import os
import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.helper import is_stable_embedding, expand_bbox, select_best_face, crop_with_margin, get_pose_name, now_ms
from app.camera.types import CameraConfig, TrackState
from app.config import FRAME_RATE

from ultralytics import YOLO

from app.ai.insight_detector import InsightFaceEngine
from app.ai.face_mesh_engine import FaceLandmarkerEngine
from app.camera.extract_person_roi import extract_person_roi
from app.config.config import envConfig
from app.events.publisher import EventPublisher
from app.recognition import embedding_store, unknown_embedding_store
from app.tracking.track_manager import TrackManager
from app.database import redis_client

from app.camera.unique_face_builder import UniqueFaceRepresentationBuilder

# unknown_manager = UnknownIdentityManager(unknown_embedding_store)

MIN_UNKNOWN_CREATION_QUALITY = float(envConfig.MIN_UNKNOWN_CREATION_QUALITY)
MIN_UNKNOWN_CREATE_FRAMES = int(envConfig.MIN_UNKNOWN_CREATE_FRAMES)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(BASE_DIR,"../../models/facemesh/face_landmarker.task")
path = os.path.abspath(path)

model = YOLO("yolov8n.pt")   
insight_engine = InsightFaceEngine()
publisher = EventPublisher(redis_client)
face_landmarker_engine = FaceLandmarkerEngine(model_path=path) 

class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"


def _open_capture(rtsp_url: str):
    if isinstance(rtsp_url, str) and rtsp_url.lower() == "webcam":
        print("[Camera] Using webcam source")
        cap = cv2.VideoCapture(0)
        print("Camera FPS:", "webcam", cap.get(cv2.CAP_PROP_FPS))
        return cap

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    print("Camera FPS:", rtsp_url, cap.get(cv2.CAP_PROP_FPS))
    return cap

def start_camera_threads(cameras: List[CameraConfig]) -> None:  
    """
    Spawn one capture thread per camera.
    """

    print(f"[Camera] Starting {len(cameras)} camera threads...")

    for cam in cameras:
        thread = threading.Thread(
            target=_camera_loop,
            args=(cam,),
            daemon=True
        )
        thread.start()

        print(f"[Camera] Thread started → {cam.code}")

def _camera_loop(cam: CameraConfig) -> None:
    print(f"[Camera] Worker started → {cam.code} && Role: {cam.camera_role}")
    target_fps = int(FRAME_RATE)

    track_manager = TrackManager(publisher=publisher, gate_type=cam.gate_type)
    uniqueFaceBuilder = UniqueFaceRepresentationBuilder(max_size=7, sim_threshold=0.90, min_frames=3, min_poses=1)
    
    track_known_buffer = {} # stoes data for one time recorgnition 
    track_state = {} # stored on-going recorgnised track
    track_identity = {} # store recorgnised track

    track_unknown_identity = {}
    track_unknown_buffer = {}
    track_unknown_meta = {} # store meta for unknown tracks to decide when to update (eg: best quality, pose count etc)

    track_embedding_state = {}

    backoff = 1.0
    max_backoff = 30.0

    interval = 1.0 / target_fps

    while True:
        print(f"[Camera] 🔌 Connecting → {cam.code}")

        cap = _open_capture(cam.rtsp_url)


        if not cap.isOpened():
            print(f"[Camera] ❌ Connect failed → {cam.code}")
            sleep_time = min(backoff + random.uniform(0, 1), max_backoff)
            print(f"[Camera] ⏳ Retry in {sleep_time:.1f}s → {cam.code}")
            time.sleep(sleep_time)
            backoff = min(backoff * 2, max_backoff)
            continue

        print(f"[Camera] ✅ Connected → {cam.code}")
        backoff = 1.0  # reset backoff on success

        last_processed = 0.0
        last_log = time.time()

        while True:
            grabbed = cap.grab()

            if not grabbed:
                print(f"[Camera] ⚠️ Stream lost → {cam.code}")
                cap.release()
                # tracker.reset()
                break  # go to reconnect loop

            now = time.time()

            if now - last_processed < interval:
                continue

            ret, frame = cap.retrieve()

            if not ret or frame is None:
                continue

            last_processed = now
            if now - last_log > 5:
                # print(f"[Camera] ❤️ Live → {cam.code}")
                last_log = now

            frame_ts = int(time.time() * 1000)
            frame_h, frame_w = frame.shape[:2]
            # DETECTION & Tracking

            results = model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                classes=[0],            # Only track 'person' (COCO class 0)
                verbose=False,
                conf=0.25,               # Adjust based on your environment
            )
 
            # Get results
            if results[0].boxes.id is not None: # gives all bbox and ids
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()

                # cleanup old tracks
                active_ids = ids.tolist()
                lost_tracks = track_manager.cleanup_lost_tracks(cam.code, active_ids)
                for tid in lost_tracks:
                    track_state.pop(tid, None)
                    track_identity.pop(tid, None)
                    track_unknown_identity.pop(tid, None)
                    track_unknown_buffer.pop(tid, None)
                    track_embedding_state.pop(tid, None)
                    track_unknown_meta.pop(tid, None)

                
                for person_id, bbox in zip(ids, boxes):

                    person_id = int(person_id)
                    x1, y1, x2, y2 = expand_bbox(bbox, frame_w, frame_h) # expending bbox by 10% default
                    expanded_bbox = np.array([x1, y1, x2, y2])
                    # update lifecycle
                    track_manager.update_track(cam.code, person_id, expanded_bbox, frame_ts, frame_w, frame_h)

                    # print(f"Track ID: {person_id} at {bbox}")

                    # Skip if already identified
                    # if person_id in track_identity or person_id in track_unknown_identity: 
                    #     print(f"[Camera {cam.code}][Person {person_id}] skip: already identified")
                    #     continue

                    if person_id in track_identity:
                        continue


                    roi_data = extract_person_roi(frame, person_id, expanded_bbox)

                    if roi_data is None:
                        # print(f"[Camera {cam.code}][Person {person_id}] skip: roi extraction failed or empty ROI")
                        continue
                    
                    person_id, roi, offset = roi_data

                    if not track_state[person_id] :
                        track_state[person_id] = TrackState.COLLECTING_FRAME

                    state = track_state[person_id]

                    # if roi.shape[0] < 120 or roi.shape[1] < 40:
                    #     print(f"[Camera {cam.code}][Person {person_id}] skip: small ROI size {roi.shape}, allowing only larger than 120x120")
                    #     continue

                    faces = insight_engine.detect_and_generate_embedding(roi, offset, cam.code)

                    # print(f"[Camera {cam.code} Person {person_id}] 🧠 InsightFace detected {len(faces)} faces")

                    if len(faces) == 0:
                        # print(f"[Camera {cam.code}][Person {person_id}] no faces returned from detector")
                        continue

                    # face quality analysis (Not for gating, just for better embedding selection and weighting)
                    faces_with_quality = []
                    for f in faces: # in one ROI may be morethan one face exits.
                        bbox = f["bbox"]
                        score = f["score"] # it is only a confidence that yeah face exits.
                        if score < envConfig.SCRFD_THRESHOLD:
                            print(f"[Camera {cam.code}][Person {person_id}] skip face: Due to SCRFD low face score {score:.2f} < 0.60")
                            continue

                        x1, y1, x2, y2 = map(int, bbox)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame_w, x2)
                        y2 = min(frame_h, y2)

                        face_img = frame[y1:y2, x1:x2]

                        if face_img.size == 0:
                            print(f"[Camera {cam.code}][Person {person_id}] skip face: extracted face image empty (bbox={x1},{y1},{x2},{y2})")
                            continue

                        analysis = face_landmarker_engine.analyze(face_img)

                        quality = insight_engine.compute_face_quality(f, face_img, analysis=analysis)

                        print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] quality computed → {quality:.3f}")
                        if quality < 0.15:
                            print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] discard face: quality too low → {quality:.3f}")
                            continue

                        f["quality"] = quality
                        f["analysis"] = analysis  # optional (for debugging / future use)

                        faces_with_quality.append(f)

                    if not faces_with_quality:
                        # print(f"[Camera {cam.code}][Person {person_id}] no faces passed quality gating")
                        continue
                    
                    track_state[person_id] = TrackState.COLLECTING_KNOWN


                    best_face = select_best_face(faces_with_quality)

                    if best_face is None:
                        print(f"[Camera {cam.code}][Person {person_id}] no best face selected (selection returned None)")
                        continue
                    quality = best_face["quality"]
                    embedding = best_face["embedding"]

                    bbox = best_face["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_w, x2)
                    y2 = min(frame_h, y2)

                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size == 0:
                        print(f"[Camera {cam.code}][Person {person_id}] skip: best face crop empty after bbox clamp")
                        continue

                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] entering stability check")
                    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] rejected by stability check")
                        continue

                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] stability accepted")

                    # ---------------------------
                    # BUFFER (TOP K = 3)
                    # ---------------------------
                    buffer = track_known_buffer.get(person_id)


                    pose_data = best_face.get("pose")
                    yaw = pose_data[0] if pose_data and len(pose_data) > 0 else None
                    pose_name = get_pose_name(yaw)
                    if pose_name is None:
                        pose_name = "unknown"

                    should_store_img = False
                    if quality > 0.5:
                        if not pose_name == "left" or not pose_name == "right":
                            should_store_img = True
                        # elif pose_name not in stored_poses_for_track:
                        #     should_store_img = True

                    entry = {
                        "embedding": embedding,
                        "quality": quality,
                        "pose_bucket": pose_name,
                        "img": face_img if should_store_img else None,
                        "ts": time.time()
                    }

                    if buffer is None:
                        track_known_buffer[person_id] = [entry]
                    else:
                        print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] buffer append → quality={quality:.3f}")
                        buffer.append(entry)

                    buffer = track_known_buffer[person_id]

                    # keep only top 3 best quality
                    buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)[:3]
                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] buffer sorted → size={len(buffer)}")
                    track_known_buffer[person_id] = buffer

                    if len(buffer) < 3:
                        print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] waiting buffer → have={len(buffer)} need=3")
                        continue
                    if track_state[person_id] != TrackState.COLLECTING:
                        continue


                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] 🔥 MATCH PIPELINE TRIGGERED")
                    # ---------------------------
                    # COMBINE (WEIGHTED MEAN)
                    # ---------------------------
                    embeddings = np.array([x["embedding"] for x in buffer])
                    weights = np.array([x["quality"] for x in buffer])

                    if weights.sum() == 0:
                        weights = np.ones_like(weights)

                    final_embedding = np.average(embeddings, axis=0, weights=weights)
                    # final_embedding /= np.linalg.norm(final_embedding)
                    norm = np.linalg.norm(final_embedding)
                    if norm == 0:
                        continue
                    final_embedding /= norm

                    # ---------------------------
                    # RECOGNITION
                    # ---------------------------
                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] running embedding match")
                    if track_state[person_id] == TrackState.COLLECTING_KNOWN:
                        match = embedding_store.find_match(final_embedding)
                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] match result → {match}")

                    # --------------------------------------------------
                    # CASE 1: Known person matched
                    # --------------------------------------------------
                    if match:

                        candidate = match["employee_id"]
                        score = match["similarity"]

                        track_identity[person_id] = candidate
                        track_state[person_id] = TrackState.MATCHED_KNOWN

                        track_manager.recognition_confirmed(
                            cam.code,
                            person_id,
                            candidate,
                            score
                        )

                        print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] identity locked → {candidate}")
                        track_known_buffer.pop(person_id, None)
                        continue

                    
                    # --------------------------------------------------
                    # CASE 2: Unknown person
                    # --------------------------------------------------

                    converted_buffer = []
                    for item in track_known_buffer[person_id]:
                        if item["quality"] > MIN_UNKNOWN_CREATION_QUALITY: # to allow more samples for unknown creation, we can be a bit lenient here
                            converted_buffer.append({
                                "embedding": item["embedding"],
                                "quality": item["quality"],
                                "pose_bucket": item["pose_bucket"],
                                "img": item["img"],
                                "ts": item["ts"]
                            })

                    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] entering unknown pipeline")
                    # if quality < MIN_UNKNOWN_CREATION_QUALITY:
                    #     print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}] skip unknown: quality {quality:.3f} below threshold {MIN_UNKNOWN_CREATION_QUALITY}")    
                    #     continue

                    buffer = track_unknown_buffer.get(person_id)
                    if buffer is None:
                        buffer = converted_buffer
                        track_unknown_buffer[person_id] = buffer
                        continue # most imp line

                    buffer = uniqueFaceBuilder.add(
                        buffer,
                        embedding,
                        quality,
                        pose_name,
                        img=face_img
                    )

                    track_unknown_buffer[person_id] = buffer

                    # if len(buffer) < MIN_UNKNOWN_CREATE_FRAMES:
                    #     continue

                    if not uniqueFaceBuilder.is_ready(buffer):
                        print(f"[Camera {cam.code}][Person {person_id}] waiting for diversity")
                        continue

                    # ---------------------------
                    # BUILD REPRESENTATION
                    # ---------------------------
                    centroid = uniqueFaceBuilder.build(buffer)
                    if centroid is None:
                        continue

                    # GET BEST FACE
                    best = uniqueFaceBuilder.get_best_face(buffer)
                    if not best or "img" not in best:
                        continue
                    best_img = best["img"]

                    ok, buffer_img = cv2.imencode(".jpg", best_img)
                    if not ok:
                        continue
                    image_bytes = buffer_img.tobytes()

                    # EXTRACT POSES
                    poses = {x["pose_bucket"] for x in buffer}

                    timestamp = int(time.time() * 1000)

                    # ---------------------------
                    # SEARCH IN UNKNOWN STORE
                    # ---------------------------

                    unknown_match = unknown_embedding_store.find_match(centroid)
                    print(f"[Camera {cam.code}][Person {person_id}] unknown_match → {unknown_match}")


                    # existing_unknown_id = track_unknown_identity.get(person_id)
                    if unknown_match:

                        unknown_id = unknown_match["unknown_id"]

                        track_unknown_identity[person_id] = unknown_id
                        track_state[person_id] = TrackState.MATCHED_UNKNOWN

                        track_manager.unknown_confirmed(
                            cam.code,
                            person_id,
                            unknown_id
                        )
                        # allow update logic to continue (DO NOT continue here)
                    # --------------------------------------------------
                    # CASE B: New unknown identity
                    # --------------------------------------------------
                    if not unknown_match:

                        if cam.camera_role != "REGISTER":
                            print(f"[Camera {cam.code}] skip: not allowed to create unknown")
                            continue

                        unknown_id = unknown_embedding_store.add_unknown(
                            centroid=centroid,
                            image_bytes=image_bytes,
                            timestamp=timestamp,
                            camera_code=cam.code,
                            embedding_count=len(buffer),
                            poses=poses
                        )

                        track_unknown_identity[person_id] = unknown_id
                        track_state[person_id] = TrackState.MATCHED_UNKNOWN

                        track_manager.unknown_confirmed(
                            cam.code,
                            person_id,
                            unknown_id
                        )

                        print(f"[Camera {cam.code}] Unknown created → {unknown_id}")

                    else:
                        unknown_id = existing_unknown_id

                        meta = track_unknown_meta.get(person_id, {})
                        pose_best = dict(meta.get("pose_best", {}))
                        pose = best.get("pose_bucket")
                        if pose is None:
                            continue
                        best_quality = best["quality"]

                        prev_pose_best = pose_best.get(pose, 0)

                        is_better_pose = best_quality > (prev_pose_best + 0.02) # to reduce update flood
                        is_new_pose = pose not in pose_best

                        should_update = is_better_pose or is_new_pose

                        if not should_update:
                            continue

                        unknown_embedding_store.update_unknown(
                            unknown_id=unknown_id,
                            centroid=centroid,
                            timestamp=timestamp,
                            camera_code=cam.code,
                            image_bytes=image_bytes,
                            poses=poses,
                            quality=best_quality
                        )

                        # update meta
                        pose_best[pose] = best_quality
                        track_unknown_meta[person_id] = {
                            "pose_best": pose_best,
                            "pose_count": len(pose_best)
                        }

                        print(f"[Camera {cam.code}][Person {person_id}] Unknown updated → {unknown_id}")                    