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

def log(cam, person_id, stage, msg):
    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}][{stage}] {msg}")


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


# =========================
# WORKER (REWRITTEN)
# =========================


def _camera_loop(cam: CameraConfig) -> None:
    print(f"[Camera] Worker started → {cam.code}")

    target_fps = int(FRAME_RATE)
    interval = 1.0 / target_fps

    track_manager = TrackManager(publisher=publisher, gate_type=cam.gate_type)
    builder = UniqueFaceRepresentationBuilder()

    track_state = {}
    track_identity = {}
    track_known_buffer = {}
    track_unknown_buffer = {}
    track_unknown_identity = {}
    track_unknown_meta = {}
    track_embedding_state = {}

    while True:
        cap = _open_capture(cam.rtsp_url)

        if not cap.isOpened():
            time.sleep(2)
            continue

        last_processed = 0.0

        while True:
            if not cap.grab():
                cap.release()
                break

            now = time.time()
            if now - last_processed < interval:
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            last_processed = now
            frame_h, frame_w = frame.shape[:2]

            results = model.track(frame, persist=True, classes=[0], conf=0.25, verbose=False)

            if results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()

            lost = track_manager.cleanup_lost_tracks(cam.code, ids.tolist())

            for tid in lost:
                track_state.pop(tid, None)
                track_identity.pop(tid, None)
                track_known_buffer.pop(tid, None)
                track_unknown_buffer.pop(tid, None)
                track_unknown_identity.pop(tid, None)
                track_unknown_meta.pop(tid, None)
                track_embedding_state.pop(tid, None)

            for person_id, bbox in zip(ids, boxes):

                person_id = int(person_id)

                if person_id in track_identity:
                    continue

                if person_id not in track_state:
                    track_state[person_id] = TrackState.COLLECTING_KNOWN
                    log(cam, person_id, "STATE", "INIT → COLLECTING_KNOWN")

                state = track_state[person_id]

                # -------------------------
                # ROI + FACE DETECTION
                # -------------------------
                x1, y1, x2, y2 = expand_bbox(bbox, frame_w, frame_h)
                roi_data = extract_person_roi(frame, person_id, np.array([x1, y1, x2, y2]))
                if roi_data is None:
                    continue

                _, roi, offset = roi_data

                faces = insight_engine.detect_and_generate_embedding(roi, offset, cam.code)
                if not faces:
                    continue

                # -------------------------
                # QUALITY FILTER
                # -------------------------
                valid_faces = []
                for f in faces:
                    if f["score"] < envConfig.SCRFD_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, f["bbox"])
                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size == 0:
                        continue

                    analysis = face_landmarker_engine.analyze(face_img)
                    quality = insight_engine.compute_face_quality(f, face_img, analysis)

                    if quality < 0.15:
                        continue

                    f["quality"] = quality
                    valid_faces.append(f)

                if not valid_faces:
                    continue

                best = select_best_face(valid_faces)
                if best is None:
                    continue

                embedding = best["embedding"]
                quality = best["quality"]

                x1, y1, x2, y2 = map(int, best["bbox"])
                face_img = frame[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue

                pose = get_pose_name(best.get("pose", [None])[0]) or "unknown"

                # =====================================================
                # 🔵 STAGE 1: KNOWN
                # =====================================================
                if state == TrackState.COLLECTING_KNOWN:

                    # stability ONLY here
                    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        log(cam, person_id, "STABILITY", "REJECTED")
                        continue

                    buffer = track_known_buffer.get(person_id, [])
                    buffer.append({
                        "embedding": embedding,
                        "quality": quality,
                        "pose_bucket": pose,
                        "img": face_img,
                        "ts": time.time()
                    })

                    buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)[:3]
                    track_known_buffer[person_id] = buffer

                    log(cam, person_id, "KNOWN", f"buffer_size={len(buffer)}")

                    if len(buffer) < 3:
                        continue

                    # combine
                    emb = np.array([x["embedding"] for x in buffer])
                    w = np.array([x["quality"] for x in buffer])
                    final = np.average(emb, axis=0, weights=w)
                    final /= np.linalg.norm(final)

                    log(cam, person_id, "KNOWN", "RUN MATCH")
                    match = embedding_store.find_match(final)

                    if match:
                        track_identity[person_id] = match["employee_id"]
                        track_state[person_id] = TrackState.MATCHED_KNOWN

                        track_manager.recognition_confirmed(
                            cam.code,
                            person_id,
                            match["employee_id"],
                            match["similarity"]
                        )

                        log(cam, person_id, "KNOWN", f"MATCHED → {match['employee_id']}")
                        track_known_buffer.pop(person_id, None)
                        continue

                    # move to unknown
                    track_state[person_id] = TrackState.COLLECTING_UNKNOWN
                    track_unknown_buffer[person_id] = [
                        x for x in buffer if x["quality"] >= MIN_UNKNOWN_CREATION_QUALITY
                    ]
                    track_known_buffer.pop(person_id, None)

                    log(cam, person_id, "STATE", "→ COLLECTING_UNKNOWN")
                    continue

                # =====================================================
                # 🔵 STAGE 2: UNKNOWN
                # =====================================================
                elif state == TrackState.COLLECTING_UNKNOWN:

                    buffer = track_unknown_buffer.get(person_id, [])
                    buffer = builder.add(buffer, embedding, quality, pose, img=face_img)
                    track_unknown_buffer[person_id] = buffer

                    if not builder.is_ready(buffer):
                        continue

                    centroid = builder.build(buffer)
                    if centroid is None:
                        continue

                    best = builder.get_best_face(buffer)
                    if not best or best["img"] is None or best["img"].size == 0:
                        continue

                    ok, buf = cv2.imencode(".jpg", best["img"])
                    if not ok:
                        continue

                    match = unknown_embedding_store.find_match(centroid)

                    if match:
                        unknown_id = match["unknown_id"]
                    else:
                        if cam.camera_role != "REGISTER":
                            continue

                        unknown_id = unknown_embedding_store.add_unknown(
                            centroid,
                            buf.tobytes(),
                            int(time.time() * 1000),
                            cam.code,
                            len(buffer),
                            {x["pose_bucket"] for x in buffer}
                        )

                    track_unknown_identity[person_id] = unknown_id
                    track_state[person_id] = TrackState.UPDATING_UNKNOWN
                    track_unknown_meta[person_id] = {"pose_best": {}, "last_update": 0}

                    log(cam, person_id, "STATE", "→ UPDATING_UNKNOWN")
                    track_manager.unknown_confirmed(cam.code, person_id, unknown_id)
                    continue

                # =====================================================
                # 🔵 STAGE 3: UPDATE
                # =====================================================
                elif state == TrackState.UPDATING_UNKNOWN:

                    unknown_id = track_unknown_identity.get(person_id)
                    if not unknown_id:
                        continue

                    buffer = track_unknown_buffer.get(person_id, [])
                    buffer = builder.add(buffer, embedding, quality, pose, img=face_img)
                    track_unknown_buffer[person_id] = buffer

                    if not builder.is_ready(buffer):
                        continue

                    centroid = builder.build(buffer)
                    best = builder.get_best_face(buffer)

                    if not best or best["img"] is None or best["img"].size == 0:
                        continue

                    meta = track_unknown_meta.get(person_id, {"pose_best": {}, "last_update": 0})
                    pose_best = meta["pose_best"]

                    # cooldown
                    if time.time() - meta["last_update"] < 2:
                        continue

                    best_quality = best["quality"]
                    prev_q = pose_best.get(pose, 0)

                    if best_quality <= prev_q + 0.02 and pose in pose_best:
                        continue

                    ok, buf = cv2.imencode(".jpg", best["img"])
                    if not ok:
                        continue

                    pose_best[pose] = best_quality

                    unknown_embedding_store.update_unknown(
                        unknown_id,
                        centroid,
                        int(time.time() * 1000),
                        cam.code,
                        buf.tobytes(),
                        {x["pose_bucket"] for x in buffer},
                        best_quality
                    )

                    track_unknown_meta[person_id] = {
                        "pose_best": pose_best,
                        "last_update": time.time()
                    }

                    log(cam, person_id, "UPDATE", f"UPDATED → {unknown_id}")


def _camera_loopp(cam: CameraConfig) -> None:
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

                    # if not track_state[person_id] :
                    #     track_state[person_id] = TrackState.COLLECTING_FRAME
                    if person_id not in track_state:
                        track_state[person_id] = TrackState.COLLECTING_KNOWN
                        print(f"[STATE][{person_id}] INIT → COLLECTING_KNOWN")


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


                    # if track_state[person_id] != TrackState.COLLECTING_KNOWN:
                    #     continue

                    state = track_state[person_id]


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
                    match = None
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

                    # ---------------------------
                    # MOVE TO UNKNOWN PIPELINE (ONLY ONCE)
                    # ---------------------------
                    if track_state[person_id] == TrackState.COLLECTING_KNOWN:
                        print(f"[STATE][{person_id}] COLLECTING_KNOWN → COLLECTING_UNKNOWN")

                        track_state[person_id] = TrackState.COLLECTING_UNKNOWN
                        track_unknown_buffer[person_id] = converted_buffer
                        track_known_buffer.pop(person_id, None)
                        continue

                    elif track_state[person_id] == TrackState.COLLECTING_UNKNOWN:

                        buffer = track_unknown_buffer.get(person_id, [])
                        buffer = uniqueFaceBuilder.add(buffer, embedding, quality, pose_name, img=face_img)
                        track_unknown_buffer[person_id] = buffer

                        print(f"[UNKNOWN][{person_id}] buffer_size={len(buffer)}")

                        if not uniqueFaceBuilder.is_ready(buffer):
                            print(f"[UNKNOWN][{person_id}] waiting for diversity")
                            continue

                        centroid = uniqueFaceBuilder.build(buffer)
                        if centroid is None:
                            continue

                        best = uniqueFaceBuilder.get_best_face(buffer)
                        if not best or best["img"] is None or best["img"].size == 0:
                            continue

                        ok, buffer_img = cv2.imencode(".jpg", best["img"])
                        if not ok:
                            continue

                        image_bytes = buffer_img.tobytes()
                        poses = {x["pose_bucket"] for x in buffer}
                        timestamp = int(time.time() * 1000)

                        # -------- MATCH UNKNOWN --------
                        unknown_match = unknown_embedding_store.find_match(centroid)
                        print(f"[UNKNOWN][{person_id}] match → {unknown_match}")

                        if unknown_match:
                            unknown_id = unknown_match["unknown_id"]
                            print(f"[UNKNOWN][{person_id}] MATCHED → {unknown_id}")
                        else:
                            if cam.camera_role != "REGISTER":
                                continue

                            unknown_id = unknown_embedding_store.add_unknown(
                                centroid=centroid,
                                image_bytes=image_bytes,
                                timestamp=timestamp,
                                camera_code=cam.code,
                                embedding_count=len(buffer),
                                poses=poses
                            )

                            print(f"[UNKNOWN][{person_id}] CREATED → {unknown_id}")

                        track_unknown_identity[person_id] = unknown_id
                        track_state[person_id] = TrackState.UPDATING_UNKNOWN

                        track_manager.unknown_confirmed(cam.code, person_id, unknown_id)

                        track_unknown_meta[person_id] = {"pose_best": {}}

                        continue

                    # =====================================================
                    # UPDATE PIPELINE (CRITICAL)
                    # =====================================================
                    elif track_state[person_id] == TrackState.UPDATING_UNKNOWN:

                        unknown_id = track_unknown_identity.get(person_id)
                        if not unknown_id:
                            continue

                        buffer = track_unknown_buffer.get(person_id, [])
                        buffer = uniqueFaceBuilder.add(buffer, embedding, quality, pose_name, img=face_img)
                        track_unknown_buffer[person_id] = buffer

                        print(f"[UPDATE][{person_id}] buffer_size={len(buffer)}")

                        if not uniqueFaceBuilder.is_ready(buffer):
                            continue

                        centroid = uniqueFaceBuilder.build(buffer)
                        best = uniqueFaceBuilder.get_best_face(buffer)

                        if not best or best["img"] is None or best["img"].size == 0:
                            continue

                        ok, buffer_img = cv2.imencode(".jpg", best["img"])
                        if not ok:
                            continue

                        meta = track_unknown_meta.get(person_id, {"pose_best": {}})
                        pose_best = meta["pose_best"]

                        prev_q = pose_best.get(pose_name, 0)

                        # 🔥 KEY LOGIC
                        if quality <= prev_q + 0.02 and pose_name in pose_best:
                            print(f"[UPDATE][{person_id}] SKIP")
                            continue

                        pose_best[pose_name] = quality

                        unknown_embedding_store.update_unknown(
                            unknown_id,
                            centroid,
                            int(time.time() * 1000),
                            cam.code,
                            buffer_img.tobytes(),
                            {x["pose_bucket"] for x in buffer},
                            quality
                        )

                        print(f"[UPDATE][{person_id}] UPDATED → {unknown_id}")

                        track_unknown_meta[person_id] = {"pose_best": pose_best}

