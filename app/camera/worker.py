import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.helper import is_stable_embedding, expand_bbox, score_face, crop_with_margin
from app.camera.types import CameraConfig
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

import os

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
    
    track_known_buffer = {} # stoes data for one time recorgnition 
    track_state = {} # stored on-going recorgnised track
    track_identity = {} # store recorgnised track

    track_unknown_identity = {}
    track_unknown_buffer = {}

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

                
                for person_id, bbox in zip(ids, boxes):

                    person_id = int(person_id)
                    x1, y1, x2, y2 = expand_bbox(bbox, frame_w, frame_h) # expending bbox by 10% default
                    expanded_bbox = np.array([x1, y1, x2, y2])
                    # update lifecycle
                    track_manager.update_track(cam.code, person_id, expanded_bbox, frame_ts, frame_w, frame_h)

                    # print(f"Track ID: {person_id} at {bbox}")

                    # Skip if already identified
                    if person_id in track_identity or person_id in track_unknown_identity: 
                        print(f"[Camera {cam.code}][Person {person_id}] skip: already identified")
                        continue



                    roi_data = extract_person_roi(frame, person_id, expanded_bbox)

                    if roi_data is None:
                        continue
                    
                    person_id, roi, offset = roi_data

                    # if roi.shape[0] < 120 or roi.shape[1] < 40:
                    #     print(f"[Camera {cam.code}][Person {person_id}] skip: small ROI size {roi.shape}, allowing only larger than 120x120")
                    #     continue

                    faces = insight_engine.detect_and_generate_embedding(roi, offset, cam.code)
                    
                    # print(f"[Camera {cam.code} Person {person_id}] 🧠 InsightFace detected {len(faces)} faces")

                    if len(faces) == 0: 
                        continue

                    # face quality analysis (Not for gating, just for better embedding selection and weighting)
                    faces_with_quality = []
                    for f in faces:
                        bbox = f["bbox"]
                        score = f["score"]

                        if score < 0.6:
                            continue

                        x1, y1, x2, y2 = map(int, bbox)

                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame_w, x2)
                        y2 = min(frame_h, y2)

                        face_img = frame[y1:y2, x1:x2]

                        if face_img.size == 0:
                            continue

                        # ---------------------------
                        # FaceMesh Analysis (NO GATING)
                        # ---------------------------
                        analysis = face_landmarker_engine.analyze(face_img)

                        # ---------------------------
                        # Compute Quality (CORE LOGIC)
                        # ---------------------------
                        quality = insight_engine.compute_face_quality(
                            f,
                            face_img,
                            analysis=analysis
                        )
                        print(f"[Camera {cam.code}][Person {person_id}] face quality: {quality:.3f}")
                        if quality < 0.45:
                            continue

                        f["quality"] = quality
                        f["analysis"] = analysis  # optional (for debugging / future use)

                        faces_with_quality.append(f)

                    if not faces_with_quality:
                        continue

                    # ---------------------------
                    # SELECT BEST FACE (by quality)
                    # ---------------------------
                    best_face = max(faces_with_quality, key=lambda f: score_face(f, roi.shape))
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
                        continue

                    # ---------------------------
                    # STABILITY CHECK
                    # ---------------------------
                    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        continue

                    # ---------------------------
                    # BUFFER (TOP K = 3)
                    # ---------------------------
                    buffer = track_known_buffer.get(person_id)

                    entry = {
                        "embedding": embedding,
                        "quality": quality
                    }

                    if buffer is None:
                        track_known_buffer[person_id] = [entry]
                    else:
                        buffer.append(entry)

                    buffer = track_known_buffer[person_id]

                    # keep only top 3 best quality
                    buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)[:3]
                    track_known_buffer[person_id] = buffer

                    # ---------------------------
                    # WAIT UNTIL ENOUGH DATA
                    # ---------------------------
                    if len(buffer) < 3:
                        continue

                    # ---------------------------
                    # COMBINE (WEIGHTED MEAN)
                    # ---------------------------
                    embeddings = np.array([x["embedding"] for x in buffer])
                    weights = np.array([x["quality"] for x in buffer])

                    if weights.sum() == 0:
                        weights = np.ones_like(weights)

                    final_embedding = np.average(embeddings, axis=0, weights=weights)
                    final_embedding /= np.linalg.norm(final_embedding)

                    # ---------------------------
                    # RECOGNITION
                    # ---------------------------
                    match = embedding_store.find_match(final_embedding)

                    # --------------------------------------------------
                    # CASE 1: Known person matched
                    # --------------------------------------------------
                    if match:

                        candidate = match["employee_id"]
                        score = match["similarity"]

                        track_identity[person_id] = candidate

                        track_manager.recognition_confirmed(
                            cam.code,
                            person_id,
                            candidate,
                            score
                        )

                        print("Identity locked:", candidate)
                        track_known_buffer.pop(person_id, None)
                        continue

                    # --------------------------------------------------
                    # CASE 2: Unknown person (no match)
                    # --------------------------------------------------

                    # ---------------------------
                    # STRICT FILTER FOR UNKNOWN
                    # ---------------------------
                    if not insight_engine.is_good_face_for_unknown(best_face, face_img):
                        print(f"[Camera {cam.code}][Person {person_id}] skip unknown: strict filter failed")
                        continue

                    # ---------------------------
                    # INIT / UPDATE BUFFER
                    # ---------------------------
                    buffer = track_unknown_buffer.get(person_id)

                    face_entry = {
                        "face": best_face,
                        "bbox": best_face["bbox"],  # add this
                        "img": face_img,
                        "quality": quality,
                        "embedding": embedding
                    }

                    if buffer is None:
                        track_unknown_buffer[person_id] = {
                            "faces": [face_entry]
                        }
                    else:
                        buffer["faces"].append(face_entry)

                    buffer = track_unknown_buffer[person_id]

                    # ---------------------------
                    # KEEP ONLY TOP-K (QUALITY)
                    # ---------------------------
                    buffer["faces"] = sorted(
                        buffer["faces"],
                        key=lambda f: f["quality"],
                        reverse=True
                    )[:5]

                    # ---------------------------
                    # FRAME COUNT GATE
                    # ---------------------------
                    if len(buffer["faces"]) < envConfig.MIN_UNKNOWN_FRAMES:
                        print(f"[Camera {cam.code}][Person {person_id}] collecting unknowns: have={len(buffer['faces'])} need={envConfig.MIN_UNKNOWN_FRAMES}")
                        continue

                    # ---------------------------
                    # QUALITY GATE
                    # ---------------------------
                    best_buffered_face = buffer["faces"][0]  # already sorted
                    min_req_quality = getattr(envConfig, "MIN_FACE_QUALITY", 0.49)

                    if best_buffered_face["quality"] < min_req_quality:
                        print(f"[Camera {cam.code}][Person {person_id}] waiting for better quality face: {best_buffered_face['quality']:.3f}")
                        continue

                    # ---------------------------
                    # COMPUTE CENTROID (WEIGHTED)
                    # ---------------------------
                    faces = buffer["faces"]

                    # Filter out small faces (discard if min(width,height) < 100)
                    try:
                        MIN_FACE_SIZE = 60
                        filtered_faces = []
                        for f in faces:
                            img = f.get("img")
                            if img is None:
                                continue
                            h, w = img.shape[:2]
                            if min(h, w) >= MIN_FACE_SIZE:
                                filtered_faces.append(f)
                            else:
                                print(f"[Camera {cam.code}][Person {person_id}] discarding small face {w}x{h}")

                        faces = filtered_faces
                    except Exception as e:
                        print(f"[Camera {cam.code}][Person {person_id}] failed during size filtering: {e}")

                    if not faces:
                        print(f"[Camera {cam.code}][Person {person_id}] no faces left after size filter, waiting for more frames")
                        continue

                    embeddings = np.array([f["embedding"] for f in faces])
                    weights = np.array([f["quality"] for f in faces])

                    if weights.sum() == 0:
                        weights = np.ones_like(weights)

                    centroid = np.average(embeddings, axis=0, weights=weights)
                    centroid /= np.linalg.norm(centroid)

                    embedding_count = len(faces)

                    # ---------------------------
                    # SEARCH IN UNKNOWN STORE
                    # ---------------------------
                    unknown_match = unknown_embedding_store.find_match(centroid)
                    print("unknown_match", unknown_match)

                    timestamp = int(time.time() * 1000)

                    # --------------------------------------------------
                    # CASE A: Existing unknown found
                    # --------------------------------------------------
                    if unknown_match:
                        unknown_id = unknown_match["unknown_id"]

                        best = buffer["faces"][0]  # already best
                        best_img = best["img"]

                        # best_img = cv2.resize(best_img, (224, 224))
                        _, buffer_img = cv2.imencode(".jpg", best_img)
                        image_bytes = buffer_img.tobytes()

                        unknown_embedding_store.update_unknown(
                            unknown_id,
                            centroid,
                            timestamp,
                            cam.code,
                            image_bytes
                        )

                        track_unknown_identity[person_id] = unknown_id

                        track_manager.unknown_confirmed(
                            cam.code,
                            person_id,
                            unknown_id,
                        )

                        print("Updated unknown:", unknown_id)
                        track_unknown_buffer.pop(person_id, None)
                        continue

                    # --------------------------------------------------
                    # CASE B: New unknown identity
                    # --------------------------------------------------
                    if cam.camera_role != "REGISTER":
                        print(f"[Camera {cam.code}] skip: not allowed to create unknown")
                        continue

                    best = buffer["faces"][0]
                    # face_img = best["img"]
                    bbox = best["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)

                    face_img = crop_with_margin(frame, x1, y1, x2, y2, margin=0.25)

                    # Log widths and heights of all images used to compute centroid
                    try:
                        image_sizes = []
                        for f in faces:
                            img = f["img"]
                            h, w = img.shape[:2]
                            image_sizes.append({"w": int(w), "h": int(h)})
                        print(f"[Camera {cam.code}][Person {person_id}] unknown_images_sizes: {image_sizes}")
                    except Exception as e:
                        print(f"[Camera {cam.code}][Person {person_id}] failed to log image sizes: {e}")

                    # face_img = cv2.resize(face_img, (224, 224))
                    _, buffer_img = cv2.imencode(".jpg", face_img)
                    image_bytes = buffer_img.tobytes()

                    unknown_id = unknown_embedding_store.add_unknown(
                        centroid,
                        image_bytes,
                        timestamp,
                        cam.code,
                        embedding_count
                    )

                    track_unknown_identity[person_id] = unknown_id
                    track_manager.unknown_confirmed(cam.code, person_id, unknown_id)

                    print("Created new unknown:", unknown_id)
                    track_unknown_buffer.pop(person_id, None)