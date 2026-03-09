import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.types import CameraConfig
from app.config import FRAME_RATE

from ultralytics import YOLO

from app.ai.insight_detector import InsightFaceEngine
from app.camera.extract_person_roi import extract_person_roi
from app.config.config import envConfig
from app.events.publisher import EventPublisher
from app.recognition import embedding_store, unknown_embedding_store
from app.tracking.track_manager import TrackManager
from app.database import redis_client


model = YOLO("yolov8n.pt")
insight_engine = InsightFaceEngine()
publisher = EventPublisher(redis_client)

class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"


def _open_capture(rtsp_url: str):
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
    print(f"[Camera] Worker started → {cam.code}")
    target_fps = int(FRAME_RATE)

    track_manager = TrackManager(publisher=publisher)
    
    track_state = {} # stored on-going recorgnised track
    track_identity = {} # store recorgnised track

    track_unknown_state = {}
    track_unknown_identity = {}
    track_unknown_buffer = {}

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


            # DETECTION & Tracking

            results = model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                classes=[0],            # Only track 'person' (COCO class 0)
                verbose=False,
                conf=0.25               # Adjust based on your environment
            )
 
            # Get results
            if results[0].boxes.id is not None: # gives all bbox and ids
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                
                for person_id, bbox in zip(ids, boxes):

                    person_id = int(person_id)
                    # update lifecycle
                    track_manager.update_track(cam.code, person_id, bbox)

                    print(f"Track ID: {person_id} at {bbox}")

                    # Skip if already identified
                    if person_id in track_identity or person_id in track_unknown_identity: 
                        continue

                    roi_data = extract_person_roi(frame, person_id, bbox)

                    if roi_data is None:
                        continue
                    
                    person_id, roi, offset = roi_data

                    faces = insight_engine.detect_and_generate_embedding(roi, offset)
                    
                    print(f"[Camera {cam.code} Person {person_id}] 🧠 InsightFace detected {len(faces)} faces")

                    if len(faces) == 0:
                        continue

                    # FACE QUALITY FILTER

                    good_faces = []

                    for f in faces:
                        if insight_engine.is_good_face(f):
                            good_faces.append(f)


                    if len(good_faces) == 0:
                        continue
                    print("good_faces are", len(good_faces))

                    track_manager.face_detected(cam.code, person_id) 

                    # CHOOSE BEST FACE
                    """
                    handling a scenario where yolo gives multiple faces for a single tracked person
                    """
                    best_face = max(good_faces, key=lambda f: f["score"])

                    embedding = best_face["embedding"]

                    # RECOGNITION
                    match = embedding_store.find_match(embedding)

                    # --------------------------------------------------
                    # CASE 1: Known person matched
                    # --------------------------------------------------
                    if match:
                        # track_manager.recognition_pending(cam.code, person_id)
                        # continue

                        candidate = match["employee_id"]

                        state = track_state.get(person_id)

                        if state is None:
                            track_state[person_id] = {"candidate": candidate, "count": 1}
                        else:
                            if state["candidate"] == candidate:
                                state["count"] += 1
                            else:
                                state["candidate"] = candidate
                                state["count"] = 1
                        if track_state[person_id]["count"] >= 3:
                            track_identity[person_id] = candidate
                            track_manager.recognition_confirmed(
                                cam.code,
                                person_id,
                                candidate
                            )
                            print("Identity locked:", candidate)

                        continue

                    # --------------------------------------------------
                    # CASE 2: Unknown candidate
                    # --------------------------------------------------

                    buffer = track_unknown_buffer.get(person_id)
                    if buffer is None:
                        track_unknown_buffer[person_id] = {
                            "embeddings": [embedding],
                            "faces": [best_face]
                        }
                    else:
                        buffer["embeddings"].append(embedding)
                        buffer["faces"].append(best_face)
                    
                    buffer = track_unknown_buffer[person_id]

                    # Stop collecting after MAX_UNKNOWN_FRAMES
                    if len(buffer["embeddings"]) > envConfig.MAX_UNKNOWN_FRAMES:
                        buffer["embeddings"] = buffer["embeddings"][:envConfig.MAX_UNKNOWN_FRAMES]
                        buffer["faces"] = buffer["faces"][:envConfig.MAX_UNKNOWN_FRAMES]

                    # If not enough frames yet → keep collecting
                    if len(buffer["embeddings"]) < envConfig.MIN_UNKNOWN_FRAMES:
                        continue

                    # Compute Track Centroid

                    embeddings = np.stack(buffer["embeddings"])

                    centroid = np.median(embeddings, axis=0)
                    centroid = centroid / np.linalg.norm(centroid)

                    # Search in Unknown Store

                    unknown_match = unknown_embedding_store.find_match(centroid)
                    timestamp = int(time.time() * 1000)

                    # CASE A: Existing unknown found
                    if unknown_match:
                        unknown_id = unknown_match["unknown_id"]

                        unknown_embedding_store.update_unknown(
                            unknown_id,
                            centroid,
                            timestamp
                        )

                        track_unknown_identity[person_id] = unknown_id

                        track_manager.unknown_confirmed(
                            cam.code,
                            person_id,
                            unknown_id
                        )

                        print("Updated unknown:", unknown_id)

                        continue

                    # CASE B: New unknown identity

                    best_face = max(buffer["faces"], key=lambda f: f["score"])

                    bbox = best_face["bbox"]
                    x1, y1, x2, y2 = map(int, bbox)

                    h, w = frame.shape[:2]

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    face_img = frame[y1:y2, x1:x2]

                    # resize for storage
                    face_img = cv2.resize(face_img, (224, 224))

                    # encode jpeg
                    _, buffer_img = cv2.imencode(".jpg", face_img)
                    image_bytes = buffer_img.tobytes()

                    unknown_id = unknown_embedding_store.add_unknown(centroid, image_bytes, timestamp)

                    track_unknown_identity[person_id] = unknown_id

                    track_manager.unknown_confirmed(cam.code, person_id, unknown_id)

                    print("Created new unknown:", unknown_id)


           