import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.ai.face_quality_filter import is_good_face
from app.camera.types import CameraConfig
from app.config import FRAME_RATE

from ultralytics import YOLO

from app.ai.insight_detector import InsightFaceEngine
from app.camera.extract_person_roi import extract_person_roi
from app.events.publisher import EventPublisher
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
    
    track_state = {}
    track_identity = {}

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

                    roi_data = extract_person_roi(frame, person_id, bbox)

                    if roi_data is None:
                        continue
                    
                    person_id, roi, offset = roi_data



                    faces = insight_engine.detect_and_generate_embedding(roi, offset)
                    
                    print(f"[Camera {cam.code} Person {person_id}] 🧠 InsightFace detected {len(faces)} faces")

                    if len(faces) > 0: 
                        track_manager.face_detected(
                            cam.code,
                            person_id,
                        ) 


                    track_manager.recognition_pending(cam.code, person_id)
                    
                    good_faces = []

                    for f in faces:
                        if insight_engine.is_good_face(f):
                            good_faces.append(f)

                    print("good_faces are", len(good_faces))

                    if len(good_faces) == 0:
                        continue
                    
                    


                    # x1, y1, x2, y2 = map(int, bbox)
                    # Now you can pass this bbox to SCRFD for face verification

           