import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np

from app.ai.insight_detector import InsightFaceDetector
from app.ai.tracker_service import ByteTrackerService
from app.camera import CameraConfig, FrameMessage
from app.camera.frame_queue import frame_queue
from app.camera.helper import motion_score, is_blurry


detector = InsightFaceDetector()
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

    
    tracker = ByteTrackerService(frame_rate=15, lost_track_buffer=30)
    track_state = {}
    track_identity = {}

    backoff = 1.0
    max_backoff = 30.0

    target_fps = 15
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


            # =========================
            # DETECTION STAGE
            # =========================

            detections = detector.detect(frame)
            if detections is None:
                detections = []

            if cam.code == "entry_1":
                print(f"[Camera {cam.code}] 🔍 Before DETECTION STAGE {len(detections)} faces")   


            # Convert to tracker format
            if len(detections) > 0:

                boxes = np.asarray(
                    [d["bbox"] for d in detections],
                    dtype=np.float32   # IMPORTANT: float32
                )

                scores = np.asarray(
                    [d["score"] for d in detections],
                    dtype=np.float32
                )

                detection_meta = detections  # keep original metadata

            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                detection_meta = []

            if cam.code == "entry_1":
                print(f"[Camera {cam.code}] 🔍 After DETECTION STAGE {len(detections)} faces")   


            #------TRACKING STAGE------

            tracks, lost_ids = tracker.update(boxes=boxes, scores=scores)

            if cam.code == "entry_1":
                print(f"[Camera {cam.code}] 🔍 Detected {len(tracks)} tracks", tracks)

            # cleanup per-track state
            for tid in lost_ids:
                track_state.pop(tid, None)
                track_identity.pop(tid, None)
                print(f"[Camera {cam.code}] 🗑️ Track {tid} lost")



            




