import threading
import time
import cv2
from typing import List
import random
from enum import Enum

from app.camera import CameraConfig, FrameMessage
from app.camera.frame_queue import frame_queue
from app.camera.helper import motion_score, is_blurry

class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"

def _open_capture(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
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


# def _camera_loop(cam: CameraConfig) -> None:
#     """
#     Capture frames from RTSP and push to frame queue.
#     """
#     print(f"[Camera] Connecting → {cam.code}")

#     cap = cv2.VideoCapture(cam.rtsp_url)

#     if not cap.isOpened():
#         print(f"[Camera] ❌ Failed to open stream → {cam.code}")
#         return

#     print(f"[Camera] ✅ Connected → {cam.code}")

#     target_fps =  15
#     interval = 1.0 / target_fps

#     last_processed = 0.0
#     last_log = time.time()

#     while True:
#         if not cap.grab():
#             print(f"[Camera] ⚠️ Stream lost → {cam.code}")
#             cap.release()
#             time.sleep(2)
#             cap = cv2.VideoCapture(cam.rtsp_url)
#             continue

#         # Rate limiting based on Time
#         # Dynamic FPS throttle
#         now = time.time()
#         if now - last_processed < interval:
#             continue

#         # Decode frame
#         ret, frame = cap.retrieve()
#         if not ret:
#             continue

#         last_processed = now

#         # push frame to queue
#         frame_queue.push(FrameMessage(camera_code=cam.code, frame=frame, timestamp=now))

#         # Heartbeat
#         if now - last_log > 3:
#             # print(f"[Camera] {cam.code} receiving frames...")
#             last_log = now


def _camera_loop(cam: CameraConfig) -> None:
    print(f"[Camera] Worker started → {cam.code}")

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
                break  # go to reconnect loop

            now = time.time()

            if now - last_processed < interval:
                continue

            ret, frame = cap.retrieve()

            if not ret:
                continue

            last_processed = now

            frame_queue.push(
                FrameMessage(
                    camera_code=cam.code,
                    frame=frame,
                    timestamp=now,
                )
            )

            if now - last_log > 5:
                # print(f"[Camera] ❤️ Live → {cam.code}")
                last_log = now