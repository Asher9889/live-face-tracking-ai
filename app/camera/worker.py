import threading
import time
import cv2
from typing import List

from app.camera import CameraConfig, FrameMessage
from app.camera.frame_queue import frame_queue
from app.camera.helper import motion_score, is_blurry


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
    """
    Capture frames from RTSP and push to frame queue.
    """
    print(f"[Camera] Connecting → {cam.code}")

    cap = cv2.VideoCapture(cam.rtsp_url)

    if not cap.isOpened():
        print(f"[Camera] ❌ Failed to open stream → {cam.code}")
        return

    print(f"[Camera] ✅ Connected → {cam.code}")

    target_fps =  15 #cam.ai_fps or 25
    interval = 1.0 / target_fps

    last_processed = 0.0
    last_log = time.time()

    while True:
        # Keep buffer fresh
        # if not cap.grab():
        #     print(f"[Camera] ⚠️ Stream lost → {cam.code}")
        #     time.sleep(2)
        #     continue

        if not cap.grab():
            print(f"[Camera] ⚠️ Stream lost → {cam.code}")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(cam.rtsp_url)
            continue

        # Rate limiting based on Time
        # Dynamic FPS throttle
        now = time.time()
        if now - last_processed < interval:
            continue

        # Decode frame
        ret, frame = cap.retrieve()
        if not ret:
            continue

        last_processed = now

        # push frame to queue
        frame_queue.push(FrameMessage(camera_code=cam.code, frame=frame, timestamp=now))

        # Heartbeat
        if now - last_log > 3:
            # print(f"[Camera] {cam.code} receiving frames...")
            last_log = now
