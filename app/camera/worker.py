import threading
import time
import cv2
from typing import List

from app.camera import CameraConfig, frame_queue


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

    frame_count = 0
    last_log = time.time()

    while True:
        # Keep buffer fresh
        if not cap.grab():
            print(f"[Camera] ⚠️ Stream lost → {cam.code}")
            time.sleep(2)
            continue

        frame_count += 1

        # FPS throttling
        if frame_count % 4 != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        # ===== Here later we will push frame to queue =====

        frame_queue.push(camera_code=cam.code, frame=frame)
    
        # Heartbeat
        now = time.time()
        if now - last_log > 3:
            print(f"[Camera] {cam.code} receiving frames...")
            last_log = now
