import os
import cv2
from datetime import datetime

SAVE_INTERVAL = 1  # save every 20 detections

_counter = 0

BASE_DIR = "debug_faces" 


def save_faces(frame, faces, camera_code):
    global _counter

    _counter += 1

    if _counter % SAVE_INTERVAL != 0:
        return

    if not faces:
        return

    cam_dir = os.path.join(BASE_DIR, camera_code)
    os.makedirs(cam_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face["bbox"]

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = f"{timestamp}_{i}.jpg"
        path = os.path.join(cam_dir, filename)

        cv2.imwrite(path, crop)
