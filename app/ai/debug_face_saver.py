import os
import cv2
from datetime import datetime
from typing import List
from app.ai.types import FaceCrop

# SAVE_INTERVAL = 20  # save every N faces
_counter = 0

BASE_DIR = "debug_faces"


def save_faces(face_crops: List[FaceCrop]) -> None:
    global _counter

    if not face_crops:
        return

    for face in face_crops:
        _counter += 1

        # if _counter % SAVE_INTERVAL != 0:
        #     continue

        cam_dir = os.path.join(BASE_DIR, face.camera_code)
        os.makedirs(cam_dir, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

        filename = f"{timestamp}.jpg"
        path = os.path.join(cam_dir, filename)

        cv2.imwrite(path, face.crop)  # or face.image depending on your field