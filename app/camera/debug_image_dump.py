import os
import time
from datetime import datetime

import cv2


class DebugImageDumper:
    """Save periodic frame and face snapshots for stream/crop debugging."""

    def __init__(self, enabled: bool = False, min_interval_sec: float = 2.0):
        self.enabled = enabled
        self.min_interval_sec = max(0.2, float(min_interval_sec))
        self._last_write = {}

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.root_dir = os.path.join(base_dir, "debug_images")

    def _should_write(self, key: str) -> bool:
        now = time.time()
        last = self._last_write.get(key)

        if last is not None and (now - last) < self.min_interval_sec:
            return False

        self._last_write[key] = now
        return True

    def dump(self, cam_code: str, person_id: int, frame=None, face=None, stage: str = "track"):
        if not self.enabled:
            return

        key = f"{cam_code}:{person_id}:{stage}"
        if not self._should_write(key):
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        person_dir = os.path.join(self.root_dir, str(cam_code), f"person_{int(person_id)}")
        os.makedirs(person_dir, exist_ok=True)

        if frame is not None and getattr(frame, "size", 0) > 0:
            frame_path = os.path.join(person_dir, f"{ts}_{stage}_frame.jpg")
            cv2.imwrite(frame_path, frame)

        if face is not None and getattr(face, "size", 0) > 0:
            face_path = os.path.join(person_dir, f"{ts}_{stage}_face.jpg")
            cv2.imwrite(face_path, face)
