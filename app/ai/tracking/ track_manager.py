import threading
from typing import Dict, List
from .tracker import SimpleTracker
import numpy as np


class TrackManager:

    def __init__(self):
        self.trackers: Dict[str, SimpleTracker] = {}
        self.lock = threading.Lock()

    def get_tracker(self, camera_code: str) -> SimpleTracker:
        with self.lock:
            if camera_code not in self.trackers:
                self.trackers[camera_code] = SimpleTracker()
            return self.trackers[camera_code]

    def update(
        self,
        camera_code: str,
        detections: List[np.ndarray],
        timestamp: float
    ):
        tracker = self.get_tracker(camera_code)
        return tracker.update(detections, timestamp)
