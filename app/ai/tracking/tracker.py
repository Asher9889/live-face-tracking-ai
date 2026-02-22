import numpy as np
import time
from typing import List
from .track import Track


class SimpleTracker:

    def __init__(self, max_age: float = 2.0, iou_threshold: float = 0.3):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    def update(self, detections: List[np.ndarray], timestamp: float):

        updated_tracks = []

        for det in detections:
            matched = False

            for track in self.tracks:
                if self._iou(track.bbox, det) > self.iou_threshold:
                    track.update(det, timestamp)
                    updated_tracks.append(track)
                    matched = True
                    break

            if not matched:
                track = Track(
                    track_id=self.next_id,
                    bbox=det,
                    last_seen=timestamp
                )
                self.next_id += 1
                self.tracks.append(track)
                updated_tracks.append(track)

        self._cleanup(timestamp)

        return updated_tracks

    def _cleanup(self, timestamp):
        self.tracks = [
            t for t in self.tracks
            if (timestamp - t.last_seen) < self.max_age
        ]

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        union = areaA + areaB - inter + 1e-6

        return inter / union
