from dataclasses import dataclass
import time
import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    last_seen: float
    age: int = 0
    hits: int = 1

    def update(self, bbox: np.ndarray, timestamp: float):
        self.bbox = bbox
        self.last_seen = timestamp
        self.age += 1
        self.hits += 1
