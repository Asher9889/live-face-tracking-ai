from dataclasses import dataclass
import numpy as np
from typing import Tuple


@dataclass
class Detection:
    camera_code: str
    timestamp: float
    frame: np.ndarray
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    pose: Tuple[float, float, float]
    age: int | None
    gender: int | None


@dataclass
class FaceCrop:
    camera_code: str
    timestamp: float
    crop: np.ndarray
    bbox: np.ndarray
    landmarks: np.ndarray