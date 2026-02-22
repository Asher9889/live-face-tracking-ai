from typing import Any, Dict
from dataclasses import dataclass, field
import numpy as np
import time
from typing import Tuple

@dataclass
class CameraConfig:
    code: str
    name: str
    gate_type: str
    rtsp_url: str
    ai_fps: int
    roi: Dict[str, Any]

@dataclass
class FrameMessage:
    camera_code: str
    frame: np.ndarray
    timestamp: float = field(default_factory=time.time)

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