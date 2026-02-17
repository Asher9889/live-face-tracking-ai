from typing import Any, Dict
from dataclasses import dataclass, field
import numpy as np
import time

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