from collections import deque
from dataclasses import dataclass

import numpy as np

from app.config import envConfig


@dataclass(frozen=True)
class FaceGeometry:
    width: int
    height: int
    area: int
    area_ratio: float


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    stage: str
    reason: str
    geometry: FaceGeometry
    median_width: float
    median_area_ratio: float
    median_quality: float | None
    tracking_samples: int


def _median(values) -> float:
    return float(np.median(values)) if values else 0.0


def _ensure_state(track_gate_state, person_id):
    state = track_gate_state.get(person_id)
    if state is None:
        window = int(envConfig.FACE_GATE_WINDOW)
        state = {
            "detect_widths": deque(maxlen=window),
            "detect_area_ratios": deque(maxlen=window),
            "tracking_widths": deque(maxlen=window),
            "tracking_area_ratios": deque(maxlen=window),
            "tracking_qualities": deque(maxlen=window),
        }
        track_gate_state[person_id] = state
    return state


def compute_face_geometry(face, frame_shape) -> FaceGeometry:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, face["bbox"])
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    area = width * height
    frame_area = max(1, frame_h * frame_w)
    area_ratio = float(area) / float(frame_area)
    return FaceGeometry(width=width, height=height, area=area, area_ratio=area_ratio)


def _build_decision(stage, geometry, state, passed, reason):
    use_tracking_history = stage in ("tracking", "registration") and state["tracking_widths"]
    width_history = state["tracking_widths"] if use_tracking_history else state["detect_widths"]
    area_history = state["tracking_area_ratios"] if use_tracking_history else state["detect_area_ratios"]
    quality_history = state["tracking_qualities"]

    return GateDecision(
        passed=passed,
        stage=stage,
        reason=reason,
        geometry=geometry,
        median_width=_median(width_history),
        median_area_ratio=_median(area_history),
        median_quality=_median(quality_history) if quality_history else None,
        tracking_samples=len(state["tracking_widths"]),
    )


def _passes_pose_guard(face) -> tuple[bool, str]:
    pose = face.get("pose")
    if pose is None:
        return False, "missing_pose"

    yaw, pitch, roll = pose

    if abs(yaw) > 50:
        return False, f"yaw={yaw:.1f}"
    if abs(roll) > 35:
        return False, f"roll={roll:.1f}"
    if abs(pitch) > 45:
        return False, f"pitch={pitch:.1f}"

    return True, "ok"


def evaluate_detection_gate(track_gate_state, person_id, face, frame_shape) -> GateDecision:
    state = _ensure_state(track_gate_state, person_id)
    geometry = compute_face_geometry(face, frame_shape)
    state["detect_widths"].append(geometry.width)
    state["detect_area_ratios"].append(geometry.area_ratio)

    if face["score"] < envConfig.SCRFD_THRESHOLD:
        return _build_decision("detection", geometry, state, False, f"score={face['score']:.2f}")

    pose_ok, pose_reason = _passes_pose_guard(face)
    if not pose_ok:
        return _build_decision("detection", geometry, state, False, pose_reason)

    if geometry.width < envConfig.MIN_DETECT_WIDTH:
        return _build_decision("detection", geometry, state, False, f"width={geometry.width}")

    if geometry.area_ratio < envConfig.MIN_DETECT_AREA_RATIO:
        return _build_decision("detection", geometry, state, False, f"area_ratio={geometry.area_ratio:.4f}")

    return _build_decision("detection", geometry, state, True, "ok")


def evaluate_tracking_gate(track_gate_state, person_id, face, frame_shape, quality) -> GateDecision:
    state = _ensure_state(track_gate_state, person_id)
    geometry = compute_face_geometry(face, frame_shape)
    detect_median_width = _median(state["detect_widths"])
    detect_median_area = _median(state["detect_area_ratios"])

    if quality < envConfig.MIN_TRACKING_QUALITY:
        return _build_decision("tracking", geometry, state, False, f"quality={quality:.3f}")

    if geometry.width < envConfig.MIN_TRACKING_WIDTH:
        return _build_decision("tracking", geometry, state, False, f"width={geometry.width}")

    if geometry.area_ratio < envConfig.MIN_TRACKING_AREA_RATIO:
        return _build_decision("tracking", geometry, state, False, f"area_ratio={geometry.area_ratio:.4f}")

    if len(state["detect_widths"]) >= envConfig.MIN_WIDTH_HISTORY and detect_median_width < envConfig.MIN_TRACKING_WIDTH:
        return _build_decision("tracking", geometry, state, False, f"median_width={detect_median_width:.1f}")

    if len(state["detect_area_ratios"]) >= envConfig.MIN_WIDTH_HISTORY and detect_median_area < envConfig.MIN_TRACKING_AREA_RATIO:
        return _build_decision("tracking", geometry, state, False, f"median_area_ratio={detect_median_area:.4f}")

    return _build_decision("tracking", geometry, state, True, "ok")


def accept_tracking_face(track_gate_state, person_id, face, frame_shape, quality) -> GateDecision:
    state = _ensure_state(track_gate_state, person_id)
    geometry = compute_face_geometry(face, frame_shape)
    state["tracking_widths"].append(geometry.width)
    state["tracking_area_ratios"].append(geometry.area_ratio)
    state["tracking_qualities"].append(float(quality))
    return _build_decision("tracking", geometry, state, True, "accepted")


def evaluate_registration_gate(track_gate_state, person_id, face, frame_shape, quality) -> GateDecision:
    state = _ensure_state(track_gate_state, person_id)
    geometry = compute_face_geometry(face, frame_shape)
    tracking_median_width = _median(state["tracking_widths"])
    tracking_median_area = _median(state["tracking_area_ratios"])
    tracking_median_quality = _median(state["tracking_qualities"])

    if len(state["tracking_widths"]) < envConfig.MIN_REGISTER_TRACKING_FRAMES:
        return _build_decision("registration", geometry, state, False, "insufficient_tracking_history")

    if quality < envConfig.MIN_REGISTER_QUALITY:
        return _build_decision("registration", geometry, state, False, f"quality={quality:.3f}")

    if tracking_median_quality < envConfig.MIN_REGISTER_QUALITY:
        return _build_decision("registration", geometry, state, False, f"median_quality={tracking_median_quality:.3f}")

    if geometry.width < envConfig.MIN_TRACKING_WIDTH:
        return _build_decision("registration", geometry, state, False, f"current_width={geometry.width}")

    if tracking_median_width < envConfig.MIN_REGISTER_WIDTH:
        return _build_decision("registration", geometry, state, False, f"median_width={tracking_median_width:.1f}")

    if geometry.area_ratio < envConfig.MIN_TRACKING_AREA_RATIO:
        return _build_decision("registration", geometry, state, False, f"current_area_ratio={geometry.area_ratio:.4f}")

    if tracking_median_area < envConfig.MIN_REGISTER_AREA_RATIO:
        return _build_decision("registration", geometry, state, False, f"median_area_ratio={tracking_median_area:.4f}")

    return _build_decision("registration", geometry, state, True, "ok")
