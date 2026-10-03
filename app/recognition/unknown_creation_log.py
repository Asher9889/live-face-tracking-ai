"""
Structured audit log for unknown-identity registration decisions.

One JSON object per line, size-rotated, so questions like "how many unknown
registrations last week had eye_sharpness below 1000" are answerable with a JSON
parser instead of regex archaeology.

Deliberately does NOT log face images or embeddings. That data already lives in
the API store; writing biometrics to a plaintext file is not worth the debugging
convenience. Only decision metadata is recorded.

Every function here is best-effort: a failure to write the audit line must never
take down the recognition pipeline, so exceptions are swallowed after a warning.
"""

import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler

from app.config.config import envConfig

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_handler = None
_configured = False


def _log_dir_for(path: str) -> str:
    d = os.path.dirname(os.path.abspath(path))
    return d or "."


def get_handler():
    """
    Lazily build the rotating file handler.

    Wrapped in a lock because several camera threads can start concurrently and
    would otherwise each attach their own handler, duplicating every line.
    """
    global _handler, _configured

    if _configured:
        return _handler

    with _lock:
        if _configured:
            return _handler

        if envConfig.UNKNOWN_CREATION_LOG_ENABLED:
            try:
                path = os.path.abspath(envConfig.UNKNOWN_CREATION_LOG_PATH)
                os.makedirs(_log_dir_for(path), exist_ok=True)
                _handler = RotatingFileHandler(
                    path,
                    maxBytes=envConfig.UNKNOWN_CREATION_LOG_MAX_BYTES,
                    backupCount=envConfig.UNKNOWN_CREATION_LOG_BACKUPS,
                    encoding="utf-8",
                )
                _handler.setFormatter(logging.Formatter("%(message)s"))
                # Standalone handler: it owns the file, not the root logger, so
                # unknown-creation lines never interleave with app logging config.
                _handler.setLevel(logging.INFO)
            except Exception as exc:
                logger.warning("[UNKNOWN_LOG] disabled, cannot open log file: %s", exc)
                _handler = None

        _configured = True
        return _handler


def _build_base_record(event: str, cam_code, track_id, camera_role: str = None):
    """Build the base record shared by all log functions."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "camera": cam_code,
        "track_id": int(track_id) if track_id is not None else None,
    }
    if camera_role:
        record["camera_role"] = camera_role
        record["can_create_unknown"] = (camera_role == "REGISTER")
    return record


def _write_record(record: dict):
    """Best-effort write of a JSON record to the rotating log."""
    if not envConfig.UNKNOWN_CREATION_LOG_ENABLED:
        return

    handler = get_handler()
    if handler is None:
        return

    # Thresholds in force at log time
    record["min_eye_sharpness"] = envConfig.MIN_UNKNOWN_EYE_SHARPNESS
    record["enforce_eye_sharpness"] = envConfig.ENFORCE_UNKNOWN_EYE_SHARPNESS
    record["min_iris_contrast_ratio"] = envConfig.MIN_UNKNOWN_IRIS_CONTRAST_RATIO
    record["min_eye_dist_ratio"] = envConfig.MIN_UNKNOWN_EYE_DIST_RATIO
    record["max_unknown_reg_yaw"] = envConfig.MAX_UNKNOWN_REG_YAW

    try:
        handler.emit(logging.LogRecord(
            name="unknown_creation",
            level=logging.INFO,
            pathname=__file__,
            lineno=0,
            msg=json.dumps(record, ensure_ascii=False),
            args=(),
            exc_info=None,
        ))
    except Exception as exc:
        logger.warning("[UNKNOWN_LOG] write failed: %s", exc)


def _coerce_fields(record: dict, fields: dict):
    """Coerce field values to JSON-native types."""
    for key, value in fields.items():
        if isinstance(value, float):
            record[key] = round(value, 4)
        elif value is None or isinstance(value, (int, str, bool)):
            record[key] = value
        else:
            record[key] = str(value)


def log_filter_stage(
    cam_code: str,
    track_id,
    camera_role: str,
    filter_stage: str,
    passed: bool,
    threshold_used,
    measured_value,
    buffer=None,
    builder=None,
    force_create: bool = False,
    **extra_fields
):
    """
    Log a single filter gate decision (face_width, eye_sharpness, eye_visibility, force_create).
    
    Call this at EACH gate (face_width, eye_sharpness, eye_visibility, force_create).
    """
    record = _build_base_record("filter_stage", cam_code, track_id, camera_role=None)
    record["filter_stage"] = filter_stage
    record["passed"] = passed
    record["threshold_used"] = threshold_used
    record["measured_value"] = measured_value
    record["force_create"] = force_create

    if buffer is not None:
        record["buffer_frames"] = len(buffer)
        poses = {item.get("pose_bucket") for item in buffer if item.get("pose_bucket")}
        record["buffer_poses"] = list(poses)
        if buffer:
            qvals = [item.get("quality", 0) for item in buffer]
            record["buffer_quality_min"] = min(qvals)
            record["buffer_quality_max"] = max(qvals)

    if builder is not None:
        record["builder_ready"] = builder.is_ready(buffer) if buffer else False
        record["builder_min_frames"] = builder.min_frames
        record["builder_min_poses"] = builder.min_poses

    _coerce_fields(record, {
        "threshold_used": threshold_used,
        "measured_value": measured_value,
    })

    _write_record(record)


def log_decision(event: str, cam_code, track_id, camera_role: str = None, **fields):
    """
    Append one decision record (kept for backward compatibility).
    
    `event` is the record type, e.g. "unknown_registered" or "unknown_rejected".
    """
    record = _build_base_record(event, cam_code, track_id, camera_role)
    _coerce_fields(record, fields)
    _write_record(record)


def _num(value):
    """Coerce numpy float32 to native float for JSON serialization."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def face_metrics(
    analysis,
    quality,
    final_quality,
    best_face_width=None,
    buffer=None,
    builder=None,
    camera_role: str = None,
    force_create: bool = False,
    filter_stage: str = None,
    threshold_used=None,
    measured_value=None,
):
    """
    Extract comparable measurements from a face pipeline iteration.
    
    Returns a dict for `log_decision` / `log_filter_stage` with all face metrics
    plus pipeline context (buffer state, builder state, thresholds in force).
    """
    analysis = analysis or {}

    def num(key):
        value = analysis.get(key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    contrast = num("iris_contrast_min")
    core = num("iris_core_brightness")
    if contrast is not None and core and core > 0:
        iris_contrast_ratio = round(contrast / core, 4)
    else:
        iris_contrast_ratio = None

    metrics = {
        "eye_sharpness": _num(analysis.get("eye_sharpness")),
        "iris_contrast": _num("iris_contrast_min"),
        "iris_contrast_ratio": iris_contrast_ratio,
        "iris_contrast_mean": _num("iris_contrast_mean"),
        "iris_core_brightness": _num("iris_core_brightness"),
        "iris_radius": _num("iris_radius"),
        "eye_aperture": _num("eye_aperture"),
        "blur": _num("blur"),
        "yaw": _num("yaw"),
        "pitch": _num("pitch"),
        "roll": _num("roll"),
        "eye_dist_ratio": _num("eye_dist_ratio"),
        "face_w_in_crop": _num(analysis.get("face_width")),
        "quality": quality,
        "final_quality": final_quality,
        "best_face_width": best_face_width,
        "min_eye_sharpness": envConfig.MIN_UNKNOWN_EYE_SHARPNESS,
        "min_iris_contrast_ratio": envConfig.MIN_UNKNOWN_IRIS_CONTRAST_RATIO,
        "min_eye_dist_ratio": envConfig.MIN_UNKNOWN_EYE_DIST_RATIO,
        "max_unknown_reg_yaw": envConfig.MAX_UNKNOWN_REG_YAW,
    }

    # Pipeline context
    if buffer is not None:
        metrics["buffer_frames"] = len(buffer)
        poses = {item.get("pose_bucket") for item in buffer if item.get("pose_bucket")}
        metrics["buffer_poses"] = list(poses)
        if buffer:
            qvals = [item.get("quality", 0) for item in buffer]
            metrics["buffer_quality_min"] = min(qvals)
            metrics["buffer_quality_max"] = max(qvals)

    if builder is not None:
        metrics["builder_ready"] = builder.is_ready(buffer) if buffer else False
        metrics["builder_min_frames"] = builder.min_frames
        metrics["builder_min_poses"] = builder.min_poses

    if camera_role:
        metrics["camera_role"] = camera_role
        metrics["can_create_unknown"] = (camera_role == "REGISTER")

    if force_create:
        metrics["force_create"] = True

    if filter_stage:
        metrics["filter_stage"] = filter_stage

    if threshold_used is not None:
        metrics["threshold_used"] = threshold_used

    return metrics
