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


def log_decision(event: str, cam_code, track_id, **fields):
    """
    Append one decision record.

    `event` is the record type, e.g. "unknown_registered" or
    "unknown_rejected". Everything else becomes a JSON field. Values that are not
    JSON-serialisable are stringified rather than dropped, so a record is never
    lost to a logging bug.
    """
    if not envConfig.UNKNOWN_CREATION_LOG_ENABLED:
        return

    handler = get_handler()
    if handler is None:
        return

    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "event": event,
        "camera": cam_code,
        "track_id": int(track_id) if track_id is not None else None,
    }

    for key, value in fields.items():
        if isinstance(value, float):
            # Round for readability; full precision is noise in an audit log.
            record[key] = round(value, 4)
        elif value is None or isinstance(value, (int, str, bool)):
            record[key] = value
        else:
            record[key] = str(value)

    # Threshold actually in force, so historical records stay interpretable if the
    # value is tuned later.
    record["min_eye_sharpness"] = envConfig.MIN_UNKNOWN_EYE_SHARPNESS
    record["enforce_eye_sharpness"] = envConfig.ENFORCE_UNKNOWN_EYE_SHARPNESS

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


def face_metrics(analysis, quality, final_quality, best_face_width=None):
    """
    Extract the comparable measurements from a face pipeline iteration.

    Returns a dict shaped for `log_decision`, so the call site stays a one-liner
    and every record carries the same field names.
    """
    analysis = analysis or {}
    return {
        "eye_sharpness": analysis.get("eye_sharpness"),
        "blur": analysis.get("blur"),
        "yaw": analysis.get("yaw"),
        "pitch": analysis.get("pitch"),
        "roll": analysis.get("roll"),
        "eye_dist_ratio": analysis.get("eye_dist_ratio"),
        "face_w_in_crop": analysis.get("face_width"),
        "quality": quality,
        "final_quality": final_quality,
        "best_face_width": best_face_width,
    }
