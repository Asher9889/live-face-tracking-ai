"""
Per-track pipeline performance log.

The camera meter (app/camera/perf.py) is per frame and answers "which stage is
slow overall". This module is per track and answers the follow-up questions:

  - which track_ids are expensive, and in which state are they expensive,
  - how often the face pipeline actually runs for a given track,
  - how long a track takes to resolve an identity (first seen -> matched),
  - which state transitions a track went through on the way.

Why rows are aggregated instead of written per frame: 18 cameras x ~5 tracks x
15 fps is ~1350 lines/second, ~100M/day. The I/O would become the cost being
measured. Instead each track accumulates in memory and emits:

  - one row when it resolves (matched or unknown created),
  - one row when it retires (tracker lost it),
  - one periodic row while it stays alive (TRACK_PERF_LOG_INTERVAL),
  - optionally one row per frame when TRACK_PERF_LOG_VERBOSE=1.

Output is JSON lines into logs/track_perf.log, rotated the same way as
logs/unknown_creation.log. No biometrics are written here — timings, states and
identity ids only.
"""

import atexit
import json
import logging
import os
import threading
import time
from logging.handlers import RotatingFileHandler

from app.config.config import envConfig

logger = logging.getLogger(__name__)

_handler = None
_handler_lock = threading.Lock()
_handler_configured = False

# A track that flips state on every frame would otherwise grow this list without
# bound for as long as the tracker keeps it alive.
MAX_STATE_EVENTS = 50


def _get_handler():
    """Build the rotating handler once, lazily, shared by all camera threads."""
    global _handler, _handler_configured

    if _handler_configured:
        return _handler

    with _handler_lock:
        if _handler_configured:
            return _handler

        if envConfig.TRACK_PERF_LOG_ENABLED:
            try:
                path = os.path.abspath(envConfig.TRACK_PERF_LOG_PATH)
                directory = os.path.dirname(path) or "."
                os.makedirs(directory, exist_ok=True)
                _handler = RotatingFileHandler(
                    path,
                    maxBytes=envConfig.TRACK_PERF_LOG_MAX_BYTES,
                    backupCount=envConfig.TRACK_PERF_LOG_BACKUPS,
                    encoding="utf-8",
                )
                _handler.setFormatter(logging.Formatter("%(message)s"))
                _handler.setLevel(logging.INFO)
            except Exception as exc:
                logger.warning("[TRACK_PERF] disabled, cannot open log file: %s", exc)
                _handler = None

        _handler_configured = True
        return _handler


def _emit(record: dict):
    """Best-effort write of one JSON line. Never raises into the frame loop."""
    handler = _get_handler()
    if handler is None:
        return
    try:
        handler.emit(
            logging.LogRecord(
                name="track_perf",
                level=logging.INFO,
                pathname=__file__,
                lineno=0,
                msg=json.dumps(record, ensure_ascii=False, default=str),
                args=(),
                exc_info=None,
            )
        )
    except Exception as exc:  # pragma: no cover - I/O failures must not kill a camera
        logger.warning("[TRACK_PERF] write failed: %s", exc)


def _stats(bucket: dict) -> dict:
    """{n, total, mean, max} for one stage, rounded for a compact line."""
    n = bucket.get("n", 0)
    total = bucket.get("total", 0.0)
    if not n:
        return {"n": 0, "total_ms": 0.0, "mean_ms": 0.0, "max_ms": 0.0}
    return {
        "n": n,
        "total_ms": round(total, 1),
        "mean_ms": round(total / n, 2),
        "max_ms": round(bucket.get("max", 0.0), 1),
    }


class TrackPerfLog:
    """
    Aggregates stage timings per (camera, track_id).

    One camera thread owns a given track_id for its whole life, so entries are
    not shared between threads in practice; the lock exists because flush_due
    walks every camera's entries from whichever thread calls it first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._tracks = {}

    # ------------------------------------------------------------- recording

    def _entry(self, cam_code, track_id):
        key = (cam_code, track_id)
        entry = self._tracks.get(key)
        if entry is None:
            now = time.time()
            entry = {
                "camera": cam_code,
                "track_id": track_id,
                "first_seen": now,
                "last_seen": now,
                "last_flush": now,
                "frames": 0,
                "state": None,
                "identity": None,
                "resolved_at": None,
                "resolution": None,
                "stages": {},
                "states": [],
            }
            self._tracks[key] = entry
        return entry

    def observe(self, cam_code, track_id, stage, ms):
        """Accumulate one stage duration for this track."""
        if not envConfig.TRACK_PERF_LOG_ENABLED or track_id is None:
            return

        with self._lock:
            entry = self._entry(cam_code, track_id)
            entry["last_seen"] = time.time()
            bucket = entry["stages"].setdefault(
                stage, {"n": 0, "total": 0.0, "max": 0.0}
            )
            bucket["n"] += 1
            bucket["total"] += float(ms)
            if float(ms) > bucket["max"]:
                bucket["max"] = float(ms)
            # Snapshot for the optional per-frame row, taken under the lock but
            # emitted after it, so a slow disk cannot stall other cameras.
            frame_row = (
                envConfig.TRACK_PERF_LOG_VERBOSE,
                entry["state"],
                bucket["n"],
                bucket["total"],
            )

        if frame_row[0]:
            _emit({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": "frame",
                "camera": cam_code,
                "track_id": track_id,
                "state": frame_row[1],
                "stage": stage,
                "ms": round(float(ms), 2),
                "stage_n": frame_row[2],
                "stage_total_ms": round(frame_row[3], 1),
            })

    def state(self, cam_code, track_id, state, identity=None):
        """
        Record a state reading for this track.

        Called once per frame for each track that entered the face pipeline, so
        it doubles as "this track was serviced again". Transitions are logged
        only when the state actually changes, and entering a resolved state
        (matched or unknown bound) emits a resolution row exactly once.
        """
        if not envConfig.TRACK_PERF_LOG_ENABLED or track_id is None:
            return

        snapshot = None
        with self._lock:
            entry = self._entry(cam_code, track_id)
            entry["last_seen"] = time.time()
            # state() runs once per frame for each track that entered the face
            # pipeline, so this is the frame counter. observe() is called once
            # per stage and must not inflate it.
            entry["frames"] += 1
            if identity:
                entry["identity"] = str(identity)

            if entry["state"] != state:
                entry["state"] = state
                if len(entry["states"]) < MAX_STATE_EVENTS:
                    entry["states"].append({
                        "state": str(state),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    })

            resolved = state in ("MATCHED_KNOWN", "UPDATING_UNKNOWN")
            if resolved and entry["resolved_at"] is None:
                entry["resolved_at"] = time.time()
                entry["resolution"] = str(state)
                snapshot = self._snapshot_locked(entry, "resolved")
            elif not resolved and entry["resolved_at"] is not None:
                # Re-verification cleared the binding; let it resolve again.
                entry["resolved_at"] = None
                entry["resolution"] = None

        if snapshot is not None:
            _emit(snapshot)

    def retire(self, cam_code, track_id, reason="lost"):
        """Emit the final row for a track and drop its accumulator."""
        if not envConfig.TRACK_PERF_LOG_ENABLED or track_id is None:
            return

        with self._lock:
            entry = self._tracks.pop((cam_code, track_id), None)
            if entry is None:
                return
            snapshot = self._snapshot_locked(entry, "retired", reason=reason)

        _emit(snapshot)

    # ---------------------------------------------------------------- output

    def _snapshot_locked(self, entry, event, reason=None):
        now = time.time()
        entry["last_flush"] = now
        return {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            "reason": reason,
            "camera": entry["camera"],
            "track_id": entry["track_id"],
            "identity": entry["identity"],
            "resolution": entry["resolution"],
            "age_s": round(now - entry["first_seen"], 2),
            "service_span_s": round(entry["last_seen"] - entry["first_seen"], 2),
            "frames": entry["frames"],
            # Frames the face pipeline ran for this track, over the track's
            # whole life. Compare with a frame rate to see how often it re-runs.
            "resolved_after_s": (
                round(entry["resolved_at"] - entry["first_seen"], 2)
                if entry["resolved_at"] else None
            ),
            "state": entry["state"],
            "states": list(entry["states"]),
            "stages": {name: _stats(b) for name, b in entry["stages"].items()},
            "stages_total_ms": round(
                sum(b.get("total", 0.0) for b in entry["stages"].values()), 1
            ),
        }

    def flush_due(self, force=False):
        """Emit periodic snapshots for tracks that are still alive."""
        if not envConfig.TRACK_PERF_LOG_ENABLED:
            return 0

        interval = envConfig.TRACK_PERF_LOG_INTERVAL
        now = time.time()
        snapshots = []

        with self._lock:
            for entry in self._tracks.values():
                if not force and (now - entry["last_flush"]) < interval:
                    continue
                entry["last_flush"] = now
                snapshots.append(self._snapshot_locked(entry, "alive"))

        for snapshot in snapshots:
            _emit(snapshot)
        return len(snapshots)

    def close(self, reason="shutdown"):
        """
        Emit final rows for every track still alive and drop them.

        The camera loops run until the process stops, so without this the
        tracks in flight at restart would disappear with no row at all.
        Registered via atexit.
        """
        if not envConfig.TRACK_PERF_LOG_ENABLED:
            return 0

        with self._lock:
            entries = list(self._tracks.values())
            self._tracks.clear()
            snapshots = [
                self._snapshot_locked(entry, "retired", reason=reason)
                for entry in entries
            ]

        for snapshot in snapshots:
            _emit(snapshot)
        return len(snapshots)

    def reset(self):
        with self._lock:
            self._tracks.clear()


# One instance for the whole process; keys are (camera, track_id) and each
# camera thread only ever touches its own keys.
track_perf = TrackPerfLog()

# Tracks still alive when the process stops are written out as retired rows so
# a restart does not silently orphan them.
atexit.register(lambda: track_perf.close("shutdown"))
