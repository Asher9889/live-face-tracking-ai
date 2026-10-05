"""
Stage timing for the camera frame loop.

Answers one question with numbers instead of guesses: which stage is eating the
frame budget, and how stale is the frame by the time it is published.

Design constraints, because an instrument that costs more than it measures is
useless:
  - time.perf_counter() + list append per stage (microseconds),
  - ONE summary line per interval, not one per frame,
  - all counters reset on flush, so memory cannot grow,
  - PERF_LOG=false turns every call into a no-op.

Usage inside _camera_loop:

    perf = PerfMeter(cam.code)
    t0 = time.perf_counter()
    results = model.track(...)
    perf.record("track", (time.perf_counter() - t0) * 1000)
    ...
    perf.tick()          # one frame seen
    perf.tick_decode()   # one frame decoded by the RTSP reader thread
    perf.flush()         # prints when the interval is due

Stages are reported as mean / p95 / max in milliseconds. Counters added with
add() are reported per frame, which is what makes "am I recognising too
often?" answerable.

Two rates are tracked per camera, because they answer different questions:
  loop fps   frames the camera loop processed (tick)
  decode fps frames the RTSP reader decoded (tick_decode)
decode ≈ 25 and loop ≈ 1  -> ffmpeg is fine, the pipeline is the bottleneck.
both ≈ 1                  -> decode/RTSP itself is starving the loop.

Environment:
    PERF_LOG=true|false       default true
    PERF_LOG_INTERVAL=<secs>  default 10
    PERF_FPS_LOG=<path>       default logs/camera_fps.log (empty/false disables)
                              one JSON row per camera per interval: fps counters
                              plus stage means, so a copied log file is enough
                              to debug from the source side without journalctl.
"""

import json
import logging
import logging.handlers
import os
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime

ENABLED = os.getenv("PERF_LOG", "true").strip().lower() in ("1", "true", "yes")
INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "10") or 10)
FPS_LOG_PATH = os.getenv("PERF_FPS_LOG", "logs/camera_fps.log").strip()


def _pct(values, q):
    """q-th percentile of an unsorted list, 0.0-1.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * q)
    return ordered[min(max(idx, 0), len(ordered) - 1)]


# --------------------------------------------------------------- fps log file
# One rotating file with one JSON row per camera per interval. Many camera
# threads write concurrently, so creation and emit share one lock. Any failure
# disables the file for good instead of warning once per interval.

_fps_lock = threading.Lock()
_fps_handler = None
_fps_disabled = False


def _get_fps_handler(path):
    global _fps_handler, _fps_disabled
    if _fps_disabled or _fps_handler is not None:
        return _fps_handler
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=3
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        _fps_handler = handler
    except OSError:
        _fps_disabled = True
    return _fps_handler


def _write_fps_row(row):
    if not FPS_LOG_PATH or FPS_LOG_PATH.lower() in ("0", "false", "no"):
        return
    with _fps_lock:
        handler = _get_fps_handler(FPS_LOG_PATH)
        if handler is None:
            return
        record = logging.LogRecord(
            "camera_fps", logging.INFO, __file__, 0, json.dumps(row), (), None
        )
        try:
            handler.emit(record)
        except OSError:
            handler.close()
            _fps_handler = None
            _fps_disabled = True


class PerfMeter:
    """
    Accumulates per-stage durations and counters, printing one summary line per
    interval. A meter is owned by a single camera thread; self_flush meters
    (used by the shared ReID encoder, which runs from many threads) take a lock
    on every record and flush themselves.
    """

    def __init__(self, name, interval=None, self_flush=False):
        self.name = name
        self.interval = float(interval if interval is not None else INTERVAL)
        self.self_flush = self_flush
        self._lock = threading.Lock()
        self._reset()
        # Decode counter: written by the RTSP reader thread, read by the loop
        # thread at flush. Delta-based so a flush never erases a concurrent
        # tick, which is why _reset() deliberately leaves both alone.
        self._dec = 0
        self._dec_base = 0

    def _reset(self):
        self._ms = defaultdict(list)
        self._cnt = defaultdict(int)
        self._frames = 0
        self._started = time.monotonic()

    # ------------------------------------------------------------- recording

    def record(self, stage, ms):
        if not ENABLED:
            return
        if self.self_flush:
            with self._lock:
                self._ms[stage].append(float(ms))
                due = (time.monotonic() - self._started) >= self.interval
            if due:
                self.flush()
            return
        self._ms[stage].append(float(ms))

    def add(self, counter, n=1):
        if not ENABLED:
            return
        if self.self_flush:
            with self._lock:
                self._cnt[counter] += int(n)
            return
        self._cnt[counter] += int(n)

    def tick(self, frames=1):
        if not ENABLED:
            return
        if self.self_flush:
            with self._lock:
                self._frames += int(frames)
            return
        self._frames += int(frames)

    def tick_decode(self, frames=1):
        """One frame decoded by the RTSP reader thread. Locked: two threads write."""
        if not ENABLED:
            return
        with self._lock:
            self._dec += int(frames)

    @contextmanager
    def stage(self, label):
        """with perf.stage("track"): ...  -> records the block duration."""
        if not ENABLED:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(label, (time.perf_counter() - t0) * 1000.0)

    # ---------------------------------------------------------------- output

    def flush(self, force=False):
        if not ENABLED:
            return False

        if self.self_flush:
            with self._lock:
                dec_pending = self._dec - self._dec_base
                if not self._ms and not self._cnt and not self._frames and not dec_pending:
                    return False
                if not force and (time.monotonic() - self._started) < self.interval:
                    return False
                dec = self._dec - self._dec_base
                self._dec_base = self._dec
                snapshot_ms = {k: list(v) for k, v in self._ms.items()}
                snapshot_cnt = dict(self._cnt)
                frames = self._frames
                elapsed = time.monotonic() - self._started
                self._reset()
        else:
            # The reader thread writes _dec concurrently, so it is sampled under
            # the lock; the stage/counter data is single-writer (this thread).
            with self._lock:
                dec_pending = self._dec - self._dec_base
            if not self._ms and not self._cnt and not self._frames and not dec_pending:
                return False
            if not force and (time.monotonic() - self._started) < self.interval:
                return False
            with self._lock:
                dec = self._dec - self._dec_base
                self._dec_base = self._dec
            snapshot_ms = {k: list(v) for k, v in self._ms.items()}
            snapshot_cnt = dict(self._cnt)
            frames = self._frames
            elapsed = time.monotonic() - self._started
            self._reset()

        elapsed = max(elapsed, 1e-6)
        samples = sum(len(v) for v in snapshot_ms.values())

        # A frame-driven meter reports FPS; a call-driven one (the shared ReID
        # encoder) has no frames of its own, so reporting fps=0.0 there would
        # be actively misleading. decode fps only appears where a reader exists.
        if frames or dec:
            head = (
                f"frames={frames} fps={frames / elapsed:.1f}"
                f" dec={dec} dec_fps={dec / elapsed:.1f}"
            )
        else:
            head = f"samples={samples}"

        # Durations: stage=mean(p95/max) ms, in the order stages were first seen.
        parts = []
        for stage, values in snapshot_ms.items():
            parts.append(
                f"{stage}={statistics.mean(values):.1f}"
                f"(p95={_pct(values, 0.95):.1f} max={max(values):.1f})"
            )

        # Counters: absolute total plus the per-frame rate that makes
        # "recognition runs every frame" visible without arithmetic.
        for counter, value in snapshot_cnt.items():
            if frames:
                parts.append(f"{counter}={value}({value / frames:.2f}/f)")
            else:
                parts.append(f"{counter}={value}")

        # File row: only frame-driven meters (the ReID meter would log a
        # constant fps=0 and drown the camera rows).
        if frames or dec:
            _write_fps_row(
                {
                    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "camera": self.name,
                    "interval_s": round(elapsed, 1),
                    "loop_frames": frames,
                    "loop_fps": round(frames / elapsed, 1),
                    "decode_frames": dec,
                    "decode_fps": round(dec / elapsed, 1),
                    "published": snapshot_cnt.get("published", 0),
                    "stages_ms": {
                        stage: {
                            "mean": round(statistics.mean(values), 1),
                            "p95": round(_pct(values, 0.95), 1),
                        }
                        for stage, values in snapshot_ms.items()
                    },
                }
            )

        print(
            f"[PERF {self.name}] {elapsed:.1f}s {head} | "
            + " ".join(parts),
            flush=True,
        )
        return True


# The ReID encoder is shared by every camera thread, so it cannot be attached to
# one camera's meter without being counted once per camera. It owns its meter
# instead, which flushes itself on the same interval.
REID_PERF = PerfMeter("reid", self_flush=True)
