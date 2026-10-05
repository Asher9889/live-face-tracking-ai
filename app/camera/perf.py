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
    perf.flush()         # prints when the interval is due

Stages are reported as mean / p95 / max in milliseconds. Counters added with
add() are reported per frame, which is what makes "am I recognising too
often?" answerable.

Environment:
    PERF_LOG=true|false       default true
    PERF_LOG_INTERVAL=<secs>  default 10
"""

import os
import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

ENABLED = os.getenv("PERF_LOG", "true").strip().lower() in ("1", "true", "yes")
INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "10") or 10)


def _pct(values, q):
    """q-th percentile of an unsorted list, 0.0-1.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(len(ordered) * q)
    return ordered[min(max(idx, 0), len(ordered) - 1)]


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
                if not self._ms and not self._cnt and not self._frames:
                    return False
                if not force and (time.monotonic() - self._started) < self.interval:
                    return False
                snapshot_ms = {k: list(v) for k, v in self._ms.items()}
                snapshot_cnt = dict(self._cnt)
                frames = self._frames
                elapsed = time.monotonic() - self._started
                self._reset()
        else:
            if not self._ms and not self._cnt and not self._frames:
                return False
            if not force and (time.monotonic() - self._started) < self.interval:
                return False
            snapshot_ms = {k: list(v) for k, v in self._ms.items()}
            snapshot_cnt = dict(self._cnt)
            frames = self._frames
            elapsed = time.monotonic() - self._started
            self._reset()

        elapsed = max(elapsed, 1e-6)
        samples = sum(len(v) for v in snapshot_ms.values())

        # A frame-driven meter reports FPS; a call-driven one (the shared ReID
        # encoder) has no frames of its own, so reporting fps=0.0 there would
        # be actively misleading.
        head = (
            f"frames={frames} fps={frames / elapsed:.1f}"
            if frames
            else f"samples={samples}"
        )

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
