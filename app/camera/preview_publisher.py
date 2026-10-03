"""
LiveKit preview publisher.

Publishes one video track per camera into a LiveKit room whose name is the camera
code (e.g. "entry_2"), plus a per-frame bbox state message on the data channel.

Design notes:
- The LiveKit Python SDK is asyncio-based, so each publisher owns a private event
  loop on a daemon thread. Callers are synchronous and simply call submit().
- submit() is drop-oldest: it stores a single pending item, so a slow network can
  never build a backlog. The newest frame always wins.
- Frame timing uses the true capture timestamp (timestamp_us), not publish time, so
  a subscriber can reason about real latency.
- Recognition runs on a different thread and may not have seen this frame. Labels
  therefore travel as per-track state resolved by the caller, never recomputed here.
"""

import asyncio
import datetime
import json
import threading
import time

import cv2
import numpy as np

from app.config.config import envConfig

LIVEKIT_AVAILABLE = True
LIVEKIT_IMPORT_ERROR = None

try:
    from livekit import api, rtc
except Exception as exc:  # pragma: no cover - depends on install
    LIVEKIT_AVAILABLE = False
    LIVEKIT_IMPORT_ERROR = exc
    api = None
    rtc = None


_SEQ_LOCK = threading.Lock()
_LAST_SEQ = 0


def _next_seq() -> int:
    """
    Process-wide monotonic frame counter.

    A publisher is created per RTSP connection, so a per-instance counter would
    restart at 1 on every reconnect. The browser drops any message whose seq is
    not greater than the last one it accepted, so a reset counter makes it
    discard the entire new session and freeze on its last accepted frame. The
    counter has to outlive any single publisher.
    """
    global _LAST_SEQ
    with _SEQ_LOCK:
        _LAST_SEQ += 1
        return _LAST_SEQ


def build_frame_state(
    cam_code,
    seq,
    capture_ts_ms,
    published_w,
    published_h,
    source_w,
    source_h,
    tracks,
):
    """
    Assemble the data-channel payload for one frame.

    Bounding boxes arrive in source-frame coordinates and are rescaled to the
    published frame, so the UI can draw directly on the picture it receives.

    tracks: iterable of dicts with track_id, bbox (source coords), state, label,
            label_confidence, label_expires_at.
    """
    sx = (published_w / source_w) if source_w else 1.0
    sy = (published_h / source_h) if source_h else 1.0

    out_tracks = []
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        out_tracks.append(
            {
                "track_id": int(t["track_id"]),
                "bbox": [
                    int(round(x1 * sx)),
                    int(round(y1 * sy)),
                    int(round(x2 * sx)),
                    int(round(y2 * sy)),
                ],
                "state": t.get("state"),
                "label": t.get("label"),
                "label_name": t.get("label_name"),
                "label_confidence": t.get("label_confidence"),
                "label_expires_at": t.get("label_expires_at"),
                "buffer_size": t.get("buffer_size"),
            }
        )

    return {
        "type": "frame_state",
        "camera_code": cam_code,
        "seq": int(seq),
        "frameTs": int(capture_ts_ms),
        "frame_width": int(published_w),
        "frame_height": int(published_h),
        "tracks": out_tracks,
    }


class PreviewPublisher:
    """
    Publishes one camera to one LiveKit room (named cam.code).

    Usage:
        pub = PreviewPublisher("entry_2")
        pub.start()
        pub.submit(frame_bgr, capture_ts_ms, seq, source_w, source_h, tracks)
        ...
        pub.stop()
    """

    TOPIC_FRAME_STATE = "frame_state"

    def __init__(self, cam_code, url=None, api_key=None, api_secret=None):
        self.cam_code = cam_code
        self.room_name = cam_code  # stable, derived from the camera API "code"

        self.url = url if url is not None else envConfig.LIVEKIT_URL
        self.api_key = api_key if api_key is not None else envConfig.LIVEKIT_API_KEY
        self.api_secret = (
            api_secret if api_secret is not None else envConfig.LIVEKIT_API_SECRET
        )

        self.target_fps = max(1, int(envConfig.PREVIEW_FPS))
        self.target_width = int(envConfig.PREVIEW_WIDTH)
        self.jpeg_quality = int(envConfig.PREVIEW_QUALITY)

        self._lock = threading.Lock()
        self._pending = None
        self._stop = threading.Event()

        self._loop = None
        self._thread = None
        self._wake = None

        self._last_publish_ts = 0.0
        self._min_interval = 1.0 / self.target_fps
        self._published_w = 0
        self._published_h = 0

    # ------------------------------------------------------------------ public

    def start(self):
        if not LIVEKIT_AVAILABLE:
            print(
                f"[Preview {self.cam_code}] disabled: livekit SDK not importable "
                f"({LIVEKIT_IMPORT_ERROR})"
            )
            return False

        missing = [
            n
            for n, v in (
                ("LIVEKIT_URL", self.url),
                ("LIVEKIT_API_KEY", self.api_key),
                ("LIVEKIT_API_SECRET", self.api_secret),
            )
            if not v
        ]
        if missing:
            print(f"[Preview {self.cam_code}] disabled: missing {', '.join(missing)}")
            return False

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name=f"preview-{self.cam_code}", daemon=True
        )
        self._thread.start()
        return True

    def submit(self, frame_bgr, capture_ts_ms, source_w, source_h, tracks):
        """
        Offer a frame for publishing. Drop-oldest: an unsent frame is discarded in
        favour of this one, so a slow subscriber cannot cause a growing delay.
        """
        if self._stop.is_set() or frame_bgr is None:
            return

        item = {
            "frame": frame_bgr,
            "capture_ts_ms": capture_ts_ms,
            "seq": _next_seq(),
            "source_w": source_w,
            "source_h": source_h,
            "tracks": tracks,
        }

        loop = self._loop
        with self._lock:
            self._pending = item

        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._notify)
            except RuntimeError:
                # Loop is shutting down; the pending item is simply dropped.
                pass

    def stop(self):
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(self._notify)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._thread = None

    # ----------------------------------------------------------------- private

    def _notify(self):
        wake = self._wake
        if wake is not None and not wake.done():
            wake.set_result(None)

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_forever())
        except Exception as exc:
            print(f"[Preview {self.cam_code}] event loop stopped: {exc}")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._loop = None

    def _mint_token(self, ttl_minutes=10):
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(f"ai-publisher-{self.cam_code}")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .with_ttl(datetime.timedelta(minutes=ttl_minutes))
        )
        return token.to_jwt()

    async def _run_forever(self):
        while not self._stop.is_set():
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Preview {self.cam_code}] session error: {exc}")
            if self._stop.is_set():
                break
            await asyncio.sleep(2.0)

    async def _session(self):
        token = self._mint_token()
        room = rtc.Room()
        await room.connect(
            self.url, token, rtc.RoomOptions(auto_subscribe=False)
        )
        print(f"[Preview {self.cam_code}] published to room '{self.room_name}'")

        source = None
        try:
            while not self._stop.is_set():
                item = await self._next_item()
                if item is None:
                    continue

                frame = item["frame"]
                h, w = frame.shape[:2]
                pub_w, pub_h = self._fit(w, h)

                if source is None:
                    source = rtc.VideoSource(pub_w, pub_h)
                    track = rtc.LocalVideoTrack.create_video_track(
                        "preview", source
                    )
                    await room.local_participant.publish_track(
                        track,
                        rtc.TrackPublishOptions(
                            source=rtc.TrackSource.SOURCE_CAMERA
                        ),
                    )
                    self._published_w, self._published_h = pub_w, pub_h
                    print(
                        f"[Preview {self.cam_code}] track live at {pub_w}x{pub_h} "
                        f"@ {self.target_fps}fps"
                    )

                small = cv2.resize(frame, (pub_w, pub_h), interpolation=cv2.INTER_AREA)
                rgb = np.ascontiguousarray(small[:, :, ::-1])  # BGR -> RGB
                vframe = rtc.VideoFrame(
                    pub_w, pub_h, rtc.VideoBufferType.RGB24, rgb.tobytes()
                )
                source.capture_frame(
                    vframe, timestamp_us=int(item["capture_ts_ms"] * 1000)
                )

                state = build_frame_state(
                    cam_code=self.cam_code,
                    seq=item["seq"],
                    capture_ts_ms=item["capture_ts_ms"],
                    published_w=pub_w,
                    published_h=pub_h,
                    source_w=item["source_w"],
                    source_h=item["source_h"],
                    tracks=item["tracks"],
                )
                await room.local_participant.publish_data(
                    json.dumps(state, separators=(",", ":")),
                    reliable=False,
                    topic=self.TOPIC_FRAME_STATE,
                )
        finally:
            try:
                await room.disconnect()
            except Exception:
                pass
            print(f"[Preview {self.cam_code}] disconnected from room '{self.room_name}'")

    def _fit(self, src_w, src_h):
        """Scale so the long edge is PREVIEW_WIDTH, keeping aspect ratio."""
        if src_w <= 0 or src_h <= 0:
            return self.target_width, max(1, self.target_width * 9 // 16)

        if src_w >= src_h:
            pub_w = self.target_width
            pub_h = max(2, int(round(src_h * (self.target_width / src_w))))
        else:
            pub_h = self.target_width
            pub_w = max(2, int(round(src_w * (self.target_width / src_h))))

        # H.264 needs even dimensions on many encoders.
        return pub_w - (pub_w % 2), pub_h - (pub_h % 2)

    async def _next_item(self, timeout=0.5):
        """
        Wait for the next frame, honouring PREVIEW_FPS and returning None when
        idle so the loop can re-check the stop flag.
        """
        self._wake = asyncio.get_running_loop().create_future()

        with self._lock:
            item = self._pending
            self._pending = None

        if item is not None:
            now = time.monotonic()
            if now - self._last_publish_ts < self._min_interval:
                return None  # too soon; this frame is intentionally dropped
            self._last_publish_ts = now
            return item

        try:
            await asyncio.wait_for(self._wake, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return None
