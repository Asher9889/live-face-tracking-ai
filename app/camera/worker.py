import os
import threading
import time
import traceback
import yaml
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.helper import is_stable_embedding_global, fast_filter, is_stable_embedding, expand_bbox, select_best_face, crop_with_margin, get_pose_name, now_ms, assign_face_to_person
from app.camera.types import CameraConfig, TrackState
from app.config import FRAME_RATE
from app.config.config import envConfig

from ultralytics import YOLO
from queue import Queue, Empty

from app.ai.insight_detector import InsightFaceEngine
from app.ai.face_mesh_engine import FaceLandmarkerEngine
from app.ai.runtime_device import resolve_torch_device, log_summary as log_device_summary
from app.camera.extract_person_roi import extract_person_roi
from app.camera.preview_publisher import PreviewPublisher
from app.camera.perf import PerfMeter
from app.camera.track_perf import track_perf
from app.camera.reid_encoder import install_reid_encoder
from app.camera.tracker_assets import resolve_tracker_config
from app.config.config import envConfig
from app.events.publisher import EventPublisher
from app.recognition import embedding_store, unknown_embedding_store
from app.recognition.unknown_creation_log import log_decision as log_unknown_decision, face_metrics
from app.tracking.track_manager import TrackEventEmitter
from app.database import redis_client

from app.camera.unique_face_builder import UniqueFaceRepresentationBuilder
from app.camera.payload_builder import build_unknown_payload

# unknown_manager = UnknownIdentityManager(unknown_embedding_store)

MIN_UNKNOWN_CREATION_QUALITY = float(envConfig.MIN_UNKNOWN_CREATION_QUALITY)
MIN_UNKNOWN_CREATE_FRAMES = int(envConfig.MIN_UNKNOWN_CREATE_FRAMES)

# Group-safety tuning: how far a person ROI may grow past the tracked body box.
ROI_PAD_X = int(os.getenv("ROI_PAD_X", "8"))
ROI_PAD_Y = int(os.getenv("ROI_PAD_Y", "20"))
# Required lead of the winning face over the runner-up. Above 0 the scene is
# genuinely ambiguous and this frame is dropped rather than risk a wrong match.
FACE_OWNERSHIP_MARGIN = float(os.getenv("FACE_OWNERSHIP_MARGIN", "0.06"))

# How many unlabelled tracks may enter the face pipeline on a single frame.
# The expensive work is rotated round-robin across frames so per-frame cost stays
# bounded and the preview path is never held up. 0 disables the limit.
RECOGNIZE_TRACKS_PER_FRAME = int(os.getenv("RECOGNIZE_TRACKS_PER_FRAME", "2"))

# How many already-matched tracks may be re-verified per frame. A matched identity
# is expensive to keep checking (same ROI/face pipeline), so it is rotated.
# 0 disables re-verification entirely.
VERIFY_TRACKS_PER_FRAME = int(os.getenv("VERIFY_TRACKS_PER_FRAME", "1"))
# A live face whose cosine similarity to the bound identity drops below this is
# treated as identity churn. The bound reference is the 3-sample centroid that
# originally matched. Matches against the employee gallery are accepted at 0.45
# (embedding_store.find_match), so a live face still above that is the same
# person; below it, the track probably changed owners.
MATCH_REVERIFY_LOWER = float(os.getenv("MATCH_REVERIFY_LOWER", "0.45"))
# Consecutive sub-threshold re-verifications before the identity is cleared.
# Guards against one bad frame (motion blur, lighting) killing a valid label.
MATCH_REVERIFY_CLEAR_AFTER = int(os.getenv("MATCH_REVERIFY_CLEAR_AFTER", "3"))

# UNRECOGNIZED_KNOWN: extra frames to retry matching before falling to UNKNOWN.
# When KNOWN collection fails (no match after 3 frames), instead of immediately
# becoming UNKNOWN (stricter gates), the track enters this retry buffer.
# Each frame re-runs find_match(). If matched → MATCHED_KNOWN restored.
# If all frames exhausted → COLLECTING_UNKNOWN (genuine unknown).
UNRECOGNIZED_MAX_FRAMES = int(os.getenv("UNRECOGNIZED_MAX_FRAMES", "5"))

# COLLECTING_KNOWN: if face crop 10-30px, upscale to 30px for recognition.
# Gives known employees at distance a chance to be matched.
COLLECT_UPSCALE_TARGET = int(os.getenv("COLLECT_UPSCALE_TARGET", "30"))
COLLECT_MIN_UPSCALE_SRC = int(os.getenv("COLLECT_MIN_UPSCALE_SRC", "10"))
# Consecutive frames with same match on upscaled face to confirm.
COLLECT_CONFIRM_FRAMES = int(os.getenv("COLLECT_CONFIRM_FRAMES", "4"))

# UNKNOWN: max consecutive quality gate failures before forcing creation.
# If quality gates (eye sharpness, eye visibility) keep rejecting, we still
# create the unknown after this many attempts so the UI doesn't show "5/5" forever.
UNKNOWN_FORCE_CREATE_AFTER = int(os.getenv("UNKNOWN_FORCE_CREATE_AFTER", "3"))

# BoT-SORT config, project-owned. The ultralytics default is used when this is
# blank, which means with_reid=False — exactly the blind-IoU association that
# swaps IDs in a crowd. Passing it explicitly keeps the tracker deterministic.
# `or` instead of a getenv default on purpose: TRACKER_YAML="" is set-but-empty,
# which would otherwise win over the project config and silently disable ReID.
TRACKER_YAML = os.getenv("TRACKER_YAML") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort.yaml")
# Person-detection confidence for the tracker (class 0 = person).
TRACK_CONF = float(os.getenv("TRACK_CONF", "0.25"))
# YOLO inference size for tracking. Larger catches distant faces in crowds,
# slower on CPU. Default 640 (ultralytics default).
TRACK_IMGSZ = int(os.getenv("TRACK_IMGSZ", "640"))

# How long (seconds) the app keeps a name bound to a track ID after the tracker
# loses it. The tracker holds lost tracks for track_buffer FRAMES (botsort yaml);
# at real fps F that is track_buffer/F seconds. This grace must not be shorter
# or the app retires an ID the tracker can still reactivate.
MIN_TRACK_GRACE = float(os.getenv("MIN_TRACK_GRACE", "1.0"))
# Extra safety margin added to the computed grace.
TRACK_GRACE_MARGIN = float(os.getenv("TRACK_GRACE_MARGIN", "0.5"))

PREVIEW_ENABLED = envConfig.PREVIEW_ENABLED

RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
RTSP_TIMEOUT_US = int(os.getenv("RTSP_TIMEOUT_US", "5000000"))
RTSP_BUFFER_SIZE = int(os.getenv("RTSP_BUFFER_SIZE", "1024000"))
CAPTURE_BACKOFF_INITIAL = float(os.getenv("CAPTURE_BACKOFF_INITIAL", "1.0"))
CAPTURE_BACKOFF_MAX = float(os.getenv("CAPTURE_BACKOFF_MAX", "30.0"))

PROFILE_WEBCAM = dict(
    yaw_threshold=20,
    pitch_threshold=25,
    roll_threshold=20,
    occlusion_threshold=0.50,
    ear_asymmetry_threshold=0.07,
    upscale_to=None,                  # no upscaling needed
    iris_radius_factor=0.035,         # min_iris_radius = face_size * factor
    min_iris_radius_ratio=0.45,
    max_iris_center_brightness=140.0,
    max_iris_brightness_asymmetry=60.0,
)
 
PROFILE_CCTV = dict(
    yaw_threshold=30,
    pitch_threshold=35,               # ceiling-mount: normal downward pitch
    roll_threshold=25,
    occlusion_threshold=0.70,         # eyeSquint is noisy at low resolution
    ear_asymmetry_threshold=0.13,     # 1px noise = 0.02-0.04 EAR at 80px
    upscale_to=160,                   # upscale before inference (landmarks improve dramatically)
    iris_radius_factor=0.025,         # smaller faces → smaller absolute iris
    min_iris_radius_ratio=0.40,       # looser ratio for small/noisy iris fitting
    max_iris_center_brightness=150.0, # slightly more tolerant for compressed CCTV frames
    max_iris_brightness_asymmetry=70.0,
    max_lateral_asymmetry=0.25,       # was 0.20 — allow mild yaw (13°) turns through
    eye_score_threshold=0.55,
)
 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

path = os.path.join(BASE_DIR,"../../models/facemesh/face_landmarker.task")
path = os.path.abspath(path)

# Explicit device: ultralytics would otherwise auto-select and this is shared by
# every camera thread, so the choice has to be visible at startup.
YOLO_DEVICE = resolve_torch_device()
# One YOLO per camera thread, created inside _camera_loop. A single shared model
# means one shared tracker + one shared ID counter across all cameras, so frames
# from different scenes interleave in the same Kalman filter and IDs bleed.
insight_engine = InsightFaceEngine()
publisher = EventPublisher(redis_client)
face_landmarker_engine = FaceLandmarkerEngine(model_path=path)
log_device_summary()

# Track buffer from the project botsort yaml, used to align the app's name-retention
# grace period with how long the tracker keeps a lost track alive.
if TRACKER_YAML and os.path.isfile(TRACKER_YAML):
    with open(TRACKER_YAML, "r") as _f:
        _TRACK_CFG = yaml.safe_load(_f) or {}
    TRACKER_YAML = os.path.abspath(TRACKER_YAML)
else:
    # No usable project tracker config. Falling back to the ultralytics default
    # means with_reid=False (IoU-only association), which silently swaps IDs in a
    # crowd — so say so loudly rather than degrading quietly.
    print(
        f"[Camera] WARNING: tracker config not found at '{TRACKER_YAML}'.\n"
        f"[Camera] Falling back to the ultralytics default with_reid=False. "
        f"Identity will churn; restore app/camera/botsort.yaml."
    )
    _TRACK_CFG = {}
    TRACKER_YAML = ""
TRACK_BUFFER_FRAMES = int(_TRACK_CFG.get("track_buffer", 30))

class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"

def log(cam, person_id, stage, msg):
    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}][{stage}] {msg}")


def format_unknown_label(unknown_id) -> str:
    """
    Human-readable label for an unidentified person: "Unknown ab12".

    The full unknown_id stays in the payload's `label` field; this only produces
    the short suffix that is safe to put on screen. The id format comes from the
    Node API, so it is not assumed to be hex, digits, or fixed length:
      - alphanumeric characters are taken from the END of the id, since that is
        where a uuid/ObjectId carries its random entropy
      - a pure-digit id is read from the end as digits
      - anything that leaves too little usable signal falls back to "Unknown"
    """
    if unknown_id is None:
        return "Unknown"

    s = str(unknown_id).strip()
    if not s:
        return "Unknown"

    # Trailing alphanumeric run, ignoring separators like "-" in a uuid.
    tail = ""
    for ch in reversed(s):
        if ch.isalnum():
            tail = ch + tail
            if len(tail) == 4:
                break
        else:
            break

    if not tail:
        return "Unknown"

    # A 4-char tail that is all one repeated character carries no information
    # (e.g. a uuid ending in "0000"), so widen the window instead of showing it.
    if len(tail) == 4 and len(set(tail)) == 1:
        wider = "".join(ch for ch in reversed(s) if ch.isalnum())[:8]
        tail = wider[-4:] if len(wider) >= 4 else wider

    # Too weak to distinguish anyone (all zeros/ones); not worth showing.
    if len(set(tail)) < 2:
        return "Unknown"

    return f"Unknown {tail.upper()}"


def pick_track_face(faces, person_bbox, cam, person_id):
    """
    Reduce a multi-face person ROI to the single face owned by this track.

    person_bbox must be the TIGHT YOLO box, not the expanded ROI, otherwise a
    neighbour's face looks equally central and the choice becomes a coin flip.
    """

    best, best_score, second_score = assign_face_to_person(faces, person_bbox)

    if best is None:
        log(cam, person_id, "OWNERSHIP", f"no face belongs to this track (candidates={len(faces)})")
        return []

    if second_score > 0 and (best_score - second_score) < FACE_OWNERSHIP_MARGIN:
        log(
            cam,
            person_id,
            "OWNERSHIP",
            f"ambiguous best={best_score:.3f} second={second_score:.3f}",
        )
        return []

    if len(faces) > 1:
        log(
            cam,
            person_id,
            "OWNERSHIP",
            f"picked from {len(faces)} faces (score={best_score:.3f}, next={second_score:.3f})",
        )

    return [best]


def seed_unknown_buffer(buffer):
    """
    Copy a retry/known buffer into the unknown buffer with one canonical schema.

    "pose_bucket" is the single pose key across the whole pipeline: every producer
    writes it and every consumer reads it. This helper guarantees the invariant at
    both seed points so a producer regression cannot raise KeyError deep inside the
    builder and silently freeze the buffer.

    Returns [] when nothing clears MIN_UNKNOWN_CREATION_QUALITY, which is the same
    as an empty buffer: builder.is_ready() is False and the track keeps collecting.
    """
    seeded = []

    for item in buffer:
        quality = item.get("quality")
        if quality is None or quality < MIN_UNKNOWN_CREATION_QUALITY:
            continue

        seeded.append({
            "embedding": item["embedding"],
            "quality": quality,
            # One key only. A frame with no usable pose is still worth keeping:
            # is_ready() only needs >= min_poses distinct buckets (1 by default),
            # and "unknown" keeps the entry usable instead of dropping a face.
            "pose_bucket": item.get("pose_bucket") or "unknown",
            "img": item.get("img"),
            "ts": item.get("ts", time.time()),
            "eye_quality": item.get("eye_quality"),
        })

    return seeded


def _open_capture(rtsp_url: str):
    if isinstance(rtsp_url, str) and rtsp_url.lower() == "webcam":
        print("[Camera] Using webcam source")
        cap = cv2.VideoCapture(0)
        print("Camera FPS:", "webcam", cap.get(cv2.CAP_PROP_FPS))
        return cap

    ffmpeg_options = [f"rtsp_transport;{RTSP_TRANSPORT}", f"stimeout;{RTSP_TIMEOUT_US}", f"buffer_size;{RTSP_BUFFER_SIZE}"]
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join(ffmpeg_options)

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    print("Camera FPS:", rtsp_url, cap.get(cv2.CAP_PROP_FPS))
    return cap

def start_camera_threads(cameras: List[CameraConfig]) -> None:  
    """
    Spawn one capture thread per camera.

    The BoT-SORT ReID checkpoint is resolved first, so a missing or
    undownloadable model aborts startup instead of killing every camera thread
    later from inside model.track(). Raises TrackerAssetError on failure.
    """
    global TRACKER_YAML

    if TRACKER_YAML:
        TRACKER_YAML = resolve_tracker_config(TRACKER_YAML)
        install_reid_encoder()

    print(f"[Camera] Starting {len(cameras)} camera threads...")

    for cam in cameras:
        thread = threading.Thread(
            target=_camera_loop,
            args=(cam,),
            daemon=True
        )
        thread.start()

        print(f"[Camera] Thread started → {cam.code}")


# =========================
# WORKER (REWRITTEN)
# =========================


def _camera_loop(cam: CameraConfig) -> None:
    print(f"[Camera] Worker started → {cam.code}")
    print("🔥 RUNNING _camera_loop")

    # Per-camera model. This gives each camera its own predictor, its own tracker
    # and its own ID sequence, so one camera's frames can never leak into another
    # camera's Kalman filter or recycle another camera's IDs.
    model = YOLO("yolov8n.pt")

    # Rotating cursor for round-robin face-pipeline scheduling.
    recognize_cursor = 0
    # Rotating cursor for matched-track re-verification.
    verify_cursor = 0

    # target_fps = int(FRAME_RATE)
    # interval = 1.0 / target_fps
    backoff = CAPTURE_BACKOFF_INITIAL

    track_event_emitter = TrackEventEmitter(publisher=publisher, gate_type=cam.gate_type)
    builder = UniqueFaceRepresentationBuilder()

    # Stage timing. One summary line every PERF_LOG_INTERVAL seconds, so the
    # cost of each pipeline stage is measured rather than guessed at. Kept out
    # of the model/face code itself: the hot path only does perf_counter reads.
    perf = PerfMeter(cam.code)
    # The stage wrappers below are shared by every person in the loop, so they
    # read the track they are currently servicing from here. Set at the top of
    # each person iteration; None means "not inside a track's work".
    cur_track = {"pid": None}

    # Wrappers rather than edited call sites: the timings must not change how
    # the pipeline behaves, and a with-block around each call would mean
    # re-indenting half the loop.
    def _face_detect(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return insight_engine.detect_and_generate_embedding(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            perf.record("face_det", ms)
            track_perf.observe(cam.code, cur_track["pid"], "face_det", ms)

    def _landmark(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return face_landmarker_engine.analyze(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            perf.record("landmark", ms)
            track_perf.observe(cam.code, cur_track["pid"], "landmark", ms)

    def _faiss(fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            perf.record("faiss", ms)
            track_perf.observe(cam.code, cur_track["pid"], "faiss", ms)

    def _http(fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            perf.record("http", ms)
            track_perf.observe(cam.code, cur_track["pid"], "http", ms)

    track_state = {}
    track_identity = {}
    # Display name for the preview overlay. track_identity stays the employee id
    # because that is what attendance events are keyed on.
    track_identity_name = {}
    # The exact embedding that won the match, used as the reference for
    # re-verification. Keyed by the same track id as track_identity.
    track_identity_embedding = {}
    # Consecutive re-verification failures per matched track.
    track_reverify_fails = {}
    track_known_buffer = {}
    track_unrecognized_buffer = {}  # {pid: [{"embedding", "quality", "pose", "img", "ts", "frame_count"}]}
    track_unknown_buffer = {}
    track_unknown_identity = {}
    track_unknown_meta = {}
    track_embedding_state = {}

    # For upscaled collection: {pid: {"matched_id": emp_id, "count": n, "frames": [...]}}
    track_upscale_collect = {}

    # Consecutive unknown quality gate failures per track.
    # After UNKNOWN_FORCE_CREATE_AFTER consecutive failures, force creation.
    track_unknown_quality_fails = {}

    # For RTSP Thread
    frame_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    frame_count = 0
    frame_errors = 0

    def _reader(cap):
        while not stop_event.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()  # drop old frame
                except:
                    pass

            # Stamp at decode, before any inference. Everything downstream derives
            # its latency from this value, so it must not be taken later.
            frame_queue.put((frame, time.time()))
            # Decode rate, counted where ffmpeg hands frames over. Split from
            # the loop's tick(): dec_fps ~25 with fps ~1 means the pipeline is
            # slow; both low means RTSP/ffmpeg itself is starving the loop.
            perf.tick_decode()


    while True:
        cap = _open_capture(cam.rtsp_url)

        if not cap.isOpened():
            sleep_time = min(backoff, CAPTURE_BACKOFF_MAX)
            print(f"[Camera] ❌ Connect failed → {cam.code}; retry in {sleep_time:.1f}s")
            time.sleep(sleep_time)
            backoff = min(backoff * 2, CAPTURE_BACKOFF_MAX)
            continue

        stop_event.clear()

        reader_thread = threading.Thread(
            target=_reader,
            args=(cap,),
            daemon=True
        )
        reader_thread.start()

        backoff = CAPTURE_BACKOFF_INITIAL

        # One LiveKit session per capture connection. Created here (not above) so a
        # reconnect tears the old session down and starts a fresh one.
        preview = None
        if PREVIEW_ENABLED:
            preview = PreviewPublisher(cam.code)
            if not preview.start():
                print(f"[Camera] {cam.code}: preview publishing disabled")
                preview = None

        # Real-time measure of this camera's processing rate, for TTL alignment.
        _fps = 0.0
        _last_ts = None
        # Start of the current frame's work; None until the first frame lands.
        frame_start = None

        # last_processed = 0.0

        while True:
            # if not cap.grab():
            #     print(f"[Camera] ⚠️ Stream lost → {cam.code}; reconnecting")
            #     cap.release()
            #     break

            # now = time.time()
            # if now - last_processed < interval:
            #     continue

            # ret, frame = cap.retrieve()
            # if not ret or frame is None:
            #     continue

            # last_processed = now

            try:
                frame, captured_at = frame_queue.get(timeout=5)
                if frame is None or frame.size == 0:
                    continue

                perf.tick()
                # How stale the frame already is when we pick it up: decode +
                # queue wait. This is latency no downstream stage caused, so it
                # is checked first when frame age looks wrong.
                perf.record("queue_wait", (time.time() - captured_at) * 1000.0)
                # Cost of the previous frame, sampled here rather than at the
                # end so the dozens of `continue` paths are measured too.
                if frame_start is not None:
                    perf.record("frame_total", (time.perf_counter() - frame_start) * 1000.0)
                    perf.flush()
                # Periodic per-track rows for whatever is still alive; cheap
                # because it only writes tracks whose interval has elapsed.
                track_perf.flush_due()
                frame_start = time.perf_counter()

                # Measure the real per-camera processing rate (EMA). The BoT-SORT
                # track buffer is expressed in FRAMES (ultralytics hardcodes
                # frame_rate=30), so the app's name-retention grace must be
                # track_buffer / actual_fps seconds for the two to retire an ID at
                # the same time. Without this a name stays bound to an ID the
                # tracker has already recycled.
                _now = time.time()
                if _last_ts is not None:
                    _inst = 1.0 / max(_now - _last_ts, 1e-6)
                    _fps = _fps * 0.9 + _inst * 0.1 if _fps > 0 else _inst
                _last_ts = _now
            except Empty:
                # A camera that stops producing frames would otherwise go silent
                # exactly when it matters most: emit whatever the window has
                # (decode counts included) before tearing the capture down.
                perf.flush(force=True)
                print(f"[Camera] ⚠️ No frames → {cam.code}; reconnecting")
                # cap.release()
                # stop_event.set()
                # break
                stop_event.set()

                reader_thread.join(timeout=2)   # wait for thread to exit safely

                cap.release()

                # Tear the LiveKit session down so a reconnect starts clean rather
                # than leaving a stale published track behind.
                if preview is not None:
                    preview.stop()

                track_state.clear()
                track_identity.clear()
                track_identity_name.clear()
                track_identity_embedding.clear()
                track_reverify_fails.clear()
                track_known_buffer.clear()
                track_unknown_buffer.clear()
                track_unknown_identity.clear()
                track_unknown_meta.clear()
                track_embedding_state.clear()

                break

            try:
                frame_h, frame_w = frame.shape[:2]

                # Detect + BoT-SORT + ReID. ReID reports its own split on the
                # global [PERF reid] line, so "track" minus "reid" is the
                # detector+association share.
                t_track = time.perf_counter()
                results = model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=TRACK_CONF,
                    verbose=False,
                    device=YOLO_DEVICE,
                    imgsz=TRACK_IMGSZ,
                    **({"tracker": TRACKER_YAML} if TRACKER_YAML else {}),
                )
                perf.record("track", (time.perf_counter() - t_track) * 1000.0)

                # A frame with nobody in it is still a frame. Detection decides only
                # what metadata rides along with the picture, never whether the
                # picture is published, so the preview stays continuous through empty
                # scenes and the face pipeline simply has nothing to do this frame.
                detections = results[0].boxes
                has_persons = detections is not None and detections.id is not None

                if has_persons:
                    boxes = detections.xyxy.cpu().numpy()
                    ids = detections.id.int().cpu().numpy()
                else:
                    boxes = np.empty((0, 4), dtype=np.float32)
                    ids = np.empty((0,), dtype=np.int64)

                # People visible this frame. Compared against face_calls in the
                # summary to show whether recognition is running for everyone.
                perf.add("persons", len(ids))

                # ---------------------------------------------------------------
                # PREVIEW PUBLISH — every frame, cheap, before the face pipeline.
                #
                # The label attached here is whatever recognition resolved on an
                # earlier frame. That is intentional: identity is track state, not
                # per-frame work, and publishing before the expensive stage keeps
                # preview latency independent of recognition cost.
                # ---------------------------------------------------------------
                t_pub = time.perf_counter()
                if preview is not None:
                    preview_tracks = []
                    for pid, bbox in zip(ids, boxes):
                        pid = int(pid)
                        label = None
                        label_name = None
                        confidence = 0.0
                        if pid in track_identity:
                            label = track_identity[pid]
                            label_name = track_identity_name.get(pid) or str(label)
                            confidence = 1.0
                        elif pid in track_unknown_identity:
                            label = track_unknown_identity[pid]
                            # Unidentified people have no name, and the raw uuid is
                            # meaningless to read, so show a short distinguishing
                            # suffix instead. The full id stays in `label`.
                            label_name = format_unknown_label(label)
                            confidence = 0.9

                        raw_state = track_state.get(pid)
                        track_entry = {
                            "track_id": pid,
                            "bbox": [float(x) for x in bbox],
                            "state": raw_state.value if raw_state is not None else None,
                            "label": label,
                            "label_name": label_name,
                            "label_confidence": confidence,
                            # Re-verification (STAGE 0) is active, so a label is
                            # no longer guaranteed for the life of the track: a
                            # track that swaps owners is cleared back to
                            # COLLECTING_KNOWN. None means "no expiry advertised".
                            "label_expires_at": None,
                        }
                        # Add buffer_size for COLLECTING_UNKNOWN to show progress
                        if raw_state == TrackState.COLLECTING_UNKNOWN:
                            buf = track_unknown_buffer.get(pid)
                            if buf:
                                track_entry["buffer_size"] = len(buf)
                        preview_tracks.append(track_entry)

                    preview.submit(
                        frame_bgr=frame,
                        capture_ts_ms=int(captured_at * 1000),
                        source_w=frame_w,
                        source_h=frame_h,
                        tracks=preview_tracks,
                    )
                    perf.record("publish", (time.perf_counter() - t_pub) * 1000.0)
                    # Frames that actually reached LiveKit vs frames processed;
                    # a gap here means publishing dropped, not the pipeline.
                    perf.add("published")
                    # The number that matters most: capture -> hand-off to
                    # LiveKit. If this grows over time the system is falling
                    # behind; if it is flat but high, a stage above is slow.
                    perf.record("age", (time.time() - captured_at) * 1000.0)

                if not has_persons:
                    # Nothing to recognise, and nothing to retire: a short empty gap
                    # must not end tracks that are still on screen in the preview.
                    continue

                # Grace for retired track IDs, aligned to the tracker's buffer:
                # the tracker keeps a lost track alive for TRACK_BUFFER_FRAMES
                # processing frames; at real fps F that is TRACK_BUFFER_FRAMES/F
                # seconds. Retire the app-side identity no earlier than that, or
                # the name detaches while the tracker still owns and can recycle
                # the ID.
                _grace = max(
                    MIN_TRACK_GRACE,
                    TRACK_BUFFER_FRAMES / max(_fps, 1.0) + TRACK_GRACE_MARGIN,
                )
                lost = track_event_emitter.cleanup_lost_tracks(cam.code, ids.tolist(), grace=_grace)

                for tid in lost:
                    # Final per-track timing row before the bookkeeping goes away.
                    track_perf.retire(cam.code, tid)
                    track_state.pop(tid, None)
                    track_identity.pop(tid, None)
                    track_identity_name.pop(tid, None)
                    track_identity_embedding.pop(tid, None)
                    track_reverify_fails.pop(tid, None)
                    track_known_buffer.pop(tid, None)
                    track_unrecognized_buffer.pop(tid, None)
                    track_unknown_buffer.pop(tid, None)
                    track_unknown_identity.pop(tid, None)
                    track_unknown_meta.pop(tid, None)
                    track_embedding_state.pop(tid, None)
                    track_upscale_collect.pop(tid, None)
                    track_unknown_quality_fails.pop(tid, None)

                # ---------------------------------------------------------------
                # FACE PIPELINE SCHEDULING
                #
                # Only unlabelled tracks need the expensive path, and only a few of
                # them per frame. Work rotates round-robin so every track is serviced
                # regularly while per-frame cost stays bounded.
                # ---------------------------------------------------------------
                eligible = [
                    (int(pid), bbox)
                    for pid, bbox in zip(ids, boxes)
                    if int(pid) not in track_identity
                ]

                if eligible and RECOGNIZE_TRACKS_PER_FRAME > 0 and len(eligible) > RECOGNIZE_TRACKS_PER_FRAME:
                    start = recognize_cursor % len(eligible)
                    process_ids = {
                        eligible[(start + i) % len(eligible)][0]
                        for i in range(RECOGNIZE_TRACKS_PER_FRAME)
                    }
                    recognize_cursor = (start + RECOGNIZE_TRACKS_PER_FRAME) % len(eligible)
                else:
                    process_ids = {pid for pid, _ in eligible}

                # Matched tracks join the same round-robin so their identity is
                # periodically re-verified against the embedding that won the
                # match. A track that has silently swapped owners now gets caught
                # instead of carrying the wrong name forever.
                # VERIFY_TRACKS_PER_FRAME <= 0 disables re-verification outright:
                # no matched track is ever re-processed, and labels then persist
                # for the life of the track.
                verify_candidates = (
                    [
                        (int(pid), bbox)
                        for pid, bbox in zip(ids, boxes)
                        if int(pid) in track_identity
                    ]
                    if VERIFY_TRACKS_PER_FRAME > 0
                    else []
                )
                if len(verify_candidates) > VERIFY_TRACKS_PER_FRAME:
                    start = verify_cursor % len(verify_candidates)
                    process_ids.update(
                        verify_candidates[(start + i) % len(verify_candidates)][0]
                        for i in range(VERIFY_TRACKS_PER_FRAME)
                    )
                    verify_cursor = (start + VERIFY_TRACKS_PER_FRAME) % len(verify_candidates)
                else:
                    # Fewer candidates than the per-frame budget: verify them all.
                    process_ids.update(pid for pid, _ in verify_candidates)

                for person_id, bbox in zip(ids, boxes):

                    person_id = int(person_id)
                    cur_track["pid"] = person_id

                    # Keep track lifecycle state updated so emit-once events are not dropped.
                    track_event_emitter.update_track(
                        cam.code,
                        person_id,
                        bbox,
                        int(captured_at * 1000),
                        frame_w,
                        frame_h
                    )

                    if person_id not in process_ids:
                        continue

                    if person_id not in track_state:
                        track_state[person_id] = TrackState.COLLECTING_KNOWN
                        log(cam, person_id, "STATE", "INIT → COLLECTING_KNOWN")
                    # else:
                    #     # 🔥 IMPORTANT DEBUG
                    #     log(cam, person_id, "DEBUG", f"EXISTING STATE → {track_state[person_id]}")

                    state = track_state[person_id]
                    # One call per serviced track per frame: logs transitions,
                    # and emits a single row when the identity resolves.
                    track_perf.state(
                        cam.code,
                        person_id,
                        state.value,
                        track_identity.get(person_id) or track_unknown_identity.get(person_id),
                    )

                    # -------------------------
                    # ROI + FACE DETECTION
                    # -------------------------
                    x1, y1, x2, y2 = expand_bbox(bbox, frame_w, frame_h)

                    # Clamp the horizontal pad so a standing neighbour's face cannot
                    # enter this track's ROI. Vertical pad keeps head room.
                    pad_x = min(ROI_PAD_X, (x2 - x1) * 0.06)
                    pad_y = min(ROI_PAD_Y, (y2 - y1) * 0.08)

                    roi_data = extract_person_roi(
                        frame, person_id, np.array([x1, y1, x2, y2]), pad_x=pad_x, pad_y=pad_y
                    )
                    if roi_data is None:
                        continue

                    _, roi, offset = roi_data

                    perf.add("face_calls", 1)
                    faces = _face_detect(roi, offset, cam.code)

                    if not faces:
                        continue

                    if len(faces) > 1:
                        # Keep only the face owned by THIS track. Discarding the whole
                        # ROI used to make two neighbours veto each other, so the entire
                        # cluster went unrecognised for as long as they stood together.
                        faces = pick_track_face(faces, bbox, cam, person_id)
                        if not faces:
                            continue

                    # Filter bad faces after detection and log rejection reason.
                    filtered_faces = []
                    required_min_width = envConfig.MIN_RECOGNITION_FACE_WIDTH
                    if state in (TrackState.COLLECTING_UNKNOWN, TrackState.UPDATING_UNKNOWN):
                        required_min_width = envConfig.MIN_UNKNOWN_REG_FACE_WIDTH

                    for f in faces:
                        # For COLLECTING_KNOWN, compute face width from bbox for upscale check
                        face_min_width = required_min_width
                        if state == TrackState.COLLECTING_KNOWN:
                            fx1, fy1, fx2, fy2 = map(int, f["bbox"])
                            face_w = fx2 - fx1
                            if face_w < COLLECT_UPSCALE_TARGET and face_w >= COLLECT_MIN_UPSCALE_SRC:
                                face_min_width = COLLECT_MIN_UPSCALE_SRC

                        filter_result = fast_filter(f, min_width=face_min_width)

                        if isinstance(filter_result, dict) and not filter_result.get("status", False):
                            reason = filter_result.get("reason", "unknown")
                            details = filter_result.get("details", "")
                            print(
                                f"[{now_ms()}][Camera {cam.code}][Person {person_id}][FAST_FILTER] "
                                f"reason={reason} details={details}"
                            )
                            continue

                        filtered_faces.append(f)

                    faces = filtered_faces
                    if not faces:
                        continue

                    # -------------------------
                    # QUALITY FILTER
                    # -------------------------
                    valid_faces = []
                    for f in faces:
                        # if f["score"] < envConfig.SCRFD_THRESHOLD:
                        #     continue

                        x1, y1, x2, y2 = map(int, f["bbox"])
                        # face_img = frame[y1:y2, x1:x2]

                        embedding = f["embedding"]

                        # 🔥 GLOBAL stability check (once per loop).
                        # Skipped for MATCHED_KNOWN: this gate compares against the
                        # track's own EMA reference, which belongs to the OLD owner.
                        # In a crowd an identity-swapped track must be allowed to
                        # reach the re-verification stage, where it is compared to
                        # the exact bound embedding and cleared if incongruent.
                        if state != TrackState.MATCHED_KNOWN and not is_stable_embedding_global(track_embedding_state, person_id, embedding):
                            print(f"[{now_ms()}][Camera {cam.code}] Unstable embedding → person_id={person_id}")
                            continue


                        face_img = crop_with_margin(frame, x1, y1, x2, y2, margin=0.2)

                        if face_img.size == 0:
                            continue

                        f["face_img"] = face_img

                        analysis = _landmark(face_img)
                        # is_valid = face_landmarker_engine.is_valid_face(analysis, cam.code) 
                        mp_score = face_landmarker_engine.score_face(analysis)

                        if not analysis.get("valid"):
                            mp_score = 0.3   # fallback, not assumption, just degradation
                        # if mp_score == 0:
                        #     mp_score = 0.3  # fallback, not assumption, just degradation

                        # if not is_valid:
                        # #     # print(f"[Camera {cam.code}] Face rejected by FaceLandmarker is_valid_face check")
                        # #     continue
                        quality = insight_engine.compute_face_quality(f, face_img, analysis)
                        if quality < 0:
                            continue

                        final_quality = ( 0.6 * quality + 0.4 * mp_score ) 
                        print(f"[{now_ms()}][Camera {cam.code}] Face quality → person_id={person_id}, quality={quality:.3f}, mp_score={mp_score:.3f}, final_quality={final_quality:.3f}")
                        if final_quality < 0.35:
                            continue

                        f["quality"] = final_quality
                        # Carried so the STAGE 2 audit record can report the value for
                        # the face that actually won selection, not just this one.
                        f["eye_sharpness"] = analysis.get("eye_sharpness")
                        valid_faces.append(f)

                    if not valid_faces:
                        continue

                    best_face = select_best_face(valid_faces)
                    if best_face is None:
                        continue

                    bx1, by1, bx2, by2 = map(int, best_face["bbox"])
                    best_face_width = bx2 - bx1

                    embedding = best_face["embedding"]
                    quality = best_face["quality"]

                    face_img = best_face.get("face_img")
                    if face_img is None or face_img.size == 0:
                        continue

                    pose = get_pose_name(best_face.get("pose", [None])[0]) or "unknown"

                    # =====================================================
                    # 🔵 STAGE 0: RE-VERIFY (matched tracks only)
                    #
                    # A MATCHED_KNOWN track normally skips the face pipeline, so
                    # its label persists forever even if the track has silently
                    # swapped to another person (the exact source of wrong names
                    # in a crowd). Tracks in this state run once per rotation:
                    # the live embedding is compared to the exact centroid the
                    # identity was matched on. Sustained divergence clears the
                    # binding so the track re-recognises instead of carrying a
                    # wrong name.
                    # =====================================================
                    if state == TrackState.MATCHED_KNOWN:
                        ref = track_identity_embedding.get(person_id)
                        if ref is None:
                            continue
                        sim = float(np.dot(embedding, ref) / (np.linalg.norm(embedding) * np.linalg.norm(ref) + 1e-9))
                        fails = track_reverify_fails.get(person_id, 0)

                        if sim >= MATCH_REVERIFY_LOWER:
                            track_reverify_fails[person_id] = 0
                            continue

                        fails += 1
                        track_reverify_fails[person_id] = fails
                        log(
                            cam, person_id, "REVERIFY",
                            f"low sim={sim:.3f} ({fails}/{MATCH_REVERIFY_CLEAR_AFTER}) → bound={track_identity.get(person_id)}",
                        )
                        if fails < MATCH_REVERIFY_CLEAR_AFTER:
                            continue

                        # Identity no longer holds. Unbind and let the standard
                        # KNOWN collection re-identify from scratch.
                        log(
                            cam, person_id, "REVERIFY",
                            f"CLEAR identity={track_identity.pop(person_id, None)} "
                            f"({track_identity_name.pop(person_id, None)})",
                        )
                        track_identity_embedding.pop(person_id, None)
                        track_reverify_fails.pop(person_id, None)
                        track_state[person_id] = TrackState.COLLECTING_KNOWN
                        track_known_buffer.pop(person_id, None)
                        track_unrecognized_buffer.pop(person_id, None)
                        track_unknown_buffer.pop(person_id, None)
                        # Reset the stability reference so the next face starts a
                        # clean re-collection instead of comparing to the old owner.
                        track_embedding_state.pop(person_id, None)
                        continue

                    # =====================================================
                    # 🔵 STAGE 1: KNOWN — WITH UPSCALE FOR SMALL FACES
                    # =====================================================
                    if state == TrackState.COLLECTING_KNOWN:

                        # --- UPSCALE SMALL FACES (10-30px → 30px) ---
                        upscaled = False
                        if best_face_width < COLLECT_UPSCALE_TARGET and best_face_width >= COLLECT_MIN_UPSCALE_SRC:
                            scale = COLLECT_UPSCALE_TARGET / best_face_width
                            new_w = int((bx2 - bx1) * scale)
                            new_h = int((by2 - by1) * scale)
                            face_img_up = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            
                            # Re-extract embedding from upscaled face
                            perf.add("upscales", 1)
                            faces_up = _face_detect(face_img_up, (0, 0), cam.code)
                            if faces_up:
                                emb_up = faces_up[0]["embedding"]
                                embedding = emb_up
                                upscaled = True
                                log(cam, person_id, "UPSCALE", f"{best_face_width}px → {new_w}px for collection")

                        # Try match on (possibly upscaled) embedding
                        match = _faiss(embedding_store.find_match, embedding)

                        if match:
                            emp_id = match["employee_id"]
                            uc = track_upscale_collect.get(person_id, {"matched_id": None, "count": 0, "frames": []})
                            
                            if uc["matched_id"] == emp_id:
                                uc["count"] += 1
                                uc["frames"].append(embedding)
                            else:
                                uc["matched_id"] = emp_id
                                uc["count"] = 1
                                uc["frames"] = [embedding]
                            
                            track_upscale_collect[person_id] = uc

                            # Check confirmation
                            if uc["count"] >= COLLECT_CONFIRM_FRAMES:
                                # 4 consecutive frames matched same person → CONFIRMED
                                track_identity[person_id] = emp_id
                                track_identity_name[person_id] = match["name"]
                                track_state[person_id] = TrackState.MATCHED_KNOWN
                                # Use the last upscaled embedding as reference
                                track_identity_embedding[person_id] = uc["frames"][-1]
                                track_reverify_fails[person_id] = 0
                                track_upscale_collect.pop(person_id, None)
                                track_known_buffer.pop(person_id, None)

                                track_event_emitter.recognition_confirmed(
                                    cam.code, person_id, emp_id, match["similarity"]
                                )
                                log(cam, person_id, "UPSCALE", f"CONFIRMED → MATCHED_KNOWN {emp_id} after {uc['count']} frames")
                                continue

                            # Matched but not yet confirmed — keep collecting
                            continue

                        # --- NO MATCH on this frame ---
                        if upscaled:
                            # Was upscaled but no match: reset counter
                            track_upscale_collect.pop(person_id, None)
                        else:
                            # Normal size face but no match: original COLLECTING_KNOWN logic
                            buffer = track_known_buffer.get(person_id, [])
                            buffer.append({
                                "embedding": embedding,
                                "quality": quality,
                                "pose_bucket": pose,
                                "img": face_img,
                                "ts": time.time()
                            })
                            buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)[:3]
                            track_known_buffer[person_id] = buffer

                            if len(buffer) < 3:
                                continue

                            # Original 3-frame centroid match
                            emb = np.array([x["embedding"] for x in buffer])
                            w = np.array([x["quality"] for x in buffer])
                            final = np.average(emb, axis=0, weights=w)
                            final /= np.linalg.norm(final)

                            match = _faiss(embedding_store.find_match, final)
                            if match:
                                track_identity[person_id] = match["employee_id"]
                                track_identity_name[person_id] = match["name"]
                                track_state[person_id] = TrackState.MATCHED_KNOWN
                                track_identity_embedding[person_id] = final
                                track_reverify_fails[person_id] = 0
                                track_known_buffer.pop(person_id, None)

                                track_event_emitter.recognition_confirmed(
                                    cam.code,
                                    person_id,
                                    match["employee_id"],
                                    match["similarity"]
                                )
                                log(cam, person_id, "KNOWN", f"MATCHED → {match['employee_id']}")
                                continue

                            # Original fallback to UNRECOGNIZED_KNOWN
                            track_state[person_id] = TrackState.UNRECOGNIZED_KNOWN
                            track_unrecognized_buffer[person_id] = seed_unknown_buffer(buffer)
                            track_known_buffer.pop(person_id, None)

                            log(cam, person_id, "STATE", f"→ UNRECOGNIZED_KNOWN (retry frames={UNRECOGNIZED_MAX_FRAMES})")
                            continue

                    # =====================================================
                    # 🔵 STAGE 1.5: UNRECOGNIZED_KNOWN — retry buffer before UNKNOWN
                    # =====================================================
                    elif state == TrackState.UNRECOGNIZED_KNOWN:
                        buffer = track_unrecognized_buffer.get(person_id, [])
                        buffer.append({
                            "embedding": embedding,
                            "quality": quality,
                            # Key must be "pose_bucket": every consumer of the
                            # unknown buffer (UniqueFaceRepresentationBuilder.is_ready,
                            # _trim, build_unknown_payload) reads item["pose_bucket"].
                            # Writing "pose" here raised KeyError('pose_bucket') on the
                            # next COLLECTING_UNKNOWN frame, which killed the track
                            # before builder.add() so the buffer never grew past its
                            # seeded size and no unknown was ever registered.
                            "pose_bucket": pose,
                            "img": face_img,
                            "ts": time.time()
                        })
                        track_unrecognized_buffer[person_id] = buffer

                        # Try match on every frame in this buffer
                        if len(buffer) >= 2:  # need at least 2 for weighted avg
                            emb = np.array([x["embedding"] for x in buffer])
                            w = np.array([x["quality"] for x in buffer])
                            final = np.average(emb, axis=0, weights=w)
                            final /= np.linalg.norm(final)

                            match = _faiss(embedding_store.find_match, final)
                            if match:
                                track_identity[person_id] = match["employee_id"]
                                track_identity_name[person_id] = match["name"]
                                track_state[person_id] = TrackState.MATCHED_KNOWN
                                track_identity_embedding[person_id] = final
                                track_reverify_fails[person_id] = 0

                                track_event_emitter.recognition_confirmed(
                                    cam.code,
                                    person_id,
                                    match["employee_id"],
                                    match["similarity"]
                                )

                                log(cam, person_id, "UNRECOGNIZED", f"RECOVERED → MATCHED_KNOWN {match['employee_id']} after {len(buffer)} retry frames")
                                track_unrecognized_buffer.pop(person_id, None)
                                continue

                        # Check if max retry frames exhausted
                        if len(buffer) >= UNRECOGNIZED_MAX_FRAMES:
                            track_state[person_id] = TrackState.COLLECTING_UNKNOWN
                            track_unknown_buffer[person_id] = seed_unknown_buffer(buffer)
                            track_unrecognized_buffer.pop(person_id, None)

                            log(cam, person_id, "STATE", f"→ COLLECTING_UNKNOWN after {UNRECOGNIZED_MAX_FRAMES} unrecognized frames")
                            # Log the seeding decision
                            log_unknown_decision(
                                "unknown_buffer_seeded",
                                cam.code,
                                person_id,
                                role=cam.camera_role,
                                buffer_size=len(track_unknown_buffer[person_id]),
                                **face_metrics(
                                    analysis,
                                    quality,
                                    final_quality,
                                    best_face_width,
                                ),
                            )
                            continue

                        # Still within retry budget — keep collecting
                        continue

                                        # =====================================================
                    # 🔵 STAGE 2: UNKNOWN
                    # =====================================================
                    elif state == TrackState.COLLECTING_UNKNOWN:
                        camera_role = cam.camera_role
                        buffer = track_unknown_buffer.get(person_id, [])

                        # ---------------- QUALITY GATES ---------------- #
                        #
                        # Each gate reports its verdict, then ONE decision point below
                        # resolves reject-vs-force-create. Previously every gate owned its
                        # own `continue` and its own counter increment, which made the
                        # force-create path unreachable:
                        #   - gate 2's `else: force_create = False` cleared a force_create
                        #     that gate 1 had just earned,
                        #   - the gate-3 `for ... else` always ran (no `break` existed) and
                        #     cancelled the force_create set on its last iteration,
                        #   - a frame failing two eye metrics counted as two consecutive
                        #     failures instead of one.
                        # The audit log now gets an explicit "force_create" stage so a
                        # forced registration is as traceable as a clean one.

                        # Consecutive frames that failed at least one gate. Reset as soon
                        # as a frame passes all three.
                        gate_failures = []
                        force_create = False

                        # ---------------- GATE 1: FACE WIDTH ---------------- #
                        if best_face_width < envConfig.MIN_UNKNOWN_REG_FACE_WIDTH:
                            gate_failures.append("face_width")
                            log_filter_stage(
                                cam.code, person_id, camera_role,
                                filter_stage="face_width",
                                passed=False,
                                threshold_used=envConfig.MIN_UNKNOWN_REG_FACE_WIDTH,
                                measured_value=best_face_width,
                                buffer=buffer,
                                builder=builder,
                                force_create=False
                            )
                            log_decision(
                                "unknown_rejected",
                                cam.code,
                                person_id,
                                reason="face_width",
                                threshold=envConfig.MIN_UNKNOWN_REG_FACE_WIDTH,
                                measured_value=best_face_width,
                                **face_metrics(
                                    analysis, quality, final_quality, best_face_width,
                                    buffer=buffer, builder=builder, camera_role=camera_role
                                ),
                            )
                        else:
                            log_filter_stage(
                                cam.code, person_id, camera_role,
                                filter_stage="face_width",
                                passed=True,
                                threshold_used=envConfig.MIN_UNKNOWN_REG_FACE_WIDTH,
                                measured_value=best_face_width,
                                buffer=buffer,
                                builder=builder,
                                force_create=False
                            )

                        # ---------------- GATE 2: EYE SHARPNESS ---------------- #
                        eye_sharpness = best_face.get("eye_sharpness")
                        min_sharpness = envConfig.MIN_UNKNOWN_EYE_SHARPNESS
                        if (
                            envConfig.ENFORCE_UNKNOWN_EYE_SHARPNESS
                            and eye_sharpness is not None
                            and eye_sharpness < min_sharpness
                        ):
                            gate_failures.append("eye_sharpness")
                            log_filter_stage(
                                cam.code, person_id, camera_role,
                                filter_stage="eye_sharpness",
                                passed=False,
                                threshold_used=min_sharpness,
                                measured_value=eye_sharpness,
                                buffer=buffer,
                                builder=builder,
                                force_create=False
                            )
                            log_decision(
                                "unknown_rejected",
                                cam.code,
                                person_id,
                                reason="eye_sharpness",
                                threshold=min_sharpness,
                                measured_value=eye_sharpness,
                                **face_metrics(
                                    analysis, quality, final_quality, best_face_width,
                                    buffer=buffer, builder=builder, camera_role=camera_role
                                ),
                            )
                        else:
                            log_filter_stage(
                                cam.code, person_id, camera_role,
                                filter_stage="eye_sharpness",
                                passed=True,
                                threshold_used=min_sharpness,
                                measured_value=eye_sharpness,
                                buffer=buffer,
                                builder=builder,
                                force_create=False
                            )

                        # ---------------- GATE 3: EYE VISIBILITY ---------------- #
                        core = analysis.get("iris_core_brightness")
                        contrast = analysis.get("iris_contrast_min")
                        if core and core > 0 and contrast is not None:
                            iris_ratio = contrast / core
                        else:
                            iris_ratio = None

                        eye_dist_ratio = analysis.get("eye_dist_ratio")
                        yaw = abs(analysis.get("yaw")) if analysis.get("yaw") is not None else None
                        min_iris_ratio = envConfig.MIN_UNKNOWN_IRIS_CONTRAST_RATIO
                        min_eye_dist = envConfig.MIN_UNKNOWN_EYE_DIST_RATIO
                        max_yaw = envConfig.MAX_UNKNOWN_REG_YAW

                        eye_visibility_passed = True
                        if envConfig.ENFORCE_UNKNOWN_EYE_VISIBILITY:
                            reasons = []
                            if yaw is not None and yaw > max_yaw:
                                reasons.append(("yaw", yaw, max_yaw))
                            if iris_ratio is not None and iris_ratio < min_iris_ratio:
                                reasons.append(("iris_contrast_ratio", iris_ratio, min_iris_ratio))
                            elif iris_ratio is None:
                                reasons.append(("iris_contrast_ratio", None, min_iris_ratio))
                            if eye_dist_ratio is not None and eye_dist_ratio < min_eye_dist:
                                reasons.append(("eye_dist_ratio", eye_dist_ratio, min_eye_dist))
                            elif eye_dist_ratio is None:
                                reasons.append(("eye_dist_ratio", None, min_eye_dist))

                            if reasons:
                                eye_visibility_passed = False
                                detail = ", ".join(
                                    f"{name}={value}" + (f" > {limit}" if name == "yaw" else f" < {limit}")
                                    for name, value, limit in reasons
                                )
                                log(cam, person_id, "UNKNOWN", f"REJECT eyes not visible: {detail}")
                                for name, value, limit in reasons:
                                    gate_failures.append(f"eye_visibility_{name}")
                                    log_filter_stage(
                                        cam.code, person_id, camera_role,
                                        filter_stage=f"eye_visibility_{name}",
                                        passed=False,
                                        threshold_used=limit,
                                        measured_value=value,
                                        buffer=buffer,
                                        builder=builder,
                                        force_create=False
                                    )
                                    log_decision(
                                        "unknown_rejected",
                                        cam.code,
                                        person_id,
                                        reason=f"eye_visibility_{name}",
                                        threshold=limit,
                                        measured_value=value,
                                        **face_metrics(
                                            analysis, quality, final_quality, best_face_width,
                                            buffer=buffer, builder=builder, camera_role=camera_role
                                        ),
                                    )

                        # Log eye visibility verdict
                        log_filter_stage(
                            cam.code, person_id, camera_role,
                            filter_stage="eye_visibility",
                            passed=eye_visibility_passed,
                            threshold_used=None,
                            measured_value=None,
                            buffer=buffer,
                            builder=builder,
                            force_create=force_create
                        )

                        # ---------------- DECISION: reject or force-create ---------------- #
                        # Counted once per frame, whichever gate(s) failed. After
                        # UNKNOWN_FORCE_CREATE_AFTER consecutive bad frames the track is
                        # registered anyway: the alternative is a track parked in
                        # COLLECTING_UNKNOWN forever with a frozen buffer.
                        if gate_failures:
                            fails = track_unknown_quality_fails.get(person_id, 0) + 1
                            track_unknown_quality_fails[person_id] = fails
                            force_create = fails >= UNKNOWN_FORCE_CREATE_AFTER

                            log_filter_stage(
                                cam.code, person_id, camera_role,
                                filter_stage="force_create",
                                passed=force_create,
                                threshold_used=UNKNOWN_FORCE_CREATE_AFTER,
                                measured_value=fails,
                                buffer=buffer,
                                builder=builder,
                                force_create=force_create,
                                failed_gates=",".join(gate_failures),
                            )
                            log(cam, person_id, "UNKNOWN",
                                f"gates failed {fails}/{UNKNOWN_FORCE_CREATE_AFTER}"
                                f" [{', '.join(gate_failures)}]"
                                f" -> {'FORCE CREATE' if force_create else 'keep collecting'}")

                            if not force_create:
                                continue
                        else:
                            track_unknown_quality_fails.pop(person_id, None)

                        buffer = track_unknown_buffer.get(person_id, [])

                        # if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        #     continue

                        # Rank buffered frames by how visible the eyes are, so the
                        # single frame we eventually store is the best look at this
                        # person rather than merely the highest-quality one.
                        buffer = builder.add(
                            buffer,
                            embedding,
                            quality,
                            pose,
                            img=face_img,
                            eye_quality=iris_ratio,
                        )
                        track_unknown_buffer[person_id] = buffer

                        if not builder.is_ready(buffer):
                            continue

                        centroid = builder.build(buffer)
                        if centroid is None:
                            continue

                        best = builder.get_best_face(buffer)
                        if not best or best["img"] is None or best["img"].size == 0:
                            continue

                        ok, buf = cv2.imencode(".jpg", best["img"])
                        if not ok:
                            continue

                        match = _faiss(unknown_embedding_store.find_match, centroid)

                        if match:
                            unknown_id = match["unknown_id"]
                            log(cam, person_id, "UNKNOWN", f"EXISTING UNKNOWN MATCHED → {unknown_id}")
                            log_unknown_decision(
                                "unknown_matched_existing",
                                cam.code,
                                person_id,
                                unknown_id=unknown_id,
                                similarity=match.get("similarity"),
                                role=cam.camera_role,
                                buffer_size=len(buffer),
                                **face_metrics(
                                    analysis, quality, final_quality, best_face_width
                                ),
                            )
                        else:
                            if cam.camera_role != "REGISTER":
                                log(cam, person_id, "UNKNOWN", f"NO MATCH → NOT CREATING (camera_role={cam.camera_role})")
                                log_unknown_decision(
                                    "unknown_not_created",
                                    cam.code,
                                    person_id,
                                    reason="camera_role_not_register",
                                    role=cam.camera_role,
                                    buffer_size=len(buffer),
                                    **face_metrics(
                                        analysis, quality, final_quality, best_face_width
                                    ),
                                )
                                continue
                            log(cam, person_id, "UNKNOWN", "NO MATCH → CREATING NEW UNKNOWN")
                            payload = build_unknown_payload(
                                buffer=buffer,
                                centroid=centroid,
                                cam_code=cam.code,
                                unknown_id=None,
                                builder=builder
                            )
                            unknown_id = _http(unknown_embedding_store.add_unknown, payload)

                            if not unknown_id:
                                log(cam, person_id, "UNKNOWN", "CREATE FAILED → STAY COLLECTING_UNKNOWN")
                                log_unknown_decision(
                                    "unknown_create_failed",
                                    cam.code,
                                    person_id,
                                    role=cam.camera_role,
                                    buffer_size=len(buffer),
                                    **face_metrics(
                                        analysis, quality, final_quality, best_face_width
                                    ),
                                )
                                continue

                            print(f"[UNKNOWN CREATED] {unknown_id} for person_id={person_id} at camera {cam.code}")
                            # eye_sharpness here is the current frame's value, not the
                            # best sample's — the builder buffer does not carry it.
                            log_unknown_decision(
                                "unknown_registered",
                                cam.code,
                                person_id,
                                unknown_id=unknown_id,
                                role=cam.camera_role,
                                buffer_size=len(buffer),
                                **face_metrics(
                                    analysis, quality, final_quality, best_face_width
                                ),
                            )

                        track_unknown_identity[person_id] = unknown_id
                        track_state[person_id] = TrackState.UPDATING_UNKNOWN
                        track_unknown_meta[person_id] = {"pose_best": {}, "last_update": 0}
                        track_unknown_quality_fails.pop(person_id, None)

                        # log(cam, person_id, "STATE", "→ UPDATING_UNKNOWN")
                        track_event_emitter.unknown_confirmed(cam.code, person_id, unknown_id)
                        continue

                    # =====================================================
                    # 🔵 STAGE 3: UPDATE (FINAL OPTIMIZED)
                    # =====================================================
                    elif state == TrackState.UPDATING_UNKNOWN:
                        if best_face_width < envConfig.MIN_UNKNOWN_REG_FACE_WIDTH:
                            continue

                        unknown_id = track_unknown_identity.get(person_id)
                        if not unknown_id:
                            continue

                        buffer = track_unknown_buffer.get(person_id, [])
                        buffer = builder.add(buffer, embedding, quality, pose, img=face_img)
                        track_unknown_buffer[person_id] = buffer

                        if not builder.is_ready(buffer):
                            continue

                        centroid = builder.build(buffer)

                        meta = track_unknown_meta.get(person_id, {
                            "pose_best": {},
                            "last_update": 0,
                            "last_attempted": {}
                        })

                        pose_best = meta["pose_best"]
                        last_attempted = meta.get("last_attempted", {})

                        # cooldown
                        if time.time() - meta["last_update"] < 2:
                            continue

                        # =====================================================
                        # 🔥 STEP 1: Build best candidate per pose
                        # =====================================================
                        pose_candidates = {}

                        for x in buffer:
                            p = x.get("pose_bucket") or "unknown"
                            q = x["quality"]

                            if p not in pose_candidates or q > pose_candidates[p]["quality"]:
                                pose_candidates[p] = x

                        # =====================================================
                        # 🔥 STEP 2: Filter poses (STRICT LOGIC)
                        # =====================================================
                        poses_to_send = {}

                        MIN_IMPROVEMENT = 0.08   # 🔥 increased
                        # MIN_SEND_QUALITY = 0.60

                        for p, data in pose_candidates.items():

                            best_quality = data["quality"]

                            local_q = pose_best.get(p, 0)
                            global_q = unknown_embedding_store.get_pose_quality(unknown_id, p)

                            effective_q = max(local_q, global_q)
                            last_q = last_attempted.get(p, 0)

                            # -----------------------------
                            # 🔴 HARD SKIP: worse or same
                            # -----------------------------
                            if best_quality <= effective_q:
                                continue

                            # -----------------------------
                            # 🔴 SKIP: micro improvement
                            # -----------------------------
                            if best_quality <= effective_q + MIN_IMPROVEMENT:
                                continue

                            # -----------------------------
                            # 🔴 SKIP: retry suppression
                            # -----------------------------
                            if best_quality <= last_q + 0.04:
                                continue

                            # -----------------------------
                            # 🔴 SKIP: low quality
                            # -----------------------------
                            # if best_quality < MIN_SEND_QUALITY:
                            #     continue

                            poses_to_send[p] = data

                            # 🔥 mark attempted (important)
                            last_attempted[p] = best_quality

                        # =====================================================
                        # 🔥 STEP 3: Nothing to send → skip
                        # =====================================================
                        if not poses_to_send:
                            continue

                        # =====================================================
                        # 🔥 STEP 4: Build payload
                        # =====================================================
                        pose_payload = {}

                        for p, data in poses_to_send.items():

                            img = data["img"]



                            if img is None or img.size == 0:
                                continue

                            h, w = img.shape[:2]
                            ok, buf = cv2.imencode(".jpg", img)

                            if not ok:
                                continue

                            pose_payload[p] = {
                                "embedding": data["embedding"].tolist(),
                                "quality": data["quality"],
                                 "faceSize": {
                                    "w": w,
                                    "h": h
                                },
                                "image": buf.tobytes(),   # 🔥 per-pose image
                                "ts": int(time.time() * 1000)
                            }

                        # nothing valid
                        if not pose_payload:
                            continue

                        # =====================================================
                        # 🔥 STEP 5: API CALL
                        # =====================================================
                        updated_id = _http(
                            unknown_embedding_store.update_unknown,
                            unknown_id,
                            centroid,
                            int(time.time() * 1000),
                            cam.code,
                            pose_payload
                        )

                        if not updated_id:
                            log(cam, person_id, "UPDATE", "UPDATE FAILED → KEEP COLLECTING")
                            continue

                        # =====================================================
                        # 🔥 STEP 6: UPDATE CACHE
                        # =====================================================
                        for p, data in poses_to_send.items():
                            pose_best[p] = data["quality"]

                            unknown_embedding_store.update_pose_quality_cache(
                                unknown_id,
                                p,
                                data["quality"]
                            )

                        track_unknown_meta[person_id] = {
                            "pose_best": pose_best,
                            "last_update": time.time(),
                            "last_attempted": last_attempted
                        }

                        log(cam, person_id, "UPDATE",
                            f"UPDATED → {unknown_id}, poses={list(poses_to_send.keys())}")
            except Exception:
                # One bad frame must never take the camera thread down.
                frame_errors += 1
                if frame_errors % 30 == 1:
                    print(f"[Camera {cam.code}] frame error #{frame_errors}: {traceback.format_exc()}")
                continue
