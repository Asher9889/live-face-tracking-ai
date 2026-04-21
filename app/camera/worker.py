import os
import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.helper import is_stable_embedding_global, fast_filter, is_stable_embedding, expand_bbox, select_best_face, crop_with_margin, get_pose_name, now_ms
from app.camera.types import CameraConfig, TrackState
from app.config import FRAME_RATE

from ultralytics import YOLO
from queue import Queue, Empty

from app.ai.insight_detector import InsightFaceEngine
from app.ai.face_mesh_engine import FaceLandmarkerEngine
from app.camera.extract_person_roi import extract_person_roi
from app.config.config import envConfig
from app.events.publisher import EventPublisher
from app.recognition import embedding_store, unknown_embedding_store
from app.tracking.track_manager import TrackEventEmitter
from app.database import redis_client

from app.camera.unique_face_builder import UniqueFaceRepresentationBuilder
from app.camera.payload_builder import build_unknown_payload

# unknown_manager = UnknownIdentityManager(unknown_embedding_store)

MIN_UNKNOWN_CREATION_QUALITY = float(envConfig.MIN_UNKNOWN_CREATION_QUALITY)
MIN_UNKNOWN_CREATE_FRAMES = int(envConfig.MIN_UNKNOWN_CREATE_FRAMES)

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

model = YOLO("yolov8n.pt")   
insight_engine = InsightFaceEngine()
publisher = EventPublisher(redis_client)
face_landmarker_engine = FaceLandmarkerEngine(model_path=path)  

class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"

def log(cam, person_id, stage, msg):
    print(f"[{now_ms()}][Camera {cam.code}][Person {person_id}][{stage}] {msg}")


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
    """

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

    # target_fps = int(FRAME_RATE)
    # interval = 1.0 / target_fps
    backoff = CAPTURE_BACKOFF_INITIAL

    track_event_emitter = TrackEventEmitter(publisher=publisher, gate_type=cam.gate_type)
    builder = UniqueFaceRepresentationBuilder()

    track_state = {}
    track_identity = {}
    track_known_buffer = {}
    track_unknown_buffer = {}
    track_unknown_identity = {}
    track_unknown_meta = {}
    track_embedding_state = {}

    # For RTSP Thread
    frame_queue = Queue(maxsize=1)
    stop_event = threading.Event()
    frame_count = 0



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

            frame_queue.put(frame)


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
                frame = frame_queue.get(timeout=5)
                if frame is None or frame.size == 0:
                    continue
            except Empty:
                print(f"[Camera] ⚠️ No frames → {cam.code}; reconnecting")
                # cap.release()
                # stop_event.set()
                # break
                stop_event.set()

                reader_thread.join(timeout=2)   # wait for thread to exit safely

                cap.release()

                track_state.clear()
                track_identity.clear()
                track_known_buffer.clear()
                track_unknown_buffer.clear()
                track_unknown_identity.clear()
                track_unknown_meta.clear()
                track_embedding_state.clear()

                break

            frame_h, frame_w = frame.shape[:2]

            results = model.track(frame, persist=True, classes=[0], conf=0.25, verbose=False)

            if results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()

            lost = track_event_emitter.cleanup_lost_tracks(cam.code, ids.tolist())

            for tid in lost:
                track_state.pop(tid, None)
                track_identity.pop(tid, None)
                track_known_buffer.pop(tid, None)
                track_unknown_buffer.pop(tid, None)
                track_unknown_identity.pop(tid, None)
                track_unknown_meta.pop(tid, None)
                track_embedding_state.pop(tid, None)

            for person_id, bbox in zip(ids, boxes):

                person_id = int(person_id)

                # Keep track lifecycle state updated so emit-once events are not dropped.
                track_event_emitter.update_track(
                    cam.code,
                    person_id,
                    bbox,
                    int(time.time() * 1000),
                    frame_w,
                    frame_h
                )

                if person_id in track_identity:
                    continue

                if person_id not in track_state:
                    track_state[person_id] = TrackState.COLLECTING_KNOWN
                    log(cam, person_id, "STATE", "INIT → COLLECTING_KNOWN")
                # else:
                #     # 🔥 IMPORTANT DEBUG
                #     log(cam, person_id, "DEBUG", f"EXISTING STATE → {track_state[person_id]}")

                state = track_state[person_id]

                # -------------------------
                # ROI + FACE DETECTION
                # -------------------------
                x1, y1, x2, y2 = expand_bbox(bbox, frame_w, frame_h)
                roi_data = extract_person_roi(frame, person_id, np.array([x1, y1, x2, y2]))
                if roi_data is None:
                    continue

                _, roi, offset = roi_data

                faces = insight_engine.detect_and_generate_embedding(roi, offset, cam.code)

                if not faces:
                    continue

                if len(faces) > 1:
                    print(f"[Camera {cam.code}] Skipping ROI with multiple faces: {len(faces)}")
                    continue

                # Filter bad faces after detetction
                faces = [f for f in faces if fast_filter(f)]
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

                    # 🔥 GLOBAL stability check (once)
                    if not is_stable_embedding_global(track_embedding_state, person_id, embedding):
                        print(f"[{now_ms()}][Camera {cam.code}] Unstable embedding → person_id={person_id}")
                        continue


                    face_img = crop_with_margin(frame, x1, y1, x2, y2, margin=0.2)

                    if face_img.size == 0:
                        continue

                    analysis = face_landmarker_engine.analyze(face_img)
                    # is_valid = face_landmarker_engine.is_valid_face(analysis, cam.code) 
                    score = face_landmarker_engine.score_face(analysis)
                    if score == 0:
                        continue
                    # if not is_valid:
                    # #     # print(f"[Camera {cam.code}] Face rejected by FaceLandmarker is_valid_face check")
                    # #     continue
                    quality = insight_engine.compute_face_quality(f, face_img, analysis)
                    final_quality = quality * score
                    if final_quality < 0.15:
                        continue

                    f["quality"] = final_quality
                    valid_faces.append(f)

                if not valid_faces:
                    continue

                best = select_best_face(valid_faces)
                if best is None:
                    continue

                embedding = best["embedding"]
                quality = best["quality"]

                x1, y1, x2, y2 = map(int, best["bbox"])
                face_img = crop_with_margin(frame, x1, y1, x2, y2, margin=0.2)

                if face_img.size == 0:
                    continue

                pose = get_pose_name(best.get("pose", [None])[0]) or "unknown"

                # =====================================================
                # 🔵 STAGE 1: KNOWN
                # =====================================================
                if state == TrackState.COLLECTING_KNOWN:

                    # stability ONLY here
                    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        log(cam, person_id, "STABILITY", "REJECTED")
                        continue

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

                    log(cam, person_id, "KNOWN", f"buffer_size={len(buffer)}")

                    if len(buffer) < 3:
                        continue

                    # combine
                    emb = np.array([x["embedding"] for x in buffer])
                    w = np.array([x["quality"] for x in buffer])
                    final = np.average(emb, axis=0, weights=w)
                    final /= np.linalg.norm(final)

                    log(cam, person_id, "KNOWN", "RUN MATCH")
                    match = embedding_store.find_match(final)

                    log(cam, person_id, "KNOWN", f"MATCH RESULT → {match['employee_id'] if match else 'NO MATCH'}")

                    if match:
                        track_identity[person_id] = match["employee_id"]
                        track_state[person_id] = TrackState.MATCHED_KNOWN

                        track_event_emitter.recognition_confirmed(
                            cam.code,
                            person_id,
                            match["employee_id"],
                            match["similarity"]
                        )

                        log(cam, person_id, "KNOWN", f"MATCHED → {match['employee_id']}")
                        track_known_buffer.pop(person_id, None)
                        continue

                    # move to unknown
                    track_state[person_id] = TrackState.COLLECTING_UNKNOWN
                    track_unknown_buffer[person_id] = [
                        x for x in buffer if x["quality"] >= MIN_UNKNOWN_CREATION_QUALITY
                    ]
                    track_known_buffer.pop(person_id, None)

                    log(cam, person_id, "STATE", "→ COLLECTING_UNKNOWN")
                    continue

                # =====================================================
                # 🔵 STAGE 2: UNKNOWN
                # =====================================================
                elif state == TrackState.COLLECTING_UNKNOWN:
                    buffer = track_unknown_buffer.get(person_id, [])

                    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
                        continue

                    buffer = builder.add(buffer, embedding, quality, pose, img=face_img)
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

                    match = unknown_embedding_store.find_match(centroid)

                    if match:
                        unknown_id = match["unknown_id"]
                        log(cam, person_id, "UNKNOWN", f"EXISTING UNKNOWN MATCHED → {unknown_id}")
                    else:
                        if cam.camera_role != "REGISTER":
                            log(cam, person_id, "UNKNOWN", f"NO MATCH → NOT CREATING (camera_role={cam.camera_role})")
                            continue
                        log(cam, person_id, "UNKNOWN", "NO MATCH → CREATING NEW UNKNOWN")
                        payload = build_unknown_payload(
                            buffer=buffer,
                            centroid=centroid,
                            cam_code=cam.code,
                            unknown_id=None,
                            builder=builder
                        )
                        unknown_id = unknown_embedding_store.add_unknown(payload)

                        if not unknown_id:
                            log(cam, person_id, "UNKNOWN", "CREATE FAILED → STAY COLLECTING_UNKNOWN")
                            continue
                        
                        print(f"[UNKNOWN CREATED] {unknown_id} for person_id={person_id} at camera {cam.code}")

                    track_unknown_identity[person_id] = unknown_id
                    track_state[person_id] = TrackState.UPDATING_UNKNOWN
                    track_unknown_meta[person_id] = {"pose_best": {}, "last_update": 0}

                    # log(cam, person_id, "STATE", "→ UPDATING_UNKNOWN")
                    track_event_emitter.unknown_confirmed(cam.code, person_id, unknown_id)
                    continue

                # =====================================================
                # 🔵 STAGE 3: UPDATE (FINAL OPTIMIZED)
                # =====================================================
                elif state == TrackState.UPDATING_UNKNOWN:

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
                        p = x["pose_bucket"]
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
                    updated_id = unknown_embedding_store.update_unknown(
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