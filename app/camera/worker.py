import os
import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.camera.helper import is_stable_embedding, expand_bbox, select_best_face, crop_with_margin, get_pose_name, now_ms
from app.camera.types import CameraConfig, TrackState
from app.config import FRAME_RATE

from ultralytics import YOLO

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

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
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

    target_fps = int(FRAME_RATE)
    interval = 1.0 / target_fps

    track_event_emitter = TrackEventEmitter(publisher=publisher, gate_type=cam.gate_type)
    builder = UniqueFaceRepresentationBuilder()

    track_state = {}
    track_identity = {}
    track_known_buffer = {}
    track_unknown_buffer = {}
    track_unknown_identity = {}
    track_unknown_meta = {}
    track_embedding_state = {}

    while True:
        cap = _open_capture(cam.rtsp_url)

        if not cap.isOpened():
            time.sleep(2)
            continue

        last_processed = 0.0

        while True:
            if not cap.grab():
                cap.release()
                break

            now = time.time()
            if now - last_processed < interval:
                continue

            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue

            last_processed = now
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

                # -------------------------
                # QUALITY FILTER
                # -------------------------
                valid_faces = []
                for f in faces:
                    if f["score"] < envConfig.SCRFD_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, f["bbox"])
                    # face_img = frame[y1:y2, x1:x2]
                    face_img = crop_with_margin(frame, x1, y1, x2, y2, margin=0.2)

                    if face_img.size == 0:
                        continue

                    analysis = face_landmarker_engine.analyze(face_img)
                    quality = insight_engine.compute_face_quality(f, face_img, analysis)

                    if quality < 0.15:
                        continue

                    f["quality"] = quality
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