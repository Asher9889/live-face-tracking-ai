import time
from app.ai.align_faces import align_faces
from app.ai.debug_face_saver import save_faces
from app.ai.debug_tracking import debug_tracking
from app.ai.face_filter import filter_and_crop
from app.ai.remove_duplicate_detections import remove_duplicate_detections
from app.ai.tracker_service import ByteTrackerService
from app.camera.frame_queue import frame_queue
from typing import List
from app.camera import FrameMessage
from app.ai.insight_detector import InsightFaceEngine
from app.camera.types import Detection

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.02  # 20ms

face_detector = InsightFaceEngine()
tracker_service = ByteTrackerService()

def start_batch_processor() -> None:
    """
    Continuously collects frames and processes batches.
    Simulates GPU inference pipeline.
    """

    print("[AI] Batch processor started")

    while True:
        batch: List[FrameMessage] = []

        start_time = time.time()

        # Collect frames for up to 20ms OR until we have 8 frames — whichever comes first.
        while len(batch) < BATCH_SIZE:
            remaining = BATCH_TIMEOUT - (time.time() - start_time)

            if remaining <= 0:
                break

            msg = frame_queue.pop(timeout=remaining)

            if msg is not None:
                batch.append(msg)

        # If no frames → continue loop
        if not batch:
            continue

        _process_batch(batch)

def _process_batch(batch: List[FrameMessage]) -> None:
    """
    Real pipeline stages: 
    Extract frames
    """ 

    frames = []
    camera_codes = []
    timestamps = []

    for msg in batch:
        frames.append(msg.frame)
        camera_codes.append(msg.camera_code)
        timestamps.append(msg.timestamp)

    # Face Detection
    detections = _run_detection(frames, camera_codes, timestamps)

    print(f"[AI] detections={len(detections)}")

    # Remove duplicates detections
    detections = remove_duplicate_detections(detections)

    # print("[AI] detections", len(detections))

    # Tracking
    # 🔥 2️⃣ Correct tracking (sequential)
    tracked_detections = tracker_service.update_sequential(detections)

    debug_tracking(tracked_detections)
    # Debug
    # print("[AI] track_ids", [d.track_id for d in tracked_detections])

   
    # face_crops = filter_and_crop(tracked_detections)

    # aligned_faces = align_faces(tracked_detections) 


    # save_faces(tracked_detections)
    # print("[AI] aligned_faces", tracked_detections)

    # print(f"[AI] usable faces={len(aligned_faces)}")



def _run_detection(frames, camera_codes, timestamps):
    # detections = []

    # for frame, cam_code, ts in zip(frames, camera_codes, timestamps):
    #     faces = face_detector.detect(frame) or []

    #     # print(f"[AI] Detected {cam_code} {len(faces)} faces")
    #     for face in faces:
    #         detections.append(
    #             Detection(
    #                 camera_code=cam_code,
    #                 timestamp=ts,
    #                 frame=frame,
    #                 bbox=face["bbox"],
    #                 landmarks=face["landmarks"],
    #                 score=face["score"],
    #                 pose=face["pose"],
    #                 age=face["age"],
    #                 gender=face["gender"]
    #             )
    #         )
    return detections


