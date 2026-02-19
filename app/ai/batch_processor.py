import time
from app.camera.frame_queue import frame_queue
from typing import List
from app.camera import FrameMessage
from app.ai.face_detector import FaceDetector

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.02  # 20ms

face_detector = FaceDetector("models/scrfd/scrfd_10g_bnkps.onnx")

def start_batch_processor() -> None:
    """
    Continuously collects frames and processes batches.
    Simulates GPU inference pipeline.
    """

    print("[AI] Batch processor started")

    while True:
        batch: List[FrameMessage] = []

        start_time = time.time()

        # Collect frames until batch full OR timeout
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
    1️⃣ Extract frames
    2️⃣ Run detection
    3️⃣ Process detections
    """ 

    frames = [msg.frame for msg in batch]
    camera_codes = [msg.camera_code for msg in batch]
    timestamps = [msg.timestamp for msg in batch]

    # 🔜 Next step will plug detector here
    detections = _run_detection(frames)

    # Temporary logging
    face_count = sum(len(d) for d in detections)
    print(f"[AI] Batch size={len(frames)} | faces={face_count}")


def _run_detection(frames):
    results = []

    for frame in frames:
        faces = face_detector.infer(frame)
        results.append(faces)

    return results

