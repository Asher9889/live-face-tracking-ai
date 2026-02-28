import threading
import time
import cv2
from typing import List
import random
from enum import Enum
import numpy as np
from app.config import FRAME_RATE

from ultralytics import YOLO

from app.ai.insight_detector import InsightFaceEngine
from app.ai.person_detector import PersonDetector
from app.ai.tracker_service import ByteTrackerService
from app.camera import CameraConfig, FrameMessage
from app.camera.extract_person_roi import extract_person_roi
from app.camera.frame_queue import frame_queue
from app.camera.helper import motion_score, is_blurry


model = YOLO("yolov8n.pt")
insight_engine = InsightFaceEngine()
class CameraState(str, Enum):
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DOWN = "DOWN"


def _open_capture(rtsp_url: str):
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

def _camera_loop(cam: CameraConfig) -> None:
    print(f"[Camera] Worker started → {cam.code}")
    target_fps = int(FRAME_RATE)

    person_detector = PersonDetector()
    tracker = ByteTrackerService(frame_rate=target_fps, lost_track_buffer=30)
    track_state = {}
    track_identity = {}

    backoff = 1.0
    max_backoff = 30.0

    interval = 1.0 / target_fps

    while True:
        print(f"[Camera] 🔌 Connecting → {cam.code}")

        cap = _open_capture(cam.rtsp_url)


        if not cap.isOpened():
            print(f"[Camera] ❌ Connect failed → {cam.code}")
            sleep_time = min(backoff + random.uniform(0, 1), max_backoff)
            print(f"[Camera] ⏳ Retry in {sleep_time:.1f}s → {cam.code}")
            time.sleep(sleep_time)
            backoff = min(backoff * 2, max_backoff)
            continue

        print(f"[Camera] ✅ Connected → {cam.code}")
        backoff = 1.0  # reset backoff on success

        last_processed = 0.0
        last_log = time.time()

        while True:
            grabbed = cap.grab()

            if not grabbed:
                print(f"[Camera] ⚠️ Stream lost → {cam.code}")
                cap.release()
                # tracker.reset()
                break  # go to reconnect loop

            now = time.time()

            if now - last_processed < interval:
                continue

            ret, frame = cap.retrieve()

            if not ret or frame is None:
                continue

            last_processed = now
            if now - last_log > 5:
                # print(f"[Camera] ❤️ Live → {cam.code}")
                last_log = now


            # DETECTION & Tracking

            results = model.track(
                frame, 
                persist=True, 
                tracker="botsort.yaml", 
                classes=[0],            # Only track 'person' (COCO class 0)
                verbose=False,
                conf=0.25               # Adjust based on your environment
            )
 
            # Get results
            if results[0].boxes.id is not None: # gives all bbox and ids
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.int().cpu().numpy()
                
                for person_id, bbox in zip(ids, boxes):
                    print(f"Track ID: {person_id} at {bbox}")
                    roi_data = extract_person_roi(frame, person_id, bbox)

                    if roi_data is None:
                        continue
                    
                    person_id, roi, offset = roi_data

                    faces = insight_engine.detect_and_recognize(roi, offset)
                    print(f"[Camera {cam.code} Person {person_id}] 🧠 InsightFace detected {len(faces)} faces")
                    # x1, y1, x2, y2 = map(int, bbox)
                    # Now you can pass this bbox to SCRFD for face verification

            # boxes, scores = person_detector.detect(frame)

            # tracks = tracker.update(boxes, scores)
            # if cam.code == "entry_1":
            #     print(f"[Camera {cam.code}] 🔍 Detected {len(tracks)} tracks", tracks)

            # detections = person_detector.detect(frame)
            # if detections is None:
            #     detections = []

            # if cam.code == "entry_1":
            #     print(f"[Camera {cam.code}] 🔍 Before DETECTION STAGE {len(detections)} faces", detections)   


            # # Convert to tracker format
            # if len(detections) > 0:

            #     boxes = np.asarray(
            #         [d["bbox"] for d in detections],
            #         dtype=np.float32   # IMPORTANT: float32
            #     )

            #     scores = np.asarray(
            #         [d["score"] for d in detections],
            #         dtype=np.float32
            #     )

            #     detection_meta = detections  # keep original metadata

            # else:
            #     boxes = np.zeros((0, 4), dtype=np.float32)
            #     scores = np.zeros((0,), dtype=np.float32)
            #     detection_meta = []

            # if cam.code == "entry_1":
            #     print(f"[Camera {cam.code}] 🔍 After DETECTION STAGE {len(detections)} faces", detections)   

            # #------TRACKING STAGE------

            # tracks, lost_ids = tracker.update(boxes=boxes, scores=scores)

            # if cam.code == "entry_1":
            #     print(f"[Camera {cam.code}] 🔍 Detected {len(tracks)} tracks", tracks)

            # # cleanup per-track state
            # for tid in lost_ids:
            #     track_state.pop(tid, None)
            #     track_identity.pop(tid, None)
            #     print(f"[Camera {cam.code}] 🗑️ Track {tid} lost")



            











# [Camera entry_1] 🔍 Detected 1 tracks [(1, array([316.2396 , 257.37665, 388.47235, 416.04968], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(2, array([324.14908, 259.61874, 398.86838, 418.42868], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(4, array([333.28256, 264.10355, 397.69003, 500.73862], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(6, array([337.56793, 264.69656, 401.38123, 568.06616], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([348.34296, 270.55643, 417.83405, 571.03436], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([349.8256 , 260.26724, 421.19815, 568.33295], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([349.4566 , 254.87216, 424.26773, 571.20886], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([352.33255, 255.24283, 426.49728, 568.53595], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([350.31976, 255.82385, 426.66214, 568.9589 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([352.81982, 256.80777, 424.58746, 570.4704 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([354.31345, 257.68774, 424.40265, 573.0898 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(7, array([348.81543, 257.79968, 428.4075 , 573.5841 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(8, array([361.95   , 262.86523, 438.47906, 573.5071 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(8, array([363.7255 , 261.012  , 449.19452, 574.753  ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(9, array([367.98117, 262.757  , 460.5776 , 573.41504], dtype=float32))]
# [Camera entry_1] 🔍 Detected 1 tracks [(9, array([366.9773 , 262.7646 , 475.77432, 572.0113 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(10, array([403.2188 , 265.22824, 466.3531 , 572.8494 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(12, array([409.14676, 263.70062, 488.9937 , 572.07996], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(13, array([448.23953, 264.2756 , 499.86487, 568.0522 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(15, array([446.52365, 275.9354 , 522.3967 , 570.9767 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(16, array([508.61835, 274.69077, 552.93713, 346.9434 ], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(19, array([583.61163, 278.311  , 636.45996, 465.07336], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(20, array([597.0525 , 277.8986 , 645.52094, 477.90134], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(21, array([604.1087 , 278.67618, 666.44183, 495.87274], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []
# [Camera entry_1] 🔍 Detected 1 tracks [(22, array([619.87244, 277.3559 , 686.14233, 506.93887], dtype=float32))]
# [Camera entry_1] 🔍 Detected 0 tracks []