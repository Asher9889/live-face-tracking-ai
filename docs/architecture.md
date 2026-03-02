1) Detection
2) Tracking
3) Track state update
4) Quality filtering
5) Embedding (controlled frequency)
6) FAISS match
7) Identity assignment
8) Event publishing


Camera Frame
(1920x1080)
       │
       ▼
YOLO Detect
       │
       ▼
Person 1 bbox → crop → person_roi_1
Person 2 bbox → crop → person_roi_2
       │
       ▼
SCRFD(person_roi_1) → face bbox
SCRFD(person_roi_2) → face bbox
       │
       ▼
Convert to global coordinates
       │
       ▼
Recognition / Tracking / Draw


1️⃣ What buffalo_l actually is

buffalo_l is not a single model.
It is a bundle of multiple models:

Model	Purpose
det_10g.onnx	SCRFD face detector
2d106det.onnx	facial landmarks
1k3d68.onnx	3D landmarks
w600k_r50.onnx	face embeddings (recognition)
genderage.onnx	gender + age

SCRFD Raw output: [Camera entry_2] 🔍 Detected 1 faces [{'bbox': array([396, 402, 487, 551], dtype=int32), 'score': 0.7635079622268677, 'landmarks': array([[411, 463],
       [424, 462],
       [398, 489],
       [413, 517],
       [423, 515]], dtype=int32), 'pose': (np.float32(-0.19837126), np.float32(-48.970337), np.float32(-5.920164)), 'size': np.int32(91), 'area': np.int32(13559), 'age': 29, 'gender': 1}]


# AI Events emitted to node
- track_created
- track_updated
- face_detected
- recognition_pending
- recognition_confirmed
- track_lost

# Mechanism Guide
| What | Mechanism |
|------|-----------|
| Track lifecycle changes | Pub/Sub events |
| Continuous bbox updates | WebSocket / Stream |
| Persistent track state | Redis Hash |
| Analytics logging | MongoDB |

YOLO track start
      ↓
track_created
      ↓
track_updated
      ↓
face_detected
      ↓
recognition_pending
      ↓
recognition_confirmed
      ↓
track_lost



Person Tracking (YOLO)

        ↓

Face Detection (SCRFD)

        ↓

Face Quality Filter

        ↓

Embedding Extraction

        ↓

Embedding Matching

        ↓

Recognition Confirmed