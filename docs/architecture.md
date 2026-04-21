# Optimization Flow
- detect → 🟢 fast_filter → 🟡 stability → crop → 🔵 MediaPipe → scoring

1) Detection
2) Tracking
3) Track state update
4) Quality filtering
5) Embedding (controlled frequency)
6) FAISS match
7) Identity assignment
8) Event publishing

# 1️⃣ What buffalo_l actually is
- buffalo_l is not a single model.
- It is a bundle of multiple models:

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

# Track Sequence
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

# Flow Diagram
Detection
   ↓
Tracking (BoTSORT)
   ↓
Embedding
   ↓
Search Known Index
   ↓
Match?
 ├─ YES → known stabilization → identity confirmed
 └─ NO
        ↓
   Search Unknown Index
        ↓
   Unknown stabilization (3 frames)
        ↓
   Create or update unknown identity

## Face Filter Metrics
### 👁️ Visual intuition

#### ✅ Yaw (left ↔ right turn)

| Yaw | Meaning | Should you keep it? |
|---:|---|---|
| 0° | Perfect frontal | ✅ YES (ideal) |
| 10–20° | Slight turn | ✅ YES |
| 25–30° | Noticeable turn | ⚠️ Borderline |
| 35°+ | Side face | ❌ NO |

**Image (add later):**

![Pose](../docs/images/yaw/0_qUZJdXJLUaZ4e3xd.jpg)
![Yaw](../docs/images/yaw/fpsyg-03-00563-g001.jpg)

#### ✅ Pitch (up ↕ down tilt)

| Pitch | Meaning | Keep? |
|---:|---|---|
| 0° | Straight | ✅ |
| 10–15° | Slight tilt | ✅ |
| 20° | Strong tilt | ⚠️ |
| 25°+ | Looking up/down heavily | ❌ |

**Image (add later):**

![Pitch](pitch.png)

👉 Cameras mounted high/low make this very common

#### ✅ Roll (head tilt sideways)

| Roll | Meaning | Keep? |
|---:|---|---|
| 0° | Straight | ✅ |
| 10–15° | Slight tilt | ✅ |
| 20° | Noticeable | ⚠️ |
| 25°+ | Tilted head | ❌ |

**Image (add later):**

![Roll](roll.png)


# 📄 Unknown Tracking Pipeline
---

## 1. 🎯 System Overview
- You have **two parallel identity systems**:
1. Known Identity (employees)
2. Unknown Identity (dynamic clustering)
---

## 2. 🧩 Core Components

### 🔹 Detection
- YOLO (`yolov8n.pt`) → detects persons
- BoT-SORT → assigns `track_id`

### 🔹 Face Engine
- InsightFace (SCRFD + ArcFace)
- FaceMesh (for pose + quality signals)

### 🔹 Tracking State

```python
track_known_buffer        # temp embeddings for known recognition 
track_state               # lifecycle tracking
track_identity            # known identities
track_unknown_identity    # unknown identities
track_unknown_buffer      # unknown temp storage
track_embedding_state     # stability tracking
```

# Load Unknown Embeddings DS:
```javascript
StoreData(
    centroid_matrix = np.array([
        A_c,
        B_c
    ], dtype=np.float32),

    pose_matrix = np.array([
        A_f,
        A_l,
        B_f,
        B_r
    ], dtype=np.float32),

    pose_owner = [
        "A",
        "A",
        "B",
        "B"
    ],

    unknown_ids = [
        "A",
        "B"
    ]
)

```