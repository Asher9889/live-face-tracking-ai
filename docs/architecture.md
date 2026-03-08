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