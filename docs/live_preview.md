# Live Preview + BBox Streaming

## 📋 Summary

Changes how video reaches the frontend. Today Node reads RTSP separately and draws
AI bboxes on top, which cannot be time-aligned. This moves the preview **into the AI
server** so the frame and its bboxes are published together from the same decode.

**Unchanged:** recognition pipeline, presence events, Redis keys, event payloads.
The `ALLOWED_EVENTS` flow is untouched — this is a parallel path, not a replacement.

---

## ❌ Current Problem

Two independent RTSP readers, two unrelated clocks.

```
Node    ──RTSP──→ 25 fps video ──→ browser
AI      ──RTSP──→ 3-5 fps bboxes ──→ Redis ──→ Node ──→ browser
```

RTP gives every client its own jitter buffer. There is no shared frame index, so
"frame 480" on Node and "frame 480" on AI are unrelated. The offset is **not
constant**:

- AI drops frames (`frame_queue` maxsize=1, drop-oldest)
- AI rate varies (YOLO + SCRFD + ArcFace + facemesh + HTTP calls per frame)
- gap between AI publishes is not a whole number of video frame intervals
- optimal offset drifts with jitter buffer resizing, load, GC, camera FPS

**Adding a fixed delay does not fix this.** It shifts the mean and leaves the error.
A hand-tuned constant is correct for ~10 minutes, then silently wrong.

Worse: if YOLO stays in the slow thread while the reader publishes independently,
you publish a *fresh* image with a *stale* box — the same bug, smaller.

---

## ✅ New Approach

The AI server becomes the single source of truth for preview video. Video and bboxes
are published together from the same decode, so **there is nothing to align**.

```
AI server
  └─ one RTSP decode
       ├─→ YOLO track (every frame)  ──→ publish (frame + bboxes) ──→ LiveKit ──→ browser
       └─→ face pipeline (subsampled) ──→ attach label to track_id ──┘
```

The label belongs to the **track**, not the frame. Once a `track_id` resolves to a
person, every later frame for that id carries the name for free — including frames
that never touched the face pipeline.

---

## 🏗️ Pipeline Split

Two consumers off one decode. Neither may guess what the other saw.

| Stage | Rate | Cost | Output |
|-------|------|------|--------|
| Decode + YOLO track | every frame (20-25 fps) | low | `track_id`, bbox, velocity |
| Face pipeline | subsampled per track | high | label for `track_id` |

### 1️⃣ Fast path (every frame)
Decode → YOLO track → publish. No face work at all. This is what feeds the preview.

> ⚠️ **YOLO must run on every published frame, in the same path that publishes.**
> If it sits in the slow thread, you ship fresh pixels with stale boxes.

### 2️⃣ Slow path (subsampled)
Face recognition runs **only for tracks without a confirmed label**, a few times per
second until it resolves, then stops for that track until the track dies.

> Re-recognising an already-labelled person 25×/sec is pure waste. Skipping it is
> the single largest throughput win available.

---

## 🏠 LiveKit Topology

Anyone on the internet may view the feed, so WebRTC is required — a raw WebSocket
cannot traverse unknown NATs, and ICE + TURN is the only thing that does.

### Room naming: one room per camera `code`

The camera API returns a stable `code` (e.g. `entry_2`). It never changes, so it
**is** the room name.

| Choice | Consequence |
|--------|-------------|
| ✅ One room per camera code | stable, deterministic, already a unique business key |
| ⚠️ One room per `gate_type` | `gate_type` is only `ENTRY`/`EXIT` — a direction, not a site |

- **Room name** = `cam.code`, e.g. `entry_2`
- One video track published per room
- **Camera code** travels in every data-channel message, and is the key binding a bbox to a track
- Camera flap ends that room's track; other cameras are unaffected
- Cost: N cameras = N rooms = N connections for a browser watching N feeds

> An unguessable room name is **not** security. The token is. The name is a lookup key.

### Security

| Requirement | Value |
|-------------|-------|
| AI publisher token | `can_publish`, `can_subscribe`, `can_publish_data` = true |
| Browser token | `can_publish` = **false**, `can_publish_data` = false |
| Token lifetime | 10 min (AI), short-lived per request (browser) |
| Region | AI server co-located with LiveKit media node |
| TURN | provisioned **and reachable** from client networks |

> A TURN relay that exists in the console but is firewalled is the classic
> "works on office wifi, fails for customers" failure. Test from a real external network.

> 🔒 **Privacy:** a token is a capability to see identifiable faces. Decide
> deliberately whether the preview stream carries the **face crop** or only
> bbox + name. The latter is a far lighter exposure, and it is much cheaper to
> choose now than to change later.

---

## ⏱️ Sync After This Change

WebRTC **bounds** the offset, it does not remove it. The SFU repackages frames and
the browser holds a jitter buffer, so displayed video lags the published frame by
roughly **200-400 ms**.

The difference: the offset is now small and stable enough for a **fixed overlay
delay to hold**.

| Approach | Preview rate | Box accuracy | Sync work |
|----------|-------------|--------------|-----------|
| Two RTSP readers (current) | full | ❌ unbounded error | ❌ impossible |
| Publish AI-processed frames only | 3-5 fps | ✅ exact | none |
| **Publish every decoded frame** | **full** | ✅ bounded, stable | **fixed delay** |

> Publish only AI-processed frames is the fallback if throughput can't be fixed:
> choppy video, but always-correct boxes. Reasonable for a presence feed, where
> correct identity matters more than smooth motion.

---

## 📬 Message Format

One message per published frame, image and metadata bundled. Separate messages can
arrive apart; that class of bug is exactly what this design removes.

```json
{
  "type": "frame_state",
  "camera_code": "entry_2",
  "seq": 10432,
  "frameTs": 1714900000123,
  "frame_width": 640,
  "frame_height": 360,
  "tracks": [
    {
      "track_id": 7,
      "bbox": [210, 120, 280, 260],
      "state": "MATCHED_KNOWN",
      "label": "EMP_12345",
      "label_confidence": 1.0,
      "label_expires_at": null
    },
    {
      "track_id": 9,
      "bbox": [300, 130, 372, 268],
      "state": "COLLECTING_UNKNOWN",
      "label": null,
      "label_confidence": 0.0,
      "label_expires_at": null
    }
  ]
}
```

Delivered on the LiveKit data channel, topic `frame_state`, unreliable.

| Field | Purpose |
|-------|---------|
| `seq` | monotonic per camera, detects gaps |
| `frameTs` | **capture** time, stamped at decode — not emit time |
| `frame_width/height` | published (downscaled) size, so UI scales without guessing |
| `bbox` | rescaled from source coords into published-frame coords |
| `state` | from existing `TrackState` |
| `label` | `null` while pending; stable synthetic id once unknown is created |
| `label_confidence` | 1.0 known, 0.9 unknown, 0.0 pending |
| `label_expires_at` | always `null` today — see gap below |

> ⚠️ `frameTs` **must** be stamped the moment the frame leaves the decoder, before
> any inference. `emit_time - capture_time` is the only honest measure of pipeline
> latency. Stamping at emit makes the delay mathematically unknowable.

> 🚧 **Known gap:** automatic label re-verification is **not implemented**.
> `label_expires_at` is emitted as `null` and labels persist for the life of the
> track, so a tracker ID switch can currently put a name on the wrong person.
> See the Label Lifecycle section — this is the next thing to build.

### Bundling
Video and metadata travel as **two synchronised channels**, not one message —
this is what WebRTC gives us, and bundling JPEG into a data message would be a
downgrade in quality and bandwidth:

| Channel | Transport | Carries |
|---------|-----------|---------|
| Video | LiveKit video track | H.264/VP8, adaptive bitrate, `timestamp_us` = capture time |
| Metadata | `frame_state` data topic | bboxes, labels, `frameTs` |

They are correlated by `frameTs`, not by send order. **Video never waits for
recognition** — that is the whole point of the split.

---

## 🔁 Full Scene State, Not Deltas

Every publish carries the **complete** current scene for that camera.

- **Idempotent** — a dropped message costs nothing; the next publish repairs it
- **Self-healing** — a reconnecting client is correct on the first message it receives
- No client-side event replay needed

> Deltas lose the UI permanently on a single dropped message. Over Redis pub/sub on
> a busy network, that will happen.

### Keep the preview stream separate from presence

| Path | Purpose | Persistence |
|------|---------|-------------|
| Preview stream (`frame_state`) | ephemeral view | ❌ never write to Redis or DB |
| `person_entered` / `unknown_entered` etc. | presence truth | ✅ existing flow |

The preview stream is a **view, not a record**. Do not treat it as an audit trail.

---

## 🏷️ Label Lifecycle

Recognition resolves once, but the label must not be permanent.

| Risk | Mitigation |
|------|-----------|
| Tracker ID switch → name on wrong person | re-verify periodically, not once |
| Face inference fails 2s (head turn) | label has confidence + timeout, not a boolean |
| Stale label after track reuses the id | `label_expires_at` forces re-check |

> 🔴 **ID switch is the main correctness risk.** If person A is labelled `EMP_12345`
> on id 7, then A steps behind B and id 7 moves to B, you show **the wrong person's
> name** — confidently wrong, which is worse than showing "unknown".

**Recognise once, then keep re-verifying** every few seconds. If the face under that
id no longer matches the stored identity, relabel.

> Verify tracker id stability against your actual scenes before tuning expiry.
> Record a group entering, count id switches, and let that set the interval.

---

## ⚙️ Queues

The decode thread hands off to the main loop via a **drop-oldest** queue
(`maxsize=1`), and the publisher holds **one** pending frame, replacing it if a
newer one arrives. Old video is worthless, so it is never queued.

```
decode thread ──→ [frame_queue maxsize=1] ──→ main loop ──→ YOLO (every frame)
                                                   │
                                                   ├──→ [pending slot] ──→ LiveKit
                                                   └──→ face pipeline (throttled)
```

- A slow browser must **never** stall recognition
- Stalled inference must **never** stall the preview
- Dropping a preview frame always beats blocking the loop

> The preview submit is fire-and-forget: it copies the frame, hands it to the
> publisher thread, and returns. No LiveKit call is ever made on the frame path.

---

## 🔧 Throughput Work

`PREVIEW_FPS` (default 15) and `PREVIEW_WIDTH` (default 640) are the two knobs.
A slow browser reduces its own stream; it does not slow recognition.

| Change | Status |
|--------|--------|
| **Skip recognition for labelled tracks** | ✅ done — labelled tracks are excluded from the eligible set |
| **Cap recognition per frame** | ✅ done — `RECOGNIZE_TRACKS_PER_FRAME` (default 2), rotated round-robin |
| **GPU inference for SCRFD/ArcFace** | ✅ done — `onnxruntime-gpu`, providers resolved at startup and logged |
| **GPU delegate for MediaPipe** | ✅ done — tries GPU, silently retries CPU |
| **Downscale published frames** | ✅ done — `PREVIEW_WIDTH`, browser upscaling to canvas is trivial |
| **Move HTTP calls out of the frame loop** | 🚧 still blocking — `update_unknown` / `create_unknown` round-trip per frame |
| **Measure real AI FPS** | 🚧 not done — required before tuning any of the above |

> Recognition is a *state machine over time*, not a per-frame result. A track that
> takes 3 frames to identify is fine at 15 fps — 200ms. Subsampling costs
> responsiveness, never correctness.
>
> The remaining HTTP round trips are the last real blocker. Until they are
> backgrounded, per-frame latency still has a network component in it.

---

## 📐 Runtime Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu`; `auto` prefers GPU, falls back |
| `PREVIEW_ENABLED` | `false` | master switch; off means zero LiveKit cost |
| `LIVEKIT_URL` | — | `wss://…` from the LiveKit project |
| `LIVEKIT_API_KEY` | — | publisher credential, `.env` only |
| `LIVEKIT_API_SECRET` | — | publisher credential, `.env` only |
| `PREVIEW_FPS` | `15` | publish rate cap |
| `PREVIEW_WIDTH` | `640` | published width; height follows source aspect |
| `RECOGNIZE_TRACKS_PER_FRAME` | `2` | `0` = unlimited (old behaviour) |
| `ROI_PAD_X` / `ROI_PAD_Y` | `8` / `20` | person ROI growth, group safety |
| `FACE_OWNERSHIP_MARGIN` | `0.06` | winning-face lead required over runner-up |

`DEVICE` resolution happens **once at import** and is shared by every camera
thread, so the startup log line is the authoritative record of what is actually
running. Read it at boot:

```
[DEVICE] pref=auto torch=cuda:0 onnx=CUDAExecutionProvider,CPUExecutionProvider mediapipe=GPU encoder=libx264
```

If that says `cuda:0` but throughput is poor, the problem is the model or the
batch size, not the provider.

---

## ✅ Checklist

**Done in this change**

- [x] YOLO tracking runs on every frame, in the publish path
- [x] Face pipeline subsampled per unlabelled track, rotated round-robin
- [x] `frameTs` stamped at decode, before inference
- [x] Complete scene state per publish (no deltas)
- [x] Preview queue drop-oldest; publisher never blocks the frame path
- [x] LiveKit room = camera `code`, deterministic
- [x] GPU-first ONNX / MediaPipe / YOLO with automatic CPU fallback
- [x] Preview defaults to off, so enabling is a deliberate act
- [x] LiveKit credentials read from `.env` only, never committed

**Still open**

- [ ] Move HTTP calls out of the frame loop (last per-frame blocker)
- [ ] Browser tokens: subscribe-only, short-lived, minted server-side
- [ ] TURN verified reachable from an external network
- [ ] Node RTSP preview path removed
- [ ] Real LiveKit end-to-end test from an external network
- [ ] Measure real AI FPS on production GPU, then tune `PREVIEW_FPS`
- [ ] Label re-verification interval, tuned from recorded id-switch data
- [ ] Decided whether preview carries the face crop or bbox + name only
- [ ] Presence event flow (`ALLOWED_EVENTS`) verified unchanged after this change

> 🔁 **Rotate the LiveKit API key/secret before deploying.** They were shared in
> plaintext while configuring this; treat them as compromised even if never
> committed. `git diff` is clean, but a secret only has to exist once to leak.
