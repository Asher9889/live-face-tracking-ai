import cv2
import numpy as np
from datetime import datetime


def is_blurry(frame, threshold=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var() < threshold

def motion_score(prev, curr):
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff.mean()
 
def is_stable_embedding(track_embedding_state, person_id, embedding, quality):
    state = track_embedding_state.get(person_id)

    # ----------------------------
    # INIT
    # ----------------------------
    if state is None:
        track_embedding_state[person_id] = {
            "ref": embedding,
            "last": embedding,
            "samples": [(embedding, quality)],
            "last_quality": quality
        }
        print(f"[IsStable][Person {person_id}] init accept (quality={quality:.3f})")
        return True

    ref = state["ref"]
    last = state["last"]
    last_quality = state["last_quality"]

    sim_ref = float(np.dot(embedding, ref))
    sim_last = float(np.dot(embedding, last))

    # ----------------------------
    # 1. STRONG IDENTITY CHECK
    # ----------------------------
    if sim_ref < 0.45: ## How similar is this face to the overall identity so far?
        print(f"[IsStable][Person {person_id}] rejected: low sim to ref {sim_ref:.3f} < 0.45")
        return False

    # ----------------------------
    # 2. TEMPORAL STABILITY
    # ----------------------------
    if sim_last < 0.40: # “How similar is this face to the previous frame?”
        print(f"[IsStable][Person {person_id}] rejected: temporal instability sim_last={sim_last:.3f} < 0.40")
        return False

    # ----------------------------
    # 3. QUALITY SAFETY (no override)
    # ----------------------------
    if quality < last_quality * 0.7:
        print(f"[IsStable][Person {person_id}] rejected: quality drop {quality:.3f} < {last_quality * 0.7:.3f} (last={last_quality:.3f})")
        return False

    # ----------------------------
    # 4. ACCEPT
    # ----------------------------
    state["samples"].append((embedding, quality))

    # ----------------------------
    # 5. DIVERSITY FILTER
    # ----------------------------
    filtered = []
    for emb, q in sorted(state["samples"], key=lambda x: x[1], reverse=True):
        if all(np.dot(emb, e) < 0.90 for e, _ in filtered):
            filtered.append((emb, q))
        if len(filtered) >= 5:
            break

    state["samples"] = filtered

    # ----------------------------
    # 6. REFERENCE UPDATE (SAFE)
    # ----------------------------
    top_embeddings = [e for e, _ in state["samples"]]

    ref_new = np.mean(top_embeddings, axis=0)
    ref_new /= np.linalg.norm(ref_new)

    state["ref"] = ref_new

    # ----------------------------
    # 7. UPDATE LAST
    # ----------------------------
    state["last"] = embedding
    state["last_quality"] = quality

    print(f"[IsStable][Person {person_id}] accepted (sim_ref={sim_ref:.3f} sim_last={sim_last:.3f} quality={quality:.3f})")

    return True

def expand_bbox(bbox, frame_w, frame_h):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    # 🔥 asymmetric expansion
    top_scale = 0.20   # more space for head
    bottom_scale = 0.05
    side_scale = 0.10

    x1 = max(0, int(x1 - w * side_scale))
    x2 = min(frame_w, int(x2 + w * side_scale))

    y1 = max(0, int(y1 - h * top_scale))     # 🔥 bigger expansion on top
    y2 = min(frame_h, int(y2 + h * bottom_scale))

    return x1, y1, x2, y2

def score_face(face, roi_shape):
    h, w, _ = roi_shape
    cx, cy = w / 2, h / 2

    x1, y1, x2, y2 = face["bbox"]
    fx = (x1 + x2) / 2
    fy = (y1 + y2) / 2

    dist = ((fx - cx)**2 + (fy - cy)**2)**0.5
    dist_score = 1 - (dist / max(w, h))

    return 0.7 * face["quality"] + 0.3 * dist_score


def select_best_face(faces_with_quality):
    """
    Select best face deterministically and stably.
    """

    if not faces_with_quality:
        return None

    # ----------------------------
    # 1. SORT (multi-criteria)
    # ----------------------------
    faces_sorted = sorted(
        faces_with_quality,
        key=lambda f: (
            f["quality"],            # primary
            f["score"],              # detection confidence
            (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])  # size
        ),
        reverse=True
    )

    best = faces_sorted[0]

    # ----------------------------
    # 2. DOMINANCE CHECK (anti-noise)
    # ----------------------------
    if len(faces_sorted) > 1:
        second = faces_sorted[1]

        # if too close → unstable selection
        if abs(best["quality"] - second["quality"]) < 0.05:
            # fallback to larger face (more reliable embedding)
            best_area = (best["bbox"][2] - best["bbox"][0]) * (best["bbox"][3] - best["bbox"][1])
            second_area = (second["bbox"][2] - second["bbox"][0]) * (second["bbox"][3] - second["bbox"][1])

            if second_area > best_area:
                best = second

    return best

def crop_with_margin(frame, x1, y1, x2, y2, margin=0.2):
    h, w = frame.shape[:2]

    # 🔥 Validate bbox
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    # 🔥 symmetric margin (total = margin, not 2x)
    mx = int(bw * margin / 2)
    my = int(bh * margin / 2)

    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)

    cropped = frame[ny1:ny2, nx1:nx2]

    # 🔥 safety check
    if cropped is None or cropped.size == 0:
        return None

    return cropped

def get_pose_name(yaw: float | None) -> str | None:
    if yaw is None:
        return None

    if yaw <= -20:
        return "left"
    elif -20 < yaw <= -10:
        return "left_mid"
    elif -10 < yaw < 10:
        return "frontal"
    elif 10 <= yaw < 20:
        return "right_mid"
    elif yaw >= 20:
        return "right"

    return None

def now_ms():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]
