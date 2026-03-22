import cv2
import numpy as np

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

    if state is None:
        track_embedding_state[person_id] = {
            "ref": embedding,
            "last": embedding,
            "samples": [(embedding, quality)],
            "last_quality": quality
        }
        return True

    ref = state["ref"]
    last = state["last"]
    last_quality = state["last_quality"]

    sim_ref = float(np.dot(embedding, ref))
    sim_last = float(np.dot(embedding, last))

    # adaptive identity check
    if sim_ref < (0.40 if len(state["samples"]) > 2 else 0.40):
        print(f"Rejected embedding for person {person_id} due to low similarity to reference: {sim_ref:.2f} threshold: {(0.40 if len(state['samples']) > 2 else 0.40):.2f}")
        return False

    # relative improvement
    quality_improved = quality > last_quality * 1.15

    if sim_last < 0.40 and not quality_improved:
        print(f"Rejected embedding for person {person_id} due to low similarity to last known embedding: {sim_last:.2f} threshold: 0.40 and no quality improvement: {quality_improved}")
        return False

    # accept
    state["samples"].append((embedding, quality))

    # top-K with diversity
    filtered = []
    for emb, q in sorted(state["samples"], key=lambda x: x[1], reverse=True):
        if all(np.dot(emb, e) < 0.95 for e, _ in filtered):
            filtered.append((emb, q))
        if len(filtered) >= 5:
            break

    state["samples"] = filtered

    # recompute reference
    top_embeddings = [e for e, _ in state["samples"]]
    ref_new = np.mean(top_embeddings, axis=0)

    norm = np.linalg.norm(ref_new)
    if norm > 0:
        ref_new /= norm

    state["ref"] = ref_new

    # update last only if not degrading badly
    if quality >= last_quality * 0.9:
        state["last"] = embedding
        state["last_quality"] = quality

    return True
    state = track_embedding_state.get(person_id)

    if state is None:
        track_embedding_state[person_id] = {
            "ref": embedding,
            "last": embedding,
            "samples": [(embedding, quality)],
            "last_quality": quality
        }
        return True

    ref = state["ref"]
    last = state["last"]
    last_quality = state["last_quality"]

    sim_ref = float(np.dot(embedding, ref))
    sim_last = float(np.dot(embedding, last))

    # 🚫 hard identity check (never relax this too much)
    if sim_ref < 0.5:
        return False

    # 🔥 improvement override
    quality_improved = quality > last_quality + 0.1

    if sim_last < 0.7 and not quality_improved:
        return False

    # ✅ accept
    state["samples"].append((embedding, quality))

    # keep only top-K by quality
    state["samples"] = sorted(
        state["samples"],
        key=lambda x: x[1],  # sort by quality
        reverse=True
    )[:5]  # K = 5 (tune this)

    if len(state["history"]) > 10:
        state["history"].pop(0)

    top_embeddings = [e for e, q in state["samples"]]
    ref_new = np.mean(top_embeddings, axis=0)
    ref_new /= np.linalg.norm(ref_new)

    state["ref"] = ref_new
    state["last"] = embedding
    state["last_quality"] = quality

    return True   
    state = track_embedding_state.get(person_id)

    # First frame → accept
    if state is None:
        track_embedding_state[person_id] = {
            "ref": embedding,
            "last": embedding,
            "history": [embedding]
        }
        return True

    ref = state["ref"]
    last = state["last"]

    sim_ref = float(np.dot(embedding, ref))
    sim_last = float(np.dot(embedding, last))

    # hard rejection (wrong face / noise)
    if sim_ref < 0.5:
        print(f"Rejected embedding for person {person_id} due to low similarity to reference: {sim_ref:.2f}")
        return False

    # temporal instability (sudden jump)
    if sim_last < 0.6:
        print(f"Rejected embedding for person {person_id} due to temporal instability: {sim_last:.2f}")
        return False

    # accept → update state
    state["history"].append(embedding)

    # keep bounded memory
    if len(state["history"]) > 10:
        state["history"].pop(0)

    # update reference (mean)
    ref_new = np.mean(state["history"], axis=0)
    ref_new /= np.linalg.norm(ref_new)

    state["ref"] = ref_new
    state["last"] = embedding

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