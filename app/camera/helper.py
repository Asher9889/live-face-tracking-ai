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
            "history": [embedding],
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
    state["history"].append(embedding)

    if len(state["history"]) > 10:
        state["history"].pop(0)

    ref_new = np.mean(state["history"], axis=0)
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