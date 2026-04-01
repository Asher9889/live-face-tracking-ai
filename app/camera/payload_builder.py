import time
import cv2


def _encode_image(img):
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]

    target_w = 160

    # ✅ Only downscale, never upscale
    if w > target_w:
        scale = target_w / w
        img = cv2.resize(img, (target_w, int(h * scale)))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    ok, buf = cv2.imencode(".jpg", img, encode_param)

    if not ok:
        return None

    return buf.tobytes()


def _build_pose_data(buffer):
    pose_map = {}

    for item in buffer:
        pose = item["pose_bucket"]
        img = item["img"]

        if img is None or img.size == 0:
            continue

        h, w = img.shape[:2]

        # encode image
        image_bytes = _encode_image(img)
        if not image_bytes:
            continue

        # keep BEST per pose
        if pose not in pose_map or item["quality"] > pose_map[pose]["quality"]:
            pose_map[pose] = {
                "embedding": item["embedding"].tolist(),
                "quality": float(item["quality"]),
                "face_size": {"w": int(w), "h": int(h)},
                "image_bytes": image_bytes,
                "ts": int(item["ts"] * 1000)
            }

    return pose_map

def build_unknown_payload(
    buffer,
    centroid,
    cam_code,
    unknown_id=None,
    builder=None,
    update_context=None
):
    """
    Build payload for both CREATE and UPDATE.
    """

    payload = {
        "unknown_id": unknown_id,
        "camera_code": cam_code,
        "timestamp": int(time.time() * 1000),

        "centroid_embedding": centroid.tolist(),
        "embedding_count": len(buffer),

        "poses": _build_pose_data(buffer),
    }

    # -------------------------
    # BEST FACE
    # -------------------------
    if builder:
        best = builder.get_best_face(buffer)
    else:
        best = max(buffer, key=lambda x: x["quality"]) if buffer else None

    if best:
        payload["best_quality"] = float(best["quality"])
        payload["best_pose"] = best["pose_bucket"]

        encoded = _encode_image(best["img"])
        if encoded:
            payload["best_image"] = encoded

    # -------------------------
    # BUILDER STATS
    # -------------------------
    if builder:
        stats = builder.get_stats(buffer)

        payload["builder_stats"] = {
            "count": stats.get("count"),
            "poses": stats.get("poses"),
            "max_quality": stats.get("max_quality"),
            "avg_quality": stats.get("avg_quality"),
        }

    # -------------------------
    # UPDATE CONTEXT
    # -------------------------
    if update_context:
        payload["update_context"] = update_context

    return payload