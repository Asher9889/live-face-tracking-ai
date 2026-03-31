import numpy as np
import time

class UniqueFaceRepresentationBuilder:
    def __init__(self, max_size=5, sim_threshold=0.90, min_frames=3, min_poses=2):
        self.max_size = max_size
        self.sim_threshold = sim_threshold
        self.min_frames = min_frames
        self.min_poses = min_poses

    # ----------------------------
    # INTERNAL: Diversity check
    # ----------------------------
    def _is_diverse(self, existing, new_emb):
        for item in existing:
            sim = float(np.dot(item["embedding"], new_emb))
            if sim > self.sim_threshold:
                return False
        return True

    # ----------------------------
    # ADD FACE
    # ----------------------------
    def add(self, buffer, embedding, quality, pose_bucket, img=None):
        if buffer is None:
            buffer = []

        timestamp = time.time()

        # 1. Replace same pose (upgrade)
        for i, item in enumerate(buffer):
            if item["pose_bucket"] == pose_bucket:
                if quality > item["quality"]:
                    buffer[i] = {
                        "embedding": embedding,
                        "quality": quality,
                        "pose_bucket": pose_bucket,
                        "img": img,
                        "ts": timestamp
                    }
                return buffer

        # 2. Diversity check
        if not self._is_diverse(buffer, embedding):
            for i, item in enumerate(buffer):
                sim = float(np.dot(item["embedding"], embedding))
                if sim > self.sim_threshold and quality > item["quality"]:
                    buffer[i] = {
                        "embedding": embedding,
                        "quality": quality,
                        "pose_bucket": pose_bucket,
                        "img": img,
                        "ts": timestamp
                    }
                    break
            return buffer

        # 3. Add new
        buffer.append({
            "embedding": embedding,
            "quality": quality,
            "pose_bucket": pose_bucket,
            "img": img,
            "ts": timestamp
        })

        # 4. Maintain size
        buffer = self._trim(buffer)

        return buffer

    # ----------------------------
    # SIZE CONTROL
    # ----------------------------
    def _trim(self, buffer):
        if len(buffer) <= self.max_size:
            return buffer

        buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)

        frontal = [x for x in buffer if x["pose_bucket"] == "frontal"]

        if frontal:
            best_frontal = frontal[0]
            others = [x for x in buffer if x != best_frontal]
            return [best_frontal] + others[:self.max_size - 1]

        return buffer[:self.max_size]

    # ----------------------------
    # READINESS CHECK (IMPORTANT)
    # ----------------------------
    def is_ready(self, buffer):
        if not buffer:
            return False

        if len(buffer) < self.min_frames:
            return False

        poses = {item["pose_bucket"] for item in buffer}
        if len(poses) < self.min_poses:
            return False

        return True

    # ----------------------------
    # GET BEST IMAGE
    # ----------------------------
    def get_best_face(self, buffer):
        if not buffer:
            return None
        return max(buffer, key=lambda x: x["quality"])

    # ----------------------------
    # DEBUG / METRICS
    # ----------------------------
    def get_stats(self, buffer):
        if not buffer:
            return {}

        return {
            "count": len(buffer),
            "poses": list({x["pose_bucket"] for x in buffer}),
            "max_quality": max(x["quality"] for x in buffer),
            "avg_quality": float(np.mean([x["quality"] for x in buffer]))
        }

    # ----------------------------
    # BUILD FINAL EMBEDDING
    # ----------------------------
    def build(self, buffer):
        if not self.is_ready(buffer):
            return None

        embeddings = np.array([x["embedding"] for x in buffer])
        weights = np.array([x["quality"] for x in buffer])

        if weights.sum() == 0:
            weights = np.ones_like(weights)

        centroid = np.average(embeddings, axis=0, weights=weights)

        norm = np.linalg.norm(centroid)
        if norm == 0:
            return None

        centroid /= norm
        return centroid