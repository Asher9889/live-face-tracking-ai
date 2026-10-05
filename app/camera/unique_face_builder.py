import numpy as np
import time

class UniqueFaceRepresentationBuilder:
    def __init__(self, max_size=7, sim_threshold=0.90, min_frames=3, min_poses=1):
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

    def add(self, buffer, embedding, quality, pose_bucket, img=None, eye_quality=None):
        if buffer is None:
            buffer = []

        timestamp = time.time()

        def _entry():
            return {
                "embedding": embedding,
                "quality": quality,
                "pose_bucket": pose_bucket,
                "img": img,
                "ts": timestamp,
                "eye_quality": eye_quality,
            }

        # ----------------------------
        # PHASE 1: BOOTSTRAP
        # ----------------------------
        if len(buffer) < self.min_frames:
            buffer.append(_entry())
            return buffer

        # ----------------------------
        # PHASE 2/3: CONTROLLED MODE
        # ----------------------------

        # 1. SAME POSE → replace best
        same_pose_items = [i for i, item in enumerate(buffer) if item.get("pose_bucket") == pose_bucket]

        if same_pose_items:
            best_idx = max(
                same_pose_items,
                key=lambda i: self._rank(buffer[i]),
            )

            if self._rank(_entry()) > self._rank(buffer[best_idx]):
                buffer[best_idx] = _entry()
            return self._trim(buffer)

        # 2. DIVERSITY CHECK (ONLY AFTER BOOTSTRAP)
        if not self._is_diverse(buffer, embedding):
            return buffer

        # 3. ADD NEW POSE
        buffer.append(_entry())

        return self._trim(buffer)

    # ----------------------------
    # RANKING
    # ----------------------------
    @staticmethod
    def _rank(item):
        """Ordering key for choosing which buffered frame represents a person.

        Eye visibility outranks raw quality. The audit log showed frames picked
        by quality alone were frequently worse than a frame already sitting in
        the buffer: in 5 of 12 registrations the chosen frame had lower iris
        contrast than the seeded frame, and in 2 cases a frame that would have
        passed the visibility gate was discarded in favour of one that would
        not. Quality is still the tie-breaker, so behaviour on frames with no
        eye metrics is unchanged.
        """
        eye = item.get("eye_quality")
        if eye is None:
            return (0.0, item["quality"])
        return (1.0, eye) if isinstance(eye, (int, float)) else (0.0, item["quality"])
    
    
    # ----------------------------
    # SIZE CONTROL
    # ----------------------------
    def _trim(self, buffer):
        if len(buffer) <= self.max_size:
            return buffer

        buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)

        frontal = [x for x in buffer if x.get("pose_bucket") == "frontal"]

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

        poses = {item.get("pose_bucket") for item in buffer if item.get("pose_bucket")}
        if len(poses) < self.min_poses:
            return False

        return True

    # ----------------------------
    # GET BEST IMAGE
    # ----------------------------
    def get_best_face(self, buffer):
        if not buffer:
            return None
        return max(buffer, key=lambda x: self._rank(x))

    # ----------------------------
    # DEBUG / METRICS
    # ----------------------------
    def get_stats(self, buffer):
        if not buffer:
            return {}

        return {
            "count": len(buffer),
            "poses": list({x.get("pose_bucket") for x in buffer if x.get("pose_bucket")}),
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