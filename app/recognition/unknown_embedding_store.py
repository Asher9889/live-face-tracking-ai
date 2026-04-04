# import numpy as np
# import requests
# import json
# import time

# from app.config.config import envConfig


# class UnknownEmbeddingStore:

#     def __init__(self, api_url, embedding_dim=512):
#         self.api_url = api_url

#         # in-memory state
#         self.unknown_ids = []
#         self.id_to_index = {}

#         self.embeddings = np.empty((0, embedding_dim), dtype=np.float32)
#         self.counts = []

#         # recent duplicate protection
#         self.recent_cache = {}  # unknown_id -> timestamp

#         # config
#         self.match_threshold = 0.50
#         self.margin_threshold = 0.03
#         self.duplicate_threshold = 0.60
#         self.recent_window_sec = 10

#     # ---------------------------------------------------
#     # Utils
#     # ---------------------------------------------------
#     def _safe_normalize(self, emb):
#         norm = np.linalg.norm(emb)
#         if norm == 0:
#             return None
#         return emb / norm

#     def _post(self, url, files, data):
#         try:
#             return requests.post(
#                 url,
#                 files=files,
#                 data=data,
#                 headers={
#                     "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
#                 },
#                 timeout=3
#             )
#         except Exception as e:
#             print("[AI] API request failed:", e)
#             return None

#     # def _blend_embedding(self, old, new, count):
#     #     updated = (old * count + new) / (count + 1)
#     #     norm = np.linalg.norm(updated)
#     #     if norm == 0:
#     #         return old
#     #     return updated / norm

#     def _quality_to_weight(self, q: float) -> float:
#         if q <= 0:
#             return 0.0

#         return min(round(q, 5), 10.0)


#     def _blend_embedding(self, old, new, old_weight, new_weight):
#         updated = (old * old_weight + new * new_weight) / (old_weight + new_weight)
#         norm = np.linalg.norm(updated)
#         if norm == 0:
#             return old
#         return updated / norm

#     # ---------------------------------------------------
#     # Load from Node
#     # ---------------------------------------------------
#     def load_unknown_embeddings(self):

#         print("[AI] Loading unknown embeddings...")

#         response = requests.get(
#             self.api_url,
#             headers={
#                 "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
#             },
#             timeout=5
#         )

#         data = response.json()

#         if not data.get("success"):
#             raise Exception("Failed to load unknown embeddings")

#         vectors = []

#         for u in data["data"]:
#             emb = np.array(u["representativeEmbedding"], dtype=np.float32)
#             emb = self._safe_normalize(emb)
#             if emb is None:
#                 continue

#             idx = len(self.unknown_ids)

#             self.unknown_ids.append(u["id"])
#             self.id_to_index[u["id"]] = idx
#             self.counts.append(u.get("embeddingCount", 1))

#             vectors.append(emb)

#         if vectors:
#             self.embeddings = np.stack(vectors)

#         print(f"[AI] Loaded {len(self.unknown_ids)} unknown identities")

#     # ---------------------------------------------------
#     # Match unknown
#     # ---------------------------------------------------
#     def find_match(self, embedding):

#         if self.embeddings.shape[0] == 0:
#             return None

#         embedding = self._safe_normalize(embedding)
#         if embedding is None:
#             return None

#         scores = np.dot(self.embeddings, embedding)

#         best_idx = np.argmax(scores)
#         best_score = scores[best_idx]

#         if best_score < self.match_threshold:
#             return None

#         # margin check (VERY important)
#         if len(scores) > 1:
#             second_score = np.partition(scores, -2)[-2]
#             if best_score - second_score < self.margin_threshold:
#                 return None

#         return {
#             "unknown_id": self.unknown_ids[best_idx],
#             "similarity": float(best_score),
#             "index": int(best_idx)
#         }

#     # ---------------------------------------------------
#     # Duplicate protection (recent window)
#     # ---------------------------------------------------
#     def _check_recent_duplicate(self, embedding):
#         now = time.time()

#         for uid, ts in list(self.recent_cache.items()):
#             if now - ts > self.recent_window_sec:
#                 del self.recent_cache[uid]
#                 continue

#             idx = self.id_to_index.get(uid)
#             if idx is None:
#                 continue

#             sim = np.dot(self.embeddings[idx], embedding)
#             if sim > self.duplicate_threshold:
#                 return uid

#         return None

#     # ---------------------------------------------------
#     # Create new unknown
#     # ---------------------------------------------------
    
#     def add_unknown(self, payload):
#         try:
#             # -------------------------
#             # Normalize centroid
#             # -------------------------
#             centroid = np.array(payload["centroid_embedding"], dtype=np.float32)
#             # centroid = self._safe_normalize(centroid)

#             if centroid is None:
#                 return None

#             # -------------------------
#             # Duplicate protection
#             # -------------------------
#             dup = self._check_recent_duplicate(centroid)
#             if dup:
#                 return dup

#             # -------------------------
#             # Extract poses
#             # -------------------------
#             poses = payload.get("poses")
#             if not poses:
#                 print("[AI] No poses found in payload")
#                 return {
#                     "error": "No poses in payload"
#                 }

#             # -------------------------
#             # Build clean poses (NO image_bytes)
#             # -------------------------
#             clean_poses = {}
#             for pose_name, pose_data in poses.items():
#                 pose_copy = dict(pose_data)
#                 pose_copy.pop("image_bytes", None)  # remove binary
#                 clean_poses[pose_name] = pose_copy

#             # -------------------------
#             # Build JSON payload
#             # -------------------------
#             data = {
#                 "unknown_id": payload.get("unknown_id"),
#                 "camera_code": payload.get("camera_code"),
#                 "timestamp": str(payload.get("timestamp")),
#                 "centroid_embedding": json.dumps(centroid.tolist()),
#                 "embedding_count": str(payload.get("embedding_count")),
#                 "poses": json.dumps(clean_poses),
#                 "builder_stats": json.dumps(payload.get("builder_stats", {}))
#             }

#             # -------------------------
#             # Build multipart request
#             # -------------------------
#             multipart_files = {}

#             # ✅ Image parts
#             for pose_name, pose_data in poses.items():
#                 img_bytes = pose_data.get("image_bytes")
#                 if img_bytes:
#                     multipart_files[f"face_{pose_name}"] = (
#                         f"{pose_name}.jpg",
#                         img_bytes,
#                         "image/jpeg"
#                     )

#             response = requests.post(
#                 # "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
#                 envConfig.NODE_CREATE_UNKNOWN_URL,
#                 files=multipart_files,
#                 data=data
#             )

#             if not response:
#                 return None

#             # -------------------------
#             # Safe response parsing
#             # -------------------------
#             try:
#                 res = response.json()
#             except Exception:
#                 print("[DEBUG] Non-JSON response:", response.text)
#                 return "debug_unknown_id"  # for webhook testing

#             if not res.get("success"):
#                 print("[AI] Node API error:", res)
#                 return None

#             unknown_id = res["data"]["unknownId"]

#             # -------------------------
#             # Update local cache
#             # -------------------------
#             idx = len(self.unknown_ids)

#             self.unknown_ids.append(unknown_id)
#             self.id_to_index[unknown_id] = idx

#             self.embeddings = np.vstack([self.embeddings, centroid])
#             self.counts.append(1)

#             self.recent_cache[unknown_id] = time.time()

#             return unknown_id

#         except Exception as e:
#             print("[AI] add_unknown failed:", e)
#             return None
    
#     # ---------------------------------------------------
#     # Update existing unknown
#     # ---------------------------------------------------
#     def update_unknown(self, unknown_id, centroid, timestamp, camera_code, image_bytes, poses, quality):

#         idx = self.id_to_index.get(unknown_id)
#         if idx is None:
#             return None

#         centroid = self._safe_normalize(centroid)
#         if centroid is None:
#             return None

#         # blend embedding locally
#         # use count as old weight, and NEW QUALITY as new weight
#         new_quality = self._quality_to_weight(quality)  # fallback (we'll improve next)
#         updated = self._blend_embedding(
#             self.embeddings[idx],
#             centroid,
#             old_weight = self.counts[idx],
#             new_weight = self.counts[idx] * new_quality
#         )

#         self.embeddings[idx] = updated
#         self.counts[idx] += 1

#         files = {
#             "face": ("face.jpg", image_bytes, "image/jpeg")
#         }

#         data = {
#             "unknownId": unknown_id,
#             "meanEmbedding": json.dumps(updated.tolist()),
#             "timestamp": str(timestamp),
#             "cameraCode": camera_code,
#             "poses": json.dumps(list(poses))
#         }

#         self._post(
#             # envConfig.NODE_UPDATE_UNKNOWN_URL, 
#             "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
#             files, 
#             data
#         )

#         self.recent_cache[unknown_id] = time.time()

#         return unknown_id

#     # ---------------------------------------------------
#     # Unified API (USE THIS IN PIPELINE)
#     # ---------------------------------------------------
#     def register_or_update(self, centroid, image_bytes, timestamp, camera_code, embedding_count, poses):

#         centroid = self._safe_normalize(centroid)
#         if centroid is None:
#             return None

#         match = self.find_match(centroid)

#         if match:
#             return self.update_unknown(
#                 match["unknown_id"],
#                 centroid,
#                 timestamp,
#                 camera_code,
#                 image_bytes,
#                 poses
#             )

#         return self.add_unknown(
#             centroid,
#             image_bytes,
#             timestamp,
#             camera_code,
#             embedding_count,
#             poses
#         )

#     # ---------------------------------------------------
#     # Debug / Metrics
#     # ---------------------------------------------------
#     def stats(self):
#         return {
#             "total_unknowns": len(self.unknown_ids),
#             "avg_embeddings_per_identity": float(np.mean(self.counts)) if self.counts else 0
#         }


# unknown_embedding_store = UnknownEmbeddingStore(
#     api_url=envConfig.NODE_LOAD_UNKNOWN_EMBEDDINGS_URL
# )


import json

import numpy as np
import requests
import time
from app.config.config import envConfig


# =============================
# SNAPSHOT DATA (IMMUTABLE)
# =============================
class StoreData:
    def __init__(self, centroid_matrix, pose_matrix, pose_owner, unknown_ids, uid_to_pose_quality, uid_to_pose_indices):
        self.centroid_matrix = centroid_matrix
        self.pose_matrix = pose_matrix
        self.pose_owner = pose_owner
        self.unknown_ids = unknown_ids
        #  for stats and smart update
        self.uid_to_pose_quality = uid_to_pose_quality
        self.uid_to_pose_indices = uid_to_pose_indices
        
    

# =============================
# MAIN STORE
# =============================
class UnknownEmbeddingStore:

    def __init__(self, api_url, embedding_dim=512):
        self.api_url = api_url
        self.embedding_dim = embedding_dim

        self._store = StoreData(
            centroid_matrix=np.empty((0, embedding_dim), dtype=np.float32),
            pose_matrix=np.empty((0, embedding_dim), dtype=np.float32),
            pose_owner=[],
            unknown_ids=[],
            uid_to_pose_quality={},
            uid_to_pose_indices={}
        )

        # thresholds
        self.centroid_threshold = 0.50
        self.pose_threshold = 0.60
        self.fallback_pose_threshold = 0.70
        self.margin_threshold = 0.03

    # -----------------------------
    # Utils
    # -----------------------------
    def _normalize(self, emb):
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm

    # -----------------------------
    # BUILD STORE
    # -----------------------------
    def _build_store(self, data):

        centroids = []
        poses = []
        pose_owner = []
        unknown_ids = []

        # 🔥 NEW
        uid_to_pose_quality = {}
        uid_to_pose_indices = {}

        pose_idx = 0
        for u in data:
            uid = u["id"]
            unknown_ids.append(uid)

            # centroid
            c = self._normalize(
                np.array(u["representativeEmbedding"], dtype=np.float32)
            )
            if c is None:
                c = np.zeros(self.embedding_dim, dtype=np.float32)
            centroids.append(c)

            # poses
            for pose_name, pose_data in u.get("poses", {}).items():
                emb = self._normalize(
                    np.array(pose_data["embedding"], dtype=np.float32)
                )
                if emb is None:
                    continue

                poses.append(emb)
                pose_owner.append(uid)

                uid_to_pose_indices[uid].append(pose_idx)
                uid_to_pose_quality[uid][pose_name] = pose_data.get("quality", 0)

                pose_idx += 1


        return StoreData(
            centroid_matrix=np.stack(centroids) if centroids else np.empty((0, self.embedding_dim)),
            pose_matrix=np.stack(poses) if poses else np.empty((0, self.embedding_dim)),
            pose_owner=pose_owner,
            unknown_ids=unknown_ids,
            uid_to_pose_quality=uid_to_pose_quality,
            uid_to_pose_indices=uid_to_pose_indices
        )



    # -----------------------------
    # Update pose quality cache
    # -----------------------------
    def update_pose_quality_cache(self, unknown_id, pose, quality):
        store = self._store

        if unknown_id not in store.uid_to_pose_quality:
            store.uid_to_pose_quality[unknown_id] = {}

        store.uid_to_pose_quality[unknown_id][pose] = quality
    

    # ------- Get pose quality for stats and smart update --------
    def get_pose_quality(self, unknown_id, pose_name):
        store = self._store
        return store.uid_to_pose_quality.get(unknown_id, {}).get(pose_name, 0)

    # -----------------------------
    # FULL RELOAD (RCU SAFE)
    # -----------------------------
    def load_unknown_embeddings(self):

        res = requests.get(
            self.api_url,
            headers={"Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"},
            timeout=3
        ).json()

        if not res.get("success"):
            raise Exception("Failed to load unknown embeddings")

        new_store = self._build_store(res["data"])

        # 🔥 atomic swap
        self._store = new_store

        print(f"[AI] Loaded {len(new_store.unknown_ids)} unknowns")

    # -----------------------------
    # MATCH
    # -----------------------------
    def find_match(self, embedding):

        store = self._store  # snapshot

        emb = self._normalize(embedding)
        if emb is None:
            return None

        if store.centroid_matrix.shape[0] == 0:
            return None

        # =============================
        # STAGE 1: CENTROID
        # =============================
        scores = np.dot(store.centroid_matrix, emb)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.centroid_threshold:

            # margin check
            if len(scores) > 1:
                second = np.partition(scores, -2)[-2]
                if best_score - second < self.margin_threshold:
                    return None

            uid = store.unknown_ids[best_idx]

            # =============================
            # STAGE 2: POSE REFINE
            # =============================
            pose_indices = store.uid_to_pose_indices.get(uid, [])

            if not pose_indices:
                return {
                    "unknown_id": uid,
                    "similarity": best_score,
                    "stage": "centroid"
                }

            pose_embs = store.pose_matrix[pose_indices]
            pose_scores = np.dot(pose_embs, emb)

            best_pose_score = float(np.max(pose_scores))

            if best_pose_score >= self.pose_threshold:
                return {
                    "unknown_id": uid,
                    "similarity": best_pose_score,
                    "stage": "pose_refined"
                }

            return None

        # =============================
        # STAGE 3: FALLBACK POSE (IMPORTANT)
        # =============================
        if store.pose_matrix.shape[0] == 0:
            return None

        pose_scores = np.dot(store.pose_matrix, emb)

        best_pose_idx = int(np.argmax(pose_scores))
        best_pose_score = float(pose_scores[best_pose_idx])

        if best_pose_score >= self.fallback_pose_threshold:
            uid = store.pose_owner[best_pose_idx]

            return {
                "unknown_id": uid,
                "similarity": best_pose_score,
                "stage": "pose_fallback"
            }

        return None

    # -----------------------------
    # CREATE (NO LOCAL STATE MUTATION)
    # -----------------------------
    def add_unknown(self, payload):
        try:
            response = requests.post(
                envConfig.NODE_CREATE_UNKNOWN_URL,
                files=payload.get("files", {}),
                data=payload.get("data", {})
            )

            res = response.json()

            if not res.get("success"):
                return None

            return res["data"]["unknownId"]

        except Exception as e:
            print("[AI] add_unknown failed:", e)
            return None

    # -----------------------------
    # UPDATE (FORWARD ONLY)
    # -----------------------------
    def update_unknown(self, unknown_id, centroid, timestamp, camera_code, poses):
        print(f"[DEBUG] Updating unknown {unknown_id} at {timestamp}, poses: {list(poses.keys())}")
        try:
            files = {}
            poses_payload = {}

            # -----------------------------------
            # 🔥 build files + clean pose payload
            # -----------------------------------
            for pose_name, pose_data in poses.items():

                image_bytes = pose_data.get("image")

                if image_bytes:
                    files[f"face_{pose_name}"] = (
                        f"{pose_name}.jpg",
                        image_bytes,
                        "image/jpeg"
                    )

                # remove image from payload (important)
                poses_payload[pose_name] = {
                    "embedding": pose_data["embedding"],
                    "quality": pose_data["quality"],
                    "faceSize": pose_data["faceSize"],
                    "ts": pose_data["ts"]
                }

            print(f"[DEBUG] Files: {list(files.keys())}")

            data = {
                "unknownId": unknown_id,
                "meanEmbedding": centroid,  # optional
                "timestamp": str(timestamp),
                "cameraCode": camera_code,
                "poses": json.dumps(poses_payload)  # 🔥 IMPORTANT
            }
            print(f"[DEBUG] Data payload: {data}")
            requests.patch(
                envConfig.NODE_UPDATE_UNKNOWN_URL,
                # "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
                files=files,
                data=data
            )

            return unknown_id

        except Exception as e:
            print("[AI] update_unknown failed:", e)
            return None

    # -----------------------------
    # STATS
    # -----------------------------
    def stats(self):
        store = self._store
        return {
            "total_unknowns": len(store.unknown_ids),
            "total_poses": len(store.pose_owner)
        }


unknown_embedding_store = UnknownEmbeddingStore(
    api_url=envConfig.NODE_LOAD_UNKNOWN_EMBEDDINGS_URL
)