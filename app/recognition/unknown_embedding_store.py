# import numpy as np
# import requests
# import json

# from app.config.config import envConfig

# class UnknownEmbeddingStore:

#     def __init__(self, api_url):
#         self.api_url = api_url

#         self.unknown_ids = []
#         self.id_to_index = {}

#         self.embeddings = np.empty((0, 512), dtype=np.float32)
#         self.counts = []

#     # ---------------------------------------------------
#     # Load unknown identities from Node API on startup
#     # ---------------------------------------------------
#     def load_unknown_embeddings(self):

#         print("[AI] Loading unknown embeddings...")

#         response = requests.get(
#             self.api_url,
#             headers={
#                 "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
#             }
#         )

#         data = response.json()

#         if not data.get("success"):
#             raise Exception("Failed to load unknown embeddings")

#         unknowns = data["data"]

#         vectors = []

#         for u in unknowns:

#             emb = np.array(u["representativeEmbedding"], dtype=np.float32)
#             emb = emb / np.linalg.norm(emb)

#             idx = len(self.unknown_ids)

#             self.unknown_ids.append(u["id"])
#             self.id_to_index[u["id"]] = idx
#             self.counts.append(u["embeddingCount"])

#             vectors.append(emb)

#         if vectors:
#             self.embeddings = np.stack(vectors)

#         print(f"[AI] Loaded {len(self.unknown_ids)} unknown identities")

#     # ---------------------------------------------------
#     # Search unknown identities
#     # ---------------------------------------------------
#     def find_match(self, embedding, threshold=0.50):

#         if self.embeddings.shape[0] == 0:
#             return None

#         scores = np.dot(self.embeddings, embedding)

#         best_idx = np.argmax(scores)
#         best_score = scores[best_idx]

#         # threshold check
#         if best_score < threshold:
#             return None

#         # # 🔥 margin check (important)
#         # if len(scores) > 1:
#         #     second_score = np.partition(scores, -2)[-2]
#         #     if best_score - second_score < 0.05:
#         #         return None

#         return {
#             "unknown_id": self.unknown_ids[best_idx],   
#             "similarity": float(best_score)
#         }
#     # ---------------------------------------------------
#     # Create new unknown identity
#     # ---------------------------------------------------
#     def add_unknown(self, centroid_embedding, image_bytes, timestamp, camera_code, embedding_count):
#         try:
#             centroid_embedding = centroid_embedding / np.linalg.norm(centroid_embedding)

#             files = {
#                 "face": ("face.jpg", image_bytes, "image/jpeg")
#             }

#             data = {
#                 "representativeEmbedding": json.dumps(centroid_embedding.tolist()),
#                 "timestamp":  str(timestamp),
#                 "cameraCode": camera_code,
#                 "embeddingCount": embedding_count
#             }

#             response = requests.post(
#                 envConfig.NODE_CREATE_UNKNOWN_URL,
#                 files=files,
#                 data=data,
#                 headers={
#                     "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
#                 }
#             )

#             data = response.json()

#             if not data.get("success"):
#                 print("[AI] Node API returned error:", data)
#                 return None

#             unknown_id = data["data"]["unknownId"]

#             idx = len(self.unknown_ids)

#             self.unknown_ids.append(unknown_id)
#             self.id_to_index[unknown_id] = idx

#             self.embeddings = np.vstack([self.embeddings, centroid_embedding])
#             self.counts.append(1)

#             return unknown_id

#         except requests.exceptions.RequestException as e:
#             print("[AI] Node API request failed:", e)

#         except Exception as e:
#             print("[AI] Unknown creation error:", e)

#         return None

#     # ---------------------------------------------------
#     # Update existing unknown identity
#     # ---------------------------------------------------
#     def update_unknown(self, unknown_id, centroid, timestamp, camera_code, image_bytes):

#         # idx = self.id_to_index[unknown_id]

#         # centroid = self.embeddings[idx]
#         # count = self.counts[idx]

#         # updated = (centroid * count + new_embedding) / (count + 1)
#         # updated = updated / np.linalg.norm(updated)

#         # self.embeddings[idx] = updated
#         # self.counts[idx] += 1

#         files = {
#             "face": ("face.jpg", image_bytes, "image/jpeg")
#         }
#         data = {
#             "unknownId": unknown_id,
#             "meanEmbedding":  json.dumps(centroid.tolist()),
#             "timestamp": str(timestamp),
#             "cameraCode": str(camera_code),
#         }

#         response = requests.post(
#             envConfig.NODE_UPDATE_UNKNOWN_URL,
#             files=files,
#             data=data,
#             headers={
#                 "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
#             }
#         )

#         return response


# unknown_embedding_store = UnknownEmbeddingStore(api_url=envConfig.NODE_LOAD_UNKNOWN_EMBEDDINGS_URL)




import numpy as np
import requests
import json
import time

from app.config.config import envConfig


class UnknownEmbeddingStore:

    def __init__(self, api_url, embedding_dim=512):
        self.api_url = api_url

        # in-memory state
        self.unknown_ids = []
        self.id_to_index = {}

        self.embeddings = np.empty((0, embedding_dim), dtype=np.float32)
        self.counts = []

        # recent duplicate protection
        self.recent_cache = {}  # unknown_id -> timestamp

        # config
        self.match_threshold = 0.50
        self.margin_threshold = 0.03
        self.duplicate_threshold = 0.60
        self.recent_window_sec = 10

    # ---------------------------------------------------
    # Utils
    # ---------------------------------------------------
    def _safe_normalize(self, emb):
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm

    def _post(self, url, files, data):
        try:
            return requests.post(
                url,
                files=files,
                data=data,
                headers={
                    "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
                },
                timeout=3
            )
        except Exception as e:
            print("[AI] API request failed:", e)
            return None

    # def _blend_embedding(self, old, new, count):
    #     updated = (old * count + new) / (count + 1)
    #     norm = np.linalg.norm(updated)
    #     if norm == 0:
    #         return old
    #     return updated / norm

    def _quality_to_weight(self, q: float) -> float:
        if q <= 0:
            return 0.0

        return min(round(q, 5), 10.0)


    def _blend_embedding(self, old, new, old_weight, new_weight):
        updated = (old * old_weight + new * new_weight) / (old_weight + new_weight)
        norm = np.linalg.norm(updated)
        if norm == 0:
            return old
        return updated / norm

    # ---------------------------------------------------
    # Load from Node
    # ---------------------------------------------------
    def load_unknown_embeddings(self):

        print("[AI] Loading unknown embeddings...")

        response = requests.get(
            self.api_url,
            headers={
                "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
            },
            timeout=5
        )

        data = response.json()

        if not data.get("success"):
            raise Exception("Failed to load unknown embeddings")

        vectors = []

        for u in data["data"]:
            emb = np.array(u["representativeEmbedding"], dtype=np.float32)
            emb = self._safe_normalize(emb)
            if emb is None:
                continue

            idx = len(self.unknown_ids)

            self.unknown_ids.append(u["id"])
            self.id_to_index[u["id"]] = idx
            self.counts.append(u.get("embeddingCount", 1))

            vectors.append(emb)

        if vectors:
            self.embeddings = np.stack(vectors)

        print(f"[AI] Loaded {len(self.unknown_ids)} unknown identities")

    # ---------------------------------------------------
    # Match unknown
    # ---------------------------------------------------
    def find_match(self, embedding):

        if self.embeddings.shape[0] == 0:
            return None

        embedding = self._safe_normalize(embedding)
        if embedding is None:
            return None

        scores = np.dot(self.embeddings, embedding)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < self.match_threshold:
            return None

        # margin check (VERY important)
        if len(scores) > 1:
            second_score = np.partition(scores, -2)[-2]
            if best_score - second_score < self.margin_threshold:
                return None

        return {
            "unknown_id": self.unknown_ids[best_idx],
            "similarity": float(best_score),
            "index": int(best_idx)
        }

    # ---------------------------------------------------
    # Duplicate protection (recent window)
    # ---------------------------------------------------
    def _check_recent_duplicate(self, embedding):
        now = time.time()

        for uid, ts in list(self.recent_cache.items()):
            if now - ts > self.recent_window_sec:
                del self.recent_cache[uid]
                continue

            idx = self.id_to_index.get(uid)
            if idx is None:
                continue

            sim = np.dot(self.embeddings[idx], embedding)
            if sim > self.duplicate_threshold:
                return uid

        return None

    # ---------------------------------------------------
    # Create new unknown
    # ---------------------------------------------------
    # def add_unknown(self, payload):

    #     try:
    #         centroid = np.array(payload["centroid_embedding"], dtype=np.float32)
    #         centroid = self._safe_normalize(centroid)

    #         if centroid is None:
    #             return None

    #         # Duplicate protection
    #         dup = self._check_recent_duplicate(centroid)
    #         if dup:
    #             return dup

    #         files = {}

    #         # send pose images
    #         poses = payload.get("poses", {})
    #         for pose_name, pose_data in poses.items():
    #             img_bytes = pose_data.get("image_bytes")
    #             if img_bytes:
    #                 files[f"face_{pose_name}"] = (
    #                     f"{pose_name}.jpg",
    #                     img_bytes,
    #                     "image/jpeg"
    #                 )

    #         # Remove image_bytes from payload
    #         clean_poses = {}

    #         for pose_name, pose_data in poses.items():
    #             pose_copy = dict(pose_data)
    #             pose_copy.pop("image_bytes", None)
    #             clean_poses[pose_name] = pose_copy

    #         data = {
    #             "payload": json.dumps({
    #                 "unknown_id": payload.get("unknown_id"),
    #                 "camera_code": payload.get("camera_code"),
    #                 "timestamp": payload.get("timestamp"),
    #                 "centroid_embedding": payload.get("centroid_embedding"),
    #                 "embedding_count": payload.get("embedding_count"),
    #                 "poses": clean_poses,
    #                 "builder_stats": payload.get("builder_stats", {})
    #             })
    #         }

    #         print("==== FULL PAYLOAD ====")
    #         print(json.dumps(payload, indent=2, default=str))

    #         response = self._post(
    #             # envConfig.NODE_CREATE_UNKNOWN_URL,
    #             "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
    #             files,
    #             data
    #         )

    #         if not response:
    #             return None

    #         res = response.json()

    #         if not res.get("success"):
    #             print("[AI] Node API error:", res)
    #             return None

    #         unknown_id = res["data"]["unknownId"]

    #         # -------------------------
    #         # Update local state
    #         # -------------------------
    #         idx = len(self.unknown_ids)

    #         self.unknown_ids.append(unknown_id)
    #         self.id_to_index[unknown_id] = idx

    #         self.embeddings = np.vstack([self.embeddings, centroid])
    #         self.counts.append(1)

    #         self.recent_cache[unknown_id] = time.time()

    #         return unknown_id

    #     except Exception as e:
    #         print("[AI] add_unknown failed:", e)
    #         return None
    
    def add_unknown(self, payload):
        try:
            # -------------------------
            # Normalize centroid
            # -------------------------
            centroid = np.array(payload["centroid_embedding"], dtype=np.float32)
            # centroid = self._safe_normalize(centroid)

            if centroid is None:
                return None

            # -------------------------
            # Duplicate protection
            # -------------------------
            dup = self._check_recent_duplicate(centroid)
            if dup:
                return dup

            # -------------------------
            # Extract poses
            # -------------------------
            poses = payload.get("poses")
            if not poses:
                print("[AI] No poses found in payload")
                return {
                    "error": "No poses in payload"
                }

            # -------------------------
            # Build clean poses (NO image_bytes)
            # -------------------------
            clean_poses = {}
            for pose_name, pose_data in poses.items():
                pose_copy = dict(pose_data)
                pose_copy.pop("image_bytes", None)  # remove binary
                clean_poses[pose_name] = pose_copy

            # -------------------------
            # Build JSON payload
            # -------------------------
            data = {
                "unknown_id": payload.get("unknown_id"),
                "camera_code": payload.get("camera_code"),
                "timestamp": str(payload.get("timestamp")),
                "centroid_embedding": json.dumps(centroid.tolist()),
                "embedding_count": str(payload.get("embedding_count")),
                "poses": json.dumps(clean_poses),
                "builder_stats": json.dumps(payload.get("builder_stats", {}))
            }

            # -------------------------
            # Build multipart request
            # -------------------------
            multipart_files = {}

            # ✅ Image parts
            for pose_name, pose_data in poses.items():
                img_bytes = pose_data.get("image_bytes")
                if img_bytes:
                    multipart_files[f"face_{pose_name}"] = (
                        f"{pose_name}.jpg",
                        img_bytes,
                        "image/jpeg"
                    )

            response = requests.post(
                # "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
                envConfig.NODE_CREATE_UNKNOWN_URL,
                files=multipart_files,
                data=data
            )

            if not response:
                return None

            # -------------------------
            # Safe response parsing
            # -------------------------
            try:
                res = response.json()
            except Exception:
                print("[DEBUG] Non-JSON response:", response.text)
                return "debug_unknown_id"  # for webhook testing

            if not res.get("success"):
                print("[AI] Node API error:", res)
                return None

            unknown_id = res["data"]["unknownId"]

            # -------------------------
            # Update local cache
            # -------------------------
            idx = len(self.unknown_ids)

            self.unknown_ids.append(unknown_id)
            self.id_to_index[unknown_id] = idx

            self.embeddings = np.vstack([self.embeddings, centroid])
            self.counts.append(1)

            self.recent_cache[unknown_id] = time.time()

            return unknown_id

        except Exception as e:
            print("[AI] add_unknown failed:", e)
            return None
    
    # ---------------------------------------------------
    # Update existing unknown
    # ---------------------------------------------------
    def update_unknown(self, unknown_id, centroid, timestamp, camera_code, image_bytes, poses, quality):

        idx = self.id_to_index.get(unknown_id)
        if idx is None:
            return None

        centroid = self._safe_normalize(centroid)
        if centroid is None:
            return None

        # blend embedding locally
        # use count as old weight, and NEW QUALITY as new weight
        new_quality = self._quality_to_weight(quality)  # fallback (we'll improve next)
        updated = self._blend_embedding(
            self.embeddings[idx],
            centroid,
            old_weight = self.counts[idx],
            new_weight = self.counts[idx] * new_quality
        )

        self.embeddings[idx] = updated
        self.counts[idx] += 1

        files = {
            "face": ("face.jpg", image_bytes, "image/jpeg")
        }

        data = {
            "unknownId": unknown_id,
            "meanEmbedding": json.dumps(updated.tolist()),
            "timestamp": str(timestamp),
            "cameraCode": camera_code,
            "poses": json.dumps(list(poses))
        }

        self._post(
            # envConfig.NODE_UPDATE_UNKNOWN_URL, 
            "https://webhook.site/a83e91fd-1397-453a-abf8-86d0c3b3e3b6",
            files, 
            data
        )

        self.recent_cache[unknown_id] = time.time()

        return unknown_id

    # ---------------------------------------------------
    # Unified API (USE THIS IN PIPELINE)
    # ---------------------------------------------------
    def register_or_update(self, centroid, image_bytes, timestamp, camera_code, embedding_count, poses):

        centroid = self._safe_normalize(centroid)
        if centroid is None:
            return None

        match = self.find_match(centroid)

        if match:
            return self.update_unknown(
                match["unknown_id"],
                centroid,
                timestamp,
                camera_code,
                image_bytes,
                poses
            )

        return self.add_unknown(
            centroid,
            image_bytes,
            timestamp,
            camera_code,
            embedding_count,
            poses
        )

    # ---------------------------------------------------
    # Debug / Metrics
    # ---------------------------------------------------
    def stats(self):
        return {
            "total_unknowns": len(self.unknown_ids),
            "avg_embeddings_per_identity": float(np.mean(self.counts)) if self.counts else 0
        }


unknown_embedding_store = UnknownEmbeddingStore(
    api_url=envConfig.NODE_LOAD_UNKNOWN_EMBEDDINGS_URL
)