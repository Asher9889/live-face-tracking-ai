import numpy as np
import requests

from app.config.config import envConfig

class UnknownEmbeddingStore:

    def __init__(self, api_url):
        self.api_url = api_url

        self.unknown_ids = []
        self.id_to_index = {}

        self.embeddings = np.empty((0, 512), dtype=np.float32)
        self.counts = []

    # ---------------------------------------------------
    # Load unknown identities from Node API on startup
    # ---------------------------------------------------
    def load_unknown_embeddings(self):

        print("[AI] Loading unknown embeddings...")

        response = requests.get(
            self.api_url,
            headers={
                "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
            }
        )

        data = response.json()

        if not data.get("success"):
            raise Exception("Failed to load unknown embeddings")

        unknowns = data["data"]

        vectors = []

        for u in unknowns:

            emb = np.array(u["representativeEmbedding"], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)

            idx = len(self.unknown_ids)

            self.unknown_ids.append(u["_id"])
            self.id_to_index[u["_id"]] = idx
            self.counts.append(u["embeddingCount"])

            vectors.append(emb)

        if vectors:
            self.embeddings = np.stack(vectors)

        print(f"[AI] Loaded {len(self.unknown_ids)} unknown identities")

    # ---------------------------------------------------
    # Search unknown identities
    # ---------------------------------------------------
    def find_match(self, embedding, threshold=0.6):

        if self.embeddings.shape[0] == 0:
            return None

        scores = np.dot(self.embeddings, embedding)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < threshold:
            return None

        return {
            "unknown_id": self.unknown_ids[best_idx],
            "similarity": float(best_score)
        }

    # ---------------------------------------------------
    # Create new unknown identity
    # ---------------------------------------------------
    def add_unknown(self, centroid_embedding, image_bytes, timestamp, camera_code):

        centroid_embedding = centroid_embedding / np.linalg.norm(centroid_embedding)

        files = {
            "image": ("face.jpg", image_bytes, "image/jpeg")
        }

        data = {
            "representativeEmbedding": centroid_embedding.tolist(),
            "firstSeen": timestamp,
            "lastSeen": timestamp,
            "cameraCode": camera_code
        }

        response = requests.post(
            envConfig.NODE_CREATE_UNKNOWN_URL,
            files=files,
            data=data,
            headers={
                "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
            }
        )

        data = response.json()

        if not data.get("success"):
            raise Exception("Failed to create unknown identity")

        unknown_id = data["data"]["unknownId"]

        idx = len(self.unknown_ids)

        self.unknown_ids.append(unknown_id)
        self.id_to_index[unknown_id] = idx

        self.embeddings = np.vstack([self.embeddings, centroid_embedding])
        self.counts.append(1)

        return unknown_id

    # ---------------------------------------------------
    # Update existing unknown identity
    # ---------------------------------------------------
    def update_unknown(self, unknown_id, new_embedding, timestamp):

        idx = self.id_to_index[unknown_id]

        centroid = self.embeddings[idx]
        count = self.counts[idx]

        updated = (centroid * count + new_embedding) / (count + 1)
        updated = updated / np.linalg.norm(updated)

        self.embeddings[idx] = updated
        self.counts[idx] += 1

        requests.patch(
            f"{envConfig.NODE_UPDATE_UNKNOWN_URL}/{unknown_id}",
            json={
                "embedding": new_embedding.tolist(),
                "timestamp": timestamp
            },
            headers={
                "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
            }
        )

        return updated

unknown_embedding_store = UnknownEmbeddingStore(api_url=envConfig.NODE_LOAD_UNKNOWN_EMBEDDINGS_URL)