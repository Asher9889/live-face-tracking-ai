import numpy as np
import requests

from app.config.config import envConfig


class EmbeddingStore:


    def __init__(self, api_url: str):
        self.api_url = api_url
        self.employee_ids = []
        self.employee_names = []
        self.embeddings = None  # numpy matrix (N x 512)

    def load_embeddings(self):
        print("[AI] Loading employee embeddings...")

        response = requests.get(self.api_url, headers={
            "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
        })

        data = response.json()
        if not data.get("success"):
            raise Exception("Failed to load embeddings")

        employees = data["data"]

        vectors = []

        for emp in employees:

            mean_embedding = np.array(emp["meanEmbedding"], dtype=np.float32)

            # normalize (safety)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            self.employee_ids.append(emp["id"])
            self.employee_names.append(emp["name"])
            vectors.append(mean_embedding)

        self.embeddings = np.stack(vectors)

        print(f"[AI] Loaded {len(self.employee_ids)} employee embeddings")

    def find_match(self, embedding, threshold=0.45):
        """
        Returns best matching employee
        """

        if self.embeddings is None:
            return None

        # ensure normalized
        embedding = embedding / np.linalg.norm(embedding)

        # cosine similarity
        scores = np.dot(self.embeddings, embedding)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < threshold:
            return None

        return {
            "employee_id": self.employee_ids[best_idx],
            "name": self.employee_names[best_idx],
            "similarity": float(best_score)
        }



# embedding_store = EmbeddingStore(api_url="http://localhost:8000/register-face")
embedding_store = EmbeddingStore(api_url=envConfig.NODE_LOAD_EMBEDDINGS_URL)
