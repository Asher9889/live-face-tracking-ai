from dataclasses import dataclass

import numpy as np
import requests

from app.config.config import envConfig


@dataclass
class EmbeddingStoreData:
    employee_ids: list
    employee_names: list
    embeddings: np.ndarray | None


class EmbeddingStore:


    def __init__(self, api_url: str):
        self.api_url = api_url
        self._store = EmbeddingStoreData(
            employee_ids=[],
            employee_names=[],
            embeddings=None,
        )

    def load_embeddings(self):
        print("[AI] Loading employee embeddings...")

        response = requests.get(self.api_url, headers={
            "Authorization": f"Bearer {envConfig.TOKEN_TO_ACCESS_NODE_API}"
        }, timeout=5)

        response.raise_for_status()

        data = response.json()
        if not data.get("success"):
            raise Exception("Failed to load embeddings")

        employees = data["data"]


        if len(employees) < 1:
            self._store = EmbeddingStoreData(
                employee_ids=[],
                employee_names=[],
                embeddings=None,
            )

            print("[AI] ⚠️ No employees found → running in UNKNOWN-ONLY mode")
            return

        employee_ids = []
        employee_names = []
        vectors = []

        for emp in employees:

            mean_embedding = np.array(emp["meanEmbedding"], dtype=np.float32)

            # normalize (safety)
            mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            employee_ids.append(emp["id"])
            employee_names.append(emp["name"])
            vectors.append(mean_embedding)

        self._store = EmbeddingStoreData(
            employee_ids=employee_ids,
            employee_names=employee_names,
            embeddings=np.stack(vectors),
        )

        print(f"[AI] Loaded {len(employee_ids)} employee embeddings")

    def find_match(self, embedding, threshold=0.45):
        """
        Returns best matching employee
        """

        store = self._store

        if store.embeddings is None:
            return None

        # ensure normalized
        # embedding = embedding / np.linalg.norm(embedding) # no need to normalize already getting normalized values

        # cosine similarity
        scores = np.dot(store.embeddings, embedding)

        best_idx = np.argmax(scores) # returns index of the largest value of similaity score
        best_score = scores[best_idx]


        if best_score < threshold:
            print(f"[AI] Threshold not met: {best_score} < {threshold}")
            return None
        print(f"[AI] Best match score above threshold: {best_score}")
        return {
            "employee_id": store.employee_ids[best_idx],
            "name": store.employee_names[best_idx],
            "similarity": float(best_score)
        }



# embedding_store = EmbeddingStore(api_url="http://localhost:8000/register-face")
embedding_store = EmbeddingStore(api_url=envConfig.NODE_LOAD_EMBEDDINGS_URL)
