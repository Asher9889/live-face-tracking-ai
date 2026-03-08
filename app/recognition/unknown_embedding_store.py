class UnknownEmbeddingStore:

    def __init__(self):
        self.unknown_ids = []
        self.embeddings = []

    def add_unknown(self, centroid_embedding):

        centroid_embedding = centroid_embedding / np.linalg.norm(centroid_embedding)

        unknown_id = f"unknown_{len(self.unknown_ids)+1}"

        self.unknown_ids.append(unknown_id)
        self.embeddings.append(centroid_embedding)

        return unknown_id

    def find_match(self, embedding, threshold=0.6):

        if len(self.embeddings) == 0:
            return None

        matrix = np.stack(self.embeddings)

        scores = np.dot(matrix, embedding)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score < threshold:
            return None

        return {
            "unknown_id": self.unknown_ids[best_idx],
            "similarity": float(best_score)
        }