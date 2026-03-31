import numpy as np


class UnknownIdentityManager:
    def __init__(self, store):
        self.store = store
        self.track_to_unknown = {}

    def create_or_update(self, person_id, buffer, cam_code, face_img):
        """
        Handles:
        - early creation
        - continuous update
        """

        centroid = self._compute_centroid(buffer)

        if centroid is None:
            return None

        # ----------------------------
        # CREATE
        # ----------------------------
        if person_id not in self.track_to_unknown:
            unknown_id = self.store.add_embedding(
                centroid,
                cam_code,
                face_img
            )

            self.track_to_unknown[person_id] = unknown_id
            return unknown_id

        # ----------------------------
        # UPDATE
        # ----------------------------
        unknown_id = self.track_to_unknown[person_id]

        for item in buffer:
            self.store.update_embedding(
                unknown_id,
                item["embedding"],
                item["quality"],
                face_img
            )

        return unknown_id

    def _compute_centroid(self, buffer):
        if not buffer:
            return None

        embeddings = np.array([x["embedding"] for x in buffer])
        weights = np.array([x["quality"] for x in buffer])

        if weights.sum() == 0:
            weights = np.ones_like(weights)

        centroid = np.average(embeddings, axis=0, weights=weights)
        centroid /= np.linalg.norm(centroid)

        return centroid