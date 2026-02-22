from insightface.app import FaceAnalysis
import numpy as np


class InsightFaceDetector:

    MIN_FACE_SIZE = 40
    MIN_SCORE = 0.5
    MAX_YAW = 45
    MAX_PITCH = 45

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=(640, 640))

        print("[AI] InsightFace detector ready")

    def detect(self, frame: np.ndarray):
        faces = self.app.get(frame)

        if faces is None:
            return []


        results = []

        for face in faces:
            bbox = face.bbox.astype(int)
            score = float(face.det_score)

            # Compute size
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = min(width, height)

            # Pose filtering   
            yaw, pitch, roll = face.pose if face.pose is not None else (0,0,0)
            age = int(face.age) if hasattr(face, "age") else None,
            gender = int(face.gender) if hasattr(face, "gender") else None,

            results.append({
                "bbox": bbox,
                "score": score,
                "landmarks": face.kps.astype(int),
                "pose": (yaw, pitch, roll),
                "size": size,
                "area": width * height,
                "age": age,
                "gender": gender
                })

            return results
