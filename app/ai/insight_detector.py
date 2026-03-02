from insightface.app import FaceAnalysis
import numpy as np

class InsightFaceEngine:

    MIN_FACE_SIZE = 30
    MIN_SCORE = 0.5

    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=det_size) # ctx_id=0 means use first GPU 
        print("[AI] InsightFace Engine Ready (GPU Enabled)")

    def detect_and_generate_embedding(self, frame: np.ndarray, offset=(0, 0)):
        """
        frame  : ROI or full frame
        offset : (x_offset, y_offset) if frame is cropped ROI
        """

        faces = self.app.get(frame)
        """
        Resize for Detection 640 * 640
        Detect Face
        Landmark Detection
        Alignment
        Recognition Resize 112 * 112 * 3. this size arcface expects.
        """

        if not faces:
            return []

        results = []

        x_offset, y_offset = offset

        for face in faces:
            score = float(face.det_score)
            if score < self.MIN_SCORE:
                continue

            bbox = face.bbox.astype(np.int32)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if min(width, height) < self.MIN_FACE_SIZE:
                continue

            # Convert to global coordinates (important if ROI used)
            global_bbox = np.array([
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset
            ])

            yaw, pitch, roll = face.pose if face.pose is not None else (0, 0, 0)

            results.append({
                "bbox": global_bbox,
                "score": score,
                "landmarks": face.kps.astype(np.int32),
                "embedding": face.embedding,   # 512-d vector
                "pose": (yaw, pitch, roll),
                "age": int(face.age) if hasattr(face, "age") else None,
                "gender": int(face.gender) if hasattr(face, "gender") else None
            })

        return results
    
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        """

        return float(np.dot(a, b))

    def is_good_face(face):

        score = face["score"]
        yaw, pitch, roll = face["pose"]

        if score < 0.35:
            return False

        if abs(yaw) > 35:
            return False

        if abs(pitch) > 25: 
            return False

        if abs(roll) > 20:
            return False

        return True    
