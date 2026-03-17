from insightface.app import FaceAnalysis
import numpy as np
import cv2
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

            embedding = face.embedding.astype(np.float32)
            # normalize once
            embedding /= np.linalg.norm(embedding)

            if min(width, height) < self.MIN_FACE_SIZE: # skip very small faces
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
                "embedding": embedding,   # 512-d vector
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

    @staticmethod
    def is_good_face(face):

        score = face["score"]
        yaw, pitch, roll = face["pose"]

        # 1. Detection confidence (tighten this)
        if score < 0.5:
            return False

        if abs(yaw) > 30:
            return False

        if abs(pitch) > 25: 
            return False

        if abs(roll) > 20:
            return False

        return True
        
    @staticmethod
    def is_good_face_for_unknown(face, face_img):
        
        score = face["score"]
        yaw, pitch, roll = face["pose"]

        if score < 0.45:
            return False

        if abs(yaw) > 35:
            return False

        if abs(pitch) > 25: 
            return False

        if abs(roll) > 20:
            return False

        h, w = face_img.shape[:2]    
        if h < 80 or w < 80:
            return False
        
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            return False

        # brightness
        brightness = np.mean(gray)
        if brightness < 40:
            return False

        return True
    
    @staticmethod
    def compute_face_quality(face, face_img):

        score = face["score"]
        yaw, pitch, roll = face["pose"]

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        h, w = face_img.shape[:2]
        size = min(h, w)

        # HARD REJECTION 
        if blur < 60:
            print("rejected frame due to low blur=======")
            return -1  # reject

        if size < 40:
            print("rejected frame due to small size=======")
            return -1  # reject

        if brightness < 40 or brightness > 220:
            print("rejected frame due to brightness=======")
            return -1  # reject

        # NORMALIZATION (important)
        blur_norm = min(blur / 300, 1.0)
        brightness_norm = brightness / 255
        size_norm = min(size / 200, 1.0)

        pose_penalty = (abs(yaw)/25 + abs(pitch)/20 + abs(roll)/25) / 3

        quality = (
            score * 0.4
            + blur_norm * 0.2
            + brightness_norm * 0.1
            + size_norm * 0.2
            - pose_penalty * 0.3
    )

        return quality
