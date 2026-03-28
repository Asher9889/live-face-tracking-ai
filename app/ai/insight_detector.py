from insightface.app import FaceAnalysis
import numpy as np
import cv2
from datetime import datetime

from app.config.config import envConfig

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class InsightFaceEngine:

    MIN_FACE_SIZE = envConfig.MIN_FACE_SIZE
    MIN_SCORE = 0.55

    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"]
        )

        self.app.prepare(ctx_id=0, det_size=det_size) # ctx_id=0 means use first GPU 
        print("[AI] InsightFace Engine Ready (GPU Enabled)")

    def detect_and_generate_embedding(self, frame: np.ndarray, offset=(0, 0), camera_code=None):
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
                # print(f"[{now}][Detect_And_Generate-Embedding] [ Camera {camera_code}] Skipped face due to low face score: {score}. Need at least {self.MIN_SCORE}.")
                continue

            bbox = face.bbox.astype(np.int32)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            embedding = face.embedding.astype(np.float32)
            # normalize once
            embedding /= np.linalg.norm(embedding)

            # print(f"[{now}][Detect_Face_Size][Camera {camera_code}] Face size: {width}x{height}, score: {score:.2f}, pose: {face.pose}, gender:{face.gender}")
            # continue 

            if min(width, height) < self.MIN_FACE_SIZE: # skip very small faces
                # print(f"[Detect_And_Generate-Embedding][Camera {camera_code}] Skipped face due to small size: {width}x{height}. Need at least {self.MIN_FACE_SIZE}.")
                continue

            # Convert to global coordinates (important if ROI used)
            global_bbox = np.array([
                bbox[0] + x_offset,
                bbox[1] + y_offset,
                bbox[2] + x_offset,
                bbox[3] + y_offset
            ])

            if face.pose is None:
                print(f"[Detect_And_Generate-Embedding][Camera {camera_code}] Face pose not available.")
                continue

            yaw, pitch, roll = face.pose
            
            print(f"[{now}][Detect_And_Generate-Embedding][Camera {camera_code}] Detected face with score: {score:.2f}, size: {width}x{height}, pose: (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}), gender: {face.gender}")

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

        if abs(yaw) > 25:
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

        if score < 0.60:
            return False
        

        if abs(yaw) > 20:
            print(f"[Unknown_Filter] Rejected face due to low yaw: {yaw}, allowing only less than 20")
            return False

        if abs(pitch) > 25: 
            print(f"[Unknown_Filter] Rejected face due to high pitch: {pitch}, allowing only less than 25")
            return False

        if abs(roll) > 20:
            print(f"[Unknown_Filter] Rejected face due to high roll: {roll}, allowing only less than 20")
            return False

        h, w = face_img.shape[:2]    
        if h < 30 or w < 30:
            print(f"[Unknown_Filter] Rejected face due to small size: {w}x{h}, allowing only larger than 30x30")
            return False
        

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < envConfig.BLUR_THRESHOLD:
            print(f"[Unknown_Filter] Rejected face due to blur: {blur_score}, allowing only greater than {envConfig.BLUR_THRESHOLD}")
            return False

        # brightness
        brightness = np.mean(gray) 
        if brightness < 40 or brightness > 220:
            print(f"[Unknown_Filter] Rejected face due to brightness: {brightness}, allowing only greater than 40")
            return False

        # print(f"[Unknown_Filter] Face passed pose and size checks: yaw={yaw}, pitch={pitch}, roll={roll}, size={w}x{h}, blur={blur_score}, brightness={brightness}")
        return True

    @staticmethod
    def compute_face_quality(face, face_img, camera_code=None):

        score = face["score"]
        yaw, pitch, roll = face["pose"]

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)

        h, w = face_img.shape[:2]
        size = min(h, w)

        # HARD REJECTION
        if abs(yaw) > 20:
            return -1
        
        if blur < envConfig.BLUR_THRESHOLD:
            print(f"[Camera {camera_code}]rejected frame due to low blur=======", blur)
            return -1  # reject

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.mean(np.sqrt(gx**2 + gy**2))
        if sobel < 20: # last is 40 getting 19 above
            print(f"[Camera {camera_code}]rejected frame due to low sobel score=======", sobel)
            return -1  # reject


        if size < 30:
            print(f"[Camera {camera_code}]rejected frame due to small size=======", size)
            return -1  # reject

        if brightness < 40 or brightness > 220:
            print(f"[Camera {camera_code}]rejected frame due to brightness=======", brightness)
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
