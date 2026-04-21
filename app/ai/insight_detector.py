from insightface.app import FaceAnalysis
import numpy as np
import cv2
from datetime import datetime

from app.config.config import envConfig

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class InsightFaceEngine:
    """
    det_score < 0.4  → very uncertain detection, likely false positive
    det_score 0.4-0.6 → partial face, occluded
    det_score > 0.6  → clean detection 
    det_score > 0.85 → near perfect detection
    """

    MIN_FACE_SIZE = envConfig.MIN_FACE_SIZE
    MIN_SCORE = 0.60

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
            # if score < self.MIN_SCORE:
            #     print(f"[{now}][Detect_And_Generate-Embedding] [ Camera {camera_code}] Skipped face due to low face score: {score}. Need at least {self.MIN_SCORE}.")
            #     continue

            bbox = face.bbox.astype(np.int32)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            embedding = face.embedding.astype(np.float32)
            # normalize once
            embedding /= np.linalg.norm(embedding)

            # print(f"[{now}][Detect_Face_Size][Camera {camera_code}] Face size: {width}x{height}, score: {score:.2f}, pose: {face.pose}, gender:{face.gender}")
            # continue 

            # if min(width, height) < self.MIN_FACE_SIZE: # skip very small faces
            #     # print(f"[Detect_And_Generate-Embedding][Camera {camera_code}] Skipped face due to small size: {width}x{height}. Need at least {self.MIN_FACE_SIZE}.")
            #     continue

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
            
            # print(f"[{now}][Detect_And_Generate-Embedding][Camera {camera_code}] Detected face with score: {score:.2f}, size: {width}x{height}, pose: (yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}), gender: {face.gender}")

            """
            print(face.bbox)        # [x1, y1, x2, y2]
            print(face.kps)         # 5 keypoints (x,y) pairs
            print(face.embedding)   # 512-D ArcFace vector ← USE THIS
            print(face.age)         # estimated age
            print(face.gender)      # 0=female, 1=male
            print(face.pose)        # yaw, pitch, roll ← filter bad angles
            """
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

    # def compute_face_quality(face, face_img, camera_code=None):

    #     score = face["score"]
    #     yaw, pitch, roll = face["pose"]

    #     yaw = abs(yaw)
    #     pitch = abs(pitch)
    #     roll = abs(roll)

    #     gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    #     blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    #     brightness = np.mean(gray)

    #     h, w = face_img.shape[:2]
    #     size = min(h, w)

    #     # HARD REJECTION
    #     if yaw > 20:
    #         return -1
        
    #     if blur < envConfig.BLUR_THRESHOLD:
    #         print(f"[Camera {camera_code}]rejected frame due to low blur=======", blur)
    #         return -1  # reject

    #     gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    #     gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    #     sobel = np.mean(np.sqrt(gx**2 + gy**2))
    #     if sobel < 20: # last is 40 getting 19 above
    #         print(f"[Camera {camera_code}]rejected frame due to low sobel score=======", sobel)
    #         return -1  # reject


    #     if size < 30:
    #         print(f"[Camera {camera_code}]rejected frame due to small size=======", size)
    #         return -1  # reject

    #     if brightness < 40 or brightness > 220:
    #         print(f"[Camera {camera_code}]rejected frame due to brightness=======", brightness)
    #         return -1  # reject

    #     # NORMALIZATION (important)
    #     blur_norm = min(blur / 300, 1.0)
    #     brightness_norm = brightness / 255
    #     size_norm = min(size / 200, 1.0)

    #     pose_penalty = (yaw/25 + pitch/20 + roll/25) / 3
    #     frontal_bonus = 1 - (yaw/20)

    #     quality = (
    #         score * 0.4
    #         + blur_norm * 0.2
    #         + brightness_norm * 0.1
    #         + size_norm * 0.2
    #         - pose_penalty * 0.3
    #     )

    #     return quality

    def compute_face_quality(self, face, face_img, analysis=None):

        # ----------------------------
        # 1. DETECTION SCORE
        # ----------------------------
        score = float(face.get("score", 0.0))
        score_norm = min(score, 1.0)

        # ----------------------------
        # 2. IMAGE METRICS
        # ----------------------------
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < envConfig.BLUR_THRESHOLD:
            return -1  # only hard reject

        blur_norm = min(blur / 300, 1.0)

        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        brightness_norm = np.exp(-((brightness - 128) ** 2) / (2 * 50 ** 2))
        contrast_norm = min(contrast / 50, 1.0)

        # ----------------------------
        # 3. SIZE
        # ----------------------------
        h, w = face_img.shape[:2]
        size = min(h, w)

        if size < 30:
            return -1

        size_norm = min(size / 160, 1.0)

        # ----------------------------
        # 4. POSE (NO FAKE VALUES)
        # ----------------------------
        pose_penalty = 0.0
        frontal_bonus = 0.0
        pose_conf = 0.0

        yaw = pitch = roll = None

        # FaceMesh (best)
        if analysis and analysis.get("valid") and analysis.get("yaw") is not None:
            yaw = abs(analysis["yaw"])
            pitch = abs(analysis["pitch"])
            roll = abs(analysis["roll"])
            pose_conf = 1.0

        # InsightFace fallback
        elif face.get("pose") is not None:
            yaw, pitch, roll = face["pose"]
            yaw, pitch, roll = abs(yaw), abs(pitch), abs(roll)
            pose_conf = 0.7

        # If pose exists → compute
        if yaw is not None:

            if yaw > 45:
                return -1  # only extreme reject

            yaw_penalty = (yaw / 30) ** 2
            pitch_penalty = (pitch / 35) ** 2
            roll_penalty = (roll / 35) ** 2

            pose_penalty = (
                yaw_penalty * 0.6 +
                pitch_penalty * 0.25 +
                roll_penalty * 0.15
            )

            frontal_bonus = np.exp(-(yaw ** 2) / (2 * 15 ** 2))

        else:
            # ❗ NO POSE → penalize uncertainty
            pose_penalty = 0.4
            frontal_bonus = 0.1
            pose_conf = 0.3

        # ----------------------------
        # 5. FACEMESH SIGNALS (NO ASSUMPTIONS)
        # ----------------------------
        eye_score = None
        expression_penalty = None
        signal_conf = 0.0

        if analysis and analysis.get("valid"):

            if analysis.get("eye_score") is not None:
                eye_score = analysis["eye_score"]
                signal_conf += 0.5

            if analysis.get("expression_score") is not None:
                expression_penalty = analysis["expression_score"]
                signal_conf += 0.5

        signal_conf = min(signal_conf, 1.0)

        # Missing signals → penalize (not assume good)
        if eye_score is None:
            eye_score = 0.5  # neutral-mid (not optimistic)

        if expression_penalty is None:
            expression_penalty = 0.2  # slight penalty

        # ----------------------------
        # 6. FINAL QUALITY
        # ----------------------------
        quality = (
            score_norm * 0.2
            + blur_norm * 0.2
            + contrast_norm * 0.1
            + brightness_norm * 0.1
            + size_norm * 0.1
            + frontal_bonus * 0.15
            + eye_score * 0.1
            - expression_penalty * 0.1
            - pose_penalty * 0.25
        )

        # ----------------------------
        # 7. CONFIDENCE (UNCERTAINTY-AWARE)
        # ----------------------------
        confidence = 0.5 * pose_conf + 0.5 * signal_conf

        # ensure it never collapses completely
        confidence = max(0.3, confidence)

        quality *= confidence

        return float(quality)