import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from app.config.config import envConfig


class FaceLandmarkerEngine:
    def __init__(
        self,
        model_path: str,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        # thresholds (configurable)
        eye_blink_threshold: float = 0.4,
        smile_threshold: float = 0.6,
        mouth_open_threshold: float = 0.4,
        yaw_threshold: float = 20,
        pitch_threshold: float = 25,
        roll_threshold: float = 20,
        blur_threshold: float = envConfig.BLUER_THRESHOLD,
        min_face_size: int = envConfig.MIN_FACE_SIZE,
    ):
        self.eye_blink_threshold = eye_blink_threshold
        self.smile_threshold = smile_threshold
        self.mouth_open_threshold = mouth_open_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            num_faces=1,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    # -----------------------------
    # MAIN ENTRY
    # -----------------------------
    # def analyze(self, face_img: np.ndarray) -> Optional[Dict]:
    #     if face_img is None or face_img.size == 0:
    #         return None

    #     h, w = face_img.shape[:2]

    #     # Size check (early reject)
    #     if min(h, w) < self.min_face_size:
    #         return None

    #     # Blur check (early reject)
    #     if self._is_blurry(face_img):
    #         return None

    #     rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    #     mp_image = self._to_mp_image(rgb)
    #     #

    #     # mp_image = vision.Image(
    #     #     image_format=vision.ImageFormat.SRGB,
    #     #     data=rgb,
    #     # )

    #     result = self.landmarker.detect(mp_image)

    #     if not result.face_landmarks:
    #         return None

    #     return {
    #         "blendshapes": result.face_blendshapes[0] if result.face_blendshapes else None,
    #         "matrix": result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None,
    #     }


    def analyze(self, face_img: np.ndarray, debug=True) -> Optional[Dict]:
        if face_img is None or face_img.size == 0:
            if debug:
                print("[Landmarker] ❌ Empty face image")
            return None

        h, w = face_img.shape[:2]

        # Size check
        if min(h, w) < self.min_face_size:
            if debug:
                print(f"[Landmarker] ❌ Rejected: small face ({w}x{h}) < {self.min_face_size}")
            return None

        # Blur check
        blur_score = cv2.Laplacian(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            if debug:
                print(f"[Landmarker] ❌ Rejected: blurry (score={blur_score:.2f} < {self.blur_threshold})")
            return None

        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = self._to_mp_image(rgb)

        result = self.landmarker.detect(mp_image)

        # print(f"[Landmarker] Detection result: {result.face_landmarks}, Blendshapes: {result.face_blendshapes}, Matrix")

        if not result.face_landmarks:
            if debug:
                print("[Landmarker] ❌ No landmarks detected (likely bad pose / occlusion)")
            return None

        return {
            "blendshapes": result.face_blendshapes[0] if result.face_blendshapes else None,
            "matrix": result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None,
        }
    # -----------------------------
    # QUALITY GATE
    # -----------------------------
    def is_valid_face(self, analysis: Dict, debug=True) -> bool:
        if analysis is None:
            if debug:
                print("[Landmarker] ❌ analysis is None")
            return False

        blend = self._blendshape_dict(analysis.get("blendshapes"))
        matrix = analysis.get("matrix")

        if blend is None:
            if debug:
                print("[Landmarker] ❌ Missing blendshapes")
            return False

        if matrix is None:
            if debug:
                print("[Landmarker] ❌ Missing transformation matrix")
            return False

        # Eye check
        left = blend.get("eyeBlinkLeft", -1)
        right = blend.get("eyeBlinkRight", -1)

        if left >= self.eye_blink_threshold or right >= self.eye_blink_threshold:
            if debug:
                print(f"[Landmarker] ❌ Eyes closed: L={left:.2f}, R={right:.2f}")
            return False

        # Expression
        smile = blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
        mouth_open = blend.get("jawOpen", 0.0)

        if smile >= self.smile_threshold or mouth_open >= self.mouth_open_threshold:
            if debug:
                print(f"[Landmarker] ❌ Expression: smile={smile:.2f}, mouth_open={mouth_open:.2f}")
            return False

        # Pose
        pose = self._extract_pose(matrix)
        if pose is None:
            if debug:
                print("[Landmarker] ❌ Pose extraction failed")
            return False

        yaw, pitch, roll = pose

        if (
            abs(yaw) >= self.yaw_threshold or
            abs(pitch) >= self.pitch_threshold or
            abs(roll) >= self.roll_threshold
        ):
            if debug:
                print(f"[Landmarker] ❌ Pose rejected: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
            return False

        if debug:
            print(f"[Landmarker] ✅ Accepted: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")

        return True
    # -----------------------------
    # HELPERS
    # -----------------------------
    def _blendshape_dict(self, blendshapes):
        if not blendshapes:
            return None
        return {b.category_name: b.score for b in blendshapes}

    # ---------- Eye ----------
    def _eyes_open(self, blend):
        left = blend.get("eyeBlinkLeft")
        right = blend.get("eyeBlinkRight")

        if left is None or right is None:
            return False  # strict

        return (left < self.eye_blink_threshold) and (right < self.eye_blink_threshold)

    # ---------- Expression ----------
    def _neutral_expression(self, blend):
        smile_l = blend.get("mouthSmileLeft", 0.0)
        smile_r = blend.get("mouthSmileRight", 0.0)
        mouth_open = blend.get("jawOpen", 0.0)

        smile = smile_l + smile_r

        return (
            smile < self.smile_threshold and
            mouth_open < self.mouth_open_threshold
        )

    # ---------- Pose ----------
    def _is_frontal(self, matrix):
        try:
            mat = np.array(matrix)

            if mat.size != 16:
                return False

            mat = mat.reshape(4, 4)
            rmat = mat[:3, :3]

            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0

            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)

            return (
                abs(yaw) < self.yaw_threshold and
                abs(pitch) < self.pitch_threshold and
                abs(roll) < self.roll_threshold
            )

        except Exception:
            return False

    # ---------- Blur ----------
    def _is_blurry(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold
    
    # ---------- Pose extraction ----------
    def _extract_pose(self, matrix):
        try:
            mat = np.array(matrix)
            if mat.size != 16:
                return None

            mat = mat.reshape(4, 4)
            rmat = mat[:3, :3]

            sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0

            pitch = np.degrees(x)
            yaw = np.degrees(y)
            roll = np.degrees(z)

            return yaw, pitch, roll

        except Exception:
            return None

    # ---------- to mediapipe image ----------
    def _to_mp_image(self, img):
        return mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img
        )