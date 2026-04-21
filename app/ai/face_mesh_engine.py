# import cv2
# import mediapipe as mp
# import numpy as np
# from typing import Optional, Dict

# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# from mediapipe.tasks.python.vision import (
#     FaceLandmarker,
#     FaceLandmarkerOptions,
#     RunningMode,
# )
# from app.config.config import envConfig


# # ---------------------------------------------------------------------------
# # MediaPipe FaceMesh landmark indices (478-point model)
# # ---------------------------------------------------------------------------
# _LM = {
#     "left_eye_outer":     33,
#     "left_eye_inner":     133,
#     "left_eye_upper":     159,
#     "left_eye_lower":     145,
#     "right_eye_outer":    263,
#     "right_eye_inner":    362,
#     "right_eye_upper":    386,
#     "right_eye_lower":    374,
#     "nose_tip":           1,
#     "mouth_left":         61,
#     "mouth_right":        291,
#     "mouth_upper":        13,
#     "mouth_lower":        14,
#     "chin":               152,
#     "left_cheek":         234,
#     "right_cheek":        454,
#     "left_brow_outer":    70,
#     "right_brow_outer":   300,
#     # Iris landmarks (478-point model only)
#     # Left iris:  center=468, edge points=469,470,471,472
#     # Right iris: center=473, edge points=474,475,476,477
#     "left_iris_center":   468,
#     "left_iris_top":      469,
#     "left_iris_right":    470,
#     "left_iris_bottom":   471,
#     "left_iris_left":     472,
#     "right_iris_center":  473,
#     "right_iris_top":     474,
#     "right_iris_right":   475,
#     "right_iris_bottom":  476,
#     "right_iris_left":    477,
# }

# _EYE_PATCH_PAD = 1.8   # wider crop to fully capture iris + surround


# class FaceLandmarkerEngine:
#     def __init__(
#         self,
#         model_path: str,
#         min_face_presence_confidence: float = 0.5,
#         min_tracking_confidence: float = 0.5,
#         # ---- expression / blink thresholds ----
#         eye_blink_threshold: float = 0.4,
#         smile_threshold: float = 0.6,
#         mouth_open_threshold: float = 0.4,
#         # ---- pose thresholds (degrees) ----
#         yaw_threshold: float = 30,
#         pitch_threshold: float = 35,
#         roll_threshold: float = 25,
#         # ---- image quality ----
#         blur_threshold: float = envConfig.BLUR_THRESHOLD,
#         min_face_size: int = envConfig.MIN_FACE_SIZE,
#         # ---- geometric thresholds ----
#         min_eye_distance_ratio: float = 0.25,
#         min_face_height_ratio: float = 0.30,
#         min_face_width_ratio: float = 0.25,
#         max_lateral_asymmetry: float = 0.20,
#         min_ear: float = 0.10,
#         min_mouth_width_ratio: float = 0.60,
#         # ------------------------------------------------------------------
#         # IRIS-BASED OCCLUSION THRESHOLDS (replaces broken skin detector)
#         # ------------------------------------------------------------------
#         # Iris radius is computed from the 5 iris landmarks.
#         # Minimum iris radius in pixels. When hand covers eye, MediaPipe
#         # projects a tiny/degenerate iris → radius collapses.
#         # For a ~250px face crop, real iris radius ≈ 8-18px.
#         # Tune: print tex_left_iris_radius_px on your camera.
#         min_iris_radius_px: float = 1.5,
#         # Maximum asymmetry ratio between left/right iris radii.
#         # A hand over one eye degenerates that iris → small radius vs normal other.
#         # ratio = min(left_r, right_r) / max(left_r, right_r)
#         # Real symmetric face: ratio > 0.6.  One eye occluded: ratio < 0.4.
#         min_iris_radius_ratio: float = 0.45,
#         # Iris center darkness: the iris center pixel should be DARK (pupil).
#         # Mean grayscale value at a small disk around iris center.
#         # Real iris center ≈ 30-80 (dark).
#         # Hand skin over it ≈ 100-200 (medium/bright).
#         # If the iris center region is brighter than this, something is in the way.
#         max_iris_center_brightness: float = 140.0,
#         # Maximum brightness ASYMMETRY between the two iris centers.
#         # If one iris center is much brighter than the other → that eye is occluded.
#         # Absolute pixel difference.
#         max_iris_brightness_asymmetry: float = 60.0,
#     ):
#         self.eye_blink_threshold = eye_blink_threshold
#         self.smile_threshold = smile_threshold
#         self.mouth_open_threshold = mouth_open_threshold
#         self.yaw_threshold = yaw_threshold
#         self.pitch_threshold = pitch_threshold
#         self.roll_threshold = roll_threshold
#         self.blur_threshold = blur_threshold
#         self.min_face_size = min_face_size
#         self.min_eye_distance_ratio = min_eye_distance_ratio
#         self.min_face_height_ratio = min_face_height_ratio
#         self.min_face_width_ratio = min_face_width_ratio
#         self.max_lateral_asymmetry = max_lateral_asymmetry
#         self.min_ear = min_ear
#         self.min_mouth_width_ratio = min_mouth_width_ratio

#         # iris params
#         self.min_iris_radius_px = min_iris_radius_px
#         self.min_iris_radius_ratio = min_iris_radius_ratio
#         self.max_iris_center_brightness = max_iris_center_brightness
#         self.max_iris_brightness_asymmetry = max_iris_brightness_asymmetry

#         options = FaceLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path=model_path),
#             running_mode=RunningMode.IMAGE,
#             output_face_blendshapes=True,
#             output_facial_transformation_matrixes=True,
#             min_face_presence_confidence=min_face_presence_confidence,
#             min_tracking_confidence=min_tracking_confidence,
#             num_faces=1,
#         )
#         self.landmarker = FaceLandmarker.create_from_options(options)

#     # =========================================================================
#     # PUBLIC: analyze()
#     # =========================================================================

#     def analyze(self, face_img: np.ndarray, debug: bool = False) -> Dict:
#         if face_img is None or face_img.size == 0:
#             return {"valid": False, "reason": "empty"}

#         h, w = face_img.shape[:2]
#         size = min(h, w)
#         aspect_ratio = w / (h + 1e-6)

#         # ------------------------------------------------------------------
#         # 1. IMAGE QUALITY
#         # ------------------------------------------------------------------
#         gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#         blur = cv2.Laplacian(gray, cv2.CV_64F).var()
#         brightness = float(np.mean(gray))
#         contrast = float(np.std(gray))

#         # ------------------------------------------------------------------
#         # 2. MEDIAPIPE INFERENCE
#         # ------------------------------------------------------------------
#         rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#         mp_image = self._to_mp_image(rgb)
#         result = self.landmarker.detect(mp_image)

#         if not result.face_landmarks:
#             return {
#                 "valid": False,
#                 "reason": "no_landmarks",
#                 "blur": blur,
#                 "brightness": brightness,
#                 "contrast": contrast,
#                 "size": size,
#             }

#         # ------------------------------------------------------------------
#         # 3. BLENDSHAPES
#         # ------------------------------------------------------------------
#         blend_dict = None
#         if result.face_blendshapes:
#             blend_dict = {
#                 b.category_name: float(b.score)
#                 for b in result.face_blendshapes[0]
#             }

#         # ------------------------------------------------------------------
#         # 4. POSE
#         # ------------------------------------------------------------------
#         matrix = (
#             result.facial_transformation_matrixes[0]
#             if result.facial_transformation_matrixes
#             else None
#         )
#         yaw = pitch = roll = None
#         if matrix is not None:
#             pose = self._extract_pose(matrix)
#             if pose:
#                 yaw, pitch, roll = pose

#         # ------------------------------------------------------------------
#         # 5. BLENDSHAPE-DERIVED SIGNALS
#         # ------------------------------------------------------------------
#         eye_score = eye_balance = None
#         if blend_dict:
#             lb = blend_dict.get("eyeBlinkLeft", 0.0)
#             rb = blend_dict.get("eyeBlinkRight", 0.0)
#             eye_score = 1.0 - max(lb, rb)
#             eye_balance = abs(lb - rb)

#         expression_score = 0.0
#         if blend_dict:
#             smile = blend_dict.get("mouthSmileLeft", 0.0) + blend_dict.get("mouthSmileRight", 0.0)
#             mouth_open = blend_dict.get("jawOpen", 0.0)
#             expression_score = max(smile, mouth_open)

#         occlusion_score = 0.0
#         if blend_dict:
#             occlusion_score = max(
#                 blend_dict.get("cheekPuff", 0.0),
#                 blend_dict.get("eyeSquintLeft", 0.0),
#                 blend_dict.get("eyeSquintRight", 0.0),
#             )

#         frontal_score = None
#         if yaw is not None and pitch is not None:
#             frontal_score = max(0.0, 1.0 - (abs(yaw) / 40.0 + abs(pitch) / 30.0) / 2.0)

#         pose_magnitude = None
#         if yaw is not None and pitch is not None and roll is not None:
#             pose_magnitude = float(np.sqrt(yaw**2 + pitch**2 + roll**2))

#         upper_face_visible = lower_face_visible = False
#         face_completeness = mouth_occluded_score = 0.0
#         eye_visibility_score = None

#         if blend_dict:
#             eye_visibility_score = 1.0 - max(
#                 blend_dict.get("eyeBlinkLeft", 0.0),
#                 blend_dict.get("eyeBlinkRight", 0.0),
#                 blend_dict.get("eyeSquintLeft", 0.0),
#                 blend_dict.get("eyeSquintRight", 0.0),
#             )
#             upper_face_visible = (
#                 eye_visibility_score > 0.4
#                 and eye_balance is not None
#                 and eye_balance < 0.4
#             )
#             jaw = blend_dict.get("jawOpen", 0.0)
#             mouth_press = (
#                 blend_dict.get("mouthPressLeft", 0.0)
#                 + blend_dict.get("mouthPressRight", 0.0)
#             )
#             mouth_shrug = blend_dict.get("mouthShrugLower", 0.0)
#             lower_face_visible = jaw < 0.4 and mouth_press < 0.4 and mouth_shrug < 0.3
#             mouth_occluded_score = max(blend_dict.get("cheekPuff", 0.0), mouth_press)
#             face_completeness = (int(upper_face_visible) + int(lower_face_visible)) / 2.0

#         # ------------------------------------------------------------------
#         # 6. GEOMETRIC SIGNALS
#         # ------------------------------------------------------------------
#         lm_list = result.face_landmarks[0]
#         geo = self._extract_geometric_signals(lm_list, img_w=w, img_h=h)

#         # ------------------------------------------------------------------
#         # 7. IRIS SIGNALS  (replaces broken skin detector)
#         # ------------------------------------------------------------------
#         iris = self._extract_iris_signals(face_img, lm_list, img_w=w, img_h=h)

#         return {
#             "valid": True,
#             "blur": blur,
#             "brightness": brightness,
#             "contrast": contrast,
#             "size": size,
#             "aspect_ratio": aspect_ratio,
#             "yaw": yaw,
#             "pitch": pitch,
#             "roll": roll,
#             "pose_magnitude": pose_magnitude,
#             "eye_score": eye_score,
#             "eye_balance": eye_balance,
#             "expression_score": expression_score,
#             "frontal_score": frontal_score,
#             "occlusion_score": occlusion_score,
#             "upper_face_visible": upper_face_visible,
#             "lower_face_visible": lower_face_visible,
#             "face_completeness": face_completeness,
#             "mouth_occluded_score": mouth_occluded_score,
#             "eye_visibility_score": eye_visibility_score,
#             **geo,
#             **iris,
#             "blendshapes": blend_dict,
#             "matrix": matrix,
#         }

#     # =========================================================================
#     # PUBLIC: is_valid_face()
#     # =========================================================================

#     def is_valid_face(self, analysis: Dict, cam_code: str, debug: bool = True) -> bool:

#         def _reject(reason: str) -> bool:
#             if debug:
#                 print(f"[Gate] [{cam_code}] ❌ {reason}")
#             return False

#         if analysis is None:
#             return _reject("Invalid analysis object: None")

#         if not analysis.get("valid"):
#             return _reject(f"Invalid analysis object: {analysis.get('reason', 'unknown')}")

#         if debug:
#             self._debug_summary(analysis)

#         # ------------------------------------------------------------------ #
#         # LAYER 1 — IMAGE QUALITY                                             #
#         # ------------------------------------------------------------------ #
#         blur = analysis.get("blur", 0)
#         size = analysis.get("size", 0)

#         if blur < self.blur_threshold:
#             return _reject(f"Blur too low: {blur:.2f} < {self.blur_threshold}")

#         if size < self.min_face_size:
#             return _reject(f"Face too small: {size}px < {self.min_face_size}px")

#         # ------------------------------------------------------------------ #
#         # LAYER 2 — POSE                                                      #
#         # ------------------------------------------------------------------ #
#         yaw = analysis.get("yaw")
#         pitch = analysis.get("pitch")
#         roll = analysis.get("roll")

#         if yaw is None or pitch is None or roll is None:
#             return _reject("Missing pose angles")

#         if abs(yaw) > self.yaw_threshold:
#             return _reject(f"Yaw out of range: {yaw:.1f}° (max {self.yaw_threshold}°)")

#         if abs(pitch) > self.pitch_threshold:
#             return _reject(f"Pitch out of range: {pitch:.1f}° (max {self.pitch_threshold}°)")

#         if abs(roll) > self.roll_threshold:
#             return _reject(f"Roll out of range: {roll:.1f}° (max {self.roll_threshold}°)")

#         # ------------------------------------------------------------------ #
#         # LAYER 3 — BLENDSHAPE EYE / EXPRESSION / OCCLUSION                  #
#         # ------------------------------------------------------------------ #
#         eye_score = analysis.get("eye_score")
#         eye_balance = analysis.get("eye_balance")

#         if eye_score is None or eye_score < 0.6:
#             return _reject(f"Eyes not sufficiently open (blendshape): eye_score={eye_score}")

#         if eye_balance is not None and eye_balance > 0.4:
#             return _reject(f"Asymmetric blink (blendshape): eye_balance={eye_balance:.2f}")

#         if not analysis.get("upper_face_visible", False):
#             return _reject("Upper face not visible (blendshape check)")

#         if not analysis.get("lower_face_visible", False):
#             return _reject("Lower face not visible (blendshape check)")

#         if analysis.get("face_completeness", 0) < 1.0:
#             return _reject(f"Partial face (blendshape): completeness={analysis.get('face_completeness'):.2f}")

#         if analysis.get("occlusion_score", 0) > 0.5:
#             return _reject(f"Face occluded (blendshape): occlusion_score={analysis.get('occlusion_score'):.2f}")

#         if analysis.get("mouth_occluded_score", 0) > 0.5:
#             return _reject(f"Mouth occluded (blendshape): mouth_occluded_score={analysis.get('mouth_occluded_score'):.2f}")

#         if analysis.get("expression_score", 0) > 0.6:
#             return _reject(f"Expression too strong: expression_score={analysis.get('expression_score'):.2f}")

#         # ------------------------------------------------------------------ #
#         # LAYER 4 — GEOMETRIC SIGNALS                                         #
#         # ------------------------------------------------------------------ #
#         face_width_ratio = analysis.get("geo_face_width_ratio")
#         face_height_ratio = analysis.get("geo_face_height_ratio")

#         if face_width_ratio is not None and face_width_ratio < self.min_face_width_ratio:
#             return _reject(f"Face too narrow (geometric): {face_width_ratio:.3f}")

#         if face_height_ratio is not None and face_height_ratio < self.min_face_height_ratio:
#             return _reject(f"Face too short (geometric): {face_height_ratio:.3f}")

#         eye_dist_ratio = analysis.get("geo_eye_distance_ratio")
#         if eye_dist_ratio is not None and eye_dist_ratio < self.min_eye_distance_ratio:
#             return _reject(f"Eyes too close / one eye missing: eye_dist_ratio={eye_dist_ratio:.3f}")

#         left_ear = analysis.get("geo_left_ear")
#         right_ear = analysis.get("geo_right_ear")

#         if left_ear is not None and left_ear < self.min_ear:
#             return _reject(f"Left eye geometrically closed: EAR={left_ear:.3f}")

#         if right_ear is not None and right_ear < self.min_ear:
#             return _reject(f"Right eye geometrically closed: EAR={right_ear:.3f}")

#         if left_ear is not None and right_ear is not None:
#             if abs(left_ear - right_ear) > 0.07:
#                 return _reject(
#                     f"EAR asymmetry (one eye occluded): "
#                     f"L={left_ear:.3f} R={right_ear:.3f} diff={abs(left_ear-right_ear):.3f}"
#                 )

#         mouth_width_ratio = analysis.get("geo_mouth_width_ratio")
#         if mouth_width_ratio is not None and mouth_width_ratio < self.min_mouth_width_ratio:
#             return _reject(f"Mouth too narrow (occluded?): {mouth_width_ratio:.3f}")

#         lateral_asymmetry = analysis.get("geo_lateral_asymmetry")
#         if lateral_asymmetry is not None and lateral_asymmetry > self.max_lateral_asymmetry:
#             return _reject(f"Lateral asymmetry too high: {lateral_asymmetry:.3f}")

#         nose_offset = analysis.get("geo_nose_midline_offset")
#         if nose_offset is not None and nose_offset > 0.15:
#             return _reject(f"Nose offset from midline: {nose_offset:.3f}")

#         # ------------------------------------------------------------------ #
#         # LAYER 5 — IRIS SIGNALS                                              #
#         #                                                                     #
#         # WHY IRIS INSTEAD OF SKIN DETECTION:                                 #
#         # Skin-based detection fails for dark/medium skin tones and glasses   #
#         # wearers because the periocular region looks skin-like regardless.   #
#         #                                                                     #
#         # Iris landmarks ARE reliable because:                                #
#         # 1. Iris radius — MediaPipe fits a circle to iris edge points.       #
#         #    When a hand occludes the eye, the model can't fit the iris       #
#         #    properly → radius collapses or becomes degenerate.               #
#         # 2. Iris center brightness — the pupil is always dark.               #
#         #    A hand over the eye makes the iris center region bright (skin).  #
#         #    This is SKIN-TONE INDEPENDENT: dark skin is still brighter       #
#         #    than a pupil.                                                     #
#         # 3. Brightness asymmetry — if one iris center is much brighter       #
#         #    than the other, that eye is covered regardless of skin tone.     #
#         # ------------------------------------------------------------------ #

#         left_r_px  = analysis.get("iris_left_radius_px")
#         right_r_px = analysis.get("iris_right_radius_px")

#         # 5a. Per-iris radius must exceed minimum
#         if left_r_px is not None and left_r_px < self.min_iris_radius_px:
#             return _reject(
#                 f"Left iris radius too small (eye occluded/degenerate): "
#                 f"radius={left_r_px:.1f}px < {self.min_iris_radius_px}px"
#             )

#         if right_r_px is not None and right_r_px < self.min_iris_radius_px:
#             return _reject(
#                 f"Right iris radius too small (eye occluded/degenerate): "
#                 f"radius={right_r_px:.1f}px < {self.min_iris_radius_px}px"
#             )

#         # 5b. Iris radius asymmetry
#         if left_r_px is not None and right_r_px is not None:
#             if left_r_px > 0 and right_r_px > 0:
#                 radius_ratio = min(left_r_px, right_r_px) / max(left_r_px, right_r_px)
#                 if radius_ratio < self.min_iris_radius_ratio:
#                     return _reject(
#                         f"Iris radius asymmetry (one eye occluded): "
#                         f"L={left_r_px:.1f}px R={right_r_px:.1f}px ratio={radius_ratio:.2f} < {self.min_iris_radius_ratio}"
#                     )

#         # 5c. Iris center brightness — pupil must be dark
#         left_brightness  = analysis.get("iris_left_center_brightness")
#         right_brightness = analysis.get("iris_right_center_brightness")

#         if left_brightness is not None and left_brightness > self.max_iris_center_brightness:
#             return _reject(
#                 f"Left iris center too bright (hand over eye — skin covering pupil): "
#                 f"brightness={left_brightness:.1f} > {self.max_iris_center_brightness}"
#             )

#         if right_brightness is not None and right_brightness > self.max_iris_center_brightness:
#             return _reject(
#                 f"Right iris center too bright (hand over eye — skin covering pupil): "
#                 f"brightness={right_brightness:.1f} > {self.max_iris_center_brightness}"
#             )

#         # 5d. Brightness asymmetry between iris centers
#         if left_brightness is not None and right_brightness is not None:
#             brightness_diff = abs(left_brightness - right_brightness)
#             if brightness_diff > self.max_iris_brightness_asymmetry:
#                 return _reject(
#                     f"Iris center brightness asymmetry (one eye occluded): "
#                     f"L={left_brightness:.1f} R={right_brightness:.1f} diff={brightness_diff:.1f} > {self.max_iris_brightness_asymmetry}"
#                 )

#         # ------------------------------------------------------------------ #
#         # ACCEPT                                                               #
#         # ------------------------------------------------------------------ #
#         if debug:
#             print("[Gate] ✅ Accepted")
#         return True

#     # =========================================================================
#     # PRIVATE: iris signals
#     # =========================================================================

#     def _extract_iris_signals(
#         self, face_img: np.ndarray, lm_list, img_w: int, img_h: int
#     ) -> Dict:
#         """
#         Extract iris-based occlusion signals using MediaPipe's iris landmarks.

#         The 478-point model includes 5 iris landmarks per eye:
#           Left:  center=468, top=469, right=470, bottom=471, left=472
#           Right: center=473, top=474, right=475, bottom=476, left=477

#         Signals:
#           1. iris_radius_px — mean distance from center to 4 edge points (pixels)
#              Real open eye: 8-20px depending on face size.
#              Occluded/degenerate: < 4px.

#           2. iris_center_brightness — mean grayscale in a small disk at iris center.
#              Real pupil: 20-80 (very dark).
#              Hand/skin over eye: 100-200 (medium bright).
#              This is SKIN-TONE INDEPENDENT — even dark skin is lighter than a pupil.

#         Returns dict with iris_left_* and iris_right_* keys.
#         """
#         gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

#         def _safe_lm(idx):
#             """Get landmark as pixel coordinate, None if index out of range."""
#             if idx >= len(lm_list):
#                 return None
#             p = lm_list[idx]
#             return int(p.x * img_w), int(p.y * img_h)

#         def _iris_signals(center_idx, edge_indices):
#             center_px = _safe_lm(center_idx)
#             if center_px is None:
#                 return None, None

#             cx, cy = center_px
#             if cx < 0 or cy < 0 or cx >= img_w or cy >= img_h:
#                 return None, None

#             # Compute radius from edge landmarks
#             radii = []
#             for idx in edge_indices:
#                 ep = _safe_lm(idx)
#                 if ep is not None:
#                     ex, ey = ep
#                     r = np.sqrt((ex - cx)**2 + (ey - cy)**2)
#                     radii.append(r)

#             radius_px = float(np.mean(radii)) if radii else None

#             # Sample brightness in a small disk at iris center
#             # Radius = half the iris radius (just the pupil zone)
#             sample_r = max(2, int((radius_px or 4) * 0.5))
#             y1 = max(0, cy - sample_r)
#             y2 = min(img_h, cy + sample_r + 1)
#             x1 = max(0, cx - sample_r)
#             x2 = min(img_w, cx + sample_r + 1)

#             if (y2 - y1) < 1 or (x2 - x1) < 1:
#                 return radius_px, None

#             # Create circular mask
#             patch_gray = gray[y1:y2, x1:x2].astype(np.float32)
#             ph, pw = patch_gray.shape
#             ys_g, xs_g = np.ogrid[:ph, :pw]
#             mask = (xs_g - pw//2)**2 + (ys_g - ph//2)**2 <= sample_r**2

#             if mask.sum() == 0:
#                 center_brightness = float(np.mean(patch_gray))
#             else:
#                 center_brightness = float(np.mean(patch_gray[mask]))

#             return radius_px, center_brightness

#         left_radius, left_brightness = _iris_signals(
#             _LM["left_iris_center"],
#             [_LM["left_iris_top"], _LM["left_iris_right"],
#              _LM["left_iris_bottom"], _LM["left_iris_left"]],
#         )

#         right_radius, right_brightness = _iris_signals(
#             _LM["right_iris_center"],
#             [_LM["right_iris_top"], _LM["right_iris_right"],
#              _LM["right_iris_bottom"], _LM["right_iris_left"]],
#         )

#         return {
#             "iris_left_radius_px":         left_radius,
#             "iris_right_radius_px":        right_radius,
#             "iris_left_center_brightness": left_brightness,
#             "iris_right_center_brightness":right_brightness,
#         }

#     # =========================================================================
#     # PRIVATE: geometric signal extraction
#     # =========================================================================

#     def _extract_geometric_signals(self, lm_list, img_w: int, img_h: int) -> Dict:
#         def lm(key):
#             p = lm_list[_LM[key]]
#             return p.x, p.y

#         all_pts = np.array([[p.x, p.y] for p in lm_list], dtype=np.float32)
#         x_min, y_min = all_pts.min(axis=0)
#         x_max, y_max = all_pts.max(axis=0)
#         face_width_norm  = float(x_max - x_min)
#         face_height_norm = float(y_max - y_min)

#         le_outer = np.array(lm("left_eye_outer"))
#         le_inner = np.array(lm("left_eye_inner"))
#         le_upper = np.array(lm("left_eye_upper"))
#         le_lower = np.array(lm("left_eye_lower"))
#         re_outer = np.array(lm("right_eye_outer"))
#         re_inner = np.array(lm("right_eye_inner"))
#         re_upper = np.array(lm("right_eye_upper"))
#         re_lower = np.array(lm("right_eye_lower"))
#         nose    = np.array(lm("nose_tip"))
#         m_left  = np.array(lm("mouth_left"))
#         m_right = np.array(lm("mouth_right"))

#         left_eye_centre  = (le_outer + le_inner) / 2.0
#         right_eye_centre = (re_outer + re_inner) / 2.0
#         eye_dist_norm  = float(np.linalg.norm(right_eye_centre - left_eye_centre))
#         eye_dist_ratio = eye_dist_norm / (face_width_norm + 1e-6)

#         def _ear(outer, inner, upper, lower):
#             eye_w = float(np.linalg.norm(outer - inner)) + 1e-6
#             dy = abs(float(upper[1] - lower[1])) * img_h
#             dx = eye_w * img_w
#             return dy / (dx + 1e-6)

#         left_ear  = _ear(le_outer, le_inner, le_upper, le_lower)
#         right_ear = _ear(re_outer, re_inner, re_upper, re_lower)

#         mouth_width_norm  = float(np.linalg.norm(m_right - m_left))
#         mouth_width_ratio = mouth_width_norm / (eye_dist_norm + 1e-6)

#         face_mid_x = float((left_eye_centre[0] + right_eye_centre[0]) / 2.0)
#         left_span  = face_mid_x - x_min
#         right_span = x_max - face_mid_x
#         lateral_asymmetry   = abs(left_span - right_span) / (face_width_norm + 1e-6)
#         nose_midline_offset = abs(float(nose[0]) - face_mid_x) / (face_width_norm + 1e-6)

#         return {
#             "geo_face_width_ratio":    face_width_norm,
#             "geo_face_height_ratio":   face_height_norm,
#             "geo_eye_distance_ratio":  eye_dist_ratio,
#             "geo_left_ear":            float(left_ear),
#             "geo_right_ear":           float(right_ear),
#             "geo_mouth_width_ratio":   float(mouth_width_ratio),
#             "geo_lateral_asymmetry":   float(lateral_asymmetry),
#             "geo_nose_midline_offset": float(nose_midline_offset),
#             "geo_eye_dist_px":         float(eye_dist_norm * img_w),
#             "geo_face_width_px":       float(face_width_norm * img_w),
#             "geo_face_height_px":      float(face_height_norm * img_h),
#         }

#     # =========================================================================
#     # PRIVATE: helpers
#     # =========================================================================

#     def _debug_summary(self, a: Dict):
#         def _fmt(value, default, spec):
#             if value is None:
#                 value = default
#             return format(value, spec)

#         print(
#             f"[Gate] "
#             f"blur={_fmt(a.get('blur'), 0.0, '.1f')}  "
#             f"size={a.get('size', 0)}px  "
#             f"yaw={_fmt(a.get('yaw'), 0.0, '.1f')}° pitch={_fmt(a.get('pitch'), 0.0, '.1f')}° roll={_fmt(a.get('roll'), 0.0, '.1f')}°  "
#             f"eye_score={_fmt(a.get('eye_score'), 0.0, '.2f')} eye_bal={_fmt(a.get('eye_balance'), 0.0, '.2f')}  "
#             f"upper={a.get('upper_face_visible')} lower={a.get('lower_face_visible')}  "
#             f"occ={_fmt(a.get('occlusion_score'), 0.0, '.2f')} mouth_occ={_fmt(a.get('mouth_occluded_score'), 0.0, '.2f')}  "
#             f"expr={_fmt(a.get('expression_score'), 0.0, '.2f')}  "
#             f"| GEO: "
#             f"w={_fmt(a.get('geo_face_width_ratio'), 0.0, '.2f')} h={_fmt(a.get('geo_face_height_ratio'), 0.0, '.2f')}  "
#             f"eye_dist={_fmt(a.get('geo_eye_distance_ratio'), 0.0, '.2f')}  "
#             f"L-EAR={_fmt(a.get('geo_left_ear'), 0.0, '.3f')} R-EAR={_fmt(a.get('geo_right_ear'), 0.0, '.3f')}  "
#             f"mouth_w={_fmt(a.get('geo_mouth_width_ratio'), 0.0, '.2f')}  "
#             f"lat_asym={_fmt(a.get('geo_lateral_asymmetry'), 0.0, '.3f')}  "
#             f"nose_off={_fmt(a.get('geo_nose_midline_offset'), 0.0, '.3f')}  "
#             f"| IRIS: "
#             f"L-r={_fmt(a.get('iris_left_radius_px'), 0.0, '.1f')}px "
#             f"R-r={_fmt(a.get('iris_right_radius_px'), 0.0, '.1f')}px  "
#             f"L-bright={_fmt(a.get('iris_left_center_brightness'), 0.0, '.1f')} "
#             f"R-bright={_fmt(a.get('iris_right_center_brightness'), 0.0, '.1f')}"
#         )

#     def _blendshape_dict(self, blendshapes):
#         if not blendshapes:
#             return None
#         return {b.category_name: b.score for b in blendshapes}

#     def _eyes_open(self, blend):
#         left  = blend.get("eyeBlinkLeft")
#         right = blend.get("eyeBlinkRight")
#         if left is None or right is None:
#             return False
#         return (left < self.eye_blink_threshold) and (right < self.eye_blink_threshold)

#     def _neutral_expression(self, blend):
#         smile = blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
#         mouth_open = blend.get("jawOpen", 0.0)
#         return smile < self.smile_threshold and mouth_open < self.mouth_open_threshold

#     def _is_frontal(self, matrix):
#         pose = self._extract_pose(matrix)
#         if pose is None:
#             return False
#         yaw, pitch, roll = pose
#         return abs(yaw) < self.yaw_threshold and abs(pitch) < self.pitch_threshold and abs(roll) < self.roll_threshold

#     def _is_blurry(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold

#     def _extract_pose(self, matrix):
#         try:
#             mat = np.array(matrix)
#             if mat.size != 16:
#                 return None
#             mat  = mat.reshape(4, 4)
#             rmat = mat[:3, :3]
#             sy   = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
#             singular = sy < 1e-6
#             if not singular:
#                 x = np.arctan2(rmat[2, 1], rmat[2, 2])
#                 y = np.arctan2(-rmat[2, 0], sy)
#                 z = np.arctan2(rmat[1, 0], rmat[0, 0])
#             else:
#                 x = np.arctan2(-rmat[1, 2], rmat[1, 1])
#                 y = np.arctan2(-rmat[2, 0], sy)
#                 z = 0
#             return np.degrees(y), np.degrees(x), np.degrees(z)
#         except Exception:
#             return None

#     def _to_mp_image(self, img):
#         return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)


















# import cv2
# import mediapipe as mp
# import numpy as np
# from typing import Optional, Dict

# from mediapipe.tasks.python import vision
# from mediapipe.tasks.python.core.base_options import BaseOptions
# from mediapipe.tasks.python.vision import (
#     FaceLandmarker,
#     FaceLandmarkerOptions,
#     RunningMode,
# )
# from app.config.config import envConfig


# # ---------------------------------------------------------------------------
# # MediaPipe FaceMesh landmark indices (478-point model)
# # ---------------------------------------------------------------------------
# _LM = {
#     "left_eye_outer":     33,
#     "left_eye_inner":     133,
#     "left_eye_upper":     159,
#     "left_eye_lower":     145,
#     "right_eye_outer":    263,
#     "right_eye_inner":    362,
#     "right_eye_upper":    386,
#     "right_eye_lower":    374,
#     "nose_tip":           1,
#     "mouth_left":         61,
#     "mouth_right":        291,
#     "mouth_upper":        13,
#     "mouth_lower":        14,
#     "chin":               152,
#     "left_cheek":         234,
#     "right_cheek":        454,
#     "left_brow_outer":    70,
#     "right_brow_outer":   300,
#     # Iris landmarks (478-point model only)
#     "left_iris_center":   468,
#     "left_iris_top":      469,
#     "left_iris_right":    470,
#     "left_iris_bottom":   471,
#     "left_iris_left":     472,
#     "right_iris_center":  473,
#     "right_iris_top":     474,
#     "right_iris_right":   475,
#     "right_iris_bottom":  476,
#     "right_iris_left":    477,
# }

# _EYE_PATCH_PAD = 1.8


# # ---------------------------------------------------------------------------
# # Camera profile presets
# # ---------------------------------------------------------------------------
# # Use these when instantiating for different deployment contexts.
# #
# # WEBCAM  — close-up, face ≥ 200px, no fixed camera angle
# # CCTV    — ceiling-mounted, face 70-150px, downward pitch is normal
# #
# # Example:
# #   engine = FaceLandmarkerEngine(model_path=..., **PROFILE_CCTV)
# #   engine = FaceLandmarkerEngine(model_path=..., **PROFILE_WEBCAM)
# # ---------------------------------------------------------------------------
# PROFILE_WEBCAM = dict(
#     yaw_threshold=20,
#     pitch_threshold=25,
#     roll_threshold=20,
#     occlusion_threshold=0.50,
#     ear_asymmetry_threshold=0.07,
#     upscale_to=None,                  # no upscaling needed
#     iris_radius_factor=0.035,         # min_iris_radius = face_size * factor
#     min_iris_radius_ratio=0.45,
#     max_iris_center_brightness=140.0,
#     max_iris_brightness_asymmetry=60.0,
# )

# PROFILE_CCTV = dict(
#     yaw_threshold=30,
#     pitch_threshold=35,               # ceiling-mount: normal downward pitch
#     roll_threshold=25,
#     occlusion_threshold=0.70,         # eyeSquint is noisy at low resolution
#     ear_asymmetry_threshold=0.13,     # 1px noise = 0.02-0.04 EAR at 80px
#     upscale_to=160,                   # upscale before inference (landmarks improve dramatically)
#     iris_radius_factor=0.025,         # smaller faces → smaller absolute iris
#     min_iris_radius_ratio=0.40,       # looser ratio for small/noisy iris fitting
#     max_iris_center_brightness=150.0, # slightly more tolerant for compressed CCTV frames
#     max_iris_brightness_asymmetry=70.0,
#     eye_score_threshold=0.55,
# )


# class FaceLandmarkerEngine:
#     def __init__(
#         self,
#         model_path: str,
#         min_face_presence_confidence: float = 0.5,
#         min_tracking_confidence: float = 0.5,
#         # ---- expression / blink thresholds ----
#         eye_blink_threshold: float = 0.4,
#         smile_threshold: float = 0.6,
#         mouth_open_threshold: float = 0.4,
#         # ---- pose thresholds ----
#         yaw_threshold: float = 20,
#         pitch_threshold: float = 25,
#         roll_threshold: float = 20,
#         # ---- image quality ----
#         blur_threshold: float = envConfig.BLUR_THRESHOLD,
#         min_face_size: int = envConfig.MIN_FACE_SIZE,
#         # ---- geometric thresholds ----
#         min_eye_distance_ratio: float = 0.25,
#         min_face_height_ratio: float = 0.30,
#         min_face_width_ratio: float = 0.25,
#         max_lateral_asymmetry: float = 0.30,
#         min_ear: float = 0.10,
#         min_mouth_width_ratio: float = 0.60,
#         # ---- blendshape occlusion threshold ----
#         # CCTV: raise to 0.70 (eyeSquint fires spuriously at low resolution)
#         # Webcam: keep at 0.50
#         occlusion_threshold: float = 0.50,
#         # ---- EAR asymmetry threshold ----
#         # CCTV: raise to 0.13 (1px quantization noise at 80px face = 0.02-0.04 EAR)
#         # Webcam: keep at 0.07
#         ear_asymmetry_threshold: float = 0.07,
#         # ---- upscaling ----
#         # CCTV: set to 160 — upscale small face crops before MediaPipe inference.
#         # Dramatically improves landmark accuracy at < 120px face size.
#         # Webcam: set to None (no upscaling needed)
#         upscale_to: Optional[int] = None,
#         # ---- iris thresholds ----
#         # iris_radius_factor: min_iris_radius = face_size * factor
#         # This makes the threshold RELATIVE to face size, so it works for both
#         # 80px CCTV faces and 300px webcam faces.
#         # CCTV:   0.025  (80px face → min 2.0px)
#         # Webcam: 0.035  (250px face → min 8.75px)
#         iris_radius_factor: float = 0.035,
#         min_iris_radius_ratio: float = 0.45,
#         max_iris_center_brightness: float = 140.0,
#         max_iris_brightness_asymmetry: float = 60.0,
#         eye_score_threshold: float = 0.50,
#     ):
#         self.eye_blink_threshold = eye_blink_threshold
#         self.smile_threshold = smile_threshold
#         self.mouth_open_threshold = mouth_open_threshold
#         self.yaw_threshold = yaw_threshold
#         self.pitch_threshold = pitch_threshold
#         self.roll_threshold = roll_threshold
#         self.blur_threshold = blur_threshold
#         self.min_face_size = min_face_size
#         self.min_eye_distance_ratio = min_eye_distance_ratio
#         self.min_face_height_ratio = min_face_height_ratio
#         self.min_face_width_ratio = min_face_width_ratio
#         self.max_lateral_asymmetry = max_lateral_asymmetry
#         self.min_ear = min_ear
#         self.min_mouth_width_ratio = min_mouth_width_ratio
#         self.occlusion_threshold = occlusion_threshold
#         self.ear_asymmetry_threshold = ear_asymmetry_threshold
#         self.upscale_to = upscale_to
#         self.iris_radius_factor = iris_radius_factor
#         self.min_iris_radius_ratio = min_iris_radius_ratio
#         self.max_iris_center_brightness = max_iris_center_brightness
#         self.max_iris_brightness_asymmetry = max_iris_brightness_asymmetry
#         self.eye_score_threshold = eye_score_threshold

#         options = FaceLandmarkerOptions(
#             base_options=BaseOptions(model_asset_path=model_path),
#             running_mode=RunningMode.IMAGE,
#             output_face_blendshapes=True,
#             output_facial_transformation_matrixes=True,
#             min_face_presence_confidence=min_face_presence_confidence,
#             min_tracking_confidence=min_tracking_confidence,
#             num_faces=1,
#         )
#         self.landmarker = FaceLandmarker.create_from_options(options)

#     # =========================================================================
#     # PUBLIC: analyze()
#     # =========================================================================

#     def analyze(self, face_img: np.ndarray, debug: bool = False) -> Dict:
#         if face_img is None or face_img.size == 0:
#             return {"valid": False, "reason": "empty"}

#         # ------------------------------------------------------------------
#         # STEP 0 — OPTIONAL UPSCALING
#         # ------------------------------------------------------------------
#         # For CCTV, face crops arrive at 70-120px. MediaPipe landmark accuracy
#         # degrades sharply below ~120px, causing cascading failures in EAR,
#         # iris radius, and blendshape scores. Upscaling to 160px first fixes this.
#         #
#         # We record the ORIGINAL size for gating (the face is actually that small
#         # in the real world) but run all inference on the upscaled image so that
#         # the extracted signals are accurate.
#         # ------------------------------------------------------------------
#         original_h, original_w = face_img.shape[:2]
#         original_size = min(original_h, original_w)

#         inference_img = face_img
#         if self.upscale_to is not None and original_size < self.upscale_to:
#             scale = self.upscale_to / original_size
#             new_w = int(original_w * scale)
#             new_h = int(original_h * scale)
#             inference_img = cv2.resize(
#                 face_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
#             )

#         h, w = inference_img.shape[:2]
#         size = original_size          # gate on REAL size, not upscaled
#         aspect_ratio = original_w / (original_h + 1e-6)

#         # ------------------------------------------------------------------
#         # 1. IMAGE QUALITY  (computed on original, not upscaled — blur is
#         #    a property of the source image, not the interpolated one)
#         # ------------------------------------------------------------------
#         gray_orig = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
#         blur = cv2.Laplacian(gray_orig, cv2.CV_64F).var()
#         brightness = float(np.mean(gray_orig))
#         contrast = float(np.std(gray_orig))

#         # ------------------------------------------------------------------
#         # 2. MEDIAPIPE INFERENCE  (on upscaled image if applicable)
#         # ------------------------------------------------------------------
#         rgb = cv2.cvtColor(inference_img, cv2.COLOR_BGR2RGB)
#         mp_image = self._to_mp_image(rgb)
#         result = self.landmarker.detect(mp_image)

#         if not result.face_landmarks:
#             return {
#                 "valid": False,
#                 "reason": "no_landmarks",
#                 "blur": blur,
#                 "brightness": brightness,
#                 "contrast": contrast,
#                 "size": size,
#             }

#         # ------------------------------------------------------------------
#         # 3. BLENDSHAPES
#         # ------------------------------------------------------------------
#         blend_dict = None
#         if result.face_blendshapes:
#             blend_dict = {
#                 b.category_name: float(b.score)
#                 for b in result.face_blendshapes[0]
#             }

#         # ------------------------------------------------------------------
#         # 4. POSE
#         # ------------------------------------------------------------------
#         matrix = (
#             result.facial_transformation_matrixes[0]
#             if result.facial_transformation_matrixes
#             else None
#         )
#         yaw = pitch = roll = None
#         if matrix is not None:
#             pose = self._extract_pose(matrix)
#             if pose:
#                 yaw, pitch, roll = pose

#         # ------------------------------------------------------------------
#         # 5. BLENDSHAPE-DERIVED SIGNALS
#         # ------------------------------------------------------------------
#         eye_score = eye_balance = None
#         if blend_dict:
#             lb = blend_dict.get("eyeBlinkLeft", 0.0)
#             rb = blend_dict.get("eyeBlinkRight", 0.0)
#             eye_score = 1.0 - max(lb, rb)
#             eye_balance = abs(lb - rb)

#         expression_score = 0.0
#         if blend_dict:
#             smile = blend_dict.get("mouthSmileLeft", 0.0) + blend_dict.get("mouthSmileRight", 0.0)
#             mouth_open = blend_dict.get("jawOpen", 0.0)
#             expression_score = max(smile, mouth_open)

#         # ------------------------------------------------------------------
#         # OCCLUSION SCORE — cheekPuff only, NOT eyeSquint
#         # ------------------------------------------------------------------
#         # eyeSquint is EXCLUDED intentionally.
#         #
#         # Why: eyeSquint is a blendshape that fires when the eyelids compress
#         # vertically. At low CCTV resolution (70-100px faces), MediaPipe cannot
#         # fit the eyelid texture accurately and eyeSquint fires constantly even
#         # on clean open eyes. This caused ~40% of clean CCTV frames to be
#         # rejected as "face occluded" even with no occlusion present.
#         #
#         # cheekPuff remains: it's a reliable signal for mask/hand over lower face
#         # and does NOT fire spuriously on small faces.
#         #
#         # For the upper-face visibility check (upper_face_visible), we still use
#         # blink blendshapes which are more robust than squint.
#         # ------------------------------------------------------------------
#         occlusion_score = blend_dict.get("cheekPuff", 0.0) if blend_dict else 0.0

#         frontal_score = None
#         if yaw is not None and pitch is not None:
#             frontal_score = max(0.0, 1.0 - (abs(yaw) / 40.0 + abs(pitch) / 30.0) / 2.0)

#         pose_magnitude = None
#         if yaw is not None and pitch is not None and roll is not None:
#             pose_magnitude = float(np.sqrt(yaw**2 + pitch**2 + roll**2))

#         upper_face_visible = lower_face_visible = False
#         face_completeness = mouth_occluded_score = 0.0
#         eye_visibility_score = None

#         if blend_dict:
#             # eye_visibility uses blink only (not squint — see above)
#             eye_visibility_score = 1.0 - max(
#                 blend_dict.get("eyeBlinkLeft", 0.0),
#                 blend_dict.get("eyeBlinkRight", 0.0),
#             )
#             upper_face_visible = (
#                 eye_visibility_score > 0.4
#                 and eye_balance is not None
#                 and eye_balance < 0.4
#             )
#             jaw = blend_dict.get("jawOpen", 0.0)
#             mouth_press = (
#                 blend_dict.get("mouthPressLeft", 0.0)
#                 + blend_dict.get("mouthPressRight", 0.0)
#             )
#             mouth_shrug = blend_dict.get("mouthShrugLower", 0.0)
#             lower_face_visible = jaw < 0.4 and mouth_press < 0.4 and mouth_shrug < 0.3
#             mouth_occluded_score = max(blend_dict.get("cheekPuff", 0.0), mouth_press)
#             face_completeness = (int(upper_face_visible) + int(lower_face_visible)) / 2.0

#         # ------------------------------------------------------------------
#         # 6. GEOMETRIC SIGNALS  (on inference image dimensions)
#         # ------------------------------------------------------------------
#         lm_list = result.face_landmarks[0]
#         geo = self._extract_geometric_signals(lm_list, img_w=w, img_h=h)

#         # ------------------------------------------------------------------
#         # 7. IRIS SIGNALS  (on inference image)
#         # ------------------------------------------------------------------
#         iris = self._extract_iris_signals(inference_img, lm_list, img_w=w, img_h=h)

#         return {
#             "valid": True,
#             "blur": blur,
#             "brightness": brightness,
#             "contrast": contrast,
#             "size": size,
#             "aspect_ratio": aspect_ratio,
#             "yaw": yaw,
#             "pitch": pitch,
#             "roll": roll,
#             "pose_magnitude": pose_magnitude,
#             "eye_score": eye_score,
#             "eye_balance": eye_balance,
#             "expression_score": expression_score,
#             "frontal_score": frontal_score,
#             "occlusion_score": occlusion_score,
#             "upper_face_visible": upper_face_visible,
#             "lower_face_visible": lower_face_visible,
#             "face_completeness": face_completeness,
#             "mouth_occluded_score": mouth_occluded_score,
#             "eye_visibility_score": eye_visibility_score,
#             **geo,
#             **iris,
#             "blendshapes": blend_dict,
#             "matrix": matrix,
#         }

#     # =========================================================================
#     # PUBLIC: is_valid_face()
#     # =========================================================================

#     def is_valid_face(self, analysis: Dict, cam_code: str = "", debug: bool = True) -> bool:

#         def _reject(reason: str) -> bool:
#             if debug:
#                 prefix = f"[{cam_code}] " if cam_code else ""
#                 print(f"[Gate] {prefix}❌ {reason}")
#             return False

#         if analysis is None:
#             return _reject("Invalid analysis object: None")

#         if not analysis.get("valid"):
#             return _reject(f"Invalid analysis object: {analysis.get('reason', 'unknown')}")

#         if debug:
#             self._debug_summary(analysis, cam_code)

#         size = analysis.get("size", 0)

#         # ------------------------------------------------------------------ #
#         # LAYER 1 — IMAGE QUALITY                                             #
#         # ------------------------------------------------------------------ #
#         blur = analysis.get("blur", 0)

#         if blur < self.blur_threshold:
#             return _reject(f"Blur too low: {blur:.2f} < {self.blur_threshold}")

#         if size < self.min_face_size:
#             return _reject(f"Face too small: {size}px < {self.min_face_size}px")

#         # ------------------------------------------------------------------ #
#         # LAYER 2 — POSE                                                      #
#         # ------------------------------------------------------------------ #
#         yaw   = analysis.get("yaw")
#         pitch = analysis.get("pitch")
#         roll  = analysis.get("roll")

#         if yaw is None or pitch is None or roll is None:
#             return _reject("Missing pose angles")

#         if abs(yaw) > self.yaw_threshold:
#             return _reject(f"Yaw out of range: {yaw:.1f}° (max {self.yaw_threshold}°)")

#         if abs(pitch) > self.pitch_threshold:
#             return _reject(f"Pitch out of range: {pitch:.1f}° (max {self.pitch_threshold}°)")

#         if abs(roll) > self.roll_threshold:
#             return _reject(f"Roll out of range: {roll:.1f}° (max {self.roll_threshold}°)")

#         # ------------------------------------------------------------------ #
#         # LAYER 3 — BLENDSHAPE EYE / EXPRESSION / OCCLUSION                  #
#         # ------------------------------------------------------------------ #
#         eye_score  = analysis.get("eye_score")
#         eye_balance = analysis.get("eye_balance")

#         if eye_score is None or eye_score < self.eye_score_threshold:
#             return _reject(f"Eyes not sufficiently open (blendshape): eye_score={eye_score}")

#         if eye_balance is not None and eye_balance > 0.4:
#             return _reject(f"Asymmetric blink (blendshape): eye_balance={eye_balance:.2f}")

#         if not analysis.get("upper_face_visible", False):
#             return _reject("Upper face not visible (blendshape check)")

#         if not analysis.get("lower_face_visible", False):
#             return _reject("Lower face not visible (blendshape check)")

#         if analysis.get("face_completeness", 0) < 1.0:
#             return _reject(f"Partial face (blendshape): completeness={analysis.get('face_completeness'):.2f}")

#         # occlusion_threshold is profile-dependent (0.50 webcam / 0.70 CCTV)
#         if analysis.get("occlusion_score", 0) > self.occlusion_threshold:
#             return _reject(
#                 f"Face occluded (cheekPuff): occlusion_score={analysis.get('occlusion_score'):.2f} "
#                 f"> {self.occlusion_threshold}"
#             )

#         if analysis.get("mouth_occluded_score", 0) > 0.5:
#             return _reject(f"Mouth occluded: mouth_occluded_score={analysis.get('mouth_occluded_score'):.2f}")

#         if analysis.get("expression_score", 0) > 0.6:
#             return _reject(f"Expression too strong: expression_score={analysis.get('expression_score'):.2f}")

#         # ------------------------------------------------------------------ #
#         # LAYER 4 — GEOMETRIC SIGNALS                                         #
#         # ------------------------------------------------------------------ #
#         face_width_ratio  = analysis.get("geo_face_width_ratio")
#         face_height_ratio = analysis.get("geo_face_height_ratio")

#         if face_width_ratio is not None and face_width_ratio < self.min_face_width_ratio:
#             return _reject(f"Face too narrow (geometric): {face_width_ratio:.3f}")

#         if face_height_ratio is not None and face_height_ratio < self.min_face_height_ratio:
#             return _reject(f"Face too short (geometric): {face_height_ratio:.3f}")

#         eye_dist_ratio = analysis.get("geo_eye_distance_ratio")
#         if eye_dist_ratio is not None and eye_dist_ratio < self.min_eye_distance_ratio:
#             return _reject(f"Eyes too close / one eye missing: eye_dist_ratio={eye_dist_ratio:.3f}")

#         left_ear  = analysis.get("geo_left_ear")
#         right_ear = analysis.get("geo_right_ear")

#         if left_ear is not None and left_ear < self.min_ear:
#             return _reject(f"Left eye geometrically closed: EAR={left_ear:.3f}")

#         if right_ear is not None and right_ear < self.min_ear:
#             return _reject(f"Right eye geometrically closed: EAR={right_ear:.3f}")

#         # EAR asymmetry threshold is profile-dependent.
#         # CCTV: 0.13 — 1px quantization at 80px face = 0.02-0.04 EAR noise per eye,
#         #              so two eyes can differ by 0.08-0.10 from noise alone.
#         # Webcam: 0.07 — high-res face, noise is negligible.
#         if left_ear is not None and right_ear is not None:
#             ear_diff = abs(left_ear - right_ear)
#             if ear_diff > self.ear_asymmetry_threshold:
#                 return _reject(
#                     f"EAR asymmetry (one eye occluded): "
#                     f"L={left_ear:.3f} R={right_ear:.3f} diff={ear_diff:.3f} "
#                     f"> {self.ear_asymmetry_threshold}"
#                 )

#         mouth_width_ratio = analysis.get("geo_mouth_width_ratio")
#         if mouth_width_ratio is not None and mouth_width_ratio < self.min_mouth_width_ratio:
#             return _reject(f"Mouth too narrow (occluded?): {mouth_width_ratio:.3f}")

#         lateral_asymmetry = analysis.get("geo_lateral_asymmetry")
#         if lateral_asymmetry is not None and lateral_asymmetry > self.max_lateral_asymmetry:
#             return _reject(f"Lateral asymmetry too high: {lateral_asymmetry:.3f}")

#         nose_offset = analysis.get("geo_nose_midline_offset")
#         if nose_offset is not None and nose_offset > 0.15:
#             return _reject(f"Nose offset from midline: {nose_offset:.3f}")

#         # ------------------------------------------------------------------ #
#         # LAYER 5 — IRIS SIGNALS                                              #
#         #                                                                     #
#         # Iris radius threshold is RELATIVE to face size (iris_radius_factor). #
#         # This makes it work correctly at both 80px CCTV and 300px webcam.   #
#         #                                                                     #
#         # Fixed absolute threshold (old approach) failed because:            #
#         #   - 4.0px threshold was correct for 250px webcam faces             #
#         #   - 80px CCTV faces have iris radius 2.0-3.5px naturally → all    #
#         #     clean faces were rejected as "occluded/degenerate"             #
#         # ------------------------------------------------------------------ #
#         min_iris_r = size * self.iris_radius_factor

#         left_r_px  = analysis.get("iris_left_radius_px")
#         right_r_px = analysis.get("iris_right_radius_px")

#         if left_r_px is not None and left_r_px < min_iris_r:
#             return _reject(
#                 f"Left iris radius too small: {left_r_px:.1f}px < {min_iris_r:.1f}px "
#                 f"(face={size}px × factor={self.iris_radius_factor})"
#             )

#         if right_r_px is not None and right_r_px < min_iris_r:
#             return _reject(
#                 f"Right iris radius too small: {right_r_px:.1f}px < {min_iris_r:.1f}px "
#                 f"(face={size}px × factor={self.iris_radius_factor})"
#             )

#         if left_r_px is not None and right_r_px is not None:
#             if left_r_px > 0 and right_r_px > 0:
#                 radius_ratio = min(left_r_px, right_r_px) / max(left_r_px, right_r_px)
#                 if radius_ratio < self.min_iris_radius_ratio:
#                     return _reject(
#                         f"Iris radius asymmetry (one eye occluded): "
#                         f"L={left_r_px:.1f}px R={right_r_px:.1f}px ratio={radius_ratio:.2f} "
#                         f"< {self.min_iris_radius_ratio}"
#                     )

#         left_brightness  = analysis.get("iris_left_center_brightness")
#         right_brightness = analysis.get("iris_right_center_brightness")

#         if left_brightness is not None and left_brightness > self.max_iris_center_brightness:
#             return _reject(
#                 f"Left iris center too bright (hand over eye): "
#                 f"brightness={left_brightness:.1f} > {self.max_iris_center_brightness}"
#             )

#         if right_brightness is not None and right_brightness > self.max_iris_center_brightness:
#             return _reject(
#                 f"Right iris center too bright (hand over eye): "
#                 f"brightness={right_brightness:.1f} > {self.max_iris_center_brightness}"
#             )

#         if left_brightness is not None and right_brightness is not None:
#             brightness_diff = abs(left_brightness - right_brightness)
#             if brightness_diff > self.max_iris_brightness_asymmetry:
#                 return _reject(
#                     f"Iris center brightness asymmetry (one eye occluded): "
#                     f"L={left_brightness:.1f} R={right_brightness:.1f} "
#                     f"diff={brightness_diff:.1f} > {self.max_iris_brightness_asymmetry}"
#                 )

#         # ------------------------------------------------------------------ #
#         # ACCEPT                                                               #
#         # ------------------------------------------------------------------ #
#         if debug:
#             prefix = f"[{cam_code}] " if cam_code else ""
#             print(f"[Gate] {prefix}✅ Accepted")
#         return True

#     # =========================================================================
#     # PRIVATE: iris signals
#     # =========================================================================

#     def _extract_iris_signals(
#         self, face_img: np.ndarray, lm_list, img_w: int, img_h: int
#     ) -> Dict:
#         gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

#         def _safe_lm(idx):
#             if idx >= len(lm_list):
#                 return None
#             p = lm_list[idx]
#             return int(p.x * img_w), int(p.y * img_h)

#         def _iris_signals(center_idx, edge_indices):
#             center_px = _safe_lm(center_idx)
#             if center_px is None:
#                 return None, None

#             cx, cy = center_px
#             if cx < 0 or cy < 0 or cx >= img_w or cy >= img_h:
#                 return None, None

#             radii = []
#             for idx in edge_indices:
#                 ep = _safe_lm(idx)
#                 if ep is not None:
#                     ex, ey = ep
#                     r = np.sqrt((ex - cx)**2 + (ey - cy)**2)
#                     radii.append(r)

#             radius_px = float(np.mean(radii)) if radii else None

#             sample_r = max(2, int((radius_px or 4) * 0.5))
#             y1 = max(0, cy - sample_r)
#             y2 = min(img_h, cy + sample_r + 1)
#             x1 = max(0, cx - sample_r)
#             x2 = min(img_w, cx + sample_r + 1)

#             if (y2 - y1) < 1 or (x2 - x1) < 1:
#                 return radius_px, None

#             patch_gray = gray[y1:y2, x1:x2].astype(np.float32)
#             ph, pw = patch_gray.shape
#             ys_g, xs_g = np.ogrid[:ph, :pw]
#             mask = (xs_g - pw//2)**2 + (ys_g - ph//2)**2 <= sample_r**2

#             center_brightness = (
#                 float(np.mean(patch_gray[mask])) if mask.sum() > 0
#                 else float(np.mean(patch_gray))
#             )

#             return radius_px, center_brightness

#         left_radius, left_brightness = _iris_signals(
#             _LM["left_iris_center"],
#             [_LM["left_iris_top"], _LM["left_iris_right"],
#              _LM["left_iris_bottom"], _LM["left_iris_left"]],
#         )
#         right_radius, right_brightness = _iris_signals(
#             _LM["right_iris_center"],
#             [_LM["right_iris_top"], _LM["right_iris_right"],
#              _LM["right_iris_bottom"], _LM["right_iris_left"]],
#         )

#         return {
#             "iris_left_radius_px":          left_radius,
#             "iris_right_radius_px":         right_radius,
#             "iris_left_center_brightness":  left_brightness,
#             "iris_right_center_brightness": right_brightness,
#         }

#     # =========================================================================
#     # PRIVATE: geometric signal extraction
#     # =========================================================================

#     def _extract_geometric_signals(self, lm_list, img_w: int, img_h: int) -> Dict:
#         def lm(key):
#             p = lm_list[_LM[key]]
#             return p.x, p.y

#         all_pts = np.array([[p.x, p.y] for p in lm_list], dtype=np.float32)
#         x_min, y_min = all_pts.min(axis=0)
#         x_max, y_max = all_pts.max(axis=0)
#         face_width_norm  = float(x_max - x_min)
#         face_height_norm = float(y_max - y_min)

#         le_outer = np.array(lm("left_eye_outer"))
#         le_inner = np.array(lm("left_eye_inner"))
#         le_upper = np.array(lm("left_eye_upper"))
#         le_lower = np.array(lm("left_eye_lower"))
#         re_outer = np.array(lm("right_eye_outer"))
#         re_inner = np.array(lm("right_eye_inner"))
#         re_upper = np.array(lm("right_eye_upper"))
#         re_lower = np.array(lm("right_eye_lower"))
#         nose    = np.array(lm("nose_tip"))
#         m_left  = np.array(lm("mouth_left"))
#         m_right = np.array(lm("mouth_right"))

#         left_eye_centre  = (le_outer + le_inner) / 2.0
#         right_eye_centre = (re_outer + re_inner) / 2.0
#         eye_dist_norm  = float(np.linalg.norm(right_eye_centre - left_eye_centre))
#         eye_dist_ratio = eye_dist_norm / (face_width_norm + 1e-6)

#         def _ear(outer, inner, upper, lower):
#             eye_w = float(np.linalg.norm(outer - inner)) + 1e-6
#             dy = abs(float(upper[1] - lower[1])) * img_h
#             dx = eye_w * img_w
#             return dy / (dx + 1e-6)

#         left_ear  = _ear(le_outer, le_inner, le_upper, le_lower)
#         right_ear = _ear(re_outer, re_inner, re_upper, re_lower)

#         mouth_width_norm  = float(np.linalg.norm(m_right - m_left))
#         mouth_width_ratio = mouth_width_norm / (eye_dist_norm + 1e-6)

#         face_mid_x = float((left_eye_centre[0] + right_eye_centre[0]) / 2.0)
#         left_span  = face_mid_x - x_min
#         right_span = x_max - face_mid_x
#         lateral_asymmetry   = abs(left_span - right_span) / (face_width_norm + 1e-6)
#         nose_midline_offset = abs(float(nose[0]) - face_mid_x) / (face_width_norm + 1e-6)

#         return {
#             "geo_face_width_ratio":    face_width_norm,
#             "geo_face_height_ratio":   face_height_norm,
#             "geo_eye_distance_ratio":  eye_dist_ratio,
#             "geo_left_ear":            float(left_ear),
#             "geo_right_ear":           float(right_ear),
#             "geo_mouth_width_ratio":   float(mouth_width_ratio),
#             "geo_lateral_asymmetry":   float(lateral_asymmetry),
#             "geo_nose_midline_offset": float(nose_midline_offset),
#             "geo_eye_dist_px":         float(eye_dist_norm * img_w),
#             "geo_face_width_px":       float(face_width_norm * img_w),
#             "geo_face_height_px":      float(face_height_norm * img_h),
#         }

#     # =========================================================================
#     # PRIVATE: helpers
#     # =========================================================================

#     def _debug_summary(self, a: Dict, cam_code: str = ""):
#         def _f(v, d, s):
#             return format(v if v is not None else d, s)

#         prefix = f"[{cam_code}] " if cam_code else ""
#         print(
#             f"[Gate] {prefix}"
#             f"blur={_f(a.get('blur'), 0.0, '.1f')}  "
#             f"size={a.get('size', 0)}px  "
#             f"yaw={_f(a.get('yaw'), 0.0, '.1f')}° "
#             f"pitch={_f(a.get('pitch'), 0.0, '.1f')}° "
#             f"roll={_f(a.get('roll'), 0.0, '.1f')}°  "
#             f"eye_score={_f(a.get('eye_score'), 0.0, '.2f')} "
#             f"eye_bal={_f(a.get('eye_balance'), 0.0, '.2f')}  "
#             f"upper={a.get('upper_face_visible')} "
#             f"lower={a.get('lower_face_visible')}  "
#             f"occ={_f(a.get('occlusion_score'), 0.0, '.2f')} "
#             f"mouth_occ={_f(a.get('mouth_occluded_score'), 0.0, '.2f')}  "
#             f"expr={_f(a.get('expression_score'), 0.0, '.2f')}  "
#             f"| GEO: "
#             f"w={_f(a.get('geo_face_width_ratio'), 0.0, '.2f')} "
#             f"h={_f(a.get('geo_face_height_ratio'), 0.0, '.2f')}  "
#             f"eye_dist={_f(a.get('geo_eye_distance_ratio'), 0.0, '.2f')}  "
#             f"L-EAR={_f(a.get('geo_left_ear'), 0.0, '.3f')} "
#             f"R-EAR={_f(a.get('geo_right_ear'), 0.0, '.3f')}  "
#             f"mouth_w={_f(a.get('geo_mouth_width_ratio'), 0.0, '.2f')}  "
#             f"lat_asym={_f(a.get('geo_lateral_asymmetry'), 0.0, '.3f')}  "
#             f"nose_off={_f(a.get('geo_nose_midline_offset'), 0.0, '.3f')}  "
#             f"| IRIS: "
#             f"L-r={_f(a.get('iris_left_radius_px'), 0.0, '.1f')}px "
#             f"R-r={_f(a.get('iris_right_radius_px'), 0.0, '.1f')}px  "
#             f"L-bright={_f(a.get('iris_left_center_brightness'), 0.0, '.1f')} "
#             f"R-bright={_f(a.get('iris_right_center_brightness'), 0.0, '.1f')}"
#         )

#     def _blendshape_dict(self, blendshapes):
#         if not blendshapes:
#             return None
#         return {b.category_name: b.score for b in blendshapes}

#     def _eyes_open(self, blend):
#         left  = blend.get("eyeBlinkLeft")
#         right = blend.get("eyeBlinkRight")
#         if left is None or right is None:
#             return False
#         return (left < self.eye_blink_threshold) and (right < self.eye_blink_threshold)

#     def _neutral_expression(self, blend):
#         smile = blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
#         mouth_open = blend.get("jawOpen", 0.0)
#         return smile < self.smile_threshold and mouth_open < self.mouth_open_threshold

#     def _is_frontal(self, matrix):
#         pose = self._extract_pose(matrix)
#         if pose is None:
#             return False
#         yaw, pitch, roll = pose
#         return (
#             abs(yaw)   < self.yaw_threshold
#             and abs(pitch) < self.pitch_threshold
#             and abs(roll)  < self.roll_threshold
#         )

#     def _is_blurry(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold

#     def _extract_pose(self, matrix):
#         try:
#             mat = np.array(matrix)
#             if mat.size != 16:
#                 return None
#             mat  = mat.reshape(4, 4)
#             rmat = mat[:3, :3]
#             sy   = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
#             singular = sy < 1e-6
#             if not singular:
#                 x = np.arctan2(rmat[2, 1], rmat[2, 2])
#                 y = np.arctan2(-rmat[2, 0], sy)
#                 z = np.arctan2(rmat[1, 0], rmat[0, 0])
#             else:
#                 x = np.arctan2(-rmat[1, 2], rmat[1, 1])
#                 y = np.arctan2(-rmat[2, 0], sy)
#                 z = 0
#             return np.degrees(y), np.degrees(x), np.degrees(z)
#         except Exception:
#             return None

#     def _to_mp_image(self, img):
#         return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)








import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from app.config.config import envConfig


# ---------------- LANDMARK INDEX ---------------- #
_LM = {
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "nose_tip": 1,
}


class FaceLandmarkerEngine:
    def __init__(
        self,
        model_path: str,
        blur_threshold: float = envConfig.BLUR_THRESHOLD,
        min_face_size: int = envConfig.MIN_FACE_SIZE,
        yaw_threshold: float = 35,
        pitch_threshold: float = 40,
        roll_threshold: float = 30,
        upscale_to: Optional[int] = 160,  # good for CCTV
    ):
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.upscale_to = upscale_to

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    # =========================================================
    # ANALYZE
    # =========================================================
    def analyze(self, face_img: np.ndarray) -> Dict:
        if face_img is None or face_img.size == 0:
            return {"valid": False, "reason": "empty"}

        h0, w0 = face_img.shape[:2]
        size = min(h0, w0)

        # -------- BLUR -------- #
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        # -------- UPSCALE (CCTV FIX) -------- #
        inference_img = face_img
        if self.upscale_to and size < self.upscale_to:
            scale = self.upscale_to / size
            inference_img = cv2.resize(
                face_img,
                (int(w0 * scale), int(h0 * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # -------- MEDIAPIPE -------- #
        rgb = cv2.cvtColor(inference_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if not result.face_landmarks:
            return {"valid": False, "reason": "no_landmarks", "blur": blur, "size": size}

        lm = result.face_landmarks[0]

        # -------- POSE -------- #
        matrix = (
            result.facial_transformation_matrixes[0]
            if result.facial_transformation_matrixes
            else None
        )
        yaw = pitch = roll = None
        if matrix is not None:
            pose = self._extract_pose(matrix)
            if pose:
                yaw, pitch, roll = pose

        # -------- GEOMETRY -------- #
        geo = self._extract_geometry(lm)

        return {
            "valid": True,
            "blur": blur,
            "size": size,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            **geo,
        }

    # =========================================================
    # GATE (MINIMAL + ROBUST)
    # =========================================================
    def is_valid_face(self, a: Dict, debug=False) -> bool:
        if not a or not a.get("valid"):
            return False

        # 1. Blur (hard reject)
        if a["blur"] < self.blur_threshold:
            print(f"Blur too low: {a['blur']:.2f} < {self.blur_threshold}")
            return False

        # 2. Size (hard reject)
        if a["size"] < self.min_face_size:
            print(f"Face too small: {a['size']}px < {self.min_face_size}px")
            return False

        # 3. Pose (loose)
        yaw, pitch = a.get("yaw"), a.get("pitch")
        if yaw is None or pitch is None:
            print(f"Invalid pose values: yaw={yaw}, pitch={pitch}")
            return False

        if abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold:
            print(f"Pose values out of bounds: yaw={yaw}, pitch={pitch}")

            return False

        # 4. Face completeness (geometry-based)
        face_area = a["face_width"] * a["face_height"]

        print(f"Face area: {face_area:.3f} (width={a['face_width']:.2f} height={a['face_height']:.2f})")

        if face_area < 0.06:
            print(f"Face area too small: {face_area:.3f} (width={a['face_width']:.2f} height={a['face_height']:.2f})")
            return False

        # 5. Partial face (eye distance)
        if a["eye_dist_ratio"] < 0.22:
            print(f"Eye distance ratio too low: {a['eye_dist_ratio']:.3f}")
            return False

        return True



    def score_face(self, a: Dict, debug=False) -> float:
        if not a or not a.get("valid"):
            return 0.0

        # -------------------------
        # HARD REJECT (only garbage)
        # -------------------------
        if a["blur"] < self.blur_threshold:
            if debug: print(f"❌ Blur too low: {a['blur']:.2f}")
            return 0.0

        if a["size"] < self.min_face_size:
            if debug: print(f"❌ Face too small: {a['size']}")
            return 0.0

        yaw, pitch = a.get("yaw"), a.get("pitch")
        if yaw is None or pitch is None:
            return 0.0

        if abs(yaw) > 60 or abs(pitch) > 60:
            # extreme pose → useless
            if debug: print(f"❌ Extreme pose: yaw={yaw}, pitch={pitch}")
            return 0.0

        # -------------------------
        # SOFT SCORING
        # -------------------------
        score = 1.0

        # 🔻 Pose penalty
        if abs(yaw) > 25:
            score *= 0.75
        if abs(pitch) > 30:
            score *= 0.75

        # # 🔻 Face completeness
        # face_area = a["face_width"] * a["face_height"]
        # if face_area < 0.10:
        #     score *= 0.7
        # elif face_area < 0.07:
        #     score *= 0.5

        # 🔻 Partial face (eyes missing / crop)
        eye_ratio = a.get("eye_dist_ratio", 1.0)
        if eye_ratio < 0.25:
            score *= 0.7
        if eye_ratio < 0.20:
            score *= 0.4

        # 🔻 Mild blur penalty (already passed hard threshold)
        blur = a.get("blur", 0)
        if blur < (self.blur_threshold * 1.5):
            score *= 0.8

        if debug:
            print(f"✅ Face score: {score:.2f}")

        return score
    # =========================================================
    # GEOMETRY
    # =========================================================
    def _extract_geometry(self, lm_list) -> Dict:
        pts = np.array([[p.x, p.y] for p in lm_list])

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)

        face_width = x_max - x_min
        face_height = y_max - y_min

        le = pts[_LM["left_eye_outer"]]
        re = pts[_LM["right_eye_outer"]]

        eye_dist = np.linalg.norm(re - le)
        eye_dist_ratio = eye_dist / (face_width + 1e-6)

        return {
            "face_width": float(face_width),
            "face_height": float(face_height),
            "eye_dist_ratio": float(eye_dist_ratio),
        }

    # =========================================================
    # POSE
    # =========================================================
    def _extract_pose(self, matrix):
        try:
            mat = np.array(matrix).reshape(4, 4)
            r = mat[:3, :3]

            sy = np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
            if sy > 1e-6:
                x = np.arctan2(r[2, 1], r[2, 2])
                y = np.arctan2(-r[2, 0], sy)
                z = np.arctan2(r[1, 0], r[0, 0])
            else:
                x = np.arctan2(-r[1, 2], r[1, 1])
                y = np.arctan2(-r[2, 0], sy)
                z = 0

            return np.degrees(y), np.degrees(x), np.degrees(z)
        except:
            return None