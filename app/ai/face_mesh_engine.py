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
# # Left eye: outer corner=33, inner corner=133, upper lid=159, lower lid=145
# # Right eye: outer corner=263, inner corner=362, upper lid=386, lower lid=374
# # Nose tip: 1
# # Mouth: left corner=61, right corner=291, upper lip=13, lower lip=14
# # Chin: 152
# # Left cheek: 234
# # Right cheek: 454
# # Left eyebrow outer: 70
# # Right eyebrow outer: 300
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
# }


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
#         yaw_threshold: float = 20,
#         pitch_threshold: float = 25,
#         roll_threshold: float = 20,
#         # ---- image quality ----
#         blur_threshold: float = envConfig.BLUR_THRESHOLD,
#         min_face_size: int = envConfig.MIN_FACE_SIZE,
#         # ---- geometric thresholds (relative, 0–1 in landmark space) ----
#         # Minimum eye distance relative to face width.
#         # If one eye is blocked the inter-eye distance collapses dramatically.
#         min_eye_distance_ratio: float = 0.25,
#         # Minimum face height relative to image height (landmark space).
#         # Catches heavily-cropped / top-of-head only faces.
#         min_face_height_ratio: float = 0.30,
#         # Minimum face width relative to image width (landmark space).
#         min_face_width_ratio: float = 0.25,
#         # Maximum allowed asymmetry between left and right landmark spans.
#         # E.g. if left side landmarks go 0.1–0.5 and right side only 0.5–0.55
#         # the face is heavily occluded on one side.
#         max_lateral_asymmetry: float = 0.20,
#         # Minimum vertical eye openness (EAR – Eye Aspect Ratio).
#         # EAR = (upper_lid_y – lower_lid_y) / eye_width.  Closed eye → ~0.
#         min_ear: float = 0.10,
#         # Minimum mouth width relative to inter-eye distance.
#         # A hand covering the mouth squashes this ratio.
#         min_mouth_width_ratio: float = 0.60,
#     ):
#         self.eye_blink_threshold = eye_blink_threshold
#         self.smile_threshold = smile_threshold
#         self.mouth_open_threshold = mouth_open_threshold
#         self.yaw_threshold = yaw_threshold
#         self.pitch_threshold = pitch_threshold
#         self.roll_threshold = roll_threshold
#         self.blur_threshold = blur_threshold
#         self.min_face_size = min_face_size

#         # geometric gate params
#         self.min_eye_distance_ratio = min_eye_distance_ratio
#         self.min_face_height_ratio = min_face_height_ratio
#         self.min_face_width_ratio = min_face_width_ratio
#         self.max_lateral_asymmetry = max_lateral_asymmetry
#         self.min_ear = min_ear
#         self.min_mouth_width_ratio = min_mouth_width_ratio

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
#     # PUBLIC: analyze()  —  signal extraction only, no hard rejections
#     # =========================================================================

#     def analyze(self, face_img: np.ndarray, debug: bool = False) -> Dict:
#         """
#         Extract all quality signals from a face crop.
#         Returns a flat dict.  Decision logic lives entirely in is_valid_face().
#         """
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
#             frontal_score = max(
#                 0.0,
#                 1.0 - (abs(yaw) / 40.0 + abs(pitch) / 30.0) / 2.0,
#             )

#         pose_magnitude = None
#         if yaw is not None and pitch is not None and roll is not None:
#             pose_magnitude = float(np.sqrt(yaw**2 + pitch**2 + roll**2))

#         # Blendshape region visibility
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
#         # 6. GEOMETRIC SIGNALS  (NEW)
#         #    All landmark coordinates are normalised to [0, 1] by MediaPipe.
#         #    We work in this normalised space so results are image-size
#         #    independent.  We still report pixel-space values where useful.
#         # ------------------------------------------------------------------
#         lm_list = result.face_landmarks[0]   # list of NormalizedLandmark

#         geo = self._extract_geometric_signals(lm_list, img_w=w, img_h=h)

#         # ------------------------------------------------------------------
#         # 7. ASSEMBLE OUTPUT
#         # ------------------------------------------------------------------
#         return {
#             "valid": True,

#             # image quality
#             "blur": blur,
#             "brightness": brightness,
#             "contrast": contrast,
#             "size": size,
#             "aspect_ratio": aspect_ratio,

#             # pose
#             "yaw": yaw,
#             "pitch": pitch,
#             "roll": roll,
#             "pose_magnitude": pose_magnitude,

#             # blendshape-derived
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

#             # ---- geometric signals (new) ----
#             **geo,

#             # raw
#             "blendshapes": blend_dict,
#             "matrix": matrix,
#         }

#     # =========================================================================
#     # PUBLIC: is_valid_face()  —  gating only, no signal extraction
#     # =========================================================================

#     def is_valid_face(self, analysis: Dict, debug: bool = False) -> bool:
#         """
#         Multi-layer quality gate.  Returns True only when all checks pass.
#         Each rejection prints a labelled reason when debug=True.
#         """

#         def _reject(reason: str) -> bool:
#             if debug:
#                 print(f"[Gate] ❌ {reason}")
#             return False

#         if analysis is None or not analysis.get("valid"):
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
#             return _reject(f"Asymmetric blink detected (one eye occluded?): eye_balance={eye_balance:.2f}")

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
#         # LAYER 4 — GEOMETRIC SIGNALS  (NEW — catches hand/mask occlusion)   #
#         # ------------------------------------------------------------------ #

#         # 4a. Face bounding box must be large enough (relative to landmark span)
#         face_width_ratio = analysis.get("geo_face_width_ratio")
#         face_height_ratio = analysis.get("geo_face_height_ratio")

#         if face_width_ratio is not None and face_width_ratio < self.min_face_width_ratio:
#             return _reject(
#                 f"Face too narrow (geometric): geo_face_width_ratio={face_width_ratio:.3f} "
#                 f"< {self.min_face_width_ratio}"
#             )

#         if face_height_ratio is not None and face_height_ratio < self.min_face_height_ratio:
#             return _reject(
#                 f"Face too short (geometric): geo_face_height_ratio={face_height_ratio:.3f} "
#                 f"< {self.min_face_height_ratio}"
#             )

#         # 4b. Inter-eye distance must be reasonable relative to face width.
#         #     When one eye is hidden the eye-centre shifts dramatically.
#         eye_dist_ratio = analysis.get("geo_eye_distance_ratio")
#         if eye_dist_ratio is not None and eye_dist_ratio < self.min_eye_distance_ratio:
#             return _reject(
#                 f"Eyes too close / one eye missing (geometric): "
#                 f"geo_eye_distance_ratio={eye_dist_ratio:.3f} < {self.min_eye_distance_ratio}"
#             )

#         # 4c. Eye Aspect Ratio (EAR) — geometric measure of eye openness.
#         #     More reliable than blendshapes for detecting physical occlusion.
#         left_ear = analysis.get("geo_left_ear")
#         right_ear = analysis.get("geo_right_ear")

#         if left_ear is not None and left_ear < self.min_ear:
#             return _reject(f"Left eye geometrically closed/occluded: EAR={left_ear:.3f} < {self.min_ear}")

#         if right_ear is not None and right_ear < self.min_ear:
#             return _reject(f"Right eye geometrically closed/occluded: EAR={right_ear:.3f} < {self.min_ear}")

#         # 4d. EAR asymmetry — one eye open, one blocked by hand/object.
#         if left_ear is not None and right_ear is not None:
#             ear_asymmetry = abs(left_ear - right_ear)
#             ear_asymmetry_threshold = 0.07   # tighter than blink blendshape
#             if ear_asymmetry > ear_asymmetry_threshold:
#                 return _reject(
#                     f"EAR asymmetry too high (one eye occluded): "
#                     f"left={left_ear:.3f}, right={right_ear:.3f}, diff={ear_asymmetry:.3f}"
#                 )

#         # 4e. Mouth width relative to inter-eye distance.
#         #     A hand covering the mouth shrinks the visible mouth width.
#         mouth_width_ratio = analysis.get("geo_mouth_width_ratio")
#         if mouth_width_ratio is not None and mouth_width_ratio < self.min_mouth_width_ratio:
#             return _reject(
#                 f"Mouth too narrow (occluded?): "
#                 f"geo_mouth_width_ratio={mouth_width_ratio:.3f} < {self.min_mouth_width_ratio}"
#             )

#         # 4f. Lateral symmetry — left half vs right half landmark span.
#         #     Occlusion on one side collapses the landmark density there.
#         lateral_asymmetry = analysis.get("geo_lateral_asymmetry")
#         if lateral_asymmetry is not None and lateral_asymmetry > self.max_lateral_asymmetry:
#             return _reject(
#                 f"Lateral asymmetry too high (partial face?): "
#                 f"geo_lateral_asymmetry={lateral_asymmetry:.3f} > {self.max_lateral_asymmetry}"
#             )

#         # 4g. Nose-to-midline offset: if nose tip is far from the horizontal
#         #     midpoint of the eye-line the face is partially cropped / occluded.
#         nose_offset = analysis.get("geo_nose_midline_offset")
#         if nose_offset is not None and nose_offset > 0.15:
#             return _reject(
#                 f"Nose offset from midline too large (asymmetric crop/occlusion): "
#                 f"geo_nose_midline_offset={nose_offset:.3f}"
#             )

#         # ------------------------------------------------------------------ #
#         # ACCEPT                                                               #
#         # ------------------------------------------------------------------ #
#         if debug:
#             print("[Gate] ✅ Accepted")
#         return True

#     # =========================================================================
#     # PRIVATE: geometric signal extraction
#     # =========================================================================

#     def _extract_geometric_signals(self, lm_list, img_w: int, img_h: int) -> Dict:
#         """
#         Compute lightweight geometric signals from the 478-point FaceMesh.

#         All raw landmark coordinates are normalised to [0,1].
#         Ratios are dimensionless and image-size independent.
#         Pixel distances are also included for debugging.
#         """
#         # Convenience accessor — returns (x_norm, y_norm) tuple.
#         def lm(key):
#             idx = _LM[key]
#             p = lm_list[idx]
#             return p.x, p.y  # normalised [0,1]

#         # All landmark positions as numpy array (N, 2)
#         all_pts = np.array([[p.x, p.y] for p in lm_list], dtype=np.float32)

#         # ---- Face bounding box (normalised) --------------------------------
#         x_min, y_min = all_pts.min(axis=0)
#         x_max, y_max = all_pts.max(axis=0)
#         face_width_norm  = float(x_max - x_min)   # fraction of image width
#         face_height_norm = float(y_max - y_min)   # fraction of image height

#         # ---- Key points ----------------------------------------------------
#         le_outer = np.array(lm("left_eye_outer"))
#         le_inner = np.array(lm("left_eye_inner"))
#         le_upper = np.array(lm("left_eye_upper"))
#         le_lower = np.array(lm("left_eye_lower"))

#         re_outer = np.array(lm("right_eye_outer"))
#         re_inner = np.array(lm("right_eye_inner"))
#         re_upper = np.array(lm("right_eye_upper"))
#         re_lower = np.array(lm("right_eye_lower"))

#         nose   = np.array(lm("nose_tip"))
#         m_left = np.array(lm("mouth_left"))
#         m_right= np.array(lm("mouth_right"))

#         # Left eye centre & right eye centre
#         left_eye_centre  = (le_outer + le_inner) / 2.0
#         right_eye_centre = (re_outer + re_inner) / 2.0

#         # ---- Inter-eye distance --------------------------------------------
#         eye_dist_norm = float(np.linalg.norm(right_eye_centre - left_eye_centre))
#         # Ratio: inter-eye dist / face width.  Collapses when one eye hidden.
#         eye_dist_ratio = eye_dist_norm / (face_width_norm + 1e-6)

#         # ---- Eye Aspect Ratio (EAR) ----------------------------------------
#         # EAR = vertical opening / horizontal eye width
#         # Uses the 4 corner / lid landmarks already extracted.
#         def _ear(outer, inner, upper, lower):
#             eye_w = float(np.linalg.norm(outer - inner)) + 1e-6
#             # Correct for image aspect ratio so EAR is meaningful
#             dy = abs(float(upper[1] - lower[1])) * img_h
#             dx = eye_w * img_w
#             return dy / (dx + 1e-6)

#         left_ear  = _ear(le_outer, le_inner, le_upper, le_lower)
#         right_ear = _ear(re_outer, re_inner, re_upper, re_lower)

#         # ---- Mouth width ratio ---------------------------------------------
#         mouth_width_norm  = float(np.linalg.norm(m_right - m_left))
#         # Normalise against inter-eye distance (scale-invariant)
#         mouth_width_ratio = mouth_width_norm / (eye_dist_norm + 1e-6)

#         # ---- Lateral symmetry ----------------------------------------------
#         # Midpoint of face (x) = midpoint of the two eye centres.
#         face_mid_x = float((left_eye_centre[0] + right_eye_centre[0]) / 2.0)
#         # Left half span: from x_min to face_mid_x
#         # Right half span: from face_mid_x to x_max
#         left_span  = face_mid_x - x_min
#         right_span = x_max - face_mid_x
#         # Asymmetry = |left_span - right_span| / face_width
#         # → 0 for symmetric face, → 1 for fully one-sided.
#         lateral_asymmetry = abs(left_span - right_span) / (face_width_norm + 1e-6)

#         # ---- Nose offset from eye midline ----------------------------------
#         # How far the nose tip x deviates from the eye-midpoint x.
#         # Normalised against face width.
#         nose_midline_offset = abs(float(nose[0]) - face_mid_x) / (face_width_norm + 1e-6)

#         return {
#             # Face bounding box (normalised to image size)
#             "geo_face_width_ratio":     face_width_norm,
#             "geo_face_height_ratio":    face_height_norm,

#             # Inter-eye distance relative to face width
#             "geo_eye_distance_ratio":   eye_dist_ratio,

#             # Per-eye EAR
#             "geo_left_ear":             float(left_ear),
#             "geo_right_ear":            float(right_ear),

#             # Mouth width relative to inter-eye distance
#             "geo_mouth_width_ratio":    float(mouth_width_ratio),

#             # Lateral symmetry (0 = perfect, 1 = half-face missing)
#             "geo_lateral_asymmetry":    float(lateral_asymmetry),

#             # Nose deviation from eye midline
#             "geo_nose_midline_offset":  float(nose_midline_offset),

#             # Pixel-space convenience values (for logging / debugging)
#             "geo_eye_dist_px":          float(eye_dist_norm * img_w),
#             "geo_face_width_px":        float(face_width_norm * img_w),
#             "geo_face_height_px":       float(face_height_norm * img_h),
#         }

#     # =========================================================================
#     # PRIVATE: helpers
#     # =========================================================================

#     def _debug_summary(self, a: Dict):
#         """Print a compact one-line summary of all gate-relevant signals."""
#         print(
#             f"[Gate] "
#             f"blur={a.get('blur', 0):.1f}  "
#             f"size={a.get('size', 0)}px  "
#             f"yaw={a.get('yaw', 0):.1f}° pitch={a.get('pitch', 0):.1f}° roll={a.get('roll', 0):.1f}°  "
#             f"eye_score={a.get('eye_score', 0):.2f} eye_bal={a.get('eye_balance', 0):.2f}  "
#             f"upper={a.get('upper_face_visible')} lower={a.get('lower_face_visible')}  "
#             f"completeness={a.get('face_completeness', 0):.2f}  "
#             f"occ={a.get('occlusion_score', 0):.2f} mouth_occ={a.get('mouth_occluded_score', 0):.2f}  "
#             f"expr={a.get('expression_score', 0):.2f}  "
#             f"| GEO: "
#             f"w={a.get('geo_face_width_ratio', 0):.2f} h={a.get('geo_face_height_ratio', 0):.2f}  "
#             f"eye_dist={a.get('geo_eye_distance_ratio', 0):.2f}  "
#             f"L-EAR={a.get('geo_left_ear', 0):.3f} R-EAR={a.get('geo_right_ear', 0):.3f}  "
#             f"mouth_w={a.get('geo_mouth_width_ratio', 0):.2f}  "
#             f"lat_asym={a.get('geo_lateral_asymmetry', 0):.3f}  "
#             f"nose_off={a.get('geo_nose_midline_offset', 0):.3f}"
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
#             return np.degrees(y), np.degrees(x), np.degrees(z)   # yaw, pitch, roll
#         except Exception:
#             return None

#     def _to_mp_image(self, img):
#         return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)












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


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices (478-point model)
# ---------------------------------------------------------------------------
_LM = {
    "left_eye_outer":     33,
    "left_eye_inner":     133,
    "left_eye_upper":     159,
    "left_eye_lower":     145,
    "right_eye_outer":    263,
    "right_eye_inner":    362,
    "right_eye_upper":    386,
    "right_eye_lower":    374,
    "nose_tip":           1,
    "mouth_left":         61,
    "mouth_right":        291,
    "mouth_upper":        13,
    "mouth_lower":        14,
    "chin":               152,
    "left_cheek":         234,
    "right_cheek":        454,
    "left_brow_outer":    70,
    "right_brow_outer":   300,
    # Iris landmarks (478-point model only)
    # Left iris:  center=468, edge points=469,470,471,472
    # Right iris: center=473, edge points=474,475,476,477
    "left_iris_center":   468,
    "left_iris_top":      469,
    "left_iris_right":    470,
    "left_iris_bottom":   471,
    "left_iris_left":     472,
    "right_iris_center":  473,
    "right_iris_top":     474,
    "right_iris_right":   475,
    "right_iris_bottom":  476,
    "right_iris_left":    477,
}

_EYE_PATCH_PAD = 1.8   # wider crop to fully capture iris + surround


class FaceLandmarkerEngine:
    def __init__(
        self,
        model_path: str,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        # ---- expression / blink thresholds ----
        eye_blink_threshold: float = 0.4,
        smile_threshold: float = 0.6,
        mouth_open_threshold: float = 0.4,
        # ---- pose thresholds (degrees) ----
        yaw_threshold: float = 30,
        pitch_threshold: float = 25,
        roll_threshold: float = 25,
        # ---- image quality ----
        blur_threshold: float = envConfig.BLUR_THRESHOLD,
        min_face_size: int = envConfig.MIN_FACE_SIZE,
        # ---- geometric thresholds ----
        min_eye_distance_ratio: float = 0.25,
        min_face_height_ratio: float = 0.30,
        min_face_width_ratio: float = 0.25,
        max_lateral_asymmetry: float = 0.20,
        min_ear: float = 0.10,
        min_mouth_width_ratio: float = 0.60,
        # ------------------------------------------------------------------
        # IRIS-BASED OCCLUSION THRESHOLDS (replaces broken skin detector)
        # ------------------------------------------------------------------
        # Iris radius is computed from the 5 iris landmarks.
        # Minimum iris radius in pixels. When hand covers eye, MediaPipe
        # projects a tiny/degenerate iris → radius collapses.
        # For a ~250px face crop, real iris radius ≈ 8-18px.
        # Tune: print tex_left_iris_radius_px on your camera.
        min_iris_radius_px: float = 4.0,
        # Maximum asymmetry ratio between left/right iris radii.
        # A hand over one eye degenerates that iris → small radius vs normal other.
        # ratio = min(left_r, right_r) / max(left_r, right_r)
        # Real symmetric face: ratio > 0.6.  One eye occluded: ratio < 0.4.
        min_iris_radius_ratio: float = 0.45,
        # Iris center darkness: the iris center pixel should be DARK (pupil).
        # Mean grayscale value at a small disk around iris center.
        # Real iris center ≈ 30-80 (dark).
        # Hand skin over it ≈ 100-200 (medium/bright).
        # If the iris center region is brighter than this, something is in the way.
        max_iris_center_brightness: float = 140.0,
        # Maximum brightness ASYMMETRY between the two iris centers.
        # If one iris center is much brighter than the other → that eye is occluded.
        # Absolute pixel difference.
        max_iris_brightness_asymmetry: float = 60.0,
    ):
        self.eye_blink_threshold = eye_blink_threshold
        self.smile_threshold = smile_threshold
        self.mouth_open_threshold = mouth_open_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self.blur_threshold = blur_threshold
        self.min_face_size = min_face_size
        self.min_eye_distance_ratio = min_eye_distance_ratio
        self.min_face_height_ratio = min_face_height_ratio
        self.min_face_width_ratio = min_face_width_ratio
        self.max_lateral_asymmetry = max_lateral_asymmetry
        self.min_ear = min_ear
        self.min_mouth_width_ratio = min_mouth_width_ratio

        # iris params
        self.min_iris_radius_px = min_iris_radius_px
        self.min_iris_radius_ratio = min_iris_radius_ratio
        self.max_iris_center_brightness = max_iris_center_brightness
        self.max_iris_brightness_asymmetry = max_iris_brightness_asymmetry

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

    # =========================================================================
    # PUBLIC: analyze()
    # =========================================================================

    def analyze(self, face_img: np.ndarray, debug: bool = False) -> Dict:
        if face_img is None or face_img.size == 0:
            return {"valid": False, "reason": "empty"}

        h, w = face_img.shape[:2]
        size = min(h, w)
        aspect_ratio = w / (h + 1e-6)

        # ------------------------------------------------------------------
        # 1. IMAGE QUALITY
        # ------------------------------------------------------------------
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        # ------------------------------------------------------------------
        # 2. MEDIAPIPE INFERENCE
        # ------------------------------------------------------------------
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        mp_image = self._to_mp_image(rgb)
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return {
                "valid": False,
                "reason": "no_landmarks",
                "blur": blur,
                "brightness": brightness,
                "contrast": contrast,
                "size": size,
            }

        # ------------------------------------------------------------------
        # 3. BLENDSHAPES
        # ------------------------------------------------------------------
        blend_dict = None
        if result.face_blendshapes:
            blend_dict = {
                b.category_name: float(b.score)
                for b in result.face_blendshapes[0]
            }

        # ------------------------------------------------------------------
        # 4. POSE
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 5. BLENDSHAPE-DERIVED SIGNALS
        # ------------------------------------------------------------------
        eye_score = eye_balance = None
        if blend_dict:
            lb = blend_dict.get("eyeBlinkLeft", 0.0)
            rb = blend_dict.get("eyeBlinkRight", 0.0)
            eye_score = 1.0 - max(lb, rb)
            eye_balance = abs(lb - rb)

        expression_score = 0.0
        if blend_dict:
            smile = blend_dict.get("mouthSmileLeft", 0.0) + blend_dict.get("mouthSmileRight", 0.0)
            mouth_open = blend_dict.get("jawOpen", 0.0)
            expression_score = max(smile, mouth_open)

        occlusion_score = 0.0
        if blend_dict:
            occlusion_score = max(
                blend_dict.get("cheekPuff", 0.0),
                blend_dict.get("eyeSquintLeft", 0.0),
                blend_dict.get("eyeSquintRight", 0.0),
            )

        frontal_score = None
        if yaw is not None and pitch is not None:
            frontal_score = max(0.0, 1.0 - (abs(yaw) / 40.0 + abs(pitch) / 30.0) / 2.0)

        pose_magnitude = None
        if yaw is not None and pitch is not None and roll is not None:
            pose_magnitude = float(np.sqrt(yaw**2 + pitch**2 + roll**2))

        upper_face_visible = lower_face_visible = False
        face_completeness = mouth_occluded_score = 0.0
        eye_visibility_score = None

        if blend_dict:
            eye_visibility_score = 1.0 - max(
                blend_dict.get("eyeBlinkLeft", 0.0),
                blend_dict.get("eyeBlinkRight", 0.0),
                blend_dict.get("eyeSquintLeft", 0.0),
                blend_dict.get("eyeSquintRight", 0.0),
            )
            upper_face_visible = (
                eye_visibility_score > 0.4
                and eye_balance is not None
                and eye_balance < 0.4
            )
            jaw = blend_dict.get("jawOpen", 0.0)
            mouth_press = (
                blend_dict.get("mouthPressLeft", 0.0)
                + blend_dict.get("mouthPressRight", 0.0)
            )
            mouth_shrug = blend_dict.get("mouthShrugLower", 0.0)
            lower_face_visible = jaw < 0.4 and mouth_press < 0.4 and mouth_shrug < 0.3
            mouth_occluded_score = max(blend_dict.get("cheekPuff", 0.0), mouth_press)
            face_completeness = (int(upper_face_visible) + int(lower_face_visible)) / 2.0

        # ------------------------------------------------------------------
        # 6. GEOMETRIC SIGNALS
        # ------------------------------------------------------------------
        lm_list = result.face_landmarks[0]
        geo = self._extract_geometric_signals(lm_list, img_w=w, img_h=h)

        # ------------------------------------------------------------------
        # 7. IRIS SIGNALS  (replaces broken skin detector)
        # ------------------------------------------------------------------
        iris = self._extract_iris_signals(face_img, lm_list, img_w=w, img_h=h)

        return {
            "valid": True,
            "blur": blur,
            "brightness": brightness,
            "contrast": contrast,
            "size": size,
            "aspect_ratio": aspect_ratio,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "pose_magnitude": pose_magnitude,
            "eye_score": eye_score,
            "eye_balance": eye_balance,
            "expression_score": expression_score,
            "frontal_score": frontal_score,
            "occlusion_score": occlusion_score,
            "upper_face_visible": upper_face_visible,
            "lower_face_visible": lower_face_visible,
            "face_completeness": face_completeness,
            "mouth_occluded_score": mouth_occluded_score,
            "eye_visibility_score": eye_visibility_score,
            **geo,
            **iris,
            "blendshapes": blend_dict,
            "matrix": matrix,
        }

    # =========================================================================
    # PUBLIC: is_valid_face()
    # =========================================================================

    def is_valid_face(self, analysis: Dict, debug: bool = True) -> bool:

        def _reject(reason: str) -> bool:
            if debug:
                print(f"[Gate] ❌ {reason}")
            return False

        if analysis is None:
            return _reject("Invalid analysis object: None")

        if not analysis.get("valid"):
            return _reject(f"Invalid analysis object: {analysis.get('reason', 'unknown')}")

        if debug:
            self._debug_summary(analysis)

        # ------------------------------------------------------------------ #
        # LAYER 1 — IMAGE QUALITY                                             #
        # ------------------------------------------------------------------ #
        blur = analysis.get("blur", 0)
        size = analysis.get("size", 0)

        if blur < self.blur_threshold:
            return _reject(f"Blur too low: {blur:.2f} < {self.blur_threshold}")

        if size < self.min_face_size:
            return _reject(f"Face too small: {size}px < {self.min_face_size}px")

        # ------------------------------------------------------------------ #
        # LAYER 2 — POSE                                                      #
        # ------------------------------------------------------------------ #
        yaw = analysis.get("yaw")
        pitch = analysis.get("pitch")
        roll = analysis.get("roll")

        if yaw is None or pitch is None or roll is None:
            return _reject("Missing pose angles")

        if abs(yaw) > self.yaw_threshold:
            return _reject(f"Yaw out of range: {yaw:.1f}° (max {self.yaw_threshold}°)")

        if abs(pitch) > self.pitch_threshold:
            return _reject(f"Pitch out of range: {pitch:.1f}° (max {self.pitch_threshold}°)")

        if abs(roll) > self.roll_threshold:
            return _reject(f"Roll out of range: {roll:.1f}° (max {self.roll_threshold}°)")

        # ------------------------------------------------------------------ #
        # LAYER 3 — BLENDSHAPE EYE / EXPRESSION / OCCLUSION                  #
        # ------------------------------------------------------------------ #
        eye_score = analysis.get("eye_score")
        eye_balance = analysis.get("eye_balance")

        if eye_score is None or eye_score < 0.6:
            return _reject(f"Eyes not sufficiently open (blendshape): eye_score={eye_score}")

        if eye_balance is not None and eye_balance > 0.4:
            return _reject(f"Asymmetric blink (blendshape): eye_balance={eye_balance:.2f}")

        if not analysis.get("upper_face_visible", False):
            return _reject("Upper face not visible (blendshape check)")

        if not analysis.get("lower_face_visible", False):
            return _reject("Lower face not visible (blendshape check)")

        if analysis.get("face_completeness", 0) < 1.0:
            return _reject(f"Partial face (blendshape): completeness={analysis.get('face_completeness'):.2f}")

        if analysis.get("occlusion_score", 0) > 0.5:
            return _reject(f"Face occluded (blendshape): occlusion_score={analysis.get('occlusion_score'):.2f}")

        if analysis.get("mouth_occluded_score", 0) > 0.5:
            return _reject(f"Mouth occluded (blendshape): mouth_occluded_score={analysis.get('mouth_occluded_score'):.2f}")

        if analysis.get("expression_score", 0) > 0.6:
            return _reject(f"Expression too strong: expression_score={analysis.get('expression_score'):.2f}")

        # ------------------------------------------------------------------ #
        # LAYER 4 — GEOMETRIC SIGNALS                                         #
        # ------------------------------------------------------------------ #
        face_width_ratio = analysis.get("geo_face_width_ratio")
        face_height_ratio = analysis.get("geo_face_height_ratio")

        if face_width_ratio is not None and face_width_ratio < self.min_face_width_ratio:
            return _reject(f"Face too narrow (geometric): {face_width_ratio:.3f}")

        if face_height_ratio is not None and face_height_ratio < self.min_face_height_ratio:
            return _reject(f"Face too short (geometric): {face_height_ratio:.3f}")

        eye_dist_ratio = analysis.get("geo_eye_distance_ratio")
        if eye_dist_ratio is not None and eye_dist_ratio < self.min_eye_distance_ratio:
            return _reject(f"Eyes too close / one eye missing: eye_dist_ratio={eye_dist_ratio:.3f}")

        left_ear = analysis.get("geo_left_ear")
        right_ear = analysis.get("geo_right_ear")

        if left_ear is not None and left_ear < self.min_ear:
            return _reject(f"Left eye geometrically closed: EAR={left_ear:.3f}")

        if right_ear is not None and right_ear < self.min_ear:
            return _reject(f"Right eye geometrically closed: EAR={right_ear:.3f}")

        if left_ear is not None and right_ear is not None:
            if abs(left_ear - right_ear) > 0.07:
                return _reject(
                    f"EAR asymmetry (one eye occluded): "
                    f"L={left_ear:.3f} R={right_ear:.3f} diff={abs(left_ear-right_ear):.3f}"
                )

        mouth_width_ratio = analysis.get("geo_mouth_width_ratio")
        if mouth_width_ratio is not None and mouth_width_ratio < self.min_mouth_width_ratio:
            return _reject(f"Mouth too narrow (occluded?): {mouth_width_ratio:.3f}")

        lateral_asymmetry = analysis.get("geo_lateral_asymmetry")
        if lateral_asymmetry is not None and lateral_asymmetry > self.max_lateral_asymmetry:
            return _reject(f"Lateral asymmetry too high: {lateral_asymmetry:.3f}")

        nose_offset = analysis.get("geo_nose_midline_offset")
        if nose_offset is not None and nose_offset > 0.15:
            return _reject(f"Nose offset from midline: {nose_offset:.3f}")

        # ------------------------------------------------------------------ #
        # LAYER 5 — IRIS SIGNALS                                              #
        #                                                                     #
        # WHY IRIS INSTEAD OF SKIN DETECTION:                                 #
        # Skin-based detection fails for dark/medium skin tones and glasses   #
        # wearers because the periocular region looks skin-like regardless.   #
        #                                                                     #
        # Iris landmarks ARE reliable because:                                #
        # 1. Iris radius — MediaPipe fits a circle to iris edge points.       #
        #    When a hand occludes the eye, the model can't fit the iris       #
        #    properly → radius collapses or becomes degenerate.               #
        # 2. Iris center brightness — the pupil is always dark.               #
        #    A hand over the eye makes the iris center region bright (skin).  #
        #    This is SKIN-TONE INDEPENDENT: dark skin is still brighter       #
        #    than a pupil.                                                     #
        # 3. Brightness asymmetry — if one iris center is much brighter       #
        #    than the other, that eye is covered regardless of skin tone.     #
        # ------------------------------------------------------------------ #

        left_r_px  = analysis.get("iris_left_radius_px")
        right_r_px = analysis.get("iris_right_radius_px")

        # 5a. Per-iris radius must exceed minimum
        if left_r_px is not None and left_r_px < self.min_iris_radius_px:
            return _reject(
                f"Left iris radius too small (eye occluded/degenerate): "
                f"radius={left_r_px:.1f}px < {self.min_iris_radius_px}px"
            )

        if right_r_px is not None and right_r_px < self.min_iris_radius_px:
            return _reject(
                f"Right iris radius too small (eye occluded/degenerate): "
                f"radius={right_r_px:.1f}px < {self.min_iris_radius_px}px"
            )

        # 5b. Iris radius asymmetry
        if left_r_px is not None and right_r_px is not None:
            if left_r_px > 0 and right_r_px > 0:
                radius_ratio = min(left_r_px, right_r_px) / max(left_r_px, right_r_px)
                if radius_ratio < self.min_iris_radius_ratio:
                    return _reject(
                        f"Iris radius asymmetry (one eye occluded): "
                        f"L={left_r_px:.1f}px R={right_r_px:.1f}px ratio={radius_ratio:.2f} < {self.min_iris_radius_ratio}"
                    )

        # 5c. Iris center brightness — pupil must be dark
        left_brightness  = analysis.get("iris_left_center_brightness")
        right_brightness = analysis.get("iris_right_center_brightness")

        if left_brightness is not None and left_brightness > self.max_iris_center_brightness:
            return _reject(
                f"Left iris center too bright (hand over eye — skin covering pupil): "
                f"brightness={left_brightness:.1f} > {self.max_iris_center_brightness}"
            )

        if right_brightness is not None and right_brightness > self.max_iris_center_brightness:
            return _reject(
                f"Right iris center too bright (hand over eye — skin covering pupil): "
                f"brightness={right_brightness:.1f} > {self.max_iris_center_brightness}"
            )

        # 5d. Brightness asymmetry between iris centers
        if left_brightness is not None and right_brightness is not None:
            brightness_diff = abs(left_brightness - right_brightness)
            if brightness_diff > self.max_iris_brightness_asymmetry:
                return _reject(
                    f"Iris center brightness asymmetry (one eye occluded): "
                    f"L={left_brightness:.1f} R={right_brightness:.1f} diff={brightness_diff:.1f} > {self.max_iris_brightness_asymmetry}"
                )

        # ------------------------------------------------------------------ #
        # ACCEPT                                                               #
        # ------------------------------------------------------------------ #
        if debug:
            print("[Gate] ✅ Accepted")
        return True

    # =========================================================================
    # PRIVATE: iris signals
    # =========================================================================

    def _extract_iris_signals(
        self, face_img: np.ndarray, lm_list, img_w: int, img_h: int
    ) -> Dict:
        """
        Extract iris-based occlusion signals using MediaPipe's iris landmarks.

        The 478-point model includes 5 iris landmarks per eye:
          Left:  center=468, top=469, right=470, bottom=471, left=472
          Right: center=473, top=474, right=475, bottom=476, left=477

        Signals:
          1. iris_radius_px — mean distance from center to 4 edge points (pixels)
             Real open eye: 8-20px depending on face size.
             Occluded/degenerate: < 4px.

          2. iris_center_brightness — mean grayscale in a small disk at iris center.
             Real pupil: 20-80 (very dark).
             Hand/skin over eye: 100-200 (medium bright).
             This is SKIN-TONE INDEPENDENT — even dark skin is lighter than a pupil.

        Returns dict with iris_left_* and iris_right_* keys.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        def _safe_lm(idx):
            """Get landmark as pixel coordinate, None if index out of range."""
            if idx >= len(lm_list):
                return None
            p = lm_list[idx]
            return int(p.x * img_w), int(p.y * img_h)

        def _iris_signals(center_idx, edge_indices):
            center_px = _safe_lm(center_idx)
            if center_px is None:
                return None, None

            cx, cy = center_px
            if cx < 0 or cy < 0 or cx >= img_w or cy >= img_h:
                return None, None

            # Compute radius from edge landmarks
            radii = []
            for idx in edge_indices:
                ep = _safe_lm(idx)
                if ep is not None:
                    ex, ey = ep
                    r = np.sqrt((ex - cx)**2 + (ey - cy)**2)
                    radii.append(r)

            radius_px = float(np.mean(radii)) if radii else None

            # Sample brightness in a small disk at iris center
            # Radius = half the iris radius (just the pupil zone)
            sample_r = max(2, int((radius_px or 4) * 0.5))
            y1 = max(0, cy - sample_r)
            y2 = min(img_h, cy + sample_r + 1)
            x1 = max(0, cx - sample_r)
            x2 = min(img_w, cx + sample_r + 1)

            if (y2 - y1) < 1 or (x2 - x1) < 1:
                return radius_px, None

            # Create circular mask
            patch_gray = gray[y1:y2, x1:x2].astype(np.float32)
            ph, pw = patch_gray.shape
            ys_g, xs_g = np.ogrid[:ph, :pw]
            mask = (xs_g - pw//2)**2 + (ys_g - ph//2)**2 <= sample_r**2

            if mask.sum() == 0:
                center_brightness = float(np.mean(patch_gray))
            else:
                center_brightness = float(np.mean(patch_gray[mask]))

            return radius_px, center_brightness

        left_radius, left_brightness = _iris_signals(
            _LM["left_iris_center"],
            [_LM["left_iris_top"], _LM["left_iris_right"],
             _LM["left_iris_bottom"], _LM["left_iris_left"]],
        )

        right_radius, right_brightness = _iris_signals(
            _LM["right_iris_center"],
            [_LM["right_iris_top"], _LM["right_iris_right"],
             _LM["right_iris_bottom"], _LM["right_iris_left"]],
        )

        return {
            "iris_left_radius_px":         left_radius,
            "iris_right_radius_px":        right_radius,
            "iris_left_center_brightness": left_brightness,
            "iris_right_center_brightness":right_brightness,
        }

    # =========================================================================
    # PRIVATE: geometric signal extraction
    # =========================================================================

    def _extract_geometric_signals(self, lm_list, img_w: int, img_h: int) -> Dict:
        def lm(key):
            p = lm_list[_LM[key]]
            return p.x, p.y

        all_pts = np.array([[p.x, p.y] for p in lm_list], dtype=np.float32)
        x_min, y_min = all_pts.min(axis=0)
        x_max, y_max = all_pts.max(axis=0)
        face_width_norm  = float(x_max - x_min)
        face_height_norm = float(y_max - y_min)

        le_outer = np.array(lm("left_eye_outer"))
        le_inner = np.array(lm("left_eye_inner"))
        le_upper = np.array(lm("left_eye_upper"))
        le_lower = np.array(lm("left_eye_lower"))
        re_outer = np.array(lm("right_eye_outer"))
        re_inner = np.array(lm("right_eye_inner"))
        re_upper = np.array(lm("right_eye_upper"))
        re_lower = np.array(lm("right_eye_lower"))
        nose    = np.array(lm("nose_tip"))
        m_left  = np.array(lm("mouth_left"))
        m_right = np.array(lm("mouth_right"))

        left_eye_centre  = (le_outer + le_inner) / 2.0
        right_eye_centre = (re_outer + re_inner) / 2.0
        eye_dist_norm  = float(np.linalg.norm(right_eye_centre - left_eye_centre))
        eye_dist_ratio = eye_dist_norm / (face_width_norm + 1e-6)

        def _ear(outer, inner, upper, lower):
            eye_w = float(np.linalg.norm(outer - inner)) + 1e-6
            dy = abs(float(upper[1] - lower[1])) * img_h
            dx = eye_w * img_w
            return dy / (dx + 1e-6)

        left_ear  = _ear(le_outer, le_inner, le_upper, le_lower)
        right_ear = _ear(re_outer, re_inner, re_upper, re_lower)

        mouth_width_norm  = float(np.linalg.norm(m_right - m_left))
        mouth_width_ratio = mouth_width_norm / (eye_dist_norm + 1e-6)

        face_mid_x = float((left_eye_centre[0] + right_eye_centre[0]) / 2.0)
        left_span  = face_mid_x - x_min
        right_span = x_max - face_mid_x
        lateral_asymmetry   = abs(left_span - right_span) / (face_width_norm + 1e-6)
        nose_midline_offset = abs(float(nose[0]) - face_mid_x) / (face_width_norm + 1e-6)

        return {
            "geo_face_width_ratio":    face_width_norm,
            "geo_face_height_ratio":   face_height_norm,
            "geo_eye_distance_ratio":  eye_dist_ratio,
            "geo_left_ear":            float(left_ear),
            "geo_right_ear":           float(right_ear),
            "geo_mouth_width_ratio":   float(mouth_width_ratio),
            "geo_lateral_asymmetry":   float(lateral_asymmetry),
            "geo_nose_midline_offset": float(nose_midline_offset),
            "geo_eye_dist_px":         float(eye_dist_norm * img_w),
            "geo_face_width_px":       float(face_width_norm * img_w),
            "geo_face_height_px":      float(face_height_norm * img_h),
        }

    # =========================================================================
    # PRIVATE: helpers
    # =========================================================================

    def _debug_summary(self, a: Dict):
        def _fmt(value, default, spec):
            if value is None:
                value = default
            return format(value, spec)

        print(
            f"[Gate] "
            f"blur={_fmt(a.get('blur'), 0.0, '.1f')}  "
            f"size={a.get('size', 0)}px  "
            f"yaw={_fmt(a.get('yaw'), 0.0, '.1f')}° pitch={_fmt(a.get('pitch'), 0.0, '.1f')}° roll={_fmt(a.get('roll'), 0.0, '.1f')}°  "
            f"eye_score={_fmt(a.get('eye_score'), 0.0, '.2f')} eye_bal={_fmt(a.get('eye_balance'), 0.0, '.2f')}  "
            f"upper={a.get('upper_face_visible')} lower={a.get('lower_face_visible')}  "
            f"occ={_fmt(a.get('occlusion_score'), 0.0, '.2f')} mouth_occ={_fmt(a.get('mouth_occluded_score'), 0.0, '.2f')}  "
            f"expr={_fmt(a.get('expression_score'), 0.0, '.2f')}  "
            f"| GEO: "
            f"w={_fmt(a.get('geo_face_width_ratio'), 0.0, '.2f')} h={_fmt(a.get('geo_face_height_ratio'), 0.0, '.2f')}  "
            f"eye_dist={_fmt(a.get('geo_eye_distance_ratio'), 0.0, '.2f')}  "
            f"L-EAR={_fmt(a.get('geo_left_ear'), 0.0, '.3f')} R-EAR={_fmt(a.get('geo_right_ear'), 0.0, '.3f')}  "
            f"mouth_w={_fmt(a.get('geo_mouth_width_ratio'), 0.0, '.2f')}  "
            f"lat_asym={_fmt(a.get('geo_lateral_asymmetry'), 0.0, '.3f')}  "
            f"nose_off={_fmt(a.get('geo_nose_midline_offset'), 0.0, '.3f')}  "
            f"| IRIS: "
            f"L-r={_fmt(a.get('iris_left_radius_px'), 0.0, '.1f')}px "
            f"R-r={_fmt(a.get('iris_right_radius_px'), 0.0, '.1f')}px  "
            f"L-bright={_fmt(a.get('iris_left_center_brightness'), 0.0, '.1f')} "
            f"R-bright={_fmt(a.get('iris_right_center_brightness'), 0.0, '.1f')}"
        )

    def _blendshape_dict(self, blendshapes):
        if not blendshapes:
            return None
        return {b.category_name: b.score for b in blendshapes}

    def _eyes_open(self, blend):
        left  = blend.get("eyeBlinkLeft")
        right = blend.get("eyeBlinkRight")
        if left is None or right is None:
            return False
        return (left < self.eye_blink_threshold) and (right < self.eye_blink_threshold)

    def _neutral_expression(self, blend):
        smile = blend.get("mouthSmileLeft", 0.0) + blend.get("mouthSmileRight", 0.0)
        mouth_open = blend.get("jawOpen", 0.0)
        return smile < self.smile_threshold and mouth_open < self.mouth_open_threshold

    def _is_frontal(self, matrix):
        pose = self._extract_pose(matrix)
        if pose is None:
            return False
        yaw, pitch, roll = pose
        return abs(yaw) < self.yaw_threshold and abs(pitch) < self.pitch_threshold and abs(roll) < self.roll_threshold

    def _is_blurry(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < self.blur_threshold

    def _extract_pose(self, matrix):
        try:
            mat = np.array(matrix)
            if mat.size != 16:
                return None
            mat  = mat.reshape(4, 4)
            rmat = mat[:3, :3]
            sy   = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0
            return np.degrees(y), np.degrees(x), np.degrees(z)
        except Exception:
            return None

    def _to_mp_image(self, img):
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=img)







