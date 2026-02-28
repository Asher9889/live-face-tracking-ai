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

    def detect_and_recognize(self, frame: np.ndarray, offset=(0, 0)):
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


# from insightface.app import FaceAnalysis
# import numpy as np


# class InsightFaceDetector:

#     MIN_FACE_SIZE = 30
#     MIN_SCORE = 0.5
#     # MAX_YAW = 45
#     # MAX_PITCH = 45

#     def __init__(self):
#         self.app = FaceAnalysis(
#             name="buffalo_l",
#             providers=["CUDAExecutionProvider"],
#         )

#         self.app.prepare(ctx_id=0, det_size=(640, 640))
#         print("[AI] InsightFace detector ready")

#     def nms(self, detections, iou_threshold=0.7):
#         if not detections:
#             return []

#         boxes = np.array([d["bbox"] for d in detections])
#         scores = np.array([d["score"] for d in detections])

#         x1 = boxes[:,0]
#         y1 = boxes[:,1]
#         x2 = boxes[:,2]
#         y2 = boxes[:,3]

#         areas = (x2 - x1) * (y2 - y1)
#         order = scores.argsort()[::-1]

#         keep = []

#         while order.size > 0:
#             i = order[0]
#             keep.append(detections[i])

#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])

#             w = np.maximum(0, xx2 - xx1)
#             h = np.maximum(0, yy2 - yy1)

#             inter = w * h
#             union = areas[i] + areas[order[1:]] - inter
#             iou = inter / (union + 1e-6)

#             inds = np.where(iou <= iou_threshold)[0]
#             order = order[inds + 1]

#         return keep

#     def detect(self, frame: np.ndarray):
#         faces = self.app.get(frame)

#         if not faces:
#             return []

#         results = []

#         for face in faces:
#             bbox = face.bbox.astype(np.int32)
#             score = float(face.det_score)

#             if score < self.MIN_SCORE: 
#                 continue

#             width = bbox[2] - bbox[0]
#             height = bbox[3] - bbox[1]
#             size = min(width, height)

#             if size < self.MIN_FACE_SIZE:
#                 continue

#             if face.pose is not None:
#                 yaw, pitch, roll = face.pose
#             else:
#                 yaw, pitch, roll = (0,0,0)

#             # if abs(yaw) > self.MAX_YAW or abs(pitch) > self.MAX_PITCH:
#             #     continue

#             age = int(face.age) if hasattr(face, "age") else None
#             gender = int(face.gender) if hasattr(face, "gender") else None

#             results.append({
#                 "bbox": bbox,
#                 "score": score,
#                 "landmarks": face.kps.astype(np.int32),
#                 "pose": (yaw, pitch, roll),
#                 "size": size,
#                 "area": width * height,
#                 "age": age,
#                 "gender": gender
#             })

#         if not results:
#             return []
#         return results
#         # return self.nms(results) # for avoiding duplicate faces for same bbox