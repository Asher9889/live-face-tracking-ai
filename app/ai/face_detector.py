import cv2
import numpy as np
import onnxruntime as ort

class FaceDetector:

    def __init__(self, model_path: str, conf_threshold=0.5, nms_threshold=0.4):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.input_size = (640, 640)

        print("[AI] SCRFD detector loaded")

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_size)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def infer(self, frame):
        h, w = frame.shape[:2]

        input_tensor = self.preprocess(frame)

        outputs = self.session.run(None, {
            self.input_name: input_tensor
        })

        return self.decode(outputs, (w, h))

    def decode(self, outputs, original_size):
        scores_list = outputs[:3]
        bbox_list = outputs[3:6]

        scores = np.concatenate([s.reshape(-1, 1) for s in scores_list])
        bboxes = np.concatenate([b.reshape(-1, 4) for b in bbox_list])

        # Filter by confidence
        mask = scores[:, 0] > self.conf_threshold
        scores = scores[mask]
        bboxes = bboxes[mask]

        if len(scores) == 0:
            return []

        # Convert to x1,y1,x2,y2
        boxes = bboxes.copy()

        # Scale to original size
        scale_x = original_size[0] / self.input_size[0]
        scale_y = original_size[1] / self.input_size[1]

        boxes[:, 0] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 2] *= scale_x
        boxes[:, 3] *= scale_y

        keep = self.nms(boxes, scores[:, 0])

        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "score": float(scores[idx])
            })

        return detections

    def nms(self, boxes, scores):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep
