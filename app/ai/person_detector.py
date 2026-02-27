from ultralytics import YOLO
import numpy as np
import cv2

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", imgsz=960):
        self.model = YOLO(model_path)
        self.imgsz = imgsz

    def detect(self, frame):
        h, w = frame.shape[:2]
        frame_small = self._resize(frame)

        results = self.model(frame_small, conf=0.3, verbose=False)

        small_h, small_w = frame_small.shape[:2]
        scale_x = w / small_w
        scale_y = h / small_h

        boxes = []
        scores = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:
                    continue

                x1_s, y1_s, x2_s, y2_s = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])

                x1 = x1_s * scale_x
                y1 = y1_s * scale_y
                x2 = x2_s * scale_x
                y2 = y2_s * scale_y

                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        return boxes, scores
    
    def _resize(self, frame):
        h, w = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        if scale >= 1:
            return frame
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))