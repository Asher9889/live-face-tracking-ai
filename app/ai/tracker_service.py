import supervision as sv
import numpy as np


class ByteTrackerService:
    def __init__(
        self,
        frame_rate: int = 15,   # MUST match actual processing FPS
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 60,     # slightly more tolerant
        minimum_matching_threshold: float = 0.3
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

    def update(self, boxes, scores):
        """
        boxes: list or np.ndarray of shape (N,4)
        scores: list or np.ndarray of shape (N,)
        """

        # Ensure numpy arrays
        if boxes is None or len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            boxes = np.asarray(boxes, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)

            # Safety: ensure valid shape
            if boxes.ndim != 2 or boxes.shape[1] != 4:
                raise ValueError("Boxes must be (N,4)")

            detections = sv.Detections(
                xyxy=boxes,
                confidence=scores,
                class_id=np.zeros(len(boxes), dtype=np.int32)
            )

        tracked = self.tracker.update_with_detections(detections)

        results = []

        for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue

            results.append((int(tid), bbox.astype(np.float32)))

        return results

    def reset(self):
        self.tracker.reset()