import supervision as sv
import numpy as np


class ByteTrackerService:
    """
    Per-camera tracker wrapper.
    One instance per camera thread.
    """

    def __init__(
        self,
        frame_rate: int = 15,
        track_activation_threshold: float = 0.20, 
        lost_track_buffer: int = 30,  # 2 seconds at 15 FPS 
        minimum_matching_threshold: float = 0.2
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        
        # Track IDs from previous frame to detect losses
        self.previous_ids = set()

    def update(self, boxes, scores):
        """
        Update tracker with new detections.
        
        Args:
            boxes: np.ndarray of shape (N, 4) with [x1, y1, x2, y2] format
            scores: np.ndarray of shape (N,) with confidence scores
            
        Returns:
            tracks: List of (track_id, bbox) tuples
            lost_ids: Set of track IDs that were removed this frame
        """
        
        # Handle empty detections
        if boxes is None or len(boxes) == 0:
            detections = sv.Detections.empty()
        else:
            # Ensure correct dtypes
            detections = sv.Detections(
                xyxy=boxes.astype(np.float32),
                confidence=scores.astype(np.float32),
                class_id=np.zeros(len(boxes), dtype=np.int32)
            )

        # Update tracker
        tracked = self.tracker.update_with_detections(detections)

        # Extract active tracks
        results = []
        current_ids = set()

        for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
            if tid is None:
                continue
            
            tid = int(tid)
            current_ids.add(tid)
            
            # Keep bbox as float32 for precision
            results.append((tid, bbox))

        # Calculate which tracks were lost this frame
        # lost_ids = self.previous_ids - current_ids
        
        # # Update state for next frame
        # self.previous_ids = current_ids.copy()

        # return results, lost_ids

        # Update state for next frame
        self.previous_ids = current_ids.copy()

        # Do NOT manually compute lost_ids
        return results, set()
    
    def reset(self):
        """Reset tracker state (e.g., when reconnecting to camera)"""
        self.tracker.reset()
        self.previous_ids = set()