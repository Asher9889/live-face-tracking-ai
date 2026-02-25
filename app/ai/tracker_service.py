import supervision as sv
import numpy as np
from collections import defaultdict
from app.camera.types import Detection


class ByteTrackerService:
    def __init__(self):
        self.trackers: dict[str, sv.ByteTrack] = {}

    def _get_tracker(self, camera_code: str):
        if camera_code not in self.trackers:
            self.trackers[camera_code] = sv.ByteTrack()
        return self.trackers[camera_code]

    def update_sequential(self, detections: list[Detection]) -> list[Detection]:

        # Group by camera
        by_camera = defaultdict(list)
        for d in detections:
            by_camera[d.camera_code].append(d)

        results = []

        # Process each camera independently
        for cam_code, dets in by_camera.items():

            tracker = self._get_tracker(cam_code)

            # Group by timestamp (frame)
            by_ts = defaultdict(list)
            for d in dets:
                by_ts[d.timestamp].append(d)

            # Sort timestamps
            sorted_ts = sorted(by_ts.keys())

            # Sequential update
            for ts in sorted_ts:

                frame_dets = by_ts[ts]

                xyxy = np.array([d.bbox for d in frame_dets])
                confidence = np.array([d.score for d in frame_dets])
                class_id = np.zeros(len(frame_dets))

                sv_dets = sv.Detections(
                    xyxy=xyxy,
                    confidence=confidence,
                    class_id=class_id
                )

                tracked = tracker.update_with_detections(sv_dets)

                for det, tid in zip(frame_dets, tracked.tracker_id):
                    det.track_id = int(tid) if tid is not None else None
                    results.append(det)

        return results