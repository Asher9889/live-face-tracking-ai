from collections import defaultdict

def debug_tracking(detections):
    """
    Print detections grouped by camera and timestamp.
    """

    grouped = defaultdict(lambda: defaultdict(list))

    for d in detections:
        grouped[d.camera_code][d.timestamp].append(d)

    for cam_code, ts_group in grouped.items():
        print(f"\n[TRACK DEBUG] Camera={cam_code}")

        for ts in sorted(ts_group.keys()):
            frame_dets = ts_group[ts]

            ids = [d.track_id for d in frame_dets]
            bboxes = [d.bbox.tolist() for d in frame_dets]

            print(
                f"ts={ts:.3f} "
                f"dets={len(frame_dets)} "
                f"ids={ids} "
                f"bbox={[d.bbox.tolist() for d in frame_dets]}"
            )