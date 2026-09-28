def extract_person_roi(frame, person_id, bbox, pad_x=8, pad_y=20):
    """
    Extract person region of interest from frame for face detection.

    Args:
        frame: full camera frame (H,W,3)
        person_id: tracking id from YOLO
        bbox: [x1,y1,x2,y2]
        pad_x: horizontal padding. Keep small: a wide pad pulls a standing
               neighbour's face into this track's ROI.
        pad_y: vertical padding, for head room above the box

    Returns:
        person_id
        roi_image
        (offset_x, offset_y) -> needed to remap SCRFD bbox
    """

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    # pads may be fractional, so convert to int AFTER padding is applied
    pad_x = int(pad_x)
    pad_y = int(pad_y)

    # expand box slightly
    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    # crop ROI
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return person_id, roi, (x1, y1)