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

    # convert to int
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # expand box slightly
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    # crop ROI
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return person_id, roi, (x1, y1)