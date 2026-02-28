def extract_person_roi(frame, person_id, bbox, pad=40):
    """
    Extract person region of interest from frame for face detection.

    Args:
        frame: full camera frame (H,W,3)
        person_id: tracking id from YOLO
        bbox: [x1,y1,x2,y2]
        pad: padding around bbox to avoid cutting head

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
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    # crop ROI
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return person_id, roi, (x1, y1)