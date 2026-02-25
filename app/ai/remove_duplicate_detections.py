def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0

def remove_duplicate_detections(detections, threshold=0.8):

    filtered = []

    for d in detections:
        duplicate = False

        for f in filtered:
            if iou(d.bbox, f.bbox) > threshold:
                duplicate = True
                break

        if not duplicate:
            filtered.append(d)

    return filtered