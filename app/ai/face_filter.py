from app.ai.face_quality import is_blurry, is_small, bad_pose
from app.ai.face_cropper import crop_face
from app.ai.types import FaceCrop


def filter_and_crop(detections):

    results = []

    for d in detections:

        if is_small(d.bbox):
            continue

        if bad_pose(d.pose):
            continue

        crop = crop_face(d.frame, d.bbox)

        if is_blurry(crop):
            continue

        results.append(
            FaceCrop(
                camera_code=d.camera_code,
                timestamp=d.timestamp,
                crop=crop,
                bbox=d.bbox,
                landmarks=d.landmarks
            )
        )

    return results