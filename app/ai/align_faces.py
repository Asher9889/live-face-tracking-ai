from typing import List
from app.ai.types import FaceCrop
from app.ai.face_aligner import align_face


def align_faces(face_crops: List[FaceCrop]) -> List[FaceCrop]:

    results = []

    for face in face_crops:

        aligned = align_face(face.crop, face.landmarks)

        if aligned is None:
            continue

        face.crop = aligned
        results.append(face)

    return results