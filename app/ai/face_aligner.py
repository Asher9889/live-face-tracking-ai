import cv2
import numpy as np

# Standard ArcFace template (112x112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray, output_size: int = 112):
    """
    Align face using 5 landmarks.
    Returns aligned face (112x112).
    """

    if landmarks.shape != (5, 2):
        return None

    src = landmarks.astype(np.float32)
    dst = REFERENCE_LANDMARKS.copy()

    # Scale template if output size changes
    if output_size != 112:
        scale = output_size / 112
        dst *= scale

    # Estimate affine transform
    M, _ = cv2.estimateAffinePartial2D(src, dst)

    if M is None:
        return None

    aligned = cv2.warpAffine(
        image,
        M,
        (output_size, output_size),
        borderValue=0.0
    )

    return aligned