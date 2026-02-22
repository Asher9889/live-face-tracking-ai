import cv2
import numpy as np


def is_blurry(image, threshold=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < threshold


def is_small(bbox, min_size=60):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return min(w, h) < min_size


def bad_pose(pose, max_angle=45):
    yaw, pitch, _ = pose
    return abs(yaw) > max_angle or abs(pitch) > max_angle