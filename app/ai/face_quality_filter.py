def is_good_face(face):

    score = face["score"]
    yaw, pitch, roll = face["pose"]

    if score < 0.35:
        return False

    if abs(yaw) > 35:
        return False

    if abs(pitch) > 25: 
        return False

    if abs(roll) > 20:
        return False

    return True