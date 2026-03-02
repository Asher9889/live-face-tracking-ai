import time


class TrackManager:

    def __init__(self, publisher):
        self.publisher = publisher
        self.tracks = {}

    def update_track(self, cam_code, person_id, bbox):

        now = time.time()

        if person_id not in self.tracks:

            self.tracks[person_id] = {
                "first_seen": now,
                "last_seen": now,
                "face_seen": False,
                "recognition_started": False,
                "recognized": False
            }

            self.publisher.publish(
                "track_created",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "bbox": bbox.tolist()
                }
            )

        else:

            self.tracks[person_id]["last_seen"] = now

            self.publisher.publish(
                "track_updated",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "bbox": bbox.tolist()
                }
            )

    def face_detected(self, cam_code, person_id):

        track = self.tracks.get(person_id)
        person_id = int(person_id)

        if not track:
            return

        if track["face_seen"]: 
            return

        if not track["face_seen"]:
            track["face_seen"] = True

            self.publisher.publish(
                "face_detected",
                {
                    "camera": cam_code,
                    "track_id": int(person_id)
                }
            )

    def recognition_pending(self, cam_code, person_id): 

        track = self.tracks.get(person_id)
    
        if not track:
            return
        
        if track["recognition_started"]:
            return

        track["recognition_started"] = True   

        self.publisher.publish(
            "recognition_pending",
            {
                "camera": cam_code,
                "track_id": int(person_id)
            }
        )

    def recognition_confirmed(self, cam_code, person_id, identity):

        track = self.tracks.get(person_id)

        if track["recognized"]:
            return

        if track:
            track["recognized"] = True

        self.publisher.publish(
            "recognition_confirmed",
            {
                "camera": cam_code,
                "track_id": int(person_id),
                "identity": identity
            }
        )

    def cleanup_lost_tracks(self, cam_code, active_ids):

        lost = []

        for tid in list(self.tracks.keys()):

            if tid not in active_ids:

                lost.append(tid)

                self.publisher.publish(
                    "track_lost",
                    {
                        "camera": cam_code,
                        "track_id": int(tid)
                    }
                )

                del self.tracks[tid]

        return lost