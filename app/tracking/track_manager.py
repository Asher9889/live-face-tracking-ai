import time


class TrackManager:

    def __init__(self, publisher, gate_type):
        self.publisher = publisher
        self.gate_type = gate_type
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

        # publish entry/exit event
        if self.gate_type == "ENTRY":
            self.publisher.publish(
                "person_entered",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "identity": identity
                }
            )

        elif self.gate_type == "EXIT":
            self.publisher.publish(
                "person_exited",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "identity": identity
                }
            )

    def unknown_confirmed(self, cam_code, person_id, unknown_id):

        track = self.tracks.get(person_id)

        if track["recognized"]:
            return

        if track:
            track["recognized"] = True

        self.publisher.publish(
            "unknown_confirmed",
            {
                "camera": cam_code,
                "track_id": int(person_id),
                "unknown_id": unknown_id
            }
        )

    def cleanup_lost_tracks(self, cam_code, active_ids):

        active_ids = set(active_ids)
        now = time.time()
        lost = []

        for tid, track in list(self.tracks.items()):

            # update last seen if still active
            if tid in active_ids:
                track["last_seen"] = now
                continue

            # check timeout
            if now - track["last_seen"] < 2.0:  # 2 seconds grace period
                continue

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

