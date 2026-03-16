import time


class TrackManager:

    def __init__(self, publisher, gate_type):
        self.publisher = publisher
        self.gate_type = gate_type
        self.tracks = {}

    def update_track(self, cam_code, person_id, bbox, frame_ts, frame_w, frame_h):

        now = time.time()

        if person_id not in self.tracks:

            self.tracks[person_id] = {
                "first_seen": now,
                "last_seen": now,
                "face_seen": False,
                "recognition_started": False,
                "recognized": False,
                "bbox": bbox.tolist(),
                "frameTs": frame_ts,
                "frame_width": frame_w,
                "frame_height": frame_h
            }

            self.publisher.publish(
                "track_created",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "bbox": bbox.tolist(),
                    "frameTs": frame_ts,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "eventTs": int(time.time() * 1000)
                }
            )

        else:

            self.tracks[person_id]["last_seen"] = now

            self.publisher.publish(
                "track_updated",
                {
                    "camera": cam_code,
                    "track_id": int(person_id),
                    "bbox": bbox.tolist(),
                    "frameTs": frame_ts,
                    "frame_width": frame_w,
                    "frame_height": frame_h,
                    "eventTs": int(time.time() * 1000)
                    
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

    def recognition_confirmed(self, cam_code, person_id, identityId):

        track = self.tracks.get(person_id)

        if track["recognized"]:
            return

        if track:
            track["recognized"] = True

        payload = {
            "person_id": identityId,
            "camera_code": cam_code,
            "track_id": int(person_id),
            "bbox": track["bbox"],
            "frameTs": track["frameTs"],
            "eventTs": int(time.time() * 1000),
            "frame_width": track["frame_width"],
            "frame_height": track["frame_height"]
        }

        self.publisher.publish("recognition_confirmed", payload)

        # publish entry/exit event
        if self.gate_type == "ENTRY":
            self.publisher.publish("person_entered", payload)

        elif self.gate_type == "EXIT":
            self.publisher.publish("person_exited", payload)

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

