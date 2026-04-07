# import time


# class TrackManager:

#     def __init__(self, publisher, gate_type):
#         self.publisher = publisher
#         self.gate_type = gate_type
#         self.tracks = {}

#     def update_track(self, cam_code, person_id, bbox, frame_ts, frame_w, frame_h):

#         now = time.time()

#         if person_id not in self.tracks:

#             self.tracks[person_id] = {
#                 "first_seen": now,
#                 "last_seen": now,
#                 "face_seen": False,
#                 "recognition_started": False,
#                 "recognized": False,
#                 "bbox": bbox.tolist(),
#                 "frameTs": frame_ts,
#                 "frame_width": frame_w,
#                 "frame_height": frame_h
#             }

#             self.publisher.publish(
#                 "track_created",
#                 {
#                     "camera_code": cam_code,
#                     "track_id": int(person_id),
#                     "bbox": bbox.tolist(),
#                     "frameTs": frame_ts,
#                     "frame_width": frame_w,
#                     "frame_height": frame_h,
#                     "eventTs": int(time.time() * 1000)
#                 }
#             )

#         else:

#             self.tracks[person_id]["last_seen"] = now

#             self.publisher.publish(
#                 "track_updated",
#                 {
#                     "camera_code": cam_code,
#                     "track_id": int(person_id),
#                     "bbox": bbox.tolist(),
#                     "frameTs": frame_ts,
#                     "frame_width": frame_w,
#                     "frame_height": frame_h,
#                     "eventTs": int(time.time() * 1000)
                    
#                 }
#             )

#     def face_detected(self, cam_code, person_id):

#         track = self.tracks.get(person_id)
#         person_id = int(person_id)

#         if not track:
#             return

#         if track["face_seen"]: 
#             return

#         if not track["face_seen"]:
#             track["face_seen"] = True

#             self.publisher.publish(
#                 "face_detected",
#                 {
#                     "camera_code": cam_code,
#                     "track_id": int(person_id)
#                 }
#             )

#     def recognition_pending(self, cam_code, person_id): 

#         track = self.tracks.get(person_id)
    
#         if not track:
#             return
        
#         if track["recognition_started"]:
#             return

#         track["recognition_started"] = True   

#         self.publisher.publish(
#             "recognition_pending",
#             {
#                 "camera_code": cam_code,
#                 "track_id": int(person_id)
#             }
#         )

#     def recognition_confirmed(self, cam_code, person_id, identityId, avg_similarity):

#         track = self.tracks.get(person_id)

#         if track["recognized"]:
#             return

#         if track:
#             track["recognized"] = True

#         confidence = self.compute_confidence(avg_similarity)
#         payload = {
#             "person_id": identityId,
#             "camera_code": cam_code,
#             "track_id": int(person_id),
#             "bbox": track["bbox"],
#             "frameTs": track["frameTs"],
#             "eventTs": int(time.time() * 1000),
#             "frame_width": track["frame_width"],
#             "frame_height": track["frame_height"],
#             "similarity": confidence
#         }

#         self.publisher.publish("recognition_confirmed", payload)

#         # publish entry/exit event
#         if self.gate_type == "ENTRY":
#             self.publisher.publish("person_entered", payload)

#         elif self.gate_type == "EXIT":
#             self.publisher.publish("person_exited", payload)

#     def unknown_confirmed(self, cam_code, person_id, unknown_id):

#         track = self.tracks.get(person_id)

#         if track["recognized"]:
#             return

#         if track:
#             track["recognized"] = True

#         self.publisher.publish(
#             "unknown_confirmed",
#             {
#                 "camera_code": cam_code,
#                 "track_id": int(person_id),
#                 "unknown_id": unknown_id
#             }
#         )

#     def cleanup_lost_tracks(self, cam_code, active_ids):

#         active_ids = set(active_ids)
#         now = time.time()
#         lost = []

#         for tid, track in list(self.tracks.items()):

#             # update last seen if still active
#             if tid in active_ids:
#                 track["last_seen"] = now
#                 continue

#             # check timeout
#             if now - track["last_seen"] < 2.0:  # 2 seconds grace period
#                 continue

#             lost.append(tid)

#             self.publisher.publish(
#                 "track_lost",
#                 {
#                     "camera_code": cam_code,
#                     "track_id": int(tid)
#                 }
#             )

#             del self.tracks[tid]

#         return lost

#     def compute_confidence(self, best_similarity, threshold=0.40, frames=3):
#         margin = best_similarity - threshold

#         if margin <= 0:
#             return 0

#         temporal_bonus = min(frames / 3, 1.0) * 2

#         confidence = threshold + margin * (1 + temporal_bonus)

#         confidence = min(confidence, 1.0)

#         return round(confidence * 100, 2)



import time


class TrackEventEmitter:
    """
    Lightweight event emitter for track lifecycle + recognition events.

    Responsibilities:
    - Track lifecycle (created, updated, lost)
    - Deduplicate events per track
    - Emit events via publisher

    Does NOT:
    - Handle recognition logic
    - Maintain business state
    """

    def __init__(self, publisher, gate_type):
        self.publisher = publisher
        self.gate_type = gate_type

        # Minimal state per track
        self.tracks = {}
        # {
        #   person_id: {
        #       "first_seen": float,
        #       "last_seen": float,
        #       "emitted_events": set()
        #   }
        # }

    # -----------------------------
    # INTERNAL: Emit once
    # -----------------------------
    def _emit_once(self, person_id, event_type, payload):
        track = self.tracks.get(person_id)
        if not track:
            return

        if event_type in track["emitted_events"]:
            return

        track["emitted_events"].add(event_type)
        self.publisher.publish(event_type, payload)

    # -----------------------------
    # TRACK LIFECYCLE
    # -----------------------------
    def update_track(self, cam_code, person_id, bbox, frame_ts, frame_w, frame_h):
        now = time.time()
        person_id = int(person_id)
        bbox_list = bbox.tolist()

        if person_id not in self.tracks:
            # NEW TRACK
            self.tracks[person_id] = {
                "first_seen": now,
                "last_seen": now,
                "emitted_events": set(),
                "bbox": bbox_list,
                "frameTs": frame_ts,
                "frame_width": frame_w,
                "frame_height": frame_h
            }

        else:
            # UPDATE TRACK
            self.tracks[person_id]["last_seen"] = now

        # Keep latest frame metadata for event payload contract.
        track = self.tracks[person_id]
        track["bbox"] = bbox_list
        track["frameTs"] = frame_ts
        track["frame_width"] = frame_w
        track["frame_height"] = frame_h

    # -----------------------------
    # FACE EVENT
    # -----------------------------
    def face_detected(self, cam_code, person_id):
        person_id = int(person_id)

        self._emit_once(
            person_id,
            "face_detected",
            {
                "camera_code": cam_code,
                "track_id": person_id,
                "eventTs": int(time.time() * 1000)
            }
        )

    # -----------------------------
    # RECOGNITION EVENTS
    # -----------------------------
    def recognition_pending(self, cam_code, person_id):
        person_id = int(person_id)

        self._emit_once(
            person_id,
            "recognition_pending",
            {
                "camera_code": cam_code,
                "track_id": person_id,
                "eventTs": int(time.time() * 1000)
            }
        )

    def recognition_confirmed(self, cam_code, person_id, identity_id, similarity):
        person_id = int(person_id)
        track = self.tracks.get(person_id)
        if not track:
            return

        payload = {
            "camera_code": cam_code,
            "track_id": person_id,
            "person_id": identity_id,
            "similarity": similarity,
            "bbox": track["bbox"],
            "frameTs": track["frameTs"],
            "frame_width": track["frame_width"],
            "frame_height": track["frame_height"],
            "eventTs": int(time.time() * 1000)
        }

        # Gate events (emit once as well)
        if self.gate_type == "ENTRY":
            self._emit_once(person_id, "person_entered", payload)

        elif self.gate_type == "EXIT":
            self._emit_once(person_id, "person_exited", payload)

    def unknown_confirmed(self, cam_code, person_id, unknown_id):
        person_id = int(person_id)
        track = self.tracks.get(person_id)
        if not track:
            return

        payload = {
            "camera_code": cam_code,
            "track_id": person_id,
            "person_id": unknown_id,
            "bbox": track["bbox"],
            "frameTs": track["frameTs"],
            "frame_width": track["frame_width"],
            "frame_height": track["frame_height"],
            "eventTs": int(time.time() * 1000)
        }

        if self.gate_type == "ENTRY":
            self._emit_once(person_id, "unknown_entered", payload)

        elif self.gate_type == "EXIT":
            self._emit_once(person_id, "unknown_exited", payload)

    # -----------------------------
    # TRACK CLEANUP
    # -----------------------------
    def cleanup_lost_tracks(self, cam_code, active_ids):
        active_ids = set(map(int, active_ids))
        now = time.time()
        lost = []

        for tid, track in list(self.tracks.items()):

            if tid in active_ids:
                track["last_seen"] = now
                continue

            # 2 sec grace
            if now - track["last_seen"] < 2.0:
                continue

            lost.append(tid)

            del self.tracks[tid]

        return lost

