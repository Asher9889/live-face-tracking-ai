

def process_track( person_id, cam, frame, frame_w, frame_h, roi, offset):
    """
    This function handles FULL lifecycle of one track:
    1. Known recognition
    2. Unknown matching / creation
    3. Unknown update
    """

    # ============================================================
    # INIT STATE
    # ============================================================
    if person_id not in track_state:
        track_state[person_id] = TrackState.COLLECTING_KNOWN

    state = track_state[person_id]

    # ============================================================
    # FACE DETECTION + QUALITY + STABILITY (COMMON STAGE)
    # ============================================================
    faces = insight_engine.detect_and_generate_embedding(roi, offset, cam.code)

    if len(faces) == 0:
        return

    # ----------------------------
    # QUALITY FILTERING
    # ----------------------------
    faces_with_quality = []

    for f in faces:
        # TODO: keep your existing quality + landmark logic
        # compute:
        # - face_img
        # - quality
        # - pose
        pass

    if not faces_with_quality:
        return

    best_face = select_best_face(faces_with_quality)
    if best_face is None:
        return

    embedding = best_face["embedding"]
    quality = best_face["quality"]
    pose_name = best_face.get("pose_bucket", "unknown")
    face_img = best_face["img"]

    # ----------------------------
    # STABILITY CHECK
    # ----------------------------
    if not is_stable_embedding(track_embedding_state, person_id, embedding, quality):
        return

    # ============================================================
    # 🔵 STAGE 1: KNOWN PIPELINE
    # ============================================================
    if state == TrackState.COLLECTING_KNOWN:

        buffer = track_known_buffer.get(person_id, [])

        # add entry
        buffer.append({
            "embedding": embedding,
            "quality": quality,
            "pose_bucket": pose_name,
            "img": face_img,
            "ts": time.time()
        })

        # keep top-3
        buffer = sorted(buffer, key=lambda x: x["quality"], reverse=True)[:3]
        track_known_buffer[person_id] = buffer

        if len(buffer) < 3:
            return

        # ----------------------------
        # BUILD EMBEDDING
        # ----------------------------
        # TODO: weighted average logic (use your existing code)
        final_embedding = build_weighted_embedding(buffer)

        # ----------------------------
        # MATCH KNOWN
        # ----------------------------
        match = embedding_store.find_match(final_embedding)

        if match:
            # -------- KNOWN MATCHED --------
            track_identity[person_id] = match["employee_id"]
            track_state[person_id] = TrackState.MATCHED_KNOWN

            track_manager.recognition_confirmed(
                cam.code,
                person_id,
                match["employee_id"],
                match["similarity"]
            )

            track_known_buffer.pop(person_id, None)
            return

        # -------- MOVE TO UNKNOWN PIPELINE --------
        track_state[person_id] = TrackState.COLLECTING_UNKNOWN

        # convert buffer → unknown buffer
        track_unknown_buffer[person_id] = buffer.copy()

        track_known_buffer.pop(person_id, None)
        return

    # ============================================================
    # 🔵 STAGE 2: UNKNOWN MATCH / CREATE (RUN ONCE)
    # ============================================================
    if state == TrackState.COLLECTING_UNKNOWN:

        buffer = track_unknown_buffer.get(person_id, [])

        buffer = uniqueFaceBuilder.add(
            buffer,
            embedding,
            quality,
            pose_name,
            img=face_img
        )

        track_unknown_buffer[person_id] = buffer

        if not uniqueFaceBuilder.is_ready(buffer):
            return

        centroid = uniqueFaceBuilder.build(buffer)
        if centroid is None:
            return

        # ----------------------------
        # SAFE IMAGE EXTRACTION
        # ----------------------------
        best = uniqueFaceBuilder.get_best_face(buffer)
        if not best or best["img"] is None:
            return

        if best["img"].size == 0:
            return

        ok, buffer_img = cv2.imencode(".jpg", best["img"])
        if not ok:
            return

        image_bytes = buffer_img.tobytes()
        poses = {x["pose_bucket"] for x in buffer}
        timestamp = int(time.time() * 1000)

        # ----------------------------
        # UNKNOWN MATCHING
        # ----------------------------
        unknown_match = unknown_embedding_store.find_match(centroid)

        if unknown_match:
            unknown_id = unknown_match["unknown_id"]
        else:
            if cam.camera_role != "REGISTER":
                return

            unknown_id = unknown_embedding_store.add_unknown(
                centroid=centroid,
                image_bytes=image_bytes,
                timestamp=timestamp,
                camera_code=cam.code,
                embedding_count=len(buffer),
                poses=poses
            )

        # store identity
        track_unknown_identity[person_id] = unknown_id

        track_manager.unknown_confirmed(
            cam.code,
            person_id,
            unknown_id
        )

        # init meta
        track_unknown_meta[person_id] = {
            "pose_best": {},
            "pose_count": 0
        }

        # 🔥 MOVE TO UPDATE MODE
        track_state[person_id] = TrackState.UPDATING_UNKNOWN
        return

    # ============================================================
    # 🔵 STAGE 3: UNKNOWN UPDATE (CONTINUOUS)
    # ============================================================
    if state == TrackState.UPDATING_UNKNOWN:

        unknown_id = track_unknown_identity.get(person_id)
        if not unknown_id:
            return

        buffer = track_unknown_buffer.get(person_id, [])

        buffer = uniqueFaceBuilder.add(
            buffer,
            embedding,
            quality,
            pose_name,
            img=face_img
        )

        track_unknown_buffer[person_id] = buffer

        if not uniqueFaceBuilder.is_ready(buffer):
            return

        centroid = uniqueFaceBuilder.build(buffer)
        best = uniqueFaceBuilder.get_best_face(buffer)

        if not best or best["img"] is None:
            return

        if best["img"].size == 0:
            return

        ok, buffer_img = cv2.imencode(".jpg", best["img"])
        if not ok:
            return

        image_bytes = buffer_img.tobytes()
        poses = {x["pose_bucket"] for x in buffer}
        timestamp = int(time.time() * 1000)

        # ----------------------------
        # POSE-BASED UPDATE LOGIC
        # ----------------------------
        meta = track_unknown_meta.get(person_id, {})
        pose_best = dict(meta.get("pose_best", {}))

        pose = best["pose_bucket"]
        best_quality = best["quality"]

        prev_quality = pose_best.get(pose, 0)

        is_better = best_quality > (prev_quality + 0.02)
        is_new_pose = pose not in pose_best

        if not (is_better or is_new_pose):
            return

        # ----------------------------
        # UPDATE UNKNOWN
        # ----------------------------
        unknown_embedding_store.update_unknown(
            unknown_id=unknown_id,
            centroid=centroid,
            timestamp=timestamp,
            camera_code=cam.code,
            image_bytes=image_bytes,
            poses=poses,
            quality=best_quality
        )

        # update meta
        pose_best[pose] = best_quality
        track_unknown_meta[person_id] = {
            "pose_best": pose_best,
            "pose_count": len(pose_best)
        }

        return