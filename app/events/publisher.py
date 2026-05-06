import json
import time


ALLOWED_EVENTS = {
    "person_entered",
    "person_exited",
    "unknown_entered",
    "unknown_exited",
    "face_detected",     
    "recognition_pending", 
}

# Events that use streams for persistence/history
STREAM_EVENTS = {
    "person_entered",
    "person_exited",
    "unknown_entered",
    "unknown_exited",
}

class EventPublisher:
    def __init__(self, redis_client):
        self.redis = redis_client

    def publish(self, event_type, payload):

        if event_type not in ALLOWED_EVENTS:
            return

        print(f"[PUBLISH] {event_type} → {payload}")
        camera_code = payload.get("camera_code")
        if camera_code is None:
            camera_code = payload.get("camera")
        if camera_code is None:
            return
        
        channel = f"live-face-tracker:camera-events:{camera_code}"
        message = {
            "event": event_type,
            **payload
        }
        # Add to stream for event history/persistence (ALLOWED_EVENTS only)
        if event_type in STREAM_EVENTS:
            self._publish_to_stream(event_type, camera_code, message)

        # Publish to Pub/Sub (real-time subscribers)
        if event_type in ALLOWED_EVENTS:
            self.redis.publish(channel, json.dumps(message))

    def _publish_to_stream(self, event_type, camera_code, message):
        """
        Persist events to Redis Stream for:
        - Event history
        - Replay capability
        - Consumer groups
        """
        stream_key = f"attendance_stream"
        
        try:
            self.redis.xadd(
                stream_key,
                {
                    "event": event_type,
                    "category": "attendance",
                    "payload": json.dumps(message),
                    "timestamp": int(time.time() * 1000)
                }
            )
            print(f"[STREAM] {event_type} → {stream_key}")
        except Exception as e:
            print(f"[STREAM ERROR] {stream_key}: {e}")