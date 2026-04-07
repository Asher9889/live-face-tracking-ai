import json


ALLOWED_EVENTS = {
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

        self.redis.publish(channel, json.dumps(message))