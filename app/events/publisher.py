import json
import time

class EventPublisher:
    def __init__(self, redis_client):
        self.redis = redis_client

    def publish(self, event_type, payload):
        camera_code = payload["camera"]
        channel = f"live-face-tracker:camera-events:{camera_code}"
        message = {
            "event": event_type,
            "timestamp": time.time(),
            "data": payload
        }

        self.redis.publish(channel, json.dumps(message))