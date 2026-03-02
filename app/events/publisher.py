import json
import time

class EventPublisher:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.channel = "ai_events"

    def publish(self, event_type, payload):
        message = {
            "event": event_type,
            "timestamp": time.time(),
            "data": payload
        }

        self.redis.publish(self.channel, json.dumps(message))