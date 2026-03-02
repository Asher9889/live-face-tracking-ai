from app.config.config import envConfig
import redis
import os
from app.config import envConfig


class RedisManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisManager, cls).__new__(cls)
            cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        self.client = redis.Redis(
            host=envConfig.REDIS_HOST,
            port=envConfig.REDIS_PORT,
            db=envConfig.REDIS_DB,
            password=envConfig.REDIS_PASSWORD,
            decode_responses=True
        )

        try:
            self.client.ping()
            print("[Redis] Connected successfully")
        except redis.exceptions.RedisError as e:
            print(f"[Redis] Connection failed: {e}")
            raise

    def get_client(self):
        return self.client