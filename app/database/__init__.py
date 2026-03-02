from .redis_client import RedisManager

redis_manager = RedisManager()
redis_client = redis_manager.get_client()

redis_client.set("test", "hello")
print(redis_client.get("test"))

__all__ = ["RedisManager", "redis_client"]