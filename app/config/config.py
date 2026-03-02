import os
from dotenv import load_dotenv

load_dotenv()

FRAME_RATE = os.getenv("FRAME_RATE", "15")

class EnvConfig:
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = int(os.getenv("REDIS_DB"))

envConfig = EnvConfig()
