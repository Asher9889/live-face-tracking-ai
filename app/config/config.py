import os
from dotenv import load_dotenv

load_dotenv()

FRAME_RATE = os.getenv("FRAME_RATE", "15")

class EnvConfig:
    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PORT = int(os.getenv("REDIS_PORT"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    REDIS_DB = int(os.getenv("REDIS_DB"))
    NODE_LOAD_EMBEDDINGS_URL = os.getenv("NODE_LOAD_EMBEDDINGS_URL")
    TOKEN_TO_ACCESS_NODE_API = os.getenv("TOKEN_TO_ACCESS_NODE_API")
    NODE_LOAD_UNKNOWN_EMBEDDINGS_URL = os.getenv("NODE_LOAD_UNKNOWN_EMBEDDINGS_URL")
    NODE_CREATE_UNKNOWN_URL = os.getenv("NODE_CREATE_UNKNOWN_URL")
    NODE_UPDATE_UNKNOWN_URL = os.getenv("NODE_UPDATE_UNKNOWN_URL")
    CAMERA_API_URL = os.getenv("CAMERA_API_URL")
    # Global override: when true, force using local webcam instead of RTSP streams
    USE_WEBCAM = os.getenv("USE_WEBCAM", "false").lower() in ("1", "true", "yes")
    MIN_UNKNOWN_FRAMES = 4
    MAX_UNKNOWN_FRAMES = 5
    MIN_FACE_SIZE = 60 # pixels
    BLUR_THRESHOLD = 30
    MIN_UNKNOWN_CREATION_QUALITY = float(os.getenv("MIN_UNKNOWN_CREATION_QUALITY"))
    MIN_UNKNOWN_CREATE_FRAMES = int(os.getenv("MIN_UNKNOWN_CREATE_FRAMES", "2"))
    SCRFD_THRESHOLD = float(os.getenv("SCRFD_THRESHOLD", "0.50"))
    DEBUG_IMAGE_DUMP_ENABLED = os.getenv("DEBUG_IMAGE_DUMP_ENABLED", "false").lower() in ("1", "true", "yes")
    DEBUG_IMAGE_DUMP_INTERVAL_SEC = float(os.getenv("DEBUG_IMAGE_DUMP_INTERVAL_SEC", "2.0"))
    DEBUG_UNKNOWN_CREATE_PAYLOAD = os.getenv("DEBUG_UNKNOWN_CREATE_PAYLOAD", "false").lower() in ("1", "true", "yes")
    DEBUG_UNKNOWN_UPDATE_PAYLOAD = os.getenv("DEBUG_UNKNOWN_UPDATE_PAYLOAD", "false").lower() in ("1", "true", "yes")

envConfig = EnvConfig()  
