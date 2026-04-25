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
    MIN_FACE_WIDTH = 60 # pixels
    BLUR_THRESHOLD = 30
    FACE_GATE_WINDOW = int(os.getenv("FACE_GATE_WINDOW", "5"))
    MIN_WIDTH_HISTORY = int(os.getenv("MIN_WIDTH_HISTORY", "3"))
    MIN_REGISTER_TRACKING_FRAMES = int(os.getenv("MIN_REGISTER_TRACKING_FRAMES", "3"))
    MIN_DETECT_WIDTH = int(os.getenv("MIN_DETECT_WIDTH", "24"))
    MIN_TRACKING_WIDTH = int(os.getenv("MIN_TRACKING_WIDTH", "36"))
    MIN_REGISTER_WIDTH = int(os.getenv("MIN_REGISTER_WIDTH", "52"))
    MIN_DETECT_AREA_RATIO = float(os.getenv("MIN_DETECT_AREA_RATIO", "0.0"))
    MIN_TRACKING_AREA_RATIO = float(os.getenv("MIN_TRACKING_AREA_RATIO", "0.0"))
    MIN_REGISTER_AREA_RATIO = float(os.getenv("MIN_REGISTER_AREA_RATIO", "0.0"))
    MIN_TRACKING_QUALITY = float(os.getenv("MIN_TRACKING_QUALITY", "0.38"))
    MIN_REGISTER_QUALITY = float(os.getenv("MIN_REGISTER_QUALITY", "0.55"))
    MIN_UNKNOWN_CREATION_QUALITY = float(os.getenv("MIN_UNKNOWN_CREATION_QUALITY"))
    MIN_UNKNOWN_CREATE_FRAMES = int(os.getenv("MIN_UNKNOWN_CREATE_FRAMES", "2"))
    SCRFD_THRESHOLD = float(os.getenv("SCRFD_THRESHOLD", "0.50"))

envConfig = EnvConfig()  
