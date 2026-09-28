import os
from dotenv import load_dotenv

load_dotenv()

FRAME_RATE = os.getenv("FRAME_RATE", "15")

class EnvConfig:
    # Device selection: "auto" (GPU when present), "cuda" (require GPU), "cpu"
    DEVICE = os.getenv("DEVICE", "auto").strip().lower()

    # ---------------- LiveKit (preview stream) ----------------
    # Secrets live in .env only. Never commit LIVEKIT_API_SECRET.
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")

    # ---------------- Preview publishing ----------------
    PREVIEW_ENABLED = os.getenv("PREVIEW_ENABLED", "false").lower() in ("1", "true", "yes")
    # Long edge of the published frame. Height follows source aspect ratio.
    PREVIEW_WIDTH = int(os.getenv("PREVIEW_WIDTH", "640"))
    PREVIEW_QUALITY = int(os.getenv("PREVIEW_QUALITY", "60"))
    # Cap on published frames per second per camera.
    PREVIEW_FPS = int(os.getenv("PREVIEW_FPS", "15"))
    # "auto" = NVENC when a CUDA device is present, else libx264.
    PREVIEW_ENCODER = os.getenv("PREVIEW_ENCODER", "auto").strip().lower()
    # Preset for the software encoder. "ultrafast" trades size for CPU time.
    PREVIEW_X264_PRESET = os.getenv("PREVIEW_X264_PRESET", "ultrafast")

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
    MIN_RECOGNITION_FACE_WIDTH = int(os.getenv("MIN_RECOGNITION_FACE_WIDTH", "30"))
    MIN_UNKNOWN_REG_FACE_WIDTH = int(os.getenv("MIN_UNKNOWN_REG_FACE_WIDTH", "45"))
    # Backward-compatible alias for older code paths.
    MIN_FACE_WIDTH = MIN_RECOGNITION_FACE_WIDTH
    BLUR_THRESHOLD = 30
    MIN_UNKNOWN_CREATION_QUALITY = float(os.getenv("MIN_UNKNOWN_CREATION_QUALITY"))
    MIN_UNKNOWN_CREATE_FRAMES = int(os.getenv("MIN_UNKNOWN_CREATE_FRAMES", "2"))
    SCRFD_THRESHOLD = float(os.getenv("SCRFD_THRESHOLD", "0.50"))

envConfig = EnvConfig()  
