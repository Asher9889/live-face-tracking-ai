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

    # Eye-region sharpness (Laplacian variance around each iris centre, measured on
    # the upscaled crop) below which a face is considered too blurred to register a
    # new unknown identity from. A blurry face still fails to match anyone, so it
    # gets stored as an unknown — which is the noise this threshold removes.
    #
    # NOT YET ACTIVE as a reject: the value is measured and logged only, so a real
    # distribution can be observed before it starts dropping faces. See
    # UNKNOWN_CREATION_LOG_PATH and ENFORCE_UNKNOWN_EYE_SHARPNESS below.
    MIN_UNKNOWN_EYE_SHARPNESS = int(os.getenv("MIN_UNKNOWN_EYE_SHARPNESS", "50"))  # was 200
    # Master switch for the eye-sharpness reject. "0" = measure and log only.
    ENFORCE_UNKNOWN_EYE_SHARPNESS = os.getenv("ENFORCE_UNKNOWN_EYE_SHARPNESS", "true").lower() in ("1","true","yes")


    # ---- Unknown-registration eye-visibility gate -------------------------
    # Production audit (104 rows, 12 registrations, all confirmed noise) showed
    # eye_sharpness carries no usable signal: the noise registrations had a
    # median of 34 against 180 for matched-existing tracks, but the ranges
    # overlap almost completely, so it cannot separate good from bad.
    #
    # Two signals did separate. The iris core must be measurably darker than the
    # surrounding iris ring, which is scale-independent and unaffected by how far
    # the face is from the camera, so it is scored as a ratio:
    #
    #     iris_contrast / iris_core_brightness
    #
    # and the eye centre-to-centre distance must stay wide enough to imply the
    # face is not turned far to one side (a profile view leaves one eye
    # compressed to nothing).
    #
    # A frame must clear BOTH. "0" disables the reject so the thresholds can be
    # tuned from logs before they drop real faces.
    # ENFORCE_UNKNOWN_EYE_VISIBILITY = bool(int(os.getenv("ENFORCE_UNKNOWN_EYE_VISIBILITY", "0")))
    ENFORCE_UNKNOWN_EYE_VISIBILITY = os.getenv("ENFORCE_UNKNOWN_EYE_VISIBILITY", "true").lower() in ("1","true","yes")
    MIN_UNKNOWN_IRIS_CONTRAST_RATIO = float(os.getenv("MIN_UNKNOWN_IRIS_CONTRAST_RATIO", "0.05"))  # was 0.15
    MIN_UNKNOWN_EYE_DIST_RATIO = float(os.getenv("MIN_UNKNOWN_EYE_DIST_RATIO", "0.40"))  # was 0.50
    # Applies only to unknown creation. Employees are unaffected.
    MAX_UNKNOWN_REG_YAW = float(os.getenv("MAX_UNKNOWN_REG_YAW", "35"))  # was 25

    # Structured audit log for unknown-registration decisions (one JSON object per
    # line, size-rotated). Face images and embeddings are deliberately NOT written
    # here — that data already lives in the API store, and putting biometrics in a
    # plaintext file is not worth the convenience of debugging.
    UNKNOWN_CREATION_LOG_ENABLED = bool(int(os.getenv("UNKNOWN_CREATION_LOG_ENABLED", "1")))
    UNKNOWN_CREATION_LOG_PATH = os.getenv("UNKNOWN_CREATION_LOG_PATH", "logs/unknown_creation.log")
    UNKNOWN_CREATION_LOG_MAX_BYTES = int(os.getenv("UNKNOWN_CREATION_LOG_MAX_BYTES", str(10 * 1024 * 1024)))
    UNKNOWN_CREATION_LOG_BACKUPS = int(os.getenv("UNKNOWN_CREATION_LOG_BACKUPS", "5"))

envConfig = EnvConfig()  
