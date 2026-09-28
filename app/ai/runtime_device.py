"""
Runtime device resolution.

Every backend is resolved GPU-first with a CPU fallback, and the resolved value is
logged. Nothing here reports intent: onnxruntime silently falls back when a
requested provider is missing, so a hardcoded CUDAExecutionProvider can print
"GPU enabled" while running entirely on CPU. Each helper returns what is actually
in use.
"""

import os
import logging

logger = logging.getLogger(__name__)

# "auto" = GPU when available, "cuda" = require GPU, "cpu" = force CPU
DEVICE_PREF = os.getenv("DEVICE", "auto").strip().lower()

_cache = {}


def _forced_cpu():
    return DEVICE_PREF == "cpu"


def _cuda_available():
    """
    A CUDA device is usable only if every layer can actually reach it:
    onnxruntime-gpu for InsightFace and torch.cuda for YOLO.
    """
    if _forced_cpu():
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            return False
        if torch.cuda.device_count() < 1:
            return False
    except Exception:
        return False

    return True


def resolve_onnx_providers():
    """
    Provider list for InsightFace / onnxruntime.

    Returns CUDAExecutionProvider first when it is genuinely available, always with
    CPUExecutionProvider retained as the fallback for any op CUDA cannot serve.
    """
    key = "onnx_providers"
    if key in _cache:
        return _cache[key]

    providers = []

    if not _forced_cpu():
        try:
            import onnxruntime as ort
            available = list(ort.get_available_providers())
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            logger.info("[DEVICE] onnxruntime available providers: %s", available)
        except Exception as exc:
            logger.warning("[DEVICE] onnxruntime provider probe failed: %s", exc)

    providers.append("CPUExecutionProvider")

    if DEVICE_PREF == "cuda" and "CUDAExecutionProvider" not in providers:
        raise RuntimeError("DEVICE=cuda requested but CUDAExecutionProvider is unavailable")

    _cache[key] = providers
    return providers


def resolve_torch_device():
    """Device string for ultralytics YOLO."""
    key = "torch_device"
    if key in _cache:
        return _cache[key]

    device = "cpu"
    if _cuda_available():
        device = "cuda:0"
    elif DEVICE_PREF == "cuda":
        raise RuntimeError("DEVICE=cuda requested but torch.cuda is unavailable")

    _cache[key] = device
    return device


def resolve_mediapipe_delegate():
    """
    MediaPipe delegate for FaceLandmarker.

    Returns the BaseOptions.Delegate member. MediaPipe raises at
    create_from_options() when the GPU delegate cannot be initialised, so the
    caller must still handle a failed GPU construction and retry on CPU.
    """
    key = "mediapipe_delegate"
    if key in _cache:
        return _cache[key]

    delegate = None
    if not _forced_cpu():
        try:
            import torch
            if torch.cuda.is_available():
                delegate = "gpu"
        except Exception:
            delegate = None

    _cache[key] = delegate
    return delegate


def resolve_video_encoder():
    """
    H.264 encoder for the preview publisher, GPU-first.

    NVENC is only useful if a CUDA device is present and PyAV was built with
    ffmpeg hardware encoding support; otherwise software libx264.
    """
    key = "video_encoder"
    if key in _cache:
        return _cache[key]

    encoder = "libx264"
    if _cuda_available():
        try:
            import av
            # PyAV exposes hardware encoders as h264_* codec names.
            codecs = set(av.codecs_available) if hasattr(av, "codecs_available") else set()
            if any("h264_nvenc" in c for c in codecs):
                encoder = "h264_nvenc"
        except Exception as exc:
            logger.warning("[DEVICE] encoder probe failed, using libx264: %s", exc)

    _cache[key] = encoder
    return encoder


def log_summary():
    """One line describing the resolved runtime, for startup diagnostics."""
    logger.info(
        "[DEVICE] pref=%s torch=%s onnx=%s mediapipe=%s encoder=%s",
        DEVICE_PREF,
        resolve_torch_device(),
        ",".join(resolve_onnx_providers()),
        resolve_mediapipe_delegate() or "cpu",
        resolve_video_encoder(),
    )
