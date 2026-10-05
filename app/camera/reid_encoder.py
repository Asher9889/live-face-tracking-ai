"""
BoT-SORT ReID encoder.

Ultralytics builds its own encoder for the tracker's `model:` entry and leaves two
problems for this project:

1. Device. `ReID.__init__` never passes a device, so AutoBackend decides from
   `torch.cuda.is_available()` alone and silently ignores DEVICE=cpu. This app
   resolves devices through app.ai.runtime_device (GPU first, CPU fallback,
   actual value logged) for every other backend, so the encoder is built the same
   way here instead of being the odd one out.

2. One embedding per crop. `ReID.__call__` hands the tracker whatever the batched
   inference returned, relying on `len(feats) != n and feats[0].shape[0] == n` to
   detect the batched path. When the ReID head returns a different number of
   embeddings than there are crops -- an occluded person, or two people inside one
   detection box -- that guess fails and the tracker receives features that are
   the wrong length or the wrong rank. BOTSORT then stacks them and numpy raises
   "setting an array element with a sequence" or "XA must be a 2-dimensional
   array", once per frame, killing tracking for that camera.

This encoder guarantees exactly one L2-normalised embedding per input crop, on any
backend, falling back to per-crop inference when the batched result is ambiguous.
"""

import logging
import os

import numpy as np
import torch

from app.ai.runtime_device import resolve_torch_device

logger = logging.getLogger(__name__)

# A detection crop must map to exactly one embedding. Anything longer is a crop
# containing extra people, which the tracker cannot use anyway.
EMBED_DIM_MIN = 64


class ReIDEncoderError(RuntimeError):
    """Raised when the ReID encoder cannot produce usable embeddings."""


def _to_1d(item) -> np.ndarray | None:
    """Flatten a single embedding to (D,) float32, or None if unusable."""
    if hasattr(item, "detach"):
        item = item.detach().cpu().numpy()
    arr = np.asarray(item, dtype=np.float32)
    if arr.ndim == 0 or arr.size < EMBED_DIM_MIN:
        return None
    return arr.reshape(-1)


def _collect(out, n: int) -> list | None:
    """
    Pull n embeddings out of a predictor result.

    Handles both shapes ultralytics can return: one result per crop, or a single
    stacked batch. Returns None when the count cannot be trusted, so the caller
    can retry one crop at a time.
    """
    if out is None:
        return None

    # Single stacked result: (n, D) or an object exposing .cpu()/.numpy().
    if isinstance(out, (list, tuple)) and len(out) == 1:
        head = out[0]
        if isinstance(head, (list, tuple)):
            head = head[0] if head else None
        if head is not None:
            stacked = _to_1d(head)
            # A stacked (n, D) tensor flattens to n*D, so split it back apart.
            if stacked is not None:
                if hasattr(out[0], "shape") and len(out[0].shape) == 2 and out[0].shape[0] == n:
                    return [row for row in np.asarray(_raw(out[0]))]
                return None

    if isinstance(out, (list, tuple)) and len(out) == n:
        feats = [_to_1d(f) for f in out]
        if all(f is not None for f in feats):
            return feats
    return None


def _raw(value):
    """numpy view of a tensor/array without flattening it."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _normalise(feats: list, n: int) -> list:
    dims = {f.shape[0] for f in feats}
    if len(dims) != 1:
        raise ReIDEncoderError(
            f"ReID returned embeddings with mismatched dimensions {sorted(dims)} for {n} "
            f"detections. The tracker's appearance matching needs one fixed vector per person."
        )
    out = []
    for feat in feats:
        norm = float(np.linalg.norm(feat))
        out.append(feat / norm if norm > 0 else feat)
    return out


class SafeReIDEncoder:
    """Drop-in replacement for ultralytics.trackers.bot_sort.ReID."""

    def __init__(self, model: str):
        from ultralytics import YOLO

        device = resolve_torch_device()
        self.device = device
        self.model_path = model

        # Device is a call-time argument: YOLO.__init__ has no device kwarg and the
        # AutoBackend is built from the first call. This init call creates the
        # backend, so the device has to be set here to reach the session.
        self.model = YOLO(model)
        embed = [len(self.model.model.model) - 2 if model.endswith(".pt") else -1]
        self.model(embed=embed, verbose=False, save=False, device=device)

        # print, not logger.info: only WARNING+ is configured, so INFO would hide
        # the one line that proves which device the encoder landed on.
        print(f"[Tracker] ReID encoder device: {device} ({os.path.basename(model)})")

    def _predict(self, crops):
        return self.model.predictor(crops)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list:
        from ultralytics.utils.ops import xywh2xyxy
        from ultralytics.utils.plotting import save_one_box

        n = int(dets.shape[0])
        if n == 0:
            return []

        boxes = xywh2xyxy(torch.from_numpy(np.ascontiguousarray(dets[:, :4], dtype=np.float32)))
        crops = [save_one_box(box, img, save=False) for box in boxes]

        feats = _collect(self._predict(crops), n)

        if feats is None:
            # Ambiguous batched result: redo one crop at a time so the
            # embedding-to-detection mapping is never guessed.
            feats = []
            for crop in crops:
                single = _collect(self._predict([crop]), 1)
                if single is None:
                    raise ReIDEncoderError(
                        "ReID model produced no usable embedding for a detection crop. "
                        "Check that the checkpoint exports an embedding head."
                    )
                feats.extend(single)

        return _normalise(feats, n)


def install_reid_encoder() -> None:
    """
    Make BOTSORT build our encoder instead of ultralytics' ReID.

    BOTSORT.__init__ resolves `ReID(args.model)` as a module global, so replacing
    the name on the bot_sort module is enough. Called once before camera threads
    start, not per camera.
    """
    from ultralytics.trackers import bot_sort

    bot_sort.ReID = SafeReIDEncoder