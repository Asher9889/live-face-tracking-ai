"""
BoT-SORT asset resolution.

Ultralytics only auto-downloads files whose names it recognises (yolov8n.pt,
yolo11n.pt, ...). A ReID model such as yolo26s-reid.onnx is NOT one of those, so
YOLO("yolo26s-reid.onnx") fails with FileNotFoundError even though the file is
published in the ultralytics assets release.

This module resolves the tracker config to a real, on-disk ReID checkpoint before
any camera thread starts:

  1. read the tracker yaml the project ships (app/camera/botsort.yaml)
  2. if it disables reid, or asks for model: auto, hand the yaml back untouched
  3. otherwise fetch the checkpoint into <repo>/models/ and raise
     TrackerAssetError if it cannot be fetched

The generated yaml is a sibling of the source yaml, so the shipped file keeps
its comments and its portable "model: yolo26s-reid.onnx" name.

Ultralytics resolves the tracker argument with check_yaml(), which accepts only
a path to a yaml file -- that is why a generated yaml is used instead of
patching the ReID path into the caller's kwargs.
"""

import os
import threading
from pathlib import Path

import yaml
from ultralytics.utils.downloads import attempt_download_asset

# app/camera/tracker_assets.py -> app/camera -> app -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default tracker config shipped with the project.
TRACKER_YAML_DEFAULT = Path(__file__).resolve().with_name("botsort.yaml")

# Where downloaded checkpoints are kept. Ultralytics keeps its own weights in
# ./weights; this project already stores adaface/facemesh/scrfd under models/.
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(REPO_ROOT / "models"))).expanduser().resolve()

# ReID checkpoints are tens of MB. A smaller file means the download failed and
# wrote an error page instead, so treat it as unusable.
MIN_WEIGHTS_BYTES = 1_000_000

_DOWNLOAD_LOCK = threading.Lock()


class TrackerAssetError(RuntimeError):
    """Raised when the tracker/ReID assets cannot be resolved. Fatal."""


def _is_usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_WEIGHTS_BYTES
    except OSError:
        return False


def _existing_checkpoint(name: str) -> Path:
    """A checkpoint already sitting next to the tracker yaml, cwd or MODEL_DIR."""
    candidates = [
        TRACKER_YAML_DEFAULT.with_name(name),
        Path.cwd() / name,
        MODEL_DIR / name,
    ]
    for candidate in candidates:
        if _is_usable(candidate):
            return candidate.resolve()
    return candidates[-1]


def ensure_reid_weights(name: str) -> Path:
    """
    Return a local path to the ReID checkpoint `name`, downloading it when absent.

    Raises TrackerAssetError if the checkpoint is missing and cannot be
    downloaded, or if it downloaded to something unusable.
    """
    name = Path(name).name
    target = _existing_checkpoint(name)

    if _is_usable(target):
        print(f"[Tracker] ReID model: {target}")
        return target

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise TrackerAssetError(
            f"Cannot create model directory '{target.parent}': {exc}\n"
            f"Set MODEL_DIR to a writable path and retry."
        ) from exc

    print(f"[Tracker] ReID model '{name}' not found -> downloading to {target} ...")

    with _DOWNLOAD_LOCK:
        if not _is_usable(target):
            try:
                # Absolute path: ultralytics downloads straight there instead of
                # dropping the file into the current working directory.
                attempt_download_asset(str(target))
            except Exception as exc:
                raise TrackerAssetError(
                    f"Failed to download the ReID model '{name}' for BoT-SORT.\n"
                    f"  target : {target}\n"
                    f"  reason : {type(exc).__name__}: {exc}\n"
                    f"  fix    : download the file manually to {target}, or point\n"
                    f"           MODEL_DIR somewhere already holding it, or set\n"
                    f"           'model: auto' in {TRACKER_YAML_DEFAULT} to use the\n"
                    f"           detector's own features instead of a ReID model."
                ) from exc

        # attempt_download_asset returns a path even when it cannot find the asset
        # in any release, so the file has to be verified rather than trusted.
        if not _is_usable(target):
            size = target.stat().st_size if target.is_file() else "missing"
            raise TrackerAssetError(
                f"Download of the ReID model '{name}' did not produce a usable file.\n"
                f"  target : {target}\n"
                f"  size   : {size} (expected >= {MIN_WEIGHTS_BYTES} bytes)\n"
                f"  fix    : download it manually to {target}, or set 'model: auto'\n"
                f"           in {TRACKER_YAML_DEFAULT}."
            )

    print(f"[Tracker] ReID model ready: {target}")
    return target


def resolve_tracker_config(tracker_yaml: str | None = None) -> str:
    """
    Return a tracker yaml path whose ReID entry points at a real checkpoint.

    `model: auto` and `with_reid: False` are returned unchanged -- both use the
    detector's features and need no download. Anything else is resolved to an
    absolute path, downloading it when required.
    """
    source = Path(tracker_yaml or TRACKER_YAML_DEFAULT).expanduser()
    if not source.is_file():
        raise TrackerAssetError(f"Tracker config not found: {source}")

    try:
        with source.open("r") as handle:
            cfg = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise TrackerAssetError(f"Tracker config {source} is not valid YAML: {exc}") from exc

    if not cfg.get("with_reid"):
        return str(source.resolve())

    model = str(cfg.get("model") or "").strip()
    if not model or model == "auto":
        return str(source.resolve())

    weights = ensure_reid_weights(model)

    resolved = source.with_name(f"{source.stem}.resolved{source.suffix}")
    cfg["model"] = str(weights)
    try:
        with resolved.open("w") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, default_flow_style=False)
    except OSError as exc:
        raise TrackerAssetError(f"Cannot write resolved tracker config {resolved}: {exc}") from exc

    print(f"[Tracker] Config: {source} -> {resolved} (model={weights})")
    return str(resolved)