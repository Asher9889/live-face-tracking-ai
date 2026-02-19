import requests
from typing import List
from urllib.parse import quote, urlsplit
from .types import CameraConfig

# from config import CAMERA_API_URL


CAMERA_API_URL = "https://live-face-tracker.mssplonline.in/api/v1/cameras"

def normalize_rtsp(rtsp_url: str, username: str, password: str) -> str:
    """
    Rebuild RTSP URL safely.
    Handles passwords containing @ or special characters.
    """
    parts = urlsplit(rtsp_url)

    # parts.netloc may contain old credentials → remove them
    host_part = parts.hostname
    port_part = f":{parts.port}" if parts.port else ""

    safe_user = quote(username, safe="")
    safe_pass = quote(password, safe="")

    rebuilt = (
        f"{parts.scheme}://{safe_user}:{safe_pass}@"
        f"{host_part}{port_part}{parts.path}"
    )

    if parts.query:
        rebuilt += f"?{parts.query}"

    return rebuilt


def fetch_cameras() -> List[CameraConfig]:
    print("[Server] Fetching camera configs...")

    try:
        res = requests.get(CAMERA_API_URL, timeout=10)
        res.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch cameras: {e}")

    payload = res.json()

    if not payload.get("success"):
        raise RuntimeError("Camera API returned success=false")

    cameras = payload.get("data", [])
    enabled_cameras = [c for c in cameras if c.get("enabled")]

    print(f"[Server] Found {len(enabled_cameras)} enabled cameras")

    final_configs = []

    for cam in enabled_cameras:
        # ===== Validate required fields =====
        if not cam.get("rtspUrl") or not cam.get("code"):
            print(f"[Server] Skipping invalid camera config: {cam}")
            continue

        creds = cam.get("credentials", {})

        rtsp_url = normalize_rtsp(
            cam["rtspUrl"],
            creds.get("username", ""),
            creds.get("password", "")
        )

        config = CameraConfig(
            code=cam["code"],
            name=cam.get("name", ""),
            gate_type=cam.get("gateType", "UNKNOWN"),
            rtsp_url=rtsp_url,
            ai_fps=cam.get("streamConfig", {}).get("aiFps", 10),
            roi=cam.get("roi", {}),
        )


        final_configs.append(config)

        # Log without password
        print(f"  → {config.code} ({config.gate_type})")

    return final_configs

