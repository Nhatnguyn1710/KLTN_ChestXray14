"""Shared global state for the API layer.

Kept intentionally simple: module-level globals were the pattern before
refactor, so we preserve exactly the same behavior, just extracted so
that routes/services can import without circular dependencies.
"""
import os
import re
from typing import Optional, Dict, Any, Tuple


# ─── Runtime globals (populated by startup.initialize) ─────────────
config: Optional[Dict[str, Any]] = None
device = None
cnn_model = None
metadata_index = None


# ─── Constants ─────────────────────────────────────────────────────
# NIH ChestX-ray14 không có nhãn "No Finding" như CheXpert.
# Giữ lại 2 hằng dưới để các call-site cũ vẫn hoạt động (no-op).
SPECIAL_LABELS: set = set()
NO_FINDING_LABEL = None
DEFAULT_THRESHOLD_BASE = 0.5
THRESHOLD_FLOOR = 0.45  # Conservative default floor to reduce false positives
_VIEW_CODE_RE = re.compile(r"\b(AP|PA)\b", re.IGNORECASE)


# ─── Folder layout ─────────────────────────────────────────────────
_BASE = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(_BASE, "uploads")
GRADCAM_FOLDER = os.path.join(_BASE, "static", "gradcam")
STATIC_FOLDER = os.path.join(_BASE, "static")
REACT_APP_INDEX = os.path.join(_BASE, "static", "dist", "index.html")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)


# ─── Helpers ───────────────────────────────────────────────────────
def load_bootstrap_config() -> Dict[str, Any]:
    """Load config early for startup-safe middleware defaults."""
    from src.utils import load_config
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    try:
        return load_config(config_path)
    except Exception:
        return {}


def resolve_cors_settings(cfg: Dict[str, Any]) -> Tuple[list, bool]:
    """Resolve CORS from env/config with secure defaults."""
    backend_cfg = cfg.get("web", {}).get("backend", {}) if isinstance(cfg, dict) else {}
    env_origins = os.environ.get("WEB_CORS_ORIGINS", "").strip()

    if env_origins:
        origins = [x.strip() for x in env_origins.split(",") if x.strip()]
    else:
        origins = backend_cfg.get(
            "cors_origins",
            [
                "http://localhost:7860",
                "http://127.0.0.1:7860",
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ],
        )

    if isinstance(origins, str):
        origins = [origins]
    if not origins:
        origins = ["http://localhost:7860"]

    allow_credentials = bool(backend_cfg.get("cors_allow_credentials", True))
    # Wildcard + credentials is unsafe and invalid in browsers.
    if "*" in origins and allow_credentials:
        allow_credentials = False

    return origins, allow_credentials


def resolve_max_upload_bytes(runtime_cfg: Optional[Dict[str, Any]]) -> int:
    """Get max upload bytes from config/env; fallback to 10MB."""
    default_mb = 10.0
    backend_cfg = {}
    if isinstance(runtime_cfg, dict):
        backend_cfg = runtime_cfg.get("web", {}).get("backend", {})

    raw_mb = backend_cfg.get("max_upload_mb")
    if raw_mb is None:
        raw_mb = os.environ.get("MAX_UPLOAD_MB")
    try:
        max_mb = float(raw_mb) if raw_mb is not None else default_mb
    except (TypeError, ValueError):
        max_mb = default_mb

    if max_mb <= 0:
        max_mb = default_mb
    return int(max_mb * 1024 * 1024)


def extract_view_code(view_text: str) -> str:
    match = _VIEW_CODE_RE.search(str(view_text or ""))
    return match.group(1).upper() if match else ""


def resolve_view_type_value(view_code: str) -> Optional[float]:
    code = str(view_code or "").strip().upper()
    if code == "AP":
        return 0.0
    if code == "PA":
        return 1.0
    return None
