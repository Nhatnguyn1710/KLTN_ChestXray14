"""Health check + labels endpoints."""
import json
import os
import torch
from fastapi import APIRouter

from src.api import state
from src.cnn.dataset import CHEXPERT_LABELS

router = APIRouter()


def _read_calibration_meta() -> dict:
    """Read model_version and generated timestamp from calibration.json."""
    calib_path = os.path.join("configs", "calibration.json")
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "model_version": data.get("model_version", ""),
            "calibration_generated": data.get("generated", ""),
        }
    except Exception:
        return {"model_version": "", "calibration_generated": ""}


@router.get("/api/health")
async def health():
    """Health check — kiểm tra trạng thái hệ thống."""
    calib = _read_calibration_meta()
    return {
        "status": "ok",
        "cnn_loaded": state.cnn_model is not None,
        "config_path": os.environ.get("CONFIG_PATH", "configs/config.yaml"),
        "max_upload_mb": round(state.resolve_max_upload_bytes(state.config) / (1024 * 1024), 2),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "labels": CHEXPERT_LABELS,
        "model_version": calib["model_version"],
        "calibration_generated": calib["calibration_generated"],
    }


@router.get("/api/labels")
async def get_labels():
    """Trả về danh sách 14 nhãn NIH ChestX-ray14."""
    labels_vi = state.config.get("labels_vi", {}) if state.config else {}
    return {
        "labels": [
            {"en": label, "vi": labels_vi.get(label, "")}
            for label in CHEXPERT_LABELS
        ]
    }
