"""Startup initialization for the X-Ray API.

Moved verbatim from the monolithic app.py — same print messages, same
fallback behavior, same optional-component handling.
"""
import os
import re

from src.utils import load_config, get_device, print_gpu_info
from src.cnn.model import load_trained_model
from src.api import state


def initialize(config_path: str = "configs/config.yaml"):
    """Khởi tạo tất cả models."""
    state.config = load_config(config_path)
    state.device = get_device(state.config)
    print_gpu_info()

    # Load CNN
    checkpoint = state.config["paths"]["densenet_checkpoint"]
    if not os.path.exists(checkpoint):
        # Fallback: auto-pick latest versioned best_model.pth from checkpoint_base_dir.
        base_dir = state.config.get("paths", {}).get("checkpoint_base_dir", "")
        candidates = []
        if base_dir and os.path.isdir(base_dir):
            for name in os.listdir(base_dir):
                if re.fullmatch(r"v\d+", name):
                    path = os.path.join(base_dir, name, "best_model.pth")
                    if os.path.isfile(path):
                        candidates.append((int(name[1:]), path))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            checkpoint = candidates[-1][1]
            print(f"  [WARN] Config checkpoint not found, fallback to latest: {checkpoint}")
    if os.path.exists(checkpoint):
        state.cnn_model = load_trained_model(checkpoint, state.config, state.device)
        print("  [OK] CNN model loaded")
    else:
        print(f"  [ERR] CNN model not trained. File not found: {checkpoint}")

    # Load Image Metadata Index (for view position lookup)
    try:
        from src.metadata import ImageMetadataIndex
        _idx_path = state.config.get("paths", {}).get(
            "metadata_index", "data/image_metadata_index.json"
        )
        if os.path.isfile(_idx_path):
            state.metadata_index = ImageMetadataIndex(_idx_path)
            print(f"  [OK] Metadata index loaded ({state.metadata_index.size:,} images)")
        else:
            print(f"  [WARN] Metadata index not found: {_idx_path}")
            print("    -> Run: python -m src.metadata --build")
    except Exception as e:
        print(f"  [ERR] Metadata index error: {e}")

    print("\n  === FastAPI Backend Ready ===\n")
