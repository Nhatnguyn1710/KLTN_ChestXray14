# src/utils.py
"""
Utility functions dùng chung cho toàn dự án.
"""

import os
import random
import yaml
import torch
import numpy as np
from pathlib import Path


def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Recursively merge two config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_recursive(config_path: Path, seen: set[Path]) -> dict:
    resolved = config_path.resolve()
    if resolved in seen:
        raise ValueError(f"Config extends cycle detected at: {resolved}")
    seen.add(resolved)

    with open(resolved, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    parent_ref = config.pop("extends", None)
    if not parent_ref:
        return config

    parent_path = Path(parent_ref)
    if not parent_path.is_absolute():
        parent_path = resolved.parent / parent_path
    parent_config = _load_config_recursive(parent_path, seen)
    return _deep_merge_dict(parent_config, config)


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load cấu hình từ file YAML."""
    return _load_config_recursive(Path(config_path), seen=set())


def set_seed(seed: int = 42):
    """Đặt seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict = None) -> torch.device:
    """Lấy device (cuda/cpu) dựa trên config và hardware."""
    if config and config.get("general", {}).get("device") == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_project_root() -> Path:
    """Trả về đường dẫn gốc của dự án."""
    return Path(__file__).parent.parent


def ensure_dir(path: str):
    """Tạo thư mục nếu chưa tồn tại."""
    os.makedirs(path, exist_ok=True)


def print_gpu_info():
    """In thông tin GPU hiện tại."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_used = torch.cuda.memory_allocated(0) / (1024**3)
        vram_free = vram_total - vram_used
        print(f"GPU: {gpu}")
        print(f"VRAM: {vram_used:.1f} GB used / {vram_total:.1f} GB total ({vram_free:.1f} GB free)")
    else:
        print("No GPU available, using CPU")
