
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.dataset import CHEXPERT_LABELS, get_normalization_stats


# ============================================================
# Per-class optimal thresholds (tính từ validation set)
# Sẽ được cập nhật sau khi train lại model v4
# ============================================================
DEFAULT_THRESHOLDS = {label: 0.5 for label in CHEXPERT_LABELS}


def get_per_class_thresholds(config: dict = None) -> dict:
 
    # 1. Thử đọc từ calibration.json (output của eval_v4.py)
    calib_path = os.path.join("configs", "calibration.json")
    if os.path.isfile(calib_path):
        import json
        with open(calib_path) as f:
            calib = json.load(f)
        if "thresholds" in calib and calib["thresholds"]:
            thresholds = DEFAULT_THRESHOLDS.copy()
            thresholds.update(calib["thresholds"])
            return thresholds

    # 2. Fallback: đọc từ config.yaml
    if config and "cnn" in config:
        custom = config["cnn"].get("thresholds", {})
        if custom:
            thresholds = DEFAULT_THRESHOLDS.copy()
            thresholds.update(custom)
            return thresholds
    return DEFAULT_THRESHOLDS.copy()


def _equivocal_band_from_config(config: dict = None) -> float:
    """Khoảng [t_low, t_high) khi chưa có thresholds_low trong calibration.json."""
    if not config:
        return 0.08
    band = config.get("web", {}).get("equivocal_band")
    if band is None:
        band = (config.get("cnn", {}) or {}).get("calibration", {}).get("equivocal_band", 0.08)
    try:
        b = float(band)
    except (TypeError, ValueError):
        b = 0.08
    return max(0.01, min(0.5, b))


def _min_tri_gap_from_config(config: dict = None) -> float:
    if not config:
        return 0.02
    g = config.get("web", {}).get("min_tri_gap")
    if g is None:
        g = (config.get("cnn", {}) or {}).get("calibration", {}).get("min_tri_gap", 0.02)
    try:
        x = float(g)
    except (TypeError, ValueError):
        x = 0.02
    return max(0.005, min(0.2, x))


def get_per_class_low_thresholds(
    config: dict = None,
    high_thresholds: dict = None,
) -> dict:
    """
    Ngưỡng dưới (t_low) cho phân tầng Âm tính / Nghi ngờ / Phát hiện.
    Ưu tiên: calibration.json key `thresholds_low` > derive từ t_high - equivocal_band.
    """
    high_thresholds = high_thresholds or DEFAULT_THRESHOLDS.copy()
    band = _equivocal_band_from_config(config)
    min_gap = _min_tri_gap_from_config(config)

    calib_path = os.path.join("configs", "calibration.json")
    if os.path.isfile(calib_path):
        import json
        with open(calib_path) as f:
            calib = json.load(f)
        if calib.get("thresholds_low"):
            lows = DEFAULT_THRESHOLDS.copy()
            lows.update(calib["thresholds_low"])
            return lows

    lows = {}
    for label in CHEXPERT_LABELS:
        hi = float(high_thresholds.get(label, 0.5))
        lo = hi - band
        lo = max(0.0, min(lo, hi - min_gap))
        lows[label] = lo
    return lows


# ============================================================
# Temperature Scaling — hiệu chỉnh xác suất output
# ============================================================
def _get_temperature(config: dict = None) -> float:
    """
    Lấy global temperature. Ưu tiên: calibration.json > config.yaml > 1.0
    """
    calib_path = os.path.join("configs", "calibration.json")
    if os.path.isfile(calib_path):
        import json
        with open(calib_path) as f:
            calib = json.load(f)
        if "temperature" in calib:
            try:
                t = float(calib["temperature"])
                if np.isfinite(t) and t > 0:
                    return t
            except (TypeError, ValueError):
                pass
    if config:
        try:
            t = float(config.get("cnn", {}).get("temperature", 1.0))
            if np.isfinite(t) and t > 0:
                return t
        except (TypeError, ValueError):
            pass
    return 1.0


def _get_per_label_temperatures(config: dict = None) -> dict:
    """
    Lấy per-label temperatures từ calibration.json.
    Fallback về global temperature cho label không có in per_label_temperatures.
    Fallback về {} (sử dụng global T đơn) nếu key không tồn tại.
    """
    calib_path = os.path.join("configs", "calibration.json")
    if not os.path.isfile(calib_path):
        return {}
    import json
    with open(calib_path) as f:
        calib = json.load(f)
    per_label = calib.get("per_label_temperatures", {})
    if not per_label:
        return {}
    global_T = float(calib.get("temperature", 1.0))
    # Validate each entry; replace invalid with global temperature
    result = {}
    for label, T in per_label.items():
        try:
            t = float(T)
            result[label] = t if np.isfinite(t) and t > 0 else global_T
        except (TypeError, ValueError):
            result[label] = global_T
    return result


def _get_per_label_isotonic(config: dict = None) -> dict:
    """
    Lấy per-label isotonic regression maps từ calibration.json.
    Trả về {label: {x: [...], y: [...]}} hoặc {} nếu không có.
    Tại inference: dùng numpy.interp(prob_raw, x, y) để map probability đã calibrate.
    """
    calib_path = os.path.join("configs", "calibration.json")
    if not os.path.isfile(calib_path):
        return {}
    import json
    with open(calib_path) as f:
        calib = json.load(f)
    per_label = calib.get("per_label_isotonic", {})
    if not per_label:
        return {}
    result = {}
    for label, m in per_label.items():
        if not isinstance(m, dict):
            continue
        x = m.get("x")
        y = m.get("y")
        if not x or not y or len(x) < 2 or len(x) != len(y):
            continue
        result[label] = {"x": list(x), "y": list(y)}
    return result


def _apply_isotonic_per_label(probs: np.ndarray, labels: list,
                              per_label_isotonic: dict) -> np.ndarray:
    """Apply isotonic mapping per-label. Probs not having a map are unchanged."""
    if not per_label_isotonic:
        return probs
    out = np.array(probs, dtype=np.float64, copy=True)
    for j, label in enumerate(labels):
        m = per_label_isotonic.get(label)
        if not m:
            continue
        x = np.asarray(m["x"], dtype=np.float64)
        y = np.asarray(m["y"], dtype=np.float64)
        out[j] = float(np.interp(out[j], x, y))
    return out


class TemperatureScaler:
    """
    Hiệu chỉnh xác suất bằng temperature scaling.
    Logit / T → sigmoid → calibrated probability.
    T > 1: output nhẹ hơn (less confident)
    T < 1: output mạnh hơn (more confident)
    """

    def __init__(self, temperature: float = 1.0):
        try:
            t = float(temperature)
        except (TypeError, ValueError):
            t = 1.0
        # Temperature must be strictly positive.
        self.temperature = t if np.isfinite(t) and t > 0 else 1.0

    def scale(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities.
        
        Converts probs → logits → logits/T → sigmoid.
        v9 note: model now outputs logits, but inference.py applies sigmoid
        before calling this method, so input is still probabilities.
        """
        if self.temperature == 1.0:
            return probs  # No scaling needed
        # Inverse sigmoid để lấy logits, rồi scale
        eps = 1e-7
        p = np.clip(probs, eps, 1.0 - eps)
        raw_logits = np.log(p / (1.0 - p))  # inverse sigmoid
        scaled = raw_logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))  # sigmoid


def _sanitize_bottom_crop_ratio(bottom_crop_ratio: float) -> float:
    """Clamp bottom-crop ratio to a safe range."""
    try:
        ratio = float(bottom_crop_ratio)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(0.4, ratio))


def _apply_bottom_crop(image_np: np.ndarray, bottom_crop_ratio: float) -> np.ndarray:
    """Crop bottom region before resize to match training preprocessing."""
    ratio = _sanitize_bottom_crop_ratio(bottom_crop_ratio)
    if ratio <= 0.0:
        return image_np
    h = int(image_np.shape[0])
    keep_h = max(1, int(round(h * (1.0 - ratio))))
    return image_np[:keep_h, :]


def load_image_rgb(image_path: str) -> Image.Image:
    """Load image and normalize orientation via EXIF before RGB conversion."""
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def _apply_clahe_preprocessing(image: Image.Image, clahe_cfg: dict) -> Image.Image:
    """Apply percentile windowing + CLAHE on grayscale channel (matches dataset.py).

    Percentile windowing (p1–p99) clips metallic artifact outliers before CLAHE
    so histogram equalization distributes contrast evenly across lung parenchyma
    instead of being skewed by a few extreme-bright pixels.
    """
    if not (clahe_cfg and clahe_cfg.get("enabled", False)):
        return image
    import cv2
    gray = np.array(image.convert("L"))
    # Percentile windowing: clip outlier pixels before CLAHE
    p_lo = float(clahe_cfg.get("percentile_lo", 1.0))
    p_hi = float(clahe_cfg.get("percentile_hi", 99.0))
    lo_val, hi_val = np.percentile(gray, [p_lo, p_hi])
    if hi_val > lo_val:
        gray = np.clip(gray, lo_val, hi_val)
        gray = ((gray - lo_val) / (hi_val - lo_val) * 255.0).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(
        clipLimit=float(clahe_cfg.get("clip_limit", 2.0)),
        tileGridSize=(int(clahe_cfg.get("tile_size", 8)), int(clahe_cfg.get("tile_size", 8))),
    )
    return Image.fromarray(clahe_obj.apply(gray))


def preprocess_image(
    image_path: str,
    image_size: int = 224,
    bottom_crop_ratio: float = 0.0,
    normalization_cfg: dict = None,
    clahe_cfg: dict = None,
    corner_erase_enabled: bool = True,
) -> torch.Tensor:
    """
    Tiền xử lý 1 ảnh X-quang cho inference.

    Args:
        image_path: Đường dẫn ảnh
        image_size: Kích thước resize
    Returns:
        Tensor [1, 3, H, W] đã normalize
    """
    mean, std = get_normalization_stats(normalization_cfg)
    # Letterbox: giữ tỉ lệ ảnh, pad mean-gray (129 ≈ mean*255 → ~0 sau Normalize)
    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,  
            fill=129,
        ),
        A.Normalize(
            mean=mean,
            std=std,
        ),
        ToTensorV2(),
    ])

    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = _apply_clahe_preprocessing(pil_image, clahe_cfg)
    image = np.array(pil_image.convert("RGB"))
    image = _apply_bottom_crop(image, bottom_crop_ratio)
    if corner_erase_enabled:
        h, w = image.shape[:2]
        cy, cx = max(1, int(h * 0.15)), max(1, int(w * 0.15))
        for ys, xs in [
            (slice(0, cy), slice(0, cx)),
            (slice(0, cy), slice(w - cx, w)),
            (slice(h - cy, h), slice(0, cx)),
            (slice(h - cy, h), slice(w - cx, w)),
        ]:
            image[ys, xs] = int(image[ys, xs].mean()) if image[ys, xs].size > 0 else 0
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0)  # Add batch dim
    return image_tensor


def _build_tta_transforms(
    image_size: int,
    normalization_cfg: dict = None,
    tta_cfg: dict = None,
) -> list:
    """Build laterality-aware TTA transforms.

    Default behaviour (when tta_cfg is None): multi-scale center crops at
    scales [1.0, 0.95, 1.05] + 4 corner crops.  Horizontal flip is only
    included for explicitly listed bilateral labels (handled at the
    logit-averaging stage, not here).

    Returns a list of (transform, is_flipped) tuples so callers can
    selectively apply flip results per label.
    """
    if tta_cfg is None:
        tta_cfg = {}
    mean, std = get_normalization_stats(normalization_cfg)
    normalize = A.Normalize(mean=mean, std=std)

    scales = tta_cfg.get("scales", [1.0, 0.95, 1.05])
    corner_crops = tta_cfg.get("corner_crops", True)
    corner_factor = float(tta_cfg.get("corner_resize_factor", 1.15))

    transforms = []  # list of (A.Compose, is_flipped:bool)

    # ── Multi-scale center crops ──
    for scale in scales:
        resize_to = max(image_size, int(round(image_size * scale)))
        transforms.append((A.Compose([
            A.LongestMaxSize(max_size=resize_to),
            A.PadIfNeeded(min_height=resize_to, min_width=resize_to,
                          border_mode=0, fill=129),
            A.CenterCrop(height=image_size, width=image_size),
            normalize, ToTensorV2(),
        ]), False))

    # ── 4 corner crops ──
    if corner_crops:
        resize_to = int(round(image_size * corner_factor))
        margin = resize_to - image_size
        if margin < 1:
            margin = 1
            resize_to = image_size + 1
        for (y_off, x_off) in [
            (0, 0), (0, margin), (margin, 0), (margin, margin),
        ]:
            transforms.append((A.Compose([
                A.LongestMaxSize(max_size=resize_to),
                A.PadIfNeeded(min_height=resize_to, min_width=resize_to,
                              border_mode=0, fill=129),
                A.Crop(x_min=x_off, y_min=y_off,
                       x_max=x_off + image_size, y_max=y_off + image_size),
                normalize, ToTensorV2(),
            ]), False))

    # ── Horizontal flip of the original (scale=1.0 center crop) ──
    # Marked is_flipped=True so callers merge selectively per label.
    transforms.append((A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=0, fill=129),
        A.HorizontalFlip(p=1.0),
        normalize, ToTensorV2(),
    ]), True))

    return transforms


def _merge_tta_logits(
    logits_list: list,
    flip_flags: list,
    labels: list,
    flip_labels: set,
) -> np.ndarray:
    """Average TTA logits, using flip results only for bilateral labels.

    Args:
        logits_list: list of 1-D numpy arrays (num_labels,) — raw logits
        flip_flags:  list of bool, same length, True = from flipped image
        labels:      ordered label names
        flip_labels: set of label names where flip is valid
    Returns:
        Averaged logits as 1-D numpy array (num_labels,).
    """
    n_labels = len(labels)
    logit_sum = np.zeros(n_labels, dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.float64)
    for logits, is_flipped in zip(logits_list, flip_flags):
        for j, label in enumerate(labels):
            if is_flipped and label not in flip_labels:
                continue  # skip flip result for lateralized labels
            logit_sum[j] += logits[j]
            counts[j] += 1.0
    counts = np.maximum(counts, 1.0)
    return (logit_sum / counts).astype(np.float32)


def _merge_tta_logits_batch(
    logits_list: list,
    flip_flags: list,
    labels: list,
    flip_labels: set,
) -> np.ndarray:
    """Batch version: merge TTA logits for N images.

    Args:
        logits_list: list of arrays, each shape (N, num_labels)
        flip_flags:  list of bool, same length as logits_list
        labels:      ordered label names
        flip_labels: set of label names where flip is valid
    Returns:
        Averaged logits, shape (N, num_labels).
    """
    n_labels = len(labels)
    N = logits_list[0].shape[0]
    logit_sum = np.zeros((N, n_labels), dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.float64)
    for logits, is_flipped in zip(logits_list, flip_flags):
        for j, label in enumerate(labels):
            if is_flipped and label not in flip_labels:
                continue
            logit_sum[:, j] += logits[:, j]
            counts[j] += 1.0
    counts = np.maximum(counts, 1.0)
    return (logit_sum / counts[None, :]).astype(np.float32)


@torch.no_grad()
def batch_tta_from_paths(
    paths: list,
    image_root: str,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    tta_cfg: dict,
    normalization_cfg: dict = None,
    clahe_cfg: dict = None,
    bottom_crop_ratio: float = 0.0,
    view_types: torch.Tensor = None,
    use_fp16: bool = True,
    corner_erase_enabled: bool = True,
) -> np.ndarray:
    """Full TTA for a batch of images loaded from disk paths.

    Matches the preprocessing in predict_with_tta() exactly: EXIF transpose,
    CLAHE, bottom crop, corner erase, then all TTA transforms (multi-scale
    center crops + corner crops + selective horizontal flip).

    Args:
        paths: list of relative paths (as stored in the chest X-ray dataset).
        image_root: dataset root directory.
        model: model in eval mode.
        device: torch device.
        image_size: target image size.
        tta_cfg: TTA config dict from config.yaml.
        normalization_cfg: normalization config.
        clahe_cfg: CLAHE config.
        bottom_crop_ratio: bottom crop ratio.
        view_types: optional (N,) tensor of view position encodings.
        use_fp16: whether to use fp16 autocast.
    Returns:
        Merged logits array of shape (N, num_labels).
    """
    from torch.amp import autocast

    flip_labels = set(tta_cfg.get("flip_labels", []))
    tta_transforms = _build_tta_transforms(
        image_size, normalization_cfg=normalization_cfg, tta_cfg=tta_cfg,
    )

    # Load and preprocess all images once (raw numpy RGB arrays)
    raw_images = []
    for rel_path in paths:
        abs_path = os.path.join(image_root, rel_path)
        if not os.path.isfile(abs_path):
            # Fallback: strip first path component (legacy CheXpert format)
            parts = rel_path.split("/", 1)
            if len(parts) > 1:
                abs_path = os.path.join(image_root, parts[1])
        pil_image = Image.open(abs_path)
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = _apply_clahe_preprocessing(pil_image, clahe_cfg)
        image_np = np.array(pil_image.convert("RGB"))
        image_np = _apply_bottom_crop(image_np, bottom_crop_ratio)
        if corner_erase_enabled:
            h, w = image_np.shape[:2]
            cy, cx = max(1, int(h * 0.15)), max(1, int(w * 0.15))
            for ys, xs in [
                (slice(0, cy), slice(0, cx)),
                (slice(0, cy), slice(w - cx, w)),
                (slice(h - cy, h), slice(0, cx)),
                (slice(h - cy, h), slice(w - cx, w)),
            ]:
                image_np[ys, xs] = int(image_np[ys, xs].mean()) if image_np[ys, xs].size > 0 else 0
        raw_images.append(image_np)

    # Run each TTA transform on all images and collect logits
    logits_per_aug = []
    flip_flags = []
    vt = view_types.to(device) if view_types is not None else None

    for t, is_flipped in tta_transforms:
        tensors = []
        for img in raw_images:
            augmented = t(image=img)
            tensors.append(augmented["image"])
        batch = torch.stack(tensors).to(device)
        with autocast("cuda", enabled=(use_fp16 and device.type == "cuda")):
            logits = model(batch, view_type=vt).detach().cpu().float().numpy()
        logits_per_aug.append(logits)
        flip_flags.append(is_flipped)

    return _merge_tta_logits_batch(logits_per_aug, flip_flags, CHEXPERT_LABELS, flip_labels)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    labels: list = None,
    per_class_thresholds: dict = None,
    temperature: float = 1.0,
    view_type: torch.Tensor = None,
    per_label_temperatures: dict = None,
    per_label_isotonic: dict = None,
) -> dict:
   
    if labels is None:
        labels = CHEXPERT_LABELS

    image_tensor = image_tensor.to(device)
    if view_type is not None:
        view_type = view_type.to(device)
    output = model(image_tensor, view_type=view_type)
    # v9: model outputs logits, apply sigmoid to get probabilities
    probs = torch.sigmoid(output).cpu().numpy()[0]  # [num_classes]

    # Calibration: ưu tiên isotonic regression (per-label) > temperature scaling.
    # Mỗi label dùng calibrator phù hợp đã chọn lúc calibrate (lưu trong calibration.json).
    if per_label_isotonic or per_label_temperatures:
        eps = 1e-7
        scaled = np.empty_like(probs)
        iso_map_dict = per_label_isotonic or {}
        for j, label in enumerate(labels):
            iso_m = iso_map_dict.get(label)
            if iso_m and iso_m.get("x") and iso_m.get("y"):
                # Isotonic mapping
                x = np.asarray(iso_m["x"], dtype=np.float64)
                y = np.asarray(iso_m["y"], dtype=np.float64)
                scaled[j] = float(np.interp(probs[j], x, y))
            else:
                # Temperature scaling fallback
                T = float((per_label_temperatures or {}).get(label, temperature))
                if T == 1.0:
                    scaled[j] = probs[j]
                else:
                    p = np.clip(probs[j], eps, 1.0 - eps)
                    logit = np.log(p / (1.0 - p))
                    scaled[j] = 1.0 / (1.0 + np.exp(-logit / T))
        probs = scaled
    elif temperature != 1.0:
        scaler = TemperatureScaler(temperature)
        probs = scaler.scale(probs)

    probabilities = {label: float(prob) for label, prob in zip(labels, probs)}

    # Per-class thresholds
    if per_class_thresholds:
        predictions = {
            label: prob >= per_class_thresholds.get(label, threshold)
            for label, prob in zip(labels, probs)
        }
    else:
        predictions = {label: prob >= threshold for label, prob in zip(labels, probs)}

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "raw_output": probs,
    }


@torch.no_grad()
def predict_with_tta(
    model: torch.nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 320,
    threshold: float = 0.5,
    labels: list = None,
    per_class_thresholds: dict = None,
    temperature: float = 1.0,
    bottom_crop_ratio: float = 0.0,
    view_type: torch.Tensor = None,
    normalization_cfg: dict = None,
    clahe_cfg: dict = None,
    per_label_temperatures: dict = None,
    per_label_isotonic: dict = None,
    tta_cfg: dict = None,
) -> dict:
    """Laterality-aware TTA prediction for a single image.

    Runs multi-scale crops + optional corner crops.  Horizontal flip is
    only averaged into bilateral labels (configured via tta_cfg.flip_labels).
    Averaging is done in logit space before temperature scaling.
    """
    if labels is None:
        labels = CHEXPERT_LABELS
    if tta_cfg is None:
        tta_cfg = {}

    flip_labels = set(tta_cfg.get("flip_labels", []))

    pil_image = Image.open(image_path)
    pil_image = ImageOps.exif_transpose(pil_image)
    pil_image = _apply_clahe_preprocessing(pil_image, clahe_cfg)
    image_np = np.array(pil_image.convert("RGB"))
    image_np = _apply_bottom_crop(image_np, bottom_crop_ratio)

    # Corner erase (match preprocess_image)
    h, w = image_np.shape[:2]
    cy, cx = max(1, int(h * 0.15)), max(1, int(w * 0.15))
    for ys, xs in [
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(w - cx, w)),
        (slice(h - cy, h), slice(0, cx)),
        (slice(h - cy, h), slice(w - cx, w)),
    ]:
        image_np[ys, xs] = int(image_np[ys, xs].mean()) if image_np[ys, xs].size > 0 else 0

    tta_transforms = _build_tta_transforms(image_size, normalization_cfg=normalization_cfg, tta_cfg=tta_cfg)
    logits_list = []
    flip_flags = []

    for t, is_flipped in tta_transforms:
        augmented = t(image=image_np)
        tensor = augmented["image"].unsqueeze(0).to(device)
        if view_type is not None:
            output = model(tensor, view_type=view_type.to(device))
        else:
            output = model(tensor)
        logits_list.append(output.cpu().numpy()[0])
        flip_flags.append(is_flipped)

    # Laterality-aware merging in logit space
    avg_logits = _merge_tta_logits(logits_list, flip_flags, labels, flip_labels)

    # Calibration on merged logits → probabilities.
    # Per-label: isotonic regression (preferred when fitted) > temperature scaling.
    raw_probs = 1.0 / (1.0 + np.exp(-avg_logits))
    probs = np.empty_like(avg_logits)
    iso_map_dict = per_label_isotonic or {}
    if per_label_isotonic or per_label_temperatures:
        for j, label in enumerate(labels):
            iso_m = iso_map_dict.get(label)
            if iso_m and iso_m.get("x") and iso_m.get("y"):
                x = np.asarray(iso_m["x"], dtype=np.float64)
                y = np.asarray(iso_m["y"], dtype=np.float64)
                probs[j] = float(np.interp(raw_probs[j], x, y))
            else:
                T = float((per_label_temperatures or {}).get(label, temperature))
                probs[j] = 1.0 / (1.0 + np.exp(-avg_logits[j] / max(T, 1e-7)))
    elif temperature != 1.0:
        probs = 1.0 / (1.0 + np.exp(-avg_logits / max(temperature, 1e-7)))
    else:
        probs = raw_probs

    probabilities = {label: float(prob) for label, prob in zip(labels, probs)}

    if per_class_thresholds:
        predictions = {
            label: prob >= per_class_thresholds.get(label, threshold)
            for label, prob in zip(labels, probs)
        }
    else:
        predictions = {label: prob >= threshold for label, prob in zip(labels, probs)}

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "raw_output": probs,
    }


def predict_study(
    model: torch.nn.Module,
    image_paths: list,
    device: torch.device,
    image_size: int = 512,
    threshold: float = 0.5,
    labels: list = None,
    per_class_thresholds: dict = None,
    temperature: float = 1.0,
    bottom_crop_ratio: float = 0.0,
    aggregation: str = "max",
    normalization_cfg: dict = None,
    clahe_cfg: dict = None,
) -> dict:
    """
    Study-level prediction: chạy inference trên nhiều view (AP + PA + Lateral)
    của cùng 1 bệnh nhân rồi tổng hợp xác suất per-label.

    Chiến lược tổng hợp (aggregation):
    - 'max'  : max-pooling per label — nếu BẤT KỲ view nào detect bệnh thì coi là dương tính.
               Phù hợp với hầu hết pathology (Effusion, Cardiomegaly, v.v.)
    - 'mean' : trung bình đơn giản — ít dùng, phù hợp khi muốn conservative hơn.
    - 'mean_top2': trung bình 2 view có xác suất cao nhất cho mỗi label.

    Args:
        model: Model đã load, ở eval mode
        image_paths: Danh sách đường dẫn ảnh (AP, PA, Lateral, ...)
        device: CUDA / CPU device
        image_size: Kích thước ảnh sau resize
        threshold: Ngưỡng quyết định mặc định
        labels: Danh sách tên label (mặc định CHEXPERT_LABELS)
        per_class_thresholds: Dict label -> threshold tối ưu từ calibration
        temperature: Temperature scaling factor
        bottom_crop_ratio: Tỉ lệ cắt phần dưới ảnh
        aggregation: Chiến lược tổng hợp ('max', 'mean', 'mean_top2')

    Returns:
        Dict với các key:
        - 'probabilities': {label: prob} đã tổng hợp
        - 'predictions':   {label: bool} sau threshold
        - 'per_view':      [{label: prob}, ...] xác suất từng view riêng lẻ
        - 'raw_output':    np.ndarray shape (num_labels,)
    """
    if labels is None:
        labels = CHEXPERT_LABELS
    if len(image_paths) == 0:
        raise ValueError("predict_study: cần ít nhất 1 image_path")

    # --- Thu xác suất từng view ---
    per_view_probs = []
    for img_path in image_paths:
        image_tensor = preprocess_image(
            img_path,
            image_size,
            bottom_crop_ratio=bottom_crop_ratio,
            normalization_cfg=normalization_cfg,
            clahe_cfg=clahe_cfg,
        )
        result = predict(
            model, image_tensor, device, threshold,
            labels=labels,
            per_class_thresholds=None,   # áp dụng threshold sau khi tổng hợp
            temperature=temperature,
        )
        per_view_probs.append(result["raw_output"])  # np.ndarray (num_labels,)

    stacked = np.stack(per_view_probs, axis=0)  # (num_views, num_labels)

    # --- Tổng hợp per-label ---
    if aggregation == "max":
        agg_probs = np.max(stacked, axis=0)
    elif aggregation == "mean_top2":
        if stacked.shape[0] >= 2:
            # Sắp xếp theo chiều view, lấy top-2 view lớn nhất cho mỗi label
            sorted_views = np.sort(stacked, axis=0)[::-1]  # (num_views, num_labels) giảm dần
            agg_probs = np.mean(sorted_views[:2], axis=0)
        else:
            agg_probs = stacked[0]
    else:  # "mean"
        agg_probs = np.mean(stacked, axis=0)

    probabilities = {label: float(p) for label, p in zip(labels, agg_probs)}

    if per_class_thresholds:
        predictions = {
            label: float(p) >= per_class_thresholds.get(label, threshold)
            for label, p in zip(labels, agg_probs)
        }
    else:
        predictions = {label: float(p) >= threshold for label, p in zip(labels, agg_probs)}

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "per_view": [
            {label: float(p) for label, p in zip(labels, view_probs)}
            for view_probs in per_view_probs
        ],
        "raw_output": agg_probs,
    }


def predict_from_path(
    image_path: str,
    config: dict,
    device: torch.device = None,
    threshold: float = 0.5,
    use_tta: bool = False,
) -> dict:
   
    if device is None:
        device = get_device(config)

    model = load_trained_model(
        config["paths"]["densenet_checkpoint"], config, device
    )

    # Lấy per-class thresholds + temperature (ưu tiên calibration.json)
    per_class_thresholds = get_per_class_thresholds(config)
    temperature = _get_temperature(config)
    per_label_temperatures = _get_per_label_temperatures(config)
    per_label_isotonic = _get_per_label_isotonic(config)
    image_size = config["cnn"]["image_size"]
    normalization_cfg = config.get("cnn", {}).get("normalization", {})
    clahe_cfg = config.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})
    bottom_crop_ratio = float(
        config.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0)
    )

    if use_tta:
        tta_cfg = config.get("cnn", {}).get("tta", {})
        result = predict_with_tta(
            model, image_path, device, image_size, threshold,
            per_class_thresholds=per_class_thresholds,
            temperature=temperature,
            bottom_crop_ratio=bottom_crop_ratio,
            normalization_cfg=normalization_cfg,
            clahe_cfg=clahe_cfg,
            per_label_temperatures=per_label_temperatures if per_label_temperatures else None,
            per_label_isotonic=per_label_isotonic if per_label_isotonic else None,
            tta_cfg=tta_cfg,
        )
    else:
        image_tensor = preprocess_image(
            image_path,
            image_size,
            bottom_crop_ratio=bottom_crop_ratio,
            normalization_cfg=normalization_cfg,
            clahe_cfg=clahe_cfg,
        )
        result = predict(
            model, image_tensor, device, threshold,
            per_class_thresholds=per_class_thresholds,
            temperature=temperature,
            per_label_temperatures=per_label_temperatures if per_label_temperatures else None,
            per_label_isotonic=per_label_isotonic if per_label_isotonic else None,
        )
    return result


def predict_study_from_paths(
    image_paths: list,
    config: dict,
    device: torch.device = None,
    threshold: float = 0.5,
    aggregation: str = None,
) -> dict:
    """
    Convenience wrapper: load model + config rồi gọi predict_study().

    Args:
        image_paths: Danh sách đường dẫn ảnh (các view của 1 study)
        config: Config dict (từ load_config())
        device: CUDA / CPU device (None = auto-detect)
        threshold: Ngưỡng quyết định mặc định
        aggregation: Chiến lược tổng hợp ('max', 'mean', 'mean_top2').
                     None → đọc từ config['cnn']['study_aggregation'] → mặc định 'max'

    Returns:
        Kết quả từ predict_study() (probabilities, predictions, per_view, raw_output)
    """
    if device is None:
        device = get_device(config)

    model = load_trained_model(
        config["paths"]["densenet_checkpoint"], config, device
    )

    per_class_thresholds = get_per_class_thresholds(config)
    temperature = _get_temperature(config)
    image_size = config["cnn"]["image_size"]
    bottom_crop_ratio = float(
        config.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0)
    )
    normalization_cfg = config.get("cnn", {}).get("normalization", {})
    clahe_cfg = config.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})
    if aggregation is None:
        aggregation = config.get("cnn", {}).get("study_aggregation", "max")

    return predict_study(
        model, image_paths, device,
        image_size=image_size,
        threshold=threshold,
        per_class_thresholds=per_class_thresholds,
        temperature=temperature,
        bottom_crop_ratio=bottom_crop_ratio,
        aggregation=aggregation,
        normalization_cfg=normalization_cfg,
        clahe_cfg=clahe_cfg,
    )


def format_prediction(result: dict, config: dict = None, top_k: int = 5) -> str:
    
    probs = result["probabilities"]

    # Sort theo xác suất giảm dần
    sorted_labels = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    lines = ["=" * 50, "KẾT QUẢ PHÂN LOẠI X-QUANG NGỰC", "=" * 50]

    labels_vi = {}
    if config:
        labels_vi = config.get("labels_vi", {})

    for i, (label, prob) in enumerate(sorted_labels[:top_k]):
        vi_name = labels_vi.get(label, "")
        bar = "█" * int(prob * 30)
        status = "⚠️" if prob >= 0.5 else "  "
        if vi_name:
            lines.append(f"{status} {label} ({vi_name}): {prob:.1%} {bar}")
        else:
            lines.append(f"{status} {label}: {prob:.1%} {bar}")

    return "\n".join(lines)
