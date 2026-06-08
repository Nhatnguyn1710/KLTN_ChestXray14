
import os
import sys
import argparse
import contextlib
import warnings
import numpy as np
import torch
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    from pytorch_grad_cam import XGradCAM
except Exception:
    XGradCAM = None

try:
    from pytorch_grad_cam import HiResCAM
except Exception:
    HiResCAM = None

try:
    from pytorch_grad_cam import GradCAMPlusPlus
except Exception:
    GradCAMPlusPlus = None

try:
    from pytorch_grad_cam import EigenCAM
except Exception:
    EigenCAM = None

try:
    from pytorch_grad_cam import LayerCAM
except Exception:
    LayerCAM = None

try:
    from pytorch_grad_cam import ScoreCAM
except Exception:
    ScoreCAM = None


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import load_config, get_device, ensure_dir
from src.cnn.model import load_trained_model
from src.cnn.inference import preprocess_image, load_image_rgb

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]


def _resolve_colormap(colormap_name: str) -> int:
    """Map config colormap name to OpenCV colormap id."""
    name = str(colormap_name or "jet").strip().lower()
    mapping = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }
    if hasattr(cv2, "COLORMAP_TURBO"):
        mapping["turbo"] = cv2.COLORMAP_TURBO
    return mapping.get(name, cv2.COLORMAP_JET)


def _resolve_cam_class(cam_method_name: str):
    """Resolve CAM implementation by name with safe fallback."""
    name = str(cam_method_name or "gradcam").strip().lower()
    mapping = {
        "gradcam": GradCAM,
        "xgradcam": XGradCAM,
        "hirescam": HiResCAM,
        "gradcam++": GradCAMPlusPlus,
        "gradcampp": GradCAMPlusPlus,
        # EigenCAM: no gradients needed — uses PCA of feature maps.
        # Much more stable for GAP-based networks where gradients are uniform.
        "eigencam": EigenCAM,
        # LayerCAM: element-wise ReLU(grad) * activation — sharper than GradCAM.
        "layercam": LayerCAM,
        # ScoreCAM: perturbation-based, most faithful but slow (N forward passes).
        "scorecam": ScoreCAM,
    }
    cam_cls = mapping.get(name, GradCAM)
    if cam_cls is None:
        return GradCAM, "gradcam"
    return cam_cls, name


def _get_target_layer(model, layer_name: str):
    """Get reference to a target layer in the model by dotted name."""
    parts = layer_name.split(".")
    layer = model.backbone if hasattr(model, "backbone") else model
    for part in parts:
        if hasattr(layer, part):
            layer = getattr(layer, part)
        else:
            raise AttributeError(f"Layer '{part}' not found in model. Full path: '{layer_name}'")
    return layer


def _body_mask(original_np: np.ndarray) -> np.ndarray:
    
    h, w = original_np.shape[:2]
    gray = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    _, fg = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep only large connected components (the body)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = int(0.05 * h * w)
    mask = np.zeros((h, w), dtype=bool)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            mask |= (labels == i)

    # Fallback: if mask still covers >92% (full-frame image with no real background),
    # keep as-is — suppression relies on corner/top-strip zeroing instead.
    if mask.sum() == 0:
        mask = fg > 0
    return mask


def _approx_lung_mask(
    original_np: np.ndarray,
    include_mediastinum: bool = True,
    dilate_iter: int = 1,
    blur_kernel: int = 31,
    side_margin_ratio: float = 0.04,
    apex_guard_ratio: float = 0.14,
) -> np.ndarray:
    
    h, w = original_np.shape[:2]
    gray = cv2.cvtColor((np.clip(original_np, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    roi = np.zeros((h, w), dtype=np.uint8)

    # Lung priors: tapered polygons reduce shoulder/arm leakage compared with
    # wide ellipses, while still covering lower pleural bases.
    # Apex starts at h*0.22 to exclude clavicle/subclavian region.
    left_poly = np.array([
        [int(w * 0.18), int(h * 0.22)],
        [int(w * 0.36), int(h * 0.22)],
        [int(w * 0.46), int(h * 0.36)],
        [int(w * 0.43), int(h * 0.78)],
        [int(w * 0.28), int(h * 0.88)],
        [int(w * 0.12), int(h * 0.72)],
        [int(w * 0.10), int(h * 0.36)],
    ], dtype=np.int32)
    right_poly = np.array([
        [int(w * 0.64), int(h * 0.22)],
        [int(w * 0.82), int(h * 0.22)],
        [int(w * 0.90), int(h * 0.36)],
        [int(w * 0.88), int(h * 0.72)],
        [int(w * 0.72), int(h * 0.88)],
        [int(w * 0.57), int(h * 0.78)],
        [int(w * 0.54), int(h * 0.36)],
    ], dtype=np.int32)
    cv2.fillPoly(roi, [left_poly], 255)
    cv2.fillPoly(roi, [right_poly], 255)
    if include_mediastinum:
        cv2.rectangle(
            roi,
            (int(w * 0.38), int(h * 0.22)),
            (int(w * 0.62), int(h * 0.80)),
            255, -1,
        )

    mask = cv2.bitwise_and(dark, roi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iter))

    # Fallback to the ROI prior if thresholding becomes too sparse.
    if float((mask > 0).mean()) < 0.05:
        mask = roi.copy()

    soft = (mask.astype(np.float32) / 255.0)
    blur_kernel = max(3, int(blur_kernel) | 1)
    soft = cv2.GaussianBlur(soft, (blur_kernel, blur_kernel), 0)

    # Suppress shoulders and image rims. Keep it soft, not binary.
    side_margin_ratio = max(0.0, min(0.12, float(side_margin_ratio)))
    apex_guard_ratio = max(0.0, min(0.50, float(apex_guard_ratio)))  # allow up to 50% top suppression
    if side_margin_ratio > 0.0:
        side_margin = max(1, int(w * side_margin_ratio))
        soft[:, :side_margin] *= np.linspace(0.05, 1.0, side_margin, dtype=np.float32)[None, :]
        soft[:, -side_margin:] *= np.linspace(1.0, 0.05, side_margin, dtype=np.float32)[None, :]
    if apex_guard_ratio > 0.0:
        apex_rows = max(1, int(h * apex_guard_ratio))
        top_fade = np.linspace(0.0, 1.0, apex_rows, dtype=np.float32)[:, None]
        soft[:apex_rows, :] *= top_fade

    max_val = float(soft.max())
    if max_val > 1e-8:
        soft = soft / max_val
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _get_pcam_attention_map(model, image_tensor) -> np.ndarray:
    """Extract PCAM attention map [H, W] via forward hook. Returns float32 [0,1] or None."""
    if not (hasattr(model, 'use_pcam') and model.use_pcam and hasattr(model, 'pcam_pool')):
        return None
    out = []
    hook = model.pcam_pool.attention.register_forward_hook(
        lambda m, i, o: out.append(torch.sigmoid(o).detach().cpu())
    )
    try:
        with torch.no_grad():
            model(image_tensor)
    finally:
        hook.remove()
    if not out:
        return None
    return out[0][0, 0].float().numpy()  # [H, W] in [0, 1]


def _generate_direct_cam(
    model,
    image_tensor,
    target_class_idx: int,
    apply_body_mask: bool = True,
) -> np.ndarray:
    """Direct CAM using classifier weights × feature activations (Zhou et al. 2016).

    FC-CAM only (Row B): W_c = W2[c] @ W1 × ReLU(features).
    Class-specific WITHOUT PCAM attention multiplication.
    PCAM attn creates a fixed spatial prior that pulls all classes to the same
    peak location → "lệch phổi" drift. Removing it keeps class discrimination
    while allowing spatially distributed heatmaps.

    W_c = combined weight vector for class c: W2[c] @ W1 → shape [1024]
    Returns float32 [H, W] raw scores (non-normalized).
    """
    feat_out = []

    hook_feat = model.backbone.features.register_forward_hook(
        lambda m, i, o: feat_out.append(torch.nn.functional.relu(o, inplace=False).detach().cpu())
    )

    try:
        with torch.no_grad():
            model(image_tensor)
    finally:
        hook_feat.remove()

    if not feat_out:
        return None

    feat = feat_out[0][0]  # [1024, H, W]

    if apply_body_mask and bool(getattr(model, "use_lung_mask", False)):
        with torch.no_grad():
            lung_mask = model._compute_body_mask(
                image_tensor, feat.shape[1], feat.shape[2]
            )
            lung_mask = lung_mask[0, 0].detach().cpu()  # [H, W]
        feat = feat * lung_mask

    
    w1 = model.classifier_head[0].weight.detach().cpu()  # [512, 1024 or 1025]
    w2 = model.classifier_head[-1].weight.detach().cpu()  # [14, 512]
    w_class = w2[target_class_idx]  # [512]
    w_combined = (w_class @ w1)[:1024]  # [1024] — drop view_position dim if present

    # CAM = ReLU(Σ_k w_k × f_k(i,j))  — NO PCAM multiply
    cam = torch.einsum('k,khw->hw', w_combined, feat).numpy()  # [H, W]
    cam = np.maximum(cam, 0)  # ReLU

    return cam.astype(np.float32)


def percentile_normalize(cam: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    """Normalize CAM to [0, 1] using percentile clipping on non-zero values."""
    nz = cam[cam > 1e-8]
    if nz.size == 0:
        return cam.copy().astype(np.float32)
    lo_val = float(np.percentile(nz, lo))
    hi_val = float(np.percentile(nz, hi))
    if hi_val - lo_val < 1e-8:
        m = float(cam.max())
        return (cam / m if m > 1e-8 else cam.copy()).astype(np.float32)
    result = (cam - lo_val) / (hi_val - lo_val)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _suppress_small_hotspots(
    cam_norm: np.ndarray,
    activation_threshold: float = 0.18,
    min_area_ratio: float = 0.0015,
) -> np.ndarray:
    """Suppress tiny isolated activation islands ("speckles") in CAM.

    Keeps only connected components above min_area_ratio among pixels whose
    activation exceeds activation_threshold.
    """
    if cam_norm is None or cam_norm.size == 0:
        return cam_norm
    activation_threshold = float(np.clip(activation_threshold, 0.0, 1.0))
    min_area_ratio = float(max(0.0, min_area_ratio))
    if activation_threshold <= 0.0 or min_area_ratio <= 0.0:
        return cam_norm

    h, w = cam_norm.shape[:2]
    binary = (cam_norm >= activation_threshold).astype(np.uint8)
    if int(binary.sum()) == 0:
        return cam_norm

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(1, int(h * w * min_area_ratio))
    keep_mask = np.zeros((h, w), dtype=bool)
    for i in range(1, num_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            keep_mask |= (labels == i)

    if not keep_mask.any():
        return cam_norm

    suppress_mask = (binary > 0) & (~keep_mask)
    if not suppress_mask.any():
        return cam_norm

    out = cam_norm.copy()
    out[suppress_mask] = 0.0
    return out


def _edge_guided_upsample(cam: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
    """Edge-aware upsampling: use joint bilateral filter to align CAM boundaries
    with anatomical edges from the original X-ray image.

    This prevents heatmap blobs from bleeding across organ boundaries
    (e.g. heart edge, lung boundary, diaphragm).
    """
    h, w = guide_rgb.shape[:2]
    cam_h, cam_w = cam.shape[:2]
    if (cam_h, cam_w) == (h, w):
        cam_up = cam
    else:
        cam_up = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert guide to grayscale for edge information
    guide_gray = cv2.cvtColor(
        (np.clip(guide_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )

    # Joint bilateral filter: smooths CAM spatially BUT preserves edges from guide
    # d=0 → auto-compute from sigmaSpace; sigmaColor=25 → respect edges;
    # sigmaSpace=9 → moderate spatial smoothing.
    if hasattr(cv2, 'ximgproc'):
        cam_filtered = cv2.ximgproc.jointBilateralFilter(
            guide_gray, cam_up.astype(np.float32), d=0, sigmaColor=25, sigmaSpace=9
        )
    else:
        # Fallback: standard bilateral filter on the CAM itself guided by its own edges
        # Plus edge-weighted blending with Canny edges from the X-ray
        cam_uint8 = np.uint8(np.clip(cam_up, 0, 1) * 255)
        cam_bilateral = cv2.bilateralFilter(cam_uint8, d=9, sigmaColor=30, sigmaSpace=9)
        # Detect anatomy edges from X-ray and slightly sharpen CAM at those boundaries
        edges = cv2.Canny(guide_gray, 30, 100)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        edge_mask = (edges_dilated > 0).astype(np.float32)
        # At edge pixels, prefer the raw (non-smoothed) CAM to preserve boundary alignment
        cam_filtered = cam_bilateral.astype(np.float32) / 255.0
        cam_raw_norm = np.clip(cam_up, 0, 1)
        cam_filtered = cam_filtered * (1.0 - edge_mask * 0.5) + cam_raw_norm * (edge_mask * 0.5)

    return np.clip(cam_filtered, 0, None).astype(np.float32)


def _add_contour_overlay(result_img: np.ndarray, cam_norm: np.ndarray,
                         levels: list = None, color: tuple = (255, 255, 255),
                         thickness: int = 1, alpha: float = 0.6) -> np.ndarray:
    """Draw subtle contour lines at key activation levels for precise localization.

    Adds thin white contour lines at specified CAM thresholds (e.g. 50%, 75%)
    so clinicians can see exact boundaries of high-activation zones.
    """
    if levels is None:
        levels = [0.5, 0.75]
    h, w = result_img.shape[:2]
    overlay = result_img.copy()

    cam_uint8 = np.uint8(np.clip(cam_norm, 0, 1) * 255)
    for level in levels:
        thresh_val = int(level * 255)
        _, binary = cv2.threshold(cam_uint8, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter out tiny contours (noise)
        min_area = int(0.001 * h * w)
        contours = [c for c in contours if cv2.contourArea(c) > min_area]
        cv2.drawContours(overlay, contours, -1, color, thickness, cv2.LINE_AA)

    result = cv2.addWeighted(result_img, 1.0 - alpha, overlay, alpha, 0)
    return result


def _resolve_target_layers(model, layer_spec):
    """Resolve one or multiple target layers from config."""
    if isinstance(layer_spec, (list, tuple)):
        names = [str(x).strip() for x in layer_spec if str(x).strip()]
    elif isinstance(layer_spec, str):
        if "," in layer_spec:
            names = [s.strip() for s in layer_spec.split(",") if s.strip()]
        else:
            names = [layer_spec.strip()]
    else:
        names = ["features.denseblock4.denselayer16.conv2"]

    if not names:
        names = ["features.denseblock4.denselayer16.conv2"]
    return [_get_target_layer(model, name) for name in names]


def _gradcam_cfg_for_label(base_cfg: dict, label: str = None) -> dict:
    """Merge optional per-label Grad-CAM overrides into the base config."""
    merged = dict(base_cfg or {})
    if not label or not isinstance(base_cfg, dict):
        return merged
    overrides = base_cfg.get("per_label_overrides", {})
    if not isinstance(overrides, dict):
        return merged
    label_override = overrides.get(label, {})
    if isinstance(label_override, dict):
        merged.update(label_override)
    return merged


# ---------------------------------------------------------------------------
# Quality assessment: anatomical zone masks & spatial metrics
# ---------------------------------------------------------------------------

def _fill_lung_zone_polygons(mask: np.ndarray, h: int, w: int, value: float = 1.0):
    """Fill generous lung-field polygons on *mask*.

    Slightly wider than ``_approx_lung_mask`` polygons to account for
    patient-size variation when evaluating quality (not for suppression).
    """
    left_poly = np.array([
        [int(w * 0.08), int(h * 0.18)],
        [int(w * 0.42), int(h * 0.18)],
        [int(w * 0.50), int(h * 0.34)],
        [int(w * 0.48), int(h * 0.84)],
        [int(w * 0.30), int(h * 0.94)],
        [int(w * 0.06), int(h * 0.76)],
        [int(w * 0.04), int(h * 0.30)],
    ], dtype=np.int32)
    right_poly = np.array([
        [int(w * 0.58), int(h * 0.18)],
        [int(w * 0.92), int(h * 0.18)],
        [int(w * 0.96), int(h * 0.30)],
        [int(w * 0.94), int(h * 0.76)],
        [int(w * 0.70), int(h * 0.94)],
        [int(w * 0.52), int(h * 0.84)],
        [int(w * 0.50), int(h * 0.34)],
    ], dtype=np.int32)
    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tmp, [left_poly, right_poly], 255)
    mask[tmp > 0] = np.maximum(mask[tmp > 0], value)


def _label_expected_zone_mask(label: str, h: int, w: int) -> np.ndarray:
    """Soft mask of the anatomically expected activation zone for *label*.

    Returns float32 ``[h, w]`` in ``[0, 1]`` where 1.0 = expected region.
    Zones are intentionally generous — the goal is to catch gross
    mis-localisations, not to enforce pixel-perfect anatomy.
    """
    mask = np.zeros((h, w), dtype=np.float32)

    if label == "Cardiomegaly":
        # Cardiac silhouette: centre–left chest, mid-vertical band
        y1, y2 = int(h * 0.22), int(h * 0.82)
        x1, x2 = int(w * 0.25), int(w * 0.75)
        mask[y1:y2, x1:x2] = 1.0

    elif label == "Effusion":
        # Costophrenic angles + lung bases (bottom ~50 %)
        y_full = int(h * 0.50)
        mask[y_full:, :] = 1.0
        # Soft fade 40-50 % to allow large effusions extending higher
        y_fade = int(h * 0.40)
        if y_full > y_fade:
            fade = np.linspace(0.0, 1.0, y_full - y_fade, dtype=np.float32)[:, None]
            mask[y_fade:y_full, :] = np.broadcast_to(fade, (y_full - y_fade, w))

    elif label == "Edema":
        # Bilateral perihilar — butterfly pattern, very broad
        y1, y2 = int(h * 0.15), int(h * 0.85)
        x1, x2 = int(w * 0.08), int(w * 0.92)
        mask[y1:y2, x1:x2] = 1.0

    elif label == "Pneumothorax":
        # Apical + peripheral lung regions
        y_lower = int(h * 0.55)
        mask[:y_lower, :] = 1.0
        # Periphery below midline
        side = int(w * 0.22)
        mask[y_lower:, :side] = 0.5
        mask[y_lower:, -side:] = 0.5

    elif label == "Consolidation":
        # Lung parenchyma (both fields + retrocardiac)
        _fill_lung_zone_polygons(mask, h, w, value=1.0)
        mx1, mx2 = int(w * 0.32), int(w * 0.68)
        my1, my2 = int(h * 0.20), int(h * 0.82)
        mask[my1:my2, mx1:mx2] = np.maximum(mask[my1:my2, mx1:mx2], 0.7)

    elif label == "Atelectasis":
        # Lower lung + hilar — but can appear anywhere
        _fill_lung_zone_polygons(mask, h, w, value=0.7)
        y_mid = int(h * 0.45)
        mask[y_mid:, :] = np.maximum(mask[y_mid:, :], 0.5)
        mx1, mx2 = int(w * 0.30), int(w * 0.70)
        my1, my2 = int(h * 0.25), int(h * 0.80)
        mask[my1:my2, mx1:mx2] = np.maximum(mask[my1:my2, mx1:mx2], 0.6)

    elif label == "Infiltration":
        # Patchy / interstitial opacities — broad lung field coverage
        _fill_lung_zone_polygons(mask, h, w, value=1.0)

    elif label == "Mass":
        # Focal lesion >3cm — anywhere in lung fields, often peripheral
        _fill_lung_zone_polygons(mask, h, w, value=1.0)

    elif label == "Nodule":
        # Focal lesion <3cm — anywhere in lung fields
        _fill_lung_zone_polygons(mask, h, w, value=1.0)

    elif label == "Pneumonia":
        # Lobar / segmental consolidation — lung parenchyma
        _fill_lung_zone_polygons(mask, h, w, value=1.0)
        mx1, mx2 = int(w * 0.30), int(w * 0.70)
        my1, my2 = int(h * 0.25), int(h * 0.85)
        mask[my1:my2, mx1:mx2] = np.maximum(mask[my1:my2, mx1:mx2], 0.7)

    elif label == "Emphysema":
        # Hyperinflated lungs — bilateral, upper lobe predominant
        _fill_lung_zone_polygons(mask, h, w, value=1.0)
        y1, y2 = int(h * 0.15), int(h * 0.55)
        mask[y1:y2, :] = np.maximum(mask[y1:y2, :], 0.9)

    elif label == "Fibrosis":
        # Reticular / honeycomb pattern — bilateral lower zones predominant
        _fill_lung_zone_polygons(mask, h, w, value=0.8)
        y_mid = int(h * 0.55)
        mask[y_mid:, :] = np.maximum(mask[y_mid:, :], 1.0)

    elif label == "Pleural_Thickening":
        # Pleural surfaces — periphery of thorax
        side = int(w * 0.20)
        mask[:, :side] = 1.0
        mask[:, -side:] = 1.0
        y_lower = int(h * 0.55)
        mask[y_lower:, :] = np.maximum(mask[y_lower:, :], 0.7)

    elif label == "Hernia":
        # Diaphragmatic hernia — lower mediastinal / sub-diaphragmatic
        y1, y2 = int(h * 0.55), int(h * 0.95)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        mask[y1:y2, x1:x2] = 1.0

    else:
        mask[:, :] = 1.0

    return mask


def _zone_activation_sums(cam_norm: np.ndarray, h: int, w: int) -> dict:
    """Return coarse anatomical zone activation sums for CAM QC/UI display."""
    zones = {}

    # Heart / mediastinum
    hm = np.zeros((h, w), dtype=bool)
    hm[int(h * 0.25):int(h * 0.78), int(w * 0.33):int(w * 0.67)] = True
    zones["heart_mediastinum"] = float(cam_norm[hm].sum())

    # Left lung regions
    left_upper = np.zeros((h, w), dtype=bool)
    left_upper[int(h * 0.15):int(h * 0.50), :int(w * 0.48)] = True
    left_upper &= ~hm
    zones["left_upper"] = float(cam_norm[left_upper].sum())

    left_mid = np.zeros((h, w), dtype=bool)
    left_mid[int(h * 0.50):int(h * 0.70), :int(w * 0.48)] = True
    left_mid &= ~hm
    zones["left_mid"] = float(cam_norm[left_mid].sum())

    left_lower = np.zeros((h, w), dtype=bool)
    left_lower[int(h * 0.70):, :int(w * 0.48)] = True
    zones["left_lower"] = float(cam_norm[left_lower].sum())

    # Right lung regions
    right_upper = np.zeros((h, w), dtype=bool)
    right_upper[int(h * 0.15):int(h * 0.50), int(w * 0.52):] = True
    right_upper &= ~hm
    zones["right_upper"] = float(cam_norm[right_upper].sum())

    right_mid = np.zeros((h, w), dtype=bool)
    right_mid[int(h * 0.50):int(h * 0.70), int(w * 0.52):] = True
    right_mid &= ~hm
    zones["right_mid"] = float(cam_norm[right_mid].sum())

    right_lower = np.zeros((h, w), dtype=bool)
    right_lower[int(h * 0.70):, int(w * 0.52):] = True
    zones["right_lower"] = float(cam_norm[right_lower].sum())

    # Top corners (artefact region)
    tc = np.zeros((h, w), dtype=bool)
    cy, cx = int(h * 0.18), int(w * 0.25)
    tc[:cy, :cx] = True
    tc[:cy, -cx:] = True
    zones["top_corners"] = float(cam_norm[tc].sum())

    # Device corridors (central + upper mediastinum)
    dc = np.zeros((h, w), dtype=bool)
    dc[int(h * 0.05):int(h * 0.25), int(w * 0.30):int(w * 0.70)] = True
    zones["device_corridors"] = float(cam_norm[dc].sum())

    return zones


def _zone_activation_ratios(cam_norm: np.ndarray, h: int, w: int) -> dict:
    """Return activation ratios by coarse anatomical zone."""
    total = float(cam_norm.sum())
    if total < 1e-6:
        return {}
    zones = _zone_activation_sums(cam_norm, h, w)
    ratios = {name: round(float(value) / total, 4) for name, value in zones.items()}
    lung_keys = ("left_upper", "left_mid", "left_lower", "right_upper", "right_mid", "right_lower")
    ratios["lungs"] = round(sum(float(zones.get(k, 0.0)) for k in lung_keys) / total, 4)
    return ratios


def _determine_primary_zone(cam_norm: np.ndarray, h: int, w: int) -> str:
    """Identify which anatomical zone contains the most CAM activation."""
    total = float(cam_norm.sum())
    if total < 1e-6:
        return "none"

    zones = _zone_activation_sums(cam_norm, h, w)
    lungs_total = (
        zones["left_upper"] + zones["left_mid"] + zones["left_lower"]
        + zones["right_upper"] + zones["right_mid"] + zones["right_lower"]
    )

    # Pick highest individual zone
    primary_base = max(zones, key=zones.get)
    if primary_base in ("top_corners", "device_corridors", "heart_mediastinum"):
        return primary_base
    # Summarise lung sub-zones
    if lungs_total > zones["heart_mediastinum"]:
        return "lungs"
    return primary_base


# Minimum expected-zone ratio per label for "plausible" classification
_QUALITY_THRESHOLDS = {
    "Cardiomegaly": 0.35,
    "Effusion": 0.28,
    "Edema": 0.35,
    "Pneumothorax": 0.25,
    "Consolidation": 0.40,
    "Atelectasis": 0.30,
    "Infiltration": 0.35,
    "Mass": 0.30,
    "Nodule": 0.30,
    "Pneumonia": 0.40,
    "Emphysema": 0.35,
    "Fibrosis": 0.35,
    "Pleural_Thickening": 0.30,
    "Hernia": 0.30,
}


def _assess_cam_quality(
    cam_norm: np.ndarray,
    label: str,
    h: int,
    w: int,
    body_mask: np.ndarray = None,
) -> dict:
    """Assess spatial quality of a post-processed CAM activation map.

    Returns a dict with keys ``classification``, ``reason``,
    ``primary_zone``, and ``final`` (sub-dict of numeric metrics) that the
    large-scale evaluation script expects.
    """
    total = float(cam_norm.sum())
    if total < 1e-6:
        return {
            "classification": "no_activation",
            "reason": "No significant CAM activation",
            "primary_zone": "none",
            "primary_fraction": 0.0,
            "zone_activations": {},
            "final": {
                "expected_zone_ratio": 0.0,
                "outside_thorax_ratio": 0.0,
                "focus_in_body_ratio": 0.0,
                "focus_on_edges_ratio": 0.0,
                "focus_on_top_corners_ratio": 0.0,
                "focus_on_bottom_ratio": 0.0,
            },
        }

    # Expected zone overlap
    zone_mask = _label_expected_zone_mask(label, h, w)
    expected_zone_ratio = float((cam_norm * zone_mask).sum()) / total

    # Outside-thorax ratio
    outside_thorax_ratio = 0.0
    if body_mask is not None:
        inv = ~body_mask if body_mask.dtype == bool else (body_mask < 0.5)
        outside_thorax_ratio = float(cam_norm[inv].sum()) / total
    focus_in_body_ratio = max(0.0, min(1.0, 1.0 - outside_thorax_ratio))

    # Top-corner artefact ratio
    cy, cx = int(h * 0.18), int(w * 0.25)
    corner_mask = np.zeros((h, w), dtype=bool)
    corner_mask[:cy, :cx] = True
    corner_mask[:cy, -cx:] = True
    corner_ratio = float(cam_norm[corner_mask].sum()) / total

    edge = max(1, int(min(h, w) * 0.06))
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:edge, :] = True
    edge_mask[-edge:, :] = True
    edge_mask[:, :edge] = True
    edge_mask[:, -edge:] = True
    edge_ratio = float(cam_norm[edge_mask].sum()) / total

    bottom_mask = np.zeros((h, w), dtype=bool)
    bottom_mask[int(h * 0.66):, :] = True
    bottom_ratio = float(cam_norm[bottom_mask].sum()) / total

    primary_zone = _determine_primary_zone(cam_norm, h, w)
    zone_activations = _zone_activation_ratios(cam_norm, h, w)
    primary_fraction = float(zone_activations.get(primary_zone, 0.0))

    # Classification
    threshold = _QUALITY_THRESHOLDS.get(label, 0.35)
    if outside_thorax_ratio > 0.15:
        classification = "pipeline_related"
        reason = f"High outside-thorax activation ({outside_thorax_ratio:.1%})"
    elif corner_ratio > 0.20:
        classification = "pipeline_related"
        reason = f"High corner artefact activation ({corner_ratio:.1%})"
    elif expected_zone_ratio >= threshold:
        classification = "plausible"
        reason = f"Expected zone ratio {expected_zone_ratio:.1%} >= {threshold:.0%}"
    else:
        classification = "model_related"
        reason = f"Expected zone ratio {expected_zone_ratio:.1%} < {threshold:.0%}"

    return {
        "classification": classification,
        "reason": reason,
        "primary_zone": primary_zone,
        "primary_fraction": round(primary_fraction, 4),
        "zone_activations": zone_activations,
        "final": {
            "expected_zone_ratio": round(expected_zone_ratio, 4),
            "outside_thorax_ratio": round(outside_thorax_ratio, 4),
            "focus_in_body_ratio": round(focus_in_body_ratio, 4),
            "focus_on_edges_ratio": round(edge_ratio, 4),
            "focus_on_top_corners_ratio": round(corner_ratio, 4),
            "focus_on_bottom_ratio": round(bottom_ratio, 4),
        },
    }


def generate_gradcam(
    model: torch.nn.Module,
    image_path: str,
    target_class_idx: int,
    image_size: int = 224,
    target_layer_name: str = "features.denseblock4.denselayer16.conv2",
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
    postprocess_cfg: dict = None,
    cam_method: str = "gradcam",
    bottom_crop_ratio: float = 0.0,
    view_type: torch.Tensor = None,
    normalization_cfg: dict = None,
    clahe_cfg: dict = None,
    corner_erase_enabled: bool = True,
    label_name: str = None,
) -> dict:
    """Generate Grad-CAM heatmap (clean pipeline — no spatial priors or heuristics).

    Steps:
    1. PCAM→GAP context switch (keeps model.gradcam_mode if present).
    2. Hook target Conv layer and compute GradCAM.
    3. Body mask — zero black background only.
    4. Gaussian smooth (kernel from config smooth_kernel, default 7).
    5. Percentile normalize (lo=1, hi=99) — robust to outlier activations.
    6. Intensity-proportional alpha blend (cam^1.2 * alpha).

    Returns a dict with keys:
        "heatmap_overlay" : np.ndarray uint8 RGB [H, W, 3]
        "grayscale_cam"   : np.ndarray float32 [H, W] in [0, 1]
        "raw_cam_overlay" : np.ndarray uint8 RGB or None (when save_raw_cam=true)
    """
    model.eval()
    target_layers = _resolve_target_layers(model, target_layer_name)
    crop_ratio = max(0.0, min(0.4, float(bottom_crop_ratio or 0.0)))
    image_tensor = preprocess_image(
        image_path,
        image_size,
        bottom_crop_ratio=crop_ratio,
        normalization_cfg=normalization_cfg,
        clahe_cfg=clahe_cfg,
        corner_erase_enabled=corner_erase_enabled,
    )
    model_device = next(model.parameters()).device
    image_tensor = image_tensor.to(model_device)

    # Load original image for overlay
    original_image_full = load_image_rgb(image_path)
    original_np_full = np.array(original_image_full).astype(np.float32) / 255.0
    display_np = original_np_full
    if crop_ratio > 0.0:
        original_image_work = original_image_full.crop(
            (0, 0, original_image_full.width,
             max(1, int(round(original_image_full.height * (1.0 - crop_ratio)))))
        )
    else:
        original_image_work = original_image_full

    # PCAM→GAP context switch: prevents attention collapse into text/device artifacts.
    use_gap_for_gradcam = True
    cam_method_name = cam_method
    if isinstance(postprocess_cfg, dict):
        use_gap_for_gradcam = bool(postprocess_cfg.get("use_gap_for_gradcam", True))
        cam_method_name = str(postprocess_cfg.get("cam_method", cam_method_name))

    cam_context = contextlib.nullcontext()
    if use_gap_for_gradcam and hasattr(model, "gradcam_mode"):
        cam_context = model.gradcam_mode()

    # If model expects view_position but caller didn't supply one, default to AP (0.0).
    # NIH/CheXpert frontal X-rays are predominantly AP. Using 0.5 (unknown) biases gradients
    # because the classifier head was NEVER trained with 0.5 — only 0.0 or 1.0.
    if view_type is None and hasattr(model, "use_view_position") and model.use_view_position:
        model_device = next(model.parameters()).device
        view_type = torch.zeros(1, 1, device=model_device)

    view_context = contextlib.nullcontext()
    if view_type is not None and not hasattr(model, "gradcam_view_context"):
        warnings.warn("view_type passed but model has no gradcam_view_context — ignored.")
    if hasattr(model, "gradcam_view_context"):
        view_context = model.gradcam_view_context(view_type)

    if cam_method_name == "cam":
        try:
            direct_cam_apply_body_mask = bool(
                (postprocess_cfg or {}).get("direct_cam_apply_body_mask", True)
            )
        except (TypeError, ValueError):
            direct_cam_apply_body_mask = True
        # Direct CAM (Zhou et al. 2016) — truly class-specific for GAP/PCAM models.
        # Uses combined classifier weights W_c = W2[c] @ W1 directly; no gradients.
        # Bypasses the GradCAM-through-PCAM limitation where \u2202logit_c/\u2202f(i,j) \u221d attn(i,j)
        # regardless of class, making all heatmaps look identical.
        with cam_context, view_context:
            grayscale_cam = _generate_direct_cam(
                model,
                image_tensor,
                target_class_idx,
                apply_body_mask=direct_cam_apply_body_mask,
            )
        if grayscale_cam is None:
            grayscale_cam = np.zeros((16, 16), dtype=np.float32)
    else:
        cam_cls, _ = _resolve_cam_class(cam_method_name)
        # SmoothGrad / EigenSmooth knobs from postprocess_cfg.
        # Reference papers:
        #   - SmoothGrad (Smilkov et al. 2017, arXiv:1706.03825): average gradients over
        #     N noisy copies of the input → removes high-frequency speckle from the heatmap.
        #     pytorch_grad_cam exposes this as `aug_smooth=True` (default 8-pass test-time aug).
        #   - EigenCAM principle (Muhammad & Yeasin 2020): project CAM onto top principal
        #     component of activations → suppresses noisy/spurious channels.
        #     pytorch_grad_cam exposes this on top of any CAM as `eigen_smooth=True`.
        # Both knobs target exactly the "chấm li ti" / scatter problem caused by sparse
        # gradients (e.g. when the backbone was trained with LSE pooling).
        try:
            aug_smooth = bool((postprocess_cfg or {}).get("aug_smooth", False))
        except (TypeError, ValueError):
            aug_smooth = False
        try:
            eigen_smooth = bool((postprocess_cfg or {}).get("eigen_smooth", False))
        except (TypeError, ValueError):
            eigen_smooth = False
        with cam_context, view_context:
            targets = [ClassifierOutputTarget(target_class_idx)]
            with cam_cls(model=model, target_layers=target_layers) as cam_obj:
                grayscale_cam = cam_obj(
                    input_tensor=image_tensor,
                    targets=targets,
                    aug_smooth=aug_smooth,
                    eigen_smooth=eigen_smooth,
                )
                grayscale_cam = grayscale_cam[0, :]  # [H, W]

        # PCAM attention gating: multiply GradCAM (class-specific channel weights)
        # by the PCAM attention map (model's spatial focus learned during training).
        # GradCAM alone with PCAM: gradient \u221d attn(i,j) * w_c \u2192 spatial info lost in channel avg.
        # Multiplying by attn re-injects the spatial focus the model actually uses.
        if not use_gap_for_gradcam:
            pcam_attn = _get_pcam_attention_map(model, image_tensor)
            if pcam_attn is not None:
                ah, aw = grayscale_cam.shape[:2]
                attn_resized = cv2.resize(pcam_attn, (aw, ah), interpolation=cv2.INTER_LINEAR)
                grayscale_cam = grayscale_cam * attn_resized

    # Save raw CAM before any post-processing for diagnostic comparison
    raw_cam_for_diag = grayscale_cam.copy()

    # ---- Anti-ringing pre-upsample low-pass --------------------------------
    # The CAM is produced at backbone resolution (typically 7x7 or 14x14)
    # and must be expanded ~16-32x to image space. Sharp resamplers (Lanczos,
    # bicubic) over-shoot at value transitions, which after percentile stretch
    # surface as bright "speckles". Standard fix (Selvaraju et al. 2017 /
    # pytorch-grad-cam `scale_cam_image`): low-pass at native resolution before
    # upsampling and use a moderate-order interpolant (INTER_CUBIC).
    try:
        pre_upsample_blur = int((postprocess_cfg or {}).get("pre_upsample_blur_kernel", 3))
    except (TypeError, ValueError):
        pre_upsample_blur = 3
    pre_upsample_blur = max(0, pre_upsample_blur)
    if pre_upsample_blur >= 3:
        pre_upsample_blur |= 1  # ensure odd
        grayscale_cam = cv2.GaussianBlur(
            grayscale_cam.astype(np.float32),
            (pre_upsample_blur, pre_upsample_blur), 0,
        )

    # Upscale CAM to full resolution — INTER_CUBIC avoids LANCZOS4 ringing
    # while giving smoother output than INTER_LINEAR.
    full_h, full_w = display_np.shape[:2]
    cam_h, cam_w = grayscale_cam.shape[:2]
    visible_h = original_image_work.height
    if crop_ratio > 0.0:
        crop_h = visible_h
        cam_cropped = grayscale_cam
        if (cam_h, cam_w) != (crop_h, full_w):
            cam_cropped = cv2.resize(
                grayscale_cam, (full_w, crop_h), interpolation=cv2.INTER_CUBIC,
            )
        cam_for_overlay = np.zeros((full_h, full_w), dtype=np.float32)
        cam_for_overlay[:crop_h, :] = cam_cropped
    else:
        if (cam_h, cam_w) != (full_h, full_w):
            cam_for_overlay = cv2.resize(
                grayscale_cam, (full_w, full_h), interpolation=cv2.INTER_CUBIC,
            )
        else:
            cam_for_overlay = grayscale_cam.copy()

    # Edge-guided refinement can make CAM look too "hard"/blocky.
    # Keep it optional (default OFF) for a softer, more natural Grad-CAM look.
    try:
        use_edge_guided = bool((postprocess_cfg or {}).get("edge_guided_upsample", False))
    except (TypeError, ValueError):
        use_edge_guided = False
    if use_edge_guided:
        cam_for_overlay = _edge_guided_upsample(cam_for_overlay, display_np)

    # Body mask — keep binary version for quality checks only.
    # Soft body mask is applied later (configurable) to allow heatmap to extend
    # outside the body for the classic paper-style "stained glass" look.
    full_body_mask = _body_mask(display_np)
    apply_body_fade = bool((postprocess_cfg or {}).get("apply_body_fade", False))
    if apply_body_fade:
        _soft_body = cv2.GaussianBlur(full_body_mask.astype(np.float32), (101, 101), 0)
        cam_for_overlay = cam_for_overlay * _soft_body
    else:
        _soft_body = None

    # Optional soft thoracic/lung mask: helps suppress shoulder/text/device drift
    # while preserving a clinically useful chest ROI.
    try:
        lung_mask_enabled = bool((postprocess_cfg or {}).get("lung_mask_enabled", False))
    except (TypeError, ValueError):
        lung_mask_enabled = False
    if lung_mask_enabled:
        try:
            lung_mask_include_mediastinum = bool(
                (postprocess_cfg or {}).get("lung_mask_include_mediastinum", True)
            )
        except (TypeError, ValueError):
            lung_mask_include_mediastinum = True
        try:
            lung_mask_strength = float((postprocess_cfg or {}).get("lung_mask_strength", 1.0))
        except (TypeError, ValueError):
            lung_mask_strength = 1.0
        try:
            lung_mask_dilate_iter = int((postprocess_cfg or {}).get("lung_mask_dilate_iter", 1))
        except (TypeError, ValueError):
            lung_mask_dilate_iter = 1
        try:
            lung_mask_blur_kernel = int((postprocess_cfg or {}).get("lung_mask_blur_kernel", 31))
        except (TypeError, ValueError):
            lung_mask_blur_kernel = 31
        try:
            lung_mask_side_margin_ratio = float(
                (postprocess_cfg or {}).get("lung_mask_side_margin_ratio", 0.04)
            )
        except (TypeError, ValueError):
            lung_mask_side_margin_ratio = 0.04
        try:
            lung_mask_apex_guard_ratio = float(
                (postprocess_cfg or {}).get("lung_mask_apex_guard_ratio", 0.14)
            )
        except (TypeError, ValueError):
            lung_mask_apex_guard_ratio = 0.14
        try:
            lung_mask_hard_threshold = float(
                (postprocess_cfg or {}).get("lung_mask_hard_threshold", 0.0)
            )
        except (TypeError, ValueError):
            lung_mask_hard_threshold = 0.0

        lung_mask_strength = max(0.0, min(1.0, lung_mask_strength))
        thoracic_mask = _approx_lung_mask(
            display_np,
            include_mediastinum=lung_mask_include_mediastinum,
            dilate_iter=max(0, lung_mask_dilate_iter),
            blur_kernel=lung_mask_blur_kernel,
            side_margin_ratio=lung_mask_side_margin_ratio,
            apex_guard_ratio=lung_mask_apex_guard_ratio,
        )
        mask_mix = (1.0 - lung_mask_strength) + lung_mask_strength * thoracic_mask
        cam_for_overlay = cam_for_overlay * mask_mix
        lung_mask_hard_threshold = max(0.0, min(0.8, lung_mask_hard_threshold))
        if lung_mask_hard_threshold > 0.0:
            cam_for_overlay[thoracic_mask < lung_mask_hard_threshold] = 0.0

    # Text/annotation corner suppression: chest X-ray datasets often embed patient markers ("L_03" etc.)
    # in the top corners. These are inside the body mask but are shortcut artifacts.
    # Zero out top-N% of image height × side-M% width on both corners.
    try:
        tc_top = float((postprocess_cfg or {}).get("text_corner_suppress_top", 0.0))
        tc_side = float((postprocess_cfg or {}).get("text_corner_suppress_side", 0.0))
    except (TypeError, ValueError):
        tc_top, tc_side = 0.0, 0.0
    if tc_top > 0 and tc_side > 0:
        top_rows = max(1, int(full_h * tc_top))
        side_cols = max(1, int(full_w * tc_side))
        try:
            top_strip_scale = float((postprocess_cfg or {}).get("text_top_strip_scale", 1.0))
        except (TypeError, ValueError):
            top_strip_scale = 1.0
        top_strip_scale = max(0.2, min(2.0, top_strip_scale))
        # Zero the entire top strip (narrow band) to catch center-placed
        # metadata text (e.g. "AP;PORT;SUPINE", "J&C") not just the corners.
        top_strip = max(1, int(full_h * tc_top * top_strip_scale))
        cam_for_overlay[:top_strip, :] = 0.0             # full-width narrow top strip
        cam_for_overlay[:top_rows, :side_cols] = 0.0    # top-left corner (wider)
        cam_for_overlay[:top_rows, -side_cols:] = 0.0   # top-right corner (wider)

    # Label-specific full-height side suppression for central findings such as
    # cardiomegaly. Keep this off by default because pleural/peripheral diseases
    # can legitimately activate near the lateral thoracic wall.
    try:
        side_strip = float((postprocess_cfg or {}).get("side_strip_suppress_ratio", 0.0))
    except (TypeError, ValueError):
        side_strip = 0.0
    side_strip = max(0.0, min(0.30, side_strip))
    if side_strip > 0.0:
        side_cols = max(1, int(full_w * side_strip))
        cam_for_overlay[:, :side_cols] = 0.0
        cam_for_overlay[:, -side_cols:] = 0.0

    # Gaussian smooth
    try:
        smooth_kernel = int((postprocess_cfg or {}).get("smooth_kernel", 7))
    except (TypeError, ValueError):
        smooth_kernel = 7
    smooth_kernel = max(3, smooth_kernel | 1)  # ensure odd, at least 3
    cam_smooth = cv2.GaussianBlur(cam_for_overlay, (smooth_kernel, smooth_kernel), 0)
    if _soft_body is not None:
        cam_smooth = cam_smooth * _soft_body
    if side_strip > 0.0:
        cam_smooth[:, :side_cols] = 0.0
        cam_smooth[:, -side_cols:] = 0.0

    # Percentile normalize: robust to bright outlier activations
    try:
        lo = float((postprocess_cfg or {}).get("percentile_lo", 1.0))
        hi = float((postprocess_cfg or {}).get("percentile_hi", 99.0))
    except (TypeError, ValueError):
        lo, hi = 1.0, 99.0
    cam_norm = percentile_normalize(cam_smooth, lo=lo, hi=hi)

    # Standalone heatmap (reference style): full gradient, no threshold, no X-ray blend.
    # Matches the professor's middle panel: dark-blue background → yellow/red hot spots.
    heatmap_standalone_bgr = cv2.applyColorMap(np.uint8(255 * cam_norm), colormap)
    heatmap_standalone_rgb = cv2.cvtColor(heatmap_standalone_bgr, cv2.COLOR_BGR2RGB)

    # Display threshold: zero out low-activation noise (uniform, no spatial bias)
    try:
        display_thresh = float((postprocess_cfg or {}).get("display_threshold", 0.20))
    except (TypeError, ValueError):
        display_thresh = 0.20
    display_thresh = max(0.0, min(0.5, display_thresh))
    if display_thresh > 0:
        cam_norm[cam_norm < display_thresh] = 0.0
        # Optional: stretch remaining values to [0, 1] for stronger contrast.
        # For aesthetic overlays (reference-like), default keeps soft intensity.
        try:
            stretch_after_thresh = bool((postprocess_cfg or {}).get("stretch_after_threshold", False))
        except (TypeError, ValueError):
            stretch_after_thresh = False
        if stretch_after_thresh:
            peak = float(cam_norm.max())
            if peak > 1e-8:
                cam_norm = cam_norm / peak

    # Remove tiny disconnected hotspots to avoid dotted/speckled artifacts.
    try:
        speckle_filter_enabled = bool((postprocess_cfg or {}).get("speckle_filter_enabled", True))
    except (TypeError, ValueError):
        speckle_filter_enabled = True
    if speckle_filter_enabled:
        try:
            speckle_thresh = float((postprocess_cfg or {}).get("speckle_activation_threshold", 0.18))
        except (TypeError, ValueError):
            speckle_thresh = 0.18
        try:
            speckle_min_area_ratio = float((postprocess_cfg or {}).get("speckle_min_area_ratio", 0.0015))
        except (TypeError, ValueError):
            speckle_min_area_ratio = 0.0015
        cam_norm = _suppress_small_hotspots(
            cam_norm,
            activation_threshold=speckle_thresh,
            min_area_ratio=speckle_min_area_ratio,
        )

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_norm), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Canonical paper-style uniform blend (pytorch-grad-cam show_cam_on_image):
    #     cam_image = (1 - image_weight) * heatmap + image_weight * img
    #     cam_image = cam_image / max(cam_image)
    # This produces the smooth "stained glass" look used in Hanif et al. and the
    # original Grad-CAM paper. No per-pixel alpha → no speckle / hard borders.
    image_weight = float(np.clip(1.0 - alpha, 0.0, 1.0))
    result_img = (1.0 - image_weight) * heatmap_rgb + image_weight * display_np
    peak = float(result_img.max())
    if peak > 1e-8:
        result_img = result_img / peak

    # Add contour overlay at 50% and 75% activation levels
    contour_enabled = bool((postprocess_cfg or {}).get("contour_overlay", True))
    result_uint8 = np.uint8(255 * np.clip(result_img, 0.0, 1.0))
    if contour_enabled and float(cam_norm.max()) > 0.01:
        contour_levels = [0.5, 0.75]
        result_uint8 = _add_contour_overlay(
            result_uint8, cam_norm,
            levels=contour_levels,
            color=(255, 255, 255),
            thickness=1,
            alpha=0.5,
        )
    heatmap_overlay = result_uint8

    # Optional raw CAM diagnostic overlay (set save_raw_cam: true in config)
    raw_cam_overlay = None
    if isinstance(postprocess_cfg, dict) and bool(postprocess_cfg.get("save_raw_cam", False)):
        raw_upscaled_visible = cv2.resize(raw_cam_for_diag, (full_w, visible_h), interpolation=cv2.INTER_LINEAR)
        raw_upscaled = np.zeros((full_h, full_w), dtype=np.float32)
        raw_upscaled[:visible_h, :] = raw_upscaled_visible
        raw_upscaled = np.clip(raw_upscaled, 0.0, None)
        if _soft_body is not None:
            raw_upscaled = raw_upscaled * _soft_body

        # Keep RAW diagnostic overlays anatomically plausible as well.
        # This preserves the un-thresholded CAM character while avoiding the
        # misleading impression that the model is attending to corners/shoulders.
        if lung_mask_enabled:
            raw_upscaled = raw_upscaled * mask_mix
            try:
                raw_lung_mask_hard_threshold = float(
                    (postprocess_cfg or {}).get(
                        "raw_lung_mask_hard_threshold",
                        max(0.0, lung_mask_hard_threshold * 0.6),
                    )
                )
            except (TypeError, ValueError):
                raw_lung_mask_hard_threshold = max(0.0, lung_mask_hard_threshold * 0.6)
            raw_lung_mask_hard_threshold = max(0.0, min(0.8, raw_lung_mask_hard_threshold))
            if raw_lung_mask_hard_threshold > 0.0:
                raw_upscaled[thoracic_mask < raw_lung_mask_hard_threshold] = 0.0
            if _soft_body is not None:
                raw_upscaled = raw_upscaled * _soft_body

        if tc_top > 0 and tc_side > 0:
            raw_upscaled[:top_strip, :] = 0.0
            raw_upscaled[:top_rows, :side_cols] = 0.0
            raw_upscaled[:top_rows, -side_cols:] = 0.0

        raw_norm = percentile_normalize(raw_upscaled, lo=lo, hi=hi)
        raw_heatmap = cv2.applyColorMap(np.uint8(255 * raw_norm), colormap)
        raw_heatmap_rgb = cv2.cvtColor(raw_heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Canonical paper-style uniform blend (same as main overlay)
        raw_image_weight = float(np.clip(1.0 - alpha, 0.0, 1.0))
        raw_overlay = (1.0 - raw_image_weight) * raw_heatmap_rgb + raw_image_weight * display_np
        raw_peak = float(raw_overlay.max())
        if raw_peak > 1e-8:
            raw_overlay = raw_overlay / raw_peak
        raw_cam_overlay = np.uint8(255 * np.clip(raw_overlay, 0.0, 1.0))

    # Quality assessment — derive label from class index if not provided
    _label = label_name
    if _label is None and 0 <= target_class_idx < len(NIH_LABELS):
        _label = NIH_LABELS[target_class_idx]
    quality = {}
    if _label:
        quality = _assess_cam_quality(cam_norm, _label, full_h, full_w, full_body_mask)

    return {
        "heatmap_overlay": heatmap_overlay,
        "heatmap_img": heatmap_standalone_rgb,   # pure colormap, no X-ray — for reference-style 3-panel
        "grayscale_cam": cam_norm,
        "raw_cam_overlay": raw_cam_overlay,
        "quality": quality,
    }


def generate_gradcam_for_top_diseases(
    model: torch.nn.Module,
    image_path: str,
    predictions: dict,
    config: dict,
    top_k: int = 3,
    threshold: float = 0.5,
    exclude_labels: list = None,
    target_labels: list = None,
    view_type: torch.Tensor = None,
) -> list:
    """Generate Grad-CAM for top predicted diseases."""
    gradcam_cfg = config.get("gradcam", {})
    normalization_cfg = config.get("cnn", {}).get("normalization", {})
    clahe_cfg = config.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})
    image_size = config["cnn"]["image_size"]
    bottom_crop_ratio = float(
        config.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0)
    )
    target_layer_name = gradcam_cfg.get(
        "target_layers",
        gradcam_cfg.get("target_layer", "features.denseblock4.denselayer16.conv2"),
    )
    cam_method = gradcam_cfg.get("cam_method", "gradcam")
    colormap = _resolve_colormap(gradcam_cfg.get("colormap", "jet"))
    alpha = float(gradcam_cfg.get("alpha", 0.4))
    if exclude_labels is None:
        exclude_labels = gradcam_cfg.get("exclude_labels", [])
    exclude_set = set(exclude_labels)

    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    eligible = [(label, prob) for label, prob in sorted_preds if label not in exclude_set]

    if target_labels is not None:
        top_diseases = [
            (label, predictions[label])
            for label in target_labels
            if label not in exclude_set and label in predictions
        ][:top_k]
    else:
        above_threshold = [(label, prob) for label, prob in eligible if prob >= threshold]
        below_threshold = [(label, prob) for label, prob in eligible if prob < threshold]
        top_diseases = (above_threshold + below_threshold)[:top_k]

    results = []
    for label, prob in top_diseases:
        label_gradcam_cfg = _gradcam_cfg_for_label(gradcam_cfg, label)
        class_idx = NIH_LABELS.index(label)
        res = generate_gradcam(
            model,
            image_path,
            class_idx,
            image_size,
            label_gradcam_cfg.get(
                "target_layers",
                label_gradcam_cfg.get("target_layer", target_layer_name),
            ),
            colormap=_resolve_colormap(label_gradcam_cfg.get("colormap", gradcam_cfg.get("colormap", "jet"))),
            alpha=float(label_gradcam_cfg.get("alpha", alpha)),
            postprocess_cfg=label_gradcam_cfg,
            cam_method=str(label_gradcam_cfg.get("cam_method", cam_method)),
            bottom_crop_ratio=bottom_crop_ratio,
            view_type=view_type,
            normalization_cfg=normalization_cfg,
            clahe_cfg=clahe_cfg,
        )
        result_item = {
            "label": label,
            "probability": prob,
            "heatmap": res["heatmap_overlay"],
            "heatmap_img": res.get("heatmap_img"),
            "grayscale": res["grayscale_cam"],
            "quality": res.get("quality", {}),
        }
        quality = result_item["quality"] if isinstance(result_item["quality"], dict) else {}
        result_item["primary_zone"] = quality.get("primary_zone", "")
        result_item["primary_fraction"] = quality.get("primary_fraction", 0.0)
        result_item["zone_activations"] = quality.get("zone_activations", {})
        if prob < threshold:
            result_item["weak_label_warning"] = (
                "Prediction is below the display threshold; localization is less reliable."
            )
        if res.get("raw_cam_overlay") is not None:
            result_item["raw_cam_overlay"] = res["raw_cam_overlay"]
        results.append(result_item)

    return results


def save_gradcam_images(
    results: list,
    output_dir: str,
    prefix: str = "gradcam",
):
    """Save heatmap overlay images to disk."""
    ensure_dir(output_dir)
    saved_paths = []
    for i, result in enumerate(results):
        label = result["label"].replace(" ", "_")
        prob = result["probability"]
        filename = f"{prefix}_{label}_{prob:.0%}.png"
        filepath = os.path.join(output_dir, filename)

        heatmap_bgr = cv2.cvtColor(result["heatmap"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, heatmap_bgr)
        saved_paths.append(filepath)

        if "raw_cam_overlay" in result and result["raw_cam_overlay"] is not None:
            raw_filename = f"{prefix}_{label}_{prob:.0%}_RAW.png"
            raw_filepath = os.path.join(output_dir, raw_filename)
            raw_bgr = cv2.cvtColor(result["raw_cam_overlay"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(raw_filepath, raw_bgr)
            saved_paths.append(raw_filepath)

    return saved_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--class_idx", type=int, default=None, help="Target class index (0-13)")
    parser.add_argument("--output_dir", type=str, default="outputs/gradcam")
    parser.add_argument("--device", type=str, default=None, help="Override device: cpu | cuda | cuda:0")
    parser.add_argument("--view_type", type=str, default="ap", choices=["ap", "pa", "lateral"],
                        help="X-ray view position: ap=frontal AP (default), pa=frontal PA, lateral")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device(config)
    model = load_trained_model(config["paths"]["densenet_checkpoint"], config, device)

    # Encode view_type: AP=0.0, PA=1.0, Lateral=2.0 (matches training encoding)
    _view_map = {"ap": 0.0, "pa": 1.0, "lateral": 2.0}
    _vt_val = _view_map.get(args.view_type.lower(), 0.0)
    cli_view_type = torch.tensor([[_vt_val]], dtype=torch.float32, device=device)

    if args.class_idx is not None:
        gradcam_cfg = config.get("gradcam", {})
        label_name = NIH_LABELS[args.class_idx]
        label_gradcam_cfg = _gradcam_cfg_for_label(gradcam_cfg, label_name)
        res = generate_gradcam(
            model, args.image, args.class_idx,
            config["cnn"]["image_size"],
            label_gradcam_cfg.get(
                "target_layers",
                label_gradcam_cfg.get("target_layer", "features.denseblock4.denselayer16.conv2"),
            ),
            colormap=_resolve_colormap(label_gradcam_cfg.get("colormap", "jet")),
            alpha=float(label_gradcam_cfg.get("alpha", 0.4)),
            postprocess_cfg=label_gradcam_cfg,
            cam_method=str(label_gradcam_cfg.get("cam_method", "gradcam")),
            normalization_cfg=config.get("cnn", {}).get("normalization", {}),
            view_type=cli_view_type,
        )
        ensure_dir(args.output_dir)
        label = label_name
        output_path = os.path.join(args.output_dir, f"gradcam_{label}.png")
        cv2.imwrite(output_path, cv2.cvtColor(res["heatmap_overlay"], cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")
        if res.get("raw_cam_overlay") is not None:
            raw_path = os.path.join(args.output_dir, f"gradcam_{label}_RAW.png")
            cv2.imwrite(raw_path, cv2.cvtColor(res["raw_cam_overlay"], cv2.COLOR_RGB2BGR))
            print(f"Saved (raw): {raw_path}")
    else:
        from src.cnn.inference import predict
        image_tensor = preprocess_image(
            args.image,
            config["cnn"]["image_size"],
            normalization_cfg=config.get("cnn", {}).get("normalization", {}),
        )
        pred_result = predict(model, image_tensor, device)

        results = generate_gradcam_for_top_diseases(
            model, args.image, pred_result["probabilities"], config,
            view_type=cli_view_type,
        )
        if results:
            paths = save_gradcam_images(results, args.output_dir)
            for p in paths:
                print(f"Saved: {p}")
        else:
            print("No significant pathology detected above threshold.")
