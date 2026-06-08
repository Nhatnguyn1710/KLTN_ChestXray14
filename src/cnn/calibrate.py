

import os
import sys
import json
import shutil
import argparse
import datetime
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# scipy is optional — graceful fallback to grid search if missing
try:
    from scipy.optimize import minimize_scalar
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.dataset import CheXpertDataset, CHEXPERT_LABELS
from src.cnn.inference import _build_tta_transforms, _merge_tta_logits_batch, batch_tta_from_paths
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_ISOTONIC = True
except ImportError:
    HAS_ISOTONIC = False


# ──────────────────────────────────────────────────────────────
# Calibration loss helpers (negative log-likelihood per Guo et al. 2017)
# ──────────────────────────────────────────────────────────────

_NLL_EPS = 1e-7


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary negative log-likelihood (log loss).

    Lower is better. Used to fit Temperature Scaling and to choose between
    Temperature Scaling and Isotonic Regression on the validation split,
    following Guo C., Pleiss G., Sun Y., Weinberger K. Q. (2017),
    "On Calibration of Modern Neural Networks", ICML.
    """
    if len(probs) == 0:
        return float("nan")
    p = np.clip(probs, _NLL_EPS, 1.0 - _NLL_EPS)
    y = labels.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _temperature_nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    probs = 1.0 / (1.0 + np.exp(-logits / T))
    return _nll(probs, labels)


# ──────────────────────────────────────────────────────────────
# Temperature fitting
# ──────────────────────────────────────────────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    max_temperature: float = 10.0,
                    min_temperature: float = 0.8) -> float:
    """Find T* that minimises negative log-likelihood for a single label.

    Returns T in [min_temperature, max_temperature]. The lower bound prevents
    pathological low-T solutions that sharpen the distribution and hurt recall.
    """
    lo, hi = float(min_temperature), float(max_temperature)
    if HAS_SCIPY:
        result = minimize_scalar(
            _temperature_nll,
            bounds=(lo, hi),
            method="bounded",
            args=(logits, labels),
            options={"xatol": 1e-4, "maxiter": 500},
        )
        return float(np.clip(result.x, lo, hi))

    T_grid = np.linspace(lo, hi, 200)
    best_T, best_loss = 1.0, float("inf")
    for T in T_grid:
        loss = _temperature_nll(T, logits, labels)
        if loss < best_loss:
            best_loss, best_T = loss, T
    return float(best_T)


def fit_isotonic(logits: np.ndarray, labels: np.ndarray):
    """Fit isotonic regression to map raw probabilities → calibrated probabilities.

    Non-parametric, monotonic alternative to Temperature Scaling. Trả về dict
    ``{"x": [...], "y": [...]}`` có thể serialize JSON. Tại inference dùng
    ``numpy.interp(prob_raw, x, y)``. Trả về None nếu sklearn không có hoặc
    dữ liệu không đủ.
    """
    if not HAS_ISOTONIC:
        return None
    if len(labels) < 50 or labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
        return None
    probs = 1.0 / (1.0 + np.exp(-logits))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    try:
        iso.fit(probs, labels.astype(float))
    except Exception:
        return None
    x_knots = iso.X_thresholds_.astype(float).tolist()
    y_knots = iso.y_thresholds_.astype(float).tolist()
    if len(x_knots) < 2:
        return None
    return {"x": x_knots, "y": y_knots}


def apply_isotonic(probs_raw: np.ndarray, iso_map: dict) -> np.ndarray:
    """Apply saved isotonic mapping to raw probabilities."""
    if not iso_map or "x" not in iso_map or "y" not in iso_map:
        return probs_raw
    x = np.asarray(iso_map["x"], dtype=np.float64)
    y = np.asarray(iso_map["y"], dtype=np.float64)
    return np.interp(probs_raw, x, y)


# ──────────────────────────────────────────────────────────────
# Threshold finding
# ──────────────────────────────────────────────────────────────

def find_threshold_youden(probs: np.ndarray, labels: np.ndarray,
                          min_thresh: float = 0.3,
                          min_pos: int = 5) -> float:
    """
    Maximise Youden-J = Sensitivity + Specificity - 1.
    Falls back to 0.5 when too few positives.
    """
    pos = labels.sum()
    neg = len(labels) - pos
    if pos < min_pos or neg < min_pos:
        return max(0.5, min_thresh)

    thresholds = np.unique(probs)
    thresholds = thresholds[(thresholds >= min_thresh) & (thresholds <= 0.95)]
    if len(thresholds) == 0:
        return max(0.5, min_thresh)

    best_j, best_t = -1.0, max(0.5, min_thresh)
    for t in thresholds:
        pred = (probs >= t).astype(float)
        tp = ((pred == 1) & (labels == 1)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def _threshold_metrics(probs: np.ndarray, labels: np.ndarray, thresh: float) -> dict:
    pred = (probs >= thresh).astype(float)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "pred_pos": int((pred == 1).sum()),
    }


def find_threshold_precision_target(
    probs: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.7,
    min_thresh: float = 0.5,
    min_pos: int = 5,
) -> float:
    """
    Choose the lowest threshold that reaches the target precision while keeping
    the highest recall among feasible candidates.

    If no threshold reaches the target precision, fall back to the threshold
    with highest precision, then highest specificity, then highest F1.
    """
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    safe_default = max(0.5, min_thresh)
    if pos < min_pos or neg < min_pos:
        return safe_default

    thresholds = np.unique(probs)
    thresholds = thresholds[(thresholds >= min_thresh) & (thresholds <= 0.995)]
    if len(thresholds) == 0:
        return safe_default

    candidates = []
    for t in thresholds:
        metrics = _threshold_metrics(probs, labels, float(t))
        candidates.append((float(t), metrics))

    feasible = [
        (t, m) for t, m in candidates
        if m["precision"] >= float(target_precision) and m["pred_pos"] > 0
    ]
    if feasible:
        feasible.sort(
            key=lambda item: (
                item[1]["recall"],
                item[1]["specificity"],
                -item[0],
            ),
            reverse=True,
        )
        return float(feasible[0][0])

    candidates.sort(
        key=lambda item: (
            item[1]["precision"],
            item[1]["specificity"],
            item[1]["f1"],
            item[0],
        ),
        reverse=True,
    )
    return float(candidates[0][0])


def find_threshold_f_beta(
    probs: np.ndarray,
    labels: np.ndarray,
    beta: float = 1.0,
    min_thresh: float = 0.3,
    min_pos: int = 5,
) -> float:
    """
    Choose the threshold that maximises F_beta score.
    beta < 1 favours precision, beta > 1 favours recall.
    F1 when beta=1.
    """
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    safe_default = max(0.5, min_thresh)
    if pos < min_pos or neg < min_pos:
        return safe_default

    thresholds = np.unique(probs)
    thresholds = thresholds[(thresholds >= min_thresh) & (thresholds <= 0.995)]
    if len(thresholds) == 0:
        return safe_default

    beta_sq = beta * beta
    best_fb, best_t = -1.0, safe_default
    for t in thresholds:
        metrics = _threshold_metrics(probs, labels, float(t))
        p, r = metrics["precision"], metrics["recall"]
        if p + r < 1e-9:
            continue
        fb = (1.0 + beta_sq) * p * r / (beta_sq * p + r + 1e-9)
        if fb > best_fb:
            best_fb, best_t = fb, float(t)
    return best_t


def find_threshold_target_sensitivity(
    probs: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.90,
    min_thresh: float = 0.10,
    min_pos: int = 5,
) -> float:
    """
    Clinical operating point: chọn threshold cao nhất sao cho recall (sensitivity)
    >= target_sensitivity. Tối đa hoá specificity trong các candidate đạt yêu cầu.
    Dùng cho critical findings (Pneumothorax, Pneumonia, Mass, Nodule) — không bỏ sót.

    Nếu không threshold nào đạt được target, fallback về threshold có recall cao nhất.
    """
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    safe_default = max(0.5, min_thresh)
    if pos < min_pos or neg < min_pos:
        return safe_default

    thresholds = np.unique(probs)
    thresholds = thresholds[(thresholds >= min_thresh) & (thresholds <= 0.995)]
    if len(thresholds) == 0:
        return safe_default

    feasible = []
    fallback_best = (-1.0, safe_default)  # (recall, threshold)
    for t in thresholds:
        m = _threshold_metrics(probs, labels, float(t))
        if m["recall"] >= float(target_sensitivity) and m["pred_pos"] > 0:
            feasible.append((float(t), m))
        if m["recall"] > fallback_best[0]:
            fallback_best = (m["recall"], float(t))

    if feasible:
        # Trong các threshold đạt sensitivity target, chọn threshold cao nhất
        # (specificity cao nhất) để giảm false positive
        feasible.sort(key=lambda item: (item[1]["specificity"], item[0]), reverse=True)
        return float(feasible[0][0])

    # Không có threshold nào đạt target → trả về threshold có recall cao nhất
    return float(fallback_best[1])


# ──────────────────────────────────────────────────────────────
# Main calibration logic
# ──────────────────────────────────────────────────────────────

def run_calibration(config: dict, checkpoint_path: str, output_path: str = "configs/calibration.json", use_tta: bool = False):
    device = get_device(config)
    cnn_cfg = config["cnn"]
    paths = config["paths"]

    # ── 1. Load model ──────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_trained_model(checkpoint_path, config, device)
    model.eval()

    # ── 2. Load validation split ──────────────────────────────
    # Ưu tiên paths.val_csv (NIH mode); fallback sang splits/val_internal_split.csv (legacy).
    val_csv = paths.get("val_csv")
    if not val_csv or not os.path.isfile(val_csv):
        val_csv = os.path.join(
            os.path.dirname(paths["train_csv"]), "splits", "val_internal_split.csv"
        )
    if not os.path.isfile(val_csv):
        sys.exit(
            f"ERROR: validation CSV not found. Tried paths.val_csv and "
            f"{val_csv}. Cấu hình paths.val_csv hoặc chạy training để sinh splits."
        )

    val_df = pd.read_csv(val_csv)
    print(f"validation images: {len(val_df):,}  |  split: {val_csv}")

    aug_cfg = cnn_cfg.get("augmentation", {})
    dataset = CheXpertDataset(
        csv_path=None,
        image_root=paths["dataset_dir"],
        image_size=cnn_cfg["image_size"],
        augmentation=False,
        uncertainty_policy=cnn_cfg.get("uncertainty_policy", "zeros"),
        dataframe=val_df,
        aug_cfg=aug_cfg,
        use_view_position=cnn_cfg.get("use_view_position", False),
        bottom_crop_ratio=aug_cfg.get("bottom_crop_ratio", 0.0),
        uncertainty_policy_per_class=cnn_cfg.get("uncertainty_policy_per_class", None),
        normalization_cfg=cnn_cfg.get("normalization", {}),
    )

    num_workers = int(cnn_cfg.get("calibration", {}).get("num_workers", 0 if os.name == "nt" else 2))
    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # ── 3. Inference ───────────────────────────────────────────
    all_logits, all_labels, all_masks = [], [], []
    use_fp16 = cnn_cfg.get("training", {}).get("use_fp16", True)

    tta_cfg = cnn_cfg.get("tta", {}) if use_tta else {}
    if use_tta:
        from src.cnn.inference import _build_tta_transforms as _bt
        tta_transforms = _bt(
            cnn_cfg["image_size"],
            normalization_cfg=cnn_cfg.get("normalization", {}),
            tta_cfg=tta_cfg,
        )
        print(f"TTA enabled: {len(tta_transforms)} augmentations (full pipeline), "
              f"flip labels: {sorted(tta_cfg.get('flip_labels', []))}")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference on val_internal" + (" (TTA)" if use_tta else "")):
            view_type = batch.get("view_type")
            if view_type is not None:
                view_type = view_type.to(device)

            if use_tta:
                merged = batch_tta_from_paths(
                    paths=batch["path"],
                    image_root=paths["dataset_dir"],
                    model=model,
                    device=device,
                    image_size=cnn_cfg["image_size"],
                    tta_cfg=tta_cfg,
                    normalization_cfg=cnn_cfg.get("normalization", {}),
                    clahe_cfg=cnn_cfg.get("augmentation", {}).get("clahe_preprocessing", None),
                    bottom_crop_ratio=float(cnn_cfg.get("augmentation", {}).get("bottom_crop_ratio", 0.0)),
                    view_types=view_type,
                    use_fp16=use_fp16,
                )
                all_logits.append(merged)
            else:
                images = batch["image"].to(device)
                with autocast("cuda", enabled=(use_fp16 and device.type == "cuda")):
                    raw = model.forward_logits(images, view_type=view_type) \
                        if hasattr(model, "forward_logits") \
                        else _get_logits(model, images, view_type)
                all_logits.append(raw.cpu().float().numpy())

            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())

    logits = np.concatenate(all_logits, axis=0)   # [N, 14]
    labels = np.concatenate(all_labels, axis=0)   # [N, 14]
    masks  = np.concatenate(all_masks,  axis=0)   # [N, 14]  (1 = label certain)

    # ── 4. Per-label calibration ───────────────────────────────
    per_label_temperatures: dict = {}
    per_label_isotonic: dict     = {}
    thresholds: dict             = {}
    label_auc: dict              = {}
    label_auprc: dict            = {}

    calib_cfg = cnn_cfg.get("calibration", {})
    threshold_strategy = str(calib_cfg.get("threshold_strategy", "precision_target")).strip().lower()
    target_precision = float(calib_cfg.get("target_precision", 0.70))
    min_pos_rare = int(calib_cfg.get("min_pos_for_rare", 10))
    min_thresh = float(calib_cfg.get("min_threshold", 0.50))
    rare_min_thresh = float(calib_cfg.get("rare_min_threshold", 0.55))
    max_temperature = float(calib_cfg.get("max_temperature", 3.0))
    min_temperature = float(calib_cfg.get("min_temperature", 0.8))
    min_recall_floor = float(calib_cfg.get("min_recall_floor", 0.05))
    f_beta_fallback = float(calib_cfg.get("f_beta_fallback", 1.0))
    use_isotonic = bool(calib_cfg.get("use_isotonic", True))
    per_label_overrides = calib_cfg.get("per_label_overrides", {})

    print(f"\n  Strategy: {threshold_strategy}  target_prec={target_precision}"
          f"  min_thresh={min_thresh}  max_T={max_temperature}"
          f"  min_recall={min_recall_floor}  f_beta_fb={f_beta_fallback}")
    if per_label_overrides:
        print(f"  Per-label overrides: {list(per_label_overrides.keys())}")
    print(f"\n  {'Label':<23} {'T*':>6}  {'thresh':>7}  "
          f"{'prec':>6}  {'recall':>6}  {'spec':>6}  {'f1':>6}  "
          f"{'AUC':>6}  {'n_pos':>6}  note")
    print("-" * 105)

    for i, label in enumerate(CHEXPERT_LABELS):
        m = masks[:, i].astype(bool)
        lgt = logits[m, i]
        lbl = labels[m, i]
        prb_raw = 1.0 / (1.0 + np.exp(-lgt))

        n_pos = int(lbl.sum())
        n_total = int(m.sum())
        note = ""

        # Per-label overrides
        label_ov = per_label_overrides.get(label, {})
        label_strategy = str(label_ov.get("strategy", threshold_strategy)).strip().lower()
        label_target_prec = float(label_ov.get("target_precision", target_precision))
        label_min_thresh = float(label_ov.get("min_threshold", min_thresh))
        label_f_beta = float(label_ov.get("f_beta", f_beta_fallback))
        label_min_recall = float(label_ov.get("min_recall_floor", min_recall_floor))

        if n_total < 10 or n_pos < 2:
            T = 1.0
            iso_map = None
            thresh = 0.5
            prb_cal = prb_raw
            note = "default(few)"
        else:
            # Fit Temperature Scaling và Isotonic Regression, chọn phương pháp có
            # negative log-likelihood thấp hơn trên validation (Guo et al. 2017).
            T = fit_temperature(lgt, lbl, max_temperature=max_temperature, min_temperature=min_temperature)
            prb_T = 1.0 / (1.0 + np.exp(-lgt / T))
            loss_T = _nll(prb_T, lbl)

            iso_map = None
            iso_map_candidate = fit_isotonic(lgt, lbl) if use_isotonic else None
            if iso_map_candidate is not None:
                prb_iso = apply_isotonic(prb_raw, iso_map_candidate)
                loss_iso = _nll(prb_iso, lbl)
            else:
                loss_iso = float("inf")

            if iso_map_candidate is not None and loss_iso < loss_T - 0.005:
                iso_map = iso_map_candidate
                T = 1.0
                prb_cal = apply_isotonic(prb_raw, iso_map)
                cal_method = "iso"
            else:
                prb_cal = prb_T
                cal_method = "T"

            min_t = rare_min_thresh if n_pos < min_pos_rare else label_min_thresh

            if label_strategy in {"fixed_logit_zero", "paper_zero", "zlpr_zero"}:
                # Paper-style ZLPR/FZLPR decision rule: label score/logit >= 0.
                # In probability space this is sigmoid(logit / T) >= 0.5 for any T > 0.
                thresh, note = 0.5, "fixed_logit0"

            elif label_strategy == "f_beta":
                # ── Direct F-beta optimisation (per-label override) ──
                thresh_cal = find_threshold_f_beta(
                    prb_cal, lbl, beta=label_f_beta, min_thresh=min_t, min_pos=min_pos_rare)
                m_cal = _threshold_metrics(prb_cal, lbl, thresh_cal)
                thresh_raw = find_threshold_f_beta(
                    prb_raw, lbl, beta=label_f_beta, min_thresh=min_t, min_pos=min_pos_rare)
                m_raw = _threshold_metrics(prb_raw, lbl, thresh_raw)

                beta_sq = label_f_beta * label_f_beta
                def _fb(m):
                    p, r = m["precision"], m["recall"]
                    return (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-9)

                if _fb(m_cal) >= _fb(m_raw):
                    thresh, note = thresh_cal, f"f{label_f_beta:.1f}(cal)"
                else:
                    T, thresh, note = 1.0, thresh_raw, f"f{label_f_beta:.1f}(raw)"
                    prb_cal = prb_raw

            elif label_strategy == "youden_j":
                thresh_cal = find_threshold_youden(
                    prb_cal, lbl, min_thresh=min_t, min_pos=min_pos_rare)
                thresh_raw = find_threshold_youden(
                    prb_raw, lbl, min_thresh=min_t, min_pos=min_pos_rare)
                m_cal = _threshold_metrics(prb_cal, lbl, thresh_cal)
                m_raw = _threshold_metrics(prb_raw, lbl, thresh_raw)
                if m_cal["f1"] >= m_raw["f1"]:
                    thresh, note = thresh_cal, "youden(cal)"
                else:
                    T, thresh, note = 1.0, thresh_raw, "youden(raw)"
                    prb_cal = prb_raw

            elif label_strategy == "target_sensitivity":
                # Clinical operating point: critical findings — không bỏ sót
                label_target_sens = float(label_ov.get("target_sensitivity", 0.90))
                # Cho phép min_threshold thấp hơn để đạt được sensitivity target
                sens_min_t = float(label_ov.get("min_threshold", 0.10))
                thresh_cal = find_threshold_target_sensitivity(
                    prb_cal, lbl,
                    target_sensitivity=label_target_sens,
                    min_thresh=sens_min_t,
                    min_pos=min_pos_rare,
                )
                thresh_raw = find_threshold_target_sensitivity(
                    prb_raw, lbl,
                    target_sensitivity=label_target_sens,
                    min_thresh=sens_min_t,
                    min_pos=min_pos_rare,
                )
                m_cal = _threshold_metrics(prb_cal, lbl, thresh_cal)
                m_raw = _threshold_metrics(prb_raw, lbl, thresh_raw)
                # Cả hai đều đạt sens target → chọn cái có specificity cao hơn
                cal_meets = m_cal["recall"] >= label_target_sens - 0.01
                raw_meets = m_raw["recall"] >= label_target_sens - 0.01
                if cal_meets and raw_meets:
                    if m_cal["specificity"] >= m_raw["specificity"]:
                        thresh, note = thresh_cal, f"sens{label_target_sens:.2f}(cal)"
                    else:
                        T, thresh, note = 1.0, thresh_raw, f"sens{label_target_sens:.2f}(raw)"
                        prb_cal = prb_raw
                elif cal_meets:
                    thresh, note = thresh_cal, f"sens{label_target_sens:.2f}(cal)"
                elif raw_meets:
                    T, thresh, note = 1.0, thresh_raw, f"sens{label_target_sens:.2f}(raw)"
                    prb_cal = prb_raw
                else:
                    thresh, note = thresh_cal, f"sens{label_target_sens:.2f}(cal,low)"

            elif label_strategy == "precision_target":
                # --- Primary: precision_target on calibrated probs ---
                thresh_cal = find_threshold_precision_target(
                    prb_cal, lbl,
                    target_precision=label_target_prec,
                    min_thresh=min_t,
                    min_pos=min_pos_rare,
                )
                m_cal = _threshold_metrics(prb_cal, lbl, thresh_cal)

                # --- Also try on raw probs (T=1) for comparison ---
                thresh_raw = find_threshold_precision_target(
                    prb_raw, lbl,
                    target_precision=label_target_prec,
                    min_thresh=min_t,
                    min_pos=min_pos_rare,
                )
                m_raw = _threshold_metrics(prb_raw, lbl, thresh_raw)

                cal_meets = m_cal["precision"] >= label_target_prec - 0.01
                raw_meets = m_raw["precision"] >= label_target_prec - 0.01

                if cal_meets and raw_meets:
                    if m_cal["f1"] >= m_raw["f1"]:
                        thresh, note, chosen_m = thresh_cal, "prec_target(cal)", m_cal
                    else:
                        T, thresh, note, chosen_m = 1.0, thresh_raw, "prec_target(raw)", m_raw
                        prb_cal = prb_raw
                elif cal_meets:
                    thresh, note, chosen_m = thresh_cal, "prec_target(cal)", m_cal
                elif raw_meets:
                    T, thresh, note, chosen_m = 1.0, thresh_raw, "prec_target(raw)", m_raw
                    prb_cal = prb_raw
                else:
                    chosen_m = None

                # Khi precision target đạt nhưng recall quá thấp → chuyển sang F-beta
                # để cân bằng precision và recall.
                if chosen_m is not None and chosen_m["recall"] < label_min_recall:
                    prb_for_fb = prb_cal if T != 1.0 else prb_raw
                    thresh_fb_cal = find_threshold_f_beta(
                        prb_for_fb, lbl, beta=label_f_beta,
                        min_thresh=min_t, min_pos=min_pos_rare)
                    thresh_fb_raw = find_threshold_f_beta(
                        prb_raw, lbl, beta=label_f_beta,
                        min_thresh=min_t, min_pos=min_pos_rare)
                    m_fb_cal = _threshold_metrics(prb_for_fb, lbl, thresh_fb_cal)
                    m_fb_raw = _threshold_metrics(prb_raw, lbl, thresh_fb_raw)
                    if m_fb_cal["f1"] >= m_fb_raw["f1"]:
                        thresh = thresh_fb_cal
                        note = f"f{label_f_beta:.0f}_floor(cal)"
                    else:
                        T, thresh = 1.0, thresh_fb_raw
                        prb_cal = prb_raw
                        note = f"f{label_f_beta:.0f}_floor(raw)"
                elif chosen_m is None:
                    thresh_fb_raw = find_threshold_f_beta(
                        prb_raw, lbl, beta=label_f_beta,
                        min_thresh=min_t, min_pos=min_pos_rare)
                    thresh_fb_cal = find_threshold_f_beta(
                        prb_cal, lbl, beta=label_f_beta,
                        min_thresh=min_t, min_pos=min_pos_rare)
                    m_fb_raw = _threshold_metrics(prb_raw, lbl, thresh_fb_raw)
                    m_fb_cal = _threshold_metrics(prb_cal, lbl, thresh_fb_cal)
                    if m_fb_cal["f1"] >= m_fb_raw["f1"]:
                        thresh = thresh_fb_cal
                        note = f"f{label_f_beta:.0f}_fallback(cal)"
                    else:
                        T, thresh = 1.0, thresh_fb_raw
                        prb_cal = prb_raw
                        note = f"f{label_f_beta:.0f}_fallback(raw)"
            else:
                raise ValueError(f"Unsupported threshold_strategy: {label_strategy}")

        per_label_temperatures[label] = round(T, 6)
        # Nếu T == 1.0 và prb_cal bị reset về prb_raw (fallback "raw"), iso_map cũng
        # phải huỷ — cuối cùng inference dùng raw probs không calibrated.
        if T == 1.0 and prb_cal is prb_raw:
            iso_map = None
        if iso_map is not None:
            per_label_isotonic[label] = iso_map
        thresholds[label] = round(thresh, 4)

        if n_total >= 10 and n_pos >= 2:
            if iso_map is not None:
                prb_final = apply_isotonic(prb_raw, iso_map)
            elif T != 1.0:
                prb_final = 1.0 / (1.0 + np.exp(-lgt / T))
            else:
                prb_final = prb_raw
            m_final = _threshold_metrics(prb_final, lbl, thresh)
        else:
            m_final = {"precision": 0.0, "recall": 0.0, "specificity": 0.0, "f1": 0.0}

        if HAS_SKLEARN and n_pos >= 2 and (n_total - n_pos) >= 2:
            try:
                auc   = roc_auc_score(lbl, prb_raw)
                auprc = average_precision_score(lbl, prb_raw)
            except Exception:
                auc = auprc = float("nan")
        else:
            auc = auprc = float("nan")

        label_auc[label]   = round(float(auc),   4) if not np.isnan(auc)   else None
        label_auprc[label] = round(float(auprc), 4) if not np.isnan(auprc) else None

        print(f"  {label:<23} {T:>6.3f}  {thresh:>7.4f}  "
              f"{m_final['precision']:>6.3f}  {m_final['recall']:>6.3f}  "
              f"{m_final['specificity']:>6.3f}  {m_final['f1']:>6.3f}  "
              f"{auc:>6.4f}  {n_pos:>6}  "
              f"[{('iso' if iso_map is not None else 'T  ')}] {note}")

    # ── 5. Summary metrics ─────────────────────────────────────
    valid_auc   = [v for v in label_auc.values()   if v is not None]
    valid_auprc = [v for v in label_auprc.values() if v is not None]
    mean_auc   = round(float(np.mean(valid_auc)),   4) if valid_auc   else 0.0
    mean_auprc = round(float(np.mean(valid_auprc)), 4) if valid_auprc else 0.0
    global_T   = round(float(np.median(list(per_label_temperatures.values()))), 6)

    print("-" * 85)
    print(f"\n  Global T (median): {global_T}")
    print(f"  mean_auc={mean_auc:.4f}  mean_auprc={mean_auprc:.4f}")

    # ── 6. Backup old calibration.json ────────────────────────
    if os.path.isfile(output_path):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = output_path.replace(".json", f".backup_{ts}.json")
        shutil.copy2(output_path, backup)
        print(f"\n  Backed up old calibration → {backup}")

    # ── 7. Write new calibration.json ─────────────────────────
    calib = {
        "model_version": os.path.basename(os.path.dirname(checkpoint_path)),
        "checkpoint": checkpoint_path,
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "temperature": global_T,
        "per_label_temperatures": per_label_temperatures,
        "per_label_isotonic": per_label_isotonic,
        "thresholds": thresholds,
        "mean_auc":   mean_auc,
        "mean_auprc": mean_auprc,
        "label_auc":  label_auc,
        "label_auprc": label_auprc,
        "threshold_policy": {
            "strategy":           threshold_strategy,
            "target_precision":   target_precision,
            "min_threshold":      min_thresh,
            "rare_min_threshold": rare_min_thresh,
            "min_pos_for_rare":   min_pos_rare,
            "max_temperature":    max_temperature,
            "min_recall_floor":   min_recall_floor,
            "f_beta_fallback":    f_beta_fallback,
            "per_label_overrides": {k: dict(v) for k, v in per_label_overrides.items()},
        },
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)

    print(f"\n  Written → {output_path}")

    return calib


# ──────────────────────────────────────────────────────────────
# Logit extraction helper (for models without forward_logits)
# ──────────────────────────────────────────────────────────────

def _get_logits(model, images, view_type=None):
    """
    Run model and return pre-sigmoid logits.
    Current CNN forward already returns raw logits, so no inverse-sigmoid
    conversion is needed here.
    """
    with torch.no_grad():
        logits = model(images, view_type=view_type)
    return logits


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrate NIH DenseNet-121 thresholds")
    parser.add_argument("--config",     default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Override checkpoint path (default: from config)")
    parser.add_argument("--output",     default="configs/calibration.json",
                        help="Output calibration JSON path")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (laterality-aware flip + merge)")
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint = args.checkpoint or config["paths"]["densenet_checkpoint"]
    if not os.path.isfile(checkpoint):
        sys.exit(f"Checkpoint not found: {checkpoint}")

    run_calibration(config, checkpoint, args.output, use_tta=args.tta)


if __name__ == "__main__":
    main()
