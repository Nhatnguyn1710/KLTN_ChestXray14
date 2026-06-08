"""
Đánh giá v2 best_model.pth trên NIH official test set.
Chạy 1 lần forward pass trên toàn bộ test set (~25.6k ảnh), sau đó bootstrap
B lần để ước lượng mean ± 95% CI cho AUC và AUPRC.

Tùy chọn --tta: dùng cùng pipeline TTA như inference/calibrate (`batch_tta_from_paths`)
để so sánh với các paper báo cáo mean AUC kèm TTA — **không cần train lại**.

Usage:
    python scripts/evaluate_v2_test.py
    python scripts/evaluate_v2_test.py --bootstrap 2000
    python scripts/evaluate_v2_test.py --config configs/config_bce.yaml --tta --output outputs_nih_bce/v1/test_results_tta.json
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.dataset import build_dataloaders, CHEXPERT_LABELS
from src.cnn.inference import batch_tta_from_paths
from torch.amp import autocast
from tqdm import tqdm


def collect_predictions(model, loader, device, use_fp16=True):
    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"]
            mask   = batch["mask"]
            view_type = batch["view_type"].to(device) if "view_type" in batch else None
            with autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
                logits = model(images, view_type=view_type)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            all_p.append(probs)
            all_y.append(labels.numpy())
            all_m.append(mask.numpy())
    return (
        np.concatenate(all_p, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_m, axis=0),
    )


def collect_predictions_tta(model, loader, device, cfg, use_fp16=True):
    """TTA đánh giá — khớp Grad-CAM / calibrate --tta (đọc ảnh từ disk + gộp logit)."""
    cnn_cfg = cfg["cnn"]
    paths_cfg = cfg["paths"]
    tta_cfg = cnn_cfg.get("tta", {})
    if not tta_cfg.get("enabled", True):
        raise ValueError(
            "cnn.tta.enabled là false trong config. Bật enabled: true trong config.yaml "
            "hoặc bỏ --tta để chạy forward đơn."
        )
    image_root = paths_cfg["dataset_dir"]
    image_size = cnn_cfg["image_size"]
    norm_cfg = cnn_cfg.get("normalization", {})
    clahe_cfg = cnn_cfg.get("augmentation", {}).get("clahe_preprocessing", {})
    bottom_crop_ratio = float(cnn_cfg.get("augmentation", {}).get("bottom_crop_ratio", 0.0))

    model.eval()
    all_p, all_y, all_m = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="TTA inference", leave=False):
            paths = batch["path"]
            if isinstance(paths, str):
                paths = [paths]
            else:
                paths = list(paths)
            view_type = batch.get("view_type")
            if view_type is not None:
                view_type = view_type.to(device)
            logits = batch_tta_from_paths(
                paths,
                image_root,
                model,
                device,
                image_size,
                tta_cfg,
                normalization_cfg=norm_cfg,
                clahe_cfg=clahe_cfg,
                bottom_crop_ratio=bottom_crop_ratio,
                view_types=view_type,
                use_fp16=use_fp16,
            )
            probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)
            all_p.append(probs)
            all_y.append(batch["labels"].numpy())
            all_m.append(batch["mask"].numpy())
    return (
        np.concatenate(all_p, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_m, axis=0),
    )


def compute_metrics(y, p, m, labels):
    """AUC + AUPRC per-class, with valid-mask filter."""
    auc, auprc = {}, {}
    for i, name in enumerate(labels):
        valid = m[:, i] > 0.5
        yt, yp = y[valid, i], p[valid, i]
        if len(np.unique(yt)) < 2:
            auc[name] = None
            auprc[name] = None
            continue
        auc[name]   = float(roc_auc_score(yt, yp))
        auprc[name] = float(average_precision_score(yt, yp))
    return auc, auprc


def bootstrap_metrics(y, p, m, labels, n_boot=1000, seed=42):
    """Bootstrap resample image indices to get CI for per-class + mean metrics."""
    rng = np.random.default_rng(seed)
    N = y.shape[0]
    auc_runs   = {name: [] for name in labels}
    auprc_runs = {name: [] for name in labels}
    auc_mean_runs, auprc_mean_runs = [], []

    for _ in tqdm(range(n_boot), desc="Bootstrap", leave=False):
        idx = rng.integers(0, N, size=N)
        ys, ps, ms = y[idx], p[idx], m[idx]
        per_auc, per_pr = compute_metrics(ys, ps, ms, labels)
        valid_auc, valid_pr = [], []
        for name in labels:
            if per_auc[name] is not None:
                auc_runs[name].append(per_auc[name])
                valid_auc.append(per_auc[name])
            if per_pr[name] is not None:
                auprc_runs[name].append(per_pr[name])
                valid_pr.append(per_pr[name])
        if valid_auc:
            auc_mean_runs.append(np.mean(valid_auc))
        if valid_pr:
            auprc_mean_runs.append(np.mean(valid_pr))

    def stats(arr):
        if not len(arr):
            return None
        a = np.array(arr)
        return {
            "mean":   float(a.mean()),
            "std":    float(a.std()),
            "ci_lo":  float(np.percentile(a, 2.5)),
            "ci_hi":  float(np.percentile(a, 97.5)),
            "n":      int(len(a)),
        }

    return {
        "auc_per_class":   {k: stats(v) for k, v in auc_runs.items()},
        "auprc_per_class": {k: stats(v) for k, v in auprc_runs.items()},
        "auc_mean":   stats(auc_mean_runs),
        "auprc_mean": stats(auprc_mean_runs),
    }


def threshold_metrics(y, p, m, labels, threshold=0.5):
    """F1 / sensitivity / specificity at fixed threshold."""
    out = {}
    for i, name in enumerate(labels):
        valid = m[:, i] > 0.5
        yt = y[valid, i].astype(int)
        yp = (p[valid, i] >= threshold).astype(int)
        if len(np.unique(yt)) < 2:
            out[name] = None
            continue
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1   = 2 * prec * sens / max(prec + sens, 1e-9)
        prevalence = (tp + fn) / max(tp + fp + tn + fn, 1)
        out[name] = {
            "f1":   round(f1, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "precision":   round(prec, 4),
            "prevalence":  round(prevalence, 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--output",
        default=None,
        help="JSON output path (mặc định: test_results_bootstrap.json hoặc test_results_tta.json nếu --tta)",
    )
    ap.add_argument(
        "--tta",
        action="store_true",
        help="Bật test-time augmentation (chậm hơn nhiều; cùng pipeline với calibrate --tta).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    ckpt   = cfg["paths"]["densenet_checkpoint"]

    out_path = args.output or (
        "outputs_nih/v2/test_results_tta.json"
        if args.tta
        else "outputs_nih/v2/test_results_bootstrap.json"
    )

    print("=" * 70)
    print("V2 EVALUATION ON NIH OFFICIAL TEST SET")
    print("=" * 70)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt}")
    print(f"Device:     {device}")
    print(f"TTA:        {'ON (paper-style)' if args.tta else 'OFF (single forward)'}")
    print(f"Bootstrap:  {args.bootstrap} iterations")
    print()

    print("[1/3] Loading model + test loader ...")
    model = load_trained_model(ckpt, cfg, device)
    _, _, test_loader, _ = build_dataloaders(cfg)
    print(f"  Test set: {len(test_loader.dataset):,} images")

    print("\n[2/3] Forward pass on full test set ...")
    use_fp16 = cfg["cnn"]["training"].get("use_fp16", True)
    if args.tta:
        probs, labels_arr, masks = collect_predictions_tta(model, test_loader, device, cfg, use_fp16=use_fp16)
    else:
        probs, labels_arr, masks = collect_predictions(model, test_loader, device, use_fp16=use_fp16)
    print(f"  Probs:  {probs.shape}")
    print(f"  Labels: {labels_arr.shape}")

    # Point estimate
    auc_pt, auprc_pt = compute_metrics(labels_arr, probs, masks, CHEXPERT_LABELS)
    valid_auc = [v for v in auc_pt.values() if v is not None]
    valid_pr  = [v for v in auprc_pt.values() if v is not None]
    point_auc   = float(np.mean(valid_auc)) if valid_auc else 0.0
    point_auprc = float(np.mean(valid_pr))  if valid_pr else 0.0

    print(f"\n  Point AUC mean:   {point_auc:.4f}")
    print(f"  Point AUPRC mean: {point_auprc:.4f}")

    print(f"\n[3/3] Bootstrap × {args.bootstrap} ...")
    boot = bootstrap_metrics(
        labels_arr, probs, masks, CHEXPERT_LABELS,
        n_boot=args.bootstrap, seed=args.seed,
    )
    thr_metrics = threshold_metrics(
        labels_arr, probs, masks, CHEXPERT_LABELS, threshold=args.threshold,
    )

    # ── Pretty print ──
    print("\n" + "=" * 70)
    print("PER-CLASS RESULTS (AUC ± 95% CI  |  AUPRC ± 95% CI  |  prev)")
    print("=" * 70)
    print(f"{'Label':<22} {'AUC (95% CI)':<22} {'AUPRC (95% CI)':<22} {'Prev':>6}")
    print("-" * 70)
    for name in CHEXPERT_LABELS:
        a = boot["auc_per_class"].get(name)
        p_ = boot["auprc_per_class"].get(name)
        prev = thr_metrics.get(name, {}).get("prevalence", 0.0) if thr_metrics.get(name) else 0.0
        a_str = f"{a['mean']:.3f} ({a['ci_lo']:.3f}-{a['ci_hi']:.3f})" if a else "N/A"
        p_str = f"{p_['mean']:.3f} ({p_['ci_lo']:.3f}-{p_['ci_hi']:.3f})" if p_ else "N/A"
        print(f"{name:<22} {a_str:<22} {p_str:<22} {prev:>6.3f}")

    print("\n" + "=" * 70)
    print("OVERALL (mean across 14 labels, bootstrap N=" + str(args.bootstrap) + ")")
    print("=" * 70)
    am = boot["auc_mean"];   pm = boot["auprc_mean"]
    if am:
        print(f"  Mean AUC:   {am['mean']:.4f}  ± {am['std']:.4f}   "
              f"(95% CI: {am['ci_lo']:.4f} – {am['ci_hi']:.4f})")
    if pm:
        print(f"  Mean AUPRC: {pm['mean']:.4f}  ± {pm['std']:.4f}   "
              f"(95% CI: {pm['ci_lo']:.4f} – {pm['ci_hi']:.4f})")

    print(f"\nThreshold = {args.threshold}  → F1 / Sens / Spec per class:")
    print("-" * 70)
    print(f"{'Label':<22} {'F1':>7} {'Sens':>7} {'Spec':>7} {'Prec':>7}")
    for name in CHEXPERT_LABELS:
        m = thr_metrics.get(name)
        if m is None:
            print(f"{name:<22} {'N/A':>7}")
        else:
            print(f"{name:<22} {m['f1']:>7.3f} {m['sensitivity']:>7.3f} "
                  f"{m['specificity']:>7.3f} {m['precision']:>7.3f}")

    # ── Save JSON ──
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "checkpoint":       ckpt,
        "test_set":         "nih_official_test",
        "tta":              bool(args.tta),
        "num_images":       int(probs.shape[0]),
        "num_classes":      len(CHEXPERT_LABELS),
        "bootstrap_n":      args.bootstrap,
        "threshold":        args.threshold,
        "point_auc_mean":   point_auc,
        "point_auprc_mean": point_auprc,
        "bootstrap":        boot,
        "threshold_metrics": thr_metrics,
        "auc_per_class_point":   auc_pt,
        "auprc_per_class_point": auprc_pt,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  → Saved: {out_path}")

    # ── Conclusion ──
    print("\n" + "=" * 70)
    print("KẾT LUẬN V2")
    print("=" * 70)
    if am and pm:
        gap_auc = am["ci_hi"] - am["ci_lo"]
        print(f"• Mean AUC = {am['mean']:.3f} (95% CI hẹp ≈ {gap_auc:.3f}) → ổn định.")
        print(f"• Mean AUPRC = {pm['mean']:.3f} → kém hơn AUC do mất cân bằng class.")
        # Find best/worst
        ranked = sorted(
            [(n, boot["auc_per_class"][n]) for n in CHEXPERT_LABELS
             if boot["auc_per_class"][n]],
            key=lambda x: x[1]["mean"], reverse=True,
        )
        if ranked:
            top3    = ranked[:3]
            bottom3 = ranked[-3:]
            print(f"• Top-3 AUC:    " + ", ".join(f"{n}={s['mean']:.3f}" for n, s in top3))
            print(f"• Bottom-3 AUC: " + ", ".join(f"{n}={s['mean']:.3f}" for n, s in bottom3))
        if am["mean"] >= 0.80:
            verdict = "TỐT — đạt mức công bố trên NIH ChestX-ray14."
        elif am["mean"] >= 0.75:
            verdict = "KHÁ — còn cải thiện được với calibration / TTA / ensemble."
        else:
            verdict = "CẦN TỐI ƯU thêm."
        print(f"• Đánh giá tổng thể: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
