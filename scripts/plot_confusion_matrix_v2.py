"""
Vẽ confusion matrix cho v2 best_model.pth trên NIH official test set.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\plot_confusion_matrix_v2.py
    .\\venv\\Scripts\\python.exe scripts\\plot_confusion_matrix_v2.py --out outputs_nih/v2/confusion_matrices.png
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.amp import autocast
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.dataset import build_dataloaders, CHEXPERT_LABELS


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_calibration(calib_path="configs/calibration.json") -> dict:
    calibration = {
        "thresholds": {lbl: 0.5 for lbl in CHEXPERT_LABELS},
        "temperature": 1.0,
        "per_label_temperatures": {},
        "per_label_isotonic": {},
    }
    if os.path.isfile(calib_path):
        with open(calib_path, encoding="utf-8") as f:
            calib = json.load(f)
        if isinstance(calib.get("thresholds"), dict):
            calibration["thresholds"].update({
                label: float(value)
                for label, value in calib["thresholds"].items()
                if label in CHEXPERT_LABELS
            })
        try:
            temperature = float(calib.get("temperature", 1.0))
            if np.isfinite(temperature) and temperature > 0:
                calibration["temperature"] = temperature
        except (TypeError, ValueError):
            pass
        if isinstance(calib.get("per_label_temperatures"), dict):
            calibration["per_label_temperatures"] = {
                label: float(value)
                for label, value in calib["per_label_temperatures"].items()
                if label in CHEXPERT_LABELS
            }
        if isinstance(calib.get("per_label_isotonic"), dict):
            calibration["per_label_isotonic"] = calib["per_label_isotonic"]
    return calibration


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def apply_calibration(raw_probs, logits, calibration, label_names):
    """Match app inference: per-label isotonic first, otherwise temperature scaling."""
    out = np.array(raw_probs, dtype=np.float64, copy=True)
    global_t = float(calibration.get("temperature", 1.0))
    per_label_t = calibration.get("per_label_temperatures", {}) or {}
    per_label_iso = calibration.get("per_label_isotonic", {}) or {}

    for j, label in enumerate(label_names):
        iso_map = per_label_iso.get(label)
        if isinstance(iso_map, dict) and iso_map.get("x") and iso_map.get("y"):
            x = np.asarray(iso_map["x"], dtype=np.float64)
            y = np.asarray(iso_map["y"], dtype=np.float64)
            if len(x) >= 2 and len(x) == len(y):
                out[:, j] = np.interp(out[:, j], x, y)
                continue

        try:
            temperature = float(per_label_t.get(label, global_t))
        except (TypeError, ValueError):
            temperature = global_t
        if np.isfinite(temperature) and temperature > 0 and abs(temperature - 1.0) > 1e-8:
            out[:, j] = sigmoid_np(logits[:, j] / temperature)

    return out.astype(np.float32)


def collect_predictions(model, loader, device, calibration=None, label_names=None, use_fp16=True):
    model.eval()
    all_p, all_y, all_m = [], [], []
    if label_names is None:
        label_names = CHEXPERT_LABELS
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=90, leave=False):
            images    = batch["image"].to(device, non_blocking=True)
            labels    = batch["labels"]
            mask      = batch["mask"]
            view_type = batch.get("view_type")
            if view_type is not None:
                view_type = view_type.to(device)
            with autocast("cuda", enabled=use_fp16 and device.type == "cuda"):
                logits = model(images, view_type=view_type)
            logits_np = logits.float().cpu().numpy()
            probs = sigmoid_np(logits_np)
            if calibration is not None:
                probs = apply_calibration(probs, logits_np, calibration, label_names)
            all_p.append(probs)
            all_y.append(labels.numpy())
            all_m.append(mask.numpy())
    return (
        np.concatenate(all_p, 0),
        np.concatenate(all_y, 0),
        np.concatenate(all_m, 0),
    )


def per_label_cm(probs, labels, mask, thresholds, label_names):
    """
    Trả về dict {label: {"tp": , "fp": , "fn": , "tn": , ...}} dùng per-label mask.
    """
    results = {}
    for i, name in enumerate(label_names):
        valid = mask[:, i] > 0.5
        yt = labels[valid, i].astype(int)
        yp = (probs[valid, i] >= thresholds.get(name, 0.5)).astype(int)

        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        total = tp + tn + fp + fn
        pos   = tp + fn
        neg   = tn + fp

        sens     = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec     = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ppv      = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv      = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        f1       = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else float("nan")
        acc      = (tp + tn) / total if total > 0 else float("nan")
        prev     = pos / total if total > 0 else float("nan")

        results[name] = dict(
            tp=tp, tn=tn, fp=fp, fn=fn,
            total=total, pos=pos, neg=neg,
            sensitivity=sens, specificity=spec,
            ppv=ppv, npv=npv, f1=f1, accuracy=acc,
            prevalence=prev,
            threshold=thresholds.get(name, 0.5),
        )
    return results


def draw_cm_grid_thesis(cm_data, label_names, out_path):
    """Hình 3.11 — nền sáng, nhãn tiếng Việt, lưới 4 cột gọn (khóa luận)."""
    ncols = 4
    nrows = (len(label_names) + ncols - 1) // ncols

    fig = plt.figure(figsize=(12.5, 3.15 * nrows), facecolor="white")
    outer = gridspec.GridSpec(
        nrows, ncols, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.06, right=0.98, top=0.97, bottom=0.04,
    )

    cmap = plt.cm.Blues
    for idx, name in enumerate(label_names):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(outer[row, col])
        d = cm_data[name]

        cm_arr = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]], dtype=float)
        vmax = max(cm_arr.max(), 1.0)
        ax.imshow(cm_arr, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")

        for r in range(2):
            for c in range(2):
                val = int(cm_arr[r, c])
                txt_color = "white" if cm_arr[r, c] > vmax * 0.55 else "#1a1a1a"
                ax.text(
                    c, r, f"{val:,}",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold", color=txt_color,
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Âm tính", "Dương tính"], fontsize=7)
        ax.set_yticklabels(["Âm tính", "Dương tính"], fontsize=7)
        ax.set_xlabel("Nhãn dự đoán", fontsize=7, labelpad=1)
        if col == 0:
            ax.set_ylabel("Nhãn thực tế", fontsize=7, labelpad=1)

        label_disp = name.replace("_", " ")
        thr = d["threshold"]
        prev_pct = d["prevalence"] * 100
        ax.set_title(
            f"{label_disp}\nNgưỡng = {thr:.3f}; tỷ lệ dương = {prev_pct:.1f}%",
            fontsize=8, fontweight="bold", pad=4,
        )

    for idx in range(len(label_names), nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(outer[row, col])
        ax.set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved (thesis style) → {out_path}")


def load_cm_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        cm_data = json.load(f)
    label_names = [lbl for lbl in CHEXPERT_LABELS if lbl in cm_data]
    if not label_names:
        label_names = list(cm_data.keys())
    return cm_data, label_names


def draw_cm_grid(cm_data, label_names, out_path, title_extra=""):
    """Vẽ 14 ô confusion matrix trong 1 figure lớn (dark dashboard style)."""
    n = len(label_names)
    ncols = 5
    nrows = (n + ncols - 1) // ncols   # 3 rows for 14 labels

    fig = plt.figure(figsize=(ncols * 3.8, nrows * 3.8 + 1.2))
    fig.patch.set_facecolor("#1a1a2e")

    # Title chính
    fig.suptitle(
        f"Confusion Matrix — NIH DenseNet121 v2{title_extra}\n"
        f"Test set  ({cm_data[label_names[0]]['total']:,} samples per label w/ valid annotation)",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35,
                              left=0.04, right=0.98, top=0.91, bottom=0.04)

    cmap_pos = plt.cm.Blues
    cmap_neg = plt.cm.Reds

    for idx, name in enumerate(label_names):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(outer[row, col])

        d = cm_data[name]
        cm_arr = np.array([[d["tn"], d["fp"]],
                           [d["fn"], d["tp"]]])

        # Màu ô: đúng → xanh, sai → đỏ nhạt
        colors = np.array([
            [cmap_pos(0.45), cmap_neg(0.45)],
            [cmap_neg(0.45), cmap_pos(0.75)],
        ])

        for r in range(2):
            for c_ in range(2):
                ax.add_patch(plt.Rectangle(
                    (c_ - 0.5, r - 0.5), 1, 1,
                    facecolor=colors[r][c_], edgecolor="white", linewidth=0.8, zorder=1
                ))
                val = cm_arr[r, c_]
                ax.text(c_, r, f"{val:,}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white", zorder=2)

        sens_str = f"{d['sensitivity']:.3f}" if not np.isnan(d['sensitivity']) else "N/A"
        spec_str = f"{d['specificity']:.3f}" if not np.isnan(d['specificity']) else "N/A"
        f1_str   = f"{d['f1']:.3f}"          if not np.isnan(d['f1'])          else "N/A"
        ppv_str  = f"{d['ppv']:.3f}"         if not np.isnan(d['ppv'])         else "N/A"

        # Ticks + labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred−", "Pred+"], fontsize=7.5, color="white")
        ax.set_yticklabels(["Actual−", "Actual+"], fontsize=7.5, color="white", rotation=90, va="center")
        ax.xaxis.set_tick_params(colors="white")
        ax.yaxis.set_tick_params(colors="white")
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        label_disp = name.replace("_", " ")
        thresh_disp = d["threshold"]
        prev_pct = d["prevalence"] * 100

        ax.set_title(
            f"{label_disp}\nThr={thresh_disp:.3f}  Prev={prev_pct:.1f}%",
            fontsize=8.5, fontweight="bold", color="white", pad=4
        )

        # Metrics dưới ô
        ax.set_xlabel(
            f"Sens={sens_str}  Spec={spec_str}\nF1={f1_str}  PPV={ppv_str}",
            fontsize=7, color="#cccccc", labelpad=4
        )

        ax.set_facecolor("#1a1a2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555555")

    # Xóa ô dư (nếu có)
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(outer[row, col])
        ax.set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_summary_table(cm_data, label_names):
    """In bảng tóm tắt ra console."""
    header = f"{'Label':<22} {'Thr':>5} {'Prev%':>6} {'Sens':>6} {'Spec':>6} {'PPV':>6} {'NPV':>6} {'F1':>6} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}"
    print("\n" + "=" * len(header))
    print("Confusion Matrix Summary — v2 Test Set")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt(v, prec=3):
        return f"{v:.{prec}f}" if (v is not None and not np.isnan(v)) else "  N/A"

    for name in label_names:
        d = cm_data[name]
        print(
            f"{name:<22} {d['threshold']:>5.3f} {d['prevalence']*100:>6.1f} "
            f"{fmt(d['sensitivity']):>6} {fmt(d['specificity']):>6} "
            f"{fmt(d['ppv']):>6} {fmt(d['npv']):>6} {fmt(d['f1']):>6} "
            f"{d['tp']:>6} {d['tn']:>6} {d['fp']:>6} {d['fn']:>6}"
        )

    # Aggregate (macro avg over non-nan)
    def macro(key):
        vals = [cm_data[n][key] for n in label_names if not np.isnan(cm_data[n][key])]
        return np.mean(vals) if vals else float("nan")

    print("-" * len(header))
    print(
        f"{'MACRO AVG':<22} {'':>5} {'':>6} "
        f"{fmt(macro('sensitivity')):>6} {fmt(macro('specificity')):>6} "
        f"{fmt(macro('ppv')):>6} {fmt(macro('npv')):>6} {fmt(macro('f1')):>6}"
    )
    print("=" * len(header))


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--calib",  default="configs/calibration.json")
    parser.add_argument("--out",    default="outputs_nih/v2/confusion_matrices.png")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--raw-probs",
        action="store_true",
        help="Use raw sigmoid probabilities while still reading thresholds from --calib.",
    )
    parser.add_argument(
        "--style",
        choices=("thesis", "dark"),
        default="thesis",
        help="thesis = Hình 3.11 (nền sáng, tiếng Việt); dark = dashboard cũ.",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        metavar="PATH",
        help="Vẽ lại từ file JSON đã có — bỏ qua inference.",
    )
    args = parser.parse_args()

    if args.from_json:
        cm_data, label_names = load_cm_json(args.from_json)
        print(f"Loaded {len(label_names)} labels from {args.from_json}")
        print_summary_table(cm_data, label_names)
        if args.style == "thesis":
            draw_cm_grid_thesis(cm_data, label_names, args.out)
        else:
            draw_cm_grid(cm_data, label_names, args.out)
        return

    config = load_config(args.config)
    device = get_device()
    print(f"Device: {device}")

    # ── Load model ──
    checkpoint = config["paths"]["densenet_checkpoint"]
    print(f"Loading checkpoint: {checkpoint}")
    model = load_trained_model(checkpoint, config, device)
    model.eval()

    # ── Load test dataloader ──
    config["cnn"]["training"]["batch_size"] = args.batch_size
    config["cnn"]["training"]["num_workers"] = args.num_workers
    _, _, test_loader, _ = build_dataloaders(config)
    print(f"Test samples: {len(test_loader.dataset):,}")

    # ── Apply calibration thresholds ──
    calibration = load_calibration(args.calib)
    thresholds = calibration["thresholds"]
    print("Thresholds from:", args.calib)
    if args.raw_probs:
        print("Probability mode: raw sigmoid (no temperature/isotonic calibration)")
        calibration_for_inference = None
    else:
        print("Probability mode: calibrated (temperature/isotonic from calibration file)")
        calibration_for_inference = calibration

    # ── Inference ──
    probs, labels, mask = collect_predictions(
        model,
        test_loader,
        device,
        calibration=calibration_for_inference,
        label_names=CHEXPERT_LABELS,
        use_fp16=not args.no_fp16,
    )
    print(f"probs shape: {probs.shape}")

    # ── Compute confusion matrices ──
    cm_data = per_label_cm(probs, labels, mask, thresholds, CHEXPERT_LABELS)

    # ── Print table ──
    print_summary_table(cm_data, CHEXPERT_LABELS)

    # ── Draw figure ──
    if args.style == "thesis":
        draw_cm_grid_thesis(cm_data, CHEXPERT_LABELS, args.out)
    else:
        draw_cm_grid(cm_data, CHEXPERT_LABELS, args.out)

    # ── Save JSON alongside ──
    json_out = args.out.replace(".png", ".json")
    with open(json_out, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"JSON saved → {json_out}")


if __name__ == "__main__":
    main()
