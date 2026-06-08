"""
So sánh Grad-CAM: Hệ thống đề xuất (v2) vs Thiết lập tham chiếu (v1).
Mỗi model dùng preprocessing đúng config của nó (CLAHE/corner erase).
Output: outputs/thesis/gradcam_de_xuat_vs_tham_chieu/
"""
from __future__ import annotations

import csv
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cnn.grad_cam import NIH_LABELS, generate_gradcam
from src.cnn.model import load_trained_model
from src.utils import get_device, load_config

OUT_DIR = "outputs/thesis/gradcam_de_xuat_vs_tham_chieu"
COMPARE_LABELS = [
    "Cardiomegaly",
    "Effusion",
    "Pneumothorax",
    "Edema",
    "Mass",
    "Atelectasis",
    "Consolidation",
    "Infiltration",
]

SETUPS = {
    "de_xuat": {
        "title": "De xuat (FZLPR v2)",
        "config": "configs/config.yaml",
        "checkpoint": "./models/nih_densenet121/v2/best_model.pth",
    },
    "tham_chieu": {
        "title": "Tham chieu (FZLPR minimal)",
        "config": "configs/config_tham_chieu.yaml",
        "checkpoint": "./models/nih_densenet121/tham_chieu/best_model.pth",
    },
}


def _preprocess_flags(cfg: dict) -> tuple[dict, bool]:
    aug = cfg.get("cnn", {}).get("augmentation", {})
    clahe_cfg = aug.get("clahe_preprocessing", {})
    if not clahe_cfg.get("enabled", False):
        clahe_cfg = None
    corner_erase_enabled = bool(aug.get("corner_erase", {}).get("enabled", True))
    return clahe_cfg, corner_erase_enabled


def _pick_manifest(cfg: dict, labels: list[str], seed: int = 42) -> list[dict]:
    test_csv = cfg["paths"]["test_csv"]
    images_dir = cfg["paths"]["images_dir"]
    df = pd.read_csv(test_csv)
    rng = np.random.default_rng(seed)
    manifest = []
    for label in labels:
        if label not in df.columns:
            continue
        subset = df[df[label] == 1]
        if subset.empty:
            continue
        row = subset.iloc[int(rng.integers(0, min(len(subset), 200)))]
        rel = row["Path"]
        img_path = os.path.join(images_dir, rel)
        if not os.path.isfile(img_path):
            parts = str(rel).split("/", 1)
            if len(parts) > 1:
                img_path = os.path.join(images_dir, parts[1])
        if os.path.isfile(img_path):
            manifest.append({"label": label, "image": img_path})
    return manifest


def _side_by_side(left: np.ndarray, right: np.ndarray, title_l: str, title_r: str) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    w = max(left.shape[1], right.shape[1])

    def pad(img):
        out = np.zeros((h, w, 3), dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out

    left = pad(left)
    right = pad(right)
    divider = np.full((h, 4, 3), 180, dtype=np.uint8)
    body = np.hstack([left, divider, right])
    bar_h = 36
    bar = np.full((bar_h, body.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(bar, title_l, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, title_r, (w + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 255, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, body])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="So sanh Grad-CAM de xuat vs tham chieu.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Chi chay mot hoac nhieu nhan (vd: Effusion Pneumothorax). Mac dinh: tat ca.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed chon anh test.")
    args = parser.parse_args()

    labels = args.labels or COMPARE_LABELS
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg_dx = load_config(SETUPS["de_xuat"]["config"])
    cfg_ref = load_config(SETUPS["tham_chieu"]["config"])
    device = get_device(cfg_dx)

    manifest = _pick_manifest(cfg_dx, labels, seed=args.seed)
    if not manifest:
        raise SystemExit("Khong tim duoc anh test cho manifest.")

    models = {}
    flags = {}
    for key, meta in SETUPS.items():
        cfg = load_config(meta["config"])
        print(f"Loading {key}: {meta['checkpoint']}")
        models[key] = load_trained_model(meta["checkpoint"], cfg, device)
        models[key].eval()
        flags[key] = _preprocess_flags(cfg)

    gcfg = cfg_dx.get("gradcam", {})
    norm_cfg = cfg_dx["cnn"].get("normalization", {})
    image_size = cfg_dx["cnn"]["image_size"]
    bottom_crop = float(cfg_dx["cnn"].get("augmentation", {}).get("bottom_crop_ratio", 0.0))
    target_layer = gcfg.get("target_layers", "features.norm5")
    cam_method = gcfg.get("cam_method", "gradcam")
    alpha = float(gcfg.get("alpha", 0.5))

    summary_rows = []

    for entry in manifest:
        label = entry["label"]
        img_path = entry["image"]
        idx = NIH_LABELS.index(label)
        print(f"\n[{label}] {img_path}")

        overlays = {}
        qualities = {}

        for key in ("de_xuat", "tham_chieu"):
            cfg = load_config(SETUPS[key]["config"])
            clahe_cfg, corner_erase = flags[key]
            res = generate_gradcam(
                models[key],
                img_path,
                target_class_idx=idx,
                image_size=image_size,
                target_layer_name=target_layer,
                alpha=alpha,
                postprocess_cfg=gcfg,
                cam_method=cam_method,
                bottom_crop_ratio=bottom_crop,
                normalization_cfg=norm_cfg,
                clahe_cfg=clahe_cfg,
                corner_erase_enabled=corner_erase,
                label_name=label,
            )
            overlays[key] = res["heatmap_overlay"]
            q = res.get("quality", {})
            qualities[key] = q.get("final", {})
            fin = qualities[key]
            print(
                f"  {key}: expected_zone={fin.get('expected_zone_ratio', 0):.3f} "
                f"corner={fin.get('focus_on_top_corners_ratio', 0):.3f} "
                f"edge={fin.get('focus_on_edges_ratio', 0):.3f} "
                f"class={q.get('classification', 'n/a')}"
            )

        combo = _side_by_side(
            overlays["de_xuat"],
            overlays["tham_chieu"],
            "De xuat",
            "Tham chieu",
        )
        out_path = os.path.join(OUT_DIR, f"{label}_compare.png")
        cv2.imwrite(out_path, cv2.cvtColor(combo, cv2.COLOR_RGB2BGR))
        print(f"  -> {out_path}")

        q_dx = qualities["de_xuat"]
        q_ref = qualities["tham_chieu"]
        summary_rows.append(
            {
                "label": label,
                "image": img_path,
                "de_xuat_expected_zone": q_dx.get("expected_zone_ratio"),
                "tham_chieu_expected_zone": q_ref.get("expected_zone_ratio"),
                "de_xuat_corner_ratio": q_dx.get("focus_on_top_corners_ratio"),
                "tham_chieu_corner_ratio": q_ref.get("focus_on_top_corners_ratio"),
                "de_xuat_edge_ratio": q_dx.get("focus_on_edges_ratio"),
                "tham_chieu_edge_ratio": q_ref.get("focus_on_edges_ratio"),
                "winner_expected_zone": (
                    "de_xuat"
                    if (q_dx.get("expected_zone_ratio") or 0) >= (q_ref.get("expected_zone_ratio") or 0)
                    else "tham_chieu"
                ),
                "winner_lower_corner": (
                    "de_xuat"
                    if (q_dx.get("focus_on_top_corners_ratio") or 1) <= (q_ref.get("focus_on_top_corners_ratio") or 1)
                    else "tham_chieu"
                ),
            }
        )

    csv_path = os.path.join(OUT_DIR, "gradcam_qc_summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n = len(summary_rows)
    exp_dx = sum(r["de_xuat_expected_zone"] or 0 for r in summary_rows) / n
    exp_ref = sum(r["tham_chieu_expected_zone"] or 0 for r in summary_rows) / n
    cor_dx = sum(r["de_xuat_corner_ratio"] or 0 for r in summary_rows) / n
    cor_ref = sum(r["tham_chieu_corner_ratio"] or 0 for r in summary_rows) / n
    win_exp = sum(1 for r in summary_rows if r["winner_expected_zone"] == "de_xuat")
    win_cor = sum(1 for r in summary_rows if r["winner_lower_corner"] == "de_xuat")

    print("\n" + "=" * 60)
    print("TOM TAT QC (trung binh tren", n, "anh)")
    print(f"  expected_zone: de_xuat={exp_dx:.3f}  tham_chieu={exp_ref:.3f}")
    print(f"  corner_ratio:  de_xuat={cor_dx:.3f}  tham_chieu={cor_ref:.3f}")
    print(f"  thang expected_zone: de_xuat {win_exp}/{n}")
    print(f"  thang it corner hon:   de_xuat {win_cor}/{n}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
