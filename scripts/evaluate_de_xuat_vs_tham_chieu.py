"""
Danh gia tong the: He thong de xuat (v2) vs Thiet lap tham chieu.
- So sanh heatmap cung anh (side-by-side + 4-panel: overlay + pure heatmap)
- QC Grad-CAM nhieu mau (labels x images_per_label)
- Xac suat du doan tren cung anh (prob calibration tren mau positive)
- Tong hop de xuat khuyen nghi

Output: outputs/thesis/de_xuat_vs_tham_chieu/
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cnn.grad_cam import NIH_LABELS, generate_gradcam
from src.cnn.inference import preprocess_image
from src.cnn.model import load_trained_model
from src.utils import get_device, load_config

OUT_DIR = ROOT / "outputs/thesis/de_xuat_vs_tham_chieu"
COMPARE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

SETUPS = {
    "de_xuat": {
        "config": "configs/config.yaml",
        "checkpoint": "models/nih_densenet121/v2/best_model.pth",
    },
    "tham_chieu": {
        "config": "configs/config_tham_chieu.yaml",
        "checkpoint": "models/nih_densenet121/tham_chieu/best_model.pth",
    },
}

IMAGES_PER_LABEL = 5
SEED = 42


def _resolve_image(images_dir: str, rel: str) -> str | None:
    p = os.path.join(images_dir, rel)
    if os.path.isfile(p):
        return p
    parts = str(rel).split("/", 1)
    if len(parts) > 1:
        p2 = os.path.join(images_dir, parts[1])
        if os.path.isfile(p2):
            return p2
    return None


def _preprocess_flags(cfg: dict) -> tuple[dict | None, bool]:
    aug = cfg.get("cnn", {}).get("augmentation", {})
    clahe_cfg = aug.get("clahe_preprocessing", {})
    clahe = clahe_cfg if clahe_cfg.get("enabled") else None
    corner = bool(aug.get("corner_erase", {}).get("enabled", True))
    return clahe, corner


def _pick_samples(cfg: dict, labels: list[str], n_per_label: int, seed: int) -> list[dict]:
    df = pd.read_csv(cfg["paths"]["test_csv"])
    images_dir = cfg["paths"]["images_dir"]
    rng = np.random.default_rng(seed)
    samples = []
    for label in labels:
        if label not in df.columns:
            continue
        subset = df[df[label] == 1]
        if subset.empty:
            continue
        k = min(n_per_label, len(subset))
        idxs = rng.choice(len(subset), size=k, replace=False)
        for i in idxs:
            row = subset.iloc[int(i)]
            img = _resolve_image(images_dir, row["Path"])
            if img:
                samples.append({"label": label, "image": img})
    return samples


def _predict_prob(model, cfg, device, img_path: str, class_idx: int) -> float:
    cnn = cfg["cnn"]
    aug = cnn.get("augmentation", {})
    clahe_cfg = aug.get("clahe_preprocessing", {})
    clahe = clahe_cfg if clahe_cfg.get("enabled") else None
    bottom = float(aug.get("bottom_crop_ratio", 0.0))
    norm = cnn.get("normalization", {})
    size = cnn["image_size"]
    t = preprocess_image(img_path, size, bottom, norm, clahe).to(device)
    with torch.no_grad():
        return float(torch.sigmoid(model(t))[0, class_idx].item())


def _run_cam(model, cfg, device, gcfg, img_path: str, class_idx: int, label: str) -> dict:
    cnn = cfg["cnn"]
    aug = cnn.get("augmentation", {})
    clahe, corner = _preprocess_flags(cfg)
    norm = cnn.get("normalization", {})
    size = cnn["image_size"]
    bottom = float(aug.get("bottom_crop_ratio", 0.0))
    return generate_gradcam(
        model=model,
        image_path=img_path,
        target_class_idx=class_idx,
        image_size=size,
        target_layer_name=gcfg.get("target_layers", "features.norm5"),
        alpha=float(gcfg.get("alpha", 0.5)),
        postprocess_cfg=gcfg,
        cam_method=gcfg.get("cam_method", "gradcam"),
        bottom_crop_ratio=bottom,
        normalization_cfg=norm,
        clahe_cfg=clahe,
        corner_erase_enabled=corner,
        label_name=label,
    )


def _label_bar(img: np.ndarray, text: str, bg=(30, 30, 30)) -> np.ndarray:
    bar = np.full((32, img.shape[1], 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _cam_color(cam_float, w: int, h: int) -> np.ndarray:
    if cam_float is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    u8 = (np.clip(cam_float, 0, 1) * 255).astype(np.uint8)
    rs = cv2.resize(u8, (w, h))
    bgr = cv2.applyColorMap(rs, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _four_panel(dx_res: dict, tc_res: dict, label: str, p_dx: float, p_tc: float) -> np.ndarray:
    o1 = dx_res["heatmap_overlay"]
    o2 = tc_res["heatmap_overlay"]
    h, w = o1.shape[:2]
    o2 = cv2.resize(o2, (w, h)) if o2.shape[:2] != (h, w) else o2
    h1 = _cam_color(dx_res.get("grayscale_cam"), w, h)
    h2 = _cam_color(tc_res.get("grayscale_cam"), w, h)

    top = np.hstack([
        _label_bar(o1, f"De xuat overlay  p={p_dx:.3f}"),
        _label_bar(o2, f"Tham chieu overlay  p={p_tc:.3f}"),
    ])
    bot = np.hstack([
        _label_bar(h1, "De xuat heatmap"),
        _label_bar(h2, "Tham chieu heatmap"),
    ])
    title = np.full((36, top.shape[1], 3), (20, 40, 80), dtype=np.uint8)
    cv2.putText(title, f"{label} — cung anh test", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 200), 2)
    return np.vstack([title, top, bot])


def main() -> None:
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparisons").mkdir(exist_ok=True)

    cfg_dx = load_config(str(ROOT / SETUPS["de_xuat"]["config"]))
    cfg_ref = load_config(str(ROOT / SETUPS["tham_chieu"]["config"]))
    device = get_device(cfg_dx)
    gcfg = cfg_dx.get("gradcam", {})

    models = {}
    for key, meta in SETUPS.items():
        cfg = load_config(str(ROOT / meta["config"]))
        ckpt = str(ROOT / meta["checkpoint"])
        print(f"Loading {key}: {ckpt}")
        models[key] = load_trained_model(ckpt, cfg, device)
        models[key].eval()

    samples = _pick_samples(cfg_dx, COMPARE_LABELS, IMAGES_PER_LABEL, SEED)
    print(f"Samples: {len(samples)} ({IMAGES_PER_LABEL} per label x {len(COMPARE_LABELS)} labels)")

    qc_rows = []
    prob_rows = []
    showcase_done: set[str] = set()

    for s in samples:
        label = s["label"]
        img_path = s["image"]
        idx = NIH_LABELS.index(label)

        p_dx = _predict_prob(models["de_xuat"], cfg_dx, device, img_path, idx)
        p_tc = _predict_prob(models["tham_chieu"], cfg_ref, device, img_path, idx)
        prob_rows.append({
            "label": label,
            "image": img_path,
            "prob_de_xuat": round(p_dx, 4),
            "prob_tham_chieu": round(p_tc, 4),
            "prob_winner": "de_xuat" if p_dx > p_tc else ("tham_chieu" if p_tc > p_dx else "tie"),
        })

        r_dx = _run_cam(models["de_xuat"], cfg_dx, device, gcfg, img_path, idx, label)
        r_tc = _run_cam(models["tham_chieu"], cfg_ref, device, gcfg, img_path, idx, label)
        q_dx = r_dx.get("quality", {})
        q_tc = r_tc.get("quality", {})
        f_dx = q_dx.get("final", {})
        f_tc = q_tc.get("final", {})

        zone_dx = float(f_dx.get("expected_zone_ratio") or 0)
        zone_tc = float(f_tc.get("expected_zone_ratio") or 0)
        cor_dx = float(f_dx.get("focus_on_top_corners_ratio") or 0)
        cor_tc = float(f_tc.get("focus_on_top_corners_ratio") or 0)

        qc_rows.append({
            "label": label,
            "image": img_path,
            "de_xuat_prob": round(p_dx, 4),
            "tham_chieu_prob": round(p_tc, 4),
            "de_xuat_expected_zone": round(zone_dx, 4),
            "tham_chieu_expected_zone": round(zone_tc, 4),
            "de_xuat_corner": round(cor_dx, 4),
            "tham_chieu_corner": round(cor_tc, 4),
            "de_xuat_qc": q_dx.get("classification", "?"),
            "tham_chieu_qc": q_tc.get("classification", "?"),
            "winner_zone": "de_xuat" if zone_dx >= zone_tc else "tham_chieu",
            "winner_corner": "de_xuat" if cor_dx <= cor_tc else "tham_chieu",
            "winner_qc_plausible": (
                "de_xuat" if q_dx.get("classification") == "plausible" and q_tc.get("classification") != "plausible"
                else ("tham_chieu" if q_tc.get("classification") == "plausible" and q_dx.get("classification") != "plausible"
                      else ("both" if q_dx.get("classification") == q_tc.get("classification") == "plausible" else "neither"))
            ),
        })

        if label not in showcase_done:
            panel = _four_panel(r_dx, r_tc, label, p_dx, p_tc)
            out_path = out / "comparisons" / f"{label}_same_image.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            showcase_done.add(label)

    # CSV
    qc_csv = out / "gradcam_qc_multisample.csv"
    with qc_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
        w.writeheader()
        w.writerows(qc_rows)

    n = len(qc_rows)
    stats = {
        "n_samples": n,
        "images_per_label": IMAGES_PER_LABEL,
        "gradcam_qc": {
            "mean_expected_zone_de_xuat": round(np.mean([r["de_xuat_expected_zone"] for r in qc_rows]), 4),
            "mean_expected_zone_tham_chieu": round(np.mean([r["tham_chieu_expected_zone"] for r in qc_rows]), 4),
            "mean_corner_de_xuat": round(np.mean([r["de_xuat_corner"] for r in qc_rows]), 4),
            "mean_corner_tham_chieu": round(np.mean([r["tham_chieu_corner"] for r in qc_rows]), 4),
            "wins_expected_zone_de_xuat": sum(1 for r in qc_rows if r["winner_zone"] == "de_xuat"),
            "wins_expected_zone_tham_chieu": sum(1 for r in qc_rows if r["winner_zone"] == "tham_chieu"),
            "wins_corner_de_xuat": sum(1 for r in qc_rows if r["winner_corner"] == "de_xuat"),
            "wins_corner_tham_chieu": sum(1 for r in qc_rows if r["winner_corner"] == "tham_chieu"),
            "plausible_de_xuat": sum(1 for r in qc_rows if r["de_xuat_qc"] == "plausible"),
            "plausible_tham_chieu": sum(1 for r in qc_rows if r["tham_chieu_qc"] == "plausible"),
            "pipeline_related_de_xuat": sum(1 for r in qc_rows if r["de_xuat_qc"] == "pipeline_related"),
            "pipeline_related_tham_chieu": sum(1 for r in qc_rows if r["tham_chieu_qc"] == "pipeline_related"),
        },
        "prob_on_positive_samples": {
            "mean_prob_de_xuat": round(np.mean([r["prob_de_xuat"] for r in prob_rows]), 4),
            "mean_prob_tham_chieu": round(np.mean([r["prob_tham_chieu"] for r in prob_rows]), 4),
            "wins_prob_de_xuat": sum(1 for r in prob_rows if r["prob_winner"] == "de_xuat"),
            "wins_prob_tham_chieu": sum(1 for r in prob_rows if r["prob_winner"] == "tham_chieu"),
        },
        "test_set_metrics_from_json": {
            "de_xuat": {"auroc": 0.7953, "auprc": 0.2539, "rare_auprc": 0.1516},
            "tham_chieu": {"auroc": 0.8018, "auprc": 0.2629, "rare_auprc": 0.2415},
        },
        "training_best_val": {
            "de_xuat": {"val_auc": 0.8301, "rare_auprc": 0.1077, "best_epoch": 20},
            "tham_chieu": {"val_auc": 0.8357, "rare_auprc": 0.1615, "best_epoch": 16},
        },
    }

    # Weighted score for thesis decision (transparent rubric)
    rubric = []
    rubric.append(("test_auroc", "tham_chieu", 3, "Chẩn đoán cốt lõi"))
    rubric.append(("test_auprc", "tham_chieu", 2, ""))
    rubric.append(("test_rare_auprc", "tham_chieu", 3, "Nhãn hiếm quan trọng lâm sàng"))
    rubric.append(("val_best_auc", "tham_chieu", 1, ""))
    rubric.append(("gradcam_expected_zone", "de_xuat", 2, "Giải thích trực quan"))
    rubric.append(("gradcam_low_corner", "de_xuat", 2, "Tránh nhiễu marker góc phim"))
    rubric.append(("gradcam_plausible_rate", "de_xuat", 2, "QC CAM ổn định"))
    rubric.append(("positive_sample_prob", "tham_chieu" if stats["prob_on_positive_samples"]["mean_prob_tham_chieu"] > stats["prob_on_positive_samples"]["mean_prob_de_xuat"] else "de_xuat", 1, "Xác suất trên mẫu dương tính"))

    scores = {"de_xuat": 0, "tham_chieu": 0}
    rubric_detail = []
    gq = stats["gradcam_qc"]
    winners_map = {
        "test_auroc": "tham_chieu",
        "test_auprc": "tham_chieu",
        "test_rare_auprc": "tham_chieu",
        "val_best_auc": "tham_chieu",
        "gradcam_expected_zone": "de_xuat" if gq["mean_expected_zone_de_xuat"] >= gq["mean_expected_zone_tham_chieu"] else "tham_chieu",
        "gradcam_low_corner": "de_xuat" if gq["mean_corner_de_xuat"] <= gq["mean_corner_tham_chieu"] else "tham_chieu",
        "gradcam_plausible_rate": "de_xuat" if gq["plausible_de_xuat"] >= gq["plausible_tham_chieu"] else "tham_chieu",
        "positive_sample_prob": "tham_chieu" if stats["prob_on_positive_samples"]["mean_prob_tham_chieu"] > stats["prob_on_positive_samples"]["mean_prob_de_xuat"] else "de_xuat",
    }
    for key, default_winner, weight, note in rubric:
        w = winners_map.get(key, default_winner)
        scores[w] += weight
        rubric_detail.append({"criterion": key, "winner": w, "weight": weight, "note": note})

    stats["rubric"] = rubric_detail
    stats["weighted_score"] = scores
    stats["recommendation"] = (
        "tham_chieu" if scores["tham_chieu"] > scores["de_xuat"]
        else ("de_xuat" if scores["de_xuat"] > scores["tham_chieu"] else "tie")
    )
    stats["recommendation_vi"] = {
        "tham_chieu": "Chon lam model phan loai chinh (do chinh xac tong the va nhan hiem tot hon).",
        "de_xuat": "Chon lam he thong trien khai neu uu tien Grad-CAM/giai thich truc quan hon do chinh xac.",
        "hybrid": "Phuong an tot nhat: checkpoint tham chieu + pipeline tien xu ly/Grad-CAM cua de xuat.",
    }

    json_path = out / "evaluation_summary.json"
    json_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("KET QUA TONG HOP (%d mau)" % n)
    print("=" * 70)
    print("Grad-CAM expected_zone:  de_xuat=%.3f  tham_chieu=%.3f  (thang %d/%d)" % (
        gq["mean_expected_zone_de_xuat"], gq["mean_expected_zone_tham_chieu"],
        gq["wins_expected_zone_de_xuat"], n))
    print("Grad-CAM corner (thap tot): de_xuat=%.3f  tham_chieu=%.3f  (thang %d/%d)" % (
        gq["mean_corner_de_xuat"], gq["mean_corner_tham_chieu"],
        gq["wins_corner_de_xuat"], n))
    print("CAM plausible: de_xuat=%d/%d  tham_chieu=%d/%d" % (
        gq["plausible_de_xuat"], n, gq["plausible_tham_chieu"], n))
    print("Prob tren mau duong: de_xuat=%.3f  tham_chieu=%.3f" % (
        stats["prob_on_positive_samples"]["mean_prob_de_xuat"],
        stats["prob_on_positive_samples"]["mean_prob_tham_chieu"]))
    print("Diem rubric: de_xuat=%d  tham_chieu=%d" % (scores["de_xuat"], scores["tham_chieu"]))
    print("KHUYEN NGHI:", stats["recommendation"])
    print("Anh so sanh:", out / "comparisons")
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
