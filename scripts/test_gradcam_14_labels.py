import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.cnn.dataset import CHEXPERT_LABELS
from src.cnn.grad_cam import generate_gradcam_for_top_diseases
from src.cnn.inference import (
    _get_per_label_isotonic,
    _get_per_label_temperatures,
    _get_temperature,
    get_per_class_thresholds,
    predict,
    preprocess_image,
)
from src.cnn.model import load_trained_model
from src.utils import get_device, load_config


def _view_tensor(view_code: str, device: torch.device) -> torch.Tensor:
    code = str(view_code or "").strip().upper()
    if code == "PA":
        value = 1.0
    elif code == "AP":
        value = 0.0
    else:
        value = 0.5
    return torch.tensor([value], dtype=torch.float32, device=device)


def _resolve_image_path(image_root: Path, rel_path: str, image_name: str) -> Path:
    candidates = [
        image_root / str(rel_path).replace("/", os.sep),
        image_root / image_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _read_positive_rows(csv_path: Path, image_root: Path) -> dict:
    positives = {label: [] for label in CHEXPERT_LABELS}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_path = _resolve_image_path(
                image_root,
                row.get("Path", ""),
                row.get("Image Index", ""),
            )
            if not image_path.is_file():
                continue
            for label in CHEXPERT_LABELS:
                if str(row.get(label, "0")).strip() == "1":
                    positives[label].append({
                        "path": str(image_path),
                        "image": row.get("Image Index", ""),
                        "patient_id": row.get("Patient ID", ""),
                        "view": row.get("View Position", ""),
                    })
    return positives


def _sample_candidates(rows: list, max_candidates: int) -> list:
    if len(rows) <= max_candidates:
        return rows
    idx = np.linspace(0, len(rows) - 1, max_candidates).round().astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]


def _load_for_display(image_path: str, size: int = 320) -> Image.Image:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 247, 250))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _overlay_to_pil(overlay_rgb: np.ndarray, size: int = 320) -> Image.Image:
    image = Image.fromarray(overlay_rgb.astype(np.uint8), mode="RGB")
    image.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (245, 247, 250))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def _quality_label(quality: dict) -> str:
    cls = str(quality.get("classification", "")).strip()
    if cls == "plausible":
        return "Dat"
    if cls == "model_related":
        return "Can xem lai"
    if cls == "pipeline_related":
        return "Nguy co lech"
    if cls == "no_activation":
        return "Khong kich hoat"
    return cls or "Chua danh gia"


def _make_contact_sheet(records: list, output_path: Path) -> None:
    tile = 280
    label_h = 82
    cols = 4
    rows = int(np.ceil(len(records) / cols))
    sheet_w = cols * tile
    sheet_h = rows * (tile + label_h) + 64
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 15)
    except Exception:
        font_title = ImageFont.load_default()
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    title = "Grad-CAM cho 14 nhan NIH ChestX-ray14 (model FZLPR)"
    draw.text((20, 18), title, fill=(20, 28, 45), font=font_title)

    for idx, record in enumerate(records):
        row = idx // cols
        col = idx % cols
        x = col * tile
        y = 64 + row * (tile + label_h)
        sheet.paste(record["thumb"], (x, y))
        draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(210, 216, 226), width=1)
        text_y = y + tile + 8
        prob = float(record["probability"])
        draw.text((x + 10, text_y), record["label"], fill=(17, 24, 39), font=font)
        draw.text(
            (x + 10, text_y + 24),
            f"p={prob:.3f} | {_quality_label(record['quality'])}",
            fill=(55, 65, 81),
            font=font_small,
        )
        primary = record["quality"].get("primary_zone", "")
        ezr = record["quality"].get("final", {}).get("expected_zone_ratio", "")
        draw.text(
            (x + 10, text_y + 46),
            f"zone={primary} | EZR={ezr}",
            fill=(75, 85, 99),
            font=font_small,
        )

    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--output_dir", default="outputs/thesis/gradcam_14_labels")
    parser.add_argument("--max_candidates", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    model = load_trained_model(config["paths"]["densenet_checkpoint"], config, device)
    model.eval()

    image_root = Path(config["paths"].get("images_dir", config["paths"]["dataset_dir"]))
    split_csv = Path(args.split_csv or config["paths"]["test_csv"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positives = _read_positive_rows(split_csv, image_root)
    thresholds = get_per_class_thresholds(config)
    temperature = _get_temperature(config)
    per_label_temperatures = _get_per_label_temperatures(config)
    per_label_isotonic = _get_per_label_isotonic(config)
    normalization_cfg = config.get("cnn", {}).get("normalization", {})
    clahe_cfg = config.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})
    bottom_crop_ratio = float(
        config.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0)
    )

    records = []
    for label in CHEXPERT_LABELS:
        rows = positives[label]
        if not rows:
            print(f"[WARN] {label}: no positive samples in {split_csv}")
            continue

        best = None
        for row in _sample_candidates(rows, args.max_candidates):
            view_type = _view_tensor(row["view"], device)
            image_tensor = preprocess_image(
                row["path"],
                config["cnn"]["image_size"],
                bottom_crop_ratio=bottom_crop_ratio,
                normalization_cfg=normalization_cfg,
                clahe_cfg=clahe_cfg,
            )
            with torch.no_grad():
                pred = predict(
                    model,
                    image_tensor,
                    device,
                    per_class_thresholds=thresholds,
                    temperature=temperature,
                    view_type=view_type,
                    per_label_temperatures=per_label_temperatures or None,
                    per_label_isotonic=per_label_isotonic or None,
                )
            prob = float(pred["probabilities"].get(label, 0.0))
            if best is None or prob > best["probability"]:
                best = {**row, "probability": prob, "pred": pred}

        view_type = _view_tensor(best["view"], device)
        gc_results = generate_gradcam_for_top_diseases(
            model,
            best["path"],
            best["pred"]["probabilities"],
            config,
            top_k=1,
            target_labels=[label],
            view_type=view_type,
        )
        if not gc_results:
            print(f"[WARN] {label}: Grad-CAM returned no result")
            continue
        item = gc_results[0]

        overlay_path = output_dir / f"{label}_gradcam.png"
        original_path = output_dir / f"{label}_original.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(item["heatmap"], cv2.COLOR_RGB2BGR))
        Image.open(best["path"]).convert("RGB").save(original_path)

        record = {
            "label": label,
            "probability": best["probability"],
            "image": best["image"],
            "patient_id": best["patient_id"],
            "view": best["view"],
            "image_path": best["path"],
            "overlay_path": str(overlay_path),
            "quality": item.get("quality", {}),
            "thumb": _overlay_to_pil(item["heatmap"], size=280),
        }
        records.append(record)

        q = record["quality"]
        final = q.get("final", {}) if isinstance(q, dict) else {}
        print(
            f"[OK] {label}: p={best['probability']:.4f}, "
            f"image={best['image']}, quality={q.get('classification')}, "
            f"EZR={final.get('expected_zone_ratio')}"
        )

    serializable = []
    for r in records:
        item = {k: v for k, v in r.items() if k != "thumb"}
        serializable.append(item)

    with (output_dir / "gradcam_14_labels_summary.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    with (output_dir / "gradcam_14_labels_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label", "probability", "image", "patient_id", "view",
            "quality", "expected_zone_ratio", "outside_thorax_ratio",
            "primary_zone", "overlay_path",
        ])
        for r in records:
            q = r.get("quality", {})
            final = q.get("final", {}) if isinstance(q, dict) else {}
            writer.writerow([
                r["label"],
                f"{float(r['probability']):.6f}",
                r["image"],
                r["patient_id"],
                r["view"],
                q.get("classification", ""),
                final.get("expected_zone_ratio", ""),
                final.get("outside_thorax_ratio", ""),
                q.get("primary_zone", ""),
                r["overlay_path"],
            ])

    _make_contact_sheet(records, output_dir / "gradcam_14_labels_grid.png")
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
