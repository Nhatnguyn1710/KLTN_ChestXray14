import argparse
import os
import re
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.inference import (
    preprocess_image,
    predict,
    predict_from_path,
    get_per_class_thresholds,
    _get_temperature,
    _get_per_label_temperatures,
)


def _resolve_image_path(config: dict, image_path: str | None) -> str:
    if image_path:
        return image_path

    valid_csv = config["paths"]["valid_csv"]
    dataset_dir = config["paths"]["dataset_dir"]
    df = pd.read_csv(valid_csv)
    if "Frontal/Lateral" in df.columns:
        frontal = df[df["Frontal/Lateral"] == "Frontal"]
        if len(frontal) > 0:
            row = frontal.iloc[0]
        else:
            row = df.iloc[0]
    else:
        row = df.iloc[0]

    rel = str(row["Path"])
    parts = rel.split("/", 1)
    if len(parts) > 1:
        return os.path.join(dataset_dir, parts[1])
    return os.path.join(dataset_dir, rel)


def _resolve_checkpoint_path(config: dict) -> str:
    ckpt = config["paths"]["densenet_checkpoint"]
    if os.path.isfile(ckpt):
        return ckpt
    base_dir = config.get("paths", {}).get("checkpoint_base_dir", "")
    candidates = []
    if base_dir and os.path.isdir(base_dir):
        for name in os.listdir(base_dir):
            if re.fullmatch(r"v\d+", name):
                path = os.path.join(base_dir, name, "best_model.pth")
                if os.path.isfile(path):
                    candidates.append((int(name[1:]), path))
    if not candidates:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    candidates.sort(key=lambda x: x[0])
    fallback = candidates[-1][1]
    print(f"[WARN] Config checkpoint not found, fallback to latest: {fallback}")
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Smoke test parity: CLI wrapper vs web-like inference pipeline.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", default=None, help="Optional absolute/relative image path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt = _resolve_checkpoint_path(config)

    image_path = _resolve_image_path(config, args.image)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    config_for_run = dict(config)
    config_for_run["paths"] = dict(config.get("paths", {}))
    config_for_run["paths"]["densenet_checkpoint"] = ckpt

    device = get_device(config)
    print(f"Device: {device}")
    print(f"Image: {image_path}")

    # A) CLI wrapper path
    cli_result = predict_from_path(
        image_path=image_path,
        config=config_for_run,
        device=device,
        threshold=args.threshold,
        use_tta=False,
    )

    # B) Web-like path (same preprocess/predict components used by API)
    model = load_trained_model(ckpt, config_for_run, device)
    per_class_thresholds = get_per_class_thresholds(config_for_run)
    temperature = _get_temperature(config_for_run)
    per_label_temperatures = _get_per_label_temperatures(config_for_run)
    bottom_crop_ratio = float(config_for_run.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0))
    normalization_cfg = config_for_run.get("cnn", {}).get("normalization", {})
    clahe_cfg = config_for_run.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})

    image_tensor = preprocess_image(
        image_path,
        config_for_run["cnn"]["image_size"],
        bottom_crop_ratio=bottom_crop_ratio,
        normalization_cfg=normalization_cfg,
        clahe_cfg=clahe_cfg,
    )
    web_like_result = predict(
        model=model,
        image_tensor=image_tensor,
        device=device,
        threshold=args.threshold,
        per_class_thresholds=per_class_thresholds,
        temperature=temperature,
        per_label_temperatures=per_label_temperatures if per_label_temperatures else None,
    )

    labels = sorted(cli_result["probabilities"].keys())
    diffs = []
    mismatch_preds = []
    for label in labels:
        p_cli = float(cli_result["probabilities"][label])
        p_web = float(web_like_result["probabilities"][label])
        diffs.append(abs(p_cli - p_web))
        if bool(cli_result["predictions"][label]) != bool(web_like_result["predictions"][label]):
            mismatch_preds.append(label)

    max_diff = max(diffs) if diffs else 0.0
    print(f"max |prob_cli - prob_web| = {max_diff:.8f}")
    if mismatch_preds:
        print(f"prediction mismatches: {mismatch_preds}")

    assert max_diff <= args.tol, f"Probability parity failed: max diff {max_diff} > tol {args.tol}"
    assert not mismatch_preds, f"Prediction parity failed for labels: {mismatch_preds}"
    print("OK: CLI and web-like inference pipelines are in parity.")


if __name__ == "__main__":
    main()
