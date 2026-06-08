

import os
import sys
import argparse
import csv
import shutil
import torch
import numpy as np
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import load_config, get_device
from src.cnn.model import load_trained_model
from src.cnn.dataset import CheXpertDataset, CHEXPERT_LABELS
import pandas as pd


TARGET_LABELS = ["Consolidation", "Pneumothorax", "Cardiomegaly"]
TOP_K = 100


def run_audit(config: dict, checkpoint_path: str, output_dir: str, top_k: int = TOP_K):
    device = get_device(config)
    cnn_cfg = config["cnn"]
    paths = config["paths"]

    # Load model
    model = load_trained_model(checkpoint_path, config, device)
    model.eval()

    # Load val_internal split (no augmentation)
    val_csv = os.path.join(os.path.dirname(paths["train_csv"]), "splits", "val_internal_split.csv")
    if not os.path.isfile(val_csv):
        print(f"ERROR: {val_csv} not found. Run training first to generate splits.")
        return

    val_df = pd.read_csv(val_csv)
    aug_cfg = cnn_cfg.get("augmentation", {})
    dataset = CheXpertDataset(
        csv_path=None,
        image_root=paths["dataset_dir"],
        image_size=cnn_cfg["image_size"],
        augmentation=False,
        uncertainty_policy=cnn_cfg["uncertainty_policy"],
        dataframe=val_df,
        aug_cfg=aug_cfg,
        use_view_position=cnn_cfg.get("use_view_position", False),
        bottom_crop_ratio=aug_cfg.get("bottom_crop_ratio", 0.0),
        uncertainty_policy_per_class=cnn_cfg.get("uncertainty_policy_per_class", None),
        normalization_cfg=cnn_cfg.get("normalization", {}),
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Inference
    all_preds = []
    all_labels = []
    all_masks = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            images = batch["image"].to(device)
            view_type = batch["view_type"].to(device) if "view_type" in batch else None
            with autocast("cuda", enabled=cnn_cfg["training"].get("use_fp16", True)):
                preds = torch.sigmoid(model(images, view_type=view_type)).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())
            all_masks.append(batch["mask"].numpy())
            all_paths.extend(batch["path"])

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Look up AP/PA and Support Devices from dataframe
    # Dataset filters to frontal only and resets index, so df aligns with loader order
    df = dataset.df

    for label_name in TARGET_LABELS:
        li = CHEXPERT_LABELS.index(label_name)
        print(f"\n--- {label_name} ---")

        # Only mask=1 samples (observed ground truth)
        observed = all_masks[:, li] > 0.5

        preds_l = all_preds[observed, li]
        labels_l = all_labels[observed, li]
        indices_l = np.where(observed)[0]

        # False Positives: y=0, pred high
        fp_mask = labels_l < 0.5
        fp_scores = preds_l[fp_mask]
        fp_indices = indices_l[fp_mask]
        fp_order = np.argsort(-fp_scores)[:top_k]

        # False Negatives: y=1, pred low
        fn_mask = labels_l > 0.5
        fn_scores = preds_l[fn_mask]
        fn_indices = indices_l[fn_mask]
        fn_order = np.argsort(fn_scores)[:top_k]

        for error_type, order, scores, base_indices in [
            ("fp", fp_order, fp_scores, fp_indices),
            ("fn", fn_order, fn_scores, fn_indices),
        ]:
            out_dir = os.path.join(output_dir, label_name, error_type)
            os.makedirs(out_dir, exist_ok=True)

            csv_path = os.path.join(out_dir, "errors.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["rank", "image_path", "ground_truth", "prediction",
                                 "view_type", "has_support_devices"])

                for rank, idx_in_subset in enumerate(order, 1):
                    global_idx = int(base_indices[idx_in_subset])
                    pred_val = float(scores[idx_in_subset])
                    gt_val = int(all_labels[global_idx, li])
                    img_path = all_paths[global_idx]

                    # AP/PA from dataframe
                    row = df.iloc[global_idx]
                    view = row.get("AP/PA", "unknown")
                    has_sd = int(row.get("Support Devices", 0) == 1) if "Support Devices" in df.columns else -1

                    writer.writerow([rank, img_path, gt_val, f"{pred_val:.4f}", view, has_sd])

                    # Copy image
                    parts = img_path.split("/", 1)
                    src_path = os.path.join(paths["dataset_dir"], parts[1] if len(parts) > 1 else img_path)
                    if os.path.isfile(src_path):
                        dst = os.path.join(out_dir, f"{rank:03d}_{os.path.basename(src_path)}")
                        shutil.copy2(src_path, dst)

            print(f"  {error_type.upper()}: {len(order)} samples → {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit top FP/FN for weak labels")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (e.g. models/densenet121/v11/best_model.pth)")
    parser.add_argument("--output_dir", type=str, default="outputs/error_audit")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    args = parser.parse_args()

    config = load_config(args.config)
    run_audit(config, args.checkpoint, args.output_dir, args.top_k)
