import argparse
import os
import sys
from itertools import islice

import torch
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils import load_config, get_device
from src.cnn.dataset import build_dataloaders
from src.cnn.model import build_model
from src.cnn.train import (
    build_criterion,
    train_one_epoch,
    validate,
)


def main():
    parser = argparse.ArgumentParser(description="Smoke regression: one train step + one validation step.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--train-batches", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=1)
    parser.add_argument("--use-config-accum", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(config)
    cnn_cfg = config["cnn"]
    train_cfg = cnn_cfg["training"]
    # Sandbox-friendly: avoid multiprocessing workers in smoke test.
    train_cfg["num_workers"] = 0

    print(f"Device: {device}")
    print("Building model + dataloaders...")
    model = build_model(config).to(device)
    train_loader, val_internal_loader, _test_loader, _valid_loader = build_dataloaders(config)

    criterion = build_criterion(train_cfg, train_loader.dataset, device)

    lr_head = float(train_cfg["learning_rate"])
    lr_backbone = lr_head * float(train_cfg.get("backbone_lr_ratio", 0.02))
    optimizer = torch.optim.AdamW(
        model.get_param_groups(lr_backbone=lr_backbone, lr_head=lr_head),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        eps=float(train_cfg.get("optimizer_eps", 1e-8)),
    )
    scaler = GradScaler("cuda", enabled=False)

    train_batches = list(islice(iter(train_loader), max(1, args.train_batches)))
    val_batches = list(islice(iter(val_internal_loader), max(1, args.val_batches)))
    accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1)) if args.use_config_accum else 1

    print(f"Running train smoke: batches={len(train_batches)}, accum_steps={accum_steps}")
    train_loss = train_one_epoch(
        model=model,
        loader=train_batches,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        use_fp16=False,
        accum_steps=accum_steps,
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        mixup_alpha=0.0,
        ema=None,
        finite_check_steps=int(train_cfg.get("finite_check_steps", 1)),
    )

    print(f"Running validation smoke: batches={len(val_batches)}")
    val_results = validate(
        model=model,
        loader=val_batches,
        criterion=criterion,
        device=device,
        use_fp16=False,
    )

    print(f"OK: train_loss={train_loss:.6f}")
    print(
        "OK: val_loss={:.6f}, val_auc_mean={:.6f}, val_auprc_mean={:.6f}".format(
            val_results["loss"],
            val_results["auc_mean"],
            val_results["auprc_mean"],
        )
    )


if __name__ == "__main__":
    main()
