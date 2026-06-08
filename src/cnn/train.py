
import os
import sys
import argparse
import time
import csv
import json
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils import load_config, set_seed, get_device, ensure_dir, print_gpu_info
from src.cnn.model import build_model
from src.cnn.dataset import build_dataloaders, CHEXPERT_LABELS


# NIH 14 nhãn — khớp với config.yaml labels.
# Core = phổ biến + lâm sàng quan trọng; Rare = tần suất thấp (<5%).
DEFAULT_CORE_LABELS = [
    "Effusion",
    "Cardiomegaly",
    "Edema",
    "Infiltration",
    "Atelectasis",
]
DEFAULT_RARE_LABELS = [
    "Consolidation",
    "Pneumothorax",
    "Hernia",
    "Emphysema",
    "Fibrosis",
]


# ============================================================
# Training Logger — ghi metrics ra CSV
# ============================================================
class TrainingLogger:
    """Ghi training metrics vào CSV file để theo dõi."""

    def __init__(self, log_path: str, resume: bool = False):
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path))
        mode = "a" if resume else "w"
        self.file = open(log_path, mode, newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        if not resume:
            self.writer.writerow([
                "epoch", "train_loss", "val_loss", "val_auc_mean", "val_auprc_mean",
                "core_auprc_mean", "rare_auprc_mean", "selection_score",
                "learning_rate", "time_seconds",
                *[f"auc_{l}" for l in CHEXPERT_LABELS],
                *[f"auprc_{l}" for l in CHEXPERT_LABELS],
            ])
            self.file.flush()

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        val_auc_mean,
        val_auprc_mean,
        core_auprc_mean,
        rare_auprc_mean,
        selection_score,
        lr,
        elapsed,
        auc_per_class,
        auprc_per_class,
    ):
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_auc_mean:.6f}",
            f"{val_auprc_mean:.6f}",
            f"{core_auprc_mean:.6f}",
            f"{rare_auprc_mean:.6f}",
            f"{selection_score:.6f}",
            f"{lr:.8f}", f"{elapsed:.1f}",
        ]
        for label in CHEXPERT_LABELS:
            auc = auc_per_class.get(label)
            row.append(f"{auc:.6f}" if auc is not None else "N/A")
        for label in CHEXPERT_LABELS:
            auprc = auprc_per_class.get(label)
            row.append(f"{auprc:.6f}" if auprc is not None else "N/A")
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


# ============================================================
# Top-K Checkpoint Tracker (v9) — auto-ensemble support
# ============================================================
class TopKCheckpointTracker:
    """
    Track & keep top-K model checkpoints ranked by a selection score.
    Inspired by kamenbliznashki/chexpert: auto-save top checkpoints for ensemble.
    
    - Saves epoch_{N}.pth for each qualifying checkpoint
    - Maintains a JSON manifest with metrics
    - Deletes old checkpoints when exceeding K to save disk space
    """

    def __init__(self, checkpoint_dir: str, k: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.k = k
        self.manifest_path = os.path.join(checkpoint_dir, "topk_manifest.json")
        self.records = []  # list of {"epoch": int, "score": float, "auc": float, "path": str}
        ensure_dir(checkpoint_dir)

        # Load existing manifest if resuming
        if os.path.isfile(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
            for record in self.records:
                if "score" not in record and "auprc" in record:
                    record["score"] = record["auprc"]

    def update(self, epoch: int, score: float, auc: float, model: nn.Module) -> bool:
        """
        Check if this epoch's metrics qualify for top-K. If yes, save checkpoint.
        Returns True if checkpoint was saved.
        """
        # Check if this qualifies
        if len(self.records) >= self.k:
            worst = min(self.records, key=lambda r: r.get("score", r.get("auprc", 0.0)))
            if score <= worst.get("score", worst.get("auprc", 0.0)):
                return False
            # Remove worst checkpoint
            if os.path.isfile(worst["path"]):
                os.remove(worst["path"])
            self.records.remove(worst)

        # Save new checkpoint
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}.pth")
        save_model_weights(ckpt_path, model, context=f"top-k epoch {epoch}")
        self.records.append({
            "epoch": epoch,
            "score": round(score, 6),
            "auc": round(auc, 6),
            "path": ckpt_path,
        })
        self.records.sort(key=lambda r: r.get("score", r.get("auprc", 0.0)), reverse=True)

        # Save manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

        return True

    def get_paths(self) -> list:
        """Return list of top-K checkpoint paths, sorted best-first."""
        return [r["path"] for r in self.records if os.path.isfile(r["path"])]

    def summary(self) -> str:
        """Human-readable summary of tracked checkpoints."""
        lines = [f"Top-{self.k} checkpoints:"]
        for i, r in enumerate(self.records):
            score = r.get("score", r.get("auprc", 0.0))
            lines.append(f"  #{i+1}: epoch {r['epoch']} - score={score:.4f}, AUC={r['auc']:.4f}")
        return "\n".join(lines)


def _iter_named_tensors(obj, prefix: str = ""):
    if torch.is_tensor(obj):
        yield prefix or "<tensor>", obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_named_tensors(value, name)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            name = f"{prefix}.{idx}" if prefix else str(idx)
            yield from _iter_named_tensors(value, name)


def _nonfinite_summary(obj, max_items: int = 8) -> tuple:
    bad = []
    bad_values = 0
    checked_values = 0
    for name, tensor in _iter_named_tensors(obj):
        if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
            continue
        checked_values += tensor.numel()
        finite = torch.isfinite(tensor)
        if not finite.all().item():
            count = int((~finite).sum().item())
            bad_values += count
            if len(bad) < max_items:
                bad.append(f"{name} ({count}/{tensor.numel()})")
    return bad, bad_values, checked_values


def assert_finite_state(obj, context: str):
    bad, bad_values, checked_values = _nonfinite_summary(obj)
    if bad_values:
        examples = "; ".join(bad)
        raise FloatingPointError(
            f"Non-finite values in {context}: {bad_values}/{checked_values}. "
            f"Examples: {examples}"
        )


def assert_model_finite(model: nn.Module, context: str):
    assert_finite_state(model.state_dict(), context)


def assert_finite_metrics(metrics: dict, context: str):
    bad = []
    for name, value in metrics.items():
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            bad.append(f"{name}={value!r}")
            continue
        if not np.isfinite(value_f):
            bad.append(f"{name}={value_f}")
    if bad:
        raise FloatingPointError(f"Non-finite metrics in {context}: {', '.join(bad)}")


def _atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def save_model_weights(path: str, model: nn.Module, context: str):
    state = model.state_dict()
    assert_finite_state(state, f"{context} model weights before save")
    _atomic_torch_save(state, path)


def _mean_available(metric_per_class: dict, labels: list) -> float:
    values = []
    for label in labels or []:
        metric = metric_per_class.get(label)
        if metric is None:
            continue
        try:
            metric_f = float(metric)
        except (TypeError, ValueError):
            continue
        if np.isfinite(metric_f):
            values.append(metric_f)
    return float(np.mean(values)) if values else 0.0


def _enrich_selection_metrics(results: dict, selection_cfg: dict = None) -> dict:
    cfg = selection_cfg if isinstance(selection_cfg, dict) else {}
    core_labels = cfg.get("core_labels", DEFAULT_CORE_LABELS)
    rare_labels = cfg.get("rare_labels", DEFAULT_RARE_LABELS)
    core_weight = float(cfg.get("core_weight", 0.75))
    rare_weight = float(cfg.get("rare_weight", 0.25))
    strategy = str(cfg.get("strategy", "weighted_group_auprc")).strip().lower()

    core_auprc = _mean_available(results.get("auprc_per_class", {}), core_labels)
    rare_auprc = _mean_available(results.get("auprc_per_class", {}), rare_labels)
    results["core_auprc_mean"] = core_auprc
    results["rare_auprc_mean"] = rare_auprc

    if strategy in {"mean_auroc", "mean_auc", "auroc", "auc"}:
        selection_score = float(results.get("auc_mean", 0.0))
    elif strategy in {"mean_auprc", "auprc", "average_precision"}:
        selection_score = float(results.get("auprc_mean", 0.0))
    elif strategy == "label_weighted_auprc":
        # v12: Label-level weighted AUPRC — mỗi label có weight riêng
        label_weights = cfg.get("label_weights", {})
        auprc_per_class = results.get("auprc_per_class", {})
        weighted_sum = 0.0
        total_w = 0.0
        for label, w in label_weights.items():
            w = float(w)
            auprc_val = auprc_per_class.get(label)
            if auprc_val is None or w <= 0:
                continue
            try:
                auprc_f = float(auprc_val)
            except (TypeError, ValueError):
                continue
            if np.isfinite(auprc_f):
                weighted_sum += w * auprc_f
                total_w += w
        selection_score = weighted_sum / max(total_w, 1e-8)
    elif strategy == "weighted_group_auprc":
        total_weight = max(core_weight + rare_weight, 1e-8)
        selection_score = (core_weight * core_auprc + rare_weight * rare_auprc) / total_weight
    else:
        selection_score = float(results.get("auprc_mean", 0.0))
    results["selection_score"] = float(selection_score)
    return results


# ============================================================
# v12: Hard-Negative Mining — mine FP từ train set
# ============================================================
@torch.no_grad()
def mine_hard_negatives(
    model: nn.Module,
    dataset,
    device: torch.device,
    hard_neg_cfg: dict,
    use_fp16: bool = True,
) -> tuple:
    """
    Forward-only trên train set (no grad, no augmentation).
    Thu thập indices: mask=1, y=0, pred > fp_threshold cho target labels.
    Dedup theo image path để tránh 1 ảnh xuất hiện quá nhiều.

    Returns:
        (indices, manifest): list of dataset indices + dict manifest for persistence.
    """
    target_labels = hard_neg_cfg.get("target_labels", ["Consolidation", "Pneumothorax", "Cardiomegaly"])
    pool_size = int(hard_neg_cfg.get("pool_size_per_label", 500))
    fp_threshold = float(hard_neg_cfg.get("fp_threshold", 0.5))

    # Tạm tắt augmentation cho mining (try/finally để đảm bảo restore)
    old_aug = dataset._augmentation
    old_transform = dataset.transform
    dataset._augmentation = False
    dataset.transform = dataset._build_transforms(False, aug_cfg=dataset._aug_cfg)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    model.eval()

    # Collect per-label FP candidates: {label_idx: [(score, global_idx, path), ...]}
    label_indices = {label: CHEXPERT_LABELS.index(label) for label in target_labels if label in CHEXPERT_LABELS}
    fp_candidates = {li: [] for li in label_indices.values()}

    try:
        global_idx = 0
        for batch in tqdm(loader, desc="[Hard-neg mining]", leave=False):
            images = batch["image"].to(device)
            labels = batch["labels"].numpy()
            masks = batch["mask"].numpy()
            paths = batch["path"]
            view_type = batch["view_type"].to(device) if "view_type" in batch else None
            bs = images.shape[0]

            with autocast('cuda', enabled=use_fp16):
                preds = torch.sigmoid(model(images, view_type=view_type)).cpu().numpy()

            for b in range(bs):
                idx = global_idx + b
                for li in label_indices.values():
                    # Chỉ mine observed negatives: mask=1, y=0, pred > threshold
                    if masks[b, li] > 0.5 and labels[b, li] < 0.5 and preds[b, li] > fp_threshold:
                        fp_candidates[li].append((float(preds[b, li]), idx, paths[b]))
            global_idx += bs
    finally:
        # Luôn restore augmentation dù mining thành công hay crash
        dataset._augmentation = old_aug
        dataset.transform = old_transform

    # Select top-K hardest per label, dedup by index
    selected = set()
    manifest = {"fp_threshold": fp_threshold, "pool_size_per_label": pool_size, "labels": {}}
    for label_name, li in label_indices.items():
        candidates = sorted(fp_candidates[li], key=lambda x: -x[0])[:pool_size]
        n_before = len(selected)
        label_entries = []
        for score, idx, path in candidates:
            selected.add(idx)
            label_entries.append({"path": path, "score": round(score, 4)})
        manifest["labels"][label_name] = {"count": len(candidates), "samples": label_entries}
        print(f"  [Hard-neg] {label_name}: {len(candidates)} FPs mined, {len(selected) - n_before} new indices")

    manifest["total_unique"] = len(selected)
    print(f"  [Hard-neg] Total unique indices: {len(selected)}")
    return list(selected), manifest


# ============================================================
# Compute Class Weights — xử lý mất cân bằng dữ liệu
# ============================================================
def compute_class_weights(dataset, device: torch.device, class_weight_max: float = 10.0) -> torch.Tensor:
    """
    Tính weight cho mỗi class dựa trên tần suất positive.
    Class hiếm (VD: Pneumothorax ~2%) → weight cao hơn.

    Args:
        dataset: CheXpertDataset
        device: torch.device
        class_weight_max: Max weight (đọc từ config, v3 mặc định 10.0)
    Returns:
        Tensor [num_classes] class weights, clamp trong [1, class_weight_max]
    """
    labels_df = dataset.df[CHEXPERT_LABELS].values  # [N, num_classes]
    # V4: Chỉ đếm label == 1 và label == 0, bỏ qua -1 (uncertain)
    pos_count = (labels_df == 1).sum(axis=0).astype(float)  # [num_classes]
    neg_count = (labels_df == 0).sum(axis=0).astype(float)  # [num_classes]
    total_valid = pos_count + neg_count  # Chỉ đếm valid labels

    # Weight = neg / pos (inverse frequency), chỉ dựa trên valid labels
    weights = neg_count / (pos_count + 1e-5)

    # v3: Clamp [1, max] — đọc max từ config, mặc định 10
    max_weight = class_weight_max
    weights = np.clip(weights, 1.0, max_weight)

    print(f"  Class weights (neg/pos ratio, clipped [1, {max_weight}]):")
    for name, w, pc, tv in zip(CHEXPERT_LABELS, weights, pos_count, total_valid):
        print(f"    {name}: weight={w:.2f}  (pos={int(pc):,}/{int(tv):,} valid, {pc/tv*100:.1f}%)")

    return torch.tensor(weights, dtype=torch.float32).to(device)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification — logits version (v9).
    
    Accepts raw logits (before sigmoid) for numerical stability.
    Uses F.logsigmoid / log(1-sigmoid) via log-sum-exp trick to avoid log(0).
    """
   
    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip_margin: float = 0.05,
        label_smoothing: float = 0.0,
        use_batch_weight: bool = False,
        batch_weight_max: float = 20.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        # Hỗ trợ per-class gamma (Tensor [C]) hoặc scalar
        if isinstance(gamma_pos, torch.Tensor):
            self.register_buffer("gamma_pos", gamma_pos)
        else:
            self.gamma_pos = gamma_pos
        if isinstance(gamma_neg, torch.Tensor):
            self.register_buffer("gamma_neg", gamma_neg)
        else:
            self.gamma_neg = gamma_neg
        self.clip_margin = clip_margin
        self.label_smoothing = label_smoothing
        self.use_batch_weight = use_batch_weight
        self.batch_weight_max = batch_weight_max

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, C] raw logits (before sigmoid) — v9 change
            targets: [B, C] binary labels (0 or 1)
            mask: [B, C] binary mask (1 = compute loss, 0 = ignore)
        """
        logits = logits.float()
        targets = targets.float()

        # Label smoothing: push targets away from 0/1 to reduce overconfidence
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Convert logits to probabilities for weighting (detached for stability)
        p = torch.sigmoid(logits)

        # --- Positive loss: -log(sigmoid(x)) = -F.logsigmoid(x) ---
        log_p = F.logsigmoid(logits)  # numerically stable log(sigmoid(x))
        pos_loss = targets * log_p  # [B, C]
        gp = self.gamma_pos
        if isinstance(gp, torch.Tensor):
            gp = gp.unsqueeze(0)
        if torch.is_tensor(gp):
            pos_weight = (1 - p.detach()) ** gp
            pos_loss = pos_weight * pos_loss
        elif gp > 0:
            pos_weight = (1 - p.detach()) ** gp
            pos_loss = pos_weight * pos_loss

        # --- Negative loss with asymmetric clipping ---
        # log(1 - sigmoid(x)) = -x + log(sigmoid(x)) = F.logsigmoid(-x)
        if self.clip_margin > 0:
            # Asymmetric clipping: shift logits up to reduce easy negative contribution
            # Equivalent to clamping p_neg = (p - margin).clamp(min=eps)
            # In logit space: shift logits down by inverse-sigmoid(margin) approximation
            neg_p = (p - self.clip_margin).clamp(min=1e-6)
            log_1_minus_neg_p = torch.log(1 - neg_p + 1e-6)
        else:
            neg_p = p
            log_1_minus_neg_p = F.logsigmoid(-logits)

        neg_loss = (1 - targets) * log_1_minus_neg_p  # [B, C]
        gn = self.gamma_neg
        if isinstance(gn, torch.Tensor):
            gn = gn.unsqueeze(0)
        if torch.is_tensor(gn):
            pt_neg = 1 - neg_p.detach()
            neg_weight = (1 - pt_neg) ** gn
            neg_loss = neg_weight * neg_loss
        elif gn > 0:
            pt_neg = 1 - neg_p.detach()
            neg_weight = (1 - pt_neg) ** gn
            neg_loss = neg_weight * neg_loss

        # Combine: -(pos + neg) * weight
        # V7: Dynamic batch weight (theo CheXpert Top1 solution)
        if self.use_batch_weight:
            with torch.no_grad():
                effective_mask = mask if mask is not None else torch.ones_like(targets)
                pos_count = (targets * effective_mask).sum(dim=0)  # [C]
                neg_count = ((1 - targets) * effective_mask).sum(dim=0)  # [C]
                batch_weights = neg_count / (pos_count + 1e-5)
                batch_weights = batch_weights.clamp(1.0, self.batch_weight_max)
                # Fallback: class không có positive trong batch → dùng static weight
                no_pos = (pos_count < 0.5)
                batch_weights[no_pos] = self.class_weights[no_pos]
            loss = -(pos_loss + neg_loss) * batch_weights.unsqueeze(0)  # [B, C]
        else:
            loss = -(pos_loss + neg_loss) * self.class_weights.unsqueeze(0)  # [B, C]

        # Apply uncertainty mask (ignore labels with -1)
        if mask is not None:
            loss = loss * mask.float()
            num_valid = mask.float().sum()
            return loss.sum() / num_valid.clamp(min=1.0)

        return loss.mean()


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with optional per-class positive weights and label mask."""

    def __init__(
        self,
        pos_weight: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.label_smoothing = float(label_smoothing or 0.0)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal BCE baseline for multi-label classification."""

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing or 0.0)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt).clamp(min=1e-6) ** self.gamma) * bce
        if mask is not None:
            mask = mask.float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        return loss.mean()


class FZLPRLoss(nn.Module):
    """
    Focal ZLPR loss for multi-label classification.

    Implements: log(1 + sum_pos exp(-s_i/tau)) +
                log(1 + sum_neg exp( s_j/tau))
    with label masks for ignored targets.
    """

    def __init__(self, tau: float = 0.2):
        super().__init__()
        self.tau = float(tau)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        logits = logits.float()
        targets = targets.float()
        if mask is None:
            valid = torch.ones_like(targets, dtype=torch.bool)
        else:
            valid = mask.bool()

        pos_mask = (targets > 0.5) & valid
        neg_mask = (targets <= 0.5) & valid
        neg_inf = torch.full_like(logits, -torch.inf)
        zero = torch.zeros((logits.size(0), 1), device=logits.device, dtype=logits.dtype)

        pos_terms = torch.where(pos_mask, -logits / self.tau, neg_inf)
        neg_terms = torch.where(neg_mask, logits / self.tau, neg_inf)

        pos_loss = torch.logsumexp(torch.cat([zero, pos_terms], dim=1), dim=1)
        neg_loss = torch.logsumexp(torch.cat([zero, neg_terms], dim=1), dim=1)
        sample_loss = pos_loss + neg_loss

        valid_samples = valid.any(dim=1).float()
        return (sample_loss * valid_samples).sum() / valid_samples.sum().clamp(min=1.0)


def build_criterion(train_cfg: dict, train_dataset, device: torch.device) -> nn.Module:
    """Build loss from cnn.training.loss."""
    loss_name = str(train_cfg.get("loss", "asl")).strip().lower()
    class_weight_max = train_cfg.get("class_weight_max", 10.0)
    label_smoothing = train_cfg.get("label_smoothing", 0.0)

    if loss_name in {"bce", "plain_bce"}:
        print("  Loss: BCEWithLogitsLoss (plain, masked)")
        print(f"  BCE label_smoothing={label_smoothing}")
        return MaskedBCEWithLogitsLoss(label_smoothing=label_smoothing)

    if loss_name in {"weighted_bce", "wbce"}:
        pos_weight = compute_class_weights(
            train_dataset,
            device,
            class_weight_max=class_weight_max,
        )
        multiply = float(train_cfg.get("class_weight_multiply", 1.0))
        pos_weight = pos_weight * multiply
        print("  Loss: Weighted BCEWithLogitsLoss (masked)")
        print(f"  pos_weight multiplier={multiply}, label_smoothing={label_smoothing}")
        return MaskedBCEWithLogitsLoss(
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )

    if loss_name == "focal":
        pos_weight = None
        if train_cfg.get("focal_use_pos_weight", False):
            pos_weight = compute_class_weights(
                train_dataset,
                device,
                class_weight_max=class_weight_max,
            )
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        print(f"  Loss: Focal BCEWithLogitsLoss (gamma={gamma}, masked)")
        return FocalBCEWithLogitsLoss(
            gamma=gamma,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )

    if loss_name in {"zlpr", "fzlpr"}:
        tau = 1.0 if loss_name == "zlpr" else float(train_cfg.get("fzlpr_tau", 0.2))
        print(f"  Loss: {loss_name.upper()} (tau={tau})")
        return FZLPRLoss(tau=tau)

    if loss_name == "asl":
        class_weights = compute_class_weights(
            train_dataset,
            device,
            class_weight_max=class_weight_max,
        )

        use_weighted_sampler = train_cfg.get("weighted_sampler", False)
        if use_weighted_sampler:
            print("  [WARN] weighted_sampler=true + class_weights in ASL = double rebalancing!")
            print("         This amplifies shortcut incentive. Consider disabling one.")

        gamma_pos = train_cfg.get("asl_gamma_pos", 0.0)
        gamma_neg = train_cfg.get("asl_gamma_neg", 4.0)
        clip_margin = train_cfg.get("asl_clip_margin", 0.05)
        num_classes = len(CHEXPERT_LABELS)
        gamma_neg_cfg = train_cfg.get("asl_gamma_neg_per_class", {})
        gamma_pos_per_class = torch.full((num_classes,), gamma_pos, dtype=torch.float32).to(device)
        gamma_neg_per_class = torch.full((num_classes,), gamma_neg, dtype=torch.float32).to(device)

        if gamma_neg_cfg:
            for i, label_name in enumerate(CHEXPERT_LABELS):
                if label_name in gamma_neg_cfg:
                    gamma_neg_per_class[i] = float(gamma_neg_cfg[label_name])
            print("  ASL per-class gamma_neg:")
            for i, label_name in enumerate(CHEXPERT_LABELS):
                print(f"    {label_name}: gamma_neg={gamma_neg_per_class[i]:.1f}")

        print(f"  Loss: ASL gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, clip_margin={clip_margin}")
        print(f"  ASL label_smoothing={label_smoothing}")
        print(f"  Batch weight: {train_cfg.get('use_batch_weight', False)} (max={train_cfg.get('batch_weight_max', class_weight_max)})")
        return AsymmetricLoss(
            class_weights,
            gamma_pos=gamma_pos_per_class,
            gamma_neg=gamma_neg_per_class,
            clip_margin=clip_margin,
            label_smoothing=label_smoothing,
            use_batch_weight=train_cfg.get("use_batch_weight", False),
            batch_weight_max=train_cfg.get("batch_weight_max", class_weight_max),
        )

    raise ValueError(
        f"Unsupported loss '{loss_name}'. Use one of: bce, weighted_bce, focal, asl, zlpr, fzlpr."
    )


# ============================================================
# Checkpoint Save / Load — hỗ trợ resume training
# ============================================================
def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    best_auc: float,
    best_metric: float,
):
    """Lưu full training state để resume nếu crash."""
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict() if scheduler else None
    scaler_state = scaler.state_dict()

    assert_finite_state(model_state, f"latest checkpoint epoch {epoch} model")
    assert_finite_state(optimizer_state, f"latest checkpoint epoch {epoch} optimizer")
    assert_finite_state(scheduler_state, f"latest checkpoint epoch {epoch} scheduler")
    assert_finite_state(scaler_state, f"latest checkpoint epoch {epoch} scaler")

    _atomic_torch_save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "scaler_state_dict": scaler_state,
            "best_auc": best_auc,
            "best_metric": best_metric,  # current objective: selection score
            "best_auprc": best_metric,   # backward-compat alias for older tooling
        },
        path,
    )


def _best_metrics_from_log(log_path: str) -> tuple:
    """
    Recover best metrics from training log when resuming old checkpoints
    that did not store best_metric (AUPRC).
    """
    if not log_path or not os.path.isfile(log_path):
        return 0.0, 0.0

    best_auc = 0.0
    best_score = 0.0
    try:
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    score = float(row.get("selection_score", row.get("val_auprc_mean", "nan")))
                    auc = float(row.get("val_auc_mean", "nan"))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(score):
                    continue
                if score > best_score:
                    best_score = score
                    best_auc = auc if np.isfinite(auc) else best_auc
    except OSError:
        return 0.0, 0.0

    return float(best_auc), float(best_score)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    resume_log_path: str = None,
) -> tuple:
    
    print(f"  Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    assert_finite_state(ckpt.get("model_state_dict", {}), f"checkpoint {path} model")
    assert_finite_state(ckpt.get("optimizer_state_dict", {}), f"checkpoint {path} optimizer")
    assert_finite_state(ckpt.get("scheduler_state_dict", {}), f"checkpoint {path} scheduler")
    assert_finite_state(ckpt.get("scaler_state_dict", {}), f"checkpoint {path} scaler")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_auc = float(ckpt.get("best_auc", 0.0))

    # v4+ checkpoints should contain best_metric (selection score / legacy AUPRC).
    # For older checkpoints, recover from training_log.csv to avoid reset-to-zero bug.
    if ckpt.get("best_metric") is not None:
        best_metric = float(ckpt["best_metric"])
    elif ckpt.get("best_auprc") is not None:
        best_metric = float(ckpt["best_auprc"])
    else:
        log_best_auc, log_best_score = _best_metrics_from_log(resume_log_path)
        if log_best_score > 0:
            best_metric = log_best_score
            best_auc = log_best_auc
        else:
            best_metric = 0.0

    if not np.isfinite(best_auc):
        best_auc = 0.0
    if not np.isfinite(best_metric):
        best_metric = 0.0
    assert_model_finite(model, f"model after loading checkpoint {path}")

    print(
        f"  Resumed from epoch {ckpt['epoch']} "
        f"(best AUC: {best_auc:.4f}, best score: {best_metric:.4f})"
    )
    return start_epoch, best_auc, best_metric


# ============================================================
# EMA (Exponential Moving Average) — v9: smoother generalization
# ============================================================
class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Produces smoother, more generalizable models than raw checkpoints.
    
    decay = 0.999 means 0.1% of new params mixed in each step.
    Uses @torch.no_grad() for efficiency.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update shadow params with exponential moving average."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
                else:
                    self.shadow[name] = param.data.clone()

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


# ============================================================
# Mixup Augmentation (v9) — improves generalization for multi-label
# ============================================================
def mixup_data(images, labels, mask, view_type=None, alpha=0.4):
    """
    Mixup: linearly interpolate between random pairs of samples.
    
    For multi-label classification, both labels AND masks are mixed
    so uncertainty masking remains correct.
    
    Args:
        images: [B, C, H, W]
        labels: [B, num_classes]  
        mask: [B, num_classes]
        alpha: Beta distribution parameter (0 = no mixing, higher = more mixing)
    Returns:
        mixed_images, mixed_labels, mixed_mask, lam
    """
    if alpha <= 0:
        return images, labels, mask, view_type, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    # For mask: keep a label masked only if BOTH samples have it masked
    mixed_mask = torch.max(mask, mask[index])
    mixed_view_type = None
    if view_type is not None:
        vt = view_type.float()
        mixed_view_type = lam * vt + (1 - lam) * vt[index]

    return mixed_images, mixed_labels, mixed_mask, mixed_view_type, lam


# ============================================================
# Train One Epoch — Gradient Accumulation + Clipping
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_fp16: bool = True,
    accum_steps: int = 1,
    max_grad_norm: float = 1.0,
    mixup_alpha: float = 0.0,
    ema: ModelEMA = None,
    finite_check_steps: int = 1,
) -> float:
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer_steps = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)  # v4: uncertainty mask
        view_type = batch["view_type"].to(device) if "view_type" in batch else None  # v5
        if not torch.isfinite(images).all():
            raise FloatingPointError(f"Non-finite input images at batch {i}")
        if not torch.isfinite(labels).all():
            raise FloatingPointError(f"Non-finite labels at batch {i}")
        if not torch.isfinite(mask).all():
            raise FloatingPointError(f"Non-finite masks at batch {i}")
        if view_type is not None and not torch.isfinite(view_type).all():
            raise FloatingPointError(f"Non-finite view_type at batch {i}")

        # v9: Mixup augmentation
        if mixup_alpha > 0:
            images, labels, mask, view_type, _ = mixup_data(
                images,
                labels,
                mask,
                view_type=view_type,
                alpha=mixup_alpha,
            )

        # Forward in autocast; loss in FP32 — ASL/BCE backward can overflow in fp16 even when loss is finite.
        with autocast('cuda', enabled=use_fp16):
            outputs = model(images, view_type=view_type)
            if not torch.isfinite(outputs).all():
                raise FloatingPointError(f"Non-finite model outputs at batch {i}")
        with autocast('cuda', enabled=False):
            loss = criterion(outputs.float(), labels, mask=mask)  # v4: pass mask
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at batch {i}: {loss.item()}"
            )
        loss_scaled = loss / accum_steps  # Scale cho accumulation

        # Backward (tích lũy gradient)
        scaler.scale(loss_scaled).backward()

        # Update weights mỗi accum_steps (hoặc batch cuối)
        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
                error_if_nonfinite=False,
            )
            if torch.isfinite(grad_norm).item():
                scaler.step(optimizer)
                scaler.update()
                optimizer_steps += 1
                if finite_check_steps > 0 and optimizer_steps % finite_check_steps == 0:
                    assert_model_finite(
                        model,
                        f"model after optimizer step {optimizer_steps} (batch {i})",
                    )
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
            else:
                print(
                    f"  [WARN] Non-finite grad norm at batch {i} (skipped optimizer step). "
                    "Consider lower LR, use_fp16=false, or reduce ASL gamma_neg / class_weight_max."
                )
                scaler.update()
                optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


# ============================================================
# Validate / Evaluate
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_fp16: bool = True,
) -> dict:
   
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_masks = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        view_type = batch["view_type"].to(device) if "view_type" in batch else None  # v5
        if not torch.isfinite(images).all():
            raise FloatingPointError(f"Non-finite validation input images at batch {num_batches}")
        if not torch.isfinite(labels).all():
            raise FloatingPointError(f"Non-finite validation labels at batch {num_batches}")
        if not torch.isfinite(mask).all():
            raise FloatingPointError(f"Non-finite validation masks at batch {num_batches}")
        if view_type is not None and not torch.isfinite(view_type).all():
            raise FloatingPointError(f"Non-finite validation view_type at batch {num_batches}")

        with autocast('cuda', enabled=use_fp16):
            outputs = model(images, view_type=view_type)
            if not torch.isfinite(outputs).all():
                raise FloatingPointError(
                    f"Non-finite validation outputs at batch {num_batches}"
                )
            loss = criterion(outputs, labels, mask=mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite validation loss at batch {num_batches}: {loss.item()}"
                )

        total_loss += loss.item()
        num_batches += 1

        # v9: model outputs logits, convert to probabilities for metrics
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_masks.append(mask.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Tính AUC-ROC + AUPRC cho mỗi class
    auc_per_class = {}
    auprc_per_class = {}
    valid_aucs = []
    valid_auprcs = []
    for i, label_name in enumerate(CHEXPERT_LABELS):
        try:
            # Lọc theo mask thật từ dataset/loss:
            #   mask=1 -> nhãn valid để tính metric
            #   mask=0 -> uncertainty/unobserved, loại khỏi metric
            valid_mask = all_masks[:, i] > 0.5
            y_true = all_labels[valid_mask, i]
            y_pred = all_preds[valid_mask, i]

            unique_labels = np.unique(y_true)
            if len(unique_labels) < 2 or len(y_true) == 0:
                auc_per_class[label_name] = None
                auprc_per_class[label_name] = None
                continue
            auc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            if not np.isnan(auc):
                auc_per_class[label_name] = auc
                valid_aucs.append(auc)
            else:
                auc_per_class[label_name] = None
            if not np.isnan(auprc):
                auprc_per_class[label_name] = auprc
                valid_auprcs.append(auprc)
            else:
                auprc_per_class[label_name] = None
        except ValueError:
            auc_per_class[label_name] = None
            auprc_per_class[label_name] = None

    auc_mean = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    auprc_mean = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0

    return {
        "loss": total_loss / max(num_batches, 1),
        "auc_mean": auc_mean,
        "auprc_mean": auprc_mean,
        "auc_per_class": auc_per_class,
        "auprc_per_class": auprc_per_class,
    }


# ============================================================
# Main Training Loop
# ============================================================
def train(config: dict, resume_path: str = None, warm_start_path: str = None):
    """Main training loop với đầy đủ tính năng."""
    set_seed(config["general"]["seed"])
    device = get_device(config)
    print_gpu_info()

    cnn_cfg = config["cnn"]
    train_cfg = cnn_cfg["training"]
    selection_cfg = train_cfg.get("model_selection", {})
    accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
    effective_batch = train_cfg["batch_size"] * accum_steps

    # --- [1/5] Build model ---
    print("\n[1/5] Building DenseNet-121 model...")
    model = build_model(config).to(device)
    params = model.count_params()
    print(f"  Total params: {params['total']:,} | Trainable: {params['trainable']:,}")

    # Freeze backbone: chỉ train head N epoch đầu (không freeze khi resume)
    # v9: freeze_backbone CŨNG áp dụng cho warm-start — protect strong label features
    freeze_epochs = train_cfg.get("freeze_backbone_epochs", 2)
    if freeze_epochs > 0 and not resume_path:
        model.freeze_backbone()
        params = model.count_params()
        tag = " (warm-start: protect backbone)" if warm_start_path else ""
        print(f"  Backbone FROZEN for {freeze_epochs} epochs{tag} (trainable: {params['trainable']:,} / {params['total']:,})")
    elif resume_path:
        print(f"  Backbone NOT frozen (resuming from checkpoint)")

    # --- [2/5] Build dataloaders ---
    print("\n[2/5] Building dataloaders...")
    train_loader, val_internal_loader, test_loader, valid_loader = build_dataloaders(config)
    msg = (
        f"  Train: {len(train_loader.dataset):,} | "
        f"Val Internal: {len(val_internal_loader.dataset):,} | "
        f"Test (held-out): {len(test_loader.dataset):,}"
    )
    if valid_loader is not None:
        msg += f" | Valid Official: {len(valid_loader.dataset):,}"
    print(msg)

    # --- [3/5] Loss, optimizer, scheduler ---
    print("\n[3/5] Setting up loss + warmup + scheduler...")
    criterion = build_criterion(train_cfg, train_loader.dataset, device)

    # Optimizer với differential LR (backbone lr nhỏ hơn head)
    lr_head = train_cfg["learning_rate"]
    backbone_lr_ratio = train_cfg.get("backbone_lr_ratio", 0.02)  # v3: 0.1 → 0.02
    lr_backbone = lr_head * backbone_lr_ratio  # v3: Backbone học chậm hơn 50x
    optimizer = torch.optim.AdamW(
        model.get_param_groups(lr_backbone=lr_backbone, lr_head=lr_head),
        weight_decay=train_cfg["weight_decay"],
        eps=float(train_cfg.get("optimizer_eps", 1e-8)),
    )

    # Scheduler: Warmup + ReduceLROnPlateau (v2: thông minh hơn cosine)
    num_epochs = train_cfg["num_epochs"]
    warmup_epochs = int(train_cfg.get("warmup_epochs", min(3, max(1, num_epochs // 5))))
    warmup_epochs = max(1, min(warmup_epochs, num_epochs))
    scheduler_type = train_cfg.get("scheduler", "plateau")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )

    # v3: Đọc scheduler params từ config
    sched_patience = train_cfg.get("scheduler_patience", 5)  # v3: 3 → 5
    sched_factor = train_cfg.get("scheduler_factor", 0.5)

    if scheduler_type == "plateau":
        # ReduceLROnPlateau: giảm LR khi val AUPRC không cải thiện đáng kể
        # threshold_mode='rel' + 0.001: yêu cầu cải thiện >= 0.1% relative thay vì 0.005 abs
        # (abs threshold quá cứng khi AUPRC đã cao, rel nhạy hơn với giai đoạn fine-tune)
        sched_min_delta = train_cfg.get("scheduler_min_delta", 0.001)
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=sched_factor, patience=sched_patience,
            threshold=sched_min_delta, threshold_mode='rel',
            min_lr=1e-7, verbose=True
        )
    elif scheduler_type == "cosine_warm":
        # v10: CosineAnnealingWarmRestarts — chu kỳ warm restart giúp thoát local minima
        # T_0: số epoch của chu kỳ đầu; T_mult: hệ số nhân chu kỳ sau mỗi lần restart
        # Gọi scheduler.step(epoch - warmup_epochs + (batch / batches_per_epoch)) per-batch
        # nhưng để đơn giản, gọi per-epoch với fractional epoch
        sched_T0 = train_cfg.get("scheduler_T0", 10)
        sched_T_mult = train_cfg.get("scheduler_T_mult", 2)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=sched_T0, T_mult=sched_T_mult, eta_min=1e-7
        )
        print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0={sched_T0}, T_mult={sched_T_mult})")
    else:
        # Cosine Annealing (fallback)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-7
        )

    scaler = GradScaler('cuda', enabled=train_cfg["use_fp16"])

    # --- [4/5] Resume (nếu có) ---
    start_epoch = 1
    best_auc = 0.0
    best_metric = 0.0      # tracker for saving best_model.pth (strict >, no threshold)
    es_best_metric = 0.0   # tracker for early stopping patience (requires > + min_delta)
    # v12: Safety floor — track peak AUPRC per label to prevent regression on strong labels
    peak_auprc_per_label = {label: 0.0 for label in CHEXPERT_LABELS}
    output_base = config["general"].get("output_dir", "./outputs")
    # base_checkpoint_dir: dùng key riêng trong config để tránh nest sai (v1/v{N})
    base_checkpoint_dir = config["paths"].get(
        "checkpoint_base_dir",
        os.path.dirname(os.path.dirname(config["paths"]["densenet_checkpoint"]))
    )
    ensure_dir(base_checkpoint_dir)

    # Warm-start: nạp chỉ model weights từ checkpoint bên ngoài (không restore optimizer/epoch)
    if warm_start_path and os.path.isfile(warm_start_path):
        print(f"\n[Warm-start] Loading model weights from: {warm_start_path}")
        ws_ckpt = torch.load(warm_start_path, map_location=device, weights_only=False)
        ws_state = ws_ckpt.get("model_state_dict", ws_ckpt)  # hỗ trợ cả checkpoint lẫn state_dict thuần
        assert_finite_state(ws_state, f"warm-start checkpoint {warm_start_path}")
        model.load_state_dict(ws_state, strict=False)
        assert_model_finite(model, f"model after warm-start {warm_start_path}")
        print("  >> Warm-start successful - optimizer/LR/epoch reset to defaults")

    is_resume = False
    if resume_path and os.path.isfile(resume_path):
        # Peek epoch từ checkpoint để chọn đúng scheduler cần restore
        # (start_epoch vẫn = 1 tại đây nên không dùng được để kiểm tra)
        _peek = torch.load(resume_path, map_location="cpu", weights_only=False)
        _ckpt_epoch = _peek.get("epoch", 0)
        _resume_scheduler = warmup_scheduler if _ckpt_epoch < warmup_epochs else main_scheduler
        del _peek  # giải phóng RAM ngay

        # Backward-compat: checkpoint cũ chưa có best_metric -> recover từ log v{N}
        resume_log_path = None
        resume_version = os.path.basename(os.path.dirname(resume_path))
        if resume_version.startswith("v") and resume_version[1:].isdigit():
            resume_log_path = os.path.join(output_base, resume_version, "training_log.csv")

        start_epoch, best_auc, best_metric = load_checkpoint(
            resume_path, model, optimizer,
            _resume_scheduler,
            scaler, device,
            resume_log_path=resume_log_path,
        )
        is_resume = True

    # v3: Auto-versioning — tạo thư mục v{N} cho cả log lẫn checkpoint
    if not is_resume:
        version = 1
        while os.path.isdir(os.path.join(output_base, f"v{version}")):
            version += 1
    else:
        # Resume: tìm version mới nhất
        version = 1
        while os.path.isdir(os.path.join(output_base, f"v{version + 1}")):
            version += 1

    # Log dir
    log_dir = os.path.join(output_base, f"v{version}")
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "training_log.csv")
    logger = TrainingLogger(log_path, resume=is_resume)

    # Checkpoint dir — versioned
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"v{version}")
    ensure_dir(checkpoint_dir)
    latest_ckpt_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    # v9: Top-K checkpoint tracker for auto-ensemble
    topk = train_cfg.get("topk_checkpoints", 5)
    topk_tracker = TopKCheckpointTracker(checkpoint_dir, k=topk)
    print(f"  Top-K checkpoint tracking: K={topk}")

    # v9: EMA (Exponential Moving Average)
    use_ema = train_cfg.get("use_ema", False)
    ema = None
    if use_ema:
        ema_decay = train_cfg.get("ema_decay", 0.999)
        ema = ModelEMA(model, decay=ema_decay)
        print(f"  EMA: ON (decay={ema_decay})")
    else:
        print(f"  EMA: OFF")

    # --- [5/5] Training loop ---
    # V4: SWA (Stochastic Weight Averaging)
    use_swa = train_cfg.get("use_swa", True)
    swa_start_epoch = int(num_epochs * 0.75)  # SWA starts at 75% of training
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_lr = train_cfg.get("swa_lr", 1e-5)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)
        print(f"  SWA: ON (start epoch {swa_start_epoch}, lr={swa_lr})")
    else:
        print(f"  SWA: OFF")

    # v10: Progressive resize — train ở resolution thấp hơn trước, tăng dần về cuối
    prog_resize_enabled = train_cfg.get("progressive_resize_enabled", False)
    prog_resize_epoch = train_cfg.get("progressive_resize_epoch", num_epochs)  # epoch chuyển lên full size
    prog_resize_start_size = train_cfg.get("progressive_resize_start_size", cnn_cfg["image_size"])  # size giai đoạn đầu
    prog_resize_full_size = cnn_cfg["image_size"]  # size gốc từ config = target cuối

    if prog_resize_enabled:
        print(f"  Progressive resize: epoch 1-{prog_resize_epoch - 1} -> {prog_resize_start_size}px, "
              f"epoch {prog_resize_epoch}-{num_epochs} -> {prog_resize_full_size}px")

    print(f"\nStarting training (v{version})...")
    print(f"  Epochs: {start_epoch} -> {num_epochs}")
    if prog_resize_enabled:
        print(f"  Image size: {prog_resize_start_size}px -> {prog_resize_full_size}px (at epoch {prog_resize_epoch})")
    else:
        print(f"  Image size: {cnn_cfg['image_size']}px")
    print(f"  Batch: {train_cfg['batch_size']} x {accum_steps} accum = {effective_batch} effective")
    print(f"  LR: backbone={lr_backbone:.6f} / head={lr_head} (ratio={backbone_lr_ratio})")
    print(f"  Warmup: {warmup_epochs} epochs -> {scheduler_type} (patience={sched_patience}, factor={sched_factor})")
    print(f"  FP16: {train_cfg['use_fp16']} | Grad clip: {train_cfg.get('max_grad_norm', 0.5)}")
    print(
        "  Model selection: "
        f"{selection_cfg.get('strategy', 'weighted_group_auprc')} "
        f"(core={selection_cfg.get('core_labels', DEFAULT_CORE_LABELS)}, "
        f"rare={selection_cfg.get('rare_labels', DEFAULT_RARE_LABELS)})"
    )
    print(f"  Early stopping: patience={train_cfg['early_stopping_patience']} on selection score")
    print(f"  Device: {device}")
    print(f"  Log: {log_path}")

    patience_counter = 0
    _current_image_size = prog_resize_start_size if prog_resize_enabled else prog_resize_full_size

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()

        # v10: Progressive resize — chuyển lên full resolution tại prog_resize_epoch
        if prog_resize_enabled:
            target_size = prog_resize_full_size if epoch >= prog_resize_epoch else prog_resize_start_size
            if target_size != _current_image_size:
                _current_image_size = target_size
                _datasets = [train_loader.dataset, val_internal_loader.dataset, test_loader.dataset]
                if valid_loader is not None:
                    _datasets.append(valid_loader.dataset)
                for ds in _datasets:
                    ds.set_image_size(_current_image_size)
                print(f"  [Progressive resize] Epoch {epoch}: switch to {_current_image_size}px")

        # Unfreeze backbone sau freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            params = model.count_params()
            print(f"\n  >>> Backbone UNFROZEN at epoch {epoch} (trainable: {params['trainable']:,})")

        # Train
        max_grad_norm = train_cfg.get("max_grad_norm", 0.5)  # v3: 1.0 → 0.5
        mixup_alpha = train_cfg.get("mixup_alpha", 0.0)  # v9: mixup
        finite_check_steps = int(train_cfg.get("finite_check_steps", 1))
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            use_fp16=train_cfg["use_fp16"],
            accum_steps=accum_steps,
            max_grad_norm=max_grad_norm,
            mixup_alpha=mixup_alpha,
            ema=ema,
            finite_check_steps=finite_check_steps,
        )

        # v12: Hard-negative replay mini-pass
        hard_neg_cfg = train_cfg.get("hard_negative", {})
        if hard_neg_cfg.get("enabled", False):
            hn_start = int(hard_neg_cfg.get("start_epoch", 12))
            hn_interval = int(hard_neg_cfg.get("mine_interval", 2))
            if epoch >= hn_start and (epoch - hn_start) % hn_interval == 0:
                hn_indices, hn_manifest = mine_hard_negatives(
                    model, train_loader.dataset, device, hard_neg_cfg,
                    use_fp16=train_cfg["use_fp16"],
                )
                # Persist manifest for post-hoc quality inspection
                hn_manifest["epoch"] = epoch
                hn_manifest_path = os.path.join(log_dir, f"hard_neg_epoch_{epoch}.json")
                with open(hn_manifest_path, "w", encoding="utf-8") as f:
                    json.dump(hn_manifest, f, indent=2, ensure_ascii=False)

                if len(hn_indices) > 0:
                    replay_subset = Subset(train_loader.dataset, hn_indices)
                    replay_loader = DataLoader(
                        replay_subset,
                        batch_size=train_cfg["batch_size"],
                        shuffle=True,
                        num_workers=train_cfg.get("num_workers", 2),
                        pin_memory=True,
                    )
                    # Scale LR down for replay to avoid overfitting on small pool
                    replay_lr_scale = float(hard_neg_cfg.get("replay_lr_scale", 0.3))
                    original_lrs = [pg["lr"] for pg in optimizer.param_groups]
                    for pg in optimizer.param_groups:
                        pg["lr"] *= replay_lr_scale

                    replay_loss = train_one_epoch(
                        model, replay_loader, criterion, optimizer, scaler, device,
                        use_fp16=train_cfg["use_fp16"],
                        accum_steps=1,  # no accumulation for small replay
                        max_grad_norm=max_grad_norm,
                        mixup_alpha=0.0,  # no mixup for replay
                        ema=ema,
                        finite_check_steps=finite_check_steps,
                    )

                    # Restore original LR
                    for pg, orig_lr in zip(optimizer.param_groups, original_lrs):
                        pg["lr"] = orig_lr

                    print(f"  [Hard-neg replay] loss={replay_loss:.4f}, "
                          f"samples={len(hn_indices)}, lr_scale={replay_lr_scale}")

        # Validate trên val_internal (44k mẫu, ổn định cho early stopping/scheduler)
        val_results = validate(
            model, val_internal_loader, criterion, device, train_cfg["use_fp16"]
        )
        val_results = _enrich_selection_metrics(val_results, selection_cfg)
        val_loss = val_results["loss"]
        val_auc = val_results["auc_mean"]
        val_auprc = val_results["auprc_mean"]
        core_auprc = val_results["core_auprc_mean"]
        rare_auprc = val_results["rare_auprc_mean"]
        selection_score = val_results["selection_score"]
        assert_finite_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_auprc": val_auprc,
                "core_auprc": core_auprc,
                "rare_auprc": rare_auprc,
                "selection_score": selection_score,
            },
            f"epoch {epoch}",
        )

        # v12: EMA validation mỗi 2 epoch — chọn best_model từ EMA nếu tốt hơn model gốc
        # Tránh eval mỗi epoch vì val trên 37k ảnh tốn ~20% thời gian epoch
        _ema_is_better = False
        if use_ema and ema is not None and epoch % 2 == 0:
            ema.apply_shadow(model)
            _ema_val = validate(model, val_internal_loader, criterion, device, train_cfg["use_fp16"])
            ema.restore(model)
            _ema_val = _enrich_selection_metrics(_ema_val, selection_cfg)
            if _ema_val["selection_score"] > selection_score:
                val_results = _ema_val
                val_loss = _ema_val["loss"]
                val_auc = _ema_val["auc_mean"]
                val_auprc = _ema_val["auprc_mean"]
                core_auprc = _ema_val["core_auprc_mean"]
                rare_auprc = _ema_val["rare_auprc_mean"]
                selection_score = _ema_val["selection_score"]
                assert_finite_metrics(
                    {
                        "ema_val_loss": val_loss,
                        "ema_val_auc": val_auc,
                        "ema_val_auprc": val_auprc,
                        "ema_selection_score": selection_score,
                    },
                    f"EMA epoch {epoch}",
                )
                _ema_is_better = True
                print(
                    "  [EMA] Better val: "
                    f"AUC={val_auc:.4f} AUPRC={val_auprc:.4f} score={selection_score:.4f} "
                    "→ dùng EMA cho checkpoint"
                )

        elapsed = time.time() - start_time

        # SWA update (after swa_start_epoch)
        in_swa = use_swa and epoch >= swa_start_epoch
        if in_swa:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
        else:
            # Normal scheduler step
            if epoch <= warmup_epochs:
                warmup_scheduler.step()
            elif scheduler_type == "plateau":
                main_scheduler.step(selection_score)
            elif scheduler_type == "cosine_warm":
                # v10: pass epoch offset so restarts align correctly after warmup
                main_scheduler.step(epoch - warmup_epochs)
            else:
                main_scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

        # Log to CSV
        logger.log(
            epoch, train_loss, val_loss, val_auc, val_auprc, core_auprc, rare_auprc, selection_score, lr, elapsed,
            val_results["auc_per_class"],
            val_results["auprc_per_class"],
        )

        # Print
        swa_tag = " [SWA]" if in_swa else ""
        print(
            f"Epoch {epoch:02d}/{num_epochs}{swa_tag} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val AUPRC: {val_auprc:.4f} | "
            f"Core: {core_auprc:.4f} | "
            f"Rare: {rare_auprc:.4f} | "
            f"Score: {selection_score:.4f} | "
            f"LR: {lr:.6f} | "
            f"Time: {elapsed:.0f}s"
        )

        # AUC + AUPRC per class (mỗi 5 epochs hoặc epoch cuối)
        if epoch % 5 == 0 or epoch == num_epochs:
            print("  AUC / AUPRC per class:")
            for label in CHEXPERT_LABELS:
                auc = val_results["auc_per_class"].get(label)
                auprc = val_results["auprc_per_class"].get(label)
                if auc is not None:
                    auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
                    print(f"    {label}: AUC={auc:.4f} | AUPRC={auprc_str}")

        # V8: Tách 2 luồng — save best_model và early stopping độc lập.
        #   best_metric: strict > (không threshold) → best_model.pth luôn là checkpoint tốt nhất
        #   es_best_metric: > + min_delta → patience chỉ reset khi cải thiện đáng kể
        es_min_delta = train_cfg.get("early_stopping_min_delta", 0.002)
        primary_metric = selection_score if selection_score > 0 else (val_auprc if val_auprc > 0 else val_auc)

        # Luồng 1: Save best_model (strict >, không min_delta)
        # v12: Safety floor dùng peak TRƯỚC epoch này (peak chưa update)
        floor_blocked = False
        if primary_metric > best_metric:
            # v12: Safety floor — chặn save best_model nếu label mạnh drop quá nhiều
            floor_cfg = selection_cfg.get("safety_floor", {})
            floor_blocked = False
            if floor_cfg.get("enabled", False) and epoch >= floor_cfg.get("warmup_epochs", 6):
                max_drop = float(floor_cfg.get("max_drop", 0.03))
                monitor_labels = floor_cfg.get("monitor_labels", [])
                for label in monitor_labels:
                    current_val = val_results["auprc_per_class"].get(label)
                    if current_val is None:
                        continue
                    drop = peak_auprc_per_label[label] - float(current_val)
                    if drop > max_drop:
                        print(
                            f"  !! Safety floor: {label} AUPRC dropped {drop:.4f} "
                            f"(peak={peak_auprc_per_label[label]:.4f}, now={float(current_val):.4f}, "
                            f"max_drop={max_drop}) → skip best_model save"
                        )
                        floor_blocked = True
                        break

            if not floor_blocked:
                best_metric = primary_metric
                best_auc = val_auc  # vẫn track AUC để log
                # v12: Nếu EMA tốt hơn → save EMA weights vào best_model
                if _ema_is_better and ema is not None:
                    ema.apply_shadow(model)
                    try:
                        save_model_weights(best_model_path, model, context=f"best EMA epoch {epoch}")
                    finally:
                        ema.restore(model)
                    print(
                        f"  >> Best model saved (EMA) to v{version}/ "
                        f"(score: {selection_score:.4f}, core: {core_auprc:.4f}, rare: {rare_auprc:.4f})"
                    )
                else:
                    save_model_weights(best_model_path, model, context=f"best epoch {epoch}")
                    print(
                        f"  >> Best model saved to v{version}/ "
                        f"(score: {selection_score:.4f}, core: {core_auprc:.4f}, rare: {rare_auprc:.4f})"
                    )

        # v12: Update peak AUPRC per label SAU khi đã check floor (dùng peak trước epoch này)
        for label in CHEXPERT_LABELS:
            auprc_val = val_results["auprc_per_class"].get(label)
            if auprc_val is not None:
                peak_auprc_per_label[label] = max(peak_auprc_per_label[label], float(auprc_val))

        # Luồng 2: Early stopping patience (> + min_delta)
        # v12: Nếu safety floor reject epoch này → cũng không reset patience
        if floor_blocked:
            patience_counter += 1
        elif primary_metric > es_best_metric + es_min_delta:
            es_best_metric = primary_metric
            patience_counter = 0
        else:
            patience_counter += 1

        # v9: Top-K checkpoint tracking (independent of best model logic)
        if selection_score > 0:
            saved_topk = topk_tracker.update(epoch, selection_score, val_auc, model)
            if saved_topk:
                print(f"  >> Top-{topk} checkpoint saved: epoch_{epoch}.pth")

        # Save latest checkpoint (luôn save mỗi epoch -> resume nếu crash)
        # Save sau khi update best_* để checkpoint phản ánh state mới nhất.
        save_checkpoint(
            latest_ckpt_path, model, optimizer,
            warmup_scheduler if epoch <= warmup_epochs else main_scheduler,
            scaler, epoch, best_auc, best_metric,
        )

        # Early stopping (nhưng không stop trong SWA phase)
        if not in_swa and patience_counter >= train_cfg["early_stopping_patience"]:
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(patience={train_cfg['early_stopping_patience']})"
            )
            break

    # --- SWA: Update BatchNorm + save SWA model ---
    if use_swa and swa_model is not None:
        print("\nUpdating SWA BatchNorm statistics...")

        # update_bn cần forward pass qua toàn bộ train data để cập nhật BN running stats.
        # Không dùng torch.optim.swa_utils.update_bn vì nó chỉ hỗ trợ loader trả tensor,
        # trong khi model.forward() cần cả view_type khi use_view_position=True.
        swa_model.train()
        with torch.no_grad():
            # Reset BN stats
            for module in swa_model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()

            for batch in tqdm(train_loader, desc="SWA BN update", leave=False):
                images = batch["image"].to(device)
                view_type = batch["view_type"].to(device) if "view_type" in batch else None
                swa_model(images, view_type=view_type)
        # Save SWA model
        swa_model_path = os.path.join(checkpoint_dir, "swa_model.pth")
        save_model_weights(swa_model_path, swa_model.module, context="SWA model")
        print(f"  SWA model saved: {swa_model_path}")

        # Evaluate SWA model on valid
        swa_model.to(device)
        swa_results = validate(
            swa_model, val_internal_loader, criterion, device, train_cfg["use_fp16"]
        )
        swa_results = _enrich_selection_metrics(swa_results, selection_cfg)
        print(
            f"  SWA Val AUC: {swa_results['auc_mean']:.4f} | "
            f"AUPRC: {swa_results['auprc_mean']:.4f} | "
            f"Score: {swa_results['selection_score']:.4f}"
        )

        # Nếu SWA tốt hơn best → dùng SWA model
        if swa_results["selection_score"] > best_metric:
            print("  >> SWA model is BETTER! Using SWA as final model.")
            shutil.copy2(swa_model_path, best_model_path)
            best_auc = swa_results["auc_mean"]
            best_metric = swa_results["selection_score"]
        else:
            print(
                f"  >> SWA model ({swa_results['selection_score']:.4f}) "
                f"vs Best ({best_metric:.4f}) - keeping best."
            )

    # --- v9: EMA evaluation ---
    if use_ema and ema is not None:
        print(f"\nEvaluating EMA model...")
        ema.apply_shadow(model)
        ema_results = validate(
            model, val_internal_loader, criterion, device, train_cfg["use_fp16"]
        )
        ema_results = _enrich_selection_metrics(ema_results, selection_cfg)
        print(
            f"  EMA Val AUC: {ema_results['auc_mean']:.4f} | "
            f"AUPRC: {ema_results['auprc_mean']:.4f} | "
            f"Score: {ema_results['selection_score']:.4f}"
        )

        if ema_results["selection_score"] > best_metric:
            print("  >> EMA model is BETTER! Using EMA as final model.")
            ema_path = os.path.join(checkpoint_dir, "ema_model.pth")
            save_model_weights(ema_path, model, context="final EMA model")
            shutil.copy2(ema_path, best_model_path)
            best_auc = ema_results["auc_mean"]
            best_metric = ema_results["selection_score"]
        else:
            print(
                f"  >> EMA ({ema_results['selection_score']:.4f}) "
                f"vs Best ({best_metric:.4f}) - keeping best."
            )

        ema.restore(model)

    # --- v9: Top-K checkpoint ensemble evaluation ---
    topk_paths = topk_tracker.get_paths()
    if len(topk_paths) >= 2:
        print(f"\n{'='*60}")
        print(f"Evaluating Top-{len(topk_paths)} checkpoint ensemble...")
        print(topk_tracker.summary())

        # Collect predictions from each checkpoint
        ensemble_preds = []
        for ckpt_path in topk_paths:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=device, weights_only=True)
            )
            ckpt_results = validate(
                model, val_internal_loader, criterion, device, train_cfg["use_fp16"]
            )
            # Re-collect raw predictions for averaging
            model.eval()
            ckpt_preds = []
            with torch.no_grad():
                for batch in val_internal_loader:
                    images = batch["image"].to(device)
                    view_type = batch["view_type"].to(device) if "view_type" in batch else None
                    with autocast('cuda', enabled=train_cfg["use_fp16"]):
                        outputs = model(images, view_type=view_type)
                    ckpt_preds.append(torch.sigmoid(outputs).cpu().numpy())
            ensemble_preds.append(np.concatenate(ckpt_preds, axis=0))

        # Average predictions across checkpoints
        avg_preds = np.mean(ensemble_preds, axis=0)

        # Collect ground truth labels + masks
        all_labels_gt = []
        all_masks_gt = []
        for batch in val_internal_loader:
            all_labels_gt.append(batch["labels"].numpy())
            all_masks_gt.append(batch["mask"].numpy())
        all_labels_gt = np.concatenate(all_labels_gt, axis=0)
        all_masks_gt = np.concatenate(all_masks_gt, axis=0)

        # Compute ensemble metrics
        valid_aucs, valid_auprcs = [], []
        for i, label in enumerate(CHEXPERT_LABELS):
            valid_mask = all_masks_gt[:, i] > 0.5
            y_true = all_labels_gt[valid_mask, i]
            y_pred = avg_preds[valid_mask, i]
            if len(np.unique(y_true)) < 2:
                continue
            auc = roc_auc_score(y_true, y_pred)
            auprc = average_precision_score(y_true, y_pred)
            if not np.isnan(auc):
                valid_aucs.append(auc)
            if not np.isnan(auprc):
                valid_auprcs.append(auprc)

        ens_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        ens_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else 0.0
        ens_results = {
            "auprc_per_class": {label: None for label in CHEXPERT_LABELS},
            "auprc_mean": ens_auprc,
        }
        # Reuse the same selection policy for ensemble comparison.
        for i, label in enumerate(CHEXPERT_LABELS):
            valid_mask = all_masks_gt[:, i] > 0.5
            y_true = all_labels_gt[valid_mask, i]
            y_pred = avg_preds[valid_mask, i]
            if len(np.unique(y_true)) < 2:
                continue
            ens_results["auprc_per_class"][label] = average_precision_score(y_true, y_pred)
        ens_results = _enrich_selection_metrics(ens_results, selection_cfg)
        ens_score = ens_results["selection_score"]

        print(f"  Ensemble Val AUC: {ens_auc:.4f} | AUPRC: {ens_auprc:.4f} | Score: {ens_score:.4f}")
        print(f"  vs Best single: AUC={best_auc:.4f} | Score={best_metric:.4f}")

        if ens_score > best_metric:
            print(f"  >> Ensemble is BETTER by +{ens_score - best_metric:.4f} score!")
            # Save ensemble manifest for later use by eval scripts
            ensemble_manifest = {
                "type": "checkpoint_ensemble",
                "checkpoints": topk_tracker.records,
                "ensemble_auc": round(ens_auc, 6),
                "ensemble_auprc": round(ens_auprc, 6),
                "ensemble_score": round(ens_score, 6),
            }
            manifest_out = os.path.join(checkpoint_dir, "ensemble_manifest.json")
            with open(manifest_out, "w", encoding="utf-8") as f:
                json.dump(ensemble_manifest, f, indent=2)
            print(f"  >> Ensemble manifest: {manifest_out}")
        else:
            print("  >> Single best model is better - no ensemble advantage.")

        # Restore best model for final evaluation
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

    # Sync final best checkpoint to runtime path for deployment/inference.
    root_ckpt_path = config["paths"]["densenet_checkpoint"]
    try:
        root_abs = os.path.abspath(root_ckpt_path)
        best_abs = os.path.abspath(best_model_path)
        if root_abs != best_abs:
            ensure_dir(os.path.dirname(root_ckpt_path))
            shutil.copy2(best_model_path, root_ckpt_path)
    except OSError as e:
        print(f"  [WARN] Failed to sync root checkpoint: {e}")

    logger.close()

    print(f"\n{'='*60}")
    print(f"Training complete! (v{version})")
    print(f"  Best validation AUC: {best_auc:.4f}")
    print(f"  Best validation score: {best_metric:.4f}")
    print(f"  Best model: {best_model_path}")
    print(f"  Root copy: {config['paths']['densenet_checkpoint']}")
    print(f"  Latest checkpoint: {latest_ckpt_path}")
    print(f"  Training log: {log_path}")

    # --- (Optional) Đánh giá cuối trên tập validation ngoài nếu được cấu hình ---
    if valid_loader is not None:
        print(f"\n{'='*60}")
        print("Evaluating best model on EXTERNAL validation set (paths.valid_csv)...")
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
        official_results = validate(
            model, valid_loader, criterion, device, train_cfg["use_fp16"]
        )

        print(f"  External Val Loss: {official_results['loss']:.4f}")
        print(f"  External Val AUC (mean): {official_results['auc_mean']:.4f}")
        print(f"  External Val AUPRC (mean): {official_results['auprc_mean']:.4f}")
        print("  External Val AUC / AUPRC per class:")
        for label in CHEXPERT_LABELS:
            auc = official_results["auc_per_class"].get(label)
            auprc = official_results["auprc_per_class"].get(label)
            if auc is not None:
                auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
                print(f"    {label}: AUC={auc:.4f} | AUPRC={auprc_str}")
    else:
        # NIH: chỉ có train/val/test — bỏ qua step này, vẫn load best model cho test eval phía dưới.
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

    # --- v12: Held-out TEST set evaluation (NEVER seen during training) ---
    # This is the ONLY time the test set is evaluated — after ALL training
    # decisions (model selection, early stopping, SWA, EMA) are finalized.
    print(f"\n{'='*60}")
    print("Evaluating best model on HELD-OUT TEST set (NIH official test_list / legacy 15% split)...")
    print("  *** This set was NEVER used for any training decision ***")
    test_results = validate(
        model, test_loader, criterion, device, train_cfg["use_fp16"]
    )
    test_results = _enrich_selection_metrics(test_results, selection_cfg)

    print(f"  Test Loss: {test_results['loss']:.4f}")
    print(f"  Test AUC (mean): {test_results['auc_mean']:.4f}")
    print(f"  Test AUPRC (mean): {test_results['auprc_mean']:.4f}")
    print(f"  Test Score: {test_results['selection_score']:.4f}")
    print(f"    Core AUPRC: {test_results['core_auprc_mean']:.4f}")
    print(f"    Rare AUPRC: {test_results['rare_auprc_mean']:.4f}")
    print("  Test AUC / AUPRC per class:")
    for label in CHEXPERT_LABELS:
        auc = test_results["auc_per_class"].get(label)
        auprc = test_results["auprc_per_class"].get(label)
        if auc is not None:
            auprc_str = f"{auprc:.4f}" if auprc is not None else "N/A"
            print(f"    {label}: AUC={auc:.4f} | AUPRC={auprc_str}")

    # Compare val_internal (used for selection) vs test (held-out) to detect overfitting
    print(f"\n  --- Overfit check: Val(selection) vs Test(held-out) ---")
    print(f"  Val  score: {best_metric:.4f}")
    print(f"  Test score: {test_results['selection_score']:.4f}")
    gap = best_metric - test_results['selection_score']
    if gap > 0.03:
        print(f"  [WARN] Val-Test gap = {gap:.4f} > 0.03 — possible overfitting to validation set!")
    elif gap > 0.01:
        print(f"  [INFO] Val-Test gap = {gap:.4f} — minor; results likely generalizable.")
    else:
        print(f"  [OK] Val-Test gap = {gap:.4f} — excellent generalization.")

    # Save test results to JSON for reproducibility
    test_out_path = os.path.join(log_dir, "test_results.json")
    test_report = {
        "split": "held_out_test",
        "num_images": len(test_loader.dataset),
        "loss": round(test_results["loss"], 6),
        "auc_mean": round(test_results["auc_mean"], 6),
        "auprc_mean": round(test_results["auprc_mean"], 6),
        "selection_score": round(test_results["selection_score"], 6),
        "core_auprc_mean": round(test_results["core_auprc_mean"], 6),
        "rare_auprc_mean": round(test_results["rare_auprc_mean"], 6),
        "auc_per_class": {
            k: round(v, 6) if v is not None else None
            for k, v in test_results["auc_per_class"].items()
        },
        "auprc_per_class": {
            k: round(v, 6) if v is not None else None
            for k, v in test_results["auprc_per_class"].items()
        },
        "val_test_gap": round(gap, 6),
    }
    with open(test_out_path, "w", encoding="utf-8") as f:
        json.dump(test_report, f, indent=2)
    print(f"  Test results saved: {test_out_path}")

    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on NIH ChestX-ray14")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training "
             "(e.g. models/densenet121/v3/latest_checkpoint.pth)",
    )
    parser.add_argument(
        "--warmstart", type=str, default=None,
        help="Path to model checkpoint/state_dict for warm-start: nạp weights nhưng KHÔNG restore "
             "optimizer/epoch — dùng để fine-tune từ model đã train sẵn "
             "(e.g. models/densenet121/v2/best_model.pth)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume_path=args.resume, warm_start_path=args.warmstart)
