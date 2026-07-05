"""Export high-resolution workflow PNG for GitHub README."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "workflow.png"
DPI = 180
W, H = 18, 14


def box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#6B7280", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.3, edgecolor=ec, facecolor=fc, transform=ax.transAxes,
    ))
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, fontweight="bold" if bold else "normal",
        color="#111827", transform=ax.transAxes, linespacing=1.25,
    )


def arrow(ax, x1, y1, x2, y2, color="#6B7280", dashed=False):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
        linewidth=1.3, color=color, linestyle="--" if dashed else "-",
        transform=ax.transAxes,
    ))


def band_bg(ax, y, h, label):
    ax.add_patch(FancyBboxPatch(
        (0.025, y), 0.95, h, boxstyle="square,pad=0",
        linewidth=1, edgecolor="#D1D5DB", facecolor="#F3F4F6", transform=ax.transAxes,
    ))
    ax.text(0.035, y + h - 0.015, label, fontsize=10, fontweight="bold",
            color="#374151", va="top", transform=ax.transAxes)


def export() -> None:
    fig, ax = plt.subplots(figsize=(W, H), dpi=DPI)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.5, 0.985, "Chest X-ray Multi-Label Diagnosis — System Workflow",
            ha="center", va="top", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.962,
            "Model Training  →  Calibration on Validation  →  Inference & Web Deployment",
            ha="center", va="top", fontsize=11, color="#6B7280")
    ax.text(0.5, 0.945, "Clinical decision support — not a substitute for physicians",
            ha="center", va="top", fontsize=9, style="italic", color="#9CA3AF")

    # --- Phase 1 ---
    band_bg(ax, 0.855, 0.095, "1. MODEL TRAINING")
    items1 = [
        ("NIH ChestX-ray14\ntrain / val / test", "#FFF", "#6B7280"),
        ("Preprocessing\nCLAHE · letterbox 448", "#FFF", "#6B7280"),
        ("Augmentation\ntrain only", "#FFF", "#6B7280"),
        ("DenseNet-121\nLSE pooling", "#EDE9FE", "#7C3AED"),
        ("14-label Head\nsigmoid outputs", "#FFF", "#6B7280"),
        ("Loss FZLPR\n14 labels", "#FFF", "#6B7280"),
    ]
    bw, bh, y1 = 0.135, 0.062, 0.875
    x_start = 0.065
    for i, (txt, fc, ec) in enumerate(items1):
        x = x_start + i * (bw + 0.012)
        box(ax, x, y1, bw, bh, txt, fc=fc, ec=ec, fs=8.5)
        if i < 5:
            arrow(ax, x + bw + 0.003, y1 + bh / 2, x + bw + 0.009, y1 + bh / 2)

    arrow(ax, 0.5, 0.855, 0.5, 0.835, color="#7C3AED")
    ax.text(0.5, 0.842, "logits on validation set", ha="center", fontsize=8.5,
            color="#7C3AED", fontweight="bold")

    # --- Phase 2 ---
    band_bg(ax, 0.735, 0.085, "2. CALIBRATION ON VALIDATION")
    box(ax, 0.12, 0.755, 0.34, 0.055,
        "Probability Calibration\nTemperature Scaling / Isotonic (lowest NLL on Val)",
        fc="#EDE9FE", ec="#7C3AED", fs=9)
    box(ax, 0.54, 0.755, 0.34, 0.055,
        "Youden-J Threshold\ndetection threshold + equivocal band (Δ = 0.08)",
        fc="#EDE9FE", ec="#7C3AED", fs=9)
    ax.text(0.27, 0.722, "T / Isotonic params", ha="center", fontsize=8, color="#7C3AED")
    ax.text(0.71, 0.722, "per-label thresholds", ha="center", fontsize=8, color="#7C3AED")
    arrow(ax, 0.27, 0.728, 0.27, 0.705, color="#7C3AED", dashed=True)
    arrow(ax, 0.71, 0.728, 0.71, 0.705, color="#7C3AED", dashed=True)

    # --- Phase 3 ---
    band_bg(ax, 0.505, 0.195, "3. INFERENCE & TRIAGE")
    items3 = [
        "Input Image\nPNG / JPG",
        "Normalization\n448 · CLAHE",
        "DenseNet-121\nload weights",
        "Apply Calibration\nfrom phase 2",
        "3-tier Classification\nper-label thresholds",
    ]
    bw3, y3 = 0.155, 0.615
    for i, txt in enumerate(items3):
        x = 0.065 + i * (bw3 + 0.012)
        fc, ec = ("#EDE9FE", "#7C3AED") if i == 2 else ("#FFF", "#6B7280")
        box(ax, x, y3, bw3, 0.055, txt, fc=fc, ec=ec, fs=8.5)
        if i < 4:
            arrow(ax, x + bw3 + 0.003, y3 + 0.027, x + bw3 + 0.009, y3 + 0.027)

    ax.text(0.5, 0.575, "prob vs threshold?", ha="center", fontsize=9,
            color="#CA8A04", fontweight="bold")
    tri = [
        ("POSITIVE\nprob ≥ threshold", "#FCA5A5", "#DC2626"),
        ("UNCERTAIN\nequivocal band", "#FDE047", "#CA8A04"),
        ("NEGATIVE\nprob < lower bound", "#86EFAC", "#16A34A"),
    ]
    for i, (txt, fc, ec) in enumerate(tri):
        x = 0.18 + i * 0.22
        box(ax, x, 0.525, 0.18, 0.048, txt, fc=fc, ec=ec, fs=8.5, bold=True)

    # --- Web & Grad-CAM ---
    band_bg(ax, 0.395, 0.105, "4. WEB & EXPLAINABILITY")
    box(ax, 0.12, 0.415, 0.52, 0.055,
        "Web App — FastAPI backend · React frontend · triage + probabilities",
        fc="#DCFCE7", ec="#16A34A", fs=9.5)
    box(ax, 0.12, 0.335, 0.30, 0.055,
        "Grad-CAM Heatmap\nup to 5 confirmed labels",
        fc="#FEF3C7", ec="#D97706", fs=9)
    arrow(ax, 0.27, 0.335, 0.27, 0.318, color="#D97706", dashed=True)
    ax.text(0.27, 0.310, "from trained model", ha="center", fontsize=7.5, color="#D97706")
    arrow(ax, 0.42, 0.362, 0.55, 0.442, color="#D97706", dashed=True)
    ax.text(0.48, 0.378, "shown on web UI", ha="center", fontsize=7.5, color="#D97706")

    # --- Footer ---
    notes = [
        "Calibration and thresholds are fit on Validation only. Test is never used for parameter selection.",
        "3-tier triage per label: Red = positive, Yellow = uncertain (Δ = 0.08), Green = negative.",
        "Purple = model core · Yellow = Grad-CAM · Green = web product.",
    ]
    ax.add_patch(FancyBboxPatch(
        (0.025, 0.03), 0.95, 0.115, boxstyle="square,pad=0",
        linewidth=0.8, edgecolor="#E5E7EB", facecolor="#FAFAFA", transform=ax.transAxes,
    ))
    for i, note in enumerate(notes):
        ax.text(0.04, 0.125 - i * 0.028, f"• {note}", ha="left", va="top",
                fontsize=8.5, color="#4B5563", transform=ax.transAxes)

    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {OUT} ({OUT.stat().st_size // 1024} KB, {DPI} DPI)")


if __name__ == "__main__":
    export()
