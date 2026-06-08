# -*- coding: utf-8 -*-
"""Sinh Hinh 3.14: nguong duoi va nguong phat hien theo tung nhan (FZLPR)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "configs" / "calibration.json"
OUT = ROOT / "thong_so" / "bieu_do" / "hinh_3_14_nguong_theo_nhan.png"

LABELS = [
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

EQUIVOCAL_BAND = 0.08
MIN_TRI_GAP = 0.02


def low_threshold(high: float) -> float:
    lo = high - EQUIVOCAL_BAND
    lo = max(0.0, min(lo, high - MIN_TRI_GAP))
    return lo


def main() -> None:
    with open(CALIB, encoding="utf-8") as f:
        calib = json.load(f)
    thresholds = calib["thresholds"]

    highs = np.array([float(thresholds[l]) for l in LABELS])
    lows = np.array([low_threshold(h) for h in highs])
    x = np.arange(len(LABELS))

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    for i in range(len(LABELS)):
        ax.plot([x[i], x[i]], [lows[i], highs[i]], color="#F29933", lw=2.2, zorder=2)
    ax.scatter(x, lows, s=58, color="#E45756", zorder=3, label="Nguong duoi")
    ax.scatter(x, highs, s=58, color="#4C78A8", zorder=3, label="Nguong phat hien")
    ax.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2, label="Tham chieu 0,5")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [l.replace("_", " ") for l in LABELS],
        rotation=35,
        ha="right",
        fontsize=9,
    )
    ax.set_ylim(0, max(0.55, highs.max() + 0.06))
    ax.set_ylabel("Nguong xac suat")
    ax.set_title(
        "Nguong duoi va nguong phat hien theo tung nhan (FZLPR, validation)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    plt.close(fig)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
