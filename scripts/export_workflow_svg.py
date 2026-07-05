"""Export a clean, HR-friendly workflow SVG (+ PNG) for the README."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG_OUT = ROOT / "docs" / "workflow.svg"
PNG_OUT = ROOT / "docs" / "workflow.png"

W, H = 1280, 880

C = {
    "bg_band": "#F8FAFC",
    "border_band": "#E2E8F0",
    "box": "#FFFFFF",
    "box_border": "#64748B",
    "model_fill": "#F5F3FF",
    "model_border": "#7C3AED",
    "product_fill": "#ECFDF5",
    "product_border": "#059669",
    "explain_fill": "#FFFBEB",
    "explain_border": "#D97706",
    "pos": "#FEE2E2",
    "pos_b": "#DC2626",
    "unc": "#FEF9C3",
    "unc_b": "#CA8A04",
    "neg": "#DCFCE7",
    "neg_b": "#16A34A",
    "text": "#0F172A",
    "muted": "#64748B",
    "arrow": "#475569",
    "accent": "#7C3AED",
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _box(x, y, w, h, lines, fill, stroke, fs=13, bold=False):
    weight = ' font-weight="600"' if bold else ""
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>',
    ]
    line_h = fs + 5
    if len(lines) == 1:
        ty = y + h / 2 + fs / 3
    else:
        ty = y + h / 2 - line_h / 2 + fs / 2 + 2
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x + w/2}" y="{ty + i * line_h}" text-anchor="middle" '
            f'font-family="Segoe UI, Arial, sans-serif" font-size="{fs}" '
            f'fill="{C["text"]}"{weight}>{_esc(line)}</text>'
        )
    return "\n    ".join(parts)


def _arrow(x1, y1, x2, y2, color=None, dashed=False):
    color = color or C["arrow"]
    dash = ' stroke-dasharray="6,4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.5" marker-end="url(#arrowhead)"{dash}/>'
    )


def _band(y, h, label):
    return (
        f'<rect x="40" y="{y}" width="{W - 80}" height="{h}" fill="{C["bg_band"]}" '
        f'stroke="{C["border_band"]}" rx="6"/>'
        f'<text x="56" y="{y + 24}" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" font-weight="700" fill="{C["muted"]}">{label}</text>'
    )


def build_svg() -> str:
    el: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
        "<defs>",
        '<marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
        f'<polygon points="0 0, 8 3, 0 6" fill="{C["arrow"]}"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        f'<text x="{W/2}" y="40" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="22" font-weight="700" fill="{C["text"]}">'
        f"Chest X-Ray Multi-Label Diagnosis — System Workflow</text>",
        f'<text x="{W/2}" y="64" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="13" fill="{C["muted"]}">'
        f"Training  →  Calibration  →  Inference  →  Web Deployment</text>",
        f'<text x="{W/2}" y="82" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" '
        f'font-size="11" font-style="italic" fill="#94A3B8">'
        f"Clinical decision support — not a substitute for physicians</text>",
    ]

    # 1 · Model Training
    y1, h1 = 100, 150
    el.append(_band(y1, h1, "1 · MODEL TRAINING"))
    bw, bh, gap = 300, 50, 18
    xs = [175, 175 + bw + gap, 175 + 2 * (bw + gap)]

    row1 = [
        (["NIH ChestX-ray14", "train / val / test"], C["box"], C["box_border"]),
        (["Preprocessing", "CLAHE · resize 448"], C["box"], C["box_border"]),
        (["Augmentation", "train set only"], C["box"], C["box_border"]),
    ]
    row2 = [
        (["DenseNet-121", "LSE pooling"], C["model_fill"], C["model_border"]),
        (["14-label head", "sigmoid output"], C["box"], C["box_border"]),
        (["FZLPR loss", "14 labels"], C["box"], C["box_border"]),
    ]
    ry1 = y1 + 36
    for i, (lines, fill, stroke) in enumerate(row1):
        el.append(_box(xs[i], ry1, bw, bh, lines, fill, stroke, fs=12))
        if i < 2:
            el.append(_arrow(xs[i] + bw, ry1 + bh / 2, xs[i + 1], ry1 + bh / 2))
    ry2 = ry1 + bh + 12
    for i, (lines, fill, stroke) in enumerate(row2):
        el.append(_box(xs[i], ry2, bw, bh, lines, fill, stroke, fs=12))
        if i < 2:
            el.append(_arrow(xs[i] + bw, ry2 + bh / 2, xs[i + 1], ry2 + bh / 2))

    el.append(_arrow(W / 2, y1 + h1, W / 2, y1 + h1 + 18, C["accent"]))
    el.append(
        f'<text x="{W/2}" y="{y1 + h1 + 14}" text-anchor="middle" '
        f'font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="600" '
        f'fill="{C["accent"]}">validation logits</text>'
    )

    # 2 · Calibration
    y2, h2 = 275, 108
    el.append(_band(y2, h2, "2 · CALIBRATION (VALIDATION SET)"))
    bw2 = 460
    el.append(_box(175, y2 + 36, bw2, 54,
                   ["Probability calibration", "Temperature Scaling or Isotonic"],
                   C["model_fill"], C["model_border"], fs=12))
    el.append(_box(175 + bw2 + 30, y2 + 36, bw2, 54,
                   ["Threshold tuning", "Youden-J per label"],
                   C["model_fill"], C["model_border"], fs=12))

    # 3 · Inference
    y3, h3 = 400, 230
    el.append(_band(y3, h3, "3 · INFERENCE & TRIAGE"))
    bw3, gap3 = 210, 16
    steps = [
        (["Input", "PNG / JPG"], C["box"], C["box_border"]),
        (["Normalize", "448 · CLAHE"], C["box"], C["box_border"]),
        (["DenseNet-121", "load weights"], C["model_fill"], C["model_border"]),
        (["Calibrate", "apply saved T"], C["box"], C["box_border"]),
        (["Decide", "3-tier output"], C["box"], C["box_border"]),
    ]
    sy = y3 + 36
    for i, (lines, fill, stroke) in enumerate(steps):
        x = 175 + i * (bw3 + gap3)
        el.append(_box(x, sy, bw3, 50, lines, fill, stroke, fs=11))
        if i < 4:
            el.append(_arrow(x + bw3, sy + 25, x + bw3 + gap3, sy + 25))

    tri_y = sy + 78
    tw, tg = 280, 24
    tri = [
        (["POSITIVE", "above threshold"], C["pos"], C["pos_b"]),
        (["UNCERTAIN", "equivocal band"], C["unc"], C["unc_b"]),
        (["NEGATIVE", "below band"], C["neg"], C["neg_b"]),
    ]
    for i, (lines, fill, stroke) in enumerate(tri):
        el.append(_box(175 + i * (tw + tg), tri_y, tw, 50, lines, fill, stroke, fs=12, bold=True))

    # 4 · Web
    y4, h4 = 645, 108
    el.append(_band(y4, h4, "4 · WEB APP & EXPLAINABILITY"))
    el.append(_box(175, y4 + 36, 340, 54,
                   ["Grad-CAM heatmap", "visual explanation"],
                   C["explain_fill"], C["explain_border"], fs=12))
    el.append(_box(535, y4 + 36, 530, 54,
                   ["Web app — FastAPI + React", "upload · predict · display results"],
                   C["product_fill"], C["product_border"], fs=12))
    el.append(_arrow(515, y4 + 63, 535, y4 + 63, C["explain_border"], dashed=True))

    # Footer legend
    el.append(
        f'<rect x="40" y="770" width="{W - 80}" height="88" fill="#FAFAFA" '
        f'stroke="#E2E8F0" rx="6"/>'
    )
    notes = [
        "Purple = model core   ·   Green = web product   ·   Yellow = Grad-CAM explainability",
        "Calibration and thresholds are fit on the validation set only.",
        "Research and educational use — physician review required before clinical decisions.",
    ]
    for i, note in enumerate(notes):
        el.append(
            f'<text x="56" y="{792 + i * 22}" font-family="Segoe UI, Arial, sans-serif" '
            f'font-size="12" fill="{C["muted"]}">{_esc(note)}</text>'
        )

    el.append("</svg>")
    return "\n  ".join(el)


def export_png_from_svg(svg_text: str) -> None:
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg_text.encode("utf-8"), write_to=str(PNG_OUT), scale=2.0)
        print(f"Wrote {PNG_OUT}")
    except ImportError:
        try:
            from PIL import Image
            import io
            import cairosvg  # noqa: F811
        except ImportError:
            print("PNG skipped (install cairosvg for PNG export). SVG is sufficient for README.")
            return


def main() -> None:
    svg = build_svg()
    SVG_OUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {SVG_OUT}")

    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(PNG_OUT), scale=2.0)
        print(f"Wrote {PNG_OUT} ({PNG_OUT.stat().st_size // 1024} KB)")
    except (ImportError, OSError):
        print("PNG skipped (optional). README uses workflow.svg — vector, always sharp on GitHub.")


if __name__ == "__main__":
    main()
