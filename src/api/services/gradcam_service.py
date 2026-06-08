"""Grad-CAM quality control — moved verbatim from app.py."""
from typing import Dict, Any


def resolve_gradcam_qc(
    item_quality: Dict[str, Any],
    gradcam_cfg: Dict[str, Any],
    disease_label: str = None,
) -> Dict[str, Any]:
    """Evaluate Grad-CAM quality.

    Primary signal is body-vs-background focus.
    Optionally apply disease-aware spatial sanity checks from config.
    """
    qc_cfg = gradcam_cfg.get("quality_gate", {}) if isinstance(gradcam_cfg, dict) else {}
    # Default enabled=False so QC is opt-in: when quality_gate section is absent
    # (clean pipeline with no quality metrics), maps are always shown.
    enabled = bool(qc_cfg.get("enabled", False))
    min_body = max(0.0, min(1.0, float(qc_cfg.get("min_focus_in_body_ratio", 0.60))))

    metrics = {}
    if isinstance(item_quality, dict):
        if isinstance(item_quality.get("final"), dict):
            metrics = item_quality.get("final", {}) or {}
        elif isinstance(item_quality.get("post"), dict):
            metrics = item_quality.get("post", {}) or {}
        elif isinstance(item_quality.get("raw"), dict):
            metrics = item_quality.get("raw", {}) or {}
        else:
            metrics = item_quality

    body = metrics.get("focus_in_body_ratio")
    edges = metrics.get("focus_on_edges_ratio")
    corners = metrics.get("focus_on_top_corners_ratio")
    bottom = metrics.get("focus_on_bottom_ratio")
    numeric_ready = isinstance(body, (int, float))

    reasons = []
    passed = True
    if enabled:
        if not numeric_ready:
            passed = False
            reasons.append("missing_quality_metrics")
        else:
            if float(body) < min_body:
                passed = False
                reasons.append("low_focus_in_body")

            # Disease-aware spatial sanity checks (lightweight heuristics).
            disease_checks = bool(qc_cfg.get("disease_location_checks", True))
            if disease_checks and disease_label:
                lower_focus_labels = set(
                    qc_cfg.get(
                        "lower_focus_labels",
                        # NIH 14 dùng "Effusion" (không phải "Pleural Effusion").
                        ["Effusion", "Edema", "Atelectasis"],
                    )
                )
                upper_focus_labels = set(
                    qc_cfg.get(
                        "upper_focus_labels",
                        ["Pneumothorax"],
                    )
                )
                max_top_for_lower = float(qc_cfg.get("max_top_ratio_for_lower_focus", 0.55))
                min_bottom_for_lower = float(qc_cfg.get("min_bottom_ratio_for_lower_focus", 0.20))
                min_top_for_upper = float(qc_cfg.get("min_top_ratio_for_upper_focus", 0.15))

                top_ok = isinstance(corners, (int, float))
                bottom_ok = isinstance(bottom, (int, float))

                if disease_label in lower_focus_labels and top_ok and bottom_ok:
                    if float(corners) > max_top_for_lower and float(bottom) < min_bottom_for_lower:
                        passed = False
                        reasons.append("unexpected_top_focus_for_lower_lung_disease")

                if disease_label in upper_focus_labels and top_ok:
                    if float(corners) < min_top_for_upper:
                        passed = False
                        reasons.append("insufficient_upper_lung_focus_for_upper_lung_disease")

    score = round(min(1.0, float(body) / max(min_body, 1e-8)), 4) if numeric_ready else 0.0

    return {
        "enabled": enabled,
        "ok": (passed if enabled else True),
        "score": score,
        "reasons": reasons,
        "metrics": {
            "focus_in_body_ratio": (round(float(body), 4) if isinstance(body, (int, float)) else None),
            "focus_on_edges_ratio": (round(float(edges), 4) if isinstance(edges, (int, float)) else None),
            "focus_on_top_corners_ratio": (round(float(corners), 4) if isinstance(corners, (int, float)) else None),
            "focus_on_bottom_ratio": (round(float(bottom), 4) if isinstance(bottom, (int, float)) else None),
        },
        "thresholds": {
            "min_focus_in_body_ratio": min_body,
            "max_top_ratio_for_lower_focus": (
                float(qc_cfg.get("max_top_ratio_for_lower_focus", 0.55))
                if bool(qc_cfg.get("disease_location_checks", True))
                else None
            ),
            "min_bottom_ratio_for_lower_focus": (
                float(qc_cfg.get("min_bottom_ratio_for_lower_focus", 0.20))
                if bool(qc_cfg.get("disease_location_checks", True))
                else None
            ),
        },
    }
