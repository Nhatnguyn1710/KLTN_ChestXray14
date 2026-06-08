"""Predict pipeline — CNN classification + optional Grad-CAM."""
import os
import csv
import uuid
from typing import Dict, Any, Optional

import torch

from src.api import state
from src.api.services.gradcam_service import resolve_gradcam_qc
from src.cnn.dataset import CHEXPERT_LABELS
from src.cnn.inference import (
    preprocess_image,
    predict,
    get_per_class_thresholds,
    get_per_class_low_thresholds,
    _min_tri_gap_from_config,
    _get_temperature,
    _get_per_label_temperatures,
    _get_per_label_isotonic,
)
from src.cnn.grad_cam import generate_gradcam_for_top_diseases

_NIH_FILENAME_METADATA_CACHE = None


def _format_nih_view_position(view_code: str) -> str:
    code = str(view_code or "").strip().upper()
    if code == "AP":
        return "AP (trước-sau)"
    if code == "PA":
        return "PA (sau-trước)"
    if code == "LATERAL":
        return "Lateral (nghiêng)"
    return "Chưa xác định"


def _load_nih_filename_metadata(config: dict) -> Dict[str, Dict[str, Any]]:
    global _NIH_FILENAME_METADATA_CACHE
    if _NIH_FILENAME_METADATA_CACHE is not None:
        return _NIH_FILENAME_METADATA_CACHE

    paths_cfg = config.get("paths", {}) if isinstance(config, dict) else {}
    csv_paths = [
        paths_cfg.get("train_csv"),
        paths_cfg.get("val_csv"),
        paths_cfg.get("test_csv"),
    ]
    index: Dict[str, Dict[str, Any]] = {}
    for csv_path in csv_paths:
        if not csv_path or not os.path.isfile(csv_path):
            continue
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = row.get("Image Index") or os.path.basename(row.get("Path", ""))
                    filename = os.path.basename(str(filename or "").strip())
                    if not filename:
                        continue
                    view_code = str(row.get("View Position", "")).strip().upper()
                    if not view_code:
                        continue
                    index[filename] = {
                        "view_position": _format_nih_view_position(view_code),
                        "view": "Frontal" if view_code in {"AP", "PA"} else view_code,
                        "ap_pa": view_code if view_code in {"AP", "PA"} else "",
                        "rel_path": row.get("Path", ""),
                        "csv_path": csv_path,
                        "match_method": "nih_filename_csv",
                    }
        except Exception as exc:
            print(f"  [Metadata] NIH CSV lookup skipped for {csv_path}: {exc}")

    _NIH_FILENAME_METADATA_CACHE = index
    print(f"  [Metadata] NIH filename index loaded ({len(index):,} images)")
    return _NIH_FILENAME_METADATA_CACHE


def _lookup_nih_metadata(config: dict, original_filename: str) -> Optional[Dict[str, Any]]:
    filename = os.path.basename(str(original_filename or "").strip())
    if not filename:
        return None
    return _load_nih_filename_metadata(config).get(filename)


def _is_unknown_view(image_meta: Dict[str, Any]) -> bool:
    view_text = str(image_meta.get("view_position", "")).strip().lower()
    return not view_text or "chưa xác định" in view_text or "chua xac dinh" in view_text

def run_predict_pipeline(
    filepath: str,
    image_filename: str,
    fallback_filename: str,
    threshold: float,
    do_gradcam: bool,
    view_position: str,
    input_w: int,
    input_h: int,
) -> Dict[str, Any]:
    """Synchronous prediction pipeline (run inside asyncio.to_thread)."""
    config = state.config
    device = state.device
    cnn_model = state.cnn_model
    metadata_index = state.metadata_index

    # --- CNN Prediction ---
    per_class_thresholds = get_per_class_thresholds(config)
    # Slider shifts all class thresholds around a base point (0.5):
    # - if slider < 0.5 -> model becomes more sensitive
    # - if slider > 0.5 -> model becomes more conservative
    slider_delta = float(threshold) - state.DEFAULT_THRESHOLD_BASE
    threshold_floor = float(
        config.get("web", {}).get(
            "threshold_floor",
            config.get("cnn", {}).get("calibration", {}).get(
                "min_threshold", state.THRESHOLD_FLOOR
            ),
        )
    )
    threshold_floor = max(0.0, min(0.95, threshold_floor))
    min_tri_gap = float(_min_tri_gap_from_config(config))
    per_class_low_thresholds = get_per_class_low_thresholds(config, per_class_thresholds)
    effective_thresholds = {}
    effective_low_thresholds = {}
    for label in CHEXPERT_LABELS:
        base_thr = float(per_class_thresholds.get(label, state.DEFAULT_THRESHOLD_BASE))
        raw_effective = base_thr + slider_delta
        # Floor applied AFTER delta so slider can't bypass it
        eff_hi = min(0.95, max(threshold_floor, raw_effective))
        base_low = float(per_class_low_thresholds.get(label, base_thr - 0.08))
        raw_low = base_low + slider_delta
        eff_lo = max(0.0, min(0.95, raw_low))
        eff_lo = min(eff_lo, eff_hi - min_tri_gap)
        eff_lo = max(0.0, eff_lo)
        effective_thresholds[label] = eff_hi
        effective_low_thresholds[label] = eff_lo
    temperature = _get_temperature(config)
    per_label_temperatures = _get_per_label_temperatures(config)
    per_label_isotonic = _get_per_label_isotonic(config)
    normalization_cfg = config.get("cnn", {}).get("normalization", {})
    clahe_cfg = config.get("cnn", {}).get("augmentation", {}).get("clahe_preprocessing", {})
    bottom_crop_ratio = float(
        config.get("cnn", {}).get("augmentation", {}).get("bottom_crop_ratio", 0.0)
    )

    # Resolve image metadata + AP/PA hint early so inference can use view_type.
    image_meta = None
    if metadata_index is not None:
        image_meta = metadata_index.lookup(filepath, image_filename)
        print(f"  [Metadata] lookup({filepath}, {image_filename}) -> {image_meta}")
    else:
        print("  [Metadata] WARNING: metadata_index is None!")
    if image_meta is None:
        image_meta = {}
    image_meta.setdefault("original_filename", image_filename or fallback_filename)

    if view_position == "auto" and _is_unknown_view(image_meta):
        nih_meta = _lookup_nih_metadata(config, image_filename or fallback_filename)
        if nih_meta:
            image_meta.update(nih_meta)
            image_meta.setdefault("original_filename", image_filename or fallback_filename)
            print(f"  [Metadata] NIH filename lookup({image_filename}) -> {nih_meta}")

    # User-selected view position overrides auto-detection
    _vp_map = {
        "PA": "PA (sau-trước)",
        "AP": "AP (trước-sau)",
        "Lateral": "Lateral (nghiêng)",
    }
    if view_position != "auto" and view_position in _vp_map:
        image_meta["view_position"] = _vp_map[view_position]
        image_meta["match_method"] = "user_selected"

    view_type_value = state.resolve_view_type_value(view_position)
    if view_type_value is None:
        inferred_code = state.extract_view_code(image_meta.get("view_position", ""))
        view_type_value = state.resolve_view_type_value(inferred_code)
    view_type_tensor = (
        torch.tensor([view_type_value], dtype=torch.float32)
        if view_type_value is not None else None
    )

    image_tensor = preprocess_image(
        filepath,
        config["cnn"]["image_size"],
        bottom_crop_ratio=bottom_crop_ratio,
        normalization_cfg=normalization_cfg,
        clahe_cfg=clahe_cfg,
    )
    result = predict(
        cnn_model,
        image_tensor,
        device,
        threshold=threshold,
        per_class_thresholds=effective_thresholds,
        temperature=temperature,
        view_type=view_type_tensor,
        per_label_temperatures=per_label_temperatures if per_label_temperatures else None,
        per_label_isotonic=per_label_isotonic if per_label_isotonic else None,
    )

    labels_vi = config.get("labels_vi", {})

    # NIH 14 labels: tất cả đều là bệnh lý, không có nhãn "No Finding".
    abnormal_detected = [
        label for label, is_pos in result["predictions"].items() if is_pos
    ]
    has_abnormal = len(abnormal_detected) > 0

    def _tri_grade_for_prob(prob_val: float, t_lo: float, t_hi: float) -> str:
        if prob_val >= t_hi:
            return "positive"
        if prob_val >= t_lo:
            return "equivocal"
        return "negative"

    # Format classification results
    classifications = []
    for label, prob in sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    ):
        label_thr = float(effective_thresholds.get(label, threshold))
        label_thr_lo = float(effective_low_thresholds.get(label, label_thr - 0.08))
        prob_val = float(prob)
        detected = bool(result["predictions"].get(label, False))
        tri_grade = _tri_grade_for_prob(prob_val, label_thr_lo, label_thr)

        classifications.append({
            "label": label,
            "label_vi": labels_vi.get(label, ""),
            "probability": round(prob_val, 4),
            "threshold": round(label_thr, 4),
            "threshold_low": round(label_thr_lo, 4),
            "margin": round(prob_val - label_thr, 4),
            "detected": detected,
            "tri_grade": tri_grade,
            "is_abnormal": True,
            "borderline": False,
        })

    # --- Top-N confirmation: demote detections beyond top-N (by margin) to borderline ---
    _max_confirmed = int(config.get("cnn", {}).get("max_confirmed_detections", 5))
    _det_abnormals = sorted(
        [c for c in classifications if c["detected"] and c.get("is_abnormal", True)],
        key=lambda c: c["margin"],
        reverse=True,
    )
    _confirmed_labels = {c["label"] for c in _det_abnormals[:_max_confirmed]}
    for c in classifications:
        if c["detected"] and c.get("is_abnormal", True) and c["label"] not in _confirmed_labels:
            c["detected"] = False
            c["borderline"] = True
            # Vượt t_high nhưng bị giới hạn top-N → chuyển sang nghi ngờ (không còn “phát hiện” chắc chắn)
            if c.get("tri_grade") == "positive":
                c["tri_grade"] = "equivocal"
    # Update abnormal_detected so GradCAM only runs on confirmed labels
    abnormal_detected = [lbl for lbl in abnormal_detected if lbl in _confirmed_labels]
    has_abnormal = len(abnormal_detected) > 0

    tri_labels_vi = {
        "negative": "Âm tính",
        "equivocal": "Nghi ngờ",
        "positive": "Phát hiện",
    }
    for c in classifications:
        g = c.get("tri_grade", "negative")
        c["tri_label_vi"] = tri_labels_vi.get(g, g)

    grades = [c.get("tri_grade", "negative") for c in classifications]
    if any(g == "positive" for g in grades):
        study_triage = "positive"
    elif any(g == "equivocal" for g in grades):
        study_triage = "equivocal"
    else:
        study_triage = "negative"
    study_triage_vi = {
        "positive": "Phát hiện bất thường",
        "equivocal": "Cần xem xét thêm (vùng trung gian)",
        "negative": "Không phát hiện bất thường rõ",
    }.get(study_triage, study_triage)

    response = {
        "classifications": classifications,
        "threshold": threshold,
        "threshold_delta": round(slider_delta, 4),
        "threshold_mode": "per_class_with_slider_delta",
        "effective_thresholds": {
            label: round(float(thr), 4)
            for label, thr in effective_thresholds.items()
        },
        "effective_low_thresholds": {
            label: round(float(thr), 4)
            for label, thr in effective_low_thresholds.items()
        },
        "study_triage": study_triage,
        "study_triage_vi": study_triage_vi,
        "positive_count": sum(1 for c in classifications if c.get("tri_grade") == "positive"),
        "equivocal_count": sum(1 for c in classifications if c.get("tri_grade") == "equivocal"),
        "negative_count": sum(1 for c in classifications if c.get("tri_grade") == "negative"),
        "detected_count": sum(
            1 for c in classifications
            if c["detected"] and c.get("is_abnormal", True)
        ),
        "gradcam_url": None,
        "gradcam_label": None,
        "gradcam_prob": None,
        "gradcam_items": [],
        "image_width": int(input_w) if input_w else None,
        "image_height": int(input_h) if input_h else None,
        "image_resolution_note": None,
    }

    # --- Grad-CAM ---
    gc_results = []
    if do_gradcam:
        try:
            gradcam_cfg = config.get("gradcam", {})
            gradcam_top_k = int(gradcam_cfg.get("top_k", 3))
            gradcam_exclude = set(gradcam_cfg.get("exclude_labels", []))

            # Every confirmed detected abnormal label gets GradCAM
            # (trừ những nhãn được liệt kê tường minh trong config.gradcam.exclude_labels).
            gradcam_target_labels = [
                label for label in abnormal_detected
                if label not in gradcam_exclude
            ]
            # Sort by probability descending — most confident first.
            gradcam_target_labels.sort(
                key=lambda l: float(result["probabilities"].get(l, 0)),
                reverse=True,
            )
            gradcam_target_labels = gradcam_target_labels[:gradcam_top_k]

            if not gradcam_target_labels:
                response["gradcam_note"] = (
                    "Khong co nhan bat thuong du do chac chan de ve Grad-CAM."
                )

            gc_results = generate_gradcam_for_top_diseases(
                cnn_model, filepath, result["probabilities"],
                config,
                top_k=gradcam_top_k,
                threshold=threshold,
                exclude_labels=list(gradcam_exclude),
                target_labels=gradcam_target_labels,
                view_type=view_type_tensor,
            )
            if gc_results:
                import cv2
                items = []
                qc_passed_count = 0
                for item in gc_results:
                    gc_filename = f"gradcam_{uuid.uuid4().hex}.png"
                    gc_path = os.path.join(state.GRADCAM_FOLDER, gc_filename)
                    heatmap_bgr = cv2.cvtColor(
                        item["heatmap"], cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(gc_path, heatmap_bgr)
                    qc = resolve_gradcam_qc(
                        item.get("quality", {}),
                        gradcam_cfg,
                        disease_label=item["label"],
                    )
                    print(f"  [GradCAM QC] {item['label']}: ok={qc.get('ok')}, "
                          f"metrics={qc.get('metrics')}, reasons={qc.get('reasons')}")
                    if qc.get("ok", False):
                        qc_passed_count += 1

                    # Zone-rule soft check (informational, never hides maps)
                    zone_rules = gradcam_cfg.get("quality_gate", {}).get("zone_rules", {})
                    primary_zone = item.get("primary_zone", "")
                    primary_fraction = round(float(item.get("primary_fraction", 0.0)), 3)
                    expected_zones = zone_rules.get(item["label"], [])
                    zone_match = (not expected_zones) or (primary_zone in expected_zones)
                    if not zone_match:
                        print(f"  [GradCAM Zone] {item['label']}: primary={primary_zone} "
                              f"({primary_fraction:.0%}) — not in expected {expected_zones}")

                    items.append({
                        "url": f"/static/gradcam/{gc_filename}",
                        "label": item["label"],
                        "probability": round(item["probability"], 4),
                        "quality": item.get("quality", {}),
                        "qc": qc,
                        "zone_activations": item.get("zone_activations", {}),
                        "primary_zone": primary_zone,
                        "primary_fraction": primary_fraction,
                        "zone_match": zone_match,
                        "weak_label_warning": item.get("weak_label_warning"),
                    })

                # Prefer showing QC-passed map first (better UX).
                items.sort(
                    key=lambda x: (
                        0 if x.get("qc", {}).get("ok", False) else 1,
                        -float(x.get("probability", 0.0)),
                    )
                )

                hide_failed = bool(
                    gradcam_cfg.get("quality_gate", {}).get("hide_failed_maps", False)
                )
                if hide_failed:
                    passed_items = [x for x in items if x.get("qc", {}).get("ok", False)]
                    if passed_items:
                        items = passed_items
                    else:
                        response["gradcam_note"] = (
                            "Grad-CAM map co chat luong thap cho anh nay; vui long chi tham khao ket qua phan loai."
                        )

                response["gradcam_items"] = items
                response["gradcam_quality"] = {
                    "enabled": bool(gradcam_cfg.get("quality_gate", {}).get("enabled", True)),
                    "passed_count": qc_passed_count,
                    "total_count": len(items),
                    "all_passed": qc_passed_count == len(items),
                }
                # Backward compatibility for old frontend keys
                response["gradcam_url"] = items[0]["url"]
                response["gradcam_label"] = items[0]["label"]
                response["gradcam_prob"] = items[0]["probability"]
        except Exception as e:
            response["gradcam_error"] = str(e)

    # Add metadata to response for frontend display
    response["image_metadata"] = {
        "original_filename": image_meta.get("original_filename", ""),
        "view_position": image_meta.get("view_position", "Chưa xác định"),
        "match_method": image_meta.get("match_method", "none"),
    }

    return response
