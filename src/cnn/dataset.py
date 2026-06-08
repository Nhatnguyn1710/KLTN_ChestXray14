
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BottomCropTransform(A.ImageOnlyTransform):
    """Crop phần dưới ảnh (vùng bụng) theo tỉ lệ."""

    def __init__(self, crop_ratio: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.crop_ratio = crop_ratio

    def apply(self, img, **params):
        if self.crop_ratio <= 0:
            return img
        h = img.shape[0]
        return img[:max(1, int(h * (1.0 - self.crop_ratio))), :]

    def get_transform_init_args_names(self):
        return ("crop_ratio",)


class CornerEraseTransform(A.ImageOnlyTransform):
    """Xóa ngẫu nhiên các góc ảnh để giảm shortcut từ text marker (L/R/...)."""

    def __init__(self, corner_ratio: float = 0.15, p_corner: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.corner_ratio = corner_ratio
        self.p_corner = p_corner

    def apply(self, img, **params):
        h, w = img.shape[:2]
        cy = max(1, int(h * self.corner_ratio))
        cx = max(1, int(w * self.corner_ratio))
        img = img.copy()
        for ys, xs in [
            (slice(0, cy),     slice(0, cx)),
            (slice(0, cy),     slice(w - cx, w)),
            (slice(h - cy, h), slice(0, cx)),
            (slice(h - cy, h), slice(w - cx, w)),
        ]:
            if np.random.random() < self.p_corner:
                img[ys, xs] = float(img[ys, xs].mean()) if img[ys, xs].size > 0 else 0.0
        return img

    def get_transform_init_args_names(self):
        return ("corner_ratio", "p_corner")


# NIH ChestX-ray14 — 14 nhãn.
# Tên CHEXPERT_LABELS giữ nguyên để backward-compat với checkpoint cũ.
# Dùng NIH_LABELS cho code mới.
CHEXPERT_LABELS = [
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
NIH_LABELS = CHEXPERT_LABELS

DEFAULT_NORMALIZATION = {
    "mean": [0.502, 0.502, 0.502],
    "std":  [0.293, 0.293, 0.293],
}


def _resolve_triplet(values, fallback):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return list(fallback)
    out = []
    for idx, item in enumerate(values):
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(float(fallback[idx]))
    return out


def get_normalization_stats(normalization_cfg: dict = None) -> tuple:
    """Trả về (mean, std) cho Normalize từ config."""
    cfg = normalization_cfg if isinstance(normalization_cfg, dict) else {}
    mean = _resolve_triplet(cfg.get("mean"), DEFAULT_NORMALIZATION["mean"])
    std  = _resolve_triplet(cfg.get("std"),  DEFAULT_NORMALIZATION["std"])
    std  = [s if abs(s) > 1e-8 else DEFAULT_NORMALIZATION["std"][i] for i, s in enumerate(std)]
    return mean, std


def encode_view_position(ap_pa) -> float:
    """AP=0.0, PA=1.0, unknown=0.5."""
    if ap_pa == "AP":
        return 0.0
    if ap_pa == "PA":
        return 1.0
    return 0.5


class CheXpertDataset(Dataset):
    """
    Dataset NIH ChestX-ray14 (14 nhãn).

    Tên class giữ là `CheXpertDataset` để backward-compat với checkpoint cũ.
    Alias: `NIHDataset = CheXpertDataset`.

    Hỗ trợ:
    - CLAHE preprocessing (tùy chọn, áp dụng cho tất cả split)
    - Augmentation đa dạng cho training, letterbox resize cho val/test
    - Uncertainty policy cho label -1 (NIH chỉ có 0/1, có thể dùng lại với dataset khác)
    - Optional view position encoding (AP/PA) đưa vào head
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        image_size: int = 224,
        augmentation: bool = False,
        uncertainty_policy: str = "ones",
        dataframe: pd.DataFrame = None,
        aug_cfg: dict = None,
        use_view_position: bool = False,
        bottom_crop_ratio: float = 0.0,
        uncertainty_policy_per_class: dict = None,
        normalization_cfg: dict = None,
    ):
        self.df = dataframe.copy() if dataframe is not None else pd.read_csv(csv_path)
        self.image_root = image_root
        self.image_size = image_size
        self.uncertainty_policy = uncertainty_policy
        self.uncertainty_policy_per_class = uncertainty_policy_per_class or {}
        self.labels = CHEXPERT_LABELS
        self.use_view_position = use_view_position
        self.bottom_crop_ratio = bottom_crop_ratio
        self.normalization_mean, self.normalization_std = get_normalization_stats(normalization_cfg)
        self._augmentation = augmentation
        self._aug_cfg = aug_cfg

        clahe_pre = (aug_cfg or {}).get("clahe_preprocessing", {})
        self.clahe_enabled       = bool(clahe_pre.get("enabled", False))
        self.clahe_clip_limit    = float(clahe_pre.get("clip_limit", 2.0))
        self.clahe_tile_size     = int(clahe_pre.get("tile_size", 8))
        self.clahe_percentile_lo = float(clahe_pre.get("percentile_lo", 1.0))
        self.clahe_percentile_hi = float(clahe_pre.get("percentile_hi", 99.0))

        corner_erase_cfg = (aug_cfg or {}).get("corner_erase", {})
        self.corner_erase_enabled = bool(corner_erase_cfg.get("enabled", True))

        self._process_uncertainty()

        # NIH CSV đã filter frontal sẵn; nếu CSV thô có cột này thì lọc thêm.
        if "Frontal/Lateral" in self.df.columns:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)

        self.transform = self._build_transforms(augmentation, aug_cfg=aug_cfg)

    def set_image_size(self, new_size: int):
        """Cập nhật image_size và rebuild transform (dùng cho progressive resize)."""
        if new_size == self.image_size:
            return
        self.image_size = new_size
        self.transform = self._build_transforms(self._augmentation, aug_cfg=self._aug_cfg)

    def _process_uncertainty(self):
        """Map NaN → 0 và xử lý label -1 theo policy.

        NIH ChestX-ray14 chỉ có 0/1, nên thực tế chỉ map NaN → 0.
        Code path này vẫn hoạt động đúng nếu CSV có -1 (defensive).

        Policies: 'ones' (-1→1), 'zeros' (-1→0), 'ignore' (giữ -1, mask khỏi loss),
        'per_class' (mỗi label dùng policy riêng từ uncertainty_policy_per_class).
        Trong mọi policy, NaN → 0 (implicit negative).
        """
        if self.uncertainty_policy == "per_class" and self.uncertainty_policy_per_class:
            for label in self.labels:
                if label not in self.df.columns:
                    continue
                series = self.df[label].copy()
                series.loc[series.isna()] = 0.0
                uncertain = series == -1.0
                policy = self.uncertainty_policy_per_class.get(label, "zeros")
                if policy == "ignore":
                    series.loc[uncertain] = -1.0
                elif policy == "ones":
                    series.loc[uncertain] = 1.0
                else:
                    series.loc[uncertain] = 0.0
                self.df[label] = series
        else:
            for label in self.labels:
                if label not in self.df.columns:
                    continue
                series = self.df[label].copy()
                series.loc[series.isna()] = 0.0
                uncertain = series == -1.0
                if self.uncertainty_policy == "ignore":
                    series.loc[uncertain] = -1.0
                elif self.uncertainty_policy == "ones":
                    series.loc[uncertain] = 1.0
                else:
                    series.loc[uncertain] = 0.0
                self.df[label] = series

    def _build_transforms(self, augmentation: bool, aug_cfg: dict = None) -> A.Compose:
        bottom_crop_ratio = self.bottom_crop_ratio

        if augmentation:
            rrc = (aug_cfg or {}).get("random_resized_crop", {})
            ssr = (aug_cfg or {}).get("shift_scale_rotate", {})
            gn  = (aug_cfg or {}).get("gauss_noise", {})
            cd  = (aug_cfg or {}).get("coarse_dropout", {})
            gd  = (aug_cfg or {}).get("grid_distortion", {})
            et  = (aug_cfg or {}).get("elastic_transform", {})
            gb  = (aug_cfg or {}).get("gaussian_blur", {})
            hf_cfg = (aug_cfg or {}).get("horizontal_flip", True)
            bc_cfg = (aug_cfg or {}).get("brightness_contrast", True)

            if isinstance(hf_cfg, bool):
                hf_p = 0.5 if hf_cfg else 0.0
            else:
                try:
                    hf_p = float(hf_cfg)
                except (TypeError, ValueError):
                    hf_p = 0.5
                hf_p = max(0.0, min(1.0, hf_p))

            if isinstance(bc_cfg, bool):
                bc_p = 0.4 if bc_cfg else 0.0
            else:
                try:
                    bc_p = float(bc_cfg)
                except (TypeError, ValueError):
                    bc_p = 0.4
                bc_p = max(0.0, min(1.0, bc_p))

            corner_steps = ([CornerEraseTransform(corner_ratio=0.15, p_corner=0.85, p=0.5)]
                            if self.corner_erase_enabled else [])
            return A.Compose([
                BottomCropTransform(crop_ratio=bottom_crop_ratio, p=1.0),
                *corner_steps,
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=129,
                ),
                # ratio=(1,1): crop vuông → không gây distortion giải phẫu.
                A.RandomResizedCrop(
                    size=(self.image_size, self.image_size),
                    scale=(rrc.get("scale_min", 0.7), rrc.get("scale_max", 1.0)),
                    ratio=(1.0, 1.0),
                    p=rrc.get("prob", 0.7),
                ),
                A.HorizontalFlip(p=hf_p),
                A.ShiftScaleRotate(
                    shift_limit=ssr.get("shift_limit", 0.08),
                    scale_limit=ssr.get("scale_limit", 0.15),
                    rotate_limit=ssr.get("rotate_limit", 20),
                    border_mode=0, p=ssr.get("prob", 0.5),
                ),
                A.GridDistortion(
                    num_steps=gd.get("num_steps", 5),
                    distort_limit=gd.get("distort_limit", 0.2),
                    border_mode=0, p=gd.get("prob", 0.2),
                ),
                A.ElasticTransform(
                    alpha=et.get("alpha", 50),
                    sigma=et.get("sigma", 5),
                    border_mode=0, p=et.get("prob", 0.15),
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=bc_p,
                ),
                A.GaussianBlur(
                    blur_limit=gb.get("blur_limit", 3),
                    p=gb.get("prob", 0.2),
                ),
                A.GaussNoise(
                    std_range=(gn.get("std_min", 0.02), gn.get("std_max", 0.1)),
                    p=gn.get("prob", 0.3),
                ),
                A.CoarseDropout(
                    num_holes_range=(cd.get("min_holes", 1), cd.get("max_holes", 6)),
                    hole_height_range=(cd.get("min_ratio", 0.02), cd.get("max_ratio", 0.1)),
                    hole_width_range=(cd.get("min_ratio", 0.02), cd.get("max_ratio", 0.1)),
                    fill=0,
                    p=cd.get("prob", 0.3),
                ),
                A.Normalize(mean=self.normalization_mean, std=self.normalization_std),
                ToTensorV2(),
            ])
        else:
            corner_steps = ([CornerEraseTransform(corner_ratio=0.15, p_corner=1.0, p=1.0)]
                            if self.corner_erase_enabled else [])
            return A.Compose([
                BottomCropTransform(crop_ratio=bottom_crop_ratio, p=1.0),
                *corner_steps,
                A.LongestMaxSize(max_size=self.image_size),
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=129,
                ),
                A.Normalize(mean=self.normalization_mean, std=self.normalization_std),
                ToTensorV2(),
            ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        relative_path = row["Path"]
        img_path = os.path.join(self.image_root, relative_path)

        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        if self.clahe_enabled:
            gray = np.array(image.convert("L"))
            # Percentile windowing trước CLAHE để loại artifact kim loại.
            lo_val, hi_val = np.percentile(gray, [self.clahe_percentile_lo, self.clahe_percentile_hi])
            if hi_val > lo_val:
                gray = np.clip(gray, lo_val, hi_val)
                gray = ((gray - lo_val) / (hi_val - lo_val) * 255.0).astype(np.uint8)
            gray = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
            ).apply(gray)
            image = Image.fromarray(gray)
        image = np.array(image.convert("RGB"))

        image_tensor = self.transform(image=image)["image"]

        raw_labels = torch.tensor(
            [row[label] for label in self.labels],
            dtype=torch.float32,
        )

        # mask=0 cho label -1 (uncertain) → bị ignore trong loss.
        if self.uncertainty_policy in ("ignore", "per_class"):
            mask   = (raw_labels >= 0).float()
            labels = raw_labels.clamp(min=0)
        else:
            mask   = torch.ones_like(raw_labels)
            labels = raw_labels

        result = {
            "image":  image_tensor,
            "labels": labels,
            "mask":   mask,
            "path":   relative_path,
        }

        if self.use_view_position:
            # Thử cả 2 tên cột: NIH dùng "View Position", legacy CheXpert dùng "AP/PA".
            try:
                if "View Position" in row.index:
                    view_raw = row["View Position"]
                elif "AP/PA" in row.index:
                    view_raw = row["AP/PA"]
                else:
                    view_raw = None
                view_val = encode_view_position(view_raw)
            except (KeyError, TypeError):
                view_val = 0.5
            result["view_type"] = torch.tensor(view_val, dtype=torch.float32)

        return result


# Alias cho code mới
NIHDataset = CheXpertDataset


def _compute_patient_strat_key(patient_df: pd.DataFrame, labels: list) -> str:
    """Tạo stratification key cho bệnh nhân dựa trên view + nhãn hiếm.

    Kết quả dạng "AP_PnX1_Con0" dùng cho train_test_split(stratify=...).
    """
    view_col = "View Position" if "View Position" in patient_df.columns else (
        "AP/PA" if "AP/PA" in patient_df.columns else None
    )
    if view_col:
        mode = patient_df[view_col].mode()
        view = str(mode.iloc[0]) if len(mode) > 0 else "AP"
    else:
        view = "UNK"

    pnx = int((patient_df.get("Pneumothorax", pd.Series(dtype=float)) == 1).any())
    con = int((patient_df.get("Consolidation", pd.Series(dtype=float)) == 1).any())

    return f"{view}_PnX{pnx}_Con{con}"


def _stratified_3way_patient_split(
    full_df: pd.DataFrame,
    labels: list,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Chia bệnh nhân thành train/val/test với stratification theo view + nhãn hiếm.

    Trả về (train_patients, val_patients, test_patients).
    Fallback sang random nếu stratum quá nhỏ.
    """
    from collections import Counter

    grouped     = full_df.groupby("patient_id")
    patient_keys = {pid: _compute_patient_strat_key(grp, labels) for pid, grp in grouped}
    patients     = list(patient_keys.keys())
    strat_labels = [patient_keys[p] for p in patients]

    counts    = Counter(strat_labels)
    min_count = min(counts.values()) if counts else 0

    if min_count < 3:
        view_only = {}
        for pid, grp in grouped:
            col = "AP/PA" if "AP/PA" in grp.columns else None
            view_only[pid] = str(grp[col].mode().iloc[0]) if col and len(grp[col].mode()) > 0 else "UNK"
        strat_labels = [view_only[p] for p in patients]
        counts    = Counter(strat_labels)
        min_count = min(counts.values()) if counts else 0
        if min_count < 3:
            strat_labels = None
            print("  [split] Fallback to random split (strata too small)")
        else:
            print(f"  [split] Fallback to view-only stratification: {dict(counts)}")
    else:
        print(f"  [split] Stratification: {len(counts)} strata, min={min_count}")

    kw = {"stratify": strat_labels} if strat_labels is not None else {}
    trainval_patients, test_patients = train_test_split(
        patients, test_size=test_ratio, random_state=seed, **kw
    )

    if strat_labels is not None:
        trainval_strat = [patient_keys[p] for p in trainval_patients]
        if min(Counter(trainval_strat).values()) < 2:
            trainval_strat = None
    else:
        trainval_strat = None

    kw2 = {"stratify": trainval_strat} if trainval_strat is not None else {}
    val_frac = val_ratio / (1.0 - test_ratio)
    train_patients, val_patients = train_test_split(
        trainval_patients, test_size=val_frac, random_state=seed, **kw2
    )

    return train_patients, val_patients, test_patients


def build_dataloaders(config: dict) -> tuple:
    """Tạo (train_loader, val_loader, test_loader, valid_loader) từ config.

    NIH mode (mặc định, dataset_mode="nih"):
        Đọc 3 CSV đã split sẵn từ prepare_nih_splits.py.

    Legacy mode (dataset_mode!="nih"):
        Tự split 70/15/15 từ paths.train_csv theo bệnh nhân.
        paths.valid_csv (được đưa ra nếu có) → external validation loader.

    valid_loader = None khi không có paths.valid_csv.
    """
    paths   = config["paths"]
    cnn_cfg = config["cnn"]
    seed    = config["general"]["seed"]
    is_nih  = config.get("dataset_mode", "nih").lower() == "nih" or (
                  "val_csv" in paths and "test_csv" in paths
              )

    if is_nih:
        train_df        = pd.read_csv(paths["train_csv"])
        val_internal_df = pd.read_csv(paths["val_csv"])
        test_df         = pd.read_csv(paths["test_csv"])
        print(f"  [NIH] Loaded CSVs: train={len(train_df):,}  val={len(val_internal_df):,}  test={len(test_df):,}")
        view_col = (
            "View Position" if "View Position" in train_df.columns
            else "AP/PA" if "AP/PA" in train_df.columns else None
        )
        if view_col:
            for name, df in [("train", train_df), ("val", val_internal_df), ("test", test_df)]:
                vc = df[view_col].value_counts()
                n  = len(df)
                print(f"    {name} {view_col}: " + ", ".join(f"{k}={v:,} ({v/n*100:.1f}%)" for k, v in vc.items()))
        for lbl in ["Pneumothorax", "Consolidation", "Hernia"]:
            if lbl in train_df.columns:
                rates = [f"{name}={(df[lbl]==1).mean()*100:.1f}%"
                         for name, df in [("train", train_df), ("val", val_internal_df), ("test", test_df)]]
                print(f"    {lbl}: {', '.join(rates)}")
    else:
        full_train_df = pd.read_csv(paths["train_csv"])
        full_train_df["patient_id"] = full_train_df["Path"].apply(lambda p: p.split("/")[2])
        split_cfg = cnn_cfg.get("split", {})
        train_patients, val_patients, test_patients = _stratified_3way_patient_split(
            full_train_df, CHEXPERT_LABELS,
            val_ratio=split_cfg.get("val_ratio", 0.15),
            test_ratio=split_cfg.get("test_ratio", 0.15),
            seed=seed,
        )
        train_df        = full_train_df[full_train_df["patient_id"].isin(train_patients)].copy()
        val_internal_df = full_train_df[full_train_df["patient_id"].isin(val_patients)].copy()
        test_df         = full_train_df[full_train_df["patient_id"].isin(test_patients)].copy()
        for df in [train_df, val_internal_df, test_df]:
            df.drop(columns=["patient_id"], inplace=True)
        split_dir = os.path.join(os.path.dirname(paths["train_csv"]), "splits")
        os.makedirs(split_dir, exist_ok=True)
        train_df.to_csv(os.path.join(split_dir, "train_split.csv"), index=False)
        val_internal_df.to_csv(os.path.join(split_dir, "val_internal_split.csv"), index=False)
        test_df.to_csv(os.path.join(split_dir, "test_split.csv"), index=False)
        print(f"  [legacy] 3-way split: {len(train_patients)} train / "
              f"{len(val_patients)} val / {len(test_patients)} test patients")

    aug_cfg  = cnn_cfg.get("augmentation", {})
    norm_cfg = cnn_cfg.get("normalization", {})
    use_vp   = cnn_cfg.get("use_view_position", False)
    bcr      = aug_cfg.get("bottom_crop_ratio", 0.0)
    upc      = cnn_cfg.get("uncertainty_policy_per_class", None)
    up       = cnn_cfg.get("uncertainty_policy", "zeros")
    bs       = cnn_cfg["training"]["batch_size"]
    nw       = cnn_cfg["training"]["num_workers"]
    img_size = cnn_cfg["image_size"]
    dd       = paths["dataset_dir"]

    def _make_dataset(df, csv_path=None, augment=False):
        return CheXpertDataset(
            csv_path=csv_path,
            image_root=dd,
            image_size=img_size,
            augmentation=augment,
            uncertainty_policy=up,
            dataframe=df,
            aug_cfg=aug_cfg if augment else None,
            use_view_position=use_vp,
            bottom_crop_ratio=bcr,
            uncertainty_policy_per_class=upc,
            normalization_cfg=norm_cfg,
        )

    train_dataset        = _make_dataset(train_df, augment=True)
    val_internal_dataset = _make_dataset(val_internal_df)
    test_dataset         = _make_dataset(test_df)

    use_weighted_sampler = cnn_cfg["training"].get("weighted_sampler", False)
    if use_weighted_sampler:
        labels_np = train_dataset.df[CHEXPERT_LABELS].values
        pos_count = (labels_np == 1).sum(axis=0).astype(float) + 1
        valid_cnt = ((labels_np == 0) | (labels_np == 1)).sum(axis=0).astype(float) + 1
        inv_freq  = valid_cnt / pos_count
        # Sample weight = max inv_freq để tránh bias khi một mẫu có nhiều nhãn hiếm cùng lúc.
        sample_weights = torch.tensor([
            float(inv_freq[labels_np[i] == 1.0].max()) if (labels_np[i] == 1.0).any() else 1.0
            for i in range(len(labels_np))
        ], dtype=torch.float64)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print(f"  Weighted sampler: ON (min_w={sample_weights.min():.2f}, max_w={sample_weights.max():.2f})")
        train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler,
                                  num_workers=nw, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True)

    val_internal_loader = DataLoader(val_internal_dataset, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)
    test_loader         = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                                     num_workers=nw, pin_memory=True)

    valid_loader = None
    if paths.get("valid_csv"):
        valid_ds     = _make_dataset(None, csv_path=paths["valid_csv"])
        valid_loader = DataLoader(valid_ds, batch_size=bs, shuffle=False,
                                  num_workers=nw, pin_memory=True)

    return train_loader, val_internal_loader, test_loader, valid_loader
