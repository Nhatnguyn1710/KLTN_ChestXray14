
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

NIH_LABELS = [
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

RARE_LABELS = ["Hernia", "Pneumonia", "Fibrosis", "Emphysema", "Edema"]


def parse_labels(finding_str: str) -> dict:
    findings = set(finding_str.split("|"))
    return {lbl: int(lbl in findings) for lbl in NIH_LABELS}


def prepare_splits(
    data_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    data_dir = Path(data_dir)
    csv_path = data_dir / "Data_Entry_2017.csv"
    train_val_txt = data_dir / "train_val_list.txt"
    test_txt = data_dir / "test_list.txt"

    for p in [csv_path, train_val_txt, test_txt]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df):,}")

    # Lọc frontal
    df = df[df["View Position"].isin(["PA", "AP"])].copy()
    print(f"  After frontal filter (PA+AP): {len(df):,}")

    # Parse labels
    label_cols = pd.DataFrame(
        df["Finding Labels"].apply(parse_labels).tolist(),
        index=df.index,
    )
    df = pd.concat([df, label_cols], axis=1)

    # Scan image directories
    print(f"\nScanning images in {data_dir} (may take ~30s) ...")
    image_index = {}
    for img_path in data_dir.rglob("*.png"):
        image_index[img_path.name] = str(img_path.relative_to(data_dir).as_posix())
    print(f"  Found {len(image_index):,} PNG files.")

    missing = [img for img in df["Image Index"] if img not in image_index]
    if missing:
        print(f"  WARNING: {len(missing)} images not found on disk. Example: {missing[:3]}")

    df["Path"] = df["Image Index"].map(image_index)
    df = df[df["Path"].notna()].copy()
    print(f"  After path resolution: {len(df):,} images")

    # Official split
    with open(train_val_txt) as f:
        train_val_set = set(f.read().splitlines())
    with open(test_txt) as f:
        test_set = set(f.read().splitlines())

    df_test = df[df["Image Index"].isin(test_set)].copy()
    df_trainval = df[df["Image Index"].isin(train_val_set)].copy()
    print(f"\n  Official test set:     {len(df_test):,} images")
    print(f"  Train+val pool:        {len(df_trainval):,} images")

    # Patient-level split on train_val pool
    patients = df_trainval["Patient ID"].unique()
    print(f"  Unique patients in train+val: {len(patients):,}")

    # Stratify by rare label composite key
    patient_rare = df_trainval.groupby("Patient ID")[RARE_LABELS].max()
    rare_key = patient_rare.reindex(patients).fillna(0).astype(int)
    rare_key_str = rare_key.apply(lambda r: "".join(r.astype(str)), axis=1).values
    key_counts = Counter(rare_key_str)
    rare_key_str = np.array([k if key_counts[k] >= 2 else "others" for k in rare_key_str])

    try:
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=seed,
            stratify=rare_key_str,
        )
    except ValueError:
        print("  WARNING: composite stratify failed, falling back to Hernia-only stratify")
        hernia_flag = patient_rare["Hernia"].reindex(patients).fillna(0).astype(int).values
        train_patients, val_patients = train_test_split(
            patients,
            test_size=val_ratio,
            random_state=seed,
            stratify=hernia_flag,
        )

    train_patients_set = set(train_patients)
    val_patients_set = set(val_patients)

    df_train = df_trainval[df_trainval["Patient ID"].isin(train_patients_set)].copy()
    df_val = df_trainval[df_trainval["Patient ID"].isin(val_patients_set)].copy()

    # Kiểm tra không overlap patient
    assert len(set(df_train["Patient ID"]) & set(df_val["Patient ID"])) == 0, "Train/Val patient overlap!"
    assert len(set(df_train["Patient ID"]) & set(df_test["Patient ID"])) == 0, "Train/Test patient overlap!"
    assert len(set(df_val["Patient ID"]) & set(df_test["Patient ID"])) == 0, "Val/Test patient overlap!"
    print("\n  Patient-level split verified — no overlap.")

    # Kiểm tra rare labels trong val
    print(f"\n  [Stratify check] Rare label counts in VAL split:")
    for lbl in RARE_LABELS:
        n = df_val[lbl].sum()
        print(f"    {lbl:<22}: {int(n):>4} images")

    # Output columns
    out_cols = ["Path", "Image Index", "Patient ID", "View Position"] + NIH_LABELS

    splits_dir = data_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    df_train[out_cols].to_csv(splits_dir / "train.csv", index=False)
    df_val[out_cols].to_csv(splits_dir / "val.csv", index=False)
    df_test[out_cols].to_csv(splits_dir / "test.csv", index=False)

    print(f"\n  Splits saved to: {splits_dir}")
    print(f"    train.csv : {len(df_train):,} images, {len(train_patients):,} patients")
    print(f"    val.csv   : {len(df_val):,} images,   {len(val_patients):,} patients")
    print(f"    test.csv  : {len(df_test):,} images")

    print(f"\n  Label distribution (Train):")
    for lbl in NIH_LABELS:
        n = df_train[lbl].sum()
        pct = 100.0 * n / len(df_train)
        print(f"    {lbl:<22}: {int(n):>6}  ({pct:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NIH ChestX-ray14 splits")
    parser.add_argument("--data_dir", default="D:/archive", help="Path to NIH archive root")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_splits(
        data_dir=args.data_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
