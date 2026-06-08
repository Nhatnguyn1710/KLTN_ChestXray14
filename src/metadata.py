
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def _md5_file(filepath: str, chunk_size: int = 65536) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _csv_path_to_disk(csv_path: str, image_root: str) -> str:
    """
    Convert CSV Path to actual disk path.
    CSV:  CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
    Disk: {image_root}/train/patient00001/study1/view1_frontal.jpg
    """
    # Strip the "CheXpert-v1.0-small/" prefix
    parts = csv_path.replace("\\", "/").split("/")
    # Find 'train' or 'valid' anchor
    for i, p in enumerate(parts):
        if p in ("train", "valid"):
            rel = "/".join(parts[i:])
            return os.path.join(image_root, rel)
    # Fallback: just join everything after first component
    return os.path.join(image_root, "/".join(parts[1:]))


def _format_view_position(frontal_lateral: str, ap_pa: str) -> str:
    """
    Build a human-readable view position string.
    Examples:
        Frontal + PA  → "PA (Posteroanterior)"
        Frontal + AP  → "AP (Anteroposterior)"
        Lateral + NaN → "Lateral"
    """
    fl = str(frontal_lateral).strip() if pd.notna(frontal_lateral) else ""
    ap = str(ap_pa).strip() if pd.notna(ap_pa) else ""

    if fl == "Lateral":
        return "Lateral (nghiêng)"
    if ap == "PA":
        return "PA (sau-trước)"
    if ap == "AP":
        return "AP (trước-sau)"
    if ap in ("LL", "RL"):
        return f"{ap} ({'nghiêng trái' if ap == 'LL' else 'nghiêng phải'})"
    if fl == "Frontal":
        return "Frontal (thẳng, chưa rõ AP/PA)"
    return "Chưa xác định"


def _filename_heuristic(filename: str) -> str:
    """Fallback: guess view position from filename text."""
    lower = filename.lower()
    if "lateral" in lower:
        return "Lateral (nghiêng, suy luận từ tên file)"
    if "frontal" in lower:
        return "Frontal (thẳng, suy luận từ tên file - chưa rõ AP/PA)"
    return "Chưa xác định"


# ---------------------------------------------------------------------------
# ImageMetadataIndex
# ---------------------------------------------------------------------------

class ImageMetadataIndex:
    """
    In-memory MD5 → metadata index for CheXpert images.

    Usage:
        idx = ImageMetadataIndex("data/image_metadata_index.json")
        meta = idx.lookup("/path/to/upload.jpg")
    """

    def __init__(self, index_path: str = "data/image_metadata_index.json"):
        self._index: Dict[str, Dict[str, Any]] = {}
        self._index_path = index_path
        if os.path.isfile(index_path):
            self._load(index_path)

    # ------ load / save ------

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self._index = json.load(f)
        print(f"  [Metadata] Loaded index: {len(self._index):,} entries from {path}")

    def _save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓ Saved index: {len(self._index):,} entries → {path} ({size_mb:.1f} MB)")

    @property
    def size(self) -> int:
        return len(self._index)

    # ------ build ------

    def build(
        self,
        csv_paths: List[str],
        image_root: str = "archive",
        save_path: Optional[str] = None,
    ) -> int:
        """
        Build MD5 → metadata index from CSV files.

        Args:
            csv_paths: list of CSV paths (train.csv, valid.csv)
            image_root: root dir containing train/ and valid/ folders
            save_path: where to save JSON (default: self._index_path)

        Returns:
            Number of indexed images.
        """
        save_path = save_path or self._index_path
        print(f"\n{'='*60}")
        print(f"  Building Image Metadata Index")
        print(f"{'='*60}")

        # Load all CSVs
        dfs = []
        for csv_p in csv_paths:
            df = pd.read_csv(csv_p)
            dfs.append(df)
            print(f"  Loaded {csv_p}: {len(df):,} rows")
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"  Total CSV rows: {len(df_all):,}")

        # Build index
        t0 = time.time()
        indexed = 0
        skipped = 0
        missing = 0

        for i, row in df_all.iterrows():
            csv_path = row["Path"]
            disk_path = _csv_path_to_disk(csv_path, image_root)

            if not os.path.isfile(disk_path):
                missing += 1
                continue

            try:
                md5 = _md5_file(disk_path)
            except Exception:
                skipped += 1
                continue

            # Extract relative path (patient.../study.../view...)
            parts = csv_path.replace("\\", "/").split("/")
            for j, p in enumerate(parts):
                if p.startswith("patient"):
                    rel_path = "/".join(parts[j:])
                    break
            else:
                rel_path = os.path.basename(csv_path)

            self._index[md5] = {
                "csv_path": csv_path,
                "rel_path": rel_path,
                "view": str(row.get("Frontal/Lateral", "")),
                "ap_pa": str(row.get("AP/PA", "")) if pd.notna(row.get("AP/PA")) else "",
                "view_position": _format_view_position(
                    row.get("Frontal/Lateral"), row.get("AP/PA")
                ),
                "sex": str(row.get("Sex", "")) if pd.notna(row.get("Sex")) else "",
                "age": int(row["Age"]) if pd.notna(row.get("Age")) else None,
            }
            indexed += 1

            # Progress
            if (i + 1) % 20000 == 0:
                elapsed = time.time() - t0
                speed = indexed / elapsed if elapsed > 0 else 0
                print(f"    [{i+1:,}/{len(df_all):,}] indexed={indexed:,} ({speed:.0f} img/s)")

        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed:.1f}s: indexed={indexed:,}, missing={missing:,}, skipped={skipped:,}")

        self._save(save_path)
        return indexed

    # ------ lookup ------

    def lookup(
        self,
        image_path: str,
        original_filename: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up metadata for an image file.

        Args:
            image_path: path to the actual image file on disk
            original_filename: original filename from upload (for heuristic fallback)

        Returns:
            dict with keys: view_position, view, ap_pa, sex, age, rel_path, csv_path, match_method
            or None if not found and no heuristic available.
        """
        # --- Strategy 1: MD5 hash match (exact) ---
        if os.path.isfile(image_path):
            try:
                md5 = _md5_file(image_path)
                print(f"  [Metadata.lookup] path={image_path}, md5={md5}, index_size={len(self._index)}, found={md5 in self._index}")
                if md5 in self._index:
                    result = dict(self._index[md5])
                    result["match_method"] = "hash_exact"
                    return result
            except Exception as e:
                print(f"  [Metadata.lookup] MD5 error: {e}")
        else:
            print(f"  [Metadata.lookup] file not found: {image_path}")

        # --- Strategy 2: Filename heuristic ---
        fname = original_filename or os.path.basename(image_path)
        view_hint = _filename_heuristic(fname)
        if view_hint != "Chưa xác định":
            return {
                "view_position": view_hint,
                "view": "Lateral" if "lateral" in fname.lower() else "Frontal",
                "ap_pa": "",
                "sex": "",
                "age": None,
                "rel_path": "",
                "csv_path": "",
                "match_method": "filename_heuristic",
            }

        # --- Not found ---
        return None

    def get_view_position(
        self,
        image_path: str,
        original_filename: Optional[str] = None,
    ) -> str:
        """Convenience: return just the view position string, or 'Chưa xác định'."""
        meta = self.lookup(image_path, original_filename)
        if meta:
            return meta.get("view_position", "Chưa xác định")
        return "Chưa xác định"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build / query image metadata index")
    parser.add_argument("--build", action="store_true", help="Build index from CSV + images")
    parser.add_argument("--config", default="configs/config.yaml", help="Config YAML path")
    parser.add_argument("--image-root", default="archive", help="Root dir with train/ and valid/")
    parser.add_argument("--output", default="data/image_metadata_index.json", help="Output index path")
    parser.add_argument("--query", type=str, default=None, help="Query: lookup metadata for an image file")
    args = parser.parse_args()

    if args.build:
        csv_paths = []
        for name in ["archive/train.csv", "archive/valid.csv"]:
            if os.path.isfile(name):
                csv_paths.append(name)
        if not csv_paths:
            print("ERROR: No CSV files found in archive/")
            sys.exit(1)

        idx = ImageMetadataIndex(args.output)
        idx.build(csv_paths, image_root=args.image_root, save_path=args.output)

    elif args.query:
        idx = ImageMetadataIndex(args.output)
        if idx.size == 0:
            print("ERROR: Index is empty. Run --build first.")
            sys.exit(1)
        meta = idx.lookup(args.query)
        if meta:
            print(json.dumps(meta, ensure_ascii=False, indent=2))
        else:
            print("Not found. View position: Chưa xác định")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
