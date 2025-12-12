#!/usr/bin/env python3
"""
Create cell-line holdout train/test splits from a LINCS AnnData dataset.

This script:
  1) Loads an .h5ad dataset (landmark or all genes)
  2) Filters by TAS score
  3) For each specified cell line, creates a split where that cell line is 100% test
  4) Saves train/test .h5ad files per split

"""

from __future__ import annotations

import argparse
from pathlib import Path
import anndata as ad


def load_adata(adata_path: Path) -> ad.AnnData:
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")
    return ad.read_h5ad(str(adata_path))


def filter_by_tas(adata: ad.AnnData, tas_threshold: float) -> ad.AnnData:
    if "tas" not in adata.obs.columns:
        raise KeyError("`tas` column not found in adata.obs")
    return adata[adata.obs["tas"] > tas_threshold].copy()


def split_by_cell_holdout(adata: ad.AnnData, holdout_cells: list[str]) -> dict[str, tuple[ad.AnnData, ad.AnnData]]:
    if "cell_iname" not in adata.obs.columns:
        raise KeyError("`cell_iname` column not found in adata.obs")

    splits: dict[str, tuple[ad.AnnData, ad.AnnData]] = {}
    present_cells = set(adata.obs["cell_iname"].astype(str).unique())

    for cell in holdout_cells:
        if cell not in present_cells:
            raise ValueError(f"Holdout cell '{cell}' not found in dataset. Available examples: {sorted(list(present_cells))[:10]} ...")

        test = adata[adata.obs["cell_iname"] == cell].copy()
        train = adata[adata.obs["cell_iname"] != cell].copy()
        splits[cell] = (train, test)

    return splits


def summarize_split(train: ad.AnnData, test: ad.AnnData) -> dict[str, int]:
    if "pert_id" not in train.obs.columns or "pert_id" not in test.obs.columns:
        raise KeyError("`pert_id` column not found in adata.obs")
    return {
        "train_samples": train.n_obs,
        "test_samples": test.n_obs,
        "train_unique_drugs": train.obs["pert_id"].nunique(),
        "test_unique_drugs": test.obs["pert_id"].nunique(),
    }


def save_split(
    fold_id: int,
    train: ad.AnnData,
    test: ad.AnnData,
    outdir: Path,
    prefix: str = "TAS_high",
) -> tuple[Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    train_path = outdir / f"train_{prefix}_fold{fold_id}.h5ad"
    test_path = outdir / f"test_{prefix}_fold{fold_id}.h5ad"
    train.write(str(train_path))
    test.write(str(test_path))
    return train_path, test_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create cell-line holdout splits from an AnnData .h5ad dataset.")
    p.add_argument("--adata", type=Path, required=True, help="Path to input .h5ad file.")
    p.add_argument("--tas-threshold", type=float, default=0.2, help="Filter: keep samples with tas > threshold.")
    p.add_argument("--holdout-cells", nargs="+", required=True, help="Cell lines (cell_iname) to hold out as test, one split per cell.")
    p.add_argument("--fold-start", type=int, default=4, help="Fold index to start numbering from (default: 4).")
    p.add_argument("--outdir", type=Path, default=Path("splits"), help="Output directory for split files.")
    p.add_argument("--prefix", type=str, default="TAS_high", help="Filename prefix for saved splits.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adata = load_adata(args.adata)
    adata = filter_by_tas(adata, args.tas_threshold)

    splits = split_by_cell_holdout(adata, args.holdout_cells)

    fold_id = args.fold_start
    for cell, (train, test) in splits.items():
        stats = summarize_split(train, test)
        train_path, test_path = save_split(fold_id, train, test, args.outdir, prefix=args.prefix)

        print(f"[Fold {fold_id}] Holdout cell: {cell}")
        print(f"  Train: {stats['train_samples']} samples | {stats['train_unique_drugs']} unique drugs")
        print(f"  Test : {stats['test_samples']} samples | {stats['test_unique_drugs']} unique drugs")
        print(f"  Saved: {train_path}")
        print(f"         {test_path}\n")

        fold_id += 1


if __name__ == "__main__":
    main()
