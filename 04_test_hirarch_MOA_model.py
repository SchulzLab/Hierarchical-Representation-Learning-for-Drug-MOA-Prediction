#!/usr/bin/env python3
"""
Evaluate trained embedding model via kNN MOA classification.

Workflow:
  1) Load train/test .h5ad for each fold
  2) Load a trained encoder checkpoint
  3) Compute embeddings for train and test
  4) Fit kNN (cosine) on train embeddings and evaluate on test
  5) Report overall and per-MOA accuracy (+ optional CSV export)

Example:
  python 04_test_hirarch_MOA_model.py \
    --folds 4 5 6 7 \
    --model-template models/nn_arcface_high_fold{fold}_tas_high.pth \
    --out per_moa_accuracy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ======================================================
# Dataset (X only; y is handled separately)
# ======================================================
class DrugDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.X[idx]


# ======================================================
# Encoder (must match training architecture)
# ======================================================
class RepresentationModel(nn.Module):
    def __init__(self, gene_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ======================================================
# Embedding computation
# ======================================================
@torch.no_grad()
def compute_embeddings(model: nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for x in dataloader:
        x = x.to(device)
        emb = model(x).detach().cpu()
        chunks.append(emb)
    return torch.cat(chunks, dim=0).numpy()


# ======================================================
# Fold evaluation
# ======================================================
def evaluate_fold(
    fold: int,
    embedding_dim: int,
    device: torch.device,
    k: int,
    train_template: str,
    test_template: str,
    model_template: str,
    batch_size: int,
) -> tuple[float, pd.DataFrame]:
    train_path = Path(train_template.format(fold=fold))
    test_path = Path(test_template.format(fold=fold))
    model_path = Path(model_template.format(fold=fold))

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"\n===== Fold {fold} =====")
    print(f"Train: {train_path}")
    print(f"Test : {test_path}")
    print(f"Model: {model_path}")

    adata_train = sc.read(str(train_path))
    adata_test = sc.read(str(test_path))

    # Ensure consistent MOA category mapping across train/test
    all_moa = pd.concat([adata_train.obs["MOA"], adata_test.obs["MOA"]]).astype("category")
    categories = all_moa.cat.categories

    adata_train.obs["MOA"] = adata_train.obs["MOA"].astype(pd.CategoricalDtype(categories=categories))
    adata_test.obs["MOA"] = adata_test.obs["MOA"].astype(pd.CategoricalDtype(categories=categories))

    y_train = adata_train.obs["MOA"].cat.codes.to_numpy()
    y_test = adata_test.obs["MOA"].cat.codes.to_numpy()

    X_train = adata_train.X.astype(np.float32)
    X_test = adata_test.X.astype(np.float32)

    dl_train = DataLoader(DrugDataset(X_train), batch_size=batch_size, shuffle=False)
    dl_test = DataLoader(DrugDataset(X_test), batch_size=batch_size, shuffle=False)

    model = RepresentationModel(gene_dim=X_train.shape[1], embedding_dim=embedding_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    train_emb = compute_embeddings(model, dl_train, device)
    test_emb = compute_embeddings(model, dl_test, device)

    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    knn.fit(train_emb, y_train)
    pred = knn.predict(test_emb)

    acc = accuracy_score(y_test, pred)
    print(f"Accuracy (k={k}): {acc:.4f}")

    per_moa = []
    for i, moa_name in enumerate(categories):
        mask = (y_test == i)
        moa_acc = accuracy_score(y_test[mask], pred[mask]) if mask.any() else np.nan
        per_moa.append({"Fold": fold, "MOA": str(moa_name), "Accuracy": moa_acc, "n_test": int(mask.sum())})
    df_per_moa = pd.DataFrame(per_moa)

    return acc, df_per_moa


# ======================================================
# CLI
# ======================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MOA classification via kNN on learned embeddings.")
    p.add_argument("--folds", nargs="+", type=int, required=True, help="Fold IDs to evaluate (e.g., 4 5 6 7).")
    p.add_argument("--k", type=int, default=5, help="k for kNN (cosine).")
    p.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension used in training.")
    p.add_argument("--batch-size", type=int, default=1000, help="Batch size for embedding computation.")
    p.add_argument("--train-template", type=str, default="splits/train_TAS_high_fold{fold}.h5ad")
    p.add_argument("--test-template", type=str, default="splits/test_TAS_high_fold{fold}.h5ad")
    p.add_argument("--model-template", type=str, default="models/nn_arcface_high_fold{fold}_tas_high.pth")
    p.add_argument("--out", type=str, default=None, help="Optional CSV path to save per-MOA accuracies.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    fold_accs = []

    for fold in args.folds:
        acc, df_per_moa = evaluate_fold(
            fold=fold,
            embedding_dim=args.embedding_dim,
            device=device,
            k=1,
            train_template=args.train_template,
            test_template=args.test_template,
            model_template=args.model_template,
            batch_size=args.batch_size,
        )
        fold_accs.append(acc)
        all_rows.append(df_per_moa)

    mean_acc = float(np.mean(fold_accs))
    print("\n===== Summary =====")
    for fold, acc in zip(args.folds, fold_accs):
        print(f"Fold {fold}: {acc:.4f}")
    print(f"Mean accuracy: {mean_acc:.4f}")

    df_all = pd.concat(all_rows, ignore_index=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(out_path, index=False)
        print(f"Saved per-MOA results to: {out_path}")


if __name__ == "__main__":
    main()
