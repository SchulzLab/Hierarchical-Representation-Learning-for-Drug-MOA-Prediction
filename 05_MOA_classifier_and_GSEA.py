#!/usr/bin/env python3
"""
Integrated Gradients gene importance + optional GSEA + visualization.

For each fold and each MOA in MOA_LIST:
  1) Load train/test .h5ad
  2) Create binary labels for "MOA vs rest"
  3) Load pretrained encoder weights from a HierArcFace checkpoint
  4) Freeze encoder, train a small binary head
  5) Compute Integrated Gradients feature importance on test
  6) Save ranked genes to CSV
  7) (Optional) run GSEA (prerank) and save significant pathways
  8) Make heatmaps and gene plots from saved results

Example:
  python 05_MOA_classifier_and_GSEA.py \
    --folds 2 \
    --moas "Topo-i" "HSP-i" "CDK-i" "mTOR-i" "HDAC-i" "PI3K-i" "EGFR-i" "JAK-i" \
    --train-template splits/train_TAS_high_fold{fold}.h5ad \
    --test-template  splits/test_TAS_high_fold{fold}.h5ad \
    --model-template models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth \
    --outdir results/ig \
    --do-gsea

Notes:
  - This assumes your pretrained checkpoint contains an encoder under keys:
      "encoder.*" or "encoder.encoder.*"
  - Integrated Gradients can be slow; consider running per MOA selectively.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from captum.attr import IntegratedGradients

# Optional dependency (only needed if --do-gsea)
try:
    import gseapy as gp
except Exception:
    gp = None

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ======================================================
# Dataset
# ======================================================
class DrugDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ======================================================
# Frozen encoder + trainable binary head
# ======================================================
class EncoderWithBinaryHead(nn.Module):
    def __init__(self, gene_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(gene_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
        )
        self.binary_head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor):
        emb = self.encoder(x)
        logits = self.binary_head(emb)
        return logits, emb


# ======================================================
# Helpers
# ======================================================
def set_seed(seed: int = 47) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_encoder(model: EncoderWithBinaryHead, ckpt_path: Path, device: torch.device) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)

    # Extract encoder weights from different possible key prefixes
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.encoder."):
            new_state[k.replace("encoder.encoder.", "")] = v
        elif k.startswith("encoder."):
            new_state[k.replace("encoder.", "")] = v

    missing, unexpected = model.encoder.load_state_dict(new_state, strict=False)
    if missing:
        print(f"⚠️ Missing encoder keys (ok if architecture differs): {missing[:5]} ...")
    if unexpected:
        print(f"⚠️ Unexpected encoder keys: {unexpected[:5]} ...")


def train_binary_head(
    model: EncoderWithBinaryHead,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
) -> EncoderWithBinaryHead:
    model.to(device)

    # Only train head
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.binary_head.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam(model.binary_head.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        n = 0

        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.squeeze(1), y)
            loss.backward()
            opt.step()

            total += loss.item() * x.size(0)
            n += x.size(0)

        epoch_loss = total / max(n, 1)
        print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def _ensure_device(model: nn.Module, device: torch.device) -> None:
    model.to(device)


def integrated_gradients_importance(
    model: EncoderWithBinaryHead,
    dataloader: DataLoader,
    device: torch.device,
    gene_names: list[str],
) -> pd.DataFrame:
    model.eval()
    _ensure_device(model, device)

    # Captum needs a callable returning logits
    ig = IntegratedGradients(lambda x: model(x)[0])

    attrs = []
    for x, _ in dataloader:
        x = x.to(device)
        x.requires_grad_(True)
        # target=0 for single-logit output
        attr = ig.attribute(x, target=0)
        attrs.append(attr.detach().cpu().numpy())

    attrs = np.concatenate(attrs, axis=0)
    importance = np.mean(np.abs(attrs), axis=0)

    df = pd.DataFrame({"Gene": list(gene_names), "Importance": importance})
    return df.sort_values("Importance", ascending=False)


def make_binary_labels(obs_moa: pd.Series, moa: str) -> np.ndarray:
    # Safer match: exact substring match can cause weird overlaps; keep contains but avoid regex surprises
    # (e.g. "Topo-i" doesn't contain regex chars, but keep regex=False)
    return obs_moa.astype(str).str.contains(moa, regex=False).astype(int).to_numpy()


def run_prerank_gsea(
    prerank_df: pd.DataFrame,
    outdir: Path,
    gene_sets: str = "Reactome_2022",
    permutation_num: int = 500,
    seed: int = 47,
):
    if gp is None:
        raise ImportError("gseapy is not installed, but --do-gsea was set. Install gseapy first.")

    outdir.mkdir(parents=True, exist_ok=True)

    res = gp.prerank(
        rnk=prerank_df,
        gene_sets=gene_sets,
        outdir=str(outdir),
        permutation_num=permutation_num,
        min_size=5,
        max_size=100,
        seed=seed,
        threads=4,
        verbose=False,
    )
    df = res.res2d.copy()
    return df


def plot_pathway_heatmap(pivot_df: pd.DataFrame, moa_colors: dict[str, str], outpath: Path) -> None:
    # Identify MOA-specific pathways where NES > 1 in only one MOA column
    threshold = 1.0
    significant = pivot_df > threshold
    moa_specific = significant.sum(axis=1) == 1
    pivot_df_plot = pivot_df[moa_specific].copy()

    if pivot_df_plot.empty:
        print("⚠️ No MOA-specific pathways found with the chosen threshold.")
        return

    row_colors = pivot_df_plot.index.map(lambda term: moa_colors.get(pivot_df_plot.loc[term].idxmax(), "lightgray"))
    fig_h = max(8, len(pivot_df_plot) * 0.3)

    g = sns.clustermap(
        pivot_df_plot,
        cmap="Blues",
        linewidths=0.5,
        row_cluster=False,
        col_cluster=False,
        xticklabels=True,
        yticklabels=True,
        row_colors=row_colors,
        figsize=(12, fig_h),
        cbar_pos=None,
    )

    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    legend_patches = [mpatches.Patch(color=c, label=m) for m, c in moa_colors.items()]
    g.figure.legend(handles=legend_patches, title="MOA", loc="center left", bbox_to_anchor=(0, 0.5))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_top_genes_bar(genes_csv: Path, moa: str, fold: int, top_n: int, outpath: Path) -> None:
    df = pd.read_csv(genes_csv).sort_values("Importance", ascending=False).head(top_n)
    if df.empty:
        return

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Gene", data=df, palette="Blues_r")
    plt.title(f"Top {top_n} Genes for {moa} (Fold {fold})")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================
# CLI
# ======================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated Gradients + (optional) GSEA pipeline.")
    p.add_argument("--folds", nargs="+", type=int, required=True)
    p.add_argument("--moas", nargs="+", type=str, required=True)

    p.add_argument("--train-template", type=str, default="splits/train_TAS_high_fold{fold}.h5ad")
    p.add_argument("--test-template", type=str, default="splits/test_TAS_high_fold{fold}.h5ad")
    p.add_argument("--model-template", type=str, default="models/nn_hierarcface_high_fold{fold}_tas_high_adaptive_margin.pth")

    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--outdir", type=Path, default=Path("results/ig"))
    p.add_argument("--do-gsea", action="store_true")
    p.add_argument("--gene-sets", type=str, default="Reactome_2022")
    p.add_argument("--gsea-perms", type=int, default=500)
    p.add_argument("--seed", type=int, default=47)

    p.add_argument("--top-pathways", type=int, default=100, help="Top N pathways per MOA for heatmap.")
    p.add_argument("--top-genes", type=int, default=10, help="Top N genes per MOA for barplots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    all_gsea_results: list[pd.DataFrame] = []

    for fold in args.folds:
        print(f"\n=== Fold {fold} ===")

        train_path = Path(args.train_template.format(fold=fold))
        test_path = Path(args.test_template.format(fold=fold))
        ckpt_path = Path(args.model_template.format(fold=fold))

        adata_train = sc.read(str(train_path))
        adata_test = sc.read(str(test_path))
        adata_train.var_names_make_unique()
        adata_test.var_names_make_unique()

        X_train = adata_train.X.astype(np.float32)
        X_test = adata_test.X.astype(np.float32)
        gene_names = list(adata_test.var_names)

        for moa in args.moas:
            print(f"\nMOA: {moa}")

            y_train = make_binary_labels(adata_train.obs["MOA"], moa)
            y_test = make_binary_labels(adata_test.obs["MOA"], moa)

            train_loader = DataLoader(DrugDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(DrugDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

            model = EncoderWithBinaryHead(gene_dim=X_train.shape[1], embedding_dim=args.embedding_dim)
            load_pretrained_encoder(model, ckpt_path, device)

            model = train_binary_head(
                model,
                train_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                patience=args.patience,
            )

            feat_df = integrated_gradients_importance(model, test_loader, device, gene_names)

            genes_out = args.outdir / f"important_genes_fold{fold}_{moa}.csv"
            feat_df.to_csv(genes_out, index=False)
            print(f"  ✅ Saved gene importances: {genes_out}")

            # Optional GSEA
            if args.do_gsea:
                prerank_df = feat_df[["Gene", "Importance"]]
                gsea_dir = args.outdir / "gsea" / f"fold{fold}" / moa
                gsea_df = run_prerank_gsea(
                    prerank_df,
                    outdir=gsea_dir,
                    gene_sets=args.gene_sets,
                    permutation_num=args.gsea_perms,
                    seed=args.seed,
                )

                sig = gsea_df[gsea_df["FDR q-val"] < 0.05].copy()
                sig["MOA"] = moa
                sig["Fold"] = fold
                all_gsea_results.append(sig)

                sig_path = gsea_dir / "significant_pathways.csv"
                sig.to_csv(sig_path, index=False)
                print(f"  ✅ Saved significant GSEA pathways: {sig_path}")

            # Bar plot of top genes
            bar_out = args.outdir / "plots" / f"top_genes_fold{fold}_{moa}.pdf"
            plot_top_genes_bar(genes_out, moa, fold, args.top_genes, bar_out)

    # Save merged GSEA results
    if args.do_gsea and len(all_gsea_results) > 0:
        merged = pd.concat(all_gsea_results, ignore_index=True)
        merged_path = args.outdir / "MOA_latent_GSEA_results_all_folds.csv"
        merged.to_csv(merged_path, index=False)
        print(f"\n✅ Saved merged GSEA results: {merged_path}")

        # Heatmap for the first fold requested
        fold_vis = args.folds[0]
        fold_df = merged[merged["Fold"] == fold_vis].copy()
        if not fold_df.empty:
            fold_df = fold_df.sort_values(["MOA", "NES"], ascending=[True, False])
            top_paths_per_moa = fold_df.groupby("MOA").head(args.top_pathways)

            pivot = top_paths_per_moa.pivot(index="Term", columns="MOA", values="NES").fillna(0.0)
            pivot["max_NES"] = pivot.max(axis=1)
            pivot = pivot.sort_values("max_NES", ascending=False).drop(columns="max_NES")

            moa_colors = {
                "Topo-i": "green",
                "HSP-i": "purple",
                "CDK-i": "blue",
                "mTOR-i": "red",
                "HDAC-i": "brown",
                "PI3K-i": "pink",
                "EGFR-i": "orange",
                "JAK-i": "cyan",
            }

            heat_out = args.outdir / "plots" / f"MOA_specific_pathways_fold{fold_vis}.pdf"
            plot_pathway_heatmap(pivot, moa_colors, heat_out)
            print(f"✅ Saved MOA-specific pathway heatmap: {heat_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
