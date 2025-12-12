#!/usr/bin/env python3
"""
Train a hierarchical ArcFace model for Drug MOA prediction using LINCS L1000 data.

Model:
  - Create Encoder
  - ArcFace head for MOA (per-class adaptive margins)
  - ArcFace head for CMAP compound labels

Training:
  - Per holdout fold
  - Early stopping
  - TAS-filtered samples only
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scanpy as sc
from sklearn.preprocessing import LabelEncoder
from pytorch_metric_learning.losses import ArcFaceLoss


# ======================================================
# ArcFace with per-class margins (MOA head)
# ======================================================
class PerClassArcFaceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        margin_vector: torch.Tensor,
        scale: float = 30.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.scale = scale
        self.margin_vector = margin_vector
        self.W = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        W = F.normalize(self.W, p=2, dim=1)

        cosine = torch.matmul(embeddings, W.t())
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        margins = self.margin_vector[labels].to(embeddings.device)
        phi = torch.cos(theta + margins.unsqueeze(1))

        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        logits *= self.scale

        return F.cross_entropy(logits, labels)


# ======================================================
# Encoder network
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
# Hierarchical ArcFace model
# ======================================================
class HierarchicalArcFace(nn.Module):
    def __init__(
        self,
        gene_dim: int,
        embedding_dim: int,
        num_moa: int,
        num_cmap: int,
        margin_moa_vector: torch.Tensor,
        margin_cmap: float,
    ):
        super().__init__()
        self.encoder = RepresentationModel(gene_dim, embedding_dim)
        self.arcface_moa = PerClassArcFaceLoss(
            num_classes=num_moa,
            embedding_size=embedding_dim,
            margin_vector=margin_moa_vector,
        )
        self.arcface_cmap = ArcFaceLoss(
            num_classes=num_cmap,
            embedding_size=embedding_dim,
            margin=margin_cmap,
            scale=30,
        )

    def forward(
        self,
        x: torch.Tensor,
        y_moa: torch.Tensor | None = None,
        y_cmap: torch.Tensor | None = None,
    ):
        embeddings = self.encoder(x)
        loss_moa = self.arcface_moa(embeddings, y_moa) if y_moa is not None else None
        loss_cmap = self.arcface_cmap(embeddings, y_cmap) if y_cmap is not None else None
        return embeddings, loss_moa, loss_cmap


# ======================================================
# Dataset
# ======================================================
class DrugDataset(Dataset):
    def __init__(self, X, y_moa, y_cmap):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_moa = torch.tensor(y_moa, dtype=torch.long)
        self.y_cmap = torch.tensor(y_cmap, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_moa[idx], self.y_cmap[idx]


# ======================================================
# Training loop
# ======================================================
def train_model(
    model,
    dataloader,
    optimizer,
    device,
    epochs=500,
    patience=30,
    lambda_cmap=0.1,
):
    model.train()
    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y_moa, y_cmap in dataloader:
            x, y_moa, y_cmap = x.to(device), y_moa.to(device), y_cmap.to(device)
            optimizer.zero_grad()
            _, loss_moa, loss_cmap = model(x, y_moa, y_cmap)
            loss = loss_moa + lambda_cmap * loss_cmap
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    return best_state


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)

    EMBED_DIM = 256
    BATCH_SIZE = 512
    EPOCHS = 500
    PATIENCE = 30

    easy_classes = [
        "HDAC-i", "TKI", "PI3K-i", "Topo-i", "mTOR-i", "HSP-i",
        "MEK/ERK-i", "CDK-i", "Aurora inh.", "EGFR-i",
        "antimicrotubule", "antimetabolite", "glucocorticoid",
        "antipsychotic", "anthelmintic", "JAK-i",
        "retinoid", "PARP-i", "antibiotic", "cardiac glycoside",
    ]

    print("🚀 Training Hierarchical ArcFace Model")

    for fold in range(1, 8):
        print(f"\n===== Fold {fold} =====")

        adata = sc.read(f"splits/train_TAS_high_fold{fold}.h5ad")
        X = adata.X.astype(np.float32)

        moa_enc = LabelEncoder()
        cmap_enc = LabelEncoder()
        y_moa = moa_enc.fit_transform(adata.obs["MOA"].astype(str))
        y_cmap = cmap_enc.fit_transform(adata.obs["cmap_name"].astype(str))

        margin_moa = torch.full((len(moa_enc.classes_),), 0.25)
        for c in easy_classes:
            if c in moa_enc.classes_:
                margin_moa[moa_enc.classes_ == c] = 0.7
        margin_moa = margin_moa.to(device)

        dataset = DrugDataset(X, y_moa, y_cmap)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = HierarchicalArcFace(
            gene_dim=X.shape[1],
            embedding_dim=EMBED_DIM,
            num_moa=len(moa_enc.classes_),
            num_cmap=len(cmap_enc.classes_),
            margin_moa_vector=margin_moa,
            margin_cmap=0.1,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        best_state = train_model(model, loader, optimizer, device, EPOCHS, PATIENCE)

        save_path = f"models/hierarcface_fold{fold}_tas_high.pth"
        torch.save(best_state, save_path)
        print(f"✅ Saved model: {save_path}")
