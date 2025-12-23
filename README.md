# Hierarchical-Representation-Learning-for-Drug-MOA-Prediction-
Code for a neural network based approach to learn a hierarchical latent space from gene expression data for the prediction of drug mechanism-of-action


![Hierarchical MoA prediction framework](figures/Figure_overview.pdf)


## Overview

Predicting drug mechanisms of action (MoAs) from transcriptional responses is a central task in drug discovery and repurposing. While many machine learning approaches improve predictive performance, they often lack biological structure and interpretability.

This work introduces a **hierarchical supervised contrastive learning framework** that learns a biologically coherent latent space from gene-expression signatures. The learned representation simultaneously captures:

- **MoA-level separation** (mechanistic organization)
- **Compound-level substructure** (drug-specific effects and dose trajectories)

The model is trained on **LINCS L1000 level 5 transcriptomic perturbation data** and generalizes across unseen drugs, unseen cell types, and even CRISPR knockdown perturbations.

## Key Contributions

- Hierarchical latent space organization using **dual ArcFace objectives**
- Improved MoA prediction performance compared to state-of-the-art baselines
- Robust generalization to:
  - unseen compounds
  - unseen cell lines
  - CRISPR perturbation profiles (without retraining)
- Biological interpretability via:
  - gene-level attribution (Integrated Gradients)
  - pathway enrichment analysis (Reactome)

---

## Data

This project uses transcriptomic perturbation profiles from the **LINCS L1000 project**.

The Level-5 consensus gene-expression signatures used in this work can be downloaded directly from the LINCS CLUE data repository:

```bash
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx


