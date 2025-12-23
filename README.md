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

## Usage

This section describes the **exact workflow** to reproduce the data preprocessing, training, and evaluation used in the paper.

The pipeline consists of four main steps:

1. Download LINCS L1000 Level-5 data  
2. Preprocess data into an AnnData file  
3. Create TAS-filtered cell-line holdout splits  
4. Train and evaluate the hierarchical ArcFace model  

---

## Step 1: Download LINCS L1000 Level-5 Data

Download the Level-5 consensus signatures (`trt_cp`) from the CLUE data repository:

```bash
wget https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx
```

Move the file to the project data directory:
```bash
mkdir -p data
mv level5_beta_trt_cp_n720216x12328.gctx data/
```



The full pipeline consists of four steps: **data preprocessing**, **split generation**, **model training**, and **evaluation**.

---

### Step 1: Data Preprocessing

Run:

```bash
python 01_data_preproc.py
```
The processed dataset is saved as: level_5_lincs_all_genes.h5ad


### Step 2: TAS-Filtered Cell-Line Holdout Splits
```bash

python 02_tas_high_splits.py \
  --adata level_5_lincs_all_genes.h5ad \
  --tas-threshold 0.2 \
  --holdout-cells A549 NCIH508 MDAMB231 LNCAP \
  --outdir splits
```
The output consists of train/test AnnData files:
splits/
├── train_TAS_high_fold4.h5ad
├── test_TAS_high_fold4.h5ad
├── train_TAS_high_fold5.h5ad
├── test_TAS_high_fold5.h5ad
...

### Step 3: Model Training

```bash
python 03_train_hirarch_MOA_model.py
```

A separate model is trained for each fold. Trained models are saved as:
models/
├── hierarcface_fold1_tas_high.pth
├── hierarcface_fold2_tas_high.pth
...

### Step 4: Model Evaluation

Evaluate MoA prediction using k-nearest neighbors in the learned latent space:
```bash
python 04_test_hirarch_MOA_model.py \
  --folds 4 5 6 7 \
  --k 1 \
  --out results/per_moa_accuracy.csv
```

The per-MoA results are optionally saved as: results/per_moa_accuracy.csv





