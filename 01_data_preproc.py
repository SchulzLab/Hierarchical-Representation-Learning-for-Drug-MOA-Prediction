import os
import pandas as pd
import scanpy as sc
import anndata as ad
import joblib  # For loading the saved scaler
from scipy.stats import rankdata, norm
import numpy as np
from sklearn.preprocessing import StandardScaler
import gzip
from cmapPy.pandasGEXpress.parse import parse
import anndata
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import plotly.express as px



current_directory = os.getcwd()
# Get the parent directory
project_directory = os.path.dirname(os.path.dirname(current_directory))

# Get the path to the subdirectory from the parent directory
path_to_project_results = os.path.join(project_directory, 'results/')
path_to_project_data = os.path.join(project_directory, 'data/')



decompressed_file_path = path_to_project_data + 'level5_beta_trt_cp_n720216x12328.gctx'

# Parse the decompressed .gctx file
gctx_data = parse(decompressed_file_path)

# Convert the GCTX data to
#  a Pandas DataFrame
lincs_data = gctx_data.data_df




## read info for data
info = pd.read_csv(path_to_project_data + 'siginfo_beta.txt', delimiter='\t')


info['sig_id'] = pd.Categorical(info['sig_id'], categories=list(lincs_data.columns), ordered=True)
info = info.sort_values('sig_id')
info.index = info.sig_id
info = info.drop('sig_id', axis=1)



only_landmark_gened = False

### We do not need the inferred genes only the landmark (978)
landmark_genes = pd.read_csv(path_to_project_data + 'geneinfo_beta.txt', delimiter='\t')
if only_landmark_gened:
    landmark_genes = landmark_genes[landmark_genes.feature_space == 'landmark'] ## drop inferred genes
else:
    landmark_genes.index = landmark_genes.gene_symbol

lincs_data.index = lincs_data.index.astype(int)


## keep only landmark genes
lincs_data = lincs_data.loc[landmark_genes.gene_id]
lincs_data = lincs_data.T

# Replace gene ids with gene names
rename_dict = dict(zip(landmark_genes['gene_id'], landmark_genes.index))
lincs_data.rename(columns=rename_dict, inplace=True)

lincs_data_f = lincs_data[lincs_data.index.isin(info.index)]
common_indices = lincs_data_f.index.intersection(info.index)

lincs_data_f = lincs_data_f.loc[common_indices]
info_f = info.loc[common_indices]


adata = anndata.AnnData(X=lincs_data_f, obs=info_f, var=landmark_genes)

compound_info = pd.read_csv(path_to_project_data + 'compoundinfo_beta.txt', delimiter='\t')
compound_info = compound_info.drop_duplicates(subset='pert_id')
compound_info_clean = compound_info.dropna(subset=['canonical_smiles']).drop_duplicates(subset='pert_id')

# Step 2: Keep only those pert_ids in the AnnData
valid_pert_ids = compound_info_clean['pert_id'].unique()
adata_filtered = adata[adata.obs['pert_id'].isin(valid_pert_ids)].copy()



# Step 3: Merge compound info into obs (all remaining pert_ids are guaranteed valid)
adata_filtered.obs = adata_filtered.obs.merge(
    compound_info_clean[['pert_id', 'moa', 'canonical_smiles']],
    on='pert_id',
    how='left'
)


MOA_info = pd.read_excel('Huang_MOA_label.xlsx', header=1)
MOA_info.rename(columns={'BRD-ID': 'pert_id'}, inplace=True)

adata_filtered.obs['pert_id'] = adata_filtered.obs['pert_id'].astype(str).str.strip()
MOA_info['pert_id'] = MOA_info['pert_id'].astype(str).str.strip()

duplicates = MOA_info[MOA_info.duplicated()]


moa_dict = dict(zip(MOA_info['pert_id'], MOA_info['MOA']))
adata_filtered.obs['MOA'] = adata_filtered.obs['pert_id'].map(moa_dict)


## at least 10 samples per drugs
pert_cell_counts = adata_filtered.obs.pert_id.value_counts()

valid_pert_ids = pert_cell_counts[pert_cell_counts >= 10].index
adata_filtered = adata_filtered[
    adata_filtered.obs['pert_id'].isin(valid_pert_ids) ,:]

adata_filtered.var = adata_filtered.var.drop(columns='gene_symbol')


### filter for low quality samples
adata_filtered = adata_filtered[adata_filtered.obs['is_hiq'] == 1]


## keep only samples with 6 and 24 hours 
adata_filtered = adata_filtered[adata_filtered.obs['pert_time'].isin([6,24])]

## IGNORE DRUGS WHERE SMILES IS RESTRICTED
adata_filtered = adata_filtered[adata_filtered.obs['canonical_smiles'] != 'restricted']

### keep only samples where moa is not NaN
adata_filtered = adata_filtered[adata_filtered.obs['MOA'].notna(), :]

# Group by MOA and count unique pert_id per MOA
moa_to_drugs = adata_filtered.obs.groupby('MOA')['pert_id'].nunique()
# Keep MOAs with at least 3 unique pert_ids
valid_moas = moa_to_drugs[moa_to_drugs >= 10].index
# Filter the data to only those MOAs
adata_filtered = adata_filtered[adata_filtered.obs['MOA'].isin(valid_moas)]

if only_landmark_gened:
    print('save file only with landmark genes')
    adata_filtered.write("level_5_lincs.h5ad")
else:
    print('save file with all genes')
    adata_filtered.write("level_5_lincs_all_genes.h5ad")


# cell_lines = ['MCF7', 'A375', 'PC3', 'HT29', 'A549', 'HA1E']

# adata_filtered = adata_filtered[adata_filtered.obs.cell_iname.isin(cell_lines)]

