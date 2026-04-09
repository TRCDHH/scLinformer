import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# ====================== Configuration ======================

attn_path = "/home/output/adamson/SOCS1_delta_attn.npy" # Path to saved delta attention matrix (shape: [N+1, N+1], including CLS token)
csv_path = "/home/output/adamson/SOCS1_delta.csv" # Path to CSV file containing gene importance scores
h5ad_path = "/home/data/adamson/control.h5ad"  # Path to AnnData object (.h5ad) containing gene expression data
perturb_gene = "SOCS1"  # The perturbed gene used as the reference node in the heatmap
top_n = 20  # Number of top genes selected for visualization
save_path = "/home/output/adamson/SOCS1_heatmap_abs.png"    # Output path for saving the heatmap figure

# ====================== Step 1: Load Data ======================

delta_attn = np.load(attn_path)
# Load attention difference matrix [N+1, N+1]

adata = sc.read_h5ad(h5ad_path)
gene_names = list(adata.var['gene_name'])
# Extract gene names from AnnData object

# ====================== Step 2: Select Top Genes ======================

df = pd.read_csv(csv_path)
df = df.sort_values(by="importance", ascending=False)

top_genes = df['target'].head(top_n).tolist()
# Select top-N most important genes

# ====================== Step 3: Build Gene Indices ======================

gene_to_idx = {g: i for i, g in enumerate(gene_names)}

if perturb_gene not in gene_to_idx:
    raise ValueError(f"{perturb_gene} not found in gene list")

# Shift by +1 to account for CLS token
perturb_idx = gene_to_idx[perturb_gene] + 1

# Keep only valid genes existing in gene list
valid_genes = [g for g in top_genes if g in gene_to_idx]

indices = [perturb_idx] + [gene_to_idx[g] + 1 for g in valid_genes]
labels = [perturb_gene] + valid_genes

print(f"Number of genes used in heatmap: {len(labels)}")

# ====================== Step 4: Extract Submatrix ======================

sub_attn = delta_attn[np.ix_(indices, indices)]

# Take absolute value of attention differences
sub_attn_abs = np.abs(sub_attn)

# Define color scaling (95th percentile for robustness)
vmax = np.percentile(sub_attn_abs, 95)

# ====================== Step 5: Visualization ======================

plt.figure(figsize=(10, 8))

sns.heatmap(
    sub_attn_abs,
    cmap="Blues",
    vmin=0,
    vmax=vmax,
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0,
    cbar_kws={
        "shrink": 0.6,
        "label": "|Δ Attention|"
    }
)

plt.title(f"{perturb_gene} Connectivity Heatmap (|Δ Attention|)", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()