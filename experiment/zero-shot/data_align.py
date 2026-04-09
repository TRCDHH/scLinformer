import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

# align rna2 to rna1 based on gene names, and save the aligned data
rna1 = sc.read_h5ad("/home/output/pre-train/pbmc/processed_data/processed_rna_uni.h5ad")
rna2 = sc.read_h5ad("/home/data/newcovid/challenge_pbmc_cellxgene_230223.h5ad")

target_genes = rna1.var_names.tolist()

gene_to_idx = {g: i for i, g in enumerate(rna2.var_names)}
target_indices = [gene_to_idx[g] if g in gene_to_idx else -1 for g in target_genes]

n_cells, n_genes = rna2.n_obs, len(target_genes)
data, row_idx, col_idx = [], [], []

for new_col, old_col in enumerate(target_indices):
    if old_col != -1:
        col = rna2.X[:, old_col]
        if hasattr(col, 'toarray'):
            col = col.toarray().flatten()
        else:
            col = np.array(col).flatten()
        rows = np.where(col != 0)[0]
        data.extend(col[rows])
        row_idx.extend(rows)
        col_idx.extend([new_col] * len(rows))

new_X = csr_matrix((data, (row_idx, col_idx)), shape=(n_cells, n_genes))

rna2_aligned = sc.AnnData(
    X=new_X,
    obs=rna2.obs.copy(),
    var=pd.DataFrame(index=target_genes)
)

for col in rna1.var.columns:
    rna2_aligned.var[col] = rna1.var[col].values

print(f"align finished: {rna2_aligned.shape}")
print(f"gene match: {sum(1 for x in target_indices if x != -1)}/{len(target_genes)}")
rna2_aligned.write_h5ad("/home/data/newcovid/challenge_pbmc_cellxgene_230223_aligned.h5ad")
print(rna2_aligned)