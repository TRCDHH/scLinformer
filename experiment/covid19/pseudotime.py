import scanpy as sc
import numpy as np
import os

# ================================
#  CONFIGURATION
# ================================
DATA_PATH = "/home/adata_with_embedding.h5ad"
OUTPUT_DIR = "/home/covid/umap_pseudotime"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_KEY = "X_emb"
TIME_KEY = "time_point"
ROOT_TIME_POINT = "D-1"
N_NEIGHBORS = 30
N_DIFFMAP_COMPS = 15

# ================================
# LOAD DATA
# ================================
adata = sc.read_h5ad(DATA_PATH)

assert EMBED_KEY in adata.obsm
assert TIME_KEY in adata.obs

# ================================
# NEIGHBORS + UMAP + DIFFMAP
# ================================
sc.pp.neighbors(adata, use_rep=EMBED_KEY, n_neighbors=N_NEIGHBORS)
sc.tl.umap(adata)
sc.tl.diffmap(adata, n_comps=N_DIFFMAP_COMPS)

# ================================
# ROOT SELECTION
# ================================
root_candidates = np.where(adata.obs[TIME_KEY] == ROOT_TIME_POINT)[0]

if len(root_candidates) == 0:
    raise ValueError(f"No {ROOT_TIME_POINT} cells found")

if len(root_candidates) == 1:
    adata.uns["iroot"] = root_candidates[0]
else:
    diff_coords = adata.obsm["X_diffmap"][root_candidates]
    dc1_values = diff_coords[:, 1]
    adata.uns["iroot"] = root_candidates[np.argmin(dc1_values)]

# ================================
# PSEUDOTIME
# ================================
sc.tl.dpt(adata)

# ================================
# UMAP PLOTTING
# ================================
sc.pl.umap(
    adata,
    color=["dpt_pseudotime"],
    show=False
)

import matplotlib.pyplot as plt
plt.savefig(f"{OUTPUT_DIR}/umap_pseudotime.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Saved: {OUTPUT_DIR}/umap_pseudotime.png")