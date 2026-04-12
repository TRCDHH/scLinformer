# %% [1] Import Libraries and Define All Custom Parameters (Top)
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc

# ---------------------- CUSTOM PARAMETERS ----------------------
DATASET = "lungatlas"
INPUT_PATH = f"/home/output/{DATASET}/adata_with_embedding.h5ad"
SAVE_PATH = f"/home/draw/other/xxx_{DATASET}.png"

# Plot parameters
CLUSTERMAP_FIGSIZE = (10, 12)
CMAP_NAME = "rocket"
COLOR_PALETTE = "Set2"
DPI_VALUE = 300

# Colorbar position [left, bottom, width, height]
CB_POSITION = [0.02, 0.35, 0.02, 0.3]

# %% [2] Create Output Directory
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# %% [3] Load Data
adata = sc.read_h5ad(INPUT_PATH)
emb = np.asarray(adata.obsm["X_emb"])
cell_type = np.array(adata.obs["cell_type"].values)

# %% [4] Sort by Cell Type
order = np.argsort(cell_type)
emb_sorted = emb[order]
cell_type_sorted = cell_type[order]

# %% [5] Prepare Cell Type Colors
unique_types = np.unique(cell_type_sorted)
palette = sns.color_palette(COLOR_PALETTE, len(unique_types))
color_map = dict(zip(unique_types, palette))
row_colors = np.array([color_map[x] for x in cell_type_sorted])

# %% [6] Plot Clustermap
sns.set(style="white")

g = sns.clustermap(
    emb_sorted,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,
    cmap=CMAP_NAME,
    figsize=CLUSTERMAP_FIGSIZE,
    xticklabels=False,
    yticklabels=False,
    cbar_pos=None
)

# %% [7] Add Custom Colorbar
mappable = g.ax_heatmap.collections[0]
cax = g.fig.add_axes(CB_POSITION)
cb = plt.colorbar(mappable, cax=cax)
cb.set_label("Embedding value", rotation=90)

# %% [8] Hide Dendrogram Legend
g.ax_row_dendrogram.set_visible(False)

# %% [9] Save and Show Plot
plt.savefig(SAVE_PATH, dpi=DPI_VALUE, bbox_inches="tight")
plt.show()
plt.close()

# %% [10] Completion Message
print(f"Saved to: {SAVE_PATH}")