# %% [1] Import Libraries and Define All Custom Parameters (Top)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc

# ---------------------- CUSTOM PARAMETERS ----------------------
DATASET = "bmmc"
ADATA_PATH = f"/home/output/{DATASET}/adata_with_embedding.h5ad"
SAVE_PATH = f"/home/draw/other/celltype_similarity_{DATASET}.png"

PLOT_FIGSIZE = (10, 8)
DPI_VALUE = 300
CMAP_NAME = "RdBu_r"
CENTER_VALUE = 0
FONT_SIZE_TITLE = 14

# %% [2] Load Data
adata = sc.read_h5ad(ADATA_PATH)
X = adata.obsm['X_emb']
labels = adata.obs['cell_type'].astype(str)

# %% [3] Compute Cell Type Centroids
print("Calculating cell type centroids...")

df = pd.DataFrame(X, index=labels.index)
df['cell_type'] = labels.values
centroids = df.groupby('cell_type').mean()
centroids = centroids.sort_index()

# %% [4] Compute Cosine Similarity Matrix
sim_matrix = cosine_similarity(centroids.values)
sim_df = pd.DataFrame(
    sim_matrix,
    index=centroids.index,
    columns=centroids.index
)

# %% [5] Plot Similarity Heatmap
plt.figure(figsize=PLOT_FIGSIZE)

sns.heatmap(
    sim_df,
    cmap=CMAP_NAME,
    center=CENTER_VALUE,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Cosine Similarity'}
)

plt.title('Cell Type Embedding Similarity (Centroid)', fontsize=FONT_SIZE_TITLE)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=DPI_VALUE)
plt.show()
plt.close()

# %% [6] Completion Message
print("Plot completed successfully!")