import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as ad

# ========== 1. 参数 ==========
dataset = "lungatlas"
input_path = f"/home/output/{dataset}/adata_with_embedding.h5ad"
save_path = f"/home/draw/other/xxx_{dataset}.png"

os.makedirs(os.path.dirname(save_path), exist_ok=True)

# ========== 2. 读取 ==========
adata = ad.read_h5ad(input_path)

emb = np.asarray(adata.obsm["X_emb"])
cell_type = np.array(adata.obs["cell_type"].values)

# ========== 3. 排序 ==========
order = np.argsort(cell_type)
emb_sorted = emb[order]
cell_type_sorted = cell_type[order]

# ========== 4. 更好看的 cell type 配色 ==========
unique_types = np.unique(cell_type_sorted)

palette = sns.color_palette("Set2", len(unique_types))  # ⭐比 husl 更柔和
color_map = dict(zip(unique_types, palette))
row_colors = np.array([color_map[x] for x in cell_type_sorted])

# ========== 5. heatmap ==========
sns.set(style="white")

g = sns.clustermap(
    emb_sorted,
    row_cluster=False,
    col_cluster=False,
    row_colors=row_colors,

    # ⭐更好看的 colormap（替换 coolwarm）
    cmap="rocket",

    figsize=(10, 12),
    xticklabels=False,
    yticklabels=False,

    # ⭐关键：关掉默认 colorbar（我们自己放）
    cbar_pos=None
)

# ========== 6. 手动加 colorbar（放左中间） ==========
mappable = g.ax_heatmap.collections[0]

cax = g.fig.add_axes([0.02, 0.35, 0.02, 0.3])  
# [left, bottom, width, height]
# 👉 left=0.02 靠左
# 👉 bottom=0.35 中间位置

cb = plt.colorbar(mappable, cax=cax)
cb.set_label("Embedding value", rotation=90)

# ========== 7. 去掉 legend（你要求的） ==========
g.ax_row_dendrogram.set_visible(False)

# ========== 8. 保存 ==========
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved to: {save_path}")