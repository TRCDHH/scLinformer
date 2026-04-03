import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import scanpy as sc

# ============================================
# 1. 读取数据
# ============================================
dataset = "bmmc"
adata = sc.read_h5ad(f'/home/output/{dataset}/adata_with_embedding.h5ad')

X = adata.obsm['X_emb']
labels = adata.obs['cell_type'].astype(str)

# ============================================
# 2. 计算每个 cell type 的平均 embedding
# ============================================
print("计算 cell type centroid...")

df = pd.DataFrame(X, index=labels.index)
df['cell_type'] = labels.values

centroids = df.groupby('cell_type').mean()

# 可选：排序（让图更整齐）
centroids = centroids.sort_index()

# ============================================
# 3. 计算相似度矩阵（cosine similarity）
# ============================================
sim_matrix = cosine_similarity(centroids.values)

sim_df = pd.DataFrame(
    sim_matrix,
    index=centroids.index,
    columns=centroids.index
)

# ============================================
# 4. 绘制热力图
# ============================================
plt.figure(figsize=(10, 8))

sns.heatmap(
    sim_df,
    cmap='RdBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Cosine Similarity'}
)

plt.title('Cell Type Embedding Similarity (Centroid)', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(f'/home/draw/other/celltype_similarity_{dataset}.png', dpi=300)
plt.show()

print("✅ 完成")