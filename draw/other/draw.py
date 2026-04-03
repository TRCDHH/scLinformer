import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import scanpy as sc
from scipy.optimize import linear_sum_assignment

import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 读取数据
# ============================================
dataset = "bmmc"
adata = sc.read_h5ad(f'/home/output/{dataset}/adata_with_embedding.h5ad')

# ============================================
# 2. 获取 embedding 和真实标签
# ============================================
X_emb = adata.obsm['X_emb']
true_labels = adata.obs['cell_type'].astype(str)

print("真实标签类别数:", len(true_labels.unique()))

# ============================================
# 3. 用 Leiden 聚类（替代 KMeans ✅）
# ============================================
print("\n正在进行 Leiden 聚类...")

sc.pp.neighbors(adata, use_rep='X_emb')
sc.tl.leiden(adata, resolution=0.3)

pred_labels = adata.obs['leiden'].astype(str)

print("聚类类别数:", len(pred_labels.unique()))

# ============================================
# 4. 匈牙利算法对齐标签（修复版 ✅）
# ============================================
def align_labels(y_true, y_pred):
    true_unique = sorted(y_true.unique())
    pred_unique = sorted(y_pred.unique())

    # 构建混淆矩阵（行=真实，列=预测）
    cm = confusion_matrix(y_true, y_pred, labels=true_unique)

    # 匈牙利算法（最大匹配）
    row_ind, col_ind = linear_sum_assignment(-cm)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if j < len(pred_unique):
            mapping[pred_unique[j]] = true_unique[i]

    aligned_pred = y_pred.map(lambda x: mapping.get(x, x))
    return aligned_pred, mapping

aligned_pred, label_mapping = align_labels(true_labels, pred_labels)

# ============================================
# 5. 计算评估指标
# ============================================
acc = accuracy_score(true_labels, aligned_pred)
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)

print("\n====== 评估指标 ======")
print(f"Accuracy (after alignment): {acc:.4f}")
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")

# ============================================
# 6. 计算混淆矩阵
# ============================================
labels = sorted(true_labels.unique())
cm = confusion_matrix(true_labels, aligned_pred, labels=labels)

# 行归一化百分比
row_sums = cm.sum(axis=1)
cm_percent = np.zeros_like(cm, dtype=float)

for i in range(len(cm)):
    if row_sums[i] > 0:
        cm_percent[i] = cm[i] / row_sums[i] * 100

# ============================================
# 7. 绘制混淆矩阵
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 12))

# Count
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=axes[0], cbar_kws={'label': 'Count'})

axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')

# Percent
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn',
            xticklabels=labels, yticklabels=labels,
            vmin=0, vmax=100,
            ax=axes[1], cbar_kws={'label': '%'})

axes[1].set_title('Confusion Matrix (%)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

fig.suptitle(f'Confusion Matrix\nACC={acc:.3f}, ARI={ari:.3f}, NMI={nmi:.3f}', fontsize=14)

plt.tight_layout()
plt.savefig(f'/home/draw/other/confusion_matrix_{dataset}.png', dpi=300)

# ============================================
# 8. 输出 mapping
# ============================================
print("\n====== 标签映射 (cluster -> cell type) ======")
for k, v in sorted(label_mapping.items(), key=lambda x: int(x[0])):
    print(f"Cluster {k} -> {v}")

# ============================================
# 9. 每类准确率
# ============================================
print("\n====== 每类准确率 ======")
for i, label in enumerate(labels):
    if row_sums[i] > 0:
        acc_i = cm[i, i] / row_sums[i]
        print(f"{label}: {acc_i:.3f} ({cm[i,i]}/{row_sums[i]})")
    else:
        print(f"{label}: N/A")

# ============================================
# 10. UMAP 可视化
# ============================================
print("\n正在生成 UMAP...")

if 'X_umap' not in adata.obsm:
    sc.tl.umap(adata)

adata.obs['true'] = true_labels
adata.obs['pred'] = pred_labels
adata.obs['aligned'] = aligned_pred
adata.obs['correct'] = (true_labels == aligned_pred).astype(str)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sc.pl.umap(adata, color='true', ax=axes[0], show=False, title='True')
sc.pl.umap(adata, color='pred', ax=axes[1], show=False, title='Cluster')
sc.pl.umap(adata, color='correct', ax=axes[2], show=False, title='Correct')

plt.tight_layout()
plt.savefig(f'/home/draw/other/umap_{dataset}.png', dpi=300)

# ============================================
# 11. 清理
# ============================================
for col in ['true', 'pred', 'aligned', 'correct']:
    if col in adata.obs.columns:
        del adata.obs[col]

print("\n✅ 完成！")