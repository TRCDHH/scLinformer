import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# ==================== 配置 ====================
attn_path = "/home/output/adamson/SOCS1_delta_attn.npy"
csv_path = "/home/output/adamson/SOCS1_delta.csv"
h5ad_path = "/home/data/adamson/control.h5ad"

perturb_gene = "SOCS1"
top_n = 20
save_path = "/home/output/adamson/SOCS1_heatmap_abs.png"

# ==================== Step 1: 读取数据 ====================
delta_attn = np.load(attn_path)  # [N+1, N+1]

adata = sc.read_h5ad(h5ad_path)
gene_names = list(adata.var['gene_name'])

# ==================== Step 2: CSV取Top基因 ====================
df = pd.read_csv(csv_path)
df = df.sort_values(by="importance", ascending=False)

top_genes = df['target'].head(top_n).tolist()

# ==================== Step 3: 构建索引（注意CLS偏移） ====================
gene_to_idx = {g: i for i, g in enumerate(gene_names)}

if perturb_gene not in gene_to_idx:
    raise ValueError(f"{perturb_gene} 不在gene list中")

# ⚠️ +1（跳过CLS）
perturb_idx = gene_to_idx[perturb_gene] + 1

valid_genes = [g for g in top_genes if g in gene_to_idx]

indices = [perturb_idx] + [gene_to_idx[g] + 1 for g in valid_genes]
labels = [perturb_gene] + valid_genes

print(f"最终基因数: {len(labels)}")

# ==================== Step 4: 提取子矩阵 ====================
sub_attn = delta_attn[np.ix_(indices, indices)]

# ==================== ⭐ 改动1：取绝对值 ====================
sub_attn_abs = np.abs(sub_attn)

# ==================== ⭐ 改动2：颜色范围（柔和关键） ====================
vmax = np.percentile(sub_attn_abs, 95)

# ==================== 画图 ====================
plt.figure(figsize=(10, 8))

sns.heatmap(
    sub_attn_abs,
    cmap="Blues",              # ⭐ 蓝色渐变
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

plt.title("SOCS1 Connectivity Heatmap (|Δ Attention|)", fontsize=14)
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()