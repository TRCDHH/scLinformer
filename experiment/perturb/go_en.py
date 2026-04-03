import os
import pandas as pd
import gseapy as gp

# ==================== 配置 ====================
csv_path = "/home/output/adamson/HSPA5_delta.csv"
outdir = "/home/output/adamson/HSPA5"
top_n = 20                 # 或者改成 importance阈值
use_threshold = False      # True: 用importance筛选

os.makedirs(outdir, exist_ok=True)

# ==================== Step 1: 读取CSV ====================
df = pd.read_csv(csv_path)

# 排序
df = df.sort_values(by="importance", ascending=False)

# ==================== Step 2: 选Top基因 ====================
if use_threshold:
    # 按importance筛选（推荐阈值0.25，可自行调整）
    top_genes = df[df['importance'] > 0.25]['target'].tolist()
else:
    # 直接取Top N
    top_genes = df['target'].head(top_n).tolist()

# 可选：去掉低信息基因（如RP11开头）
top_genes = [g for g in top_genes if not g.startswith("RP11")]

print(f"使用基因数量: {len(top_genes)}")
print("前10个基因:", top_genes[:10])

# ==================== Step 3: GO富集 ====================
enr = gp.enrichr(
    gene_list=top_genes,
    gene_sets="GO_Biological_Process_2021",
    organism="human",
    outdir=outdir,
    cutoff=0.05
)

# 获取结果
res = enr.results

# 保存结果
res.to_csv(os.path.join(outdir, "GO_results.csv"), index=False)

print("\nTop GO terms:")
print(res[['Term', 'Adjusted P-value']].head(10))

# ==================== Step 4: 画图 ====================

# Bar plot
gp.barplot(
    res,
    title="SOCS1 GO Enrichment",
    cutoff=0.05,
    ofname=os.path.join(outdir, "barplot.png")
)

# Dot plot（推荐论文用）
gp.dotplot(
    res,
    title="SOCS1 GO Enrichment",
    cutoff=0.05,
    top_term=10,
    ofname=os.path.join(outdir, "dotplot.png")
)

print(f"\n结果已保存到: {outdir}")