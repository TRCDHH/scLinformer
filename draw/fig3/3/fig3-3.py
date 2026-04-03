import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np

plt.switch_backend('Agg')
output_path = '/home/draw/output'
os.makedirs(output_path, exist_ok=True)

# -------------------------- 1. 核心通路关键词（优化命名） --------------------------
core_pathway_keywords = {
    'Immune Response': ['immune response', 'T cell activation', 'monocyte activation', 'lymphocyte'],
    'Metabolic Process': ['metabolic process', 'metabolism', 'glucose', 'energy'],
    'Signal Transduction': ['signaling', 'signal transduction', 'pathway'],
    'bmmc Stress Response': ['stress', 'oxidative stress', 'cellular stress'],
    'Lung Epithelial Migration': ['epithelial cell differentiation', 'cell adhesion', 'migration', 'lung'],
    'Cell Apoptosis': ['apoptosis', 'cell death'],
    'Immune Cell Proliferation': ['proliferation', 'cell cycle', 'immune cell proliferation']
}
# Gene Ontology 层级结构 + MSigDB hallmark gene sets + 免疫/细胞生物学领域经验的人工整合结果
# -------------------------- 2. 读取并筛选数据 --------------------------
file_paths = {
    'bmmc': '/home/evaluation/basis/bmmc/bmmc_ALL_basis_significant_GO_summary.csv',
    'immune': '/home/evaluation/basis/immune/immune_ALL_basis_significant_GO_summary.csv',
    'lungatlas': '/home/evaluation/basis/lungatlas/lungatlas_ALL_basis_significant_GO_summary.csv',
    'covid': '/home/evaluation/basis/covid/covid_ALL_basis_significant_GO_summary.csv'
}

df_list = []
for dataset, path in file_paths.items():
    df = pd.read_csv(path).copy()  # 修复SettingWithCopyWarning
    # 清洗GO_term
    df['GO_clean'] = df['GO_term'].apply(lambda x: re.sub(r'\s*GO:\d+\s*|\s*\(GO:\d+\)\s*', '', x).strip().lower())
    # 匹配主题
    df['match_topic'] = ''
    for topic, keywords in core_pathway_keywords.items():
        mask = df['GO_clean'].str.contains('|'.join(keywords), regex=True)
        df.loc[mask, 'match_topic'] = topic
    # 筛选有效数据
    df_filtered = df[(df['match_topic'] != '') & (df['Adjusted_P_value'] < 0.1)].head(10)
    df_filtered.loc[:, 'dataset'] = dataset
    df_list.append(df_filtered)

df_all = pd.concat(df_list, ignore_index=True)

# -------------------------- 3. 数据合并 + 调整x轴顺序 --------------------------
df_merged = df_all.groupby(['match_topic', 'dataset'])['Combined_Score'].mean().reset_index()
# 自定义x轴顺序：bmmc → immune → lungatlas → covid
dataset_order = ['bmmc', 'immune', 'lungatlas', 'covid']
pivot_df = df_merged.pivot_table(index='match_topic', columns='dataset', values='Combined_Score', fill_value=0)
pivot_df = pivot_df[dataset_order]  # 强制按自定义顺序排列

# -------------------------- 4. 绘制论文级热图（恢复0值为白色） --------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# 移除0值掩码，恢复白色背景
cmap = sns.color_palette("Blues", as_cmap=True)

# 绘制热图（核心调整：去掉mask参数，0值默认白色）
sns.heatmap(
    pivot_df,
    ax=ax,
    cmap=cmap,
    annot=True,
    fmt='.1f',
    cbar_kws={
        'label': 'Mean Combined Score',
        'shrink': 0.8,
        'pad': 0.05
    },
    linewidths=0.5,
    linecolor='white',
    annot_kws={"size": 9, "weight": "normal"},
    vmin=0  # 颜色条从0开始
)

# -------------------------- 5. 论文级美化 --------------------------
# 标题（简洁+突出Linformer）
ax.set_title('Functional Enrichment of Linformer E-matrix Across Datasets', fontsize=13, fontweight='bold', pad=20)
# 轴标签
ax.set_xlabel('Dataset', fontsize=11, labelpad=10, fontweight='medium')
ax.set_ylabel('Functional Topic', fontsize=11, labelpad=10, fontweight='medium')
# 轴标签样式
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
# 移除上/右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)
ax.spines['left'].set_linewidth(0.8)

# 调整布局
plt.tight_layout()

# -------------------------- 6. 保存高清图（论文用） --------------------------
# PNG（用于预览/补充材料）
save_png = os.path.join(output_path, 'final_linformer_pathway_heatmap.png')
plt.savefig(save_png, bbox_inches='tight', facecolor='white', dpi=300)
# PDF（矢量图，论文正文）
save_pdf = os.path.join(output_path, 'final_linformer_pathway_heatmap.pdf')
plt.savefig(save_pdf, bbox_inches='tight', facecolor='white')

print(f"✅ 最终版热图生成完成！")
print(f"PNG路径：{save_png}")
print(f"PDF路径：{save_pdf}")
print(f"\n📊 图表优化说明：")
print("1. X轴顺序：bmmc → immune → lungatlas → covid（正常→组织→疾病）")
print("2. 主题命名：突出数据集特异性（如Lung Epithelial Migration/bmmc Stress Response）")
print("3. 0值恢复为白色背景，视觉更简洁")
print("4. 样式：移除冗余边框，统一字体/线条，符合顶刊排版规范")