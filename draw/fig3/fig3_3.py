# %% [1] Import Libraries and Define All Custom Parameters (Top)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# ---------------------- CUSTOM PARAMETERS ----------------------
OUTPUT_PATH = '/home/draw/output'
FONT_FAMILY = 'DejaVu Sans'
FONT_SIZE_GLOBAL = 10
FONT_SIZE_ANNOT = 9
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
AXES_LINE_WIDTH = 0.8
PLOT_FIGSIZE = (10, 8)
DPI_VALUE = 300
VMIN_VALUE = 0
P_VALUE_THRESHOLD = 0.1
TOP_N_GO = 10

# File paths
FILE_PATHS = {
    'bmmc': '/home/evaluation/basis/bmmc/bmmc_ALL_basis_significant_GO_summary.csv',
    'immune': '/home/evaluation/basis/immune/immune_ALL_basis_significant_GO_summary.csv',
    'lungatlas': '/home/evaluation/basis/lungatlas/lungatlas_ALL_basis_significant_GO_summary.csv',
    'covid': '/home/evaluation/basis/covid/covid_ALL_basis_significant_GO_summary.csv'
}

# Core pathway keywords
CORE_PATHWAY_KEYWORDS = {
    'Immune Response': ['immune response', 'T cell activation', 'monocyte activation', 'lymphocyte'],
    'Metabolic Process': ['metabolic process', 'metabolism', 'glucose', 'energy'],
    'Signal Transduction': ['signaling', 'signal transduction', 'pathway'],
    'bmmc Stress Response': ['stress', 'oxidative stress', 'cellular stress'],
    'Lung Epithelial Migration': ['epithelial cell differentiation', 'cell adhesion', 'migration', 'lung'],
    'Cell Apoptosis': ['apoptosis', 'cell death'],
    'Immune Cell Proliferation': ['proliferation', 'cell cycle', 'immune cell proliferation']
}

# Dataset display order
DATASET_ORDER = ['bmmc', 'immune', 'lungatlas', 'covid']

# %% [2] Global Plot Settings
plt.switch_backend('Agg')
plt.rcParams['font.family'] = FONT_FAMILY
plt.rcParams['font.size'] = FONT_SIZE_GLOBAL
plt.rcParams['axes.linewidth'] = AXES_LINE_WIDTH
os.makedirs(OUTPUT_PATH, exist_ok=True)

# %% [3] Load and Process Data
df_list = []
for dataset, path in FILE_PATHS.items():
    df = pd.read_csv(path).copy()
    # Clean GO terms
    df['GO_clean'] = df['GO_term'].apply(lambda x: re.sub(r'\s*GO:\d+\s*|\s*\(GO:\d+\)\s*', '', x).strip().lower())
    # Match pathway topics
    df['match_topic'] = ''
    for topic, keywords in CORE_PATHWAY_KEYWORDS.items():
        mask = df['GO_clean'].str.contains('|'.join(keywords), regex=True)
        df.loc[mask, 'match_topic'] = topic
    # Filter significant results
    df_filtered = df[(df['match_topic'] != '') & (df['Adjusted_P_value'] < P_VALUE_THRESHOLD)].head(TOP_N_GO)
    df_filtered.loc[:, 'dataset'] = dataset
    df_list.append(df_filtered)

df_all = pd.concat(df_list, ignore_index=True)

# %% [4] Aggregate and Pivot Data
df_merged = df_all.groupby(['match_topic', 'dataset'])['Combined_Score'].mean().reset_index()
pivot_df = df_merged.pivot_table(index='match_topic', columns='dataset', values='Combined_Score', fill_value=0)
pivot_df = pivot_df[DATASET_ORDER]

# %% [5] Plot Heatmap
fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=DPI_VALUE)
cmap = sns.color_palette("Blues", as_cmap=True)

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
    annot_kws={"size": FONT_SIZE_ANNOT, "weight": "normal"},
    vmin=VMIN_VALUE
)

# %% [6] Plot Styling
ax.set_title('Functional Enrichment of Linformer E-matrix Across Datasets',
             fontsize=FONT_SIZE_TITLE, fontweight='bold', pad=20)
ax.set_xlabel('Dataset', fontsize=FONT_SIZE_LABEL, labelpad=10, fontweight='medium')
ax.set_ylabel('Functional Topic', fontsize=FONT_SIZE_LABEL, labelpad=10, fontweight='medium')

ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right', fontsize=FONT_SIZE_TICK)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZE_TICK)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(AXES_LINE_WIDTH)
ax.spines['left'].set_linewidth(AXES_LINE_WIDTH)

plt.tight_layout()

# %% [7] Save Figures
save_png = os.path.join(OUTPUT_PATH, 'final_linformer_pathway_heatmap.png')
plt.savefig(save_png, bbox_inches='tight', facecolor='white', dpi=DPI_VALUE)

save_pdf = os.path.join(OUTPUT_PATH, 'final_linformer_pathway_heatmap.pdf')
plt.savefig(save_pdf, bbox_inches='tight', facecolor='white')

plt.show()
plt.close()

# %% [8] Print Completion Message
print("Final heatmap generated successfully!")
print("PNG path:", save_png)
print("PDF path:", save_pdf)