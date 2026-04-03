import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from textwrap import wrap
from matplotlib.lines import Line2D

# -------------------------- 【你只需要修改这里】--------------------------
CSV_PATH = "/home/evaluation/basis/bmmc/bmmc_ALL_basis_significant_GO_summary.csv"
SAVE_PATH = "/home/draw/fig3/linformer_go_bmmc.png"
TOP_GO      = 10
# ===================================================

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font_scale=1.1)

# 1. 读取数据
df = pd.read_csv(CSV_PATH)

# 2. 数据处理
df['neglog10P'] = -np.log10(df['Adjusted_P_value'])
df['GO_short'] = df['GO_term'].str.replace(r" \(GO:\d+\)", "", regex=True).str.strip()

df_basis = df.groupby(['basis_idx', 'GO_short'], as_index=False).agg({
    'neglog10P': 'max',
    'Combined_Score': 'max'
})

# 选 top GO
top_go_names = (
    df_basis.groupby('GO_short')['neglog10P']
    .max()
    .nlargest(TOP_GO)
    .index
)

df_plot = df_basis[df_basis['GO_short'].isin(top_go_names)].copy()

#  log 压缩大小
df_plot['size_scaled'] = np.log1p(df_plot['Combined_Score'])

# GO 排序
go_order = (
    df_plot.groupby('GO_short')['neglog10P']
    .max()
    .sort_values(ascending=True)
    .index
)

df_plot['GO_short'] = pd.Categorical(
    df_plot['GO_short'],
    categories=go_order,
    ordered=True
)

# ========================== ✅ 最小修改：固定布局 不溢出 ==========================
# 用 GridSpec 给图例单独留空间，永远不溢出
fig = plt.figure(figsize=(15, 7))
gs = GridSpec(1, 2, width_ratios=[5, 2])  # 主图6 + 右侧1.3专门放图例

ax = fig.add_subplot(gs[0])   # 主图
ax_leg = fig.add_subplot(gs[1]) # 独立图例区域
ax_leg.axis('off')

palette = ["#4A90E2", "#5CDB96", "#F5A623", "#E2596B", "#9B6DD9", "#5CC8DB"]

sns.scatterplot(
    data=df_plot,
    x="neglog10P",
    y="GO_short",
    hue="basis_idx",
    size="size_scaled",
    sizes=(80, 500),
    palette=palette,
    edgecolor="#f0f0f0",
    linewidth=1,
    alpha=0.9,
    legend=False,
    ax=ax
)

# ================= 手动构建 legend（完全不变） =================
unique_basis = sorted(df_plot['basis_idx'].unique())
hue_handles = [
    Line2D([0], [0], marker='o', color='w',
           label=f'Basis {b}',
           markerfacecolor=palette[i % len(palette)],
           markersize=8)
    for i, b in enumerate(unique_basis)
]

quantiles = df_plot['Combined_Score'].quantile([0.25, 0.5, 0.75, 0.95]).values
size_values = np.unique(np.round(quantiles, 0))

def map_size(v, vmin, vmax, smin=80, smax=500):
    v_log = np.log1p(v)
    vmin_log = np.log1p(vmin)
    vmax_log = np.log1p(vmax)
    return smin + (v_log - vmin_log) / (vmax_log - vmin_log) * (smax - smin)

vmin = df_plot['Combined_Score'].min()
vmax = df_plot['Combined_Score'].max()

size_handles = [
    Line2D([0], [0], marker='o', color='gray',
           label=f"{int(v)}",
           markerfacecolor='gray',
           markersize=np.sqrt(map_size(v, vmin, vmax))
    )
    for v in size_values
]

# ================= ✅ 图例放在独立区域，永不溢出、不重叠 =================
# 1. Basis 图例（放在右侧上方）
leg1 = ax_leg.legend(
    hue_handles, [h.get_label() for h in hue_handles],
    title="Basis", loc="upper left", frameon=False, labelspacing=1.2, ncol=2
)

# 2. Score 图例（放在右侧下方）
leg2 = ax_leg.legend(
    size_handles, [h.get_label() for h in size_handles],
    title="Combined Score", loc="lower left", frameon=False, labelspacing=1.2, ncol=2
)

ax_leg.add_artist(leg1)

# 主图样式（不变）
ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11)
ax.set_ylabel("")
ax.grid(axis='x', alpha=0.15)
sns.despine(left=True, bottom=True, ax=ax)

# 保存
plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight', facecolor="white")
plt.close()

print("✅ 修复完成：图例永不溢出！")