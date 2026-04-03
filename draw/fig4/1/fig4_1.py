import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# ===================== 配置 =====================
perturb_gene = 'SOCS1'
CSV_PATH = "/home/output/adamson/GO_SOCS1/GO_results.csv"
SAVE_DIR = "/home/draw/fig4"
PADJ_THRESHOLD = 0.05
TOP_N = 12

# 👇 核心参数（控制紧凑度）- 已调整：柱子更细，缝隙更小
Y_SCALE = 0.35       # 增大间距系数，让柱子更紧凑
HEIGHT_RATIO = 0.65  # 减小柱子高度比例，让柱子变细（原0.92→0.65）

# =================================================
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. 读取数据
df = pd.read_csv(CSV_PATH)
df_sig = df[df["Adjusted P-value"] < PADJ_THRESHOLD].copy()

if len(df_sig) == 0:
    print("无显著富集结果！")
    exit()

# 2. 排序
df_sig["-log10(Padj)"] = -np.log10(df_sig["Adjusted P-value"])
df_plot = df_sig.sort_values("-log10(Padj)", ascending=False).head(TOP_N)
df_plot = df_plot.iloc[::-1]

# 3. 清洗 + 自动换行（更论文风格）
def clean_and_wrap_term(term, max_len=50):
    term_clean = re.sub(r"\s*\(GO:\d+\)", "", term)
    
    if len(term_clean) > max_len:
        words = term_clean.split()
        lines = []
        current = ""
        for w in words:
            if len(current + " " + w) < max_len:
                current += " " + w
            else:
                lines.append(current.strip())
                current = w
        lines.append(current.strip())
        return "\n".join(lines)
    
    return term_clean

df_plot["Term_clean"] = df_plot["Term"].apply(clean_and_wrap_term)

# 4. 动态画布（关键）
FIG_HEIGHT = max(5, len(df_plot) * 0.5)

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
    "figure.dpi": 300,
    "figure.figsize": (10, FIG_HEIGHT),
    "savefig.bbox": "tight"
})

fig, ax = plt.subplots()

# 5. 配色
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(df_plot)))

# 6. 核心：y压缩 + height匹配
y = np.arange(len(df_plot)) * Y_SCALE

ax.barh(
    y=y,
    width=df_plot["-log10(Padj)"],
    height=Y_SCALE * HEIGHT_RATIO,  # 柱子高度 = 0.35 * 0.65 = 0.2275
    color=colors,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.95
)

ax.set_yticks(y)
ax.set_yticklabels(df_plot["Term_clean"])

# 7. 美化
ax.set_xlabel("-log10(Adjusted P-value)", fontsize=13, labelpad=10)
ax.set_ylabel("Biological Process", fontsize=13, labelpad=10)
ax.set_title(f"GO Enrichment of DEGs upon {perturb_gene} Perturbation", fontsize=13, pad=12)

ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11, pad=2)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 右侧留白（避免顶边）
ax.set_xlim(0, df_plot["-log10(Padj)"].max() * 1.1)

# 轻微网格（更像论文）
ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout(pad=1)

# 8. 保存
png_path = os.path.join(SAVE_DIR, "enrichment_barh_final.png")

plt.savefig(png_path, dpi=300)
plt.close()

print(f"✅ 完成！图片已保存到 {SAVE_DIR}")