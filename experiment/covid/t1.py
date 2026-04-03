import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from scipy.spatial.distance import cdist
import warnings


def draw(adata, name):
    print(adata)
    
    # 保存路径
    fig_dir = f"/home/mycode/experiment/covid/{name}"
    os.makedirs(fig_dir, exist_ok=True)
    
    # ================================
    # 0️⃣ 基础设置
    # ================================
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=300, figsize=(6, 6))
    sc.settings.figdir = fig_dir

    # ================================
    # 1️⃣ 检查必要数据
    # ================================
    assert "X_emb" in adata.obsm.keys(), "❌ 没有找到 X_emb embedding"
    assert "time_point" in adata.obs.columns, "❌ 没有找到 time_point 列"
    
    # 检查时间点数据
    available_times = adata.obs["time_point"].unique()
    print(f"可用时间点: {available_times}")

    # ================================
    # 2️⃣ 构建邻接图（基于 embedding）
    # ================================
    sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=30)

    # ================================
    # 3️⃣ UMAP（用于可视化）
    # ================================
    sc.tl.umap(adata)

    # ================================
    # 4️⃣ Diffusion Map（关键修复：使用 neighbors 的表示）
    # ================================
    # 确保 diffmap 使用基于 X_emb 构建的邻居图
    sc.tl.diffmap(adata, n_comps=15)

    # ================================
    # 5️⃣ 智能选择 root 细胞（关键改进）
    # ================================
    root_candidates = np.where(adata.obs["time_point"] == "D-1")[0]

    if len(root_candidates) == 0:
        raise ValueError("❌ 没有找到 D-1 时间点，无法设置 root")
    
    if len(root_candidates) == 1:
        # 只有一个 D-1 细胞，直接使用
        adata.uns["iroot"] = root_candidates[0]
        print(f"⚠️ 只有一个 D-1 细胞，使用索引 {root_candidates[0]} 作为 root")
    else:
        # 多个 D-1 细胞：选择在 diffusion space 中最"中心"的细胞
        # 或者选择 diffusion component 1 值最小的（假设是起点）
        diff_coords = adata.obsm["X_diffmap"][root_candidates]
        
        # 方法1：选择 DC1 最小的（假设 DC1 捕捉了时间轴）
        dc1_values = diff_coords[:, 1]  # 第0列是常数1，第1列是第一个DC
        root_idx = root_candidates[np.argmin(dc1_values)]
        
        # 方法2（备选）：选择几何中心最近的细胞
        # centroid = diff_coords.mean(axis=0)
        # distances = cdist(diff_coords, [centroid])
        # root_idx = root_candidates[np.argmin(distances)]
        
        adata.uns["iroot"] = root_idx
        print(f"✅ 从 {len(root_candidates)} 个 D-1 细胞中选择索引 {root_idx} 作为 root")

    # ================================
    # 6️⃣ 计算 pseudotime
    # ================================
    sc.tl.dpt(adata)

    assert "dpt_pseudotime" in adata.obs.columns, "❌ dpt 计算失败"
    
    # 检查 pseudotime 范围
    pt_range = adata.obs["dpt_pseudotime"].describe()
    print(f"Pseudotime 统计:\n{pt_range}")

    # ================================
    # 7️⃣ UMAP + pseudotime 可视化
    # ================================
    sc.pl.umap(
        adata,
        color=["dpt_pseudotime"],
        save="_pseudotime.png",
        show=False
    )

    # ================================
    # 8️⃣ Pseudotime vs 时间（核心验证图，修复采样逻辑）
    # ================================
    # 动态确定时间顺序（只包含数据中存在的）
    all_timepoints = ["D-1", "D3", "D7", "D10", "D14", "D28"]
    time_order = [t for t in all_timepoints if t in available_times]
    
    if len(time_order) == 0:
        time_order = sorted(available_times)  # 回退到字母排序
    
    # 转 dataframe
    df = pd.DataFrame({
        "time_point": adata.obs["time_point"].values,
        "pseudotime": adata.obs["dpt_pseudotime"].values
    })

    # 修复：安全采样，避免不必要的 replace=True
    time_counts = df["time_point"].value_counts()
    print(f"各时间点细胞数:\n{time_counts}")
    
    # 确定采样数：最多1000，但至少保留所有细胞如果总数不足
    n_target = min(1000, time_counts.min())
    
    sampled_dfs = []
    for tp in time_order:
        tp_df = df[df["time_point"] == tp]
        if len(tp_df) > n_target:
            sampled_dfs.append(tp_df.sample(n=n_target, random_state=42))
        else:
            sampled_dfs.append(tp_df)  # 不采样，保留全部
            if len(tp_df) < 100:
                warnings.warn(f"时间点 {tp} 只有 {len(tp_df)} 个细胞，考虑合并或排除")
    
    df_sampled = pd.concat(sampled_dfs)

    # 绘制 violin plot
    fig, ax = plt.subplots(figsize=(8, 5))
    
    sns.violinplot(
        data=df_sampled,
        x="time_point",
        y="pseudotime",
        order=time_order,
        inner="box",
        palette="viridis",
        ax=ax
    )
    
    ax.set_xlabel("Time Point", fontsize=12)
    ax.set_ylabel("Pseudotime", fontsize=12)
    ax.set_title(f"Pseudotime Distribution by Time Point\n({name})", fontsize=14)
    
    # 添加样本数标注
    for i, tp in enumerate(time_order):
        count = time_counts[tp]
        ax.annotate(f"n={count}", xy=(i, ax.get_ylim()[1]), 
                   ha='center', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/pseudotime_violin.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 额外：绘制箱线图版本（更清晰显示中位数）
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="time_point",
        y="pseudotime",
        order=time_order,
        palette="viridis",
        ax=ax
    )
    ax.set_xlabel("Time Point", fontsize=12)
    ax.set_ylabel("Pseudotime", fontsize=12)
    ax.set_title(f"Pseudotime by Time Point (Boxplot)\n({name})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/pseudotime_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ================================
    # 9️⃣ 多标签可视化
    # ================================
    color_vars = []
    for var in ["cell_type", "covid_status", "time_point", "patient_id"]:
        if var in adata.obs.columns:
            color_vars.append(var)
    
    for var in color_vars:
        try:
            sc.pl.umap(
                adata,
                color=[var],
                save=f"_{var}.png",
                show=False,
                legend_loc='on data' if adata.obs[var].nunique() < 30 else 'right margin'
            )
        except Exception as e:
            print(f"⚠️ 绘制 {var} 时出错: {e}")

    # ================================
    # 🔟 额外：Pseudotime 与时间的相关性验证
    # ================================
    # 将时间转换为数值计算相关性
    time_to_num = {t: i for i, t in enumerate(time_order)}
    df["time_num"] = df["time_point"].map(time_to_num)
    
    # 计算 Spearman 相关性（非参数，适合顺序数据）
    from scipy.stats import spearmanr
    corr, pval = spearmanr(df["time_num"], df["pseudotime"])
    
    print(f"\n📊 验证结果:")
    print(f"   Spearman 相关性 (Time vs Pseudotime): {corr:.3f}")
    print(f"   P-value: {pval:.2e}")
    
    if corr < 0.3:
        print("   ⚠️ 警告: 相关性较低，pseudotime 可能未很好捕捉时间趋势")
    elif corr > 0.7:
        print("   ✅ 强正相关，pseudotime 很好反映了时间进程")
    
    # 保存统计结果
    with open(f"{fig_dir}/pseudotime_stats.txt", "w") as f:
        f.write(f"Sample: {name}\n")
        f.write(f"Cells: {adata.n_obs}\n")
        f.write(f"Time points: {list(available_times)}\n")
        f.write(f"Spearman correlation: {corr:.4f} (p={pval:.2e})\n")
        f.write(f"\nPseudotime by timepoint:\n")
        f.write(df.groupby("time_point")["pseudotime"].describe().to_string())

    print(f"\n✅ 全部完成，图已保存到: {fig_dir}")


# ================================
# 主程序
# ================================
if __name__ == "__main__":
    # 读取数据
    adata = sc.read_h5ad('/home/output/newcovid/adata_with_embedding.h5ad')
    print(f"数据形状: {adata.shape}")
    print(f"Cell types: {adata.obs['cell_type'].unique()}")

    # 提取特定细胞类型和患者
    sub = adata[
        (adata.obs["cell_type"] == "T CD8 Naive")
    ].copy()

    if sub.n_obs == 0:
        raise ValueError("❌ 没有符合条件的细胞，检查 cell_type 和 patient_id")
    
    print(f"\n子集形状: {sub.shape}")
    draw(sub, "Monocyte_CD14")