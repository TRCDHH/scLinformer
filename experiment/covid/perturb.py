import torch
import scanpy as sc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from scLinformer.model import RNAEncoder
from scLinformer.utils import Dataset
import os

dataset_name = "newcovid"
perturb_gene = "IRF7"   # IL6 IRF7
top_n = 50  
os.makedirs(f'/home/output/{dataset_name}/{perturb_gene}', exist_ok=True)
model = RNAEncoder(n_genes=2000)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(f'/home/output/{dataset_name}/model/rna_encoder.pth'))

rna = sc.read_h5ad(f'/home/output/{dataset_name}/processed_data/processed_rna.h5ad')
print(rna.var_names)
gene_names_all = rna.var_names.tolist()  # 全部2000基因
n_sample_cells = 3000
sc.pp.subsample(rna, n_obs=n_sample_cells, random_state=42)

rna_mat = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
data_idx = [i for i in range(rna_mat.shape[0])]


def get_attn_E(dataloader):
    attn_E_list = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True)
        for idx, data in enumerate(pbar):
            if torch.cuda.is_available():
                data['data'] = data['data'].cuda()
            mu, attn_E = model(data['data'], get_attn_E=True)
            attn_E_list.append(attn_E.cpu().numpy())

    attn_E_final = np.concatenate(attn_E_list, axis=0)
    attn_E_final = attn_E_final.mean(axis=0)  # (2001, 64)
    return attn_E_final


# ========== 计算原始数据的注意力权重 ==========
rna_dataset = Dataset(rna_mat, data_idx)
rna_dataloader = DataLoader(
    rna_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=4, 
    drop_last=False
)
attn_E_original = get_attn_E(rna_dataloader)


# ========== 扰动指定基因后计算注意力权重 ==========
if perturb_gene and perturb_gene in gene_names_all:
    perturb_idx = gene_names_all.index(perturb_gene)
    
    # 复制数据并将指定基因表达设为0
    rna_mat_perturbed = rna_mat.copy()
    rna_mat_perturbed[:, perturb_idx] = 0
    
    # 创建扰动后的dataloader
    rna_dataset_perturbed = Dataset(rna_mat_perturbed, data_idx)
    rna_dataloader_perturbed = DataLoader(
        rna_dataset_perturbed, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4, 
        drop_last=False
    )
    
    attn_E_perturbed = get_attn_E(rna_dataloader_perturbed)
    
    # ========== 计算注意力权重差值 ==========
    attn_E_diff = attn_E_original - attn_E_perturbed  # (2001, 64)
    
    # 分离CLS token和基因 (第0行是CLS, 1-2000是基因)
    gene_attn_diff = attn_E_diff[1:, :]  # (2000, 64)
    
    # ========== 64个heads平均，取绝对值 ==========
    # 在64个heads上平均，然后取绝对值
    gene_attn_change = gene_attn_diff.mean(axis=1)           # (2000,) 平均变化
    gene_attn_change_abs = np.abs(gene_attn_change)          # (2000,) 绝对值
    
    # ========== 找出变化最大的基因 ==========
    top_indices = np.argsort(gene_attn_change_abs)[-top_n:][::-1]
    
    print(f"扰动基因: {perturb_gene} (索引: {perturb_idx})")
    print(f"\n=== 注意力变化最大的Top {top_n} 基因 ===")
    for i, idx in enumerate(top_indices[:10]):
        gene_name = gene_names_all[idx]
        change = gene_attn_change[idx]
        marker = " <- 被扰动基因" if idx == perturb_idx else ""
        print(f"  {i+1}. {gene_name}: {change:.6f}{marker}")
    
    # ========== 保存到CSV ==========
    import pandas as pd
    
    # 创建完整结果表
    results_df = pd.DataFrame({
        'gene_name': gene_names_all,
        'gene_idx': range(2000),
        'is_perturbed': [i == perturb_idx for i in range(2000)],
        'attn_change_mean': gene_attn_change,      # 平均变化（有正负）
        'attn_change_abs': gene_attn_change_abs    # 绝对值
    })
    
    # 按绝对变化排序
    results_df_sorted = results_df.sort_values('attn_change_abs', ascending=False)
    
    # 保存所有基因
    csv_path_all = f'/home/output/{dataset_name}/{perturb_gene}/attn_diff_all_genes.csv'
    results_df_sorted.to_csv(csv_path_all, index=False)
    print(f"\n✓ 已保存所有基因: {csv_path_all}")
    
    # 保存Top N
    csv_path_top = f'/home/output/{dataset_name}/{perturb_gene}/attn_diff_top{top_n}_genes.csv'
    results_df_sorted.head(top_n).to_csv(csv_path_top, index=False)
    print(f"✓ 已保存Top {top_n}: {csv_path_top}")
    
    # 显示被扰动基因信息
    perturbed_info = results_df[results_df['is_perturbed'] == True]
    print(f"\n=== 被扰动基因 '{perturb_gene}' ===")
    print(f"  注意力变化: {perturbed_info['attn_change_mean'].values[0]:.6f}")
    print(f"  绝对变化:   {perturbed_info['attn_change_abs'].values[0]:.6f}")
    print(f"  排名:       {results_df_sorted.index.get_loc(perturb_idx) + 1}")

    # ========== 基因富集分析 ==========
    import gseapy as gp
    import os

    # 取Top N基因
    # top_gene_list = results_df_sorted.head(top_n)['gene_name'].tolist()

    top_df = results_df_sorted.head(top_n)
    filtered_df = top_df[
        (~top_df['gene_name'].str.contains(r'^AC|^AL|^RP11|^(TR|IG)'))
    ]
    top_gene_list = filtered_df['gene_name'].tolist()

    print(f"\n=== 开始基因富集分析 (Top {top_n}) ===")

    # 输出目录
    enrich_dir = f'/home/output/{dataset_name}/{perturb_gene}/enrichment'
    os.makedirs(enrich_dir, exist_ok=True)

    # ========== GO Biological Process ==========
    go_results = gp.enrichr(
        gene_list=top_gene_list,
        gene_sets='GO_Biological_Process_2021',
        organism='human',
        outdir=enrich_dir,
        cutoff=0.5
    )

    # 保存GO结果
    go_path = os.path.join(enrich_dir, 'go_enrichment.csv')
    go_results.results.to_csv(go_path, index=False)
    print(f"✓ GO富集已保存: {go_path}")
    
else:
    print(f"警告: 基因 '{perturb_gene}' 不在列表中或未指定")