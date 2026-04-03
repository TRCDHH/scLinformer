from scLinformer.scLinformerModel import Model
import torch
import pandas as pd
import gseapy as gp
import os
from tqdm import tqdm

# plot_basis_go_enrichment_dual(model, outdir="/home/output/attention")
def run_all_basis_go_analysis(
    model,
    dataset_name,
    outdir,
    topk_genes=100,
    gene_set="GO_Biological_Process_2021",
    organism="human",
    adj_p_cutoff=0.05,
    save_top_terms=5,
    tensor = None
):
    """
    统计所有 basis 的 GO enrichment，
    只保存有显著结果的 basis，
    每个 basis 只保存前 save_top_terms 个 GO term，
    所有结果汇总到一个 csv。
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== Running ALL basis GO analysis for {dataset_name} =====")

    # --------------------------------------------------
    # 1️⃣ 取 E
    # --------------------------------------------------
    if tensor is None:
        E = model.rna_encoder.trans[0].attn.E_k.detach().cpu()
    else:
        E = tensor
    gene_names = model.rna_data_var.index.tolist()

    n_basis = E.shape[1]
    n_head = E.shape[0]

    print(f"Total basis: {n_basis}")

    significant_basis_count = 0
    summary_records = []

    # --------------------------------------------------
    # 2️⃣ 遍历所有 basis 和 head
    # -------------------------------------------------- 
    for basis_idx in tqdm(range(n_basis), desc=f"{dataset_name} basis"):
        for head_idx in range(n_head):
            E_head = E[head_idx]

            weights = E_head[basis_idx] 

            # 取 topk genes（按绝对值）
            _, idx = torch.topk(torch.abs(weights), topk_genes)
            idx = idx.reshape(-1).tolist() 
            top_genes = [
                gene_names[i - 1]
                for i in idx
                if i > 0 and (i - 1) < len(gene_names)
            ]

            # ----------------------------
            # GO enrichment
            # ----------------------------
            try:
                enr = gp.enrichr(
                    gene_list=top_genes,
                    gene_sets=[gene_set],
                    organism=organism,
                    outdir=None,
                    cutoff=0.5
                )
            except Exception as e:
                print(f"Basis {basis_idx} error: {e}")
                continue

            results = enr.results

            if results.empty:
                continue

            # 筛选显著
            sig_results = results[
                results["Adjusted P-value"] < adj_p_cutoff
            ]

            if sig_results.empty:
                continue

            significant_basis_count += 1

            # 只保存前 save_top_terms 个 GO
            top_sig = (
                sig_results
                .sort_values("Adjusted P-value")
                .head(save_top_terms)
            )

            for _, row in top_sig.iterrows():
                summary_records.append({
                    "dataset": dataset_name,
                    "head_idx": head_idx,
                    "basis_idx": basis_idx,
                    "GO_term": row["Term"],
                    "Adjusted_P_value": row["Adjusted P-value"],
                    "Combined_Score": row["Combined Score"]
                })

    # --------------------------------------------------
    # 3️⃣ 汇总保存
    # --------------------------------------------------
    summary_df = pd.DataFrame(summary_records)

    summary_path = os.path.join(
        outdir,
        f"{dataset_name}_ALL_basis_significant_GO_summary.csv"
    )

    summary_df.to_csv(summary_path, index=False)

    enrichment_ratio = significant_basis_count / (n_basis*n_head)

    print("\n===== Summary =====")
    print(f"Total basis: {n_basis}")
    print(f"Significant basis: {significant_basis_count}")
    print(f"Enrichment ratio: {enrichment_ratio:.3f}")
    print(f"Saved to: {summary_path}")

    return summary_df, enrichment_ratio


def save_flattened_E_matrix(
    model,
    dataset_name,
    outdir,
    tensor=None,
    remove_cls_token=True,
    abs_weight=False
):
    """
    将 E (n_head, k, 1+n_gene) 展平成 long format CSV：

    列：
    head_idx, basis_idx, gene_idx, gene_name, weight

    参数：
    - remove_cls_token: 是否去掉 index=0（CLS token）
    - abs_weight: 是否保存绝对值（用于重要性分析）
    """

    os.makedirs(outdir, exist_ok=True)

    print(f"\n===== Saving flattened E matrix for {dataset_name} =====")

    # --------------------------------------------------
    # 1️⃣ 取 E
    # --------------------------------------------------
    if tensor is None:
        E = model.rna_encoder.trans[0].attn.E_k.detach().cpu()
    else:
        E = tensor

    gene_names = model.rna_data_var.index.tolist()

    n_head, n_basis, n_gene_total = E.shape

    print(f"E shape: {E.shape}")

    records = []

    # --------------------------------------------------
    # 2️⃣ 遍历
    # --------------------------------------------------
    for head_idx in tqdm(range(n_head), desc="Flatten E"):
        for basis_idx in range(n_basis):

            weights = E[head_idx, basis_idx]  # shape: (1 + n_gene)

            for gene_idx in range(n_gene_total):

                # 跳过 CLS token
                if remove_cls_token and gene_idx == 0:
                    continue

                real_idx = gene_idx - 1 if remove_cls_token else gene_idx

                if real_idx >= len(gene_names) or real_idx < 0:
                    continue

                weight = weights[gene_idx].item()

                if abs_weight:
                    weight = abs(weight)

                records.append({
                    "dataset": dataset_name,
                    "head_idx": head_idx,
                    "basis_idx": basis_idx,
                    "gene_idx": real_idx,
                    "gene_name": gene_names[real_idx],
                    "weight": weight
                })

    # --------------------------------------------------
    # 3️⃣ 保存
    # --------------------------------------------------
    df = pd.DataFrame(records)

    save_path = os.path.join(
        outdir,
        f"{dataset_name}_flattened_E_matrix.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"\nSaved flattened matrix to: {save_path}")
    print(f"Total rows: {len(df)}")

    return df


dataset_list = ["bmmc", "immune", "lungatlas", "covid"]
# dataset_list = ["covid"]
# parameters
for dataset_name in dataset_list:
    rna_data_path = f'/home/output/{dataset_name}/processed_data/processed_rna.h5ad'
    rna_model_path = f'/home/output/{dataset_name}/model/rna_encoder.pth'
    output_path = f'/home/evaluation/basis/{dataset_name}'

    model = Model(
        RNAData = rna_data_path,
        output_path = output_path,
        process_data=False
    )
    model.rna_encoder.load_state_dict(torch.load(rna_model_path))

    # run_all_basis_go_analysis(
    #     model=model,
    #     dataset_name=dataset_name,
    #     outdir=output_path,
    #     topk_genes=30,
    #     organism="human"
    # )

    save_flattened_E_matrix(
        model = model,
        dataset_name = dataset_name,
        outdir = output_path,
        tensor=None,
        remove_cls_token=True,
        abs_weight=False
    )

    del model