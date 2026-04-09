import torch
import scanpy as sc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from scLinformer.model import RNAEncoder
from scLinformer.utils import Dataset
import os
import pandas as pd
import gseapy as gp

# ================================
#  CONFIGURATION
# ================================
DATASET_NAME = "newcovid"
MODEL_PATH = f"/home/output/{DATASET_NAME}/model/rna_encoder.pth"
DATA_PATH = f"/home/output/{DATASET_NAME}/processed_data/processed_rna.h5ad"
OUTPUT_DIR = f"/home/output/{DATASET_NAME}"

PERTURB_GENE = "IRF7"
TOP_N = 50
N_SAMPLES = 3000

BATCH_SIZE = 64
NUM_WORKERS = 4

N_GENES = 2000

# ================================
# OUTPUT DIR
# ================================
os.makedirs(f"{OUTPUT_DIR}/{PERTURB_GENE}", exist_ok=True)

# ================================
# MODEL
# ================================
model = RNAEncoder(n_genes=N_GENES)
if torch.cuda.is_available():
    model = model.cuda()

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ================================
# DATA
# ================================
rna = sc.read_h5ad(DATA_PATH)
gene_names_all = rna.var_names.tolist()

print(f"Loaded genes: {len(gene_names_all)}")

sc.pp.subsample(rna, n_obs=N_SAMPLES, random_state=42)

rna_mat = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
data_idx = list(range(rna_mat.shape[0]))

# ================================
# ATTENTION EXTRACTION FUNCTION
# ================================
def get_attn_E(dataloader):
    attn_E_list = []

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for data in pbar:
            if torch.cuda.is_available():
                data["data"] = data["data"].cuda()

            _, attn_E = model(data["data"], get_attn_E=True)
            attn_E_list.append(attn_E.cpu().numpy())

    attn_E = np.concatenate(attn_E_list, axis=0)
    attn_E = attn_E.mean(axis=0)  # (2001, 64)

    return attn_E

# ================================
# ORIGINAL ATTENTION
# ================================
rna_dataset = Dataset(rna_mat, data_idx)
rna_loader = DataLoader(
    rna_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False
)

attn_E_original = get_attn_E(rna_loader)

# ================================
# GENE PERTURBATION
# ================================
if PERTURB_GENE and PERTURB_GENE in gene_names_all:

    perturb_idx = gene_names_all.index(PERTURB_GENE)

    # perturb expression (set to 0)
    rna_mat_perturbed = rna_mat.copy()
    rna_mat_perturbed[:, perturb_idx] = 0

    rna_dataset_perturbed = Dataset(rna_mat_perturbed, data_idx)
    rna_loader_perturbed = DataLoader(
        rna_dataset_perturbed,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    attn_E_perturbed = get_attn_E(rna_loader_perturbed)

    # ================================
    # ATTENTION DIFFERENCE
    # ================================
    attn_E_diff = attn_E_original - attn_E_perturbed  # (2001, 64)

    gene_attn_diff = attn_E_diff[1:, :]  # remove CLS token

    gene_attn_change = gene_attn_diff.mean(axis=1)
    gene_attn_change_abs = np.abs(gene_attn_change)

    top_indices = np.argsort(gene_attn_change_abs)[-TOP_N:][::-1]

    print(f"\nPerturbed gene: {PERTURB_GENE} (index: {perturb_idx})")
    print(f"\n=== Top {TOP_N} genes with highest attention change ===")

    for i, idx in enumerate(top_indices[:10]):
        gene_name = gene_names_all[idx]
        change = gene_attn_change[idx]
        mark = " <- perturbed gene" if idx == perturb_idx else ""
        print(f"{i+1}. {gene_name}: {change:.6f}{mark}")

    # ================================
    # SAVE RESULTS
    # ================================
    results_df = pd.DataFrame({
        "gene_name": gene_names_all,
        "gene_idx": range(N_GENES),
        "is_perturbed": [i == perturb_idx for i in range(N_GENES)],
        "attn_change_mean": gene_attn_change,
        "attn_change_abs": gene_attn_change_abs
    })

    results_df_sorted = results_df.sort_values("attn_change_abs", ascending=False)

    out_dir = f"{OUTPUT_DIR}/{PERTURB_GENE}"

    results_df_sorted.to_csv(
        f"{out_dir}/attn_diff_all_genes.csv",
        index=False
    )

    results_df_sorted.head(TOP_N).to_csv(
        f"{out_dir}/attn_diff_top{TOP_N}_genes.csv",
        index=False
    )

    print(f"\n✓ Saved results to: {out_dir}")

    # ================================
    # PERTURBED GENE SUMMARY
    # ================================
    perturbed_info = results_df[results_df["is_perturbed"]]

    print(f"\n=== Perturbed Gene Summary: {PERTURB_GENE} ===")
    print(f"Mean change: {perturbed_info['attn_change_mean'].values[0]:.6f}")
    print(f"Abs change:  {perturbed_info['attn_change_abs'].values[0]:.6f}")
    print(
        f"Rank: {results_df_sorted.index.get_loc(perturb_idx) + 1}"
    )

    # ================================
    # GENE SET ENRICHMENT (GO BP)
    # ================================
    top_df = results_df_sorted.head(TOP_N)

    filtered_df = top_df[
        ~top_df["gene_name"].str.contains(r"^AC|^AL|^RP11|^(TR|IG)")
    ]

    top_gene_list = filtered_df["gene_name"].tolist()

    enrich_dir = f"{out_dir}/enrichment"
    os.makedirs(enrich_dir, exist_ok=True)

    print(f"\n=== Running GO enrichment (Top {TOP_N}) ===")

    go_results = gp.enrichr(
        gene_list=top_gene_list,
        gene_sets="GO_Biological_Process_2021",
        organism="human",
        outdir=enrich_dir,
        cutoff=0.5
    )

    go_results.results.to_csv(
        f"{enrich_dir}/go_enrichment.csv",
        index=False
    )

    print(f"✓ GO enrichment saved: {enrich_dir}")

else:
    print(f"Warning: gene '{PERTURB_GENE}' not found in dataset")