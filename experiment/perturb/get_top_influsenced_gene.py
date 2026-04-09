from scLinformer.scLinformerModel import Model
from scLinformer.utils import RNA_data_preprocessing, Dataset
from scLinformer.model import RNAEncoder
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
import pandas as pd
from scipy import sparse

# parameters
perturb_gene = "gene"  # Name of the perturbed gene.
rna_encoder_model_path = '/home/output/norman/model/rna_encoder.pth'  # Path to the pretrained RNA encoder model
processed_perturb_data_path = '/home/data/perturb.h5ad'  # Path to the perturbed dataset (AnnData format, .h5ad)
processed_control_data_path = '/home/data/control.h5ad'  # Path to the control (non-perturbed) dataset (AnnData format, .h5ad)
output_path = '/home/output'  # Directory where results will be saved
top_n = 100  # Number of top genes to select based on importance scores

model = RNAEncoder()
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(rna_encoder_model_path))
n_cell = 32

rna_control = sc.read_h5ad(processed_control_data_path)
gene_names = rna_control.var['gene_name']
rna_control = rna_control.X.toarray()  if sparse.issparse(rna_control.X)  else np.asarray(rna_control.X)
rna_rd = sc.read_h5ad(processed_perturb_data_path)
rna_rd = rna_rd.X.toarray()  if sparse.issparse(rna_rd.X)  else np.asarray(rna_rd.X)

idx_ctrl = np.random.choice(rna_control.shape[0], n_cell, replace=False)
idx_rd = np.random.choice(rna_rd.shape[0], n_cell, replace=False)

control_dataset = Dataset(rna_control, idx_ctrl)
rd_dataset = Dataset(rna_rd, idx_rd)
control_dataloader = DataLoader(control_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)
rd_dataloader = DataLoader(rd_dataset, batch_size=64, shuffle=False, num_workers=4, drop_last=False)

def get_attnetion(dataloader):
    attn_sum = None
    total_cells = 0
    model.eval()
    with torch.no_grad():
        with tqdm(total = len(dataloader), dynamic_ncols=True):
            print('rna predicting...')
            for idx, data in enumerate(dataloader):
                if torch.cuda.is_available():
                    data['data'] = data['data'].cuda()
                    mu, attn = model(data['data'], get_attn=True)
                    attn = attn.detach().cpu().sum(dim=0)  # [G, G]
                    if attn_sum is None:
                        attn_sum = attn
                    else:
                        attn_sum += attn
                    total_cells += attn.shape[0]
    return attn_sum / total_cells

control_cell_attn = get_attnetion(control_dataloader)
rd_cell_attn = get_attnetion(rd_dataloader)
delta_attn = rd_cell_attn - control_cell_attn
np.save(f"output_path/{perturb_gene}_delta_attn.npy", delta_attn.cpu().numpy())

def get_top_perturb_genes(delta_attn, gene_names, perturb_gene, top_n=100):
    """
    Revised version: supports multi-gene perturbation such as "A+B".
    Automatically aggregates the importance scores from multiple perturbed genes.
    """
    # 1. Remove CLS token
    delta_attn = delta_attn[1:, 1:]

    # 2. Parse perturbation genes (support "+" separator)
    perturb_genes = perturb_gene.split("+")
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # 3. Get indices of all perturbed genes
    perturb_idxs = []
    for g in perturb_genes:
        if g not in gene_to_idx:
            raise ValueError(f"Gene {g} is not found in the gene list!")
        perturb_idxs.append(gene_to_idx[g])

    # 4. Aggregate attention impact from multiple source genes
    #    (sum of absolute values = overall importance)
    total_importance = torch.zeros(delta_attn.shape[1], device=delta_attn.device)
    for idx in perturb_idxs:
        imp = torch.abs(delta_attn[idx])
        total_importance += imp

    # 5. Set self-importance of perturbed genes to zero (exclude themselves)
    for idx in perturb_idxs:
        total_importance[idx] = 0

    # 6. Select Top-N genes
    values, indices = torch.topk(total_importance, top_n)

    # 7. Build output DataFrame
    df = pd.DataFrame({
        "source": perturb_gene,  # keep "A+B" format
        "target": [gene_names[i] for i in indices.tolist()],
        "importance": values.cpu().tolist()
    })

    return df

# ====================== Run & Save ======================
df = get_top_perturb_genes(delta_attn, gene_names, perturb_gene, top_n=top_n)
df.to_csv(f"output_path/{perturb_gene}_delta.csv", index=False)
print(f"Done! Results saved to: {perturb_gene}_delta.csv")
print(f"Top {top_n} genes most influenced by {perturb_gene}")