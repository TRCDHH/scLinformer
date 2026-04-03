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

perturb_gene = "KLF1+CEBPA"
top_n = 100
model = RNAEncoder(n_genes=5045)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('/home/output/norman/model/rna_encoder.pth'))
n_cell = 32

rna_control = sc.read_h5ad('/home/data/norman/control.h5ad')
gene_names = rna_control.var['gene_name']
rna_control = rna_control.X.toarray()  if sparse.issparse(rna_control.X)  else np.asarray(rna_control.X)
rna_rd = sc.read_h5ad(f'/home/data/norman/{perturb_gene}.h5ad')
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
# print(delta_attn, rd_cell_attn, control_cell_attn)
np.save(f"/home/output/norman/{perturb_gene}_delta_attn.npy", delta_attn.cpu().numpy())

# ====================== ✅ 关键修正：支持多基因扰动 ======================
def get_top_perturb_genes(delta_attn, gene_names, perturb_gene, top_n=100):
    """
    修正版：支持 perturb_gene = "A+B" 多基因扰动
    自动合并两个扰动基因的重要性分数
    """
    # 1. 去掉 CLS token
    delta_attn = delta_attn[1:, 1:]

    # 2. 拆分扰动基因（支持 + 连接）
    perturb_genes = perturb_gene.split("+")
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # 3. 获取所有扰动基因的索引
    perturb_idxs = []
    for g in perturb_genes:
        if g not in gene_to_idx:
            raise ValueError(f"基因 {g} 不在基因列表中！")
        perturb_idxs.append(gene_to_idx[g])

    # 4. 合并多个源基因的 attention 影响（取绝对值求和 = 总重要性）
    total_importance = torch.zeros(delta_attn.shape[1], device=delta_attn.device)
    for idx in perturb_idxs:
        imp = torch.abs(delta_attn[idx])
        total_importance += imp

    # 5. 把扰动基因自身的重要性置0（排除自己）
    for idx in perturb_idxs:
        total_importance[idx] = 0

    # 6. 取TopN
    values, indices = torch.topk(total_importance, top_n)

    # 7. 构建输出表格
    df = pd.DataFrame({
        "source": perturb_gene,          # 保留 A+B 格式
        "target": [gene_names[i] for i in indices.tolist()],
        "importance": values.cpu().tolist()
    })

    return df

# ====================== 运行 & 保存 ======================
df = get_top_perturb_genes(delta_attn, gene_names, perturb_gene, top_n=top_n)
df.to_csv(f"/home/output/norman/{perturb_gene}_delta.csv", index=False)
print(f"✅ 完成！结果已保存：{perturb_gene}_delta.csv")
print(f"✅ 受 {perturb_gene} 影响最大的Top{top_n}个基因")