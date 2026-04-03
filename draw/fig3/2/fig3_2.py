import torch
import scanpy as sc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
from scLinformer.model import RNAEncoder
from scLinformer.utils import Dataset

# ======================
# 你的模型加载（不变）
# ======================
dataset_name = "lungatlas"
model = RNAEncoder(n_genes=2000)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(f'/home/output/{dataset_name}/model/rna_encoder.pth'))

# ======================
# 读取数据
# ======================
rna = sc.read_h5ad(f'/home/output/{dataset_name}/processed_data/processed_rna.h5ad')
gene_names_all = rna.var_names.tolist()  # 全部2000基因

# ======================
# ✅ 论文专用：只画36个经典marker
# ======================
marker_genes = [
    # ======================
    # 通用免疫（所有数据集都有）
    # ======================
    "CD3D","CD3E","CD4","CD8A","CD8B","IL7R","LEF1","CCR7","CD28","ICOS","TNFRSF4",
    "NKG7","GNLY","GZMA","GZMB","KLRD1","KLRB1","FCGR3A","FCGR3B",
    "CD19","MS4A1","CD79A","CD79B","CD24","CD38","IGHM","IGHD","JCHAIN","CD27",
    "CD14","FCGR1A","ITGAX","CD86","CD1C","CLEC10A","CLEC9A","XCR1","IRF8","LYZ",
    "IL1B","TNF","CCL3","CCL4","CXCL8","IL6","CSF1R","IL10","TGFB1",
    "ISG15","MX1","IFIT1","IFIT2","IFIT3","OAS1","OAS2","OAS3","STAT1","STAT2",
    "MKI67","PCNA","TOP2A","CCNB1","CDK1","TYMS","BIRC5","TUBA1B",
    # ======================
    # BMMC 骨髓特有（必加！你现在缺的就是这些）
    # ======================
    "MPO","ELANE","AZU1","CEACAM8","CSF3R","LCAT","CEACAM6","MMP9","LTF","LCN2",
    "CD34","PROM1","KIT","FLT3","MEIS1","HOXA9","HLF","CRHBP","AVP","PROCR","SPINK2",
    "HBB","HBA1","HBA2","GATA1","KLF1","TFRC","AHSP","BPGM","HBD","HBE1",
    "PPBP","PF4","ITGA2B","TUBB1","GP1BA","GP9",
    "PECAM1","VWF","CDH5","KDR","FLT1",
    "COL1A1","COL3A1","DCN","LUM","PDGFRA","PDGFRB",
    # ======================
    # PBMC 特有（免疫外周血）
    # ======================
    "S100A8","S100A9","FCN1","CD16","FCER1A","CD33","TCL1A","CD200","ITGB2",
    # ======================
    # Lung 肺特有
    # ======================
    "EPCAM","KRT18","KRT5","KRT17","SFTPA1","SFTPA2","SFTPB","SFTPC","SCGB1A1","SCGB3A2",
    "FOXJ1","MUC5AC","MUC1","AGER","PDPN","CLDN18","NAPSA","SOX9","NKX2-1",
    "MARCO","FABP4","CD68","SFTPC","SFTPD","CHI3L1","KIT","FOXF1","WT1",
    # ======================
    # Immune 专属（免疫富集）
    # ======================
    "CD27","CD45RA","CD45RO","PTPRC","SELL","ITGA4","ITGB7","CCR6","CXCR3","CXCR5",
    "IL2RA","IL2RG","IL4R","IL17A","IL17F","IFNG","FOXP3","CTLA4","PDCD1","LAG3"
]

# 找到这些基因在2000基因里的位置
keep_idx = [i for i, g in enumerate(gene_names_all) if g in marker_genes]
gene_names_plot = [gene_names_all[i] for i in keep_idx]
print(f"✅ 实际画图基因数量：{len(keep_idx)} 个")

# ======================
# 只取3000个细胞（不动）
# ======================
n_sample_cells = 3000
sc.pp.subsample(rna, n_obs=n_sample_cells, random_state=42)

rna_mat = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
data_idx = [i for i in range(rna_mat.shape[0])]

# ======================
# Dataset & Dataloader（不动）
# ======================
rna_dataset = Dataset(rna_mat, data_idx)
rna_dataloader = DataLoader(
    rna_dataset, 
    batch_size=64, 
    shuffle=False, 
    num_workers=4, 
    drop_last=False
)

# ======================
# 提取基因embedding（不动，模型依旧输入2000基因）
# ======================
def get_gene_embed(dataloader):
    gene_embed_list = []
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True)
        for idx, data in enumerate(pbar):
            if torch.cuda.is_available():
                data['data'] = data['data'].cuda()
            mu, gene_embed = model(data['data'], get_gene_embed=True)
            gene_embed_list.append(gene_embed.cpu().numpy())

    gene_embed_final = np.concatenate(gene_embed_list, axis=0)
    gene_embed_final = gene_embed_final.mean(axis=0)  # (2000, 128)
    return gene_embed_final

# ======================
# 获取完整2000基因embedding
# ======================
gene_emb_full = get_gene_embed(rna_dataloader)
print("✅ 模型输出基因embedding形状:", gene_emb_full.shape)  # (2000,128)

# ======================
# ✅ 只筛选marker基因用于画图
# ======================
gene_emb_plot = gene_emb_full[keep_idx]

# ======================
# 构建 KNN 图（K=10 黄金值）
# ======================
k = 10
knn_graph = kneighbors_graph(
    gene_emb_plot, 
    n_neighbors=k, 
    metric="cosine",
    include_self=False
)

# ======================
# 绘制基因互作网络（最终定稿：无圈、深蓝线、清晰）
# ======================
G = nx.from_scipy_sparse_array(knn_graph)
G = nx.relabel_nodes(G, {i: gene_names_plot[i] for i in range(len(gene_names_plot))})

plt.figure(figsize=(14, 14), dpi=300)

# 1. 布局：让节点紧凑但不拥挤
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)

# 2. 画边（关键：深蓝色、不透明，看得清！）
nx.draw_networkx_edges(
    G, pos,
    alpha=0.6,      # 不透明，清晰
    width=0.8,      # 线条稍粗
    edge_color="#2E86AB"  # 深蓝色，比浅蓝色清晰很多
)

# 3. 画节点（关键：完全隐藏圆圈，node_size=0 或 linewidth=0）
# 这里用 node_size=0 彻底隐藏圆圈，只保留文字和线条
nx.draw_networkx_nodes(
    G, pos,
    node_size=0,    # 节点大小设为0 → 完全消失
    alpha=0
)

# 4. 画基因名（字体大小、颜色都调好）
nx.draw_networkx_labels(
    G, pos,
    font_size=12,           # 大小合适
    font_weight="normal",  # 不加粗，更简洁
    font_color="black"
)

plt.axis("off")
plt.title("Gene Interaction Network (Marker Genes)", fontsize=20, pad=20)
plt.tight_layout()
plt.savefig(f"/home/draw/fig3/gene_knn_final_{dataset_name}.png", dpi=300, bbox_inches="tight")
plt.show()