# %% [markdown]
# # Gene Interaction Network Visualization (Marker Gene KNN Graph)
# Pipeline:
# 1. Load model and data
# 2. Extract gene embeddings
# 3. Build KNN gene similarity network
# 4. Plot gene interaction network

# %%
# Import dependencies
import torch
import scanpy as sc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
from scLinformer.model import RNAEncoder
from scLinformer.utils import Dataset

# %%
# Configuration
dataset_name = "lungatlas"
rna_model_path = f'/home/output/{dataset_name}/model/rna_encoder.pth'
processed_rna_data_path = f'/home/output/{dataset_name}/processed_data/processed_rna.h5ad'
output_figure_path = f"/home/draw/fig3/gene_knn_final_{dataset_name}.png"

# %%
# Load trained model
model = RNAEncoder(n_genes=2000)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(rna_model_path))

# %%
# Load data and define marker genes
rna = sc.read_h5ad(processed_rna_data_path)
gene_names_all = rna.var_names.tolist()

marker_genes = [
    "CD3D","CD3E","CD4","CD8A","CD8B","IL7R","LEF1","CCR7","CD28","ICOS","TNFRSF4",
    "NKG7","GNLY","GZMA","GZMB","KLRD1","KLRB1","FCGR3A","FCGR3B",
    "CD19","MS4A1","CD79A","CD79B","CD24","CD38","IGHM","IGHD","JCHAIN","CD27",
    "CD14","FCGR1A","ITGAX","CD86","CD1C","CLEC10A","CLEC9A","XCR1","IRF8","LYZ",
    "IL1B","TNF","CCL3","CCL4","CXCL8","IL6","CSF1R","IL10","TGFB1",
    "ISG15","MX1","IFIT1","IFIT2","IFIT3","OAS1","OAS2","OAS3","STAT1","STAT2",
    "MKI67","PCNA","TOP2A","CCNB1","CDK1","TYMS","BIRC5","TUBA1B",
    "MPO","ELANE","AZU1","CEACAM8","CSF3R","LCAT","CEACAM6","MMP9","LTF","LCN2",
    "CD34","PROM1","KIT","FLT3","MEIS1","HOXA9","HLF","CRHBP","AVP","PROCR","SPINK2",
    "HBB","HBA1","HBA2","GATA1","KLF1","TFRC","AHSP","BPGM","HBD","HBE1",
    "PPBP","PF4","ITGA2B","TUBB1","GP1BA","GP9",
    "PECAM1","VWF","CDH5","KDR","FLT1",
    "COL1A1","COL3A1","DCN","LUM","PDGFRA","PDGFRB",
    "S100A8","S100A9","FCN1","CD16","FCER1A","CD33","TCL1A","CD200","ITGB2",
    "EPCAM","KRT18","KRT5","KRT17","SFTPA1","SFTPA2","SFTPB","SFTPC","SCGB1A1","SCGB3A2",
    "FOXJ1","MUC5AC","MUC1","AGER","PDPN","CLDN18","NAPSA","SOX9","NKX2-1",
    "MARCO","FABP4","CD68","SFTPC","SFTPD","CHI3L1","KIT","FOXF1","WT1",
    "CD27","CD45RA","CD45RO","PTPRC","SELL","ITGA4","ITGB7","CCR6","CXCR3","CXCR5",
    "IL2RA","IL2RG","IL4R","IL17A","IL17F","IFNG","FOXP3","CTLA4","PDCD1","LAG3"
]

# %%
# Filter marker gene indices
keep_idx = [i for i, g in enumerate(gene_names_all) if g in marker_genes]
gene_names_plot = [gene_names_all[i] for i in keep_idx]
print(f"Used marker genes count: {len(keep_idx)}")

# %%
# Subsample cells
n_sample_cells = 3000
sc.pp.subsample(rna, n_obs=n_sample_cells, random_state=42)

# %%
# Prepare dataset and dataloader
rna_mat = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
data_idx = list(range(rna_mat.shape[0]))

rna_dataset = Dataset(rna_mat, data_idx)
rna_dataloader = DataLoader(
    rna_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    drop_last=False
)

# %%
# Function to extract gene embeddings
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
    gene_embed_final = gene_embed_final.mean(axis=0)
    return gene_embed_final

# %%
# Get gene embeddings
gene_emb_full = get_gene_embed(rna_dataloader)
print("Gene embedding shape:", gene_emb_full.shape)

# %%
# Build KNN graph
gene_emb_plot = gene_emb_full[keep_idx]
k = 10
knn_graph = kneighbors_graph(
    gene_emb_plot,
    n_neighbors=k,
    metric="cosine",
    include_self=False
)

# %%
# Construct network graph
G = nx.from_scipy_sparse_array(knn_graph)
G = nx.relabel_nodes(G, {i: gene_names_plot[i] for i in range(len(gene_names_plot))})

# %%
# Plot and save gene network
plt.figure(figsize=(14, 14), dpi=300)
pos = nx.spring_layout(G, seed=42, k=0.15, iterations=100)

nx.draw_networkx_edges(G, pos, alpha=0.6, width=0.8, edge_color="#2E86AB")
nx.draw_networkx_nodes(G, pos, node_size=0, alpha=0)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight="normal", font_color="black")

plt.axis("off")
plt.title("Gene Interaction Network (Marker Genes)", fontsize=20, pad=20)
plt.tight_layout()
plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
plt.show()