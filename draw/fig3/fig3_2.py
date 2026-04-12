# %% [1] Import Libraries and Define All Custom Parameters (Top)
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

# ---------------------- CUSTOM PARAMETERS ----------------------
DATASET_NAME = "lungatlas"
N_GENES = 2000
MODEL_PATH = f'/home/output/{DATASET_NAME}/model/rna_encoder.pth'
DATA_PATH = f'/home/output/{DATASET_NAME}/processed_data/processed_rna.h5ad'
SAVE_FIG_PATH = f"/home/draw/fig3/gene_knn_final_{DATASET_NAME}.png"

# Sampling parameters
N_SAMPLE_CELLS = 3000
BATCH_SIZE = 64
NUM_WORKERS = 4
RANDOM_STATE = 42

# Graph parameters
K_NEIGHBORS = 10
LAYOUT_SEED = 42
SPRING_K = 0.15
SPRING_ITER = 100

# Plot parameters
PLOT_FIGSIZE = (14, 14)
DPI_VALUE = 300
EDGE_ALPHA = 0.6
EDGE_WIDTH = 0.8
EDGE_COLOR = "#2E86AB"
FONT_SIZE = 12
FONT_COLOR = "black"
TITLE_SIZE = 20
TITLE_PAD = 20

# Marker genes list
MARKER_GENES = [
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

# %% [2] Load Model
model = RNAEncoder(n_genes=N_GENES)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(MODEL_PATH))

# %% [3] Load and Preprocess Data
rna = sc.read_h5ad(DATA_PATH)
gene_names_all = rna.var_names.tolist()

# Filter marker genes
keep_idx = [i for i, g in enumerate(gene_names_all) if g in MARKER_GENES]
gene_names_plot = [gene_names_all[i] for i in keep_idx]
print("Number of genes for plotting:", len(keep_idx))

# Subsample cells
sc.pp.subsample(rna, n_obs=N_SAMPLE_CELLS, random_state=RANDOM_STATE)
rna_mat = rna.X.toarray() if sparse.issparse(rna.X) else np.asarray(rna.X)
data_idx = [i for i in range(rna_mat.shape[0])]

# %% [4] Create DataLoader
rna_dataset = Dataset(rna_mat, data_idx)
rna_dataloader = DataLoader(
    rna_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=False
)

# %% [5] Extract Gene Embeddings
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

gene_emb_full = get_gene_embed(rna_dataloader)
print("Gene embedding shape:", gene_emb_full.shape)

# %% [6] Filter Embeddings for Marker Genes
gene_emb_plot = gene_emb_full[keep_idx]

# %% [7] Build KNN Graph
knn_graph = kneighbors_graph(
    gene_emb_plot,
    n_neighbors=K_NEIGHBORS,
    metric="cosine",
    include_self=False
)

# %% [8] Plot Gene Interaction Network
G = nx.from_scipy_sparse_array(knn_graph)
G = nx.relabel_nodes(G, {i: gene_names_plot[i] for i in range(len(gene_names_plot))})

plt.figure(figsize=PLOT_FIGSIZE, dpi=DPI_VALUE)
pos = nx.spring_layout(G, seed=LAYOUT_SEED, k=SPRING_K, iterations=SPRING_ITER)

# Draw edges
nx.draw_networkx_edges(
    G, pos,
    alpha=EDGE_ALPHA,
    width=EDGE_WIDTH,
    edge_color=EDGE_COLOR
)

# Draw nodes (hidden)
nx.draw_networkx_nodes(
    G, pos,
    node_size=0,
    alpha=0
)

# Draw labels
nx.draw_networkx_labels(
    G, pos,
    font_size=FONT_SIZE,
    font_weight="normal",
    font_color=FONT_COLOR
)

plt.axis("off")
plt.title("Gene Interaction Network (Marker Genes)", fontsize=TITLE_SIZE, pad=TITLE_PAD)
plt.tight_layout()

plt.savefig(SAVE_FIG_PATH, dpi=DPI_VALUE, bbox_inches="tight")
plt.show()
plt.close()