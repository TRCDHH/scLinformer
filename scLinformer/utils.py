import os
import random
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import sparse
import anndata as ad
import scanpy as sc
from sklearn import metrics
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import connected_components
from matplotlib.backends.backend_pdf import PdfPages

class Dataset(Dataset):
    def __init__(self, data, ids, gene_idx = None, cell_type_idx=None, batch_idx=None):
        self.dataset = data        
        self.ids = ids
        if gene_idx is not None:
            if isinstance(gene_idx, pd.Series):
                gene_idx = gene_idx.to_numpy()
            if gene_idx.dtype.kind in {'U', 'O'}:  # object/string
                self.gene2idx = {g: i for i, g in enumerate(gene_idx)}
                gene_idx = np.array([self.gene2idx[g] for g in gene_idx], dtype=np.int64)
            self.gene_idx = torch.from_numpy(gene_idx).long()  # LongTensor
        else:
            self.gene_idx = None
        # cell_type_idx 
        if cell_type_idx is not None:
            if isinstance(cell_type_idx, pd.Series):
                if hasattr(cell_type_idx, 'cat'):
                    cell_type_idx = cell_type_idx.cat.codes.to_numpy()
                else:
                    cell_type_idx = cell_type_idx.to_numpy()
            if cell_type_idx.dtype == np.object_ or cell_type_idx.dtype.kind in {'U', 'O'}:
                unique_types = np.unique(cell_type_idx)
                type2idx = {t: i for i, t in enumerate(unique_types)}
                cell_type_idx = np.array([type2idx[t] for t in cell_type_idx], dtype=np.int64)
            cell_type_idx = cell_type_idx.astype(np.int64)
            self.cell_type_idx = torch.from_numpy(cell_type_idx).long()
        else:
            self.cell_type_idx = None
        # batch_idx
        if batch_idx is not None:
            if isinstance(batch_idx, pd.Series):
                if hasattr(batch_idx, 'cat'):
                    batch_idx = batch_idx.cat.codes.to_numpy()
                else:
                    batch_idx = batch_idx.to_numpy()
            if batch_idx.dtype == np.object_ or batch_idx.dtype.kind in {'U', 'O'}:
                unique_types = np.unique(batch_idx)
                type2idx = {t: i for i, t in enumerate(unique_types)}
                batch_idx = np.array([type2idx[t] for t in batch_idx], dtype=np.int64)
            batch_idx = batch_idx.astype(np.int64)
            self.batch_idx = torch.from_numpy(batch_idx).long()
        else:
            self.batch_idx = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
         # Extract expression vector for a single cell
        x = self.dataset[self.ids[idx], :]
        # Remove missing genes (filled with -1 after outer concat)
        valid_mask = x != -1
        x = x[valid_mask]
        # Convert to tensor
        x = torch.from_numpy(x).float()
        output = {"data": x}
        # gene idx
        if self.gene_idx is not None:
            gene_idx = self.gene_idx[valid_mask]
            output["gene_idx"] = gene_idx
        # cell type idx
        if self.cell_type_idx is not None:
            output["cell_type_idx"] = self.cell_type_idx[self.ids[idx]]
        # batch idx
        if self.batch_idx is not None:
            output["batch_idx"] = self.batch_idx[self.ids[idx]]

        return output
    
def setSeed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

class EarlyStopping:           
    def __init__(self, patience, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        model.save_model_dict()
        self.val_loss_min = val_loss


def tensor2adata(r2r, mu,val=1e-4):
    r2r = torch.cat(r2r)
    mu = torch.cat(mu)
    
    r2r = torch.masked_fill(r2r, r2r < val, 0)
    r2r = sparse.csr_matrix(r2r)
    r2r = ad.AnnData(r2r)
    mu = mu.numpy() 
    r2r.obsm["X_emb"] = mu
    return r2r


def calculate_cluster_index(adata, type):
    if not "neighbors" in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep='X_emb')
    
    sc.tl.leiden(adata)
    
    ARI = metrics.adjusted_rand_score(adata.obs[type], adata.obs['leiden'])
    AMI = metrics.adjusted_mutual_info_score(adata.obs[type], adata.obs['leiden'])
    NMI = metrics.normalized_mutual_info_score(adata.obs[type], adata.obs['leiden'])
    HOM = metrics.homogeneity_score(adata.obs[type], adata.obs['leiden'])
    
    return ARI, AMI, NMI, HOM

def compute_batch_metrics(adata, batch_key='batch', k=15):
    """
    Compute batch effect metrics: Batch ASW and Graph Connectivity
    Returns a dict with both metrics
    """
    # Batch ASW (silhouette score for batch mixing)
    X = adata.obsm['X_emb']  
    labels = adata.obs[batch_key].astype(str).values
    sil_samples = silhouette_samples(X, labels, metric='euclidean')
    batch_asw = np.mean(sil_samples)

    # Graph Connectivity (GC)
    if not "neighbors" in adata.uns.keys():
        sc.pp.neighbors(adata, use_rep='X_emb')
    gc_scores = []
    for batch in adata.obs[batch_key].unique():
        sub = adata[adata.obs[batch_key]==batch]
        n_comp, labels_sub = connected_components(csgraph=sub.obsp['connectivities'], directed=False)
        gc = np.max(np.bincount(labels_sub)) / sub.n_obs
        gc_scores.append(gc)
    graph_conn = np.mean(gc_scores)

    return {'Batch_ASW': batch_asw, 'Graph_Connectivity': graph_conn}

def draw_tsne(data, title, color):
    sc.settings.set_figure_params(dpi=120, facecolor='white')
    sc.pp.neighbors(data, use_rep='X_emb')
    sc.tl.tsne(data, use_rep='X_emb')
    fig = sc.pl.tsne(data, color=color, title=title, return_fig=True)
    return fig


def draw_reg_plot(eval_adata,
                  cell_type,
                  reg_type='mean',
                  axis_keys={"x": "pred", "y": "stimulated"},
                  condition_key='condition',
                  gene_draw=None,
                  top_gene_list=None,
                  title=None,
                  fontsize=14
                 ):
    df_case = eval_adata[(eval_adata.obs[condition_key]==axis_keys["y"])].to_df()
    df_pred = eval_adata[(eval_adata.obs[condition_key]==axis_keys["x"])].to_df()
    if reg_type=='mean':
        mean_case = df_case.mean().values.reshape(-1, 1)
        mean_pred = df_pred.mean().values.reshape(-1, 1)
    elif reg_type=='var':
        mean_case = df_case.var().values.reshape(-1, 1)
        mean_pred = df_pred.var().values.reshape(-1, 1)
    data = np.hstack((mean_case, mean_pred))
    data_df = pd.DataFrame(data, columns=['case', 'predict'], index=df_case.columns)
    
    sns.set(color_codes=True)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    sns.regplot(x='case', y='predict', data=data_df, ax=ax)
    if gene_draw is not None:
        texts = []
        x = mean_case
        y = mean_pred
        for i in gene_draw:
            j = eval_adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            ax.plot(x_bar, y_bar, "o", color="red", markersize=5)
        adjust_text(
                texts,
                x=x,
                y=y,
                arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
                force_points=(0.0, 0.0),
        )
    if top_gene_list is not None:
        data_deg = data_df.loc[top_gene_list, :]
        r_top = round(data_deg['case'].corr(data_deg['predict'], method='pearson'), 3)
        xt = 0.1 * np.max(data_df['case'])
        yt = 0.85 * np.max(data_df['predict'])
        ax.text(xt, yt, s='$R^2_{top 100 genes}$=' + str(round(r_top*r_top,3)), fontsize=fontsize, color='black')
    r = round(data_df['case'].corr(data_df['predict'], method='pearson'), 3)
    xt = 0.1 * np.max(data_df['case'])
    yt = 0.75 * np.max(data_df['predict'])
    ax.text(xt, yt, s='$R^2_{all genes}$=' + str(round(r*r,3)), fontsize=fontsize, color='black')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('The Linear Regression of True and Predict Expression '+ reg_type +' of ' + cell_type)

    return round(r*r,3),round(r_top*r_top,3), fig


def RNA_data_preprocessing(
    RNA_data,
    normalize_total = True,
    log1p = True,
    use_hvg = True,
    n_top_genes = 3000,
    save_data = False,
    file_path = None,
    use_universal_model = False,
    use_bins = False,
    n_bins = 16,
    use_cell_type: bool = True,
    use_batch: bool = True,
    gene_vocab = None
):
    RNA_data_processed = RNA_data.copy()
    
    RNA_data_processed.var_names_make_unique()
    
    if not normalize_total or not log1p or not use_hvg:
        print('prefered to process data with default settings to keep the best result.')

    if use_universal_model and gene_vocab is not None:
        print('filter genes not in universal gene list.')
        RNA_data_processed = add_gene_id_and_filter(RNA_data_processed, gene_vocab)

    if use_cell_type:
        print('get cell type idx.')
        RNA_data_processed.obs['cell_type_idx'] = (
            RNA_data_processed.obs['cell_type']
            .astype('category')
            .cat.codes
        )

    if use_batch:
        print('get batch idx.')
        RNA_data_processed.obs['batch_idx'] = (
            RNA_data_processed.obs['batch']
            .astype('category')
            .cat.codes
        )
    
    if normalize_total:
        print('normalize size factor.')
        sc.pp.normalize_total(RNA_data_processed)
        
    if log1p:
        print('log transform RNA data.')
        sc.pp.log1p(RNA_data_processed)
    
    if use_hvg:
        print('choose top '+str(n_top_genes)+' genes for following training.')
        sc.pp.highly_variable_genes(RNA_data_processed, n_top_genes=n_top_genes)
        RNA_data_processed = RNA_data_processed[:, RNA_data_processed.var['highly_variable']]

    if use_bins:
        print("Binning RNA expression...")
        X = RNA_data_processed.X
        if sparse.issparse(X):
            data_sample = np.random.choice(X.data, size=min(len(X.data), 500_000), replace=False)
        else:
            flat = X.ravel()
            data_sample = np.random.choice(flat, size=min(len(flat), 500_000), replace=False)
        bins = np.quantile(data_sample[data_sample > 0], np.linspace(0, 1, n_bins))
        bins = np.unique(np.concatenate([[0], bins]))  
        if sparse.issparse(X):
            X.data = np.digitize(X.data, bins[1:]) - 1 
            X.data = np.clip(X.data, 0, n_bins - 1)
        else:
            X = np.digitize(X, bins[1:]) - 1
            X = np.clip(X, 0, n_bins - 1)
            RNA_data_processed.X = X
    
    if save_data:
        print('writing processed RNA data to target file.')
        if not use_universal_model:
            RNA_data_processed.write_h5ad(os.path.join(file_path, 'processed_rna.h5ad'))
        else:
            RNA_data_processed.write_h5ad(os.path.join(file_path, 'processed_rna_uni.h5ad'))
            
    return RNA_data_processed


def five_fold_split_dataset(
    RNA_data, 
    seed = 114514
):
    if not seed is None:
        setSeed(seed)
    temp = [i for i in range(len(RNA_data.obs_names))]
    random.shuffle(temp)
    id_list = []
    test_count = int(0.2 * len(temp))
    validation_count = int(0.16 * len(temp))
    for i in range(5):
        test_id = temp[: test_count]
        validation_id = temp[test_count: test_count + validation_count]
        train_id = temp[test_count + validation_count:]
        temp.extend(test_id)
        temp = temp[test_count: ]
        id_list.append([train_id, validation_id, test_id])
    return id_list


def add_gene_id_and_filter(adata, gene_vocab):
    genes = adata.var_names.tolist()
    gene_ids = np.array([gene_vocab.get(g, -1) for g in genes])
    adata.var["gene_idx"] = gene_ids

    total = len(genes)
    matched = np.sum(gene_ids != -1)
    print(f"Total genes: {total}")
    print(f"Matched genes: {matched}")
    print(f"Coverage: {matched/total:.4f}")

    mask = gene_ids != -1
    adata = adata[:, mask].copy()
    print(f"Removed {np.sum(~mask)} unmatched genes")

    return adata

def evaluate_sc_embedding(
        adata,
        embedding_key="latent",
        celltype_key="cell_type",
        batch_key="batch",
        outdir="./evaluation",
        n_neighbors=15,
        resolutions=[0.3, 0.5, 0.8, 1.0]
    ):

    os.makedirs(outdir, exist_ok=True)

    # 设置图大小（解决 PDF 太窄）
    sc.set_figure_params(dpi=120, figsize=(8,8))

    print("Computing neighbors graph...")
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)

    print("Computing UMAP...")
    sc.tl.umap(adata)

    print("Computing tSNE...")
    sc.tl.tsne(adata, use_rep=embedding_key)

    ############################################
    # clustering metrics
    ############################################

    results = []

    print("Evaluating clustering...")

    best_ari = -1
    best_metrics = None

    for r in resolutions:

        key = f"leiden_{r}"
        sc.tl.leiden(adata, resolution=r, key_added=key)

        ari = metrics.adjusted_rand_score(
            adata.obs[celltype_key],
            adata.obs[key]
        )

        ami = metrics.adjusted_mutual_info_score(
            adata.obs[celltype_key],
            adata.obs[key]
        )

        nmi = metrics.normalized_mutual_info_score(
            adata.obs[celltype_key],
            adata.obs[key]
        )

        hom = metrics.homogeneity_score(
            adata.obs[celltype_key],
            adata.obs[key]
        )

        results.append({
            "resolution": r,
            "ARI": ari,
            "AMI": ami,
            "NMI": nmi,
            "HOM": hom
        })

        if ari > best_ari:
            best_ari = ari
            best_metrics = (ari, ami, nmi, hom)

    cluster_df = pd.DataFrame(results)

    ############################################
    # batch metrics
    ############################################

    print("Computing batch metrics...")

    X = adata.obsm[embedding_key]

    batch_labels = adata.obs[batch_key].astype(str).values
    sil_batch = silhouette_samples(X, batch_labels, metric="euclidean")
    batch_asw = 1 - np.mean(np.abs(sil_batch))

    ############################################
    # cell-type ASW
    ############################################

    cell_labels = adata.obs[celltype_key].astype(str).values
    sil_cell = silhouette_samples(X, cell_labels, metric="euclidean")

    # scale to 0-1 (standard in single-cell papers)
    cell_asw = (np.mean(sil_cell) + 1) / 2

    ############################################
    # Graph connectivity
    ############################################

    gc_scores = []

    for b in adata.obs[batch_key].unique():

        sub = adata[adata.obs[batch_key]==b].copy()

        sc.pp.neighbors(sub, use_rep=embedding_key, n_neighbors=n_neighbors)

        n_comp, labels = connected_components(
            csgraph=sub.obsp["connectivities"],
            directed=False
        )

        gc = np.max(np.bincount(labels)) / sub.n_obs
        gc_scores.append(gc)

    graph_conn = np.mean(gc_scores)

    batch_df = pd.DataFrame({
        "Cell_ASW":[cell_asw],
        "Batch_ASW":[batch_asw],
        "Graph_Connectivity":[graph_conn]
    })

    ############################################
    # summary metrics（论文最常用）
    ############################################

    summary_df = pd.DataFrame({
        "ARI":[best_metrics[0]],
        "AMI":[best_metrics[1]],
        "NMI":[best_metrics[2]],
        "HOM":[best_metrics[3]],
        "Cell_ASW":[cell_asw],
        "Batch_ASW":[batch_asw],
        "Graph_Connectivity":[graph_conn]
    })

    ############################################
    # save metrics
    ############################################

    cluster_df.to_csv(os.path.join(outdir,"cluster_metrics.csv"),index=False)
    batch_df.to_csv(os.path.join(outdir,"batch_metrics.csv"),index=False)
    summary_df.to_csv(os.path.join(outdir,"summary_metrics.csv"),index=False)

    ############################################
    # plots
    ############################################

    print("Drawing plots...")

    with PdfPages(os.path.join(outdir,"embedding_plots.pdf")) as pdf:
        fig = sc.pl.umap(
            adata,
            color=celltype_key,
            show=False,
            return_fig=True,
            size=10,
            legend_loc="right margin"
        )
        fig.set_size_inches(10,8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig = sc.pl.umap(
            adata,
            color=batch_key,
            show=False,
            return_fig=True,
            size=10,
            legend_loc="right margin"
        )
        fig.set_size_inches(10,8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig = sc.pl.tsne(
            adata,
            color=celltype_key,
            show=False,
            return_fig=True,
            size=10,
            legend_loc="right margin"
        )
        fig.set_size_inches(10,8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig = sc.pl.tsne(
            adata,
            color=batch_key,
            show=False,
            return_fig=True,
            size=10,
            legend_loc="right margin"
        )
        fig.set_size_inches(10,8)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

    print("Evaluation finished.")