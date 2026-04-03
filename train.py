from scLinformer.scLinformerModel import Model
import scanpy as sc
import os
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import connected_components
from matplotlib.backends.backend_pdf import PdfPages

# parameters
rna_data_path = '/home/output/newcovid/processed_data/processed_rna.h5ad'
output_path = '/home/output/newcovid'

model = Model(
    RNAData = rna_data_path,
    output_path = output_path,
    train_ids=np.load("/home/data/newcovid/train_ids.npy").tolist(),
    test_ids=np.load("/home/data/newcovid/test_ids.npy").tolist(),
    valid_ids=np.load("/home/data/newcovid/valid_ids.npy").tolist(),
    process_data=False,
    save_processed_data=True,
    use_batch=False
)

model.train_model()
model.test_model()
# adata = model.test_model(use_test=False)
# adata.write_h5ad(os.path.join(output_path, "adata_with_embedding.h5ad"))

# def evaluate_sc_embedding(
#         adata,
#         embedding_key="X_emb",
#         celltype_key="cell_type",
#         batch_key="batch",
#         outdir="./evaluation",
#         n_neighbors=15,
#         resolutions=[0.3, 0.5, 0.8, 1.0]
#     ):

#     os.makedirs(outdir, exist_ok=True)

#     # 设置图大小（解决 PDF 太窄）
#     sc.set_figure_params(dpi=120, figsize=(8,8))

#     print("Computing neighbors graph...")
#     sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)

#     print("Computing UMAP...")
#     sc.tl.umap(adata)

#     ############################################
#     # clustering metrics
#     ############################################

#     results = []

#     print("Evaluating clustering...")

#     best_ari = -1
#     best_metrics = None

#     for r in resolutions:

#         key = f"leiden_{r}"
#         sc.tl.leiden(adata, resolution=r, key_added=key)

#         ari = metrics.adjusted_rand_score(
#             adata.obs[celltype_key],
#             adata.obs[key]
#         )

#         ami = metrics.adjusted_mutual_info_score(
#             adata.obs[celltype_key],
#             adata.obs[key]
#         )

#         nmi = metrics.normalized_mutual_info_score(
#             adata.obs[celltype_key],
#             adata.obs[key]
#         )

#         hom = metrics.homogeneity_score(
#             adata.obs[celltype_key],
#             adata.obs[key]
#         )

#         results.append({
#             "resolution": r,
#             "ARI": ari,
#             "AMI": ami,
#             "NMI": nmi,
#             "HOM": hom
#         })

#         if ari > best_ari:
#             best_ari = ari
#             best_metrics = (ari, ami, nmi, hom)

#     cluster_df = pd.DataFrame(results)
    
#     ############################################
#     # summary metrics（论文最常用）
#     ############################################

#     summary_df = pd.DataFrame({
#         "ARI":[best_metrics[0]],
#         "AMI":[best_metrics[1]],
#         "NMI":[best_metrics[2]],
#         "HOM":[best_metrics[3]],
#     })

#     ############################################
#     # save metrics
#     ############################################

#     cluster_df.to_csv(os.path.join(outdir,"cluster_metrics.csv"),index=False)
#     summary_df.to_csv(os.path.join(outdir,"summary_metrics.csv"),index=False)

#     ############################################
#     # plots
#     ############################################

#     print("Drawing plots...")

#     with PdfPages(os.path.join(outdir,"embedding_plots.pdf")) as pdf:
#         fig = sc.pl.umap(
#             adata,
#             color=celltype_key,
#             show=False,
#             return_fig=True,
#             size=10,
#             legend_loc="right margin"
#         )
#         fig.set_size_inches(10,8)
#         pdf.savefig(fig, bbox_inches="tight")
#         plt.close()

#         fig = sc.pl.umap(
#             adata,
#             color="time_point",
#             show=False,
#             return_fig=True,
#             size=10,
#             legend_loc="right margin"
#         )
#         fig.set_size_inches(10,8)
#         pdf.savefig(fig, bbox_inches="tight")
#         plt.close()

#         fig = sc.pl.umap(
#             adata,
#             color="covid_status",
#             show=False,
#             return_fig=True,
#             size=10,
#             legend_loc="right margin"
#         )
#         fig.set_size_inches(10,8)
#         pdf.savefig(fig, bbox_inches="tight")
#         plt.close()

#     print("Evaluation finished.")

# evaluate_sc_embedding(
#     adata,
#     embedding_key="X_emb",
#     celltype_key="cell_type",
#     batch_key="batch",
#     outdir=output_path
# )