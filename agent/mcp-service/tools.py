from scLinformer.scLinformerModel import Model
import numpy as np
import pymysql
import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import silhouette_samples
from scipy.sparse.csgraph import connected_components
import json
import traceback

base_url = "http://localhost:8001/static"

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
    sc.set_figure_params(dpi=120, figsize=(8, 8))

    # Compute only UMAP
    print("Computing neighbors graph...")
    sc.pp.neighbors(adata, use_rep=embedding_key, n_neighbors=n_neighbors)

    print("Computing UMAP...")
    sc.tl.umap(adata)

    # Store results
    cluster_results = []
    best_ari = -1
    best_metrics = None
    has_cell_type = celltype_key in adata.obs.columns
    has_batch = batch_key in adata.obs.columns

    # Cell type clustering metrics (only calculated if cell_type exists)
    if has_cell_type:
        print("Evaluating cell type clustering...")
        for r in resolutions:
            key = f"leiden_{r}"
            sc.tl.leiden(adata, resolution=r, key_added=key)

            ari = metrics.adjusted_rand_score(adata.obs[celltype_key], adata.obs[key])
            ami = metrics.adjusted_mutual_info_score(adata.obs[celltype_key], adata.obs[key])
            nmi = metrics.normalized_mutual_info_score(adata.obs[celltype_key], adata.obs[key])
            hom = metrics.homogeneity_score(adata.obs[celltype_key], adata.obs[key])

            cluster_results.append({
                "resolution": r, "ARI": ari, "AMI": ami, "NMI": nmi, "HOM": hom
            })

            if ari > best_ari:
                best_ari = ari
                best_metrics = (ari, ami, nmi, hom)

        cluster_df = pd.DataFrame(cluster_results)
        cluster_df.to_csv(os.path.join(outdir, "cluster_metrics.csv"), index=False)
    else:
        cluster_df = pd.DataFrame()
        print("No cell_type detected, skipping cell type clustering evaluation")

    # Batch/cell type ASW + graph connectivity
    X = adata.obsm[embedding_key]
    cell_asw = np.nan
    batch_asw = np.nan
    graph_conn = np.nan

    # Cell type silhouette score
    if has_cell_type:
        cell_labels = adata.obs[celltype_key].astype(str).values
        sil_cell = silhouette_samples(X, cell_labels, metric="euclidean")
        cell_asw = (np.mean(sil_cell) + 1) / 2

    # Batch silhouette score + graph connectivity
    if has_batch:
        batch_labels = adata.obs[batch_key].astype(str).values
        batch_asw_samples = silhouette_samples(X, batch_labels, metric="euclidean")
        batch_asw = np.mean(batch_asw_samples)

        gc_scores = []
        for b in adata.obs[batch_key].unique():
            sub = adata[adata.obs[batch_key] == b].copy()
            sc.pp.neighbors(sub, use_rep=embedding_key, n_neighbors=n_neighbors)
            n_comp, labels = connected_components(csgraph=sub.obsp["connectivities"], directed=False)
            gc = np.max(np.bincount(labels)) / sub.n_obs
            gc_scores.append(gc)
        graph_conn = np.mean(gc_scores)

    # Save batch metrics
    batch_df = pd.DataFrame({
        "Cell_ASW": [cell_asw],
        "Batch_ASW": [batch_asw],
        "Graph_Connectivity": [graph_conn]
    })
    batch_df.to_csv(os.path.join(outdir, "batch_metrics.csv"), index=False)

    # Summary metrics
    summary_data = {
        "ARI": [best_metrics[0] if best_metrics else np.nan],
        "AMI": [best_metrics[1] if best_metrics else np.nan],
        "NMI": [best_metrics[2] if best_metrics else np.nan],
        "HOM": [best_metrics[3] if best_metrics else np.nan],
        "Cell_ASW": [cell_asw],
        "Batch_ASW": [batch_asw],
        "Graph_Connectivity": [graph_conn]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(outdir, "summary_metrics.csv"), index=False)

    # Generate UMAP plots only
    print("Drawing UMAP plots (PNG)...")

    # Cell type UMAP
    if has_cell_type:
        fig = sc.pl.umap(
            adata, color=celltype_key, show=False, return_fig=True, size=10, legend_loc="right margin"
        )
        fig.set_size_inches(10, 8)
        fig.savefig(os.path.join(outdir, "umap_cell_type.png"), bbox_inches="tight", dpi=150)
        plt.close()
        relative_path = os.path.join(outdir, "umap_cell_type.png").replace('\\', '/').split('runtime_data/')[-1]
        summary_data['umap_cell_path'] = f"{base_url}/{relative_path}"

    # Batch UMAP
    if has_batch:
        fig = sc.pl.umap(
            adata, color=batch_key, show=False, return_fig=True, size=10, legend_loc="right margin"
        )
        fig.set_size_inches(10, 8)
        fig.savefig(os.path.join(outdir, "umap_batch.png"), bbox_inches="tight", dpi=150)
        plt.close()
        relative_path = os.path.join(outdir, "umap_batch.png").replace('\\', '/').split('runtime_data/')[-1]
        summary_data['umap_batch_path'] = f"{base_url}/{relative_path}"

    print("Evaluation finished!")
    return summary_data

# Database connection configuration
db_config = {
    'host': '47.102.40.253',
    'user': 'root',
    'password': 'crm114514',
    'database': 'agent_db',
    'port': 3306
}

def get_db_connection():
    """Get database connection"""
    return pymysql.connect(**db_config)


def run_task_process(task_id, dataset_id, task_folder):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
            select dataset_path from dataset where id = %s
            """
            cursor.execute(sql, (dataset_id))
            dataset_path = cursor.fetchone()[0]

            adata = sc.read_h5ad(dataset_path)
            use_cell_type = 'cell_type' in adata.obs.columns
            use_batch = 'batch' in adata.obs.columns
            model = Model(
                RNAData=adata,
                process_data=True,
                save_processed_data=True,
                use_cell_type=use_cell_type,
                use_batch=use_batch
            )
            processed_data_path = os.path.join(task_folder, 'processed_data')

            sql = """
            update dataset set processed_path = %s where id = %s
            """
            cursor.execute(sql, (processed_data_path, dataset_id))
            conn.commit()

            model.train_model()
            adata = model.test_model(use_test=False)
            result = evaluate_sc_embedding(
                adata,
                embedding_key="X_emb",
                batch_key="batch",
                celltype_key="cell_type",
                outdir=task_folder,
            )
            print(result)
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.integer, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(i) for i in obj]
                else:
                    return obj
            # Convert entire result
            result = convert_to_json_serializable(result)
            # Now can safely convert to JSON
            result_json = json.dumps(result, ensure_ascii=False)

            sql = """
            update task set complete_time = now(), result = %s where id = %s
            """
            cursor.execute(sql, (result_json, task_id))
            conn.commit()
    except Exception as e:
        # Get full error stack trace
        error_msg = traceback.format_exc()
        short_error = str(e)
        print(f"Task execution failed:\n{error_msg}")
        # Store error in database
        try:
            fail_result = json.dumps({
                "status": "failed",
                "error": short_error,
                "detail": error_msg
            }, ensure_ascii=False)

            with conn.cursor() as cursor:
                sql = """
                update task 
                set complete_time = now(), 
                    result = %s 
                where id = %s
                """
                cursor.execute(sql, (fail_result, task_id))
                conn.commit()
        except Exception as db_e:
            print("Database write failed:", str(db_e))
    finally:
        conn.close()
