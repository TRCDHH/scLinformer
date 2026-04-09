import os
import pandas as pd
import gseapy as gp

# ====================== Configuration ======================
csv_path = "/home/output/adamson/HSPA5_delta.csv"  # Path to the input CSV file containing gene importance scores
outdir = "/home/output/adamson/HSPA5" # Directory to save enrichment results and plots
top_n = 20   # Number of top genes to select (used when use_threshold = False)
use_threshold = False   # If True, select genes based on importance threshold instead of top_n
importance_threshold = 0.25   # Threshold for filtering genes by importance (used when use_threshold = True)
gene_set = "GO_Biological_Process_2021"  # Gene set library used for enrichment analysis
organism = "human"  # Organism name for enrichment analysis
cutoff = 0.05  # Adjusted p-value cutoff for filtering significant terms
plot_title = "GO Enrichment" # Title used for visualization plots
top_term = 10  # Number of top enriched terms to display in dot plot

# ====================== Setup ======================

os.makedirs(outdir, exist_ok=True)

# ====================== Step 1: Load CSV ======================

df = pd.read_csv(csv_path)

# Sort genes by importance score (descending)
df = df.sort_values(by="importance", ascending=False)

# ====================== Step 2: Select Top Genes ======================

if use_threshold:
    # Select genes based on importance threshold
    top_genes = df[df['importance'] > importance_threshold]['target'].tolist()
else:
    # Select top N genes
    top_genes = df['target'].head(top_n).tolist()

# Optional: remove low-information genes (e.g., RP11 genes)
top_genes = [g for g in top_genes if not g.startswith("RP11")]

print(f"Number of selected genes: {len(top_genes)}")
print("Top 10 genes:", top_genes[:10])

# ====================== Step 3: GO Enrichment ======================

enr = gp.enrichr(
    gene_list=top_genes,
    gene_sets=gene_set,
    organism=organism,
    outdir=outdir,
    cutoff=cutoff
)

# Get enrichment results
res = enr.results

# Save results
res.to_csv(os.path.join(outdir, "GO_results.csv"), index=False)

print("\nTop GO terms:")
print(res[['Term', 'Adjusted P-value']].head(10))

# ====================== Step 4: Visualization ======================

# Bar plot
gp.barplot(
    res,
    title=plot_title,
    cutoff=cutoff,
    ofname=os.path.join(outdir, "barplot.png")
)

# Dot plot (recommended for publication)
gp.dotplot(
    res,
    title=plot_title,
    cutoff=cutoff,
    top_term=top_term,
    ofname=os.path.join(outdir, "dotplot.png")
)

print(f"\nResults saved to: {outdir}")