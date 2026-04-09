#!/bin/bash

# ====================== Pipeline Runner ======================
# This script runs the full analysis pipeline step-by-step.
# Please make sure all parameters in each script are properly set BEFORE running.

set -e  # stop if any command fails

echo "=================================================="
echo "Starting Analysis Pipeline"
echo "=================================================="

# ====================== Step 1 ======================
echo ""
echo "Step 1: Perturbation Importance Analysis"
echo "Please ensure parameters are correctly set in perturbation script"
echo "--------------------------------------------------"

read -p "Press Enter to continue..."

python /home/code/get_top_perturb_genes.py

echo "Step 1 finished"

# ====================== Step 2 ======================
echo ""
echo "Step 2: GO Enrichment Analysis"
echo "Please check gene selection strategy (top_n / threshold)"
echo "--------------------------------------------------"

read -p "Press Enter to continue..."

python /home/code/go_enrichment.py

echo "Step 2 finished"

# ====================== Step 3 ======================
echo ""
echo "Step 3: Attention Heatmap Visualization"
echo "Ensure perturb_gene and top_n are consistent"
echo "--------------------------------------------------"

read -p "Press Enter to continue..."

python /home/code/heatmap.py

echo "Step 3 finished"

# ====================== Done ======================

echo ""
echo "=================================================="
echo "Pipeline Completed Successfully!"
echo "Check results in output directory"
echo "=================================================="