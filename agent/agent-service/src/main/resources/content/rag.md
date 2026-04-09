# 📚 scRNA-seq Technology Knowledge Base

## 💫 ScLinformer Technical Features

**Architecture Highlights**
- Linear attention mechanism, O(n) complexity instead of O(n²), no lag on long sequences
- Optimized for sparse single-cell data, more efficient zero-value handling
- Built-in batch-aware encoding, automatic technical noise correction

**Suitable Scenarios**
- Fast dimensionality reduction for large sample sizes (>10k cells)
- Multi-batch data integration
- Resource-constrained environments (works even with small GPU memory)

**Output Results**
- Low-dimensional embedding (latent representation)
- Batch-corrected expression matrix
- Cell clustering pre-annotations

---

## 📊 Other Model Knowledge Reference (Coming Soon~)

Understanding these helps understand ScLinformer's design philosophy, but they are not yet available:

| Model | Core Method | Application | Features |
|-----|---------|---------|------|
| scVI | Variational Autoencoder + Deep Generative Model | Batch Correction, Denoising, Dimensionality Reduction | Probabilistic modeling, uncertainty quantification |
| TotalVI | Joint Variational Inference | CITE-seq (RNA+Protein) | Multi-modal integration |
| scArches | Transfer Learning Architecture | Reference Mapping, Query to Reference | No retraining needed |
| CellBender | Deep Generative Model | Remove Ambient RNA Contamination | Physical mechanism modeling |

> 💡 ScLinformer absorbs the advantages of the above models while optimizing for efficiency bottlenecks~

---

## 🔧 Data Format Guide

**Input Support**
- `.h5ad` (Scanpy AnnData, most recommended)
- 10x Genomics standard output (auto-read)
- `.loom` (Cross-platform compatible)

**Preprocessing Suggestions**
- Basic filtering: min_genes=200, min_cells=3
- Select ~2000 highly variable genes
- Normalization + log1p transformation

---

## ❓ FAQ

**Q: Can ScLinformer results be compared with scVI?**
A: Different embedding spaces, cannot compare values directly. Use downstream tasks (clustering accuracy, batch entropy) for evaluation~

**Q: My data has only 500 cells, can I use it?**
A: Yes! ScLinformer is stable for small samples, but large samples can better leverage the speed advantage ⚡

**Q: What about CITE-seq analysis?**
A: Currently ScLinformer focuses on RNA, multi-modal version in development, stay tuned 🙌

**Q: How long does a task take?**
A: ~2-5 minutes for 10k cells, ~10-15 minutes for 100k cells, much faster than traditional methods!

---

## 🎯 Analysis Workflow Suggestion

1. **Upload Data** → Auto format check
2. **Create Task** → ScLinformer auto processing
3. **Get Results** → Downstream analysis (clustering, visualization, marker identification)

Ask me anytime for explanations of specific steps~ ✨
