from scLinformer.scLinformerModel import Model
from scLinformer.utils import RNA_data_preprocessing
import scanpy as sc

# parameters
rna_data_path = '/home/data/norman/control.h5ad'
# rna = sc.read_h5ad(rna_data_path)
# rna = RNA_data_preprocessing(
#     rna,
#     normalize_total = False,
#     log1p = False,
#     use_hvg = True,
#     n_top_genes = 2000,
#     save_data = True,
#     file_path = '/home/data/norman',    
#     use_cell_type = False,
#     use_batch = False,
# )

output_path = '/home/output/norman'
rna = sc.read_h5ad(rna_data_path)
if 'cell_type_idx' not in rna.obs.columns:
    rna.obs['cell_type_idx'] = (
        rna.obs['cell_type']
        .astype('category')
        .cat.codes
    )
# print(rna.obs['cell_type'].unique())
# print(rna.obs['cell_type_idx'].unique())


model = Model(
    RNAData = rna,
    output_path = output_path,
    n_genes=5045,
    process_data=False,
    use_cell_type=True,
    use_batch=False
)
model.train_model()
# model.test_model()