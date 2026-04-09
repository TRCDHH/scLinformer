from scLinformer.scLinformerModel import Model
import scanpy as sc

# parameters
rna_data_path = '/home/data/control.h5ad'
output_path = '/home/output'
rna = sc.read_h5ad(rna_data_path)

model = Model(
    RNAData = rna,
    output_path = output_path,
    save_processed_data = True
)
model.train_model()
