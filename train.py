from scLinformer.scLinformerModel import Model

# parameters
rna_data_path = '/home/data/pbmc/processed_rna.h5ad'
output_path = '/home/output/test'

model = Model(
    RNAData = rna_data_path,
    output_path = output_path,
    process_data=True
)

model.train_model(epochs=1)
model.test_model()