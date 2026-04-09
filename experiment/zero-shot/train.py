from scLinformer.scLinformerModel import Model
import numpy as np

# parameters
rna_data_path = '/home/data/newcovid/challenge_pbmc_cellxgene_230223_aligned.h5ad'
output_path = '/home/output/pre-train/newcovid'

model = Model(
    RNAData = rna_data_path,
    output_path = output_path,
    train_ids=np.load("/home/data/newcovid/train_ids.npy").tolist(),
    test_ids=np.load("/home/data/newcovid/test_ids.npy").tolist(),
    valid_ids=np.load("/home/data/newcovid/valid_ids.npy").tolist(),
    process_data=True,
    save_processed_data=True,
    use_batch=False,
    use_cell_type=True,
    use_universal_model=True,
    use_hvg=False
)

# model.train_model()
model.test_model(model_path="/home/output/pre-train/pbmc/model")