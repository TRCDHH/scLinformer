import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import sparse
import json
from .model import *
from .utils import *


class Model:
    def __init__(self,        
        RNAData, 
        train_ids = None,
        valid_ids = None,
        test_ids = None,      
        output_path: str = './scLinformer_output/',
        gene_vocab_path = None,
        n_genes: int = 2000,
        process_data: bool = True,
        save_processed_data: bool = False,
        use_universal_model: bool = False,
        use_cell_type: bool = True,
        use_batch: bool = True,
        use_hvg: bool = True
    ):
        super(Model, self).__init__()

        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, 'processed_data'), exist_ok=True)

        self.use_universal_model = use_universal_model
        if use_universal_model:
            if gene_vocab_path is not None:
                with open(gene_vocab_path, "r") as f:
                    gene_vocab = json.load(f)
            else:
                base_dir = os.path.dirname(__file__)
                json_path = os.path.join(base_dir, "default_gene_vocab.json")
                with open(json_path, "r") as f:
                    gene_vocab = json.load(f)
        else:
            gene_vocab = None
        # process data
        if isinstance(RNAData, str):
            RNAData = sc.read_h5ad(RNAData)
        if process_data:
            print('processing data ...')
            RNAData = RNA_data_preprocessing(RNAData, use_hvg=use_hvg,use_batch=use_batch,use_cell_type=use_cell_type,n_top_genes=n_genes, use_universal_model=use_universal_model, save_data=save_processed_data, file_path=os.path.join(self.output_path, 'processed_data'), gene_vocab=gene_vocab)
        print(RNAData)
        if use_universal_model:
            self.rna_encoder = RNAEncoder(n_genes=n_genes,use_gene_embed = True, num_gene_embed=len(gene_vocab))
        else:
            self.rna_encoder = RNAEncoder(n_genes=n_genes)
        if use_batch:
            self.rna_decoder = RNADecoder(output_dim=n_genes, n_batch=len(RNAData.obs['batch_idx'].unique()))
        else:
            self.rna_decoder = RNADecoder(output_dim=n_genes)

        if use_cell_type:
            self.cell_type_discriminator = SimpleClassifier(num_classes=len(RNAData.obs['cell_type_idx'].unique()))
        else:
            self.cell_type_discriminator = SimpleClassifier(num_classes=1)
    
        if torch.cuda.is_available():
            self.rna_encoder = self.rna_encoder.cuda()
            self.rna_decoder = self.rna_decoder.cuda()
            self.cell_type_discriminator = self.cell_type_discriminator.cuda()
        
        self.rna_data_obs = RNAData.obs
        self.rna_data_var = RNAData.var
        self.rna_data  = RNAData.X.toarray()  if sparse.issparse(RNAData.X)  else np.asarray(RNAData.X)
        self.rna_reconstruct_loss = nn.MSELoss(reduction='mean')
        self.cell_type_classification_loss = nn.CrossEntropyLoss()
        # split dataset
        if train_ids is None or valid_ids is None or test_ids is None:
            id_list = five_fold_split_dataset(RNAData)
            self.train_idxs, self.valid_idxs, self.test_idxs = id_list[0]
            np.save(os.path.join(self.output_path, 'processed_data', 'train_ids.npy'), np.array(self.train_idxs))
            np.save(os.path.join(self.output_path, 'processed_data', 'valid_ids.npy'), np.array(self.valid_idxs))
            np.save(os.path.join(self.output_path, 'processed_data', 'test_ids.npy'), np.array(self.test_idxs))
        else:
            self.train_idxs = train_ids
            self.valid_idxs = valid_ids 
            self.test_idxs = test_ids
        print(f"train dataset: {len(self.train_idxs)}\nvalid dataset: {len(self.valid_idxs)}\ntest dataset: {len(self.test_idxs)}")
        print('model initialized ... \n')

    def set_train(self):
        self.rna_encoder.train()
        self.rna_decoder.train()
        self.cell_type_discriminator.train()
     
    
    def set_eval(self):
        self.rna_encoder.eval()
        self.rna_decoder.eval()
        self.cell_type_discriminator.eval()

    def R2R(self, x):
        rna = x['data']
        gene_idx = x['gene_idx'] if 'gene_idx' in x else None
        cell_type_idx = x['cell_type_idx'] if 'cell_type_idx' in x else None
        batch_idx = x['batch_idx'] if 'batch_idx' in x else None
        mu = self.rna_encoder(rna, gene_idx)
        r2r = self.rna_decoder(mu, batch_idx)
        # classifier
        if cell_type_idx is not None:
            cell_type_logits = self.cell_type_discriminator(mu)
            cell_loss = self.cell_type_classification_loss(cell_type_logits, cell_type_idx)
        else:
            cell_loss = None
        # reconstruct
        reconstruct_loss = self.rna_reconstruct_loss(r2r, rna)
        # loss
        loss = reconstruct_loss
        if cell_loss is not None:
            loss += cell_loss * 1e-3
        return loss, reconstruct_loss, cell_loss
    
    def train_model(
            self,
            model_path = None,
            rna_encoder_lr = 0.0001,
            rna_decoder_lr = 0.001,
            cell_type_discriminator_lr = 0.001,
            batch_size = 64,
            patience = 10,
            epochs = 100,
            seed: int = 114514):
        
        if not seed is None:
            setSeed(seed)

        # load model if needed
        if model_path is not None:
            self.rna_encoder.load_state_dict(torch.load(os.path.join(model_path, 'rna_encoder.pth')))
            self.rna_decoder.load_state_dict(torch.load(os.path.join(model_path, 'rna_decoder.pth')))
            self.cell_type_discriminator.load_state_dict(torch.load(os.path.join(model_path, 'cell_type_discriminator.pth')))
        # dataloader
        train_dataset = Dataset(self.rna_data, self.train_idxs, 
                                gene_idx=self.rna_data_var['gene_idx'] if 'gene_idx' in self.rna_data_var.columns else None, 
                                cell_type_idx=self.rna_data_obs['cell_type_idx'] if 'cell_type_idx' in self.rna_data_obs.columns else None,
                                batch_idx=self.rna_data_obs['batch_idx'] if 'batch_idx' in self.rna_data_obs.columns else None)
        valid_dataset = Dataset(self.rna_data, self.valid_idxs, gene_idx=self.rna_data_var['gene_idx'] if 'gene_idx' in self.rna_data_var.columns else None, 
                                cell_type_idx=self.rna_data_obs['cell_type_idx'] if 'cell_type_idx' in self.rna_data_obs.columns else None,
                                batch_idx=self.rna_data_obs['batch_idx'] if 'batch_idx' in self.rna_data_obs.columns else None)
        if(len(train_dataset) % batch_size == 1):
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        else:
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
        if(len(valid_dataset) % batch_size == 1):
            validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
        else:
            validation_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        # optimizers
        rna_encoder_optimizer = torch.optim.AdamW(self.rna_encoder.parameters(), lr=rna_encoder_lr)
        rna_decoder_optimizer = torch.optim.AdamW(self.rna_decoder.parameters(), lr=rna_decoder_lr)
        rna_discriminator_cell_type_optimizer = torch.optim.AdamW(self.cell_type_discriminator.parameters(), lr=cell_type_discriminator_lr, weight_decay=1e-4)
        # eraly stop
        early_stop = EarlyStopping(patience=patience)
        # training
        with tqdm(total = epochs, dynamic_ncols=True) as pbar:
            pbar.set_description('train rna ...')
            for epoch in range(epochs):
                train_loss, val_loss = [], []
                train_test_loss, val_test_loss = [], []
                self.set_train()
                for idx, data in enumerate(train_dataloader): 
                    if torch.cuda.is_available():
                        data['data'] = data['data'].cuda()
                        if "gene_idx" in data:
                            data['gene_idx'] = data['gene_idx'].cuda()
                        if "cell_type_idx" in  data:
                            data['cell_type_idx'] = data['cell_type_idx'].cuda()
                        if "batch_idx" in data:
                            data['batch_idx'] = data['batch_idx'].cuda()

                    # train encoder, decoder, translator
                    loss, reconstruct_loss, cell_loss = self.R2R(data)
                    loss.backward()
                    rna_encoder_optimizer.step()
                    rna_decoder_optimizer.step()
                    rna_discriminator_cell_type_optimizer.step()
                    rna_encoder_optimizer.zero_grad()
                    rna_decoder_optimizer.zero_grad()
                    rna_discriminator_cell_type_optimizer.zero_grad()

                    train_loss.append(reconstruct_loss.item())
                    if cell_loss is not None:
                        train_test_loss.append(cell_loss.item())

                # validation
                self.set_eval()
                for idx, data in enumerate(validation_dataloader):
                    if torch.cuda.is_available():
                        data['data'] = data['data'].cuda()
                        if "gene_idx" in data:
                            data['gene_idx'] = data['gene_idx'].cuda()
                        if "cell_type_idx" in  data:
                            data['cell_type_idx'] = data['cell_type_idx'].cuda()
                        if "batch_idx" in data:
                            data['batch_idx'] = data['batch_idx'].cuda()
                    loss, val_reconstruct_loss, cell_loss_val = self.R2R(data)
                    val_loss.append(val_reconstruct_loss.item())
                    if cell_loss is not None:
                        val_test_loss.append(cell_loss.item())

                early_stop(np.mean(val_loss), self)
                time.sleep(0.01)
                pbar.update(1)
                pbar.set_postfix(
                    train='{:.4f}'.format(np.mean(train_loss)), 
                    val='{:.4f}'.format(np.mean(val_loss)),
                    cell_train='{:.4f}'.format(np.mean(train_test_loss)), 
                    cell_val='{:.4f}'.format(np.mean(val_test_loss))
                )
                
                if early_stop.early_stop:
                    print("Early stoped ... \n")
                    break
    
    def test_model(
        self,
        model_path = None,
        batch_size: int = 64,
        use_test: bool = True
    ):
        # load_model
        if model_path is not None:
            self.rna_encoder.load_state_dict(torch.load(os.path.join(model_path, 'rna_encoder.pth')))
        else:
            self.rna_encoder.load_state_dict(torch.load(os.path.join(self.output_path, 'model', 'rna_encoder.pth')))
        # dataloader
        test_dataset = Dataset(self.rna_data, self.test_idxs, 
                               gene_idx=self.rna_data_var['gene_idx'] if 'gene_idx' in self.rna_data_var.columns else None,
                               batch_idx=self.rna_data_obs['batch_idx'] if 'batch_idx' in self.rna_data_obs.columns else None)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        
        self.set_eval()
        """ record the predicted data """
        MU_predict = []

        with torch.no_grad():
            with tqdm(total = len(test_dataloader), dynamic_ncols=True) as pbar:
                pbar.set_description('RNA predicting...')
                for idx, data in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        data['data'] = data['data'].cuda()
                        if "gene_idx" in data:
                            data['gene_idx'] = data['gene_idx'].cuda()
                    mu = self.rna_encoder(data['data'], data['gene_idx'] if "gene_idx" in data else None)
                    MU_predict.append(mu.cpu())
                    time.sleep(0.01)
                    pbar.update(1)

        adata = tensor2adata([], MU_predict)
        adata.obs = self.rna_data_obs.iloc[self.test_idxs, :].copy()

        # print('test finished ... \n')
        if use_test:    
            evaluate_sc_embedding(
                adata,
                embedding_key="X_emb",
                celltype_key="cell_type",
                batch_key="batch",
                outdir=self.output_path
            )
        return adata

    def save_model_dict(self):
        torch.save(self.rna_encoder.state_dict(), os.path.join(self.output_path, 'model', 'rna_encoder.pth'))
        torch.save(self.rna_decoder.state_dict(), os.path.join(self.output_path, 'model', 'rna_decoder.pth'))
        torch.save(self.cell_type_discriminator.state_dict(), os.path.join(self.output_path, 'model', 'cell_type_discriminator.pth'))