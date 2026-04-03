import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import sparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
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
        lantent_dim: int = 1024,
        rna_encoder_hid_dim: int = 128,
        rna_decoder_hid_dim: int = 256,
        process_data: bool = True,
        save_processed_data: bool = False,
        use_universal_model: bool = False,
        use_cell_type: bool = True,
        use_batch: bool = True
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
            RNAData = RNA_data_preprocessing(RNAData,use_batch=use_batch,use_cell_type=use_cell_type,n_top_genes=n_genes, use_universal_model=use_universal_model, save_data=save_processed_data, file_path=os.path.join(self.output_path, 'processed_data'), gene_vocab=gene_vocab)
        print(RNAData)
        if use_universal_model:
            self.rna_encoder = RNAEncoder(n_genes=n_genes,use_gene_embed = True, num_gene_embed=len(gene_vocab))
        else:
            self.rna_encoder = RNAEncoder(n_genes=n_genes)
        if use_batch:
            self.rna_decoder = RNADecoder(output_dim=n_genes, n_batch=len(RNAData.obs['batch_idx'].unique()))
        else:
            self.rna_decoder = RNADecoder(output_dim=n_genes)
        self.rna_discriminator = Discriminator()
        #self.cell_type_discriminator = nn.Linear(32, len(RNAData.obs['cell_type_idx'].unique()))
        if use_cell_type:
            self.cell_type_discriminator = SimpleClassifier(num_classes=len(RNAData.obs['cell_type_idx'].unique()))
        else:
            self.cell_type_discriminator = SimpleClassifier(num_classes=1)

        if use_universal_model:
            self.fix_rna_encoder = RNAEncoder(use_gene_embed = True, num_gene_embed=len(gene_vocab))
        else:
            self.fix_rna_encoder = RNAEncoder()
    
        if torch.cuda.is_available():
            self.rna_encoder = self.rna_encoder.cuda()
            self.rna_decoder = self.rna_decoder.cuda()
            self.rna_discriminator = self.rna_discriminator.cuda()
            self.fix_rna_encoder = self.fix_rna_encoder.cuda()
            self.cell_type_discriminator = self.cell_type_discriminator.cuda()
        
        self.rna_data_obs = RNAData.obs
        self.rna_data_var = RNAData.var
        self.rna_data  = RNAData.X.toarray()  if sparse.issparse(RNAData.X)  else np.asarray(RNAData.X)
        self.contrastive_loss = DCL(temperature=0.1)
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
        self.rna_discriminator.train()
        self.cell_type_discriminator.train()
     
    
    def set_eval(self):
        self.rna_encoder.eval()
        self.rna_decoder.eval()
        self.rna_discriminator.eval()
        self.cell_type_discriminator.eval()

    def R2R(self, x):
        rna = x['data']
        gene_idx = x['gene_idx'] if 'gene_idx' in x else None
        cell_type_idx = x['cell_type_idx'] if 'cell_type_idx' in x else None
        batch_idx = x['batch_idx'] if 'batch_idx' in x else None
        mu = self.rna_encoder(rna, gene_idx)
        r2r = self.rna_decoder(mu, batch_idx)
        # contrastive
        # self.update_fixed_rna_encoder()
        # real_mu = self.fix_rna_encoder(rna, gene_idx)
        # rec_mu = self.fix_rna_encoder(r2r, gene_idx)
        # contrastive_loss = self.contrastive_loss(real_mu, rec_mu) + self.contrastive_loss(rec_mu, real_mu)
        # gan
        # rec_disc_out= self.rna_discriminator(r2r)
        # gan_loss = - torch.mean(rec_disc_out)
        # classifier
        if cell_type_idx is not None:
            cell_type_logits = self.cell_type_discriminator(mu)
            cell_loss = self.cell_type_classification_loss(cell_type_logits, cell_type_idx)
        else:
            cell_loss = None
        # reconstruct
        reconstruct_loss = self.rna_reconstruct_loss(r2r, rna)
        # loss
        loss = reconstruct_loss #+ 1e-3 * contrastive_loss + gan_loss * 1e-3
        if cell_loss is not None:
            loss += cell_loss * 1e-3
        return loss, reconstruct_loss, 0, 0, cell_loss
    
    def train_model(
            self,
            model_path = None,
            rna_encoder_lr = 0.0001,
            rna_decoder_lr = 0.001,
            rna_discriminator_lr = 0.001,
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
            self.rna_discriminator.load_state_dict(torch.load(os.path.join(model_path, 'rna_discriminator.pth')))
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
        rna_discriminator_optimizer = torch.optim.AdamW(self.rna_discriminator.parameters(), lr=rna_discriminator_lr)
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
                    
                    # train discriminator
                    # r2r = self.rna_decoder(self.rna_encoder(data['data'], data['gene_idx'] if 'gene_idx' in data else None))
                    # real_disc_out_r = self.rna_discriminator(data['data'])
                    # rec_disc_out_r = self.rna_discriminator(r2r.detach())
                    # disc_loss_r = torch.mean(rec_disc_out_r) - torch.mean(real_disc_out_r)
                    # gradient_penalty_r = self.compute_gradient_penalty(self.rna_discriminator, data['data'], r2r.detach(), device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    # loss_d_r = disc_loss_r + gradient_penalty_r
                    # rna_discriminator_optimizer.zero_grad()
                    # loss_d_r.backward()
                    # rna_discriminator_optimizer.step()

                    # train encoder, decoder, translator
                    loss, reconstruct_loss, contrastive_loss, gan_loss, cell_loss = self.R2R(data)
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
                    loss, val_reconstruct_loss, val_contrastive_loss, val_gan_loss, cell_loss_val = self.R2R(data)
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
            self.rna_decoder.load_state_dict(torch.load(os.path.join(model_path, 'rna_decoder.pth')))
        else:
            self.rna_encoder.load_state_dict(torch.load(os.path.join(self.output_path, 'model', 'rna_encoder.pth')))
            self.rna_decoder.load_state_dict(torch.load(os.path.join(self.output_path, 'model', 'rna_decoder.pth')))
        # dataloader
        test_dataset = Dataset(self.rna_data, self.test_idxs, 
                               gene_idx=self.rna_data_var['gene_idx'] if 'gene_idx' in self.rna_data_var.columns else None,
                               batch_idx=self.rna_data_obs['batch_idx'] if 'batch_idx' in self.rna_data_obs.columns else None)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
        
        self.set_eval()
        """ record the predicted data """
        MU_predict = []
        R2R_predict = []

        with torch.no_grad():
            with tqdm(total = len(test_dataloader), dynamic_ncols=True) as pbar:
                pbar.set_description('RNA predicting...')
                for idx, data in enumerate(test_dataloader):
                    if torch.cuda.is_available():
                        data['data'] = data['data'].cuda()
                        if "gene_idx" in data:
                            data['gene_idx'] = data['gene_idx'].cuda()
                        if "batch_idx" in data:
                            data['batch_idx'] = data['batch_idx'].cuda()
                    mu = self.rna_encoder(data['data'], data['gene_idx'] if "gene_idx" in data else None)
                    if "batch_idx" in data:
                        r2r = self.rna_decoder(mu, data['batch_idx'])
                    else:
                        r2r = self.rna_decoder(mu)
                    MU_predict.append(mu.cpu())
                    R2R_predict.append(r2r.cpu())
                    time.sleep(0.01)
                    pbar.update(1)

        adata = tensor2adata(R2R_predict, MU_predict)
        adata.obs = self.rna_data_obs.iloc[self.test_idxs, :].copy()

        # print('drawing tsne figures ...')
        # """ draw umap if needed """
        # fig_MU = draw_tsne(adata, 'mu', 'cell_type')
        # fig_batch = draw_tsne(adata, 'mu', 'batch')
        
        # fig_list = [fig_MU, fig_batch]
        # with PdfPages(self.output_path + '/tSNE.pdf') as pdf:
        #     for i in range(len(fig_list)):
        #         pdf.savefig(figure=fig_list[i], dpi=200, bbox_inches='tight')
        #         plt.close()
        
        # index_MU = calculate_cluster_index(adata, 'cell_type')
        # index_matrix = pd.DataFrame([index_MU])
        # index_matrix.columns = ['ARI', 'AMI', 'NMI', 'HOM']
        # index_matrix.index = ['cell type']
        # index_matrix.to_csv(self.output_path + '/cluster_index.csv')

        # batch_metrics = compute_batch_metrics(adata, batch_key='batch')
        # batch_df = pd.DataFrame([batch_metrics])
        # batch_df.index = ['batch']
        # batch_df.to_csv(self.output_path + '/batch_index.csv')

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
        torch.save(self.rna_discriminator.state_dict(), os.path.join(self.output_path, 'model', 'rna_discriminator.pth'))
        torch.save(self.cell_type_discriminator.state_dict(), os.path.join(self.output_path, 'model', 'cell_type_discriminator.pth'))
        

    @torch.no_grad()
    def update_fixed_rna_encoder(self):
        for param, param_fixed in zip(self.rna_encoder.parameters(),
                                    self.fix_rna_encoder.parameters()):
            param_fixed.data.copy_(param.data)
        
    @torch.no_grad()
    def update_fixed_encoder(self):
        for param, param_fixed in zip(self.rna_encoder.parameters(),
                                    self.fix_rna_encoder.parameters()):
            param_fixed.data.copy_(param.data)

        for param, param_fixed in zip(self.atac_encoder.parameters(),
                                    self.fix_atac_encoder.parameters()):
            param_fixed.data.copy_(param.data)
        
        for param, param_fixed in zip(self.translator.parameters(),
                                    self.fix_translator.parameters()):
            param_fixed.data.copy_(param.data)

    def compute_gradient_penalty(self, discriminator, real_samples, fake_samples, device):
        alpha = torch.rand(real_samples.size(0), 1, device=device)
        alpha = alpha.expand_as(real_samples)
        
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated = interpolated.to(device)
        interpolated.requires_grad_(True)
        
        d_interpolated = discriminator(interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        lambda_gp = 10
        return lambda_gp * gradient_penalty


SMALL_NUM = np.log(1e-45)
class DCL(object):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.5, weight_fn=None):
        super(DCL, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn

    def __call__(self, z1, z2):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cross_view_distance = torch.mm(z1, z2.t())
        positive_loss = -torch.diag(cross_view_distance) / self.temperature
        if self.weight_fn is not None:
            positive_loss = positive_loss * self.weight_fn(z1, z2)
        neg_similarity = torch.cat((torch.mm(z1, z1.t()), cross_view_distance), dim=1) / self.temperature
        neg_mask = torch.eye(z1.size(0), device=z1.device).repeat(1, 2)
        negative_loss = torch.logsumexp(neg_similarity + neg_mask * SMALL_NUM, dim=1, keepdim=False)
        return (positive_loss + negative_loss).mean()