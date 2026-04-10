import torch
import torch.nn as nn
import torch.nn.functional as F

class RNAEncoder(nn.Module):
    def __init__(self, 
                 n_genes=2000, 
                 n_out=32,           
                 n_layers=2,
                 d_model=128, 
                 nhead=4, 
                 noise=0.05, 
                 k=64, 
                 use_gene_embed=False, 
                 num_gene_embed=48292,
                 dropout=0.1
                 ):
        super(RNAEncoder, self).__init__()

        self.n_genes = n_genes
        self.d_model = d_model
        self.n_out = n_out
        self.noise = noise
        self.use_gene_embed = use_gene_embed
        self.n_layers = n_layers
        
        self.gene_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        if use_gene_embed:
            self.gene_idx_embed = GeneEmbeding(num_gene_embed, d_model)
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, n_genes, d_model) * 0.01)
        
        self.trans = nn.ModuleList([
            LinformerBlock(
                dim=d_model, 
                seq_len=n_genes + 1,  # +1 for CLS token
                k=k, 
                heads=nhead, 
                mlp_dim=d_model*4,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.to_latent = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_out)  
        )
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def add_noise(self, x, noise_std=0.05):
        if self.training and noise_std > 0:
            return x + torch.randn_like(x) * noise_std
        return x

    def forward(self, x, gene_idx=None, get_attn = False, get_gene_embed=False, get_attn_E=False):
        batch_size = x.size(0)
        x = self.add_noise(x, noise_std=self.noise if hasattr(self, 'training') else 0)
        x = x.unsqueeze(-1)

        x = self.gene_embed(x)
        if self.use_gene_embed and gene_idx is not None:
            if gene_idx is None:
                gene_idx = torch.arange(self.n_genes, device=x.device).unsqueeze(0).expand(batch_size, -1)
            x = x + self.gene_idx_embed(gene_idx)
        else:
            x = x + self.pos_embed

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        if get_attn_E:
            for i in range(self.n_layers):
                if i==self.n_layers-1:
                    x, attn_E = self.trans[i](x, False, True)
                else:
                    x = self.trans[i](x)
        elif get_attn:
            for i in range(self.n_layers):
                if i==self.n_layers-1:
                    x, attn = self.trans[i](x, True)
                else:
                    x = self.trans[i](x)
        else:
            for layer in self.trans:
                x = layer(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        mu = self.to_latent(cls_output)
        
        if get_attn_E:
            return mu, attn_E
        elif get_attn:
            return mu, attn
        
        if get_gene_embed:
            gene_embed = x[:, 1:]
            return mu, gene_embed
        
        return mu
    
    def encode(self, x, gene_idx=None):
        mu= self.forward(x, gene_idx)
        return mu                     

class RNADecoder(nn.Module):
    def __init__(
        self, 
        input_dim = 32,
        hidden_dim = 128,
        output_dim = 2000,
        dropout=0.1,
        n_batch = None
        ):
        super(RNADecoder, self).__init__()
        self.n_batch = n_batch
        if self.n_batch is not None:
            input_dim += self.n_batch
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, output_dim),
        )
        
    
    def forward(self, x, batch_idx=None):
        if batch_idx is not None and self.n_batch is not None:
            batch_onehot = F.one_hot(batch_idx, num_classes=self.n_batch).float().to(x.device)
            x = torch.cat([x, batch_onehot], dim=-1)
        x = self.net(x)
        return x
        
class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k, heads=4):
        super().__init__()

        self.dim = dim
        self.seq_len = seq_len
        self.k = k
        self.heads = heads
        self.head_dim = dim // heads

        assert self.head_dim * heads == dim

        # Q K V projection
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Linformer projection matrices
        self.E_k = nn.Parameter(torch.randn(heads, k, seq_len))
        self.E_v = nn.Parameter(torch.randn(heads, k, seq_len))

        self.out = nn.Linear(dim, dim)

    def forward(self, x, get_attn=False, get_attn_E=False):
        """
        x: (batch, seq_len, dim)
        """

        b, n, d = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # split heads
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        # shapes
        # q = (b, h, n, d)
        # k = (b, h, n, d)

        # Linformer projection
        k = torch.einsum("hkn,bhnd->bhkd", self.E_k, k)
        v = torch.einsum("hkn,bhnd->bhkd", self.E_v, v)

        # attention
        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        if get_attn:
            A_full = torch.einsum("bhnk,hkm->bhnm", attn, self.E_k)
            A_full = A_full.mean(dim=1)
        
        if get_attn_E:
            attn_E = attn.mean(dim=1)

        out = torch.matmul(attn, v)

        # merge heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, n, d)

        if get_attn_E:
            return self.out(out), attn_E
        if get_attn:
            return self.out(out), A_full
        
        return self.out(out)
    
class LinformerBlock(nn.Module):
    def __init__(self, dim, seq_len, k, heads, mlp_dim, dropout=0.1):
        super().__init__()

        self.attn = LinformerSelfAttention(
            dim=dim,
            seq_len=seq_len,
            k=k,
            heads=heads
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, get_attn=False, get_attn_E=False):

        if get_attn_E:
            y, attn_E = self.attn(self.norm1(x), False, True)
            x = x + y
        elif get_attn:
            y, attn = self.attn(self.norm1(x), True)
            x = x + y
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        if get_attn_E:
            return x, attn_E
        if get_attn:
            return x, attn
        
        return x
    
class GeneEmbeding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embsize)
        return x
    
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes, input_dim=32):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x)