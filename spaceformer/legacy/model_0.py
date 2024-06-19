import torch
import math
import numpy as np
import scanpy as sc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from .covet import covet_sqrt
from .utils import _sce_loss, _get_logger
from tensorboardX import SummaryWriter
from functools import partial
from .dataset import _SpaceFormerDataset
import os

class _MLPCelltype(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(_MLPCelltype, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    

class _MeanCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(_MeanCelltype, self).__init__()
        self.n_neighs = n_neighs
        self.fc1 = nn.Linear(input_dim + input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, real_edge_mask, fake_edge_mask):
        node_indices = torch.nonzero(fake_edge_mask > 0)[:, 1]
        niche_feat = x[node_indices].reshape(x.shape[0], self.n_neighs, -1)
        res = torch.sum(niche_feat, dim=1) / self.n_neighs
        hidden = torch.cat((x, res), dim=1)
        out = self.fc1(hidden)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class _MeanAddCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(_MeanAddCelltype, self).__init__()
        self.n_neighs = n_neighs
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, real_edge_mask, fake_edge_mask):
        node_indices = torch.nonzero(fake_edge_mask > 0)[:, 1]
        niche_feat = x[node_indices].reshape(x.shape[0], self.n_neighs, -1)
        res = torch.sum(niche_feat, dim=1) / self.n_neighs
        hidden = x + res
        out = self.fc1(hidden)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class _CovetCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(_CovetCelltype, self).__init__()
        self.n_neighs = n_neighs
        self.fc1 = nn.Linear(input_dim + 32 * 32, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, coordinates, highly_variable_genes):
        res = covet_sqrt(highly_variable_genes.cpu().numpy(), coordinates.cpu().numpy(), self.n_neighs)
        res = torch.from_numpy(res.reshape(-1, 32 * 32)).to(x.device)
        hidden = torch.cat((x, res), dim=1)
        out = self.fc1(hidden)
        out = self.activation(out)
        out = self.fc2(out)
        return out


class _Attention(nn.Module):
    def __init__(self, d_model):
        super(_Attention, self).__init__()
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d_model)
        max_values, _ = torch.max(scores, dim=1, keepdim=True)
        scores = torch.exp(scores - max_values)

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class _GlobalTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim):
        super(_GlobalTransformerCelltype, self).__init__()
        self.encoder = _Attention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _Attention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x):
        hidden, encode_weights = self.encoder(x)
        hidden = self.dropout1(hidden)
        hidden = hidden + x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        return h


class _LocalAttention(nn.Module):
    def __init__(self, d_model):
        super(_LocalAttention, self).__init__()
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.d_model)
        scores.masked_fill_(mask, -1e9)
        max_values, _ = torch.max(scores, dim=1, keepdim=True)
        scores = torch.exp(scores - max_values)

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class _LocalTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim):
        super(_LocalTransformerCelltype, self).__init__()
        self.encoder = _LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )


    def forward(self, x, real_edge_mask, fake_edge_mask):
        hidden, encode_weights = self.encoder(x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        return h


class _SpatialAttention(nn.Module):
    def __init__(self, d_model, gamma, attn_hidden_dim=None):
        super(_SpatialAttention, self).__init__()
        self.gamma = gamma
        self.d_model = d_model

        if attn_hidden_dim is None:
            attn_hidden_dim = d_model
        self.attn_hidden_dim = attn_hidden_dim

        self.Q_real = nn.Linear(d_model, attn_hidden_dim, bias=False)
        self.Q_fake = nn.Linear(d_model, attn_hidden_dim, bias=False)
        self.K_real = nn.Linear(d_model, attn_hidden_dim, bias=False)
        self.K_fake = nn.Linear(d_model, attn_hidden_dim, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, real_edge_mask, fake_edge_mask):
        Q_real = self.Q_real(x)
        K_real = self.K_real(x)
        Q_fake = self.Q_fake(x)
        K_fake = self.K_fake(x)
        V = self.V(x)

        real_scores = torch.matmul(Q_real, K_real.transpose(0, 1)) / math.sqrt(self.attn_hidden_dim)
        real_scores.masked_fill_(real_edge_mask, -1e9)
        real_scores_max, _ = torch.max(real_scores, dim=1, keepdim=True)
        fake_scores = torch.matmul(Q_fake, K_fake.transpose(0, 1)) / math.sqrt(self.attn_hidden_dim)
        fake_scores.masked_fill_(fake_edge_mask, -1e9)
        fake_scores_max, _ = torch.max(fake_scores, dim=1, keepdim=True)
        max_scores = torch.maximum(real_scores_max, fake_scores_max)

        # real_scores = real_scores - max_scores
        real_scores = torch.exp(real_scores) / (1 + self.gamma)
        # fake_scores = fake_scores - max_scores
        fake_scores = self.gamma * torch.exp(fake_scores) / (1 + self.gamma)

        scores = real_scores + fake_scores

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class SpaceFormer(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim, gamma):
        """SpaceFormer Model

        :param cell_mask_rate: Probablity of masking a cell in training
        :param gene_mask_rate: Probablity of masking a feature (gene) in training
        :param dropout: Dropout rate duing training
        :param input_dim: Input dimension (number of genes)
        :param ffn_dim: Dimension of hidden layer
        :param gamma: Contrast of global / local attention.
        """
        super(SpaceFormer, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = _SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x, real_edge_mask, fake_edge_mask):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff1(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon

    def transform():
        raise NotImplemented("")

    def get_gene_attn(self, separate_q_k:bool=False):
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.encoder.Q_real.weight.detach().cpu()
        k = self.encoder.K_real.weight.detach().cpu()
        if separate_q_k:
            return q, k
        else:
            return q.T @ k

    def get_attn_and_embedding(self, dataset: _SpaceFormerDataset, device:str='cuda'):
        """Get attention weights and embeddings
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        encode_weights = []
        embeddings = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in loader:
            inputs = batch[0].to(device).squeeze(0)
            real_edge_mask = batch[2].to(device).squeeze(0)
            fake_edge_mask = batch[3].to(device).squeeze(0)
            _, _, encode_weight, embedding = self(inputs, real_edge_mask, fake_edge_mask)
            encode_weights.append(encode_weight.detach().cpu().numpy())
            embeddings.append(embedding.detach().cpu().numpy())
        return encode_weights, embeddings

    def fit(self, dataset: _SpaceFormerDataset, device:str='cuda', optim_type:str='adam', lr:float=1e-4, weight_decay:float=0., 
            warmup=8, max_epoch:int=200, loss_fn:str='sce', log_dir:str='log/'):
        """Create a PyTorch Dataset from a list of adata

        :param dataset: Dataset to be trained on
        :param device: Device to be used ("cpu" or "cuda")
        :param optim_type: Optimizer for fitting
        :param lr: Learning rate
        :param weight_decay: Weight decay factor
        :param warmup: Use higher training rate for earlier epochs
        :param max_epoch: maximum number of epochs
        :param loss_fn: Loss function for training
        :param log_dir: Directory to save logs

        :return: A `torch.Dataset` including all data.
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()
        opt_args = dict(lr=lr, weight_decay=weight_decay)

        if optim_type == "adam":
            optimizer = optim.Adam(parameters, **opt_args)
        elif optim_type == 'SGD':
            optimizer = optim.SGD(parameters, **opt_args)

        scheduler = lambda epoch :( 1 + np.cos((epoch - warmup) * np.pi / (max_epoch - warmup)) ) * 0.5 if epoch >= warmup else (epoch / warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        
        if loss_fn == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'sce':
            criterion = partial(_sce_loss, alpha=3)
        elif loss_fn == "mse":
            criterion = nn.MSELoss()    

        os.makedirs(log_dir, exist_ok=True)
        logger = _get_logger('train', log_dir)
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(max_epoch):
            self.train()
            train_loss = 0
            for batch in loader:
                inputs = batch[0].to(device).squeeze(0)
                labels = batch[1].to(device).squeeze(0)
                real_edge_mask = batch[2].to(device).squeeze(0)
                fake_edge_mask = batch[3].to(device).squeeze(0)
                x_init, x_recon, encode_weights, embedding = self(inputs, real_edge_mask, fake_edge_mask)
                loss = criterion(x_init, x_recon)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch + 1}: train_loss {train_loss / len(loader)}")
            writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            scheduler.step()


class _ThreeWayAttention(nn.Module):
    def __init__(self, d_in, local_d=None, global_d=None, d_out=None):
        super(_ThreeWayAttention, self).__init__()
        self.d_in = d_in

        if local_d is None:
            local_d = d_in
        if global_d is None:
            global_d = d_in
        if d_out is None:
            d_out = d_in

        self.local_Q = nn.Linear(d_in, local_d, bias=False)
        self.local_K = nn.Linear(d_in, local_d, bias=False)
        self.local_V = nn.Linear(d_in, d_out, bias=True)

        self.global_Q = nn.Linear(d_in, global_d, bias=False)
        self.global_K = nn.Linear(d_in, global_d, bias=False)
        self.global_V = nn.Linear(d_in, d_out, bias=True)

        self.essential_score = nn.Linear(d_in, 1, bias=False)
        self.essential_V = nn.Linear(d_in, d_out, bias=True)

    def forward(self, adj_matrix, x, masked_x=None, x_bar=None, get_details=False):
        if x_bar is None:
            x_bar = torch.mean(x, axis=0, keepdim=True)
            # Note: x_bar can have multiple pseudobulks.

        if masked_x is None:
            masked_x = x

        local_Q = self.local_Q(masked_x)
        local_K = self.local_K(x)
        local_V = self.local_V(x)

        global_Q = self.global_Q(masked_x)
        global_K = self.global_K(x_bar)
        global_V = self.global_V(x_bar)

        essential_scores = self.essential_score(masked_x)
        essential_V = self.essential_V(masked_x) # note this is still masked x

        local_scores = torch.matmul(local_Q, local_K.transpose(0, 1)) / math.sqrt(self.d_in)
        local_scores.masked_fill_(~adj_matrix, -1e9)
        
        global_scores = torch.matmul(global_Q, global_K.transpose(0, 1)) / math.sqrt(self.d_in)

        max_local_score, _ = torch.max(local_scores, dim=1, keepdim=True)
        max_global_score, _ = torch.max(global_scores, dim=1, keepdim=True)
        max_score, _ = torch.max(torch.cat([max_local_score, max_global_score, essential_scores], dim=1), dim=1, keepdim=True)

        local_scores = local_scores - max_score
        global_scores = global_scores - max_score
        essential_scores = essential_scores - max_score

        local_scores = torch.exp(local_scores)
        global_scores = torch.exp(global_scores)
        essential_scores = torch.exp(essential_scores)

        # Global and local attention sums up to 1 after transform.
        sum_score = torch.sum(local_scores, dim=-1, keepdim=True) + torch.sum(global_scores, dim=-1, keepdim=True) + essential_scores
        local_attn = local_scores / sum_score
        global_attn = global_scores / sum_score
        essential_attn = essential_scores / sum_score

        local_res = torch.matmul(local_attn, local_V)
        global_res = torch.matmul(global_attn, global_V)
        essential_res = essential_attn * essential_V
        res = local_res + global_res + essential_res

        if get_details:
            return res, (local_res, global_res, essential_res), (local_attn, global_attn, essential_attn)
        else:
            return res
    

class SpaceFormerLite(nn.Module):
    def __init__(self, d_in, local_d, global_d):
        """SpaceFormerLite Model (encoder only)

        :param cell_mask_rate: Probablity of masking a cell in training
        :param gene_mask_rate: Probablity of masking a feature (gene) in training
        :param dropout: Dropout rate duing training
        :param input_dim: Input dimension (number of genes)
        :param ffn_dim: Dimension of hidden layer
        :param gamma: Contrast of global / local attention.
        """
        super(SpaceFormerLite, self).__init__()

        # Three way attention
        self.spatial_gather = _ThreeWayAttention(d_in, local_d, global_d, d_in)

    def masking(self, x, mask_rate):
        # All cells are masked
        random_mask = torch.rand(x.shape, device=x.get_device()) < mask_rate
        out_x = x.clone()
        out_x.masked_fill_(random_mask, 0.)

        return out_x

    def forward(self, adj_matrix, x, masked_x, get_details=False):
        # Spatial component
        z, sub_res, attn = self.spatial_gather(adj_matrix, x, masked_x, get_details=True)

        # Max pooling with the input
        x_recon = torch.maximum(z, masked_x)

        if get_details:
            
            return x_recon, z, sub_res, attn
        else:
            return x_recon

    def fit(self, dataset: _SpaceFormerDataset, masking_rate=0.5, device:str='cuda', optim_type:str='adam', lr:float=1e-4, weight_decay:float=0., 
            warmup=8, max_epoch:int=200, loss_fn:str='mse', log_dir:str='log/'):
        """Create a PyTorch Dataset from a list of adata

        :param dataset: Dataset to be trained on
        :param device: Device to be used ("cpu" or "cuda")
        :param optim_type: Optimizer for fitting
        :param lr: Learning rate
        :param weight_decay: Weight decay factor
        :param warmup: Use higher training rate for earlier epochs
        :param max_epoch: maximum number of epochs
        :param loss_fn: Loss function for training
        :param log_dir: Directory to save logs

        :return: A `torch.Dataset` including all data.
        """
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()
        opt_args = dict(lr=lr, weight_decay=weight_decay)

        if optim_type == "adam":
            optimizer = optim.Adam(parameters, **opt_args)
        elif optim_type == 'SGD':
            optimizer = optim.SGD(parameters, **opt_args)

        scheduler = lambda epoch :( 1 + np.cos((epoch - warmup) * np.pi / (max_epoch - warmup)) ) * 0.5 if epoch >= warmup else (epoch / warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        
        if loss_fn == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'sce':
            criterion = partial(_sce_loss, alpha=3)
        elif loss_fn == "mse":
            criterion = nn.MSELoss()

        os.makedirs(log_dir, exist_ok=True)
        logger = _get_logger('train', log_dir)
        writer = SummaryWriter(logdir=log_dir)

        for epoch in range(max_epoch):
            self.train()
            train_loss = 0
            for x, adj_matrix in loader:
                x = x.to(device).squeeze(0)
                adj_matrix = adj_matrix.to(device).squeeze(0)
                masked_x = self.masking(x, masking_rate)
                x_recon = self(adj_matrix, x, masked_x)
                loss = criterion(x, x_recon)
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.info(f"Epoch {epoch + 1}: train_loss {train_loss / len(loader)}")
            writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            scheduler.step()

    def transform():
        raise NotImplemented("")

    def get_local_gene_attn(self, separate_q_k:bool=False, to_numpy=True):
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.spatial_gather.local_Q.weight.detach().cpu()
        k = self.spatial_gather.local_K.weight.detach().cpu()
        if separate_q_k:
            if to_numpy:
                return q.numpy(), k.numpy()
            else:
                return q, k
        else:
            if to_numpy:
                (q.T @ k).numpy()
            else:
                return q.T @ k
        
    def get_global_gene_attn(self, separate_q_k:bool=False, to_numpy=True):
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.spatial_gather.global_Q.weight.detach().cpu()
        k = self.spatial_gather.global_K.weight.detach().cpu()
        if separate_q_k:
            if to_numpy:
                return q.numpy(), k.numpy()
            else:
                return q, k
        else:
            if to_numpy:
                (q.T @ k).numpy()
            else:
                return q.T @ k
        
    def get_essential_gene_attn(self, to_numpy=True):
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        temp = self.spatial_gather.essential_score.weight.detach().cpu()
        if to_numpy:
            return temp.numpy()
        else:
            return temp

class _SpatialTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim, gamma):
        super(_SpatialTransformerCelltype, self).__init__()
        self.encoder = _SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, real_edge_mask, fake_edge_mask):
        hidden, encode_weights = self.encoder(x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        return h


def _build_model_celltype(args):
    if args.model == 'MLP':
        model = _MLPCelltype(args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'Mean':
        model = _MeanCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'MeanAdd':
        model = _MeanAddCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'Covet':
        model = _CovetCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'GlobalTransformer':
        model = _GlobalTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'LocalTransformer':
        model = _LocalTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'SpatialTransformer':
        model = _SpatialTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim, args.gamma)
    return model

class _GlobalTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim):
        super(_GlobalTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = _Attention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _Attention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)
        
        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon
    

class _LocalTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim):
        super(_LocalTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = _LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x, real_edge_mask, fake_edge_mask):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff1(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon


class _SpatialTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim, gamma):
        """
        """
        super(_SpatialTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = _SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.ff1 = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm4 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def encoding_mask_nodes(self, x, cell_mask_rate, gene_mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(cell_mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        mask_x = x[mask_nodes].clone()
        gene_mask = torch.zeros_like(mask_x, dtype=torch.bool).to(x.device)
        for i in range(mask_x.size(0)):
            num_to_zero =  int(gene_mask_rate * mask_x.size(1))
            indices_to_zero = torch.randperm(mask_x.size(1))[:num_to_zero]
            mask_x[i, indices_to_zero] = 0
            gene_mask[i, indices_to_zero] = True
        
        out_x = x.clone()
        out_x[mask_nodes] = mask_x

        return out_x, mask_nodes, gene_mask

    def forward(self, x, real_edge_mask, fake_edge_mask):
        use_x, mask_nodes, gene_mask = self.encoding_mask_nodes(x, self.cell_mask_rate, self.gene_mask_rate)

        hidden, encode_weights = self.encoder(use_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + use_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        hidden[mask_nodes][gene_mask] = 0

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        recon = self.norm4(recon + self.ff1(recon))

        h = self.head(recon)

        x_init = x[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)
        x_recon = h[mask_nodes][gene_mask].view(mask_nodes.shape[0], -1)

        return x_init, x_recon, encode_weights, recon


def _build_model_pretrain(args):
    if args.model == 'GlobalTransformer':
        model = _GlobalTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim)
    elif args.model == 'LocalTransformer':
        model = _LocalTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim)
    elif args.model == 'SpatialTransformer':
        model = _SpatialTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim, args.gamma)
    return model


class _MLPImputation(nn.Module):
    def __init__(self, input_dim, hidden_dim, imputation_rate):
        super(_MLPImputation, self).__init__()
        self.imputation_rate = imputation_rate

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list), device=x.device)[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        recon = self.mlp(drop_x)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = recon[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon
    
    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)

    def evaluation(self, raw, gene_list):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        recon = self.mlp(drop_x)

        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]
            gene_init = raw[:, gene_idx]
            gene_recon = recon[:, gene_idx]

            mask = torch.nonzero(gene_init, as_tuple=True)[0]
            if len(np.unique(gene_init[mask].detach().cpu().numpy())) < 2:
                continue
            
            pearson, _ = pearsonr(gene_init[mask].detach().cpu().numpy(), gene_recon[mask].detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init[mask].detach().cpu().numpy(), gene_recon[mask].detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list
    

class _MeanImputation(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, imputation_rate):
        super(_MeanImputation, self).__init__()
        self.n_neighs = n_neighs
        self.imputation_rate = imputation_rate

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list), device=x.device)[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, real_edge_mask, fake_edge_mask):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        node_indices = torch.nonzero(fake_edge_mask > 0)[:, 1]
        niche_feat = drop_x[node_indices].reshape(drop_x.shape[0], self.n_neighs, -1)
        res = torch.sum(niche_feat, dim=1) / self.n_neighs
        hidden = torch.cat((drop_x, res), dim=1)
        
        recon = self.mlp(hidden)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = recon[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon
    
    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)

    def evaluation(self, raw, gene_list, real_edge_mask, fake_edge_mask):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        node_indices = torch.nonzero(fake_edge_mask > 0)[:, 1]
        niche_feat = drop_x[node_indices].reshape(drop_x.shape[0], self.n_neighs, -1)
        res = torch.sum(niche_feat, dim=1) / self.n_neighs
        hidden = torch.cat((drop_x, res), dim=1)

        recon = self.mlp(hidden)

        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = recon[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list
    

class _CovetImputation(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, imputation_rate):
        super(_CovetImputation, self).__init__()
        self.imputation_rate = imputation_rate
        self.n_neighs = n_neighs

        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 32 * 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list), device=x.device)[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, coordinates, highly_variable_genes):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        res = covet_sqrt(highly_variable_genes.cpu().numpy(), coordinates.cpu().numpy(), self.n_neighs)
        res = torch.from_numpy(res.reshape(-1, 32 * 32)).to(x.device)
        hidden = torch.cat((drop_x, res), dim=1)

        recon = self.mlp(hidden)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = recon[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon
    
    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)

    def evaluation(self, raw, gene_list, coordinates, highly_variable_genes):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        res = covet_sqrt(highly_variable_genes.cpu().numpy(), coordinates.cpu().numpy(), self.n_neighs)
        res = torch.from_numpy(res.reshape(-1, 32 * 32)).to(drop_x.device)
        hidden = torch.cat((drop_x, res), dim=1)
        
        recon = self.mlp(hidden)

        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = recon[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list


class _GlobalTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate):
        super(_GlobalTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = Attention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _Attention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list))[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        hidden, encode_weights = self.encoder(drop_x)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = h[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon

    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)
    
    def evaluation(self, raw, gene_list):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        hidden, encode_weights = self.encoder(drop_x)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)
        
        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]
            gene_init = raw[:, gene_idx]
            gene_recon = h[:, gene_idx]

            mask = torch.nonzero(gene_init, as_tuple=True)[0]
            if len(np.unique(gene_init[mask].detach().cpu().numpy())) < 2:
                continue

            pearson, _ = pearsonr(gene_init[mask].detach().cpu().numpy(), gene_recon[mask].detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init[mask].detach().cpu().numpy(), gene_recon[mask].detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list
    

class _LocalTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate):
        super(_LocalTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = _LocalAttention(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _LocalAttention(input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list))[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, real_edge_mask, fake_edge_mask):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        hidden, encode_weights = self.encoder(drop_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = h[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon

    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)
    
    def evaluation(self, raw, gene_list, real_edge_mask, fake_edge_mask):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        hidden, encode_weights = self.encoder(drop_x, real_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)
        
        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = h[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list


class _SpatialTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate, gamma):
        super(_SpatialTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = _SpatialAttention(input_dim, gamma)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(input_dim)

        self.ff = nn.Sequential(
            nn.Linear(input_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(input_dim)

        self.decoder = _SpatialAttention(input_dim, gamma)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(input_dim)

        self.head = nn.Linear(input_dim, input_dim)

    def drop(self, x, gene_list, num_drop):
        mask = torch.zeros_like(x, dtype=bool)
    
        for i in range(x.shape[0]):
            random_indices = torch.randperm(len(gene_list))[:num_drop]
            dropout_indices = gene_list[random_indices]
            mask[i, dropout_indices] = True
        
        drop_x = x.clone()
        drop_x[mask] = 0

        return drop_x, mask

    def forward(self, x, gene_list, real_edge_mask, fake_edge_mask):
        drop_x, drop_mask = self.drop(x, gene_list, int(self.imputation_rate * len(gene_list)))
        
        hidden, encode_weights = self.encoder(drop_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)

        x_init = x[drop_mask].reshape(x.shape[0], -1)
        x_recon = h[drop_mask].reshape(x.shape[0], -1)

        return x_init, x_recon
    
    def preprocess(self, x):
        adata = sc.AnnData(X=x.detach().cpu().numpy())
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        X = adata.X
        return torch.from_numpy(X).to(x.device)
    
    def evaluation(self, raw, gene_list, real_edge_mask, fake_edge_mask):
        drop_raw, drop_mask = self.drop(raw, gene_list, int(self.imputation_rate * len(gene_list)))
        drop_x = self.preprocess(drop_raw)

        hidden, encode_weights = self.encoder(drop_x, real_edge_mask, fake_edge_mask)
        hidden = self.dropout1(hidden)
        hidden = hidden + drop_x
        hidden = self.norm1(hidden)

        hidden = self.norm2(hidden + self.ff(hidden))

        recon, decode_weights = self.decoder(hidden, real_edge_mask, fake_edge_mask)
        recon = self.dropout2(recon)
        recon = recon + hidden
        recon = self.norm3(recon)

        h = self.head(recon)
        
        pearson_list = []
        spearman_list = []
        for i in range(len(gene_list)):
            gene_idx = gene_list[i]

            gene_mask_i = drop_mask[:, gene_idx]
            
            gene_init = raw[:, gene_idx][gene_mask_i]
            gene_recon = h[:, gene_idx][gene_mask_i]

            pearson, _ = pearsonr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            pearson_list.append(pearson)
            spearman, _ = spearmanr(gene_init.detach().cpu().numpy(), gene_recon.detach().cpu().numpy())
            spearman_list.append(spearman)

        return pearson_list, spearman_list


def _build_model_imputation(args):
    if args.model == 'MLP':
        model = _MLPImputation(args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'Mean':
        model = _MeanImputation(args.n_neighs, args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'Covet':
        model = _CovetImputation(args.n_neighs, args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'GlobalTransformer':
        model = _GlobalTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate)
    elif args.model == 'LocalTransformer':
        model = _LocalTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate)
    elif args.model == 'SpatialTransformer':
        model = _SpatialTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate, args.gamma)
    return model