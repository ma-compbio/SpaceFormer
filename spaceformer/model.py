import torch
import math
import numpy as np
import scanpy as sc
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from .covet import covet_sqrt
from .utils import sce_loss, get_logger
from tensorboardX import SummaryWriter
from functools import partial
import os

class MLPCelltype(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPCelltype, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    

class MeanCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(MeanCelltype, self).__init__()
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


class MeanAddCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(MeanAddCelltype, self).__init__()
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


class CovetCelltype(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, output_dim):
        super(CovetCelltype, self).__init__()
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


class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
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


class GlobalTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim):
        super(GlobalTransformerCelltype, self).__init__()
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

        self.decoder = Attention(input_dim)
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


class LocalAttention(nn.Module):
    def __init__(self, d_model):
        super(LocalAttention, self).__init__()
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


class LocalTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim):
        super(LocalTransformerCelltype, self).__init__()
        self.encoder = LocalAttention(input_dim)
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


class SpatialAttention(nn.Module):
    def __init__(self, d_model, gamma):
        super(SpatialAttention, self).__init__()
        self.gamma = gamma
        self.d_model = d_model

        self.Q_real = nn.Linear(d_model, d_model, bias=False)
        self.Q_fake = nn.Linear(d_model, d_model, bias=False)
        self.K_real = nn.Linear(d_model, d_model, bias=False)
        self.K_fake = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, real_edge_mask, fake_edge_mask):
        Q_real = self.Q_real(x)
        K_real = self.K_real(x)
        Q_fake = self.Q_fake(x)
        K_fake = self.K_fake(x)
        V = self.V(x)

        real_scores = torch.matmul(Q_real, K_real.transpose(0, 1)) / math.sqrt(self.d_model)
        real_scores.masked_fill_(real_edge_mask, -1e9)
        real_scores_max, _ = torch.max(real_scores, dim=1, keepdim=True)
        fake_scores = torch.matmul(Q_fake, K_fake.transpose(0, 1)) / math.sqrt(self.d_model)
        fake_scores.masked_fill_(fake_edge_mask, -1e9)
        fake_scores_max, _ = torch.max(fake_scores, dim=1, keepdim=True)
        max_scores = torch.maximum(real_scores_max, fake_scores_max)

        real_scores = real_scores - max_scores
        real_scores = torch.exp(real_scores) / (1 + self.gamma)
        fake_scores = fake_scores - max_scores
        fake_scores = self.gamma * torch.exp(fake_scores) / (1 + self.gamma)

        scores = real_scores + fake_scores

        attn = scores / torch.sum(scores, dim=-1, keepdim=True)
        cntx = torch.matmul(attn, V)
    
        return cntx, attn


class SpatialTransformerCelltype(nn.Module):
    def __init__(self, dropout, input_dim, ffn_dim, hidden_dim, output_dim, gamma):
        super(SpatialTransformerCelltype, self).__init__()
        self.encoder = SpatialAttention(input_dim, gamma)
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

        self.decoder = SpatialAttention(input_dim, gamma)
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


def build_model_celltype(args):
    if args.model == 'MLP':
        model = MLPCelltype(args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'Mean':
        model = MeanCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'MeanAdd':
        model = MeanAddCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'Covet':
        model = CovetCelltype(args.n_neighs, args.input_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'GlobalTransformer':
        model = GlobalTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'LocalTransformer':
        model = LocalTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim)
    elif args.model == 'SpatialTransformer':
        model = SpatialTransformerCelltype(args.dropout, args.input_dim, args.ffn_dim, args.hidden_dim, args.output_dim, args.gamma)
    return model


class SpaceFormer(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim, gamma):
        super(SpaceFormer, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = SpatialAttention(input_dim, gamma)
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

        self.decoder = SpatialAttention(input_dim, gamma)
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

def train(model, dataset, device='cuda', optim_type='adam', lr=1e-4, weight_decay=0, warmup=8, max_epoch=200, loss_fn='sce', alpha=3, log_dir='log/'):
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    parameters = model.parameters()
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
        criterion = partial(sce_loss, alpha=alpha)
    elif loss_fn == "mse":
        criterion = nn.MSELoss()    

    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = SummaryWriter(logdir=log_dir)

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        for batch in loader:
            inputs = batch[0].to(device).squeeze(0)
            labels = batch[1].to(device).squeeze(0)
            real_edge_mask = batch[2].to(device).squeeze(0)
            fake_edge_mask = batch[3].to(device).squeeze(0)
            x_init, x_recon, encode_weights, embedding = model(inputs, real_edge_mask, fake_edge_mask)
            loss = criterion(x_init, x_recon)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"Epoch {epoch + 1}: train_loss {train_loss / len(loader)}")
        writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
        writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
        scheduler.step()

class GlobalTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim):
        super(GlobalTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

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

        self.decoder = Attention(input_dim)
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
    

class LocalTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim):
        super(LocalTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = LocalAttention(input_dim)
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


class SpatialTransformerPretrain(nn.Module):
    def __init__(self, cell_mask_rate, gene_mask_rate, dropout, input_dim, ffn_dim, gamma):
        super(SpatialTransformerPretrain, self).__init__()
        self.cell_mask_rate = cell_mask_rate
        self.gene_mask_rate = gene_mask_rate

        self.encoder = SpatialAttention(input_dim, gamma)
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

        self.decoder = SpatialAttention(input_dim, gamma)
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


def build_model_pretrain(args):
    if args.model == 'GlobalTransformer':
        model = GlobalTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim)
    elif args.model == 'LocalTransformer':
        model = LocalTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim)
    elif args.model == 'SpatialTransformer':
        model = SpatialTransformerPretrain(args.cell_mask_rate, args.gene_mask_rate, args.dropout, args.input_dim, args.ffn_dim, args.gamma)
    return model


class MLPImputation(nn.Module):
    def __init__(self, input_dim, hidden_dim, imputation_rate):
        super(MLPImputation, self).__init__()
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
    

class MeanImputation(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, imputation_rate):
        super(MeanImputation, self).__init__()
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
    

class CovetImputation(nn.Module):
    def __init__(self, n_neighs, input_dim, hidden_dim, imputation_rate):
        super(CovetImputation, self).__init__()
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


class GlobalTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate):
        super(GlobalTransformerImputation, self).__init__()

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

        self.decoder = Attention(input_dim)
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
    

class LocalTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate):
        super(LocalTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = LocalAttention(input_dim)
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


class SpatialTransformerImputation(nn.Module):
    def __init__(self, input_dim, ffn_dim, dropout, imputation_rate, gamma):
        super(SpatialTransformerImputation, self).__init__()

        self.imputation_rate = imputation_rate

        self.encoder = SpatialAttention(input_dim, gamma)
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

        self.decoder = SpatialAttention(input_dim, gamma)
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


def build_model_imputation(args):
    if args.model == 'MLP':
        model = MLPImputation(args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'Mean':
        model = MeanImputation(args.n_neighs, args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'Covet':
        model = CovetImputation(args.n_neighs, args.input_dim, args.hidden_dim, args.imputation_rate)
    elif args.model == 'GlobalTransformer':
        model = GlobalTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate)
    elif args.model == 'LocalTransformer':
        model = LocalTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate)
    elif args.model == 'SpatialTransformer':
        model = SpatialTransformerImputation(args.input_dim, args.ffn_dim, args.dropout, args.imputation_rate, args.gamma)
    return model