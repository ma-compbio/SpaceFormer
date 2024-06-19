import torch
import math
import numpy as np
import scanpy as sc
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from .utils import _sce_loss, _get_logger
from tensorboardX import SummaryWriter
from functools import partial
from .dataset import _SpaceFormerDataset
import os
from typing import Literal

class _ThreeWayAttention(nn.Module):
    def __init__(self, d_in, local_d=None, global_d=None, d_out=None, 
                 ego_bottleneck=None, local_bottleneck=None, global_bottleneck=None):
        super(_ThreeWayAttention, self).__init__()
        self.d_in = d_in

        if local_d is None:
            local_d = d_in
        if global_d is None:
            global_d = d_in
        if d_out is None:
            d_out = d_in

        self.ego_score = nn.Linear(d_in, 1, bias=False)
        if ego_bottleneck is None:
            self.ego_V = nn.Linear(d_in, d_out, bias=True)
        else:
            self.ego_V = nn.Sequential(nn.Linear(d_in, ego_bottleneck, bias=False),
                                       nn.Linear(ego_bottleneck, d_out, bias=True))

        self.local_Q = nn.Linear(d_in, local_d, bias=False)
        self.local_K = nn.Linear(d_in, local_d, bias=False)
        if local_bottleneck is None:
            self.local_V = nn.Linear(d_in, d_out, bias=True)
        else:
            self.local_V = nn.Sequential(nn.Linear(d_in, local_bottleneck, bias=False),
                                         nn.Linear(local_bottleneck, d_out, bias=True))

        self.global_Q = nn.Linear(d_in, global_d, bias=False)
        self.global_K = nn.Linear(d_in, global_d, bias=False)
        if global_bottleneck is None:
            self.global_V = nn.Linear(d_in, d_out, bias=True)
        else:
            self.global_V = nn.Sequential(nn.Linear(d_in, global_bottleneck, bias=False),
                                          nn.Linear(global_bottleneck, d_out, bias=True))
        

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

        ego_scores = self.ego_score(masked_x)
        ego_V = self.ego_V(masked_x) # note this is still masked x

        local_scores = torch.matmul(local_Q, local_K.transpose(0, 1)) / math.sqrt(self.d_in)
        local_scores.masked_fill_(~adj_matrix, -1e9)
        
        global_scores = torch.matmul(global_Q, global_K.transpose(0, 1)) / math.sqrt(self.d_in)

        max_local_score, _ = torch.max(local_scores, dim=1, keepdim=True)
        max_global_score, _ = torch.max(global_scores, dim=1, keepdim=True)
        max_score, _ = torch.max(torch.cat([max_local_score, max_global_score, ego_scores], dim=1), dim=1, keepdim=True)

        local_scores = local_scores - max_score
        global_scores = global_scores - max_score
        ego_scores = ego_scores - max_score

        local_scores = torch.exp(local_scores)
        global_scores = torch.exp(global_scores)
        ego_scores = torch.exp(ego_scores)

        # Global and local attention sums up to 1 after transform.
        sum_score = torch.sum(local_scores, dim=-1, keepdim=True) + torch.sum(global_scores, dim=-1, keepdim=True) + ego_scores
        local_attn = local_scores / sum_score
        global_attn = global_scores / sum_score
        ego_attn = ego_scores / sum_score

        local_res = torch.matmul(local_attn, local_V)
        global_res = torch.matmul(global_attn, global_V)
        ego_res = (masked_x + ego_attn) * ego_V
        res = local_res + global_res + ego_res

        if get_details:
            return res, (local_res, global_res, ego_res), (local_attn, global_attn, ego_attn)
        else:
            return res
    

class SpaceFormer0(nn.Module):
    def __init__(self, features: list[str] | int, local_d, global_d, *,
                 ego_bottleneck=None, local_bottleneck=None, global_bottleneck=None):
        """SpaceFormerLite Model (encoder only)

        :param features: input feature (gene) names or the number of features
        :param local_d: 
        :param global_d: 
        """
        super(SpaceFormer0, self).__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        # Three way attention
        d_in = len(self.features)
        self.spatial_gather = _ThreeWayAttention(d_in, local_d, global_d, d_in,
                                                 ego_bottleneck, local_bottleneck, global_bottleneck)

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
        # x_recon = torch.maximum(z, masked_x)
        x_recon = z

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
                return (q.T @ k).numpy()
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
                return (q.T @ k).numpy()
            else:
                return q.T @ k
        
    def get_ego_gene_attn(self, to_numpy=True):
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        temp = self.spatial_gather.ego_score.weight.detach().cpu()
        if to_numpy:
            return temp.numpy()
        else:
            return temp

    def _get_transform(self, layer):
        if isinstance(layer, nn.Linear):
            v = layer.weight.detach().cpu()
            b = layer.bias.detach().cpu()
            return v, None, b
        else:
            v1 = layer[0].weight.detach().cpu()
            v2 = layer[1].weight.detach().cpu()
            b = layer[1].bias.detach().cpu()
            return v1, v2, b

    def get_transform(self, which=Literal["ego", "local", "global"], to_numpy=True):
        if which == "ego":
            temp = self._get_transform(self.spatial_gather.ego_V)
        elif which == "local":
            temp = self._get_transform(self.spatial_gather.local_V)
        elif which == "global":
            temp = self._get_transform(self.spatial_gather.global_V)
        
        if to_numpy:
            return tuple(i.numpy() if i is not None else None for i in temp)
        else:
            return temp


class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        return torch.exp(self._weight)

    def forward(self, x):
        return x @ self.weight.T


class NonNegBias(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self._bias = torch.nn.Parameter(torch.zeros(1, d))

    @property
    def bias(self):
        return torch.exp(self._bias)

    def forward(self, x):
        return x + self.bias


class _BilinearAttention(nn.Module):
    def __init__(self, d_in, d_ego, d_local, d_global, d_out=None):
        super(_BilinearAttention, self).__init__()
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.d_ego = d_ego
        self.d_local = d_local
        self.d_global = d_global

        self.bias = NonNegBias(d_out)

        self.qk_ego = nn.Linear(d_in, d_ego, bias=False)
        self.v_ego = NonNegLinear(d_ego, d_out, bias=False)

        self.q_local = nn.Linear(d_in, d_local, bias=False)
        self.k_local = NonNegLinear(d_in, d_local, bias=False)
        self.v_local = NonNegLinear(d_local, d_out, bias=False)

        if self.d_global > 0:
            self.q_global = nn.Linear(d_in, d_global, bias=False)
            self.k_global = NonNegLinear(d_in, d_global, bias=False)
            self.v_global = NonNegLinear(d_global, d_out, bias=False)
        else:
            self.q_global = None
            self.k_global = None
            self.v_global = None

    def forward(self, adj_matrix, x, masked_x=None, x_bar=None, superego=False):
        if x_bar is None:
            x_bar = torch.mean(x, axis=0, keepdim=True) # in which case, m := 1
            # Note: x_bar can have multiple pseudobulks.

        if masked_x is None:
            masked_x = x

        ego_scores = self.qk_ego(masked_x) / x.shape[1] 
        if superego:
            max_ego_score, _ = torch.max(ego_scores, dim=1, keepdim=True)
            ego_scores = torch.exp(ego_scores - max_ego_score)
            ego_attn = ego_scores / torch.sum(ego_scores, dim=-1, keepdim=True)
            return torch.exp(self.v_ego(ego_attn)), None, None
        
        q_local = self.q_local(masked_x) # n * g -> n * d
        k_local = self.k_local(x) # n * g -> n * d
        # (d * n * 1) x (d * 1 * n) -> d * n * n
        local_scores = q_local.transpose(-1, -2)[:, :, None] * k_local.transpose(-1, -2)[:, None, :] / x.shape[1] / x.shape[1]
        local_scores.masked_fill_(~adj_matrix, -1e9)
        local_scores = local_scores.transpose(-2, -3) # n * d * n

        max_local_score, _ = torch.max(local_scores.reshape((local_scores.shape[0], -1)), dim=1, keepdim=True)
        max_ego_score, _ = torch.max(ego_scores, dim=1, keepdim=True)

        if self.d_global > 0:
            q_global = self.q_global(masked_x) # n * g -> n * d
            k_global = self.k_global(x_bar)    # m * g -> m * d
            global_scores = q_global.transpose(-1, -2)[:, :, None] * k_global.transpose(-1, -2)[:, None, :] / x.shape[1] / x.shape[1]
            global_scores = global_scores.transpose(-2, -3) # n * d * m
            max_score, _ = torch.max(torch.cat([global_scores.reshape((global_scores.shape[0], -1)), 
                                                max_ego_score, 
                                                max_local_score], dim=1), dim=1, keepdim=True)
        else:
            max_score, _ = torch.max(torch.cat([max_ego_score, max_local_score], dim=1), dim=1, keepdim=True)

        ego_scores = torch.exp(ego_scores - max_score)
        local_scores = torch.exp(local_scores - max_score[:, :, None])
        local_scores = torch.sum(local_scores, dim=-1)

        if self.d_global > 0:
            global_scores = torch.exp(global_scores - max_score[:, :, None])
            global_scores = torch.sum(global_scores, dim=-1)
            sum_score = (torch.sum(ego_scores, dim=-1, keepdim=True) + 
                        torch.sum(local_scores, dim=-1, keepdim=True) + 
                        torch.sum(global_scores, dim=-1, keepdim=True))
            global_attn = global_scores / sum_score
            global_res = self.v_global(global_attn)
        else:
            sum_score = (torch.sum(ego_scores, dim=-1, keepdim=True) + 
                        torch.sum(local_scores, dim=-1, keepdim=True))

        ego_attn = ego_scores / sum_score
        local_attn = local_scores / sum_score

        ego_res = self.v_ego(ego_attn)
        local_res = self.v_local(local_attn)
        
        if self.d_global > 0:
            res = ego_res + local_res + global_res
        else:
            res = ego_res + local_res
            global_res = None
            global_attn = None

        res = self.bias(res)
        return res, (ego_res, local_res, global_res), (ego_attn, local_attn, global_attn)

        

class SpaceFormer(nn.Module):
    def __init__(self, features: list[str] | int, d_ego, d_local, d_global):
        """SpaceFormer2 Model 

        :param features: input feature (gene) names or the number of features
        :param local_d: 
        :param global_d: 
        """
        super(SpaceFormer, self).__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        # Three way attention
        d_in = len(self.features)
        self.spatial_gather = _BilinearAttention(d_in, d_ego, d_local, d_global, d_in)

    def masking(self, x, mask_rate):
        # All cells are masked
        random_mask = torch.rand(x.shape, device=x.get_device()) < mask_rate
        out_x = x.clone()
        out_x.masked_fill_(random_mask, 0.)

        return out_x

    def forward(self, adj_matrix, x, masked_x, get_details=False, superego=False):
        # Spatial component
        z, sub_res, attn = self.spatial_gather(adj_matrix, x, masked_x, superego=superego)

        # Max pooling with the input
        # x_recon = torch.maximum(z, masked_x)
        x_recon = z

        if get_details:
            return x_recon, z, sub_res, attn
        else:
            return x_recon

    def fit(self, dataset: _SpaceFormerDataset, masking_rate=0.0, device:str='cuda', optim_type:str='adam', lr:float=1e-4, weight_decay:float=0., 
            warmup=8, max_epoch:int=100, stop_eps=1e-3, stop_tol=3, loss_fn:str='mse', log_dir:str='log/'):
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
        self.train()

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()
        opt_args = dict(lr=lr, weight_decay=weight_decay)

        if optim_type == "adam":
            optimizer = optim.Adam(parameters, **opt_args)
        elif optim_type == 'SGD':
            optimizer = optim.SGD(parameters, **opt_args)

        # scheduler = lambda epoch :( 1 + np.cos((epoch - warmup) * np.pi / (max_epoch - warmup)) ) * 0.5 if epoch >= warmup else (epoch / warmup)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
        
        if loss_fn == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'sce':
            criterion = partial(_sce_loss, alpha=3)
        elif loss_fn == "mse":
            criterion = nn.MSELoss()

        os.makedirs(log_dir, exist_ok=True)
        logger = _get_logger('train', log_dir)
        # writer = SummaryWriter(logdir=log_dir)

        cnt = 0
        last_avg_loss = np.inf
        for epoch in range(max_epoch):
            total_loss = 0
            for x, adj_matrix in loader:
                x = x.to(device).squeeze(0)
                adj_matrix = adj_matrix.to(device).squeeze(0)
                masked_x = self.masking(x, masking_rate)
                x_recon = self(adj_matrix, x, masked_x)
                
                loss = criterion(x, x_recon)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(loader)
            logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss}:.5f")
            if last_avg_loss - avg_loss < stop_eps:
                cnt += 1
            if cnt >= stop_tol:
                logger.info(f"Stopping criterion met.")
                break
            # writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            # writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            # scheduler.step()
        else:
            logger.info(f"Maximum iterations reached.")

        return self

    def transform(self, x, adj_matrix):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.BoolTensor(adj_matrix)
            
            res, (ego_res, local_res, global_res), (ego_attn, local_attn, global_attn) = self(adj_matrix, x, get_details=True)
            return res, (ego_res, local_res, global_res), (ego_attn, local_attn, global_attn)


    def get_bias(self) -> np.array:
        b = self.spatial_gather.bias.bias.detach().cpu().numpy()
        return b.T

    def get_ego_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_ego.weight.detach().cpu().numpy()
        return qk, v.T

    def get_local_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        q = self.spatial_gather.q_local.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_local.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_local.weight.detach().cpu().numpy()
        return q, k, v.T
        
    def get_global_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.spatial_gather.q_global.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_global.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_global.weight.detach().cpu().numpy()
        return q, k, v.T
       
    def get_local_q_score(self) -> np.array:
        pass


class _BilinearAttention2(nn.Module):
    def __init__(self, d_in, d_ego, d_local, d_global, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.d_ego = d_ego
        self.d_local = d_local
        self.d_global = d_global

        self.bias = NonNegBias(d_out)

        self.qk_ego = NonNegLinear(d_in, d_ego, bias=False)
        self.v_ego = NonNegLinear(d_ego, d_out, bias=False)

        self.q_local = NonNegLinear(d_in, d_local, bias=False)
        self.k_local = NonNegLinear(d_in, d_local, bias=False)
        self.v_local = NonNegLinear(d_local, d_out, bias=False)

        if self.d_global > 0:
            self.q_global = NonNegLinear(d_in, d_global, bias=False)
            self.k_global = NonNegLinear(d_in, d_global, bias=False)
            self.v_global = NonNegLinear(d_global, d_out, bias=False)
        else:
            self.q_global = None
            self.k_global = None
            self.v_global = None

    def forward(self, adj_matrix, x, masked_x=None, x_bar=None):
        if x_bar is None:
            x_bar = torch.mean(x, axis=0, keepdim=True) # in which case, m := 1
            # Note: x_bar can have multiple pseudobulks.

        if masked_x is None:
            masked_x = x

        ego_scores = self.qk_ego(masked_x) # / x.shape[1]
        
        q_local = self.q_local(masked_x) # n * g -> n * d
        k_local = self.k_local(x) # n * g -> n * d
        # (d * n * 1) x (d * 1 * n) -> d * n * n
        local_scores = q_local.transpose(-1, -2)[:, :, None] * k_local.transpose(-1, -2)[:, None, :] # / x.shape[1] / x.shape[1]
        local_scores.masked_fill_(~adj_matrix, -1e9)
        local_scores = local_scores.transpose(-2, -3) # n * d * n
        local_scores = torch.sum(local_scores, dim=-1)

        if self.d_global > 0:
            q_global = self.q_global(masked_x) # n * g -> n * d
            k_global = self.k_global(x_bar)    # m * g -> m * d
            global_scores = q_global.transpose(-1, -2)[:, :, None] * k_global.transpose(-1, -2)[:, None, :] # / x.shape[1] / x.shape[1]
            global_scores = global_scores.transpose(-2, -3) # n * d * m

        if self.d_global > 0:
            global_scores = torch.sum(global_scores, dim=-1)
            sum_score = (torch.sum(ego_scores, dim=-1, keepdim=True) + 
                        torch.sum(local_scores, dim=-1, keepdim=True) + 
                        torch.sum(global_scores, dim=-1, keepdim=True))
            global_attn = global_scores / sum_score
            global_res = self.v_global(global_attn)
        else:
            sum_score = (torch.sum(ego_scores, dim=-1, keepdim=True) + 
                        torch.sum(local_scores, dim=-1, keepdim=True))

        ego_attn = ego_scores# / sum_score
        local_attn = local_scores# / sum_score

        ego_res = self.v_ego(ego_attn)
        local_res = self.v_local(local_attn)
        
        if self.d_global > 0:
            res = ego_res + local_res + global_res
        else:
            res = ego_res + local_res
            global_res = None
            global_attn = None

        res = self.bias(res)
        return res, (ego_res, local_res, global_res), (ego_attn, local_attn, global_attn)

        

class SpaceFormer2(nn.Module):
    def __init__(self, features: list[str] | int, d_ego, d_local, d_global):
        """SpaceFormer2 Model 

        :param features: input feature (gene) names or the number of features
        :param local_d: 
        :param global_d: 
        """
        super().__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        # Three way attention
        d_in = len(self.features)
        self.spatial_gather = _BilinearAttention2(d_in, d_ego, d_local, d_global, d_in)

    def masking(self, x, mask_rate):
        # All cells are masked
        random_mask = torch.rand(x.shape, device=x.get_device()) < mask_rate
        out_x = x.clone()
        out_x.masked_fill_(random_mask, 0.)

        return out_x

    def forward(self, adj_matrix, x, masked_x, get_details=False):
        # Spatial component
        z, sub_res, attn = self.spatial_gather(adj_matrix, x, masked_x)

        # Max pooling with the input
        # x_recon = torch.maximum(z, masked_x)
        x_recon = z

        if get_details:
            return x_recon, z, sub_res, attn
        else:
            return x_recon

    def fit(self, dataset: _SpaceFormerDataset, masking_rate=0.0, device:str='cuda', optim_type:str='adam', lr:float=1e-4, weight_decay:float=0., 
            warmup=8, max_epoch:int=100, stop, loss_fn:str='mse', log_dir:str='log/', verbose=False):
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
        self.spatial_gather.verbose = verbose

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

    def transform(self, data):
        raise NotImplemented("")
        self.eval()

    def get_bias(self) -> np.array:
        b = self.spatial_gather.bias.bias.detach().cpu().numpy()
        return b.T

    def get_ego_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_ego.weight.detach().cpu().numpy()
        return qk, v.T

    def get_local_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        q = self.spatial_gather.q_local.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_local.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_local.weight.detach().cpu().numpy()
        return q, k, v.T
        
    def get_global_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention matrix
        """
        q = self.spatial_gather.q_global.weight.detach().cpu().numpy()
        k = self.spatial_gather.k_global.weight.detach().cpu().numpy()
        v = self.spatial_gather.v_global.weight.detach().cpu().numpy()
        return q, k, v.T
       
    def get_local_q_score(self) -> np.array:
        pass