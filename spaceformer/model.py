import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .utils import _sce_loss, _get_logger
from tensorboardX import SummaryWriter
from functools import partial
from .dataset import _SpaceFormerDataset
import os
from typing import Literal


class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        return self.elu(self._weight) + 1

    def forward(self, x):
        return x @ self.weight.T

class NonNegLinear2(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 3)
        self.shift = torch.nn.Parameter(torch.randn(1))
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        return self.elu(self._weight) + 1
    
    @property
    def weight2(self):
        return self.weight + self.shift

    def forward(self, x):
        return x @ self.weight.T
    
    def rofward(self, x):
        return x @ self.weight2

class NonNegBias(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self._bias = torch.nn.Parameter(torch.zeros(1, d))
        self.elu = nn.ELU()

    @property
    def bias(self):
        return self.elu(self._bias) + 1

    def forward(self, x):
        return x + self.bias


class BilinearAttention(nn.Module):
    def __init__(self, d_in, d_ego, d_local, d_global, d_out=None):
        super(BilinearAttention, self).__init__()
        if d_out is None:
            d_out = d_in
        self.d_in = d_in
        self.d_ego = d_ego
        self.d_local = d_local
        self.d_global = d_global

        self.bias = NonNegBias(d_out)

        # self.qk_ego = nn.Linear(d_in, d_ego, bias=False)
        self.v_ego = NonNegLinear2(d_ego, d_out, bias=False)

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

        # ego_scores = self.qk_ego(masked_x) / x.shape[1] 
        ego_scores = self.v_ego.rofward(masked_x) / x.shape[1] 
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
        self.spatial_gather = BilinearAttention(d_in, d_ego, d_local, d_global, d_in)

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
            return x_recon, sub_res, attn
        else:
            return x_recon

    def fit(self, dataset: _SpaceFormerDataset, masking_rate=0.0, device:str='cuda', optim_type:str='adam', *, lr:float=1e-4, weight_decay:float=0., 
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
            logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}")
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
        # qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
        qk = self.spatial_gather.v_ego.weight2.detach().cpu().numpy()
        v = self.spatial_gather.v_ego.weight.detach().cpu().numpy()
        return qk.T, v.T

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
       
    def score_cells(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        res = {}
        qk_ego, v_ego = self.get_ego_transform()
        for i in range(qk_ego.shape[0]):
            res[f'u_ego_{i}'] = x @ qk_ego[i, :]
        q_local, k_local, v_local = self.get_local_transform()
        for i in range(q_local.shape[0]):
            res[f'q_local_{i}'] = x @ q_local[i, :]
            res[f'k_local_{i}'] = x @ k_local[i, :]
        if self.spatial_gather.d_global > 0:
            q_global, k_global, v_global = self.get_global_transform()
        for i in range(q_global.shape[0]):
            res[f'q_global_{i}'] = x @ q_global[i, :]
        return res

    def get_top_features(self, top_k=5):
        res = {}
        features = np.array(self.features)
        qk_ego, v_ego = self.get_ego_transform()
        for i in range(qk_ego.shape[0]):
            res[f'U_ego_{i}'] = features[np.argsort(-qk_ego[i, :])[:top_k]].tolist()
            # res[f'V_ego_{i}'] = features[np.argsort(-v_ego[i, :])[:top_k]].tolist()
        q_local, k_local, v_local = self.get_local_transform()
        for i in range(q_local.shape[0]):
            res[f'Q_local_{i}'] = features[np.argsort(-q_local[i, :])[:top_k]].tolist()
            res[f'K_local_{i}'] = features[np.argsort(-k_local[i, :])[:top_k]].tolist()
            res[f'V_local_{i}'] = features[np.argsort(-v_local[i, :])[:top_k]].tolist()
        if self.spatial_gather.d_global > 0:
            q_global, k_global, v_global = self.get_global_transform()
            for i in range(q_global.shape[0]):
                res[f'Q_global_{i}'] = features[np.argsort(-q_global[i, :])[:top_k]].tolist()
                res[f'K_global_{i}'] = features[np.argsort(-k_global[i, :])[:top_k]].tolist()
                res[f'V_global_{i}'] = features[np.argsort(-v_global[i, :])[:top_k]].tolist()
        return res
    
    