import torch
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .utils import _sce_loss, _get_logger
from tensorboardX import SummaryWriter
from functools import partial
from .dataset import SteamboatDataset
import os
from typing import Literal


class NonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) / 10 - 2)
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        return self.elu(self._weight) + 1

    def forward(self, x):
        return x @ self.weight.T
    
class NormNonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) / 10 - 2)
        self.sigmoid = nn.Sigmoid()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        temp =  self.sigmoid(self._weight)
        return temp / temp.sum()

    def forward(self, x):
        return x @ self.weight.T

class ReverseNonNegLinear:
    def __init__(self, bnnl):
        self.bnnl = bnnl
    
    @property
    def weight(self):
        return self.bnnl.thgiew

    def forward(self, x):
        return self.bnnl.rofward(x)
    
    def __call__(self, x):
        return self.forward(x)

class BidirNonNegLinear(nn.Module):
    def __init__(self, d_in, d_out, bias) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.randn(d_out, d_in) - 2)
        self._shift = torch.nn.Parameter(torch.zeros(1, d_in))
        self.elu = nn.ELU()
        if bias:
            raise NotImplementedError()

    @property
    def weight(self):
        return self.elu(self._weight) + 1

    @property
    def thgiew(self):
        return (self.elu(self._weight) + 1) * (self.elu(self._shift) + 1)

    def forward(self, x):
        return x @ self.weight.T
    
    def rofward(self, x):
        return x @ self.thgiew
    
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

        self.qk_ego = nn.Linear(d_in, d_ego, bias=False)
        self.v_ego = BidirNonNegLinear(d_ego, d_out, bias=False)

        self.q_local = NormNonNegLinear(d_in, d_local, bias=False) # each row of the weight matrix is a metagene (x -> x @ w.T)
        self.k_local = NonNegLinear(d_in, d_local, bias=False) # each row ...
        self.v_local = NonNegLinear(d_local, d_out, bias=False) # each column ..

        self.q_global = NormNonNegLinear(d_in, d_global, bias=False) # each row ..
        self.k_global = NonNegLinear(d_in, d_global, bias=False) # each row ..
        self.v_global = NonNegLinear(d_global, d_out, bias=False) # each column ..

        # penalty / regularlization
        self.cos_sim = nn.CosineSimilarity(dim=1) # each row is a metagene
        self.cell_cos_sim = nn.CosineSimilarity(dim=0) # each col is a list of cells
        # remember some variables during forward
        self.k_local_emb = None
        self.q_local_emb = None

    def local_cell_cos(self):
        temp = self.cell_cos_sim(self.k_local_emb, self.q_local_emb).sum()
        # self.k_local_emb = None
        # self.q_local_emb = None
        return temp

    def local_cos(self):
        return self.cos_sim(self.q_local.weight, self.v_local.weight.T).sum() # response v should be different from receiver q
    
    def global_cos(self):
        return (self.cos_sim(self.q_global.weight, self.v_global.weight.T).sum() + # response v should be different from receiver q
                self.cos_sim(self.q_global.weight, self.k_global.weight).sum())  

    def othorgonality(self, m):
        temp = m @ m.transpose(0, 1)
        eye = torch.eye(m.shape[0]).to(m.get_device())
        return (temp - eye).pow(2.0).sum()
    
    def q_othorgonality(self):
        return self.othorgonality(self.q_local.weight) + self.othorgonality(self.q_global.weight) + self.othorgonality(self.v_ego.weight2.transpose(0, 1))

    def score_local(self, adj_matrix, x, masked_x=None):
        if masked_x is None:
            masked_x = x
        # q_local = self.v_local.rofward(masked_x) # n * g -> n * d
        q_local = self.q_local(masked_x)
        k_local = self.k_local(x) # n * g -> n * d
        # (d * n * 1) x (d * 1 * n) -> d * n * n
        local_scores = q_local.transpose(-1, -2)[:, :, None] * k_local.transpose(-1, -2)[:, None, :] / x.shape[1] / x.shape[1]
        local_scores.masked_fill_(~adj_matrix, -1e9)
        local_scores = local_scores.transpose(-2, -3) # n * d * n
        self.k_local_emb = k_local
        self.q_local_emb = q_local
        return local_scores

    def score_local_sparse(self, adj_list, x, masked_x=None):
        if masked_x is None:
            masked_x = x
        # q_local = self.v_local.rofward(masked_x) # n * g -> n * d
        q_local = self.q_local(masked_x)
        k_local = self.k_local(x) # n * g -> n * d
        self.k_local_emb = k_local
        self.q_local_emb = q_local
        q_local = q_local[adj_list[1, :], :] # n * g --v-> kn * d
        k_local = k_local[adj_list[0, :], :] # n * g --u-> kn * d
        local_scores = (q_local * k_local) # nk * d
        if adj_list.shape[0] == 3: # masked for unequal neighbors
            local_scores.masked_fill_((adj_list[2, :] == 0).reshape([-1, 1]), 0.)
        n = local_scores.shape[0] // x.shape[0]
        local_scores = local_scores.reshape([x.shape[0], n, self.d_local]) / x.shape[1] / x.shape[1] # n * k * d 
        local_scores = local_scores.transpose(-1, -2)
        return local_scores

    def score_global(self, x, masked_x=None, x_bar=None):
        if masked_x is None:
            masked_x = x
        if x_bar is None:
            x_bar = torch.mean(x, axis=0, keepdim=True)
        # q_global = self.v_global.rofward(masked_x) # n * g -> n * d
        q_global = self.q_global(masked_x)
        k_global = self.k_global(x_bar)    # m * g -> m * d
        global_scores = q_global.transpose(-1, -2)[:, :, None] * k_global.transpose(-1, -2)[:, None, :] / x.shape[1] / x.shape[1]
        global_scores = global_scores.transpose(-2, -3) # n * d * m
        self.k_global_emb = k_global
        self.q_global_emb = q_global
        return global_scores

    def forward(self, adj_matrix, x, masked_x=None, x_bar=None, sparse_graph=True, get_details=False):
        if x_bar is None:
            x_bar = torch.mean(x, axis=0, keepdim=True) # in which case, m := 1
            # Note: x_bar can have multiple pseudobulks.

        if masked_x is None:
            masked_x = x

        ego_emb = self.qk_ego(masked_x)
        ego_scores = (ego_emb / x.shape[1]) ** 2
        # ego_scores = (self.v_ego.rofward(masked_x) / x.shape[1]) ** 2

        if sparse_graph:
            local_scores = self.score_local_sparse(adj_matrix, x, masked_x)
        else:
            local_scores = self.score_local(adj_matrix, x, masked_x)

        global_scores = self.score_global(x, masked_x, x_bar)

        max_list = []
        if self.d_ego > 0:
            max_ego_score, _ = torch.max(ego_scores, dim=1, keepdim=True)
            # max_list.append(max_ego_score)
        if self.d_local > 0:
            max_local_score, _ = torch.max(local_scores.reshape((local_scores.shape[0], -1)), dim=1, keepdim=True)
            max_list.append(max_local_score)
        if self.d_global > 0:
            max_global_score, _ = torch.max(global_scores.reshape((global_scores.shape[0], -1)), dim=1, keepdim=True)
            max_list.append(max_global_score)
        
        max_score, _ = torch.max(torch.cat(max_list, dim=1), dim=1, keepdim=True)

        sum_exp_score = 1e-3
        if self.d_ego > 0:
            exp_ego_scores = ego_scores # torch.exp(ego_scores - max_score)
            sum_exp_score += torch.sum(exp_ego_scores, dim=-1, keepdim=True)
        if self.d_local > 0:
            exp_local_scores = local_scores # torch.exp(local_scores - max_score[:, :, None])
            exp_local_scores = torch.sum(exp_local_scores, dim=-1)
            sum_exp_score += torch.sum(exp_local_scores, dim=-1, keepdim=True)
        if self.d_global > 0:
            exp_global_scores = global_scores # torch.exp(global_scores - max_score[:, :, None])
            exp_global_scores = torch.sum(exp_global_scores, dim=-1)
            sum_exp_score += torch.sum(exp_global_scores, dim=-1, keepdim=True)

        # print(local_scores.shape, exp_local_scores.shape, sum_exp_score.shape)
        # print(global_scores.shape, exp_global_scores.shape, sum_exp_score.shape)
        res = 0.
        if self.d_ego > 0:
            ego_attn = exp_ego_scores / sum_exp_score
            ego_res = self.v_ego(ego_attn)
            res += ego_res
        else:
            ego_scores = None
            ego_attn = None
            ego_res = None

        if self.d_local > 0:
            if get_details:
                local_norm_attn = local_scores / sum_exp_score[:, :, None]
            local_attn = exp_local_scores / sum_exp_score
            local_res = self.v_local(local_attn)
            res += local_res
        else:
            local_norm_attn = None
            local_attn = None
            local_res = None

        if self.d_global > 0:
            if get_details:
                global_norm_attn = global_scores / sum_exp_score[:, :, None]
            global_attn = exp_global_scores / sum_exp_score
            global_res = self.v_global(global_attn)
            res += global_res
        else:
            global_norm_attn = None
            global_attn = None
            global_res = None

        res = self.bias(res)
        if get_details:
            return res, {
                'embq': (ego_emb, self.q_local_emb, self.q_global_emb),
                'embk': (None, self.k_local_emb, self.k_global_emb),
                'attnp': (ego_attn, local_norm_attn, global_norm_attn),
                'attnm': (ego_attn, local_attn, global_attn),
                'trivia': (ego_res, local_res, global_res)}
        else:
            return res


    
class Steamboat(nn.Module):
    def __init__(self, features: list[str] | int, d_ego, d_local, d_global):
        """Steamboat Model 

        :param features: input feature (gene) names or the number of features
        :param local_d: 
        :param global_d: 
        """
        super(Steamboat, self).__init__()

        if isinstance(features, list):
            self.features = features
        else:
            self.features = [f'feature_{i}' for i in range(features)]

        # Three way attention
        d_in = len(self.features)
        self.spatial_gather = BilinearAttention(d_in, d_ego, d_local, d_global, d_in)

    def masking(self, x, mask_rate, masking_method):
        out_x = x.clone()
        # All cells are masked
        if masking_method == 'full':
            random_mask = torch.rand(x.shape, device=x.get_device()) < mask_rate
            out_x.masked_fill_(random_mask, 0.)
        elif masking_method == 'feature':
            random_mask = torch.rand([1, x.shape[1]], device=x.get_device()) < mask_rate
            out_x.masked_fill_(random_mask, 0.)
        else:
            raise ValueError("Unknown random mask method.")
        return out_x

    def forward(self, adj_matrix, x, masked_x, sparse_graph=True, get_details=False):
        return self.spatial_gather(adj_matrix, x, masked_x, sparse_graph=sparse_graph, get_details=get_details)

    def fit(self, dataset: SteamboatDataset, 
            masking_rate=0.0, masking_method='full', 
            device:str='cuda', 
            *, orthogonal=0., similarity_penalty=0.0, 
            opt=None, opt_args=None, max_epoch:int=100, stop_eps=1e-4, stop_tol=10, 
            loss_fn:str='mse', log_dir:str='log/', report_per=10):
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
        :param log_dir: report per how many epoch. 0 to only report before termination. negative number to never report.

        :return: A `torch.Dataset` including all data.
        """
        self.train()

        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        parameters = self.parameters()

        if opt_args is None:
            opt_args = {}

        if opt is None:
            optimizer = optim.Adam(parameters, **opt_args)
        else:
            optimizer = opt(parameters, **opt_args)
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
        best_loss = np.inf
        for epoch in range(max_epoch):
            total_loss = 0.
            total_penalty = 0.
            for x, adj_matrix in loader:
                x = x.squeeze(0).to(device)
                adj_matrix = adj_matrix.squeeze(0).to(device)
                masked_x = self.masking(x, masking_rate, masking_method)
                x_recon = self(adj_matrix, x, masked_x, sparse_graph=dataset.sparse_graph)
                
                loss = criterion(x, x_recon)
                total_loss += loss.item()

                loss = loss # * x.shape[0] / 10000 # handle size differences among datasets; larger dataset has higher weight

                reg = 0.
                if orthogonal > 0.:
                    reg = self.spatial_gather.q_othorgonality() * orthogonal + reg
                if similarity_penalty > 0.:
                    reg = (self.spatial_gather.local_cos() + self.spatial_gather.local_cell_cos()) * similarity_penalty + reg
                if reg != 0.:
                    total_penalty += reg.item()
                optimizer.zero_grad()
                (loss + reg).backward()
                optimizer.step()

            avg_loss = total_loss / len(loader)
            avg_penalty = total_penalty / len(loader)

            if best_loss - (avg_loss + avg_penalty) < stop_eps:
                cnt += 1
            else:
                cnt = 0
            if report_per >= 0 and cnt >= stop_tol:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}, reg {avg_penalty:.6f}")
                logger.info(f"Stopping criterion met.")
                break
            elif report_per > 0 and (epoch % report_per) == 0:
                logger.info(f"Epoch {epoch + 1}: train_loss {avg_loss:.5f}, reg {avg_penalty:.6f}")
            best_loss = min(best_loss, avg_loss + avg_penalty)

            # writer.add_scalar('Train_Loss', train_loss / len(loader), epoch)
            # writer.add_scalar('Learning_Rate', optimizer.state_dict()["param_groups"][0]["lr"], epoch)
            # scheduler.step()
        else:
            logger.info(f"Maximum iterations reached.")
        self.fit_loss = avg_loss
        self.eval()
        return self

    def transform(self, x, adj_matrix):
        self.eval()
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.BoolTensor(adj_matrix)
            
            return self(adj_matrix, x, get_details=True)


    def get_bias(self) -> np.array:
        b = self.spatial_gather.bias.bias.detach().cpu().numpy()
        return b.T

    def get_ego_transform(self) -> np.array:
        """Get gene attention matrix
        
        :param separate_q_k: If True, return Q and K. Otherwise, return Q.T @ K.

        :return: Gene attention vectors
        """
        # qk = self.spatial_gather.qk_ego.weight.detach().cpu().numpy()
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
    
    def score_local(self, x, adj_matrix):
        with torch.no_grad():
            return self.spatial_gather.score_local(x, adj_matrix).cpu().numpy()
    
    def score_global(self, x, x_bar=None):
        with torch.no_grad():
            return self.spatial_gather.score_global(x, x_bar=x_bar).cpu().numpy()