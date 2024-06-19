import argparse
import random
import torch
import os
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import optim as optim


def _get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger



def _sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def _create_loss(loss_fn, alpha_l=3):
    if loss_fn == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    elif loss_fn == 'sce':
        loss = partial(sce_loss, alpha=alpha_l)
    elif loss_fn == "mse":
        loss = nn.MSELoss()    
    return loss


def _create_optimizer(optim_type, model, lr, weight_decay):
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    if optim_type == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif optim_type == 'SGD':
        optimizer = optim.SGD(parameters, **opt_args)

    return optimizer


def set_random_seed(seed: int) -> None:
    """Reset seed for Numpy and PyTorch

    :param seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def _build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--dataset", type=str, default='brain')
    parser.add_argument("--data_dir", type=str, default='/home/guanghaw/Single-Cell-Transformer/transformer/data')
    parser.add_argument("--train_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--valid_ids", type=int, nargs="+", default=[0])
    parser.add_argument("--test_ids", type=int, nargs="+", default=[])
    parser.add_argument("--drop_rates", type=float, nargs="+", default=[0])
    parser.add_argument("--n_neighs", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.02) 
    parser.add_argument("--gene_thres", type=int, default=250)

    parser.add_argument("--model", type=str, default='MLP')
    parser.add_argument("--input_dim", type=int, default=3999)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--ffn_dim", type=int, default=8192)
    parser.add_argument("--output_dim", type=int, default=19)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cell_mask_rate", type=float, default=0.5)
    parser.add_argument("--gene_mask_rate", type=float, default=0.5)
    parser.add_argument("--imputation_rate", type=float, default=0.2)

    parser.add_argument("--optim_type", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--loss_fn", type=str, default="crossentropy")
    parser.add_argument("--max_grad_norm", type=float, default=8.0)

    parser.add_argument("--spatial", action='store_true', default=False)

    args = parser.parse_args()
    return args
