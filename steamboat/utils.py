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
    if (logger.hasHandlers()):
        logger.handlers.clear()
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
