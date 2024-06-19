from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
from tqdm import tqdm
import scipy as sp

class _SpaceFormerDataset(Dataset):
    def __init__(self, data_list):
        super(_SpaceFormerDataset, self).__init__()
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['X'], sample['adj_matrix']


def prep_adatas(adatas: list[sc.AnnData], n_neighs: int = 8) -> list[sc.AnnData]:
    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    return adatas

def make_dataset(adatas: list[sc.AnnData]) -> Dataset:
    """Create a PyTorch Dataset from a list of adata
    The input data should be a list of AnnData that contains 1. raw counts or normalized counts
    :param adatas: A list of `SCANPY AnnData`
    :param n_neighs: Number of neighbors in the spatial graph
    :param pp: If True, normalize and log transform the data.
    :param inplace: If True, modify the AnnData inplace.

    :return: A `torch.Dataset` including all data.
    """
    datasets = []
    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        data_dict = {}

        # Gather expression profile
        data_dict['X'] = torch.from_numpy(adata.X.astype(np.float32))
        
        # Gather spatial graph
        data_dict['adj_matrix'] = torch.from_numpy((adata.obsp['spatial_connectivities'] == 1).toarray())

        datasets.append(data_dict)
        
    return _SpaceFormerDataset(datasets)