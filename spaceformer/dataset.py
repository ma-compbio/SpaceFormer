from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
from tqdm import tqdm
import scipy as sp

class SpaceFormerDataset(Dataset):
    def __init__(self, data_list, sparse_graph):
        super().__init__()
        self.data = data_list
        self.sparse_graph = sparse_graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['X'], sample['adj']
    

def prep_adatas(adatas: list[sc.AnnData], n_neighs: int = 8) -> list[sc.AnnData]:
    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, zero_center=False)
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    return adatas

def make_dataset(adatas: list[sc.AnnData], sparse_graph=False) -> Dataset:
    """Create a PyTorch Dataset from a list of adata
    The input data should be a list of AnnData that contains 1. raw counts or normalized counts
    :param adatas: A list of `SCANPY AnnData`
    :param sparse_graph: Use adjacency list. 

    :return: A `torch.Dataset` including all data.
    """
    datasets = []

    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        data_dict = {}

        # Gather expression profile
        data_dict['X'] = torch.from_numpy(adata.X.astype(np.float32))
        
        # Gather spatial graph
        if sparse_graph:
            u, v = adata.obsp['spatial_connectivities'].nonzero()
            k0 = u.shape[0] / adata.shape[0]
            k = int(np.round(k0))

            if np.abs(k - k0) > 1e-6:
                raise ValueError("Sparse graph only supports k-NN graph where each node has exactly k neighbors.")
            
            order = np.argsort(v)
            u = u[order]
            v = v[order]
            v = v.reshape([-1, k])
            u = u.reshape([-1, k])
            if not (v == np.arange(adata.shape[0])[:, None]).all():
                raise ValueError("Sparse graph only supports k-NN graph where each node has exactly k neighbors.")
            
            data_dict['adj'] = torch.from_numpy(u)

        else:
            data_dict['adj'] = torch.from_numpy((adata.obsp['spatial_connectivities'] == 1).toarray())

        datasets.append(data_dict)
        
    return SpaceFormerDataset(datasets, sparse_graph)