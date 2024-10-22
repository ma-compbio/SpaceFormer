from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
from tqdm import tqdm
import scipy as sp
from typing import Union
import scipy.sparse
import warnings

class SteamboatDataset(Dataset):
    def __init__(self, data_list, sparse_graph):
        super().__init__()
        self.data = data_list
        self.sparse_graph = sparse_graph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample['X'], sample['adj']
    

def prep_adatas(adatas: list[sc.AnnData], n_neighs: int = 8, log_norm=True) -> list[sc.AnnData]:
    with warnings.catch_warnings(action="ignore"):
        warnings.simplefilter("ignore")
        for i in tqdm(range(len(adatas))):
            adata = adatas[i]
            if log_norm:
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
            # sc.pp.scale(adata, zero_center=False)
            sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
    return adatas


def make_dataset(adatas: list[sc.AnnData], sparse_graph=True, mask_var=None) -> Dataset:
    """Create a PyTorch Dataset from a list of adata
    The input data should be a list of AnnData that contains 1. raw counts or normalized counts
    :param adatas: A list of `SCANPY AnnData`
    :param sparse_graph: Use adjacency list. 
    :param mask_var: Column in `var` to select variables. Default: `obs.highly_variable` if available, otherwise no filtering. Specify `False` to use all genes.
    :return: A `torch.Dataset` including all data.
    """
    # Sanity checks
    if mask_var is None:
        if 'highly_variable' in adatas[0].var.columns:
            mask_var = 'highly_variable'
        else:
            mask_var = False

    if mask_var:
        for i in range(len(adatas)):
            assert mask_var in adatas[i].var.columns, f"Not all adatas have {mask_var} in var"
        temp = adatas[0].var[mask_var]
        for i in range(1, len(adatas)):
            assert (adatas[i].var[mask_var] == temp).all(), f"Not all adatas have {mask_var} in var"

    datasets = []
    unequal_nbs = []

    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        data_dict = {}

        # Gather expression profile
        X = adata.X
        if mask_var:
            X = X[:, adata.var[mask_var]]
        if isinstance(adata.X, sp.sparse.spmatrix):
            data_dict['X'] = torch.from_numpy(X.astype(np.float32).toarray())
        else:
            data_dict['X'] = torch.from_numpy(X.astype(np.float32))
        
        # Gather spatial graph
        if sparse_graph:
            have_equal_deg = True
            v, u = adata.obsp['spatial_connectivities'].nonzero()
            k0 = u.shape[0] / adata.shape[0]
            k = int(np.round(k0))

            order = np.argsort(v)
            u = u[order]
            v = v[order]

            if np.abs(k - k0) < 1e-6 and (v.reshape([-1, k]) == np.arange(adata.shape[0])[:, None]).all():
                data_dict['adj'] = torch.from_numpy(np.vstack([u, v]))
            else:
                ks = np.array(adata.obsp['spatial_connectivities'].sum(axis=0)).squeeze().astype(int)
                max_k = int(ks.max())
                unequal_nbs.append(i)
                aligned_u = np.zeros((adata.shape[0], max_k), dtype=int)
                aligned_v = np.zeros((adata.shape[0], max_k), dtype=int)
                align_mask = np.zeros((adata.shape[0], max_k), dtype=int)

                pt = 0
                for i in range(adata.shape[0]):
                    pt2 = pt + ks[i]
                    aligned_u[i, :] = v[pt]
                    aligned_v[i, :] = v[pt]

                    aligned_u[i, :ks[i]] = u[pt:pt2]
                    assert (v[pt:pt2] == i).all()
                    align_mask[i, :ks[i]] = 1
                    pt = pt2
                        
                data_dict['adj'] = torch.from_numpy(np.vstack([aligned_u.flatten(), 
                                                               aligned_v.flatten(), 
                                                               align_mask.flatten()]))

        else:
            data_dict['adj'] = torch.from_numpy((adata.obsp['spatial_connectivities'] == 1).toarray())

        datasets.append(data_dict)
        
    if unequal_nbs:
        print("Not all cells in the following samples have the same number of neighbors:")
        print(*unequal_nbs, sep=', ', end='.\n')
        print("Steamboat can handle this. You can safely ignore this warning if this is expected.")

    return SteamboatDataset(datasets, sparse_graph)