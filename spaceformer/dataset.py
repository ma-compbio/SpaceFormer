from torch.utils.data import Dataset
import numpy as np
import torch
import squidpy as sq
import scanpy as sc
from tqdm import tqdm
import scipy as sp

class SpaceFormerDataset(Dataset):
    def __init__(self, data_list, spatial):
        self.data = data_list
        self.spatial = spatial

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        X = sample['X']
        labels = sample['labels']
        raw_X = sample['raw_X']
        highly_variable_genes = sample['highly_variable_genes']
        if self.spatial:
            coordinates = sample['coordinates']
            real_edge_mask = sample['real_edge_mask']
            fake_edge_mask = sample['fake_edge_mask']
            return X, labels, real_edge_mask, fake_edge_mask, coordinates, highly_variable_genes, raw_X
        else:
            return X, labels, raw_X


def make_dataset(adatas, label, n_neighs=8, use_hvg=True):
    all_labels = set()
    for i in range(len(adatas)):
        adata = adatas[i]
        if label not in adata.obs:
            raise RuntimeError(f"Cannot find '{label}' in adata.obs.")
        all_labels |= set(adata.obs[label])

    LABEL_DICT = {k: i for i, k in enumerate(all_labels)}

    datasets = []
    for i in tqdm(range(len(adatas))):
        adata = adatas[i]
        data_dict = {}
        data_dict['raw_X'] = torch.from_numpy(adata.X.astype(np.float32))
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=64)
        sc.pp.scale(adata, zero_center=False)
        
        # Gather expression profile
        data_dict['X'] = torch.from_numpy(adata.X.astype(np.float32))
        
        # Gather HVGs
        if use_hvg:
            if 'highly_variable' not in adata:
                raise RuntimeError("Cannot find 'highly_variable' in adata.var.")
        else:
            adata.var['highly_variable'] = True
        highly_variable_genes = adata.X[:, adata.var['highly_variable'].to_numpy()]
        data_dict['highly_variable_genes'] = torch.from_numpy(highly_variable_genes)
        data_dict['coordinates'] = torch.from_numpy(adata.obsm['spatial'])

        # Gather labels
        if label not in adata.obs:
            raise RuntimeError(f"Cannot find '{label}' in adata.obs.")
        labels = torch.tensor([LABEL_DICT[l] for l in adata.obs[label]])
        data_dict['labels'] = labels

        # Gather spatial graph
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs)
        csr_matrix = adata.obsp['spatial_connectivities']
        rows, cols = csr_matrix.nonzero()
        values = np.ones_like(rows)
        edge_mask = sp.sparse.csr_matrix((values, (cols, rows)), shape=(adata.X.shape[0], adata.X.shape[0])).toarray()
        real_edge_mask = torch.from_numpy(edge_mask == 0)
        fake_edge_mask = torch.from_numpy(edge_mask == 1)

        data_dict['real_edge_mask'] = real_edge_mask
        data_dict['fake_edge_mask'] = fake_edge_mask
        
        datasets.append(data_dict)
        
    return SpaceFormerDataset(datasets, True), LABEL_DICT


class BrainDataset(Dataset):
    def __init__(self, data_list, spatial):
        self.data = data_list
        self.spatial = spatial

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        X = sample['X']
        labels = sample['labels']
        raw_X = sample['raw_X']
        highly_variable_genes = sample['highly_variable_genes']
        if self.spatial:
            coordinates = sample['coordinates']
            real_edge_mask = sample['real_edge_mask']
            fake_edge_mask = sample['fake_edge_mask']
            return X, labels, real_edge_mask, fake_edge_mask, coordinates, highly_variable_genes, raw_X
        else:
            return X, labels, raw_X