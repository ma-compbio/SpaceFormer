import os
import torch
import pandas as pd
import scanpy as sc
import scipy as sp
import squidpy as sq
import numpy as np
from utils.const import BRAIN_FILENAMES, BRAIN_LABEL_DICT, HUMAN_4000_FILENAMES, MOUSE_FILENAMES, GSEA_LABEL_DICT

def load_brain_data(args):
    data_list = []
    if args.dataset == 'brain':
        for i in range(len(BRAIN_FILENAMES)):
            id = BRAIN_FILENAMES[i]
            obs = pd.read_csv(os.path.join(args.data_dir, 'brain', f"{id}.features.csv"), index_col=0)
            var = pd.read_csv(os.path.join(args.data_dir, 'brain', f"{id}.genes.csv"), index_col=0)
            data = pd.read_csv(os.path.join(args.data_dir, 'brain', f"{id}.matrix.csv"))
            adata = sc.AnnData(X=sp.sparse.csr_matrix((data['val'], (data['col'] - 1, data['row'] - 1)), 
                                           shape=(len(obs), len(var))),
                   obs=obs,
                   var=var)
            data_dict = {}

            data_dict['raw_X'] = torch.from_numpy(adata.X.toarray())

            # X
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(adata, n_top_genes=64)
            sc.pp.scale(adata, zero_center=False)
            highly_variable_genes = adata.X[:, adata.var['highly_variable'].to_numpy()].toarray()
            data_dict['highly_variable_genes'] = torch.from_numpy(highly_variable_genes)

            X = adata.X.toarray()
            data_dict['X'] = torch.from_numpy(X)

            # label
            labels = torch.tensor([BRAIN_LABEL_DICT[l] for l in adata.obs['cluster_L2']])
            data_dict['labels'] = labels

            if args.spatial:
                adata.obsm['spatial'] = adata.obs[['adjusted.x', 'adjusted.y']].to_numpy()
                data_dict['coordinates'] = torch.from_numpy(adata.obsm['spatial'])
                sq.gr.spatial_neighbors(adata, n_neighs=args.n_neighs)
                csr_matrix = adata.obsp['spatial_connectivities']
                rows, cols = csr_matrix.nonzero()
                
                values = np.ones_like(rows)
                edge_mask = sp.sparse.csr_matrix((values, (cols, rows)), shape=(adata.X.shape[0], adata.X.shape[0])).toarray()
                real_edge_mask = torch.from_numpy(edge_mask == 0)
                fake_edge_mask = torch.from_numpy(edge_mask == 1)

                data_dict['real_edge_mask'] = real_edge_mask
                data_dict['fake_edge_mask'] = fake_edge_mask

            data_list.append(data_dict)
    elif args.dataset == 'GSEA':
        raw_adata = sc.read_h5ad(os.path.join(args.data_dir, 'GSEA_merfish.h5ad')).raw.to_adata()
        sc.pp.filter_genes(raw_adata, min_cells=3)
        sc.pp.filter_cells(raw_adata, min_genes=3)
        for i in range(4, 12):
            donor_id = 'MsBrainAgingSpatialDonor_' + str(i)
            for j in range(3):
                data_dict = {}    
                print(f'Processing Data Sample {j + (i - 4) * 3}')
                slice_id = str(j)
                adata = raw_adata[(raw_adata.obs['donor_id'] == donor_id) & (raw_adata.obs['slice'] == slice_id)].copy()
                data_dict['raw_X'] = torch.from_numpy(adata.X)

                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=32)
                sc.pp.scale(adata, zero_center=False)
                highly_variable_genes = adata.X[:, adata.var['highly_variable'].to_numpy()]
                data_dict['highly_variable_genes'] = torch.from_numpy(highly_variable_genes)

                X = adata.X
                data_dict['X'] = torch.from_numpy(X)

                # label
                labels = torch.tensor([GSEA_LABEL_DICT[l] for l in adata.obs['clust_annot']])
                data_dict['labels'] = labels

                if args.spatial:
                    data_dict['coordinates'] = torch.from_numpy(adata.obsm['spatial'])
                    sq.gr.spatial_neighbors(adata, n_neighs=args.n_neighs)
                    csr_matrix = adata.obsp['spatial_connectivities']
                    rows, cols = csr_matrix.nonzero()
                    
                    values = np.ones_like(rows)
                    edge_mask = sp.sparse.csr_matrix((values, (cols, rows)), shape=(adata.X.shape[0], adata.X.shape[0])).toarray()
                    real_edge_mask = torch.from_numpy(edge_mask == 0)
                    fake_edge_mask = torch.from_numpy(edge_mask == 1)

                    data_dict['real_edge_mask'] = real_edge_mask
                    data_dict['fake_edge_mask'] = fake_edge_mask

                data_list.append(data_dict)
    elif args.dataset == 'GSEA_imputed':
        raw_adata = sc.read_h5ad(os.path.join(args.data_dir, 'GSEA_imputed.h5ad')).raw.to_adata()
        sc.pp.filter_genes(raw_adata, min_cells=3)
        sc.pp.filter_cells(raw_adata, min_genes=3)
        print(raw_adata.X.shape)
        for i in range(4, 12):
            donor_id = 'MsBrainAgingSpatialDonor_' + str(i)
            for j in range(3):
                data_dict = {}    
                print(f'Processing Data Sample {j + (i - 4) * 3}')
                slice_id = str(j)
                adata = raw_adata[(raw_adata.obs['donor_id'] == donor_id) & (raw_adata.obs['slice'] == slice_id)].copy()
                data_dict['raw_X'] = torch.from_numpy(adata.X.toarray())

                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=64)
                sc.pp.scale(adata, zero_center=False)
                highly_variable_genes = adata.X.toarray()[:, adata.var['highly_variable'].to_numpy()]
                data_dict['highly_variable_genes'] = torch.from_numpy(highly_variable_genes)

                X = adata.X.toarray()
                data_dict['X'] = torch.from_numpy(X)

                # label
                labels = torch.tensor([GSEA_LABEL_DICT[l] for l in adata.obs['clust_annot']])
                data_dict['labels'] = labels

                if args.spatial:
                    data_dict['coordinates'] = torch.from_numpy(adata.obsm['spatial'])
                    sq.gr.spatial_neighbors(adata, n_neighs=args.n_neighs)
                    csr_matrix = adata.obsp['spatial_connectivities']
                    rows, cols = csr_matrix.nonzero()
                    
                    values = np.ones_like(rows)
                    edge_mask = sp.sparse.csr_matrix((values, (cols, rows)), shape=(adata.X.shape[0], adata.X.shape[0])).toarray()
                    real_edge_mask = torch.from_numpy(edge_mask == 0)
                    fake_edge_mask = torch.from_numpy(edge_mask == 1)

                    data_dict['real_edge_mask'] = real_edge_mask
                    data_dict['fake_edge_mask'] = fake_edge_mask

                data_list.append(data_dict)
    
    return data_list


