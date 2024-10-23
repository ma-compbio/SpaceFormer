import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .model import Steamboat
from typing import Literal
from torch import nn
import scanpy as sc
import numpy as np
import torch
from .dataset import SteamboatDataset
import scipy as sp
from tqdm import tqdm

palettes = {
    'ncr10': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'npg3': ['#952522', '#0a4b84', '#98752b']
}

def rank(x, axis=1):
    return np.argsort(np.argsort(x, axis=axis), axis=axis)

def plot_transforms(model: Steamboat, top: int = 3, reorder: bool = False, 
                    figsize: str | tuple[float, float] = 'auto', 
                    qkv_colors: list[str] = palettes['npg3'],
                    vmin: float = 0., vmax: float = 1.):
    """Plot all metagenes

    :param model: Steamboat model
    :param top: Number of top genes per metagene to plot, defaults to 3
    :param reorder: Reorder the genes by metagene, or keep the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    :param qkv_colors: Colors for the bar plot showing the magnitude of each metagene before normalization, defaults to palettes['npg3']
    :param vmin: minimum value in the color bar, defaults to 0.
    :param vmax: maximum value in the color bar, defaults to 1.
    """
    assert len(qkv_colors) == 3, f"Expect a color palette with at 3 colors, get {len(qkv_colors)}."
    d_ego : int = model.spatial_gather.d_ego
    d_loc : int = model.spatial_gather.d_local
    d_glb : int = model.spatial_gather.d_global
    d : int = d_ego + d_loc + d_glb

    qk_ego, v_ego = model.get_ego_transform()
    q_local, k_local, v_local = model.get_local_transform()
    q_global, k_global, v_global = model.get_global_transform()

    if top > 0:
        if reorder:
            rank_v_ego = np.argsort(-v_ego, axis=1)[:, :top]
            rank_q_local = np.argsort(-np.abs(q_local), axis=1)[:, :top]
            rank_k_local = np.argsort(-k_local, axis=1)[:, :top]
            rank_v_local = np.argsort(-v_local, axis=1)[:, :top]
            rank_q_global = np.argsort(-np.abs(q_global), axis=1)[:, :top]
            rank_k_global = np.argsort(-k_global, axis=1)[:, :top]
            rank_v_global = np.argsort(-v_global, axis=1)[:, :top]
            feature_mask = {}
            for i in rank_v_ego:
                for j in i:
                    feature_mask[j] = None
            for i in range(d_loc):
                for j in rank_k_local[i, :]:
                    feature_mask[j] = None
                for j in rank_q_local[i, :]:
                    feature_mask[j] = None
                for j in rank_v_local[i, :]:
                    feature_mask[j] = None
            for i in range(d_glb):
                for j in rank_k_global[i, :]:
                    feature_mask[j] = None
                for j in rank_q_global[i, :]:
                    feature_mask[j] = None
                for j in rank_v_global[i, :]:
                    feature_mask[j] = None
            feature_mask = list(feature_mask.keys())
        else:
            rank_v_ego = rank(v_ego)
            rank_q_local = rank(np.abs(q_local))
            rank_k_local = rank(k_local)
            rank_v_local = rank(v_local)
            rank_q_global = rank(np.abs(q_global))
            rank_k_global = rank(k_global)
            rank_v_global = rank(v_global)
            max_rank = np.max(np.vstack([rank_v_ego, 
                                        rank_q_local, 
                                        rank_k_local, 
                                        rank_v_local, 
                                        rank_q_global, 
                                        rank_k_global, 
                                        rank_v_global]), axis=0)
            feature_mask = (max_rank > (max_rank.max() - 3))
            
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (d_ego * 0.36 + (d_loc + d_glb) * 0.49 + 2 + .5, len(chosen_features) * 0.15 + .25 + .75)
    # print(figsize)
    fig, axes = plt.subplots(2, d + 1, sharey='row', sharex='col',
                                          figsize=figsize, 
                                          height_ratios=(.75, len(chosen_features) * .15 + .25),
                                          width_ratios=[2] * d_ego + [3] * (d_loc + d_glb) + [.5])
    plot_axes = axes[1]
    bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': vmax, 'vmin': vmin}

    # Local
    #
    for i in range(0, d_loc + d_glb + d_ego):
        title = ''
        if i < d_ego:
            what = f'{i}'
            if i == (d_ego - 1) // 2:
                if d_ego % 2 == 0:
                    title += '          '
                title += 'Ego'
            labels = ('u', 'v')
            to_plot = np.vstack((qk_ego[i, feature_mask],
                                 v_ego[i, feature_mask])).T
            color = qkv_colors[1:]
        elif i < d_loc + d_ego:
            j = i - d_ego
            what = f'{j}'
            if i == (d_loc - 1) // 2 + d_ego:
                if d_loc % 2 == 0:
                    title += '          '
                title += 'Local'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_local[j, feature_mask],
                                 q_local[j, feature_mask],
                                 v_local[j, feature_mask])).T
            color = qkv_colors
        else:
            j = i - d_ego - d_loc
            what = f'{j}'
            if i == (d_glb - 1) // 2 + d_ego + d_loc:
                if d_glb % 2 == 0:
                    title += '          '
                title += 'Global'
            labels = ('k', 'q', 'v')
            to_plot = np.vstack((k_global[j, feature_mask],
                                 q_global[j, feature_mask],
                                 v_global[j, feature_mask])).T
            color = qkv_colors
        
        true_vmax = to_plot.max(axis=0)
        # print(true_vmax)
        to_plot /= true_vmax
 
        bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax, color=color)
        bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax))
        bar_axes[i].set_yscale('log')
        bar_axes[i].set_title(title, size=10, fontweight='bold')
        if i != 0:
            bar_axes[i].get_yaxis().set_visible(False)
        for pos in ['right', 'top', 'left']:
            if pos == 'left' and i == 0:
                continue
            else:
                bar_axes[i].spines[pos].set_visible(False)
        sns.heatmap(to_plot, xticklabels=labels, ax=plot_axes[i], 
                    **common_params, cbar_ax=cbar_ax)
        plot_axes[i].set_xlabel(f"{what}")
        
    # All text straight up
    for i in range(d_ego + d_loc + d_glb):
        plot_axes[i].set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)

    for i in range(1, d_ego + d_loc + d_glb):
        plot_axes[i].get_yaxis().set_visible(False)

    # Remove duplicate cbars
    bar_axes[-1].set_visible(False)

    plot_axes[-1].get_yaxis().set_visible(False)
    plot_axes[-1].get_xaxis().set_visible(False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plot_axes[-1].spines[pos].set_visible(False)
    # axes[-1].set_visible(False)

    fig.align_xlabels()
    plt.tight_layout()

    # Subplots sep lines [credit: https://stackoverflow.com/a/55465138]
    r = fig.canvas.get_renderer()
    get_bbox = lambda ax: ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
    bboxes = np.array(list(map(get_bbox, axes[1,:-1].flat)), mpl.transforms.Bbox)

    #Get the minimum and maximum extent, get the coordinate half-way between those
    xmax = bboxes[:, 1, 0]
    xmin = bboxes[:, 0, 0]

    # print(xmax, xmin)

    xs = np.c_[xmax[:-1], xmin[1:]].mean(axis=1)

    # for x in xmax:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="red", linewidth=1.)
    #     fig.add_artist(line)

    # for x in xmin:
    #     line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="blue", linewidth=1.)
    #     fig.add_artist(line)

    for i, x in enumerate(xs):
        if i in (d_ego - 1, d_ego + d_loc - 1):
            line = plt.Line2D([x, x],[0, 1], transform=fig.transFigure, color="black", linewidth=.5)
            fig.add_artist(line)


def plot_transform(model, scope: Literal['ego', 'local', 'global'], d, 
                   top: int = 3, reorder: bool = False, 
                   figsize: str | tuple[float, float] = 'auto'):
    """Plot a single set of metagenes (q, k, v)

    :param model: Steamboat model
    :param scope: type of the factor: 'ego', 'local', 'global'
    :param d: Number of the head to be plotted
    :param top: Top genes to plot, defaults to 3
    :param reorder: Reorder the genes or use the orginal ordering, defaults to False
    :param figsize: Size of the figure, defaults to 'auto'
    """
    if scope == 'ego':
        qk_ego, v_ego = model.get_ego_transform()
        assert False, "Not implemented for ego."
    elif scope == 'local':
        q, k, v = model.get_local_transform()
    elif scope == 'global':
        q, k, v = model.get_global_transform()
    else:
        assert False, "scope must be local or global."

    q = q[d, :]
    k = k[d, :]
    v = v[d, :]
    
    if top > 0:
        rank_q = np.argsort(-q)[:top]
        rank_k = np.argsort(-k)[:top]
        rank_v = np.argsort(-v)[:top]
        feature_mask = {}
        for j in rank_k:
            feature_mask[j] = None
        for j in rank_q:
            feature_mask[j] = None
        for j in rank_v:
            feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
        chosen_features = np.array(model.features)[feature_mask]
    else:
        feature_mask = list(range(len(model.features)))
        chosen_features = np.array(model.features)

    if figsize == 'auto':
        figsize = (.65, len(chosen_features) * 0.15 + .75)
    # print(figsize)
    fig, ax = plt.subplots(figsize=figsize)
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds'}

    to_plot = np.vstack((k[feature_mask],
                         q[feature_mask],
                         v[feature_mask])).T
    true_vmax = to_plot.max(axis=0)
    # print(true_vmax)
    to_plot /= true_vmax

    sns.heatmap(to_plot, xticklabels=['k', 'q', 'v'], ax=ax, **common_params)
    
    # ax.set_xticklabels(plot_axes[i].get_xticklabels(), rotation=0)
    # ax.get_yaxis().set_visible(False)

    plt.tight_layout()




def annotate_adatas(adatas: list[sc.AnnData], dataset: SteamboatDataset, model: Steamboat, 
                    device='cuda', get_recon=False):
    """_summary_

    :param adatas: _description_
    :param dataset: _description_
    :param model: _description_
    :param device: _description_, defaults to 'cuda'
    """
    # Safeguards
    assert len(adatas) == len(dataset), "mismatch in lenghths of adatas and dataset"
    for adata, data in zip(adatas, dataset):
        assert adata.shape[0] == data[0].shape[0], f"adata[{i}] has {adata.shape[0]} cells but dataset[{i}] has {data[0].shape[0]}."


    # Assignments
    # d_ego: int = model.spatial_gather.d_ego
    d_local: int = model.spatial_gather.d_local
    # d_global: int = model.spatial_gather.d_global

    for i, (x, adj) in tqdm(enumerate(dataset), total=len(dataset)):
            x = x.to(device)
            adj = adj.to(device)
            with torch.no_grad():
                res, details = model(adj, x, x, sparse_graph=True, get_details=True)
                
            if get_recon:
                adatas[i].obsm['X_recon'] = res.cpu().numpy()

            scopes = ['ego', 'local', 'global']

            # raw emb q
            for which, what in enumerate(scopes):
                if details['embq'][which] is not None:
                    adatas[i].obsm[f'X_{what}_q'] = details['embq'][which].cpu().numpy()
                else:
                    adatas[i].obsm[f'X_{what}_q'] = np.zeros([adatas[i].shape[0], 0])

            # raw emb k
            if details['embk'][1] is not None:
                adatas[i].obsm[f'X_local_k'] = details['embk'][1].cpu().numpy()
            else:
                adatas[i].obsm[f'X_local_k'] = np.zeros([adatas[i].shape[0], 0])

            if details['embk'][2] is not None:
                adatas[i].uns[f'X_global_k'] = details['embk'][2].cpu().numpy()
            else:
                adatas[i].uns[f'X_global_k'] = np.zeros([1, 0])

            # attn (as embedding)
            for which, what in enumerate(scopes):
                if details['attnm'][which] is not None:
                    adatas[i].obsm[f'X_{what}_attn'] = details['attnm'][which].cpu().numpy()
                else:
                    adatas[i].obsm[f'X_{what}_attn'] = np.zeros([adatas[i].shape[0], 0])

            # local attention (as graph)
            for j in range(d_local):
                w = details['attnp'][1].cpu().numpy()[:, j, :].flatten()
                uv = adj.cpu().numpy()
                u = uv[0]
                v = uv[1]
                if uv.shape[0] == 3: # masked for unequal neighbors
                    m = (uv[2] > 0)
                    w, u, v = w[m], u[m], v[m]
                adatas[i].obsp[f'local_attn_{j}'] = sp.sparse.csr_matrix((w, (u, v)), 
                                                                         shape=(adatas[i].shape[0], 
                                                                                adatas[i].shape[0]))


def gather_obsm(adata: sc.AnnData, adatas: list[sc.AnnData]):
    """_summary_

    :param adata: _description_
    :param adatas: _description_
    """
    pass


def neighbors(adata: sc.AnnData,
              use_rep: str = 'X_local_q', 
              key_added: str = 'steamboat_emb',
              metric='cosine', 
              neighbors_kwargs: dict = None):
    """A thin wrapper for scanpy.pp.neighbors for Steamboat functionalities

    :param adata: AnnData object to be processed
    :param use_rep: embedding to be used, 'X_local_q' or 'X_local_attn' (if very noisy data), defaults to 'X_local_q'
    :param key_added: key in obsp to store the resulting similarity graph, defaults to 'steamboat_emb'
    :param metric: metric for similarity graph, defaults to 'cosine'
    :param neighbors_kwargs: Other parameters for scanpy.pp.neighbors if desired, defaults to None
    :return: hands over what scanpy.pp.neighbors returns
    """
    if neighbors_kwargs is None:
        neighbors_kwargs = {}
    return sc.pp.neighbors(adata, use_rep=use_rep, key_added=key_added, metric=metric, **neighbors_kwargs)


def leiden(adata: sc.AnnData, resolution: float = 1., *,
            obsp='steamboat_emb_connectivities',
            key_added='steamboat_clusters',
            leiden_kwargs: dict = None):
    """A thin wrapper for scanpy.tl.leiden to cluster for cell types (for spatial domain segmentation, use `segment`).

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param obsp: obsp key to be used, defaults to 'steamboat_emb_connectivities'
    :param key_added: obs key to be added for resulting clusters, defaults to 'steamboat_clusters'
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: hands over what scanpy.tl.leiden returns
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}
    return sc.tl.leiden(adata, obsp=obsp, key_added=key_added, resolution=resolution, **leiden_kwargs)
    

def segment(adata: sc.AnnData, resolution: float = 1., *,
            embedding_key: str = 'steamboat_emb_connectivities',
            key_added: str = 'steamboat_spatial_domain',
            obsp_summary: str = 'steamboat_summary_connectivities',
            obsp_combined: str = 'steamboat_combined_connectivities', 
            spatial_graph_threshold: float = 0.0,
            leiden_kwargs: dict = None):
    """Spatial domain segmentation using Steamboat embeddings and graphs

    :param adata: AnnData object to be processed
    :param resolution: resolution for Leiden clustering, defaults to 1.
    :param embedding_key: key in obsp for similarity graph (by running `neighbors`), defaults to 'steamboat_emb_connectivities'
    :param key_added: obs key for semgentaiton result, defaults to 'steamboat_spatial_domain'
    :param obsp_summary: obsp key for summary spatial graph, defaults to 'steamboat_summary_connectivities'
    :param obsp_combined: obsp key for combined spatial and similarity graphs, defaults to 'steamboat_combined_connectivities'
    :param spatial_graph_threshold: threshold to include/exclude an edge, a larger number will make the program run faster but potentially less accurate, defaults to 0.0
    :param leiden_kwargs: Other parameters for scanpy.tl.leiden if desired, defaults to None
    :return: _descripthands over what scanpy.tl.leiden returnsion_
    """
    if leiden_kwargs is None:
        leiden_kwargs = {}

    temp = None
    j = 0
    while f'local_attn_{j}' in adata.obsp:
        if temp is None:
            temp = adata.obsp[f'local_attn_{j}'].copy() ** 4
        else:
            temp += adata.obsp[f'local_attn_{j}'] ** 4
        j += 1
    temp = temp.sqrt().sqrt()
    temp.data /= temp.data.max()
    temp.data[temp.data < spatial_graph_threshold] = 0
    temp.eliminate_zeros()
    adata.obsp[obsp_summary] = temp
    adata.obsp[obsp_combined] = adata.obsp[embedding_key] * (adata.obsp[obsp_summary]) + (adata.obsp[obsp_summary])
    adata.obsp[obsp_combined].eliminate_zeros() 
    return sc.tl.leiden(adata, obsp=obsp_combined, key_added=key_added, resolution=resolution, **leiden_kwargs)




def plot_transforms_combined(model, top=3, reorder=False, figsize='auto'):
    d_ego = model.spatial_gather.d_ego
    d_loc = model.spatial_gather.d_local
    d_glb = model.spatial_gather.d_global
    d = d_ego + d_loc + d_glb

    qk_ego, v_ego = model.get_ego_transform()
    q_local, k_local, v_local = model.get_local_transform()
    q_global, k_global, v_global = model.get_global_transform()

    if reorder:
        rank_v_ego = np.argsort(-v_ego, axis=1)[:, :top]
        rank_q_local = np.argsort(-np.abs(q_local), axis=1)[:, :top]
        rank_k_local = np.argsort(-k_local, axis=1)[:, :top]
        rank_v_local = np.argsort(-v_local, axis=1)[:, :top]
        rank_q_global = np.argsort(-np.abs(q_global), axis=1)[:, :top]
        rank_k_global = np.argsort(-k_global, axis=1)[:, :top]
        rank_v_global = np.argsort(-v_global, axis=1)[:, :top]
        feature_mask = {}
        for i in rank_v_ego:
            for j in i:
                feature_mask[j] = None
        for i in range(d_loc):
            for j in rank_k_local[i, :]:
                feature_mask[j] = None
        for i in range(d_loc):
            for j in rank_q_local[i, :]:
                feature_mask[j] = None
        for i in range(d_loc):
            for j in rank_v_local[i, :]:
                feature_mask[j] = None
        for i in range(d_glb):
            for j in rank_k_global[i, :]:
                feature_mask[j] = None
        for i in range(d_glb):
            for j in rank_q_global[i, :]:
                feature_mask[j] = None
        for i in range(d_glb):
            for j in rank_v_global[i, :]:
                feature_mask[j] = None
        feature_mask = list(feature_mask.keys())
    else:
        rank_v_ego = rank(v_ego)
        rank_q_local = rank(np.abs(q_local))
        rank_k_local = rank(k_local)
        rank_v_local = rank(v_local)
        rank_q_global = rank(np.abs(q_global))
        rank_k_global = rank(k_global)
        rank_v_global = rank(v_global)
        max_rank = np.max(np.vstack([rank_v_ego, rank_q_local, rank_k_local, rank_v_local, rank_q_global, rank_k_global, rank_v_global]), axis=0)
        feature_mask = (max_rank > (max_rank.max() - 3))
        
    chosen_features = np.array(model.features)[feature_mask]

    if figsize == 'auto':
        figsize = (d * 0.2 + 4, len(chosen_features) * 0.15 + 1)
    print(figsize)
    fig, (cbar_axes, axes) = plt.subplots(2, 7, sharey='row', 
                                          figsize=figsize, 
                                          height_ratios=(1, len(chosen_features) + 1),
                                          width_ratios=(d_ego, d_loc, d_loc, d_loc, d_glb, d_glb, d_glb))
    
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "horizontal"}, 'square': True,
                     'vmin': 0.}

    #
    labels = ([f'$V_{{{i}}}$' for i in range(d_ego)])
    vmax = np.ceil(v_ego[:, feature_mask].max())
    sns.heatmap(v_ego[:, feature_mask].T, xticklabels=labels, ax=axes[0], 
                **common_params, cbar_ax=cbar_axes[0], vmax=vmax)
    axes[0].set_xlabel('Components\n╰── Ego ──╯')

    # Local
    #
    labels = ([f'$V_{{{i}}}$' for i in range(d_loc)])
    vmax = np.ceil(v_local[:, feature_mask].max())
    sns.heatmap(v_local[:, feature_mask].T, xticklabels=labels, ax=axes[3], 
                **common_params, cbar_ax=cbar_axes[3], vmax=vmax)
    axes[3].set_xlabel('Response')

    #
    labels = ([f'$Q_{{{i}}}$' for i in range(d_loc)])
    vmax = np.ceil(q_local[:, feature_mask].max())
    sns.heatmap(q_local[:, feature_mask].T, xticklabels=labels, ax=axes[2], 
                **common_params, cbar_ax=cbar_axes[2], vmax=vmax)
    axes[2].set_xlabel('Receiver\n╰' + '─' * (d_loc * 3 // 2) + '─ Local ─' + '─' * (d_loc * 3 // 2) + '╯')

    #
    labels = ([f'$K_{{{i}}}$' for i in range(d_loc)])
    vmax = np.ceil(k_local[:, feature_mask].max())
    sns.heatmap(k_local[:, feature_mask].T, xticklabels=labels, ax=axes[1], 
                **common_params, cbar_ax=cbar_axes[1], vmax=vmax)
    axes[1].set_xlabel('Sender')

    # Global
    #
    labels = ([f'$V_{{{i}}}$' for i in range(d_glb)])
    vmax = np.ceil(v_global[:, feature_mask].max())
    sns.heatmap(v_global[:, feature_mask].T, xticklabels=labels, ax=axes[6], 
                **common_params, cbar_ax=cbar_axes[6], vmax=vmax)
    axes[6].set_xlabel('Resp.')

    #
    labels = ([f'$Q_{{{i}}}$' for i in range(d_glb)])
    vmax = np.ceil(q_global[:, feature_mask].max())
    sns.heatmap(q_global[:, feature_mask].T, xticklabels=labels, ax=axes[5], 
                **common_params, cbar_ax=cbar_axes[5], vmax=vmax)
    axes[5].set_xlabel('Rec.\n╰─── Global ───╯')

    #
    labels = ([f'$K_{{{i}}}$' for i in range(d_glb)])
    vmax = np.ceil(k_global[:, feature_mask].max())
    sns.heatmap(k_global[:, feature_mask].T, xticklabels=labels, ax=axes[4], 
                **common_params, cbar_ax=cbar_axes[4], vmax=vmax)
    axes[4].set_xlabel('Send.')

    for i in range(7):
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
    
    fig.align_xlabels()
    plt.tight_layout()