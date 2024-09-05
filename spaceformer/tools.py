import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .model import SpaceFormer
from typing import Literal
from torch import nn
import scanpy as sc
import numpy as np

palettes = {
    'ncr10': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85'],
    'npg3': ['#952522', '#0a4b84', '#98752b']
}

def rank(x, axis=1):
    return np.argsort(np.argsort(x, axis=axis), axis=axis)

def plot_transforms(model, top=3, reorder=False, figsize='auto', qkv_colors=palettes['npg3']):
    assert len(qkv_colors) == 3, f"Expect a color palette with at 3 colors, get {len(qkv_colors)}."
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

    if figsize == 'auto':
        figsize = (d_ego * 0.36 + (d_loc + d_glb) * 0.49 + 2 + .5, len(chosen_features) * 0.15 + .25 + .75)
    # print(figsize)
    fig, axes = plt.subplots(2, d_ego + d_loc + d_glb + 1, sharey='row', sharex='col',
                                          figsize=figsize, 
                                          height_ratios=(.75, len(chosen_features) * .15 + .25),
                                          width_ratios=[2] * d_ego + [3] * (d_loc + d_glb) + [.5])
    plot_axes = axes[1]
    bar_axes = axes[0]
    cbar_ax = plot_axes[-1].inset_axes([0.0, 0.1, 1.0, .8])
    common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
                     'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
                     'vmax': 1., 'vmin': 0.}

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


# def plot_transforms(model, top=3, reorder=False, figsize='auto', qkv_colors=palettes['npg3']):
#     assert len(qkv_colors) == 3, f"Expect a color palette with at 3 colors, get {len(qkv_colors)}."
#     d_ego = model.spatial_gather.d_ego
#     d_loc = model.spatial_gather.d_local
#     d_glb = model.spatial_gather.d_global
#     d = d_ego + d_loc + d_glb
# 
#     qk_ego, v_ego = model.get_ego_transform()
#     q_local, k_local, v_local = model.get_local_transform()
#     q_global, k_global, v_global = model.get_global_transform()
# 
#     if reorder:
#         rank_v_ego = np.argsort(-v_ego, axis=1)[:, :top]
#         rank_q_local = np.argsort(-np.abs(q_local), axis=1)[:, :top]
#         rank_k_local = np.argsort(-k_local, axis=1)[:, :top]
#         rank_v_local = np.argsort(-v_local, axis=1)[:, :top]
#         rank_q_global = np.argsort(-np.abs(q_global), axis=1)[:, :top]
#         rank_k_global = np.argsort(-k_global, axis=1)[:, :top]
#         rank_v_global = np.argsort(-v_global, axis=1)[:, :top]
#         feature_mask = {}
#         for i in rank_v_ego:
#             for j in i:
#                 feature_mask[j] = None
#         for i in range(d_loc):
#             for j in rank_k_local[i, :]:
#                 feature_mask[j] = None
#             for j in rank_q_local[i, :]:
#                 feature_mask[j] = None
#             for j in rank_v_local[i, :]:
#                 feature_mask[j] = None
#         for i in range(d_glb):
#             for j in rank_k_global[i, :]:
#                 feature_mask[j] = None
#             for j in rank_q_global[i, :]:
#                 feature_mask[j] = None
#             for j in rank_v_global[i, :]:
#                 feature_mask[j] = None
#         feature_mask = list(feature_mask.keys())
#     else:
#         rank_v_ego = rank(v_ego)
#         rank_q_local = rank(np.abs(q_local))
#         rank_k_local = rank(k_local)
#         rank_v_local = rank(v_local)
#         rank_q_global = rank(np.abs(q_global))
#         rank_k_global = rank(k_global)
#         rank_v_global = rank(v_global)
#         max_rank = np.max(np.vstack([rank_v_ego, 
#                                      rank_q_local, 
#                                      rank_k_local, 
#                                      rank_v_local, 
#                                      rank_q_global, 
#                                      rank_k_global, 
#                                      rank_v_global]), axis=0)
#         feature_mask = (max_rank > (max_rank.max() - 3))
#         
#     chosen_features = np.array(model.features)[feature_mask]
# 
#     if figsize == 'auto':
#         figsize = (d_ego * 0.2 + (d_loc + d_glb) * 0.4 + 2 + .5, len(chosen_features) * 0.16 + .25 + .75)
#     # print(figsize)
#     fig, (bar_axes, axes) = plt.subplots(2, 1 + d_loc + d_glb + 1, sharey='row', sharex='col',
#                                           figsize=figsize, 
#                                           height_ratios=(.75, len(chosen_features) * .16 + .25),
#                                           width_ratios=[d_ego] + [3] * (d_loc + d_glb) + [.5])
#     cbar_ax = axes[-1].inset_axes([0.0, 0.05, 1.0, .9])
#     common_params = {'linewidths': .05, 'linecolor': 'gray', 'yticklabels': chosen_features, 
#                      'cmap': 'Reds', 'cbar_kws': {"orientation": "vertical"}, 'square': True,
#                      'vmax': 1., 'vmin': 0.}
# 
#     # Local
#     #
#     for i in range(0, d_loc + d_glb + 1):
#         if i == 0:
#             what = 'Ego\;(v)'
#             labels = ([f'{i}' for i in range(d_ego)])
#             to_plot = v_ego[:, feature_mask].T.copy()
#             color = qkv_colors[2]
#             true_vmax_2 = qk_ego[:, feature_mask].T.max(axis=0)
#             color2 = qkv_colors[1]
#         elif i - 1 < d_loc:
#             j = i - 1
#             what = f'Local_{j}'
#             labels = ('k', 'q', 'v')
#             to_plot = np.vstack((k_local[j, feature_mask],
#                                  q_local[j, feature_mask],
#                                  v_local[j, feature_mask])).T
#             color = qkv_colors
#         else:
#             j = i - 1 - d_loc
#             what = f'Global_{j}'
#             labels = ('k', 'q', 'v')
#             to_plot = np.vstack((k_global[j, feature_mask],
#                                  q_global[j, feature_mask],
#                                  v_global[j, feature_mask])).T
#             color = qkv_colors
#         
#         true_vmax = to_plot.max(axis=0)
#         # print(true_vmax)
#         to_plot /= true_vmax
#         
#         if i == 0:
#             bar_axes[i].bar(np.arange(len(true_vmax)) + .3, true_vmax_2, color=color2, width=.4)
#             bar_axes[i].bar(np.arange(len(true_vmax)) + .7, true_vmax, color=color, width=.4)
#         else:
#             bar_axes[i].bar(np.arange(len(true_vmax)) + .5, true_vmax, color=color)
#         
#         bar_axes[i].set_xticks(np.arange(len(true_vmax)) + .5, [''] * len(true_vmax))
#         bar_axes[i].set_yscale('log')
#         if i != 0:
#             bar_axes[i].get_yaxis().set_visible(False)
#         for pos in ['right', 'top', 'left']:
#             if pos == 'left' and i == 0:
#                 continue
#             else:
#                 bar_axes[i].spines[pos].set_visible(False)
#         sns.heatmap(to_plot, xticklabels=labels, ax=axes[i], 
#                     **common_params, cbar_ax=cbar_ax)
#         axes[i].set_xlabel(f"${what}$")
#         
#     # All text straight up
#     for i in range(1 + d_loc + d_glb):
#         axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=0)
# 
#     # Remove duplicate cbars
#     bar_axes[-1].set_visible(False)
# 
#     axes[-1].get_yaxis().set_visible(False)
#     axes[-1].get_xaxis().set_visible(False)
#     for pos in ['right', 'top', 'bottom', 'left']:
#         axes[-1].spines[pos].set_visible(False)
#     # axes[-1].set_visible(False)
# 
#     fig.align_xlabels()
#     plt.tight_layout()


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

