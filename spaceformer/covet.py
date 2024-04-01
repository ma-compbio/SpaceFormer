import numpy
import sklearn as skl
import numpy as np


def mean_aggregation(expression, spatial, k_neighbors=8):
    n, g = expression.shape
    k = k_neighbors

    knn_graph = skl.neighbors.kneighbors_graph(spatial, k, mode='distance')
    knn_graph_inds = knn_graph.nonzero()[1].reshape([n, k])
    weights = 1 / k
    niche = weights * expression[knn_graph_inds.flatten(), :].reshape([n, k, g])
    res = niche.sum(axis=1)
    return res


def _svdenv_core(expr, spatial, k, override_mean=None):
    n, g = expr.shape
    knn_graph = skl.neighbors.kneighbors_graph(spatial, k, mode='distance')
    knn_graph_inds = knn_graph.nonzero()[1].reshape([n, k])
    weights = np.sqrt(1 / (k - 1))  # Simply sd with ddof=1 for now

    if override_mean is None:
        if isinstance(expr, numpy.ndarray) and not isinstance(expr, numpy.matrix):
            expr -= expr.mean(axis=0, keepdims=True)
        else:
            temp = expr.mean(axis=0)
            if temp.shape != (1, g):
                raise RuntimeError(f"Matrix type {type(expr)} is not supported.")
            expr -= temp
    else:
        expr = expr - override_mean
    expr = np.array(expr)
    niche = weights * np.array(expr[knn_graph_inds.flatten(), :]).reshape([n, k, g])

    svd_u, svd_s, svd_v = np.linalg.svd(niche)
    return svd_s, svd_v


def covet_sqrt(expression, spatial, k_neighbors=8, flatten=False, override_mean=None):
    """Find the COVET_SQRT matrices of each cell

    This will produce a gene x gene matrix for each cell, summarizing its niche of `k_neighbor` cells.
    :param expression: Expression profile of the cells (cell x gene)
    :param spatial: Spatial profile of the cells (cell x coordinates)
    :param k_neighbors: Number of cells in each niche
    :param flatten: if True, return a cell x (gene*gene) matrix; otherwise, return a cell x gene x gene tensor.
    :param override_mean: provide a different mean in calculation
    :return:
    """
    svd_s, svd_v = _svdenv_core(expression, spatial, k_neighbors, override_mean)
    res = (svd_s[:, :, None] * svd_v[:, :k_neighbors, :]).transpose(0, 2, 1) @ svd_v[:, :k_neighbors, :]

    if flatten:
        return res.reshape((res.shape[0], -1))
    else:
        return res


def svdenv(expression, spatial, k_neighbors=8, flatten=False, override_mean=None, genre='abs'):
    """Find the SVD based niche summary
    For 'abs' and 'sqr' approach, the niche for each cell is a G x k matrix.
    For 'covet' approach, the niche for each cell is a G x G matrix.

    :param expression: Expression profile of the cells (cell x gene); log1p normalized expression recommended.
    :param spatial: Spatial profile of the cells (cell x coordinates)
    :param k_neighbors: Number of cells in each niche
    :param flatten: if True, return a matrix (i.e., a vector for each cell);
        if False, return a tensor (i.e., a matrix for each cell).
    :param override_mean: provide a different mean in calculation
    :param genre: 'abs', 'sqr', or 'covet'
    :return: a numpy.ndarray of niche summary
    """
    svd_s, svd_v = _svdenv_core(expression, spatial, k_neighbors, override_mean)
    if genre == 'abs':
        res = np.abs(svd_s[:, :, None] * svd_v[:, :k_neighbors, :]).transpose(0, 2, 1)
    elif genre == 'sqr':
        res = np.square(np.sqrt(svd_s[:, :, None]) * svd_v[:, :k_neighbors, :]).transpose(0, 2, 1)
    elif genre == 'covet':
        res = (svd_s[:, :, None] * svd_v[:, :k_neighbors, :]).transpose(0, 2, 1) @ svd_v[:, :k_neighbors, :]
    else:
        raise ValueError(f"genre must be 'abs', 'sqr', or 'covet', but get {genre}.")

    if flatten:
        return res.reshape((res.shape[0], -1))
    else:
        return res