"""kNN graph and MST construction (Paper 1 — Lee 2000)."""

import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components


def build_knn_graph(pts: np.ndarray, k: int = 6):
    """Build a symmetric kNN graph as a sparse weight matrix.

    Parameters
    ----------
    pts : (N, 3) array
    k   : number of neighbours per point

    Returns
    -------
    W : (N, N) sparse CSR matrix with Euclidean edge weights.
    """
    k = min(k, len(pts) - 1)
    tree = KDTree(pts)
    dists, idxs = tree.query(pts, k=k + 1)
    N = len(pts)
    rows, cols, data = [], [], []
    for i in range(N):
        for j_local in range(1, k + 1):
            j = idxs[i, j_local]
            d = dists[i, j_local]
            rows += [i, j]
            cols += [j, i]
            data += [d, d]
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    return W


def build_mst(W):
    """Return the minimum spanning tree of weight matrix W (sparse CSR)."""
    return minimum_spanning_tree(W)


def cut_long_edges(
    mst,
    pts: np.ndarray,
    factor: float = 3.0,
) -> tuple:
    """Remove MST edges longer than ``factor * median_edge_length``.

    Returns
    -------
    labels : (N,) int array of connected-component labels (0-indexed).
    n_components : int
    """
    cx = mst.tocoo()
    if len(cx.data) == 0:
        return np.zeros(len(pts), dtype=int), 1
    threshold = factor * np.median(cx.data)
    mask = cx.data <= threshold
    kept = csr_matrix(
        (cx.data[mask], (cx.row[mask], cx.col[mask])),
        shape=mst.shape,
    )
    n_comp, labels = connected_components(kept, directed=False)
    return labels, n_comp
