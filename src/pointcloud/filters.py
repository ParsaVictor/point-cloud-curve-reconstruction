"""Statistical and density-based outlier filters for 3D point clouds."""

import numpy as np
from scipy.spatial import KDTree


def zscore_filter(pts: np.ndarray, threshold: float = 2.5) -> np.ndarray:
    """Remove points whose distance to their nearest neighbour is a statistical outlier.

    Uses median-absolute-deviation (MAD) so the threshold is robust to the
    heavy tails that outliers themselves introduce.

    Parameters
    ----------
    pts : (N, 3) array
    threshold : float
        Points with MAD-score > threshold are removed.

    Returns
    -------
    mask : (N,) bool array — True means keep.
    """
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn_dists = dists[:, 1]
    median = np.median(nn_dists)
    mad = np.median(np.abs(nn_dists - median))
    if mad == 0:
        return np.ones(len(pts), dtype=bool)
    scores = np.abs(nn_dists - median) / (1.4826 * mad)
    return scores <= threshold


def knn_outlier_filter(
    pts: np.ndarray,
    k: int = 8,
    mad_factor: float = 3.0,
) -> np.ndarray:
    """Robust kNN-distance outlier filter (Paper 1 — Lee 2000 variant).

    A point is flagged if its mean k-NN distance exceeds
    ``mad_factor * MAD`` above the global median.

    Returns
    -------
    mask : (N,) bool — True means keep.
    """
    k = min(k, len(pts) - 1)
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    median = np.median(mean_dists)
    mad = np.median(np.abs(mean_dists - median))
    threshold = median + mad_factor * (1.4826 * mad)
    return mean_dists <= threshold


def density_filter(
    pts: np.ndarray,
    k: int = 8,
    remove_quantile: float = 0.05,
) -> np.ndarray:
    """Remove the lowest-density points measured by inverse mean kNN distance.

    Parameters
    ----------
    remove_quantile : float
        Fraction of the lowest-density points to discard (0.05 → bottom 5 %).

    Returns
    -------
    mask : (N,) bool — True means keep.
    """
    k = min(k, len(pts) - 1)
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    densities = 1.0 / (mean_dists + 1e-9)
    threshold = np.quantile(densities, remove_quantile)
    return densities >= threshold


def lof_filter(
    pts: np.ndarray,
    k: int = 10,
    fraction: float = 0.02,
) -> np.ndarray:
    """Local Outlier Factor filter — removes the top-``fraction`` LOF scorers.

    Implemented from scratch (no sklearn dependency) using the standard LOF
    definition so the package stays lightweight for CI.

    Returns
    -------
    mask : (N,) bool — True means keep.
    """
    if len(pts) < k + 1:
        return np.ones(len(pts), dtype=bool)

    k = min(k, len(pts) - 1)
    tree = KDTree(pts)
    dists, idxs = tree.query(pts, k=k + 1)
    k_dists = dists[:, k]  # k-th nearest neighbour distance

    # Reachability distances and local reachability density
    reach = np.maximum(dists[:, 1:], k_dists[idxs[:, 1:]])
    lrd = 1.0 / (reach.mean(axis=1) + 1e-12)

    # LOF score
    lof = np.array([
        lrd[idxs[i, 1:]].mean() / (lrd[i] + 1e-12)
        for i in range(len(pts))
    ])

    threshold = np.quantile(lof, 1.0 - fraction)
    return lof <= threshold
