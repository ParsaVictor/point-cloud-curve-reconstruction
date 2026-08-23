"""Curve-fitting and curve-preservation utilities (Papers 3–5)."""

import numpy as np
from scipy.spatial import KDTree


def fit_poly_curve(
    pts: np.ndarray,
    degree: int = 2,
    n_samples: int = 200,
) -> np.ndarray:
    """Fit a polynomial curve to a 3D point cloud via PCA projection.

    Projects points onto the first principal component, fits a polynomial of
    the requested degree in that 1-D space, and maps back to 3-D.

    Parameters
    ----------
    pts    : (N, 3) point array
    degree : polynomial degree (1 = line, 2 = parabola, …)
    n_samples : number of points to sample on the fitted curve

    Returns
    -------
    curve_pts : (n_samples, 3) array of points on the fitted curve.
                Returns empty array if fewer than ``degree + 1`` points given.
    """
    if len(pts) < degree + 1:
        return np.empty((0, 3))

    center = pts.mean(axis=0)
    centered = pts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axis = Vt[0]

    t = centered @ axis
    t_min, t_max = t.min(), t.max()

    # Project secondary directions for transverse fitting
    ax2 = Vt[1] if Vt.shape[0] > 1 else np.zeros(3)
    y = centered @ ax2

    coeffs = np.polyfit(t, y, deg=min(degree, len(pts) - 1))
    t_eval = np.linspace(t_min, t_max, n_samples)
    y_eval = np.polyval(coeffs, t_eval)

    curve_pts = center + np.outer(t_eval, axis) + np.outer(y_eval, ax2)
    return curve_pts


def preserve_curve_points(
    pts: np.ndarray,
    mask: np.ndarray,
    tol_factor: float = 2.5,
    min_cluster: int = 5,
    n_samples: int = 300,
) -> np.ndarray:
    """Recover points near fitted curves that outlier filters removed.

    Fits a curve to the *kept* points (``mask == True``), then re-adds any
    removed point whose distance to the fitted curve is within
    ``tol_factor * local_bandwidth``.

    Parameters
    ----------
    pts        : (N, 3) full point array
    mask       : (N,) bool — current keep/remove state
    tol_factor : distance tolerance as a multiple of median kNN distance
    min_cluster : minimum points required to fit a curve

    Returns
    -------
    new_mask : (N,) bool — updated keep/remove mask.
    """
    new_mask = mask.copy()
    kept = pts[mask]
    if len(kept) < min_cluster:
        return new_mask

    curve = fit_poly_curve(kept, degree=2, n_samples=n_samples)
    if len(curve) == 0:
        return new_mask

    # Local bandwidth = median kNN distance of kept points
    tree_kept = KDTree(kept)
    k = min(6, len(kept) - 1)
    dists, _ = tree_kept.query(kept, k=k + 1)
    bw = np.median(dists[:, 1:].mean(axis=1))
    tolerance = tol_factor * bw

    curve_tree = KDTree(curve)
    removed_idx = np.where(~mask)[0]
    if len(removed_idx) == 0:
        return new_mask

    removed_pts = pts[removed_idx]
    d_to_curve, _ = curve_tree.query(removed_pts)
    recover = d_to_curve <= tolerance
    new_mask[removed_idx[recover]] = True
    return new_mask
