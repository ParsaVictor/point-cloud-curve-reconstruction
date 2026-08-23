"""Combined denoising + reconstruction pipeline (Paper 5 / final notebook)."""

import numpy as np

from .filters import knn_outlier_filter, density_filter, lof_filter
from .curves import preserve_curve_points


def run_pipeline(
    pts: np.ndarray,
    k_knn: int = 8,
    mad_factor: float = 3.5,
    density_quantile: float = 0.02,
    lof_fraction: float = 0.01,
    curve_tol_factor: float = 2.5,
    curve_min_cluster: int = 5,
    preserve_curves: bool = True,
) -> dict:
    """Run the full denoising pipeline and return labelled results.

    Pipeline stages
    ---------------
    1. Robust kNN outlier filter (MAD-based)
    2. Density filter (remove lowest-density points)
    3. LOF filter (local outlier factor)
    4. Curve-point recovery (optional — restores points near fitted curves)

    Parameters
    ----------
    pts              : (N, 3) raw point cloud
    k_knn            : neighbours for kNN and density filters
    mad_factor       : MAD multiplier for kNN filter aggressiveness
    density_quantile : bottom quantile removed by density filter
    lof_fraction     : top fraction removed by LOF filter
    curve_tol_factor : tolerance for curve-point recovery
    curve_min_cluster: min points needed to fit a recovery curve
    preserve_curves  : whether to run Stage 4 curve recovery

    Returns
    -------
    dict with keys:
        'pts'      — (N, 3) original points
        'mask'     — (N,) bool, True = kept
        'clean'    — (M, 3) kept points
        'n_input'  — N
        'n_output' — M
        'labels'   — (N,) int, -1 = removed, 0 = kept (for compatibility)
    """
    if len(pts) == 0:
        return {
            "pts": pts, "mask": np.array([], dtype=bool),
            "clean": pts, "n_input": 0, "n_output": 0,
            "labels": np.array([], dtype=int),
        }

    # Stage 1 — robust kNN filter
    mask = knn_outlier_filter(pts, k=k_knn, mad_factor=mad_factor)

    # Stage 2 — density filter (applied globally)
    mask &= density_filter(pts, k=k_knn, remove_quantile=density_quantile)

    # Stage 3 — LOF filter
    if mask.sum() > 0:
        lof_mask = np.zeros(len(pts), dtype=bool)
        lof_mask[mask] = lof_filter(pts[mask], fraction=lof_fraction)
        mask &= lof_mask

    # Stage 4 — curve-point recovery
    if preserve_curves and mask.sum() >= curve_min_cluster:
        mask = preserve_curve_points(
            pts, mask,
            tol_factor=curve_tol_factor,
            min_cluster=curve_min_cluster,
        )

    clean = pts[mask]
    labels = np.where(mask, 0, -1)
    return {
        "pts": pts,
        "mask": mask,
        "clean": clean,
        "n_input": len(pts),
        "n_output": len(clean),
        "labels": labels,
    }
