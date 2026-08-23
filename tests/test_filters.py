"""Tests for pointcloud.filters — run on synthetic data, no dataset required."""

import numpy as np
import pytest
from pointcloud.filters import (
    zscore_filter,
    knn_outlier_filter,
    density_filter,
    lof_filter,
)


def _helix(n=200, noise=0.0):
    t = np.linspace(0, 4 * np.pi, n)
    pts = np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)])
    if noise > 0:
        pts += np.random.default_rng(0).normal(0, noise, pts.shape)
    return pts


def _with_outliers(pts, n_out=10, spread=5.0):
    rng = np.random.default_rng(42)
    outliers = rng.uniform(-spread, spread, (n_out, 3))
    combined = np.vstack([pts, outliers])
    true_mask = np.array([True] * len(pts) + [False] * n_out)
    return combined, true_mask


class TestZscoreFilter:
    def test_returns_bool_array(self):
        pts = _helix(100)
        mask = zscore_filter(pts)
        assert mask.dtype == bool
        assert mask.shape == (100,)

    def test_keeps_clean_curve(self):
        pts = _helix(150)
        mask = zscore_filter(pts, threshold=3.0)
        assert mask.sum() >= 130

    def test_removes_spread_outliers(self):
        pts, true_mask = _with_outliers(_helix(200), n_out=15, spread=8.0)
        mask = zscore_filter(pts, threshold=2.5)
        # Outliers should be mostly removed
        outlier_kept = (~true_mask & mask).sum()
        assert outlier_kept < 8

    def test_single_point(self):
        mask = zscore_filter(np.array([[0.0, 0.0, 0.0]]))
        assert mask.shape == (1,)

    def test_uniform_data(self):
        pts = np.ones((10, 3))
        mask = zscore_filter(pts)
        assert mask.shape == (10,)


class TestKnnOutlierFilter:
    def test_output_shape(self):
        pts = _helix(80)
        mask = knn_outlier_filter(pts)
        assert mask.shape == (80,)
        assert mask.dtype == bool

    def test_removes_obvious_outliers(self):
        pts, true_mask = _with_outliers(_helix(200), n_out=20, spread=10.0)
        mask = knn_outlier_filter(pts, k=8, mad_factor=2.5)
        # Most inliers kept
        assert (true_mask & mask).sum() >= 160
        # Most outliers removed
        assert (~true_mask & mask).sum() < 15

    def test_k_larger_than_points_clamped(self):
        pts = np.random.rand(5, 3)
        mask = knn_outlier_filter(pts, k=100)
        assert mask.shape == (5,)

    def test_tight_threshold_removes_more(self):
        pts, _ = _with_outliers(_helix(200), n_out=10)
        loose = knn_outlier_filter(pts, mad_factor=5.0).sum()
        tight = knn_outlier_filter(pts, mad_factor=1.5).sum()
        assert tight <= loose


class TestDensityFilter:
    def test_output_shape(self):
        pts = _helix(100)
        mask = density_filter(pts)
        assert mask.shape == (100,)

    def test_fraction_removed(self):
        pts = _helix(200)
        mask = density_filter(pts, remove_quantile=0.10)
        # At most 10 % + small rounding should be removed
        assert mask.sum() >= 175

    def test_quantile_zero_keeps_all(self):
        pts = _helix(50)
        mask = density_filter(pts, remove_quantile=0.0)
        assert mask.all()

    def test_k_clamped(self):
        pts = np.random.rand(4, 3)
        mask = density_filter(pts, k=50)
        assert mask.shape == (4,)


class TestLofFilter:
    def test_output_shape(self):
        pts = _helix(100)
        mask = lof_filter(pts)
        assert mask.shape == (100,)

    def test_fraction_removed(self):
        pts = _helix(200)
        mask = lof_filter(pts, fraction=0.05)
        removed = (~mask).sum()
        # Allow ±2 from exactly 5%
        assert abs(removed - 10) <= 4

    def test_too_few_points_returns_all(self):
        pts = np.random.rand(3, 3)
        mask = lof_filter(pts, k=10)
        assert mask.all()

    def test_removes_extreme_outlier(self):
        pts = _helix(150)
        lone = np.array([[100.0, 100.0, 100.0]])
        combined = np.vstack([pts, lone])
        mask = lof_filter(combined, fraction=0.02)
        assert not mask[-1], "The lone outlier at (100,100,100) should be removed"
