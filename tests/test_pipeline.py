"""Integration tests for the full denoising pipeline."""

import numpy as np
import pytest
from pointcloud.pipeline import run_pipeline


def _noisy_curve(n=300, n_outliers=30):
    rng = np.random.default_rng(7)
    t = np.linspace(0, 2 * np.pi, n)
    curve = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
    curve += rng.normal(0, 0.02, curve.shape)
    outliers = rng.uniform(-3, 3, (n_outliers, 3))
    pts = np.vstack([curve, outliers])
    true_inlier = np.array([True] * n + [False] * n_outliers)
    return pts, true_inlier


class TestRunPipeline:
    def test_output_keys(self):
        pts, _ = _noisy_curve(200, 20)
        result = run_pipeline(pts)
        for key in ("pts", "mask", "clean", "n_input", "n_output", "labels"):
            assert key in result

    def test_n_input_equals_len_pts(self):
        pts, _ = _noisy_curve(150)
        result = run_pipeline(pts)
        assert result["n_input"] == len(pts)

    def test_clean_matches_mask(self):
        pts, _ = _noisy_curve(150)
        result = run_pipeline(pts)
        assert result["clean"].shape == (result["mask"].sum(), 3)

    def test_removes_spread_outliers(self):
        pts, true_inlier = _noisy_curve(300, 30)
        result = run_pipeline(pts)
        outliers_kept = (~true_inlier & result["mask"]).sum()
        assert outliers_kept < 20, f"Too many outliers kept: {outliers_kept}"

    def test_retains_most_inliers(self):
        pts, true_inlier = _noisy_curve(300, 30)
        result = run_pipeline(pts)
        inliers_kept = (true_inlier & result["mask"]).sum()
        assert inliers_kept >= 250

    def test_empty_input(self):
        result = run_pipeline(np.empty((0, 3)))
        assert result["n_input"] == 0
        assert result["n_output"] == 0

    def test_labels_values(self):
        pts, _ = _noisy_curve(100)
        result = run_pipeline(pts)
        unique = set(np.unique(result["labels"]))
        assert unique.issubset({-1, 0})

    def test_preserve_curves_disabled(self):
        pts, _ = _noisy_curve(200)
        r_on = run_pipeline(pts, preserve_curves=True)
        r_off = run_pipeline(pts, preserve_curves=False)
        # With curve recovery, output should be >= without it
        assert r_on["n_output"] >= r_off["n_output"]

    def test_idempotent_on_clean_data(self):
        rng = np.random.default_rng(99)
        t = np.linspace(0, 2 * np.pi, 100)
        pts = np.column_stack([np.cos(t), np.sin(t), np.zeros(100)])
        pts += rng.normal(0, 0.005, pts.shape)
        result = run_pipeline(pts, mad_factor=5.0, density_quantile=0.001, lof_fraction=0.001)
        assert result["n_output"] >= 90
