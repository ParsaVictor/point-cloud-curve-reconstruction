"""Tests for pointcloud.curves."""

import numpy as np
import pytest
from pointcloud.curves import fit_poly_curve, preserve_curve_points


def _arc(n=80, noise=0.01):
    rng = np.random.default_rng(3)
    t = np.linspace(0, np.pi, n)
    pts = np.column_stack([np.cos(t), np.sin(t), np.zeros(n)])
    return pts + rng.normal(0, noise, pts.shape)


class TestFitPolyCurve:
    def test_output_shape(self):
        pts = _arc(60)
        curve = fit_poly_curve(pts, degree=2, n_samples=100)
        assert curve.shape == (100, 3)

    def test_too_few_points_returns_empty(self):
        pts = np.random.rand(1, 3)
        curve = fit_poly_curve(pts, degree=2)
        assert len(curve) == 0

    def test_linear_fit_on_line(self):
        t = np.linspace(0, 5, 50)
        pts = np.column_stack([t, np.zeros(50), np.zeros(50)])
        curve = fit_poly_curve(pts, degree=1, n_samples=50)
        assert curve.shape[0] == 50
        # Fitted curve should span roughly the same range as input
        assert curve[:, 0].max() >= 4.0

    def test_degree_zero_clamped(self):
        pts = _arc(30)
        curve = fit_poly_curve(pts, degree=0, n_samples=20)
        assert len(curve) > 0


class TestPreserveCurvePoints:
    def test_mask_unchanged_when_already_clean(self):
        pts = _arc(100)
        mask = np.ones(len(pts), dtype=bool)
        new_mask = preserve_curve_points(pts, mask)
        assert new_mask.sum() == mask.sum()

    def test_recovers_near_curve_points(self):
        pts = _arc(120)
        # Remove every 5th point
        mask = np.ones(len(pts), dtype=bool)
        mask[::5] = False
        new_mask = preserve_curve_points(pts, mask, tol_factor=3.0)
        # Some removed points should be recovered
        assert new_mask.sum() > mask.sum()

    def test_does_not_recover_far_outliers(self):
        pts = _arc(80)
        outliers = np.array([[100.0, 100.0, 100.0], [99.0, 98.0, 97.0]])
        combined = np.vstack([pts, outliers])
        mask = np.array([True] * len(pts) + [False] * 2)
        new_mask = preserve_curve_points(combined, mask, tol_factor=2.0)
        # The far outliers should NOT be recovered
        assert not new_mask[len(pts)]
        assert not new_mask[len(pts) + 1]

    def test_output_is_superset_of_input_mask(self):
        pts = _arc(60)
        mask = np.ones(len(pts), dtype=bool)
        mask[10:20] = False
        new_mask = preserve_curve_points(pts, mask)
        # Curve recovery never removes previously-kept points
        assert (new_mask | ~mask).all()
