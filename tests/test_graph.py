"""Tests for pointcloud.graph — synthetic data, no external dataset."""

import numpy as np
import pytest
from pointcloud.graph import build_knn_graph, build_mst, cut_long_edges


def _line(n=50):
    t = np.linspace(0, 10, n)
    return np.column_stack([t, np.zeros(n), np.zeros(n)])


class TestBuildKnnGraph:
    def test_shape(self):
        pts = _line(30)
        W = build_knn_graph(pts, k=4)
        assert W.shape == (30, 30)

    def test_symmetric(self):
        pts = _line(20)
        W = build_knn_graph(pts, k=3)
        diff = (W - W.T).data
        assert len(diff) == 0 or np.allclose(diff, 0)

    def test_non_negative_weights(self):
        pts = _line(15)
        W = build_knn_graph(pts, k=4)
        assert (W.data >= 0).all()

    def test_k_clamped_to_n_minus_1(self):
        pts = np.random.rand(5, 3)
        W = build_knn_graph(pts, k=100)
        assert W.shape == (5, 5)


class TestBuildMst:
    def test_mst_has_n_minus_1_edges(self):
        pts = _line(20)
        W = build_knn_graph(pts, k=4)
        mst = build_mst(W)
        cx = mst.tocoo()
        # MST of connected graph: exactly N-1 directed edges in upper triangle
        assert len(cx.data) == 19

    def test_mst_weights_non_negative(self):
        pts = _line(15)
        W = build_knn_graph(pts, k=4)
        mst = build_mst(W)
        assert (mst.data >= 0).all()


class TestCutLongEdges:
    def test_two_clusters_become_two_components(self):
        # Two well-separated clusters — the long MST bridge should be cut
        c1 = np.random.default_rng(0).normal([0, 0, 0], 0.05, (30, 3))
        c2 = np.random.default_rng(1).normal([10, 0, 0], 0.05, (30, 3))
        pts = np.vstack([c1, c2])
        W = build_knn_graph(pts, k=5)
        mst = build_mst(W)
        labels, n_comp = cut_long_edges(mst, pts, factor=2.0)
        assert n_comp >= 2

    def test_returns_correct_shapes(self):
        pts = _line(25)
        W = build_knn_graph(pts, k=4)
        mst = build_mst(W)
        labels, n_comp = cut_long_edges(mst, pts)
        assert labels.shape == (25,)
        assert isinstance(n_comp, (int, np.integer))

    def test_high_factor_keeps_one_component(self):
        pts = _line(20)
        W = build_knn_graph(pts, k=4)
        mst = build_mst(W)
        _, n_comp = cut_long_edges(mst, pts, factor=1000.0)
        assert n_comp == 1
