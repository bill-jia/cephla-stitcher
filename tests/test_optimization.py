"""Tests for cephla_tilefusion.optimization module."""

import numpy as np
import pytest

from cephla_tilefusion.optimization import (
    solve_global,
    two_round_optimization,
    links_from_pairwise_metrics,
)


class TestSolveGlobal:
    """Tests for solve_global function."""

    def test_two_tiles_simple(self):
        """Test optimization with two tiles and known offset."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 5.0]), "w": 1.0},
        ]
        shifts = solve_global(links, n_tiles=2, fixed_indices=[0])

        assert shifts.shape == (2, 2)
        # Tile 0 should be at origin
        assert np.allclose(shifts[0], [0, 0])
        # Tile 1 should be at the measured offset
        assert np.allclose(shifts[1], [10, 5])

    def test_three_tiles_chain(self):
        """Test optimization with three tiles in a chain."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 1.0},
        ]
        shifts = solve_global(links, n_tiles=3, fixed_indices=[0])

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0])
        assert np.allclose(shifts[2], [20, 0])

    def test_overdetermined_system(self):
        """Test optimization with more links than needed."""
        # Triangle of three tiles with redundant links
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([0.0, 10.0]), "w": 1.0},
            {"i": 0, "j": 2, "t": np.array([10.0, 10.0]), "w": 1.0},
        ]
        shifts = solve_global(links, n_tiles=3, fixed_indices=[0])

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0], atol=0.1)
        assert np.allclose(shifts[2], [10, 10], atol=0.1)

    def test_weighted_links(self):
        """Test that weights affect the solution."""
        # Two conflicting measurements with different weights
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 0, "j": 1, "t": np.array([20.0, 0.0]), "w": 3.0},
        ]
        shifts = solve_global(links, n_tiles=2, fixed_indices=[0])

        # Should be closer to 20 due to higher weight
        assert shifts[1, 0] > 15

    def test_no_links(self):
        """Test with no links (should return zeros)."""
        shifts = solve_global([], n_tiles=3, fixed_indices=[0])
        assert np.allclose(shifts, 0)


class TestTwoRoundOptimization:
    """Tests for two_round_optimization function."""

    def test_no_outliers(self):
        """Test with consistent links (no outliers to remove)."""
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 1.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 1.0},
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.5, abs_thresh=5.0, iterative=False
        )

        assert np.allclose(shifts[0], [0, 0])
        assert np.allclose(shifts[1], [10, 0])
        assert np.allclose(shifts[2], [20, 0])

    def test_outlier_removal(self):
        """Test that outliers are removed."""
        # Use more extreme outlier and tighter chain
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 2.0},
            {
                "i": 0,
                "j": 2,
                "t": np.array([1000.0, 0.0]),
                "w": 0.1,
            },  # Clear outlier with low weight
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.3, abs_thresh=50.0, iterative=False
        )

        # Result should be closer to 20 than to the outlier value
        assert shifts[2, 0] < 100  # Not dominated by outlier

    def test_iterative_mode(self):
        """Test iterative outlier removal converges."""
        # Simple chain with one clear outlier
        links = [
            {"i": 0, "j": 1, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 1, "j": 2, "t": np.array([10.0, 0.0]), "w": 2.0},
            {"i": 0, "j": 2, "t": np.array([500.0, 0.0]), "w": 0.5},  # Clear outlier
        ]
        shifts = two_round_optimization(
            links, n_tiles=3, fixed_indices=[0], rel_thresh=0.3, abs_thresh=50.0, iterative=True
        )

        # Should converge to something reasonable (not dominated by outlier)
        assert shifts[2, 0] < 100


class TestLinksFromPairwiseMetrics:
    """Tests for links_from_pairwise_metrics function."""

    def test_basic_conversion(self):
        """Test basic conversion from pairwise metrics."""
        metrics = {
            (0, 1): (10, 5, 0.9),
            (1, 2): (10, 0, 0.8),
        }
        links = links_from_pairwise_metrics(metrics)

        assert len(links) == 2
        assert links[0]["i"] == 0
        assert links[0]["j"] == 1
        assert np.allclose(links[0]["t"], [10, 5])
        assert links[0]["w"] == pytest.approx(np.sqrt(0.9))

    def test_empty_metrics(self):
        """Test with empty metrics."""
        links = links_from_pairwise_metrics({})
        assert links == []

    def test_weight_calculation(self):
        """Test that weights are sqrt of scores."""
        metrics = {
            (0, 1): (0, 0, 0.25),
            (1, 2): (0, 0, 1.0),
        }
        links = links_from_pairwise_metrics(metrics)

        assert links[0]["w"] == pytest.approx(0.5)
        assert links[1]["w"] == pytest.approx(1.0)
