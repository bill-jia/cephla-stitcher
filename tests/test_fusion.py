"""Tests for cephla_tilefusion.fusion module."""

import numpy as np
import pytest

from cephla_tilefusion.fusion import accumulate_tile_shard, normalize_shard, blend_numba_2d


class TestAccumulateTileShard:
    """Tests for accumulate_tile_shard function."""

    def test_basic_accumulation(self):
        """Test basic weighted accumulation."""
        fused = np.zeros((1, 100, 100), dtype=np.float32)
        weight = np.zeros((1, 100, 100), dtype=np.float32)
        sub = np.ones((1, 20, 20), dtype=np.float32) * 100
        w2d = np.ones((20, 20), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub, w2d, 10, 10)

        # Check accumulated region
        assert fused[0, 15, 15] == 100.0
        assert weight[0, 15, 15] == 1.0
        # Check outside region
        assert fused[0, 0, 0] == 0.0
        assert weight[0, 0, 0] == 0.0

    def test_weighted_accumulation(self):
        """Test that weights are applied correctly."""
        fused = np.zeros((1, 50, 50), dtype=np.float32)
        weight = np.zeros((1, 50, 50), dtype=np.float32)
        sub = np.ones((1, 10, 10), dtype=np.float32) * 100
        w2d = np.ones((10, 10), dtype=np.float32) * 0.5

        accumulate_tile_shard(fused, weight, sub, w2d, 0, 0)

        assert fused[0, 5, 5] == 50.0  # 100 * 0.5
        assert weight[0, 5, 5] == 0.5

    def test_overlapping_accumulation(self):
        """Test accumulation of overlapping tiles."""
        fused = np.zeros((1, 50, 50), dtype=np.float32)
        weight = np.zeros((1, 50, 50), dtype=np.float32)
        sub1 = np.ones((1, 20, 20), dtype=np.float32) * 100
        sub2 = np.ones((1, 20, 20), dtype=np.float32) * 100
        w2d = np.ones((20, 20), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub1, w2d, 0, 0)
        accumulate_tile_shard(fused, weight, sub2, w2d, 10, 10)

        # Non-overlapping region
        assert fused[0, 5, 5] == 100.0
        assert weight[0, 5, 5] == 1.0
        # Overlapping region
        assert fused[0, 15, 15] == 200.0
        assert weight[0, 15, 15] == 2.0

    def test_boundary_handling(self):
        """Test that out-of-bounds pixels are ignored."""
        fused = np.zeros((1, 20, 20), dtype=np.float32)
        weight = np.zeros((1, 20, 20), dtype=np.float32)
        sub = np.ones((1, 10, 10), dtype=np.float32) * 100
        w2d = np.ones((10, 10), dtype=np.float32)

        # Place partially outside
        accumulate_tile_shard(fused, weight, sub, w2d, 15, 15)

        # Only 5x5 should be inside
        assert fused[0, 17, 17] == 100.0
        assert fused[0, 10, 10] == 0.0

    def test_multichannel(self):
        """Test accumulation with multiple channels."""
        fused = np.zeros((3, 50, 50), dtype=np.float32)
        weight = np.zeros((3, 50, 50), dtype=np.float32)
        sub = np.ones((3, 10, 10), dtype=np.float32)
        sub[0] *= 100
        sub[1] *= 200
        sub[2] *= 300
        w2d = np.ones((10, 10), dtype=np.float32)

        accumulate_tile_shard(fused, weight, sub, w2d, 0, 0)

        assert fused[0, 5, 5] == 100.0
        assert fused[1, 5, 5] == 200.0
        assert fused[2, 5, 5] == 300.0


class TestNormalizeShard:
    """Tests for normalize_shard function."""

    def test_basic_normalization(self):
        """Test basic weight normalization."""
        fused = np.ones((1, 10, 10), dtype=np.float32) * 200
        weight = np.ones((1, 10, 10), dtype=np.float32) * 2

        normalize_shard(fused, weight)

        assert np.allclose(fused, 100.0)

    def test_zero_weight_handling(self):
        """Test that zero weights result in zero values."""
        fused = np.ones((1, 10, 10), dtype=np.float32) * 100
        weight = np.zeros((1, 10, 10), dtype=np.float32)

        normalize_shard(fused, weight)

        assert np.allclose(fused, 0.0)

    def test_varying_weights(self):
        """Test normalization with varying weights."""
        fused = np.array([[[100, 200], [300, 400]]], dtype=np.float32)
        weight = np.array([[[1, 2], [4, 4]]], dtype=np.float32)

        normalize_shard(fused, weight)

        expected = np.array([[[100, 100], [75, 100]]], dtype=np.float32)
        assert np.allclose(fused, expected)


class TestBlendNumba2D:
    """Tests for blend_numba_2d function."""

    def test_equal_weights(self):
        """Test blending with equal weights."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.ones(10, dtype=np.float32)
        wx_i = np.ones(10, dtype=np.float32)
        wy_j = np.ones(10, dtype=np.float32)
        wx_j = np.ones(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        assert np.allclose(result, 150.0)

    def test_weighted_blend(self):
        """Test blending with different weights."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.ones(10, dtype=np.float32) * 0.75
        wx_i = np.ones(10, dtype=np.float32)
        wy_j = np.ones(10, dtype=np.float32) * 0.25
        wx_j = np.ones(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        # (0.75 * 100 + 0.25 * 200) / (0.75 + 0.25) = 125
        assert np.allclose(result, 125.0)

    def test_zero_total_weight(self):
        """Test fallback when total weight is zero."""
        sub_i = np.ones((10, 10), dtype=np.float32) * 100
        sub_j = np.ones((10, 10), dtype=np.float32) * 200
        wy_i = np.zeros(10, dtype=np.float32)
        wx_i = np.zeros(10, dtype=np.float32)
        wy_j = np.zeros(10, dtype=np.float32)
        wx_j = np.zeros(10, dtype=np.float32)
        out = np.zeros((10, 10), dtype=np.float32)

        result = blend_numba_2d(sub_i, sub_j, wy_i, wx_i, wy_j, wx_j, out)

        # Should fall back to sub_i
        assert np.allclose(result, 100.0)
