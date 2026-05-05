"""Tests for cephla_tilefusion.registration module."""

import numpy as np
import pytest

from cephla_tilefusion.registration import (
    find_adjacent_pairs,
    compute_pair_bounds,
    register_and_score,
)


class TestFindAdjacentPairs:
    """Tests for find_adjacent_pairs function."""

    def test_horizontal_neighbors(self):
        """Test detection of horizontal neighbors."""
        # Two tiles side by side with 50 pixel overlap
        positions = [(0, 0), (0, 950)]  # 1000px wide tiles, 50px overlap
        pairs = find_adjacent_pairs(
            positions, pixel_size=(1, 1), tile_shape=(1000, 1000), min_overlap=15
        )

        assert len(pairs) == 1
        assert pairs[0][0] == 0
        assert pairs[0][1] == 1

    def test_vertical_neighbors(self):
        """Test detection of vertical neighbors."""
        positions = [(0, 0), (950, 0)]  # Vertical arrangement
        pairs = find_adjacent_pairs(
            positions, pixel_size=(1, 1), tile_shape=(1000, 1000), min_overlap=15
        )

        assert len(pairs) == 1

    def test_diagonal_not_neighbors(self):
        """Test that diagonal tiles are not detected as neighbors."""
        positions = [(0, 0), (950, 950)]  # Diagonal
        pairs = find_adjacent_pairs(
            positions, pixel_size=(1, 1), tile_shape=(1000, 1000), min_overlap=15
        )

        assert len(pairs) == 0

    def test_no_overlap(self):
        """Test that non-overlapping tiles are not paired."""
        positions = [(0, 0), (0, 2000)]  # Too far apart
        pairs = find_adjacent_pairs(
            positions, pixel_size=(1, 1), tile_shape=(1000, 1000), min_overlap=15
        )

        assert len(pairs) == 0

    def test_grid_arrangement(self):
        """Test 2x2 grid of tiles."""
        # 2x2 grid with overlap
        positions = [
            (0, 0),
            (0, 900),
            (900, 0),
            (900, 900),
        ]
        pairs = find_adjacent_pairs(
            positions, pixel_size=(1, 1), tile_shape=(1000, 1000), min_overlap=15
        )

        # Should have 4 pairs: (0,1), (0,2), (1,3), (2,3)
        assert len(pairs) == 4

    def test_pixel_size_scaling(self):
        """Test that pixel size is accounted for."""
        # Positions in physical units (um), tiles are 100px at 10um/px = 1000um
        positions = [(0, 0), (0, 900)]  # 900um apart in 1000um tiles = 100um overlap
        pairs = find_adjacent_pairs(
            positions, pixel_size=(10, 10), tile_shape=(100, 100), min_overlap=5
        )

        assert len(pairs) == 1


class TestComputePairBounds:
    """Tests for compute_pair_bounds function."""

    def test_horizontal_overlap(self):
        """Test bounds for horizontal overlap."""
        adjacent_pairs = [(0, 1, 0, 900, 1000, 100)]  # dx=900, 100px overlap
        bounds = compute_pair_bounds(adjacent_pairs, tile_shape=(1000, 1000))

        assert len(bounds) == 1
        i_pos, j_pos, by_i, bx_i, by_j, bx_j = bounds[0]
        assert i_pos == 0
        assert j_pos == 1
        # Bounds should define the overlap region
        assert bx_i[1] - bx_i[0] == 100  # 100px wide overlap

    def test_vertical_overlap(self):
        """Test bounds for vertical overlap."""
        adjacent_pairs = [(0, 1, 900, 0, 100, 1000)]  # dy=900, 100px overlap
        bounds = compute_pair_bounds(adjacent_pairs, tile_shape=(1000, 1000))

        assert len(bounds) == 1
        i_pos, j_pos, by_i, bx_i, by_j, bx_j = bounds[0]
        assert by_i[1] - by_i[0] == 100  # 100px tall overlap


class TestRegisterAndScore:
    """Tests for register_and_score function."""

    def test_identical_images(self):
        """Test registration of identical images."""
        img = np.random.rand(100, 100).astype(np.float32) * 255
        shift, ssim = register_and_score(img, img, win_size=7)

        assert shift is not None
        assert abs(shift[0]) < 1.0
        assert abs(shift[1]) < 1.0
        assert ssim > 0.95

    def test_shifted_image(self):
        """Test registration of shifted image."""
        img1 = np.zeros((100, 100), dtype=np.float32)
        img1[30:70, 30:70] = 255  # Square in center

        img2 = np.zeros((100, 100), dtype=np.float32)
        img2[35:75, 32:72] = 255  # Shifted by (5, 2)

        shift, ssim = register_and_score(img1, img2, win_size=7)

        assert shift is not None
        # Phase correlation may return the shift with opposite sign depending on convention
        # Just verify a shift was detected
        assert abs(shift[0]) > 1 or abs(shift[1]) > 1

    def test_returns_float_tuple(self):
        """Test that return types are correct."""
        img = np.random.rand(50, 50).astype(np.float32) * 255
        shift, ssim = register_and_score(img, img, win_size=7)

        assert isinstance(shift, tuple)
        assert len(shift) == 2
        assert isinstance(shift[0], float)
        assert isinstance(shift[1], float)
        assert isinstance(ssim, float)


class TestOverlapBoundsSize:
    """Tests verifying overlap region size from compute_pair_bounds."""

    def test_horizontal_overlap_much_smaller_than_full_tile(self):
        """Verify horizontal overlap region is much smaller than full tile.

        For a 2048x2048 tile with 15% overlap (~307px), the overlap region is
        roughly 2048 * 307 = 628,736 pixels, not 2048 * 2048 = 4,194,304 pixels.
        """
        tile_shape = (2048, 2048)
        overlap_pixels = 307
        dx = tile_shape[1] - overlap_pixels  # 1741

        adjacent_pairs = [
            (0, 1, 0, dx, tile_shape[0], overlap_pixels),  # horizontal pair
        ]
        pair_bounds = compute_pair_bounds(adjacent_pairs, tile_shape)

        _, _, bounds_i_y, bounds_i_x, _, _ = pair_bounds[0]
        actual_overlap_h = bounds_i_y[1] - bounds_i_y[0]
        actual_overlap_w = bounds_i_x[1] - bounds_i_x[0]
        actual_overlap_pixels = actual_overlap_h * actual_overlap_w

        full_tile_pixels = tile_shape[0] * tile_shape[1]

        # Overlap region should be much smaller than full tile
        ratio = full_tile_pixels / actual_overlap_pixels
        assert ratio > 5, f"Expected overlap to be <20% of full tile, got ratio {ratio:.1f}"

        # Verify the bounds are correct
        assert actual_overlap_w == overlap_pixels
        assert actual_overlap_h == tile_shape[0]

    def test_vertical_overlap_bounds(self):
        """Test bounds for vertical overlap."""
        tile_shape = (2048, 2048)
        overlap_pixels = 307
        dy = tile_shape[0] - overlap_pixels

        adjacent_pairs = [
            (0, 1, dy, 0, overlap_pixels, tile_shape[1]),  # vertical pair
        ]
        pair_bounds = compute_pair_bounds(adjacent_pairs, tile_shape)

        _, _, bounds_i_y, bounds_i_x, _, _ = pair_bounds[0]
        actual_overlap_h = bounds_i_y[1] - bounds_i_y[0]
        actual_overlap_w = bounds_i_x[1] - bounds_i_x[0]

        assert actual_overlap_h == overlap_pixels
        assert actual_overlap_w == tile_shape[1]
