"""Tests for cephla_tilefusion.utils module."""

import numpy as np
import pytest

from cephla_tilefusion.utils import make_1d_profile, shift_array, compute_ssim, to_numpy


class TestMake1DProfile:
    """Tests for make_1d_profile function."""

    def test_no_blend(self):
        """Profile with no blending should be all ones."""
        prof = make_1d_profile(100, 0)
        assert prof.shape == (100,)
        assert np.allclose(prof, 1.0)
        assert prof.dtype == np.float32

    def test_blend_edges(self):
        """Profile should ramp at edges."""
        prof = make_1d_profile(100, 10)
        # Start should be 0
        assert prof[0] == 0.0
        # End should be close to 0
        assert prof[-1] < 0.2
        # Middle should be 1
        assert prof[50] == 1.0

    def test_blend_symmetry(self):
        """Profile should be symmetric."""
        prof = make_1d_profile(100, 20)
        assert np.allclose(prof[:20], prof[-20:][::-1])

    def test_blend_larger_than_half(self):
        """Blend larger than half length should be capped."""
        prof = make_1d_profile(20, 15)
        # Should cap at length // 2 = 10
        assert prof.shape == (20,)
        # Profile should still be valid with ramps at edges
        assert prof[0] < prof[10]  # Ramp up from start
        assert prof[-1] < prof[10]  # Ramp down to end

    def test_short_profile(self):
        """Short profile should still work."""
        prof = make_1d_profile(5, 2)
        assert prof.shape == (5,)


class TestShiftArray:
    """Tests for shift_array function."""

    def test_no_shift(self):
        """Zero shift should return similar array."""
        arr = np.random.rand(50, 50).astype(np.float32)
        shifted = shift_array(arr, [0, 0])
        assert shifted.shape == arr.shape
        # With order=1 interpolation, should be very close
        assert np.allclose(shifted, arr, atol=1e-5)

    def test_integer_shift(self):
        """Integer shift should translate array."""
        arr = np.zeros((50, 50), dtype=np.float32)
        arr[20:30, 20:30] = 1.0
        shifted = shift_array(arr, [5.0, 5.0])
        # Peak should move
        assert shifted[25, 25] > 0.5

    def test_preserves_dtype(self):
        """Output should be same dtype as input."""
        arr = np.random.rand(20, 20).astype(np.float32)
        shifted = shift_array(arr, [1.0, 1.0])
        assert shifted.dtype == np.float32


class TestComputeSSIM:
    """Tests for compute_ssim function."""

    def test_identical_images(self):
        """SSIM of identical images should be 1.0."""
        arr = np.random.rand(50, 50).astype(np.float32) * 255
        ssim = compute_ssim(arr, arr, win_size=7)
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_different_images(self):
        """SSIM of different images should be less than 1."""
        arr1 = np.random.rand(50, 50).astype(np.float32) * 255
        arr2 = np.random.rand(50, 50).astype(np.float32) * 255
        ssim = compute_ssim(arr1, arr2, win_size=7)
        assert ssim < 1.0

    def test_similar_images(self):
        """SSIM of similar images should be high."""
        arr1 = np.random.rand(50, 50).astype(np.float32) * 255
        arr2 = arr1 + np.random.rand(50, 50).astype(np.float32) * 10
        ssim = compute_ssim(arr1, arr2, win_size=7)
        assert ssim > 0.8


class TestToNumpy:
    """Tests for to_numpy function."""

    def test_numpy_passthrough(self):
        """Numpy array should pass through unchanged."""
        arr = np.array([1, 2, 3])
        result = to_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_list_conversion(self):
        """List should be converted to numpy."""
        lst = [1, 2, 3]
        result = to_numpy(lst)
        assert isinstance(result, np.ndarray)
