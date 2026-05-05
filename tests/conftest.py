"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from cephla_tilefusion import set_use_gpu

# Force the CPU backend for the entire test session. The unit tests pass plain
# numpy arrays to backend-dispatched helpers (shift_array, compute_ssim, ...),
# which would fail under the GPU backend because the GPU implementations
# expect cupy arrays. Forcing CPU also makes the suite deterministic on hosts
# with and without a GPU.
set_use_gpu(False)


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_tile(rng):
    """Generate a sample tile image."""
    return rng.random((100, 100), dtype=np.float32) * 65535


@pytest.fixture
def sample_multichannel_tile(rng):
    """Generate a sample multi-channel tile."""
    return rng.random((3, 100, 100), dtype=np.float32) * 65535
