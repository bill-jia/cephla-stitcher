"""
Shared utilities for tilefusion.

GPU/CPU detection, array operations, and helper functions.

The compute backend is selected at runtime. ``GPU_AVAILABLE`` reports whether
the GPU stack (cupy + cucim) imported successfully on this machine; when True
the backend defaults to GPU but callers can opt into CPU via
:func:`set_use_gpu`. ``USING_GPU`` reflects the *current* selection and is
mutated in place by ``set_use_gpu`` together with the dispatched bindings
(``xp``, ``match_histograms``, ``block_reduce``, ``phase_cross_correlation``).
Other modules that need to react to toggles should import this module
(``from . import utils``) and read ``utils.<name>`` rather than binding the
attribute at import time.
"""

import numpy as np

# CPU implementations are always available.
from skimage.exposure import match_histograms as _match_histograms_cpu
from skimage.measure import block_reduce as _block_reduce_cpu
from skimage.registration import phase_cross_correlation as _phase_cross_correlation_cpu
from scipy.ndimage import shift as _shift_cpu
from skimage.metrics import structural_similarity as _ssim_cpu

# GPU implementations are loaded lazily. If any import fails we fall back to
# CPU-only mode for the rest of the process.
try:
    import cupy as _cp
    from cupyx.scipy.ndimage import shift as _cp_shift
    from cucim.skimage.exposure import match_histograms as _match_histograms_gpu
    from cucim.skimage.measure import block_reduce as _block_reduce_gpu
    from cucim.skimage.registration import (
        phase_cross_correlation as _phase_cross_correlation_gpu,
    )
    from opm_processing.imageprocessing.ssim_cuda import (
        structural_similarity_cupy_sep_shared as _ssim_cuda,
    )

    GPU_AVAILABLE = True
except Exception:
    _cp = None
    _cp_shift = None
    _match_histograms_gpu = None
    _block_reduce_gpu = None
    _phase_cross_correlation_gpu = None
    _ssim_cuda = None
    GPU_AVAILABLE = False


# Public, mutable bindings. ``set_use_gpu`` rebinds these in-place.
cp = _cp  # cupy module (or None if GPU not available)
cp_shift = _cp_shift
USING_GPU = GPU_AVAILABLE
xp = _cp if GPU_AVAILABLE else np
match_histograms = _match_histograms_gpu if GPU_AVAILABLE else _match_histograms_cpu
block_reduce = _block_reduce_gpu if GPU_AVAILABLE else _block_reduce_cpu
phase_cross_correlation = (
    _phase_cross_correlation_gpu if GPU_AVAILABLE else _phase_cross_correlation_cpu
)


def set_use_gpu(use_gpu: bool) -> None:
    """Select the compute backend for subsequent operations.

    Parameters
    ----------
    use_gpu : bool
        If True, route registration/fusion through cupy + cucim. Raises
        ``RuntimeError`` if the GPU stack failed to import. If False, force
        the numpy/scipy/skimage CPU backend even when a GPU is available.

    Notes
    -----
    This rebinds module-level attributes (``xp``, ``USING_GPU``, and the
    dispatched skimage callables). Other modules that already imported these
    names with ``from .utils import ...`` will not pick up the change; they
    must instead access them via ``utils.<name>``.
    """
    global USING_GPU, xp, match_histograms, block_reduce, phase_cross_correlation

    if use_gpu and not GPU_AVAILABLE:
        raise RuntimeError(
            "GPU backend requested but GPU dependencies (cupy, cucim) are not "
            "installed or no CUDA device is available. Install the 'gpu' extra "
            "or pass use_gpu=False."
        )

    USING_GPU = bool(use_gpu)
    if USING_GPU:
        xp = _cp
        match_histograms = _match_histograms_gpu
        block_reduce = _block_reduce_gpu
        phase_cross_correlation = _phase_cross_correlation_gpu
    else:
        xp = np
        match_histograms = _match_histograms_cpu
        block_reduce = _block_reduce_cpu
        phase_cross_correlation = _phase_cross_correlation_cpu


def shift_array(arr, shift_vec):
    """Shift array using GPU if active, else CPU fallback."""
    if USING_GPU and cp_shift is not None:
        return cp_shift(arr, shift=shift_vec, order=1, prefilter=False)
    return _shift_cpu(arr, shift=shift_vec, order=1, prefilter=False)


def compute_ssim(arr1, arr2, win_size: int) -> float:
    """SSIM wrapper that routes to GPU kernel or CPU skimage."""
    if USING_GPU and _ssim_cuda is not None:
        return float(_ssim_cuda(arr1, arr2, win_size=win_size))
    arr1_np = np.asarray(arr1)
    arr2_np = np.asarray(arr2)
    data_range = float(arr1_np.max() - arr1_np.min())
    if data_range == 0:
        data_range = 1.0
    return float(_ssim_cpu(arr1_np, arr2_np, win_size=win_size, data_range=data_range))


def make_1d_profile(length: int, blend: int) -> np.ndarray:
    """
    Create a linear ramp profile over `blend` pixels at each end.

    Parameters
    ----------
    length : int
        Number of pixels.
    blend : int
        Ramp width.

    Returns
    -------
    prof : (length,) float32
        Linear profile.
    """
    blend = min(blend, length // 2)
    prof = np.ones(length, dtype=np.float32)
    if blend > 0:
        ramp = np.linspace(0, 1, blend, endpoint=False, dtype=np.float32)
        prof[:blend] = ramp
        prof[-blend:] = ramp[::-1]
    return prof


def to_numpy(arr):
    """Convert array to numpy, handling both CPU and GPU arrays."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Move array to current device (GPU if active, else CPU)."""
    return xp.asarray(arr)
