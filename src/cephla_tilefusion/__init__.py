"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D microscopy.

A Python library for stitching tiled microscopy images with support for
OME-TIFF, individual TIFF folders, and Zarr formats.

Based on the tilefusion module from opm-processing-v2:
https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py

Original author: Doug Shepherd (https://github.com/dpshepherd), QI2lab, Arizona State University
"""

from . import utils as _utils
from .core import TileFusion
from .utils import GPU_AVAILABLE, set_use_gpu
from .flatfield import (
    calculate_flatfield,
    apply_flatfield,
    apply_flatfield_region,
    save_flatfield,
    load_flatfield,
    HAS_BASICPY,
)


def is_using_gpu() -> bool:
    """Return ``True`` if the GPU backend is currently active."""
    return _utils.USING_GPU


def __getattr__(name):
    # Resolve ``USING_GPU`` lazily so callers always see the live setting,
    # even after :func:`set_use_gpu` has been used to switch backends.
    if name == "USING_GPU":
        return _utils.USING_GPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.1"
__all__ = [
    "TileFusion",
    "GPU_AVAILABLE",
    "USING_GPU",
    "is_using_gpu",
    "set_use_gpu",
    "__version__",
    "calculate_flatfield",
    "apply_flatfield",
    "apply_flatfield_region",
    "save_flatfield",
    "load_flatfield",
    "HAS_BASICPY",
]
