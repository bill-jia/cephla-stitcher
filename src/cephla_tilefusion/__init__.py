"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D microscopy.

A Python library for stitching tiled microscopy images with support for
OME-TIFF, individual TIFF folders, and Zarr formats.

Based on the tilefusion module from opm-processing-v2:
https://github.com/QI2lab/opm-processing-v2/blob/tilefusion2D/src/opm_processing/imageprocessing/tilefusion.py

Original author: Doug Shepherd (https://github.com/dpshepherd), QI2lab, Arizona State University
"""

from .core import TileFusion
from .utils import USING_GPU
from .flatfield import (
    calculate_flatfield,
    apply_flatfield,
    apply_flatfield_region,
    save_flatfield,
    load_flatfield,
    HAS_BASICPY,
)

__version__ = "0.1.1"
__all__ = [
    "TileFusion",
    "USING_GPU",
    "__version__",
    "calculate_flatfield",
    "apply_flatfield",
    "apply_flatfield_region",
    "save_flatfield",
    "load_flatfield",
    "HAS_BASICPY",
]
