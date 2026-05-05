"""
TileFusion - GPU/CPU-accelerated tile registration and fusion for 2D OME-TIFF stacks.

Main orchestration class that composes registration, fusion, optimization, and I/O modules.
"""

import gc
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorstore as ts
import tifffile
from tqdm import trange, tqdm

from .utils import (
    USING_GPU,
    block_reduce,
    cp,
    make_1d_profile,
    xp,
)
from .registration import (
    compute_pair_bounds,
    find_adjacent_pairs,
    register_and_score,
    register_pair_worker,
)
from .fusion import accumulate_tile_shard, normalize_shard
from .optimization import links_from_pairwise_metrics, solve_global, two_round_optimization
from .flatfield import apply_flatfield, apply_flatfield_region
from .io import (
    load_ome_tiff_metadata,
    load_individual_tiffs_metadata,
    load_ome_tiff_tiles_metadata,
    load_zarr_metadata,
    load_ngff_ome_zarr_metadata,
    read_ome_tiff_tile,
    read_ome_tiff_region,
    read_individual_tiffs_tile,
    read_individual_tiffs_region,
    read_ome_tiff_tiles_tile,
    read_ome_tiff_tiles_region,
    read_zarr_tile,
    read_zarr_region,
    read_ngff_ome_zarr_tile,
    read_ngff_ome_zarr_region,
    is_ngff_ome_zarr,
    create_zarr_store,
    write_ngff_metadata,
    write_scale_group_metadata,
)


class TileFusion:
    """
    GPU/CPU-accelerated tile registration and fusion for 2D OME-TIFF stacks.

    Parameters
    ----------
    tiff_path : str or Path
        Path to the OME-TIFF file containing tiled images with stage positions.
    output_path : str or Path, optional
        Output path for fused Zarr. If None, derived from input path.
    blend_pixels : tuple of int
        Feather widths (by, bx).
    downsample_factors : tuple of int
        Block-reduce factors for registration.
    ssim_window : int
        Window size for SSIM.
    threshold : float
        SSIM acceptance threshold.
    multiscale_factors : sequence of int
        Downsampling factors for multiscale.
    resolution_multiples : sequence
        Resolution multipliers per scale level.
    max_workers : int
        Maximum parallel I/O workers.
    debug : bool
        If True, prints debug info.
    metrics_filename : str
        Filename for storing registration metrics.
    channel_to_use : int
        Channel index for registration.
    multiscale_downsample : str
        Either "stride" (default) or "block_mean" to control multiscale reduction.
    """

    def __init__(
        self,
        tiff_path: Union[str, Path],
        output_path: Union[str, Path] = None,
        blend_pixels: Tuple[int, int] = (0, 0),
        downsample_factors: Tuple[int, int] = (1, 1),
        ssim_window: int = 15,
        threshold: float = 0.5,
        multiscale_factors: Sequence[int] = (2, 4, 8, 16),
        resolution_multiples: Sequence[Union[int, Sequence[int]]] = (
            (1, 1),
            (2, 2),
            (4, 4),
            (8, 8),
            (16, 16),
        ),
        max_workers: int = 8,
        debug: bool = False,
        metrics_filename: str = "metrics.json",
        channel_to_use: int = 0,
        multiscale_downsample: str = "stride",
        region: Optional[str] = None,
        flatfield: Optional[np.ndarray] = None,
        darkfield: Optional[np.ndarray] = None,
        registration_z: Optional[int] = None,
        registration_t: int = 0,
    ):
        self.tiff_path = Path(tiff_path)
        if not self.tiff_path.exists():
            raise FileNotFoundError(f"Path not found: {self.tiff_path}")

        self.output_path = (
            Path(output_path)
            if output_path
            else self.tiff_path.parent / f"{self.tiff_path.stem}_fused.ome.zarr"
        )

        # Detect and load format
        self._is_zarr_format = False
        self._is_ngff_ome_zarr_format = False
        self._is_individual_tiffs_format = False
        self._is_ome_tiff_tiles_format = False
        self._metadata = {}

        if self.tiff_path.is_dir():
            # Check for ome_tiff tiles format first (ome_tiff/ folder with .ome.tiff files)
            ome_tiff_folder = self.tiff_path / "ome_tiff"
            if ome_tiff_folder.exists() and list(ome_tiff_folder.glob("*.ome.tiff"))[:1]:
                self._is_ome_tiff_tiles_format = True
                self._metadata = load_ome_tiff_tiles_metadata(self.tiff_path)
            else:
                zarr_json = self.tiff_path / "zarr.json"
                if zarr_json.exists():
                    with open(zarr_json) as f:
                        meta = json.load(f)
                    attrs = meta.get("attributes", {})
                    if "per_index_metadata" in attrs:
                        self._is_zarr_format = True
                        self._metadata = load_zarr_metadata(self.tiff_path)
                    elif is_ngff_ome_zarr(self.tiff_path):
                        self._is_ngff_ome_zarr_format = True
                        self._metadata = load_ngff_ome_zarr_metadata(self.tiff_path)
                    else:
                        self._is_individual_tiffs_format = True
                        self._metadata = load_individual_tiffs_metadata(self.tiff_path)
                elif is_ngff_ome_zarr(self.tiff_path):
                    self._is_ngff_ome_zarr_format = True
                    self._metadata = load_ngff_ome_zarr_metadata(self.tiff_path)
                else:
                    self._is_individual_tiffs_format = True
                    self._metadata = load_individual_tiffs_metadata(self.tiff_path)
        else:
            self._metadata = load_ome_tiff_metadata(self.tiff_path)
            # Close the metadata handle immediately - we use thread-local handles
            # for thread-safe concurrent reads instead of sharing this handle.
            if "tiff_handle" in self._metadata:
                self._metadata.pop("tiff_handle").close()

        # Extract common properties
        self.n_tiles = self._metadata["n_tiles"]
        self.n_series = self._metadata["n_series"]
        self.Y, self.X = self._metadata["shape"]
        self.channels = self._metadata["channels"]
        self.time_dim = self._metadata.get("time_dim", 1)
        self.position_dim = self._metadata.get("position_dim", self.n_tiles)
        self._pixel_size = self._metadata["pixel_size"]
        self._tile_positions = self._metadata["tile_positions"]
        self._tile_identifiers = self._metadata.get("tile_identifiers", [])
        self._unique_regions = self._metadata.get("unique_regions", [])

        # Snapshot the unfiltered tile listing so set_region() can re-filter
        # without re-walking the dataset (important for HCS plates where
        # discovery is the dominant cost).
        self._all_tile_positions = list(self._tile_positions)
        self._all_tile_identifiers = list(self._tile_identifiers)
        self._region = None
        # Region filter (if requested) is applied at the end of __init__ once
        # registration/fusion state has been initialized.

        # Z-stack and time series properties
        self.n_z = self._metadata.get("n_z", 1)
        self.n_t = self._metadata.get("n_t", 1)
        self.dz_um = self._metadata.get("dz_um", 1.0)
        self._time_folders = self._metadata.get("time_folders", None)
        self._middle_z = self.n_z // 2  # Use middle z-level for registration

        # Registration z/t selection (validate after n_z/n_t are known)
        if registration_z is None:
            self._registration_z = self._middle_z
        else:
            if registration_z < 0 or registration_z >= self.n_z:
                raise ValueError(f"registration_z={registration_z} out of range [0, {self.n_z})")
            self._registration_z = registration_z

        if registration_t < 0 or registration_t >= self.n_t:
            raise ValueError(f"registration_t={registration_t} out of range [0, {self.n_t})")
        self._registration_t = registration_t

        # Configuration
        self.downsample_factors = tuple(downsample_factors)
        self.ssim_window = int(ssim_window)
        self.threshold = float(threshold)
        self.multiscale_factors = tuple(multiscale_factors)
        self.resolution_multiples = [
            r if hasattr(r, "__len__") else (r, r) for r in resolution_multiples
        ]
        self._max_workers = int(max_workers)
        self._debug = bool(debug)
        self.metrics_filename = metrics_filename
        self._blend_pixels = tuple(blend_pixels)
        self.channel_to_use = channel_to_use

        if multiscale_downsample not in ("stride", "block_mean"):
            raise ValueError('multiscale_downsample must be "stride" or "block_mean".')
        self.multiscale_downsample = multiscale_downsample

        self._update_profiles()
        self.chunk_shape = (1, 1024, 1024)
        self.chunk_y, self.chunk_x = self.chunk_shape[-2:]

        # State
        self.pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, float]] = {}
        self.global_offsets: Optional[np.ndarray] = None
        self.offset: Optional[Tuple[float, float]] = None
        self.unpadded_shape: Optional[Tuple[int, int]] = None
        self.padded_shape: Optional[Tuple[int, int]] = None
        self.pad = (0, 0)
        self.fused_ts = None
        self.center = None

        # Flatfield correction (optional)
        self._flatfield = flatfield  # Shape (C, Y, X) or None
        self._darkfield = darkfield  # Shape (C, Y, X) or None

        # Validate flatfield/darkfield shapes match tile dimensions
        expected_shape = (self.channels, self.Y, self.X)
        if flatfield is not None and flatfield.shape != expected_shape:
            raise ValueError(
                f"flatfield.shape {flatfield.shape} does not match expected "
                f"tile shape {expected_shape} (channels, Y, X)"
            )
        if darkfield is not None and darkfield.shape != expected_shape:
            raise ValueError(
                f"darkfield.shape {darkfield.shape} does not match expected "
                f"tile shape {expected_shape} (channels, Y, X)"
            )

        # Thread-local storage for TiffFile handles (thread-safe concurrent access)
        self._thread_local = threading.local()
        self._handles_lock = threading.Lock()
        self._all_handles: List[tifffile.TiffFile] = []

        # Apply region filter now that state attributes are all initialized.
        # Formats without tile identifiers (e.g. plain OME-TIFF) silently skip
        # the filter for backward compatibility.
        if region is not None and self._all_tile_identifiers:
            self.set_region(region)

    def set_region(self, region: Optional[str]) -> None:
        """Re-filter active tiles to a single region without reloading metadata.

        Lets a caller stitch many regions of an HCS plate without paying the
        per-region discovery cost (which scans every tile in the dataset).
        Pass ``None`` to clear the filter and expose all tiles again.

        Resets registration and fusion state (``pairwise_metrics``,
        ``global_offsets``, fused output, etc.) since they are region-specific.
        Callers that want to reuse an instance across regions should also
        update ``output_path`` and ``metrics_filename`` between regions.
        """
        if region is None:
            positions = list(self._all_tile_positions)
            identifiers = list(self._all_tile_identifiers)
        else:
            if not self._all_tile_identifiers:
                raise ValueError(
                    "Cannot filter by region: dataset has no tile identifiers"
                )
            positions = []
            identifiers = []
            for pos, tile_id in zip(self._all_tile_positions, self._all_tile_identifiers):
                if len(tile_id) >= 2 and tile_id[0] == region:
                    positions.append(pos)
                    identifiers.append(tile_id)
            if not positions:
                raise ValueError(f"No tiles found for region '{region}'")

        self._region = region
        self._tile_positions = positions
        self._tile_identifiers = identifiers
        self.n_tiles = len(positions)
        self.n_series = self.n_tiles
        self.position_dim = self.n_tiles
        self._metadata["tile_positions"] = positions
        self._metadata["tile_identifiers"] = identifiers
        self._metadata["n_tiles"] = self.n_tiles

        # Reset per-region registration/fusion state.
        self.pairwise_metrics = {}
        self.global_offsets = None
        self.offset = None
        self.unpadded_shape = None
        self.padded_shape = None
        self.pad = (0, 0)
        self.fused_ts = None
        self.center = None

    def close(self) -> None:
        """
        Close any open file handles to release resources.

        This should be called when finished using a TileFusion instance,
        or use it as a context manager (``with TileFusion(...) as tf:``)
        for automatic cleanup. Important for OME-TIFF inputs where file
        handles are kept open for performance.

        Warning
        -------
        Only call this method when all read operations are complete. Calling
        ``close()`` while other threads are still reading tiles will close
        their handles mid-operation, causing errors.
        """
        # THREAD SAFETY NOTE:
        # This method is NOT safe to call while other threads are actively reading.
        # The design assumes close() is called only after all work is complete.
        #
        # Race condition scenario:
        #   1. Thread A calls _get_thread_local_handle(), gets handle
        #   2. Main thread calls close(), closes all handles
        #   3. Thread A calls handle.series[idx].asarray() -> ERROR (closed file)
        #
        # We chose documentation over a complex fix (reference counting, read-write
        # locks) because:
        #   - The context manager pattern naturally prevents this issue
        #   - Adding synchronization would hurt performance for the common case
        #   - Users explicitly calling close() should know their threads are done
        #
        # Safe usage patterns:
        #   - Use context manager: with TileFusion(...) as tf: ...
        #   - Call close() only after ThreadPoolExecutor.shutdown(wait=True)
        #   - Use single-threaded access when manually managing lifecycle

        # Close all thread-local handles
        with self._handles_lock:
            for handle in self._all_handles:
                try:
                    handle.close()
                except (OSError, AttributeError):
                    pass  # Best-effort cleanup: handle may be invalid or already closed
            self._all_handles.clear()

        # Reset thread-local storage so future calls to _get_thread_local_handle()
        # will create new handles. Note: This only affects threads that access
        # self._thread_local AFTER this point. Threads that cached a handle reference
        # before close() was called will still have stale (closed) handles, but
        # _get_thread_local_handle() now checks for closed handles and creates new ones.
        self._thread_local = threading.local()

    def __enter__(self) -> "TileFusion":
        """Enter the runtime context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context and close file handles."""
        try:
            self.close()
        except Exception:
            # If there was no exception in the with-block, propagate the close() failure.
            # If there was an original exception, suppress close() errors so we don't mask it.
            if exc_type is None:
                raise

    def __del__(self):
        """
        Destructor to ensure file handles are closed.

        Note: This is a fallback safety net only. Python does not guarantee
        when (or if) __del__ is called. Always prefer using the context
        manager protocol (``with TileFusion(...) as tf:``) or explicitly
        calling ``close()`` for reliable resource cleanup.
        """
        try:
            self.close()
        except (OSError, AttributeError, TypeError):
            pass  # Object may be partially initialized, or close() may fail during shutdown

    def _get_thread_local_handle(self) -> Optional[tifffile.TiffFile]:
        """
        Get or create a thread-local TiffFile handle for the current thread.

        Each thread gets its own file handle to ensure thread-safe concurrent
        reads. This avoids race conditions that can occur when multiple threads
        share a single file descriptor (seek + read is not atomic on Windows).

        Returns
        -------
        tifffile.TiffFile or None
            Thread-local handle for OME-TIFF files, None for other formats.
        """
        # Only applies to OME-TIFF format (not zarr, individual tiffs, etc.)
        if (
            self._is_zarr_format
            or self._is_ngff_ome_zarr_format
            or self._is_individual_tiffs_format
            or self._is_ome_tiff_tiles_format
        ):
            return None

        # Check if this thread already has a valid (open) handle.
        # NOTE: There is a race condition between this check and using the handle -
        # another thread could call close() after validation but before the handle
        # is used. This is documented behavior; callers must ensure close() is only
        # called after all read operations complete.
        if hasattr(self._thread_local, "tiff_handle"):
            handle = self._thread_local.tiff_handle
            # Verify handle exists and is not closed.
            # We check filehandle.closed which is a reliable indicator.
            if (
                handle is not None
                and handle.filehandle is not None
                and not handle.filehandle.closed
            ):
                return handle
            # Handle was closed or invalid - will create a new one below

        # Create a new handle for this thread
        handle = tifffile.TiffFile(self.tiff_path)
        self._thread_local.tiff_handle = handle

        # Track for cleanup
        with self._handles_lock:
            self._all_handles.append(handle)

        return handle

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def tile_positions(self) -> List[Tuple[float, float]]:
        """Stage positions for each tile (y, x)."""
        return self._tile_positions

    @tile_positions.setter
    def tile_positions(self, positions: Sequence[Tuple[float, float]]):
        if any(len(p) != 2 for p in positions):
            raise ValueError("Each position must be a 2-tuple.")
        self._tile_positions = [tuple(p) for p in positions]

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Pixel size in (y, x)."""
        return self._pixel_size

    @pixel_size.setter
    def pixel_size(self, size: Tuple[float, float]):
        if len(size) != 2:
            raise ValueError("pixel_size must be a 2-tuple.")
        self._pixel_size = tuple(float(x) for x in size)

    @property
    def blend_pixels(self) -> Tuple[int, int]:
        """Feather widths in (by, bx)."""
        return self._blend_pixels

    @blend_pixels.setter
    def blend_pixels(self, bp: Tuple[int, int]):
        if len(bp) != 2:
            raise ValueError("blend_pixels must be a 2-tuple.")
        self._blend_pixels = tuple(bp)
        self._update_profiles()

    @property
    def max_workers(self) -> int:
        """Maximum concurrent I/O workers."""
        return self._max_workers

    @max_workers.setter
    def max_workers(self, mw: int):
        if mw < 1:
            raise ValueError("max_workers must be >= 1.")
        self._max_workers = int(mw)

    @property
    def debug(self) -> bool:
        """Debug flag for verbose logging."""
        return self._debug

    @debug.setter
    def debug(self, flag: bool):
        self._debug = bool(flag)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _update_profiles(self) -> None:
        """Recompute 1D feather profiles from blend_pixels."""
        by, bx = self._blend_pixels
        self.y_profile = make_1d_profile(self.Y, by)
        self.x_profile = make_1d_profile(self.X, bx)

    # -------------------------------------------------------------------------
    # I/O methods (delegate to format-specific loaders)
    # -------------------------------------------------------------------------

    def _read_tile(self, tile_idx: int, z_level: int = None, time_idx: int = None) -> np.ndarray:
        """Read a single tile from the input data (all channels)."""
        if z_level is None:
            z_level = self._registration_z  # Default to registration z-level
        if time_idx is None:
            time_idx = self._registration_t  # Default to registration timepoint

        if self._is_zarr_format:
            zarr_ts = self._metadata["tensorstore"]
            is_3d = self._metadata.get("is_3d", False)
            tile = read_zarr_tile(zarr_ts, tile_idx, is_3d)
        elif self._is_ngff_ome_zarr_format:
            tile = read_ngff_ome_zarr_tile(
                self._metadata["tile_stores"],
                self._metadata["tile_identifiers"],
                self._metadata["tile_axes"],
                tile_idx,
                z_level=z_level,
                time_idx=time_idx,
            )
        elif self._is_individual_tiffs_format:
            tile = read_individual_tiffs_tile(
                self._metadata["image_folder"],
                self._metadata["channel_names"],
                self._metadata["tile_identifiers"],
                tile_idx,
                z_level=z_level,
                time_idx=time_idx,
                time_folders=self._time_folders,
            )
        elif self._is_ome_tiff_tiles_format:
            tile = read_ome_tiff_tiles_tile(
                self._metadata["ome_tiff_folder"],
                self._metadata["tile_identifiers"],
                self._metadata["tile_file_map"],
                tile_idx,
                self._metadata["axes"],
                z_level=z_level,
                time_idx=time_idx,
            )
        else:
            # Use thread-local handle for thread-safe concurrent reads
            handle = self._get_thread_local_handle()
            tile = read_ome_tiff_tile(self.tiff_path, tile_idx, handle)

        # Apply flatfield correction if enabled
        if self._flatfield is not None:
            tile = apply_flatfield(tile, self._flatfield, self._darkfield)

        return tile

    def _read_tile_region(
        self,
        tile_idx: int,
        y_slice: slice,
        x_slice: slice,
        z_level: int = None,
        time_idx: int = None,
    ) -> np.ndarray:
        """Read a region of a tile from the input data."""
        if z_level is None:
            z_level = self._registration_z  # Default to registration z-level
        if time_idx is None:
            time_idx = self._registration_t  # Default to registration timepoint

        if self._is_zarr_format:
            zarr_ts = self._metadata["tensorstore"]
            is_3d = self._metadata.get("is_3d", False)
            region = read_zarr_region(
                zarr_ts, tile_idx, y_slice, x_slice, self.channel_to_use, is_3d
            )
        elif self._is_ngff_ome_zarr_format:
            region = read_ngff_ome_zarr_region(
                self._metadata["tile_stores"],
                self._metadata["tile_identifiers"],
                self._metadata["tile_axes"],
                tile_idx,
                y_slice,
                x_slice,
                channel_idx=self.channel_to_use,
                z_level=z_level,
                time_idx=time_idx,
            )
        elif self._is_individual_tiffs_format:
            region = read_individual_tiffs_region(
                self._metadata["image_folder"],
                self._metadata["channel_names"],
                self._metadata["tile_identifiers"],
                tile_idx,
                y_slice,
                x_slice,
                self.channel_to_use,
                z_level=z_level,
                time_idx=time_idx,
                time_folders=self._time_folders,
            )
        elif self._is_ome_tiff_tiles_format:
            region = read_ome_tiff_tiles_region(
                self._metadata["ome_tiff_folder"],
                self._metadata["tile_identifiers"],
                self._metadata["tile_file_map"],
                tile_idx,
                self._metadata["axes"],
                y_slice,
                x_slice,
                self.channel_to_use,
                z_level=z_level,
                time_idx=time_idx,
            )
        else:
            # Use thread-local handle for thread-safe concurrent reads
            handle = self._get_thread_local_handle()
            region = read_ome_tiff_region(self.tiff_path, tile_idx, y_slice, x_slice, handle)

        # Apply flatfield correction if enabled
        if self._flatfield is not None:
            region = apply_flatfield_region(
                region,
                self._flatfield,
                self._darkfield,
                y_slice,
                x_slice,
                channel_idx=self.channel_to_use,
            )

        return region

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    def refine_tile_positions_with_cross_correlation(
        self,
        downsample_factors: Tuple[int, int] = None,
        ssim_window: int = None,
        ch_idx: int = 0,
        threshold: float = None,
        parallel: Optional[bool] = None,
    ) -> None:
        """
        Detect and score overlaps between neighboring tile pairs via cross-correlation.

        Parameters
        ----------
        parallel : bool, optional
            If None (default), auto-detects: enabled for multi-file formats
            (Zarr, individual TIFFs, OME-TIFF tiles), disabled for single-file
            OME-TIFF (due to I/O contention).
        """
        df = downsample_factors or self.downsample_factors
        sw = ssim_window or self.ssim_window
        th = threshold if threshold is not None else self.threshold
        self.pairwise_metrics.clear()

        max_shift = (100, 100)

        # Find adjacent pairs
        adjacent_pairs = find_adjacent_pairs(
            self._tile_positions, self._pixel_size, (self.Y, self.X)
        )

        if self._debug:
            print(f"Found {len(adjacent_pairs)} adjacent tile pairs to register")

        # Compute bounds
        pair_bounds = compute_pair_bounds(adjacent_pairs, (self.Y, self.X))

        # Auto-detect parallel mode if not specified
        if parallel is None:
            # Parallel helps for individual TIFFs (separate files)
            # but hurts for single-file OME-TIFF (I/O contention)
            is_multi_file = (
                self._is_zarr_format
                or self._is_individual_tiffs_format
                or self._is_ome_tiff_tiles_format
            )
            parallel = is_multi_file

        # Use parallel processing for CPU mode with enough pairs
        use_parallel = parallel and not USING_GPU and len(pair_bounds) > 4

        if use_parallel:
            self._register_parallel(pair_bounds, df, sw, th, max_shift)
        else:
            self._register_sequential(pair_bounds, df, sw, th, max_shift)

    def _register_parallel(
        self,
        pair_bounds: List[Tuple],
        df: Tuple[int, int],
        sw: int,
        th: float,
        max_shift: Tuple[int, int],
    ) -> None:
        """Register tile pairs using parallel I/O and compute.

        Uses batching only when estimated memory exceeds 30% of available RAM.
        """
        import psutil

        n_pairs = len(pair_bounds)
        n_workers = min(cpu_count(), n_pairs, self._max_workers)
        io_workers = min(n_pairs, self._max_workers)
        print(
            f"Parallel registration: {n_pairs} pairs, {n_workers} compute workers, {io_workers} I/O workers"
        )

        # Estimate memory needed based on actual overlap size
        if pair_bounds:
            total_pixels = 0
            for _, _, bounds_i_y, bounds_i_x, _, _ in pair_bounds:
                patch_h = bounds_i_y[1] - bounds_i_y[0]
                patch_w = bounds_i_x[1] - bounds_i_x[0]
                total_pixels += patch_h * patch_w
            # 4 bytes per float32 pixel, 2 patches per pair
            estimated_memory = total_pixels * 4 * 2
        else:
            estimated_memory = 0

        available_ram = psutil.virtual_memory().available
        ram_budget = int(available_ram * 0.3)
        needs_batching = estimated_memory > ram_budget

        def read_pair_patches(args):
            i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x = args
            try:
                patch_i = self._read_tile_region(
                    i_pos, slice(bounds_i_y[0], bounds_i_y[1]), slice(bounds_i_x[0], bounds_i_x[1])
                )
                patch_j = self._read_tile_region(
                    j_pos, slice(bounds_j_y[0], bounds_j_y[1]), slice(bounds_j_x[0], bounds_j_x[1])
                )
                return (i_pos, j_pos, patch_i, patch_j)
            except Exception:
                return (i_pos, j_pos, None, None)

        if needs_batching:
            # Batched approach for large datasets
            avg_pair_bytes = max(1, estimated_memory // n_pairs) if n_pairs > 0 else 1
            batch_size = max(16, ram_budget // avg_pair_bytes)
            n_batches = (n_pairs + batch_size - 1) // batch_size

            print(
                f"Processing {n_pairs} pairs in {n_batches} batches (RAM limited, {n_workers} workers)"
            )

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_pairs)
                batch = pair_bounds[start:end]

                with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
                    patches = list(io_executor.map(read_pair_patches, batch))

                work_items = [
                    (i, j, pi, pj, df, sw, th, max_shift)
                    for i, j, pi, pj in patches
                    if pi is not None
                ]

                desc = f"register {batch_idx+1}/{n_batches}"
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    results = list(
                        tqdm(
                            executor.map(register_pair_worker, work_items),
                            total=len(work_items),
                            desc=desc,
                            leave=True,
                        )
                    )

                for i_pos, j_pos, dy_s, dx_s, score in results:
                    if dy_s is not None:
                        self.pairwise_metrics[(i_pos, j_pos)] = (dy_s, dx_s, score)

                del patches, work_items, results
                gc.collect()
        else:
            # Simple approach - load all at once
            with ThreadPoolExecutor(max_workers=io_workers) as io_executor:
                patches = list(io_executor.map(read_pair_patches, pair_bounds))

            work_items = [
                (i, j, pi, pj, df, sw, th, max_shift) for i, j, pi, pj in patches if pi is not None
            ]

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(
                    tqdm(
                        executor.map(register_pair_worker, work_items),
                        total=len(work_items),
                        desc="register",
                        leave=True,
                    )
                )

            for i_pos, j_pos, dy_s, dx_s, score in results:
                if dy_s is not None:
                    self.pairwise_metrics[(i_pos, j_pos)] = (dy_s, dx_s, score)

            del patches, work_items, results
            gc.collect()

    def _register_sequential(
        self,
        pair_bounds: List[Tuple],
        df: Tuple[int, int],
        sw: int,
        th: float,
        max_shift: Tuple[int, int],
    ) -> None:
        """Register tile pairs sequentially (GPU mode or small datasets)."""
        io_executor = ThreadPoolExecutor(max_workers=2)

        for i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x in tqdm(
            pair_bounds, desc="register", leave=True
        ):

            def read_patch(idx, y_bounds, x_bounds):
                return self._read_tile_region(
                    idx, slice(y_bounds[0], y_bounds[1]), slice(x_bounds[0], x_bounds[1])
                )

            try:
                future_i = io_executor.submit(read_patch, i_pos, bounds_i_y, bounds_i_x)
                future_j = io_executor.submit(read_patch, j_pos, bounds_j_y, bounds_j_x)
                patch_i = future_i.result()
                patch_j = future_j.result()
            except Exception as e:
                if self._debug:
                    print(f"Error reading patches for ({i_pos}, {j_pos}): {e}")
                continue

            arr_i = xp.asarray(patch_i)
            arr_j = xp.asarray(patch_j)

            reduce_block = (1, df[0], df[1]) if arr_i.ndim == 3 else tuple(df)
            g1 = block_reduce(arr_i, reduce_block, xp.mean)
            g2 = block_reduce(arr_j, reduce_block, xp.mean)

            try:
                shift_ds, ssim_val = register_and_score(g1, g2, win_size=sw, debug=self._debug)
            except Exception as e:
                if self._debug:
                    print(f"Registration failed for ({i_pos}, {j_pos}): {e}")
                continue

            if shift_ds is None:
                continue
            score = float(max(ssim_val, 1e-6))
            if th != 0.0 and score < th:
                continue

            dy_s, dx_s = [int(np.round(shift_ds[k] * df[k])) for k in range(2)]

            if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
                if self._debug:
                    print(f"Dropping link {(i_pos, j_pos)} shift=({dy_s}, {dx_s})")
                continue

            self.pairwise_metrics[(i_pos, j_pos)] = (dy_s, dx_s, round(score, 3))

        io_executor.shutdown(wait=True)

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    def optimize_shifts(
        self,
        method: str = "ONE_ROUND",
        rel_thresh: float = 0.3,
        abs_thresh: float = 5.0,
        iterative: bool = False,
    ) -> None:
        """
        Globally optimize tile shifts.

        Parameters
        ----------
        method : {'ONE_ROUND', 'TWO_ROUND_SIMPLE', 'TWO_ROUND_ITERATIVE'}
        rel_thresh : float
            Relative threshold for link removal.
        abs_thresh : float
            Absolute threshold for link removal.
        iterative : bool
            If True, repeat outlier removal until convergence.
        """
        links = links_from_pairwise_metrics(self.pairwise_metrics)
        if not links:
            self.global_offsets = np.zeros((self.position_dim, 2), dtype=np.float64)
            return

        n = len(self._tile_positions)
        fixed = [0]

        if method == "ONE_ROUND":
            d_opt = solve_global(links, n, fixed)
        elif method.startswith("TWO_ROUND"):
            d_opt = two_round_optimization(
                links, n, fixed, rel_thresh, abs_thresh, method.endswith("ITERATIVE")
            )
        else:
            raise ValueError(f"Unknown method {method}")

        self.global_offsets = d_opt

    # -------------------------------------------------------------------------
    # Metrics persistence
    # -------------------------------------------------------------------------

    def save_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """Save pairwise_metrics to a JSON file."""
        path = Path(filepath)
        out = {f"{i},{j}": list(v) for (i, j), v in self.pairwise_metrics.items()}
        with open(path, "w") as f:
            json.dump(out, f)

    def load_pairwise_metrics(self, filepath: Union[str, Path]) -> None:
        """Load pairwise_metrics from a JSON file."""
        path = Path(filepath)
        with open(path, "r") as f:
            data = json.load(f)
        self.pairwise_metrics = {tuple(map(int, k.split(","))): tuple(v) for k, v in data.items()}

    # -------------------------------------------------------------------------
    # Fused image space
    # -------------------------------------------------------------------------

    def _compute_fused_image_space(self) -> None:
        """Compute fused image physical shape and offset based on tile positions."""
        pos = np.array(self._tile_positions)
        min_y, min_x = pos.min(axis=0)
        max_y = pos[:, 0].max() + self.Y * self._pixel_size[0]
        max_x = pos[:, 1].max() + self.X * self._pixel_size[1]

        sy = int(np.ceil((max_y - min_y) / self._pixel_size[0]))
        sx = int(np.ceil((max_x - min_x) / self._pixel_size[1]))

        self.unpadded_shape = (sy, sx)
        self.offset = (min_y, min_x)
        self.center = ((max_x - min_x) / 2, (max_y - min_y) / 2)

    def _pad_to_chunk_multiple(self) -> None:
        """Pad unpadded_shape to exact multiples of chunk shape."""
        ty, tx = self.chunk_y, self.chunk_x
        sy, sx = self.unpadded_shape

        py = (-sy) % ty
        px = (-sx) % tx

        self.pad = (py, px)
        self.padded_shape = (sy + py, sx + px)

    def _create_fused_tensorstore(self, output_path: Union[str, Path]) -> None:
        """Create the output Zarr v3 store for the fused image."""
        out = Path(output_path)
        # 5D shape: (T, C, Z, Y, X)
        full_shape = [self.n_t, self.channels, self.n_z, *self.padded_shape]
        shard_chunk = [1, 1, 1, self.chunk_y * 2, self.chunk_x * 2]
        codec_chunk = [1, 1, 1, self.chunk_y, self.chunk_x]
        self.shard_chunk = shard_chunk

        self.fused_ts = create_zarr_store(
            out, tuple(full_shape), tuple(codec_chunk), tuple(shard_chunk), self.max_workers
        )

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------

    def _fuse_tiles(
        self, mode: str = "blended", chunked: bool = True, ram_fraction: float = 0.4
    ) -> None:
        """Fuse all tiles into output, looping over z-levels and time points."""
        total_planes = self.n_t * self.n_z
        plane_idx = 0

        for t in range(self.n_t):
            for z in range(self.n_z):
                plane_idx += 1
                if total_planes > 1:
                    print(f"Fusing plane {plane_idx}/{total_planes} (t={t}, z={z})...")

                if mode == "direct":
                    self._fuse_tiles_direct_plane(z_level=z, time_idx=t)
                elif chunked:
                    self._fuse_tiles_chunked_plane(z_level=z, time_idx=t, ram_fraction=ram_fraction)
                else:
                    self._fuse_tiles_full_plane(z_level=z, time_idx=t)

    def _fuse_tiles_direct_plane(self, z_level: int = 0, time_idx: int = 0) -> None:
        """Fuse tiles using direct placement for a single z/t plane."""
        import psutil

        offsets = [
            (
                int((y - self.offset[0]) / self._pixel_size[0]),
                int((x - self.offset[1]) / self._pixel_size[1]),
            )
            for (y, x) in self._tile_positions
        ]
        pad_Y, pad_X = self.padded_shape

        available_ram = psutil.virtual_memory().available
        output_bytes = pad_Y * pad_X * self.channels * 2
        use_memory = output_bytes < 0.45 * available_ram

        show_progress = self.n_t == 1 and self.n_z == 1  # Only show progress for single plane

        if use_memory:
            if show_progress:
                print(f"Direct mode: using in-memory buffer ({output_bytes / 1e9:.2f} GB)")
            output = np.zeros((1, self.channels, 1, pad_Y, pad_X), dtype=np.uint16)

            iterator = (
                trange(len(offsets), desc="placing tiles", leave=True)
                if show_progress
                else range(len(offsets))
            )
            for t_idx in iterator:
                oy, ox = offsets[t_idx]
                tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

                y_end = min(oy + self.Y, pad_Y)
                x_end = min(ox + self.X, pad_X)
                tile_h = y_end - oy
                tile_w = x_end - ox

                if tile_h > 0 and tile_w > 0:
                    output[0, :, 0, oy:y_end, ox:x_end] = tile_all[:, :tile_h, :tile_w]

            if show_progress:
                print("Writing to disk...")
            self.fused_ts[time_idx : time_idx + 1, :, z_level : z_level + 1, :, :].write(
                output
            ).result()
            del output
        else:
            if show_progress:
                print(f"Direct mode: writing directly to disk")
            iterator = (
                trange(len(offsets), desc="placing tiles", leave=True)
                if show_progress
                else range(len(offsets))
            )
            for t_idx in iterator:
                oy, ox = offsets[t_idx]
                tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

                y_end = min(oy + self.Y, pad_Y)
                x_end = min(ox + self.X, pad_X)
                tile_h = y_end - oy
                tile_w = x_end - ox

                if tile_h > 0 and tile_w > 0:
                    tile_region = tile_all[:, :tile_h, :tile_w].astype(np.uint16)
                    # Shape: (1, C, 1, h, w)
                    self.fused_ts[
                        time_idx : time_idx + 1, :, z_level : z_level + 1, oy:y_end, ox:x_end
                    ].write(tile_region[np.newaxis, :, np.newaxis, :, :]).result()

        gc.collect()

    def _fuse_tiles_full_plane(self, z_level: int = 0, time_idx: int = 0) -> None:
        """Fuse all tiles using full-image accumulator for a single z/t plane."""
        offsets = [
            (
                int((y - self.offset[0]) / self._pixel_size[0]),
                int((x - self.offset[1]) / self._pixel_size[1]),
            )
            for (y, x) in self._tile_positions
        ]
        pad_Y, pad_X = self.padded_shape
        C = self.channels

        fused_block = np.zeros((C, pad_Y, pad_X), dtype=np.float32)
        weight_sum = np.zeros_like(fused_block)
        w2d = self.y_profile[:, None] * self.x_profile[None, :]

        show_progress = self.n_t == 1 and self.n_z == 1
        iterator = (
            trange(len(offsets), desc="fusing", leave=True)
            if show_progress
            else range(len(offsets))
        )

        for t_idx in iterator:
            oy, ox = offsets[t_idx]
            tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

            if tile_all.shape[0] == 1 and C > 1:
                tile_all = np.broadcast_to(tile_all, (C, tile_all.shape[1], tile_all.shape[2]))

            accumulate_tile_shard(fused_block, weight_sum, tile_all, w2d, oy, ox)

        normalize_shard(fused_block, weight_sum)
        # Write to 5D output: (T, C, Z, Y, X)
        self.fused_ts[time_idx, :, z_level, :pad_Y, :pad_X].write(
            fused_block.astype(np.uint16)
        ).result()

        del fused_block, weight_sum
        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _fuse_tiles_chunked_plane(
        self, z_level: int = 0, time_idx: int = 0, ram_fraction: float = 0.4
    ) -> None:
        """Fuse tiles using memory-efficient chunked processing for a single z/t plane."""
        import psutil

        available_ram = psutil.virtual_memory().available
        usable_ram = int(available_ram * ram_fraction)
        bytes_per_pixel = 4 * 2 * self.channels
        max_pixels = usable_ram // bytes_per_pixel
        block_size = int(np.sqrt(max_pixels))
        block_size = (block_size // self.chunk_y) * self.chunk_y
        block_size = max(block_size, self.chunk_y * 2)
        block_size = min(block_size, 10240)

        pad_Y, pad_X = self.padded_shape

        if block_size >= max(pad_Y, pad_X):
            if self.n_t == 1 and self.n_z == 1:
                print(f"Image fits in RAM budget, using full mode")
            return self._fuse_tiles_full_plane(z_level=z_level, time_idx=time_idx)

        show_progress = self.n_t == 1 and self.n_z == 1
        if show_progress:
            print(f"Using chunked mode: {block_size}x{block_size} blocks")

        tile_bounds = []
        for y, x in self._tile_positions:
            oy = int((y - self.offset[0]) / self._pixel_size[0])
            ox = int((x - self.offset[1]) / self._pixel_size[1])
            tile_bounds.append((oy, oy + self.Y, ox, ox + self.X))

        n_blocks_y = (pad_Y + block_size - 1) // block_size
        n_blocks_x = (pad_X + block_size - 1) // block_size
        total_blocks = n_blocks_y * n_blocks_x
        C = self.channels

        block_idx = 0
        for block_y in range(0, pad_Y, block_size):
            for block_x in range(0, pad_X, block_size):
                block_idx += 1
                by_end = min(block_y + block_size, pad_Y)
                bx_end = min(block_x + block_size, pad_X)
                bh, bw = by_end - block_y, bx_end - block_x

                overlapping = []
                for t_idx, (ty0, ty1, tx0, tx1) in enumerate(tile_bounds):
                    if ty1 > block_y and ty0 < by_end and tx1 > block_x and tx0 < bx_end:
                        overlapping.append(t_idx)

                if not overlapping:
                    continue

                fused_block = np.zeros((C, bh, bw), dtype=np.float32)
                weight_sum = np.zeros_like(fused_block)

                desc = f"block {block_idx}/{total_blocks}"
                iterator = (
                    tqdm(overlapping, desc=desc, leave=False) if show_progress else overlapping
                )
                for t_idx in iterator:
                    tile_all = self._read_tile(t_idx, z_level=z_level, time_idx=time_idx)

                    if tile_all.shape[0] == 1 and C > 1:
                        tile_all = np.broadcast_to(
                            tile_all, (C, tile_all.shape[1], tile_all.shape[2])
                        )

                    ty0, ty1, tx0, tx1 = tile_bounds[t_idx]

                    oy0 = max(ty0, block_y) - block_y
                    oy1 = min(ty1, by_end) - block_y
                    ox0 = max(tx0, block_x) - block_x
                    ox1 = min(tx1, bx_end) - block_x

                    sy0 = max(block_y - ty0, 0)
                    sy1 = sy0 + (oy1 - oy0)
                    sx0 = max(block_x - tx0, 0)
                    sx1 = sx0 + (ox1 - ox0)

                    w2d = self.y_profile[sy0:sy1, None] * self.x_profile[None, sx0:sx1]

                    for c in range(C):
                        fused_block[c, oy0:oy1, ox0:ox1] += tile_all[c, sy0:sy1, sx0:sx1] * w2d
                        weight_sum[c, oy0:oy1, ox0:ox1] += w2d

                mask = weight_sum > 0
                fused_block[mask] /= weight_sum[mask]

                # Write to 5D output: (T, C, Z, Y, X)
                self.fused_ts[time_idx, :, z_level, block_y:by_end, block_x:bx_end].write(
                    fused_block.astype(np.uint16)
                ).result()

                del fused_block, weight_sum

        gc.collect()
        if USING_GPU and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    # -------------------------------------------------------------------------
    # Multiscale pyramid
    # -------------------------------------------------------------------------

    def _create_multiscales(
        self,
        omezarr_path: Path,
        factors: Sequence[int] = (2, 4, 8),
    ) -> None:
        """Build NGFF multiscales by downsampling Y/X iteratively (not Z or T)."""
        inp = None
        for idx, factor in enumerate(factors):
            out_path = omezarr_path / f"scale{idx + 1}" / "image"
            if inp is not None:
                del inp
            prev = omezarr_path / f"scale{idx}" / "image"
            inp = ts.open(
                {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(prev)}}
            ).result()

            factor_to_use = factors[idx] // factors[idx - 1] if idx > 0 else factors[0]
            # 5D shape: (T, C, Z, Y, X)
            _, _, _, Y, X = inp.shape
            new_y, new_x = Y // factor_to_use, X // factor_to_use

            chunk_y = min(1024, new_y)
            chunk_x = min(1024, new_x)

            self.padded_shape = (new_y, new_x)
            self.chunk_y, self.chunk_x = chunk_y, chunk_x

            self._create_fused_tensorstore(output_path=out_path)

            for y0 in trange(0, new_y, chunk_y, desc=f"scale{idx + 1}", leave=True):
                by = min(chunk_y, new_y - y0)
                in_y0 = y0 * factor_to_use
                in_y1 = min(Y, (y0 + by) * factor_to_use)
                for x0 in range(0, new_x, chunk_x):
                    bx = min(chunk_x, new_x - x0)
                    in_x0 = x0 * factor_to_use
                    in_x1 = min(X, (x0 + bx) * factor_to_use)

                    # Read 5D slab: (T, C, Z, h, w)
                    slab = inp[:, :, :, in_y0:in_y1, in_x0:in_x1].read().result()
                    if self.multiscale_downsample == "stride":
                        down = slab[..., ::factor_to_use, ::factor_to_use]
                    else:
                        arr = xp.asarray(slab)
                        # Only downsample Y, X (last 2 dims)
                        block = (1, 1, 1, factor_to_use, factor_to_use)
                        down_arr = block_reduce(arr, block_size=block, func=xp.mean)
                        down = (
                            cp.asnumpy(down_arr)
                            if USING_GPU and cp is not None
                            else np.asarray(down_arr)
                        )
                    down = down.astype(slab.dtype, copy=False)
                    self.fused_ts[:, :, :, y0 : y0 + by, x0 : x0 + bx].write(down).result()

            write_scale_group_metadata(omezarr_path / f"scale{idx + 1}")

    def _generate_ngff_zarr3_json(
        self,
        omezarr_path: Path,
        resolution_multiples: Sequence[Union[int, Sequence[int]]],
        dataset_name: str = "image",
        version: str = "0.5",
    ) -> None:
        """Write OME-NGFF v0.5 multiscales JSON for Zarr3."""
        write_ngff_metadata(
            omezarr_path,
            self._pixel_size,
            self.center,
            resolution_multiples,
            dataset_name,
            version,
        )

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full tile fusion pipeline end-to-end."""
        metrics_path = self.tiff_path.parent / self.metrics_filename

        try:
            self.load_pairwise_metrics(metrics_path)
            print(f"Loaded {len(self.pairwise_metrics)} pairwise metrics from {metrics_path}")
        except FileNotFoundError:
            print("Computing pairwise registration metrics...")
            self.refine_tile_positions_with_cross_correlation(
                downsample_factors=self.downsample_factors,
                ch_idx=self.channel_to_use,
                threshold=self.threshold,
            )
            self.save_pairwise_metrics(metrics_path)
            print(f"Saved {len(self.pairwise_metrics)} pairwise metrics to {metrics_path}")

        if len(self.pairwise_metrics) == 0:
            print("No overlapping tile pairs found. Using stage positions directly.")
        else:
            print("Optimizing global tile positions...")

        self.optimize_shifts(
            method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
        )

        # Apply offsets
        self._tile_positions = [
            tuple(np.array(pos) + off * np.array(self.pixel_size))
            for pos, off in zip(self._tile_positions, self.global_offsets)
        ]

        print("Computing fused image space...")
        self._compute_fused_image_space()
        self._pad_to_chunk_multiple()
        if self.n_t > 1 or self.n_z > 1:
            print(
                f"Output size: {self.n_t}T x {self.channels}C x {self.n_z}Z x "
                f"{self.padded_shape[0]} x {self.padded_shape[1]}"
            )
        else:
            print(f"Output size: {self.padded_shape[0]} x {self.padded_shape[1]}")

        scale0 = self.output_path / "scale0" / "image"
        scale0.parent.mkdir(parents=True, exist_ok=True)
        self._create_fused_tensorstore(output_path=scale0)

        print("Fusing tiles...")
        self._fuse_tiles()

        write_scale_group_metadata(self.output_path / "scale0")

        print("Building multiscale pyramid...")
        self._create_multiscales(self.output_path, factors=self.multiscale_factors)
        self._generate_ngff_zarr3_json(
            self.output_path, resolution_multiples=self.resolution_multiples
        )

        print(f"Done! Output: {self.output_path}")

    def stitch_all_regions(self) -> None:
        """Stitch all regions in the dataset, creating separate outputs per region.

        Creates output folder structure: {input_name}_fused/{region}.ome.zarr
        """
        if not self._unique_regions:
            print("No multiple regions detected. Running standard stitching...")
            self.run()
            return

        if len(self._unique_regions) == 1:
            print(
                f"Only one region ({self._unique_regions[0]}) found. Running standard stitching..."
            )
            self.run()
            return

        # Create output folder
        output_folder = self.tiff_path.parent / f"{self.tiff_path.stem}_fused"
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Found {len(self._unique_regions)} regions: {self._unique_regions}")
        print(f"Output folder: {output_folder}")

        for i, region in enumerate(self._unique_regions):
            print(f"\n{'='*60}")
            print(f"Processing region {i+1}/{len(self._unique_regions)}: {region}")
            print(f"{'='*60}")

            self.set_region(region)
            self.output_path = output_folder / f"{region}.ome.zarr"
            self.metrics_filename = f"metrics_{region}.json"
            self.run()

        print(f"\n{'='*60}")
        print(f"All regions complete! Output: {output_folder}")
