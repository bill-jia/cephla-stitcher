"""
Flatfield correction module using BaSiCPy.

Provides functions to calculate and apply flatfield (and optionally darkfield)
correction for microscopy images.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from basicpy import BaSiC

    HAS_BASICPY = True
except ImportError:
    HAS_BASICPY = False


def calculate_flatfield(
    tiles: List[np.ndarray],
    use_darkfield: bool = False,
    constant_darkfield: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Calculate flatfield (and optionally darkfield) using BaSiCPy.

    Parameters
    ----------
    tiles : list of ndarray
        List of tile images, each with shape (C, Y, X) or (Y, X) for single-channel.
        2D arrays are automatically converted to 3D with shape (1, Y, X).
    use_darkfield : bool
        Whether to also compute darkfield correction.
    constant_darkfield : bool
        If True, darkfield is reduced to a single constant value (median) per
        channel. This is physically appropriate since dark current is typically
        uniform across the sensor. Default is True.

    Returns
    -------
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X), float32.
    darkfield : ndarray or None
        Darkfield correction array with shape (C, Y, X), or None if not computed.
        If constant_darkfield=True, each channel slice will be a constant value.

    Raises
    ------
    ImportError
        If basicpy is not installed.
    ValueError
        If tiles list is empty or tiles have inconsistent shapes.
    """
    if not HAS_BASICPY:
        raise ImportError(
            "basicpy is required for flatfield calculation. Install with: pip install basicpy"
        )

    if not tiles:
        raise ValueError("tiles list is empty")

    # Validate tile dimensionality: only 2D (Y, X) or 3D (C, Y, X) supported
    for i, t in enumerate(tiles):
        if t.ndim not in (2, 3):
            raise ValueError(f"Tile {i} has {t.ndim} dimensions; expected 2 (Y, X) or 3 (C, Y, X)")

    # Support 2D (Y, X) arrays by converting to 3D (1, Y, X)
    tiles = [t[np.newaxis, ...] if t.ndim == 2 else t for t in tiles]

    # Get shape from first tile
    n_channels = tiles[0].shape[0]
    tile_shape = tiles[0].shape[1:]  # (Y, X)

    # Validate all tiles have same shape
    for i, tile in enumerate(tiles):
        if tile.shape[0] != n_channels:
            raise ValueError(f"Tile {i} has {tile.shape[0]} channels, expected {n_channels}")
        if tile.shape[1:] != tile_shape:
            raise ValueError(f"Tile {i} has shape {tile.shape[1:]}, expected {tile_shape}")

    # Calculate flatfield per channel
    flatfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32)
    darkfield = np.zeros((n_channels,) + tile_shape, dtype=np.float32) if use_darkfield else None

    for ch in range(n_channels):
        # Stack channel data from all tiles: shape (n_tiles, Y, X)
        channel_stack = np.stack([tile[ch] for tile in tiles], axis=0)

        # Create BaSiC instance and fit
        basic = BaSiC(get_darkfield=use_darkfield, smoothness_flatfield=1.0)
        try:
            basic.fit(channel_stack)
        except Exception as exc:
            raise RuntimeError(
                f"BaSiCPy flatfield fitting failed for channel {ch} "
                f"with data shape {channel_stack.shape}"
            ) from exc

        flatfield[ch] = basic.flatfield.astype(np.float32)

        if use_darkfield:
            if constant_darkfield:
                # Use median value for constant darkfield (more robust than mean)
                df_value = np.median(basic.darkfield)
                darkfield[ch] = np.full(tile_shape, df_value, dtype=np.float32)
            else:
                darkfield[ch] = basic.darkfield.astype(np.float32)

    return flatfield, darkfield


def apply_flatfield(
    tile: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile.

    Formula:
        If darkfield is provided: corrected = (raw - darkfield) / flatfield
        Otherwise: corrected = raw / flatfield

    Parameters
    ----------
    tile : ndarray
        Input tile with shape (C, Y, X).
    flatfield : ndarray
        Flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield correction array with shape (C, Y, X).

    Returns
    -------
    corrected : ndarray
        Corrected tile with shape (C, Y, X), cast back to the input dtype.
        For integer dtypes, values are clipped to the valid range before
        casting (e.g., negative values clipped to 0 for unsigned types).

    Raises
    ------
    ValueError
        If tile and flatfield shapes are incompatible.
    """
    # Validate shapes
    if tile.shape != flatfield.shape:
        raise ValueError(
            f"Tile shape {tile.shape} does not match flatfield shape {flatfield.shape}"
        )
    if darkfield is not None and tile.shape != darkfield.shape:
        raise ValueError(
            f"Tile shape {tile.shape} does not match darkfield shape {darkfield.shape}"
        )

    # Convert to float32 to avoid underflow with unsigned integer types
    tile_f = tile.astype(np.float32)
    # For flatfield values <= 1e-6, use 1.0 to avoid division by zero/near-zero
    flatfield_safe = np.where(flatfield > 1e-6, flatfield, 1.0).astype(np.float32)

    if darkfield is not None:
        corrected = (tile_f - darkfield.astype(np.float32)) / flatfield_safe
    else:
        corrected = tile_f / flatfield_safe

    # Clip to valid range for integer dtypes to avoid wraparound
    if np.issubdtype(tile.dtype, np.integer):
        info = np.iinfo(tile.dtype)
        corrected = np.clip(corrected, info.min, info.max)

    return corrected.astype(tile.dtype)


def apply_flatfield_region(
    region: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray],
    y_slice: slice,
    x_slice: slice,
    channel_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Apply flatfield correction to a tile region.

    Parameters
    ----------
    region : ndarray
        Input region with shape (C, h, w) or (h, w).
    flatfield : ndarray
        Full flatfield correction array with shape (C, Y, X).
    darkfield : ndarray, optional
        Full darkfield correction array with shape (C, Y, X).
    y_slice, x_slice : slice
        Slices defining the region within the full tile.
    channel_idx : int, optional
        Index of the channel that ``region`` represents within the flatfield.
        Required when ``region`` is 2D and the flatfield has multiple channels;
        defaults to 0 for a single-channel flatfield. When ``region`` is 3D,
        if provided, only the matching flatfield slice is used (region must
        have one channel) — otherwise the channel counts must match.

    Returns
    -------
    corrected : ndarray
        Corrected region with same shape as input.

    Raises
    ------
    ValueError
        If region and flatfield shapes are incompatible, or ``channel_idx`` is
        out of range.
    """
    n_ff_channels = flatfield.shape[0]

    if channel_idx is not None and not (0 <= channel_idx < n_ff_channels):
        raise ValueError(
            f"channel_idx={channel_idx} is out of range for flatfield with "
            f"{n_ff_channels} channels"
        )

    if region.ndim == 2:
        if channel_idx is None:
            if n_ff_channels != 1:
                raise ValueError(
                    f"Region is 2D but flatfield has {n_ff_channels} channels; "
                    "pass channel_idx to select which slice to apply"
                )
            ch = 0
        else:
            ch = channel_idx
        ff_region = flatfield[ch, y_slice, x_slice]
        df_region = darkfield[ch, y_slice, x_slice] if darkfield is not None else None
    else:
        if channel_idx is not None:
            if region.shape[0] != 1:
                raise ValueError(
                    f"channel_idx was provided but region has {region.shape[0]} "
                    "channels; expected 1"
                )
            ff_region = flatfield[channel_idx : channel_idx + 1, y_slice, x_slice]
            df_region = (
                darkfield[channel_idx : channel_idx + 1, y_slice, x_slice]
                if darkfield is not None
                else None
            )
        else:
            if region.shape[0] != n_ff_channels:
                raise ValueError(
                    f"Region has {region.shape[0]} channels but flatfield has "
                    f"{n_ff_channels} channels"
                )
            ff_region = flatfield[:, y_slice, x_slice]
            df_region = darkfield[:, y_slice, x_slice] if darkfield is not None else None

    # Convert to float32 to avoid underflow with unsigned integer types
    region_f = region.astype(np.float32)
    # For flatfield values <= 1e-6, use 1.0 to avoid division by zero/near-zero
    ff_safe = np.where(ff_region > 1e-6, ff_region, 1.0).astype(np.float32)

    if df_region is not None:
        corrected = (region_f - df_region.astype(np.float32)) / ff_safe
    else:
        corrected = region_f / ff_safe

    # Clip to valid range for integer dtypes to avoid wraparound
    if np.issubdtype(region.dtype, np.integer):
        info = np.iinfo(region.dtype)
        corrected = np.clip(corrected, info.min, info.max)

    return corrected.astype(region.dtype)


def save_flatfield(
    path: Path,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray] = None,
) -> None:
    """
    Save flatfield (and optionally darkfield) to a .npy file.

    Parameters
    ----------
    path : Path
        Output path (should end with .npy).
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray, optional
        Darkfield array with shape (C, Y, X).
    """
    data = {
        "flatfield": flatfield.astype(np.float32),
        "darkfield": darkfield.astype(np.float32) if darkfield is not None else None,
        "channels": flatfield.shape[0],
        "shape": flatfield.shape[1:],
    }
    np.save(path, data, allow_pickle=True)


def load_flatfield(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load flatfield (and optionally darkfield) from a .npy file.

    Parameters
    ----------
    path : Path
        Path to .npy file.

    Returns
    -------
    flatfield : ndarray
        Flatfield array with shape (C, Y, X).
    darkfield : ndarray or None
        Darkfield array with shape (C, Y, X), or None if not present.

    Raises
    ------
    OSError
        If the file cannot be read (not found, permission denied, etc.).
    ValueError
        If the file format is invalid (not a dictionary with 'flatfield' key).
    """
    try:
        loaded = np.load(path, allow_pickle=True)
    except OSError as exc:
        raise OSError(f"Cannot read flatfield file '{path}': {exc}") from exc

    try:
        data = loaded.item()
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Invalid flatfield file format at '{path}'. "
            "Expected a NumPy .npy file containing a dictionary as saved by "
            "`save_flatfield` (with keys like 'flatfield' and 'darkfield')."
        ) from exc

    if not isinstance(data, dict) or "flatfield" not in data:
        raise ValueError(
            f"Invalid flatfield file format at '{path}'. "
            "Expected a dictionary with at least a 'flatfield' entry."
        )

    flatfield = data["flatfield"]
    darkfield = data.get("darkfield", None)
    return flatfield, darkfield
