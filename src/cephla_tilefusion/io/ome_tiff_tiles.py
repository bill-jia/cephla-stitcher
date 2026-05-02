"""
OME-TIFF tiles format reader.

Reads folder format with one OME-TIFF file per tile (all channels in one file).
Files are in ome_tiff/ subfolder with naming: {region}_{fov}.ome.tiff
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tifffile


def load_ome_tiff_tiles_metadata(folder_path: Path) -> Dict[str, Any]:
    """
    Load metadata from OME-TIFF tiles folder format.

    Parameters
    ----------
    folder_path : Path
        Path to the dataset folder containing ome_tiff/ subfolder.

    Returns
    -------
    dict
        Metadata dictionary with tile positions, channels, etc.
    """
    folder_path = Path(folder_path)
    ome_tiff_folder = folder_path / "ome_tiff"

    if not ome_tiff_folder.exists():
        raise ValueError(f"ome_tiff folder not found in {folder_path}")

    # Find all OME-TIFF files
    tiff_files = sorted(ome_tiff_folder.glob("*.ome.tiff"))
    if not tiff_files:
        tiff_files = sorted(ome_tiff_folder.glob("*.ome.tif"))
    if not tiff_files:
        raise ValueError(f"No .ome.tiff files found in {ome_tiff_folder}")

    # Load coordinates from parent folder
    coords_path = folder_path / "coordinates.csv"
    if not coords_path.exists():
        # Try in 0/ subfolder
        coords_path = folder_path / "0" / "coordinates.csv"
    if not coords_path.exists():
        raise ValueError(f"coordinates.csv not found in {folder_path}")

    coords = pd.read_csv(coords_path)
    n_tiles = len(coords)

    # Build tile file map from filenames: {region}_{fov}.ome.tiff
    tile_file_map = {}  # (region, fov) -> file path

    for tiff_file in tiff_files:
        stem = tiff_file.stem.replace(".ome", "")
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            region, fov_str = parts
            fov = int(fov_str)
            tile_file_map[(region, fov)] = tiff_file

    # Build tile identifiers from coordinates, only including tiles with files
    tile_identifiers = []
    tile_positions_temp = []
    region_fov_counts = {}

    for _, row in coords.iterrows():
        region = row["region"]
        fov = region_fov_counts.get(region, 0)
        region_fov_counts[region] = fov + 1

        # Only include if file exists
        if (region, fov) in tile_file_map:
            tile_identifiers.append((region, fov))
            x_um = row["x (mm)"] * 1000
            y_um = row["y (mm)"] * 1000
            tile_positions_temp.append((y_um, x_um))

    n_tiles = len(tile_identifiers)

    if n_tiles == 0:
        raise ValueError("No matching tiles found between coordinates.csv and ome_tiff files")

    # Read first file to get shape and channel info
    first_file = tile_file_map.get(tile_identifiers[0])
    if first_file is None:
        first_file = tiff_files[0]

    with tifffile.TiffFile(first_file) as tif:
        series = tif.series[0]
        shape = series.shape
        axes = series.axes

    # Parse axes to get dimensions
    # Possible axes: CYX, ZCYX, TCYX, TZCYX
    axes = axes.upper()
    if axes == "CYX":
        channels = shape[0]
        Y, X = shape[1], shape[2]
        file_n_z = 1
        file_n_t = 1
    elif axes == "ZCYX":
        file_n_z = shape[0]
        channels = shape[1]
        Y, X = shape[2], shape[3]
        file_n_t = 1
    elif axes == "TCYX":
        file_n_t = shape[0]
        channels = shape[1]
        Y, X = shape[2], shape[3]
        file_n_z = 1
    elif axes == "TZCYX":
        file_n_t = shape[0]
        file_n_z = shape[1]
        channels = shape[2]
        Y, X = shape[3], shape[4]
    else:
        # Fallback: assume last two are Y, X
        Y, X = shape[-2], shape[-1]
        channels = shape[-3] if len(shape) >= 3 else 1
        file_n_z = 1
        file_n_t = 1

    # Get channel names from OME metadata if available
    channel_names = []
    try:
        with tifffile.TiffFile(first_file) as tif:
            if tif.ome_metadata:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(tif.ome_metadata)
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                for channel in root.findall(".//ome:Channel", ns):
                    name = channel.get("Name")
                    if name:
                        channel_names.append(name)
    except Exception:
        pass

    if not channel_names:
        channel_names = [f"Channel_{i}" for i in range(channels)]

    # Load acquisition parameters
    params_path = folder_path / "acquisition parameters.json"
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        magnification = params.get("objective", {}).get("magnification", 10.0)
        sensor_pixel_um = params.get("sensor_pixel_size_um", 7.52)
        pixel_size_um = sensor_pixel_um / magnification
        n_z = params.get("Nz", 1)
        n_t = params.get("Nt", 1)
        dz_um = params.get("dz(um)", 1.0)
    else:
        pixel_size_um = 0.752
        n_z = file_n_z
        n_t = file_n_t
        dz_um = 1.0

    pixel_size = (pixel_size_um, pixel_size_um)

    # Use tile positions built earlier (only for tiles with files)
    tile_positions = tile_positions_temp

    # Extract unique regions
    unique_regions = []
    seen = set()
    for tile_id in tile_identifiers:
        region = tile_id[0]
        if region not in seen:
            unique_regions.append(region)
            seen.add(region)

    return {
        "n_tiles": n_tiles,
        "n_series": n_tiles,
        "shape": (Y, X),
        "channels": channels,
        "channel_names": channel_names,
        "n_z": n_z,
        "n_t": n_t,
        "dz_um": dz_um,
        "time_dim": n_t,
        "position_dim": n_tiles,
        "pixel_size": pixel_size,
        "tile_positions": tile_positions,
        "tile_identifiers": tile_identifiers,
        "unique_regions": unique_regions,
        "ome_tiff_folder": ome_tiff_folder,
        "tile_file_map": tile_file_map,
        "axes": axes,
    }


def _get_tile_file(ome_tiff_folder: Path, tile_id: tuple, tile_file_map: Dict) -> Path:
    """Get the OME-TIFF file path for a tile."""
    if tile_id in tile_file_map:
        return tile_file_map[tile_id]

    # Fallback: construct filename - try both .ome.tiff and .ome.tif
    region, fov = tile_id
    path = ome_tiff_folder / f"{region}_{fov}.ome.tiff"
    if not path.exists():
        path = ome_tiff_folder / f"{region}_{fov}.ome.tif"
    return path


def read_ome_tiff_tiles_tile(
    ome_tiff_folder: Path,
    tile_identifiers: List[tuple],
    tile_file_map: Dict,
    tile_idx: int,
    axes: str,
    z_level: int = 0,
    time_idx: int = 0,
) -> np.ndarray:
    """
    Read all channels of a tile from OME-TIFF tiles format.

    Parameters
    ----------
    ome_tiff_folder : Path
        Path to ome_tiff folder.
    tile_identifiers : list of tuple
        Tile identifiers: (region, fov) tuples.
    tile_file_map : dict
        Mapping from tile_id to file path.
    tile_idx : int
        Index of the tile.
    axes : str
        Axes string from OME-TIFF (e.g., "CYX", "ZCYX").
    z_level : int
        Z-level index (default 0).
    time_idx : int
        Time point index (default 0).

    Returns
    -------
    np.ndarray
        Tile data as (C, Y, X) array.
    """
    tile_id = tile_identifiers[tile_idx]
    file_path = _get_tile_file(ome_tiff_folder, tile_id, tile_file_map)

    data = tifffile.imread(file_path)
    axes = axes.upper()

    # Extract the correct slice based on axes
    if axes == "CYX":
        return data  # Already (C, Y, X)
    elif axes == "ZCYX":
        return data[z_level]  # (Z, C, Y, X) -> (C, Y, X)
    elif axes == "TCYX":
        return data[time_idx]  # (T, C, Y, X) -> (C, Y, X)
    elif axes == "TZCYX":
        return data[time_idx, z_level]  # (T, Z, C, Y, X) -> (C, Y, X)
    elif axes == "YX":
        return data[np.newaxis, ...]  # (Y, X) -> (1, Y, X)
    else:
        # Fallback: assume last 3 dims are C, Y, X
        if data.ndim == 3:
            return data
        elif data.ndim == 4:
            return data[z_level]
        elif data.ndim == 5:
            return data[time_idx, z_level]
        return data


def read_ome_tiff_tiles_region(
    ome_tiff_folder: Path,
    tile_identifiers: List[tuple],
    tile_file_map: Dict,
    tile_idx: int,
    axes: str,
    y_slice: slice,
    x_slice: slice,
    channel_idx: int = 0,
    z_level: int = 0,
    time_idx: int = 0,
) -> np.ndarray:
    """
    Read a region of a single channel from OME-TIFF tiles format.

    Parameters
    ----------
    ome_tiff_folder : Path
        Path to ome_tiff folder.
    tile_identifiers : list of tuple
        Tile identifiers: (region, fov) tuples.
    tile_file_map : dict
        Mapping from tile_id to file path.
    tile_idx : int
        Index of the tile.
    axes : str
        Axes string from OME-TIFF.
    y_slice : slice
        Y region to read.
    x_slice : slice
        X region to read.
    channel_idx : int
        Channel index.
    z_level : int
        Z-level index (default 0).
    time_idx : int
        Time point index (default 0).

    Returns
    -------
    np.ndarray
        Tile region as float32.
    """
    # Read full tile and extract region
    tile_data = read_ome_tiff_tiles_tile(
        ome_tiff_folder,
        tile_identifiers,
        tile_file_map,
        tile_idx,
        axes,
        z_level,
        time_idx,
    )

    # tile_data is (C, Y, X), extract channel and region
    return tile_data[channel_idx, y_slice, x_slice].astype(np.float32)
