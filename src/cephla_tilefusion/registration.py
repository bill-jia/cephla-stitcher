"""
Tile registration algorithms.

Phase cross-correlation based registration with SSIM scoring.
"""

from typing import Any, Tuple, Union

import numpy as np

from .utils import (
    USING_GPU,
    block_reduce,
    compute_ssim,
    match_histograms,
    phase_cross_correlation,
    shift_array,
    to_numpy,
    xp,
    cp,
)


def register_pair_worker(args: Tuple) -> Tuple:
    """
    Worker function for parallel registration of a tile pair.

    Parameters
    ----------
    args : tuple
        (i_pos, j_pos, patch_i, patch_j, df, sw, th, max_shift)

    Returns
    -------
    tuple
        (i_pos, j_pos, dy_s, dx_s, score) or (i_pos, j_pos, None, None, None) on failure
    """
    i_pos, j_pos, patch_i, patch_j, df, sw, th, max_shift = args

    try:
        # Downsample
        reduce_block = (1, df[0], df[1]) if patch_i.ndim == 3 else tuple(df)
        g1 = block_reduce(patch_i, reduce_block, np.mean)
        g2 = block_reduce(patch_j, reduce_block, np.mean)

        # Squeeze to 2D if needed
        while g1.ndim > 2 and g1.shape[0] == 1:
            g1 = g1[0]
            g2 = g2[0]

        # Match histograms
        g2 = match_histograms(g2, g1)

        # Phase cross-correlation
        shift, _, _ = phase_cross_correlation(
            g1.astype(np.float32),
            g2.astype(np.float32),
            normalization="phase",
            upsample_factor=10,
        )

        # Apply shift and compute SSIM
        g2s = shift_array(g2, shift_vec=shift)
        ssim_val = compute_ssim(g1, g2s, win_size=sw)

        # Scale shift back to original resolution
        dy_s, dx_s = int(np.round(shift[0] * df[0])), int(np.round(shift[1] * df[1]))

        # Check thresholds
        if th != 0.0 and ssim_val < th:
            return (i_pos, j_pos, None, None, None)
        if abs(dy_s) > max_shift[0] or abs(dx_s) > max_shift[1]:
            return (i_pos, j_pos, None, None, None)

        return (i_pos, j_pos, dy_s, dx_s, round(ssim_val, 3))

    except Exception:
        return (i_pos, j_pos, None, None, None)


def register_and_score(
    g1: Any,
    g2: Any,
    win_size: int,
    debug: bool = False,
) -> Union[Tuple[Tuple[float, float], float], Tuple[None, None]]:
    """
    Histogram-match g2->g1, compute subpixel shift, and SSIM.

    Parameters
    ----------
    g1, g2 : array-like
        Fixed and moving patches (YX).
    win_size : int
        SSIM window.
    debug : bool
        If True, print intermediate info.

    Returns
    -------
    shift : (dy, dx)
        Subpixel shift.
    ssim_val : float
        SSIM score.
    """
    arr1 = xp.asarray(g1, dtype=xp.float32)
    arr2 = xp.asarray(g2, dtype=xp.float32)
    while arr1.ndim > 2 and arr1.shape[0] == 1:
        arr1 = arr1[0]
        arr2 = arr2[0]

    arr2 = match_histograms(arr2, arr1)
    shift, _, _ = phase_cross_correlation(
        arr1,
        arr2,
        disambiguate=True,
        normalization="phase",
        upsample_factor=10,
        overlap_ratio=0.5,
    )
    shift_apply = xp.asarray(shift, dtype=xp.float32)
    g2s = shift_array(arr2, shift_vec=shift_apply)
    ssim_val = compute_ssim(arr1, g2s, win_size=win_size)
    out_shift = to_numpy(shift_apply)
    return tuple(float(s) for s in out_shift), float(ssim_val)


def find_adjacent_pairs(tile_positions, pixel_size, tile_shape, min_overlap=15):
    """
    Find adjacent tile pairs for registration.

    Parameters
    ----------
    tile_positions : list of (y, x) tuples
        Stage positions for each tile.
    pixel_size : tuple of (py, px)
        Pixel size in physical units.
    tile_shape : tuple of (Y, X)
        Tile dimensions in pixels.
    min_overlap : int
        Minimum overlap in pixels.

    Returns
    -------
    adjacent_pairs : list of tuples
        Each tuple: (i_pos, j_pos, dy, dx, overlap_y, overlap_x)
    """
    n_pos = len(tile_positions)
    Y, X = tile_shape
    adjacent_pairs = []

    for i_pos in range(n_pos):
        for j_pos in range(i_pos + 1, n_pos):
            phys = np.array(tile_positions[j_pos]) - np.array(tile_positions[i_pos])
            vox_off = np.round(phys / np.array(pixel_size)).astype(int)
            dy, dx = vox_off

            overlap_y = Y - abs(dy)
            overlap_x = X - abs(dx)

            # Check if tiles are adjacent
            is_horizontal_neighbor = abs(dy) < min_overlap and overlap_x >= min_overlap
            is_vertical_neighbor = abs(dx) < min_overlap and overlap_y >= min_overlap

            if is_horizontal_neighbor or is_vertical_neighbor:
                adjacent_pairs.append((i_pos, j_pos, dy, dx, overlap_y, overlap_x))

    return adjacent_pairs


def compute_pair_bounds(adjacent_pairs, tile_shape):
    """
    Compute overlap bounds for each adjacent pair.

    Parameters
    ----------
    adjacent_pairs : list
        Output from find_adjacent_pairs.
    tile_shape : tuple of (Y, X)
        Tile dimensions.

    Returns
    -------
    pair_bounds : list of tuples
        Each tuple: (i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x)
    """
    Y, X = tile_shape
    pair_bounds = []

    for i_pos, j_pos, dy, dx, overlap_y, overlap_x in adjacent_pairs:
        bounds_i_y = (max(0, dy), min(Y, Y + dy))
        bounds_i_x = (max(0, dx), min(X, X + dx))
        bounds_j_y = (max(0, -dy), min(Y, Y - dy))
        bounds_j_x = (max(0, -dx), min(X, X - dx))

        if bounds_i_y[1] > bounds_i_y[0] and bounds_i_x[1] > bounds_i_x[0]:
            pair_bounds.append((i_pos, j_pos, bounds_i_y, bounds_i_x, bounds_j_y, bounds_j_x))

    return pair_bounds
