"""
Global position optimization.

Least-squares optimization of tile positions from pairwise measurements.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def solve_global(links: List[Dict[str, Any]], n_tiles: int, fixed_indices: List[int]) -> np.ndarray:
    """
    Solve a linear least-squares for all 2 axes at once,
    given weighted pairwise links and fixed tile indices.

    Parameters
    ----------
    links : list of dict
        Each dict has keys: 'i', 'j', 't' (2D offset), 'w' (weight).
    n_tiles : int
        Total number of tiles.
    fixed_indices : list of int
        Indices of tiles to fix at origin.

    Returns
    -------
    shifts : ndarray of shape (n_tiles, 2)
        Optimized shifts for each tile.
    """
    shifts = np.zeros((n_tiles, 2), dtype=np.float64)
    for axis in range(2):
        m = len(links) + len(fixed_indices)
        A = np.zeros((m, n_tiles), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        row = 0
        for link in links:
            i, j = link["i"], link["j"]
            t, w = link["t"][axis], link["w"]
            A[row, j] = w
            A[row, i] = -w
            b[row] = w * t
            row += 1
        for idx in fixed_indices:
            A[row, idx] = 1.0
            b[row] = 0.0
            row += 1
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        shifts[:, axis] = sol
    return shifts


def two_round_optimization(
    links: List[Dict[str, Any]],
    n_tiles: int,
    fixed_indices: List[int],
    rel_thresh: float,
    abs_thresh: float,
    iterative: bool,
) -> np.ndarray:
    """
    Perform two-round (or iterative two-round) robust optimization:
    1. Solve on all links.
    2. Remove any link whose residual > max(abs_thresh, rel_thresh * median(residuals)).
    3. Re-solve on the remaining links.
    If iterative=True, repeat step 2 + 3 until no more links are removed.

    Parameters
    ----------
    links : list of dict
        Pairwise link data.
    n_tiles : int
        Total number of tiles.
    fixed_indices : list of int
        Tiles to fix at origin.
    rel_thresh : float
        Relative threshold (fraction of median residual).
    abs_thresh : float
        Absolute threshold for residual.
    iterative : bool
        If True, iterate until convergence.

    Returns
    -------
    shifts : ndarray of shape (n_tiles, 2)
        Optimized shifts.
    """
    shifts = solve_global(links, n_tiles, fixed_indices)

    def compute_res(ls: List[Dict[str, Any]], sh: np.ndarray) -> np.ndarray:
        return np.array([np.linalg.norm(sh[l["j"]] - sh[l["i"]] - l["t"]) for l in ls])

    work = links.copy()
    res = compute_res(work, shifts)
    if len(res) == 0:
        return shifts
    cutoff = max(abs_thresh, rel_thresh * np.median(res))
    outliers = set(np.where(res > cutoff)[0])

    if iterative:
        while outliers:
            for k in sorted(outliers, reverse=True):
                work.pop(k)
            if not work:
                break
            shifts = solve_global(work, n_tiles, fixed_indices)
            res = compute_res(work, shifts)
            if len(res) == 0:
                break
            cutoff = max(abs_thresh, rel_thresh * np.median(res))
            outliers = set(np.where(res > cutoff)[0])
    else:
        for k in sorted(outliers, reverse=True):
            work.pop(k)
        if work:
            shifts = solve_global(work, n_tiles, fixed_indices)

    return shifts


def links_from_pairwise_metrics(
    pairwise_metrics: Dict[Tuple[int, int], Tuple[int, int, float]],
) -> List[Dict[str, Any]]:
    """
    Convert pairwise_metrics dict to list of link dicts.

    Parameters
    ----------
    pairwise_metrics : dict
        Keys are (i, j) tuples, values are (dy, dx, score) tuples.

    Returns
    -------
    links : list of dict
        Each dict has 'i', 'j', 't', 'w' keys.
    """
    links = []
    for (i, j), v in pairwise_metrics.items():
        links.append(
            {
                "i": i,
                "j": j,
                "t": np.array(v[:2], dtype=np.float64),
                "w": np.sqrt(v[2]),
            }
        )
    return links
