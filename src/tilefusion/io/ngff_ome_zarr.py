"""
Standard NGFF OME-Zarr multi-tile reader.

Reads datasets where each tile/FOV is an independent multiscales NGFF image
group (per OME-NGFF spec), with stage positions encoded in the image's own
``coordinateTransformations`` translation transform on the scale-0 dataset.

Supported layouts:

- ``flat``: root group with numbered/named child image groups (e.g. bioformats2raw).
- ``hcs``: NGFF HCS plate/well/field hierarchy.
- ``sibling``: directory containing sibling ``.ome.zarr`` subdirectories.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorstore as ts

# Physical unit -> micrometer conversion factor.
_UNIT_TO_UM = {
    "micrometer": 1.0,
    "micrometre": 1.0,
    "micron": 1.0,
    "um": 1.0,
    "µm": 1.0,
    "μm": 1.0,
    "nanometer": 1e-3,
    "nanometre": 1e-3,
    "nm": 1e-3,
    "millimeter": 1e3,
    "millimetre": 1e3,
    "mm": 1e3,
    "centimeter": 1e4,
    "centimetre": 1e4,
    "cm": 1e4,
    "meter": 1e6,
    "metre": 1e6,
    "m": 1e6,
}


def _read_group_metadata(path: Path) -> Optional[Dict[str, Any]]:
    """Read NGFF group metadata (v3 ``zarr.json`` or v2 ``.zattrs``).

    Returns a dict normalized to the v3 shape ``{"attributes": {...}, ...}`` so
    downstream code can read ``attributes`` uniformly.
    """
    zj = path / "zarr.json"
    if zj.exists():
        try:
            with open(zj) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    zattrs = path / ".zattrs"
    if zattrs.exists():
        try:
            with open(zattrs) as f:
                attrs = json.load(f)
            return {"attributes": attrs, "zarr_format": 2}
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _get_ome_attrs(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Extract OME-NGFF attribute namespace.

    NGFF v0.5 nests under ``attributes.ome``; older (v0.4) versions place
    fields like ``multiscales`` / ``plate`` directly under ``attributes``.
    """
    attrs = meta.get("attributes", {}) or {}
    if isinstance(attrs.get("ome"), dict):
        return attrs["ome"]
    return attrs


def _has_multiscales(path: Path) -> bool:
    """True if ``path`` is an NGFF image group (has ``multiscales`` attr)."""
    meta = _read_group_metadata(path)
    if meta is None:
        return False
    return "multiscales" in _get_ome_attrs(meta)


def _has_child_ngff_images(root: Path) -> bool:
    """True if ``root`` has at least one immediate child that is an NGFF image."""
    if not root.is_dir():
        return False
    try:
        for sub in root.iterdir():
            if sub.is_dir() and _has_multiscales(sub):
                return True
    except OSError:
        return False
    return False


def _has_sibling_ngff_images(root: Path) -> bool:
    """True if ``root`` has no ``zarr.json`` but contains >= 2 sibling NGFF images."""
    if not root.is_dir():
        return False
    if (root / "zarr.json").exists() or (root / ".zattrs").exists():
        return False
    count = 0
    try:
        for sub in root.iterdir():
            if sub.is_dir() and _has_multiscales(sub):
                count += 1
                if count >= 2:
                    return True
    except OSError:
        return False
    return False


def is_ngff_ome_zarr(root: Path) -> bool:
    """Cheap check for ``TileFusion`` format detection."""
    root = Path(root)
    if not root.is_dir():
        return False
    meta = _read_group_metadata(root)
    if meta is not None:
        ome = _get_ome_attrs(meta)
        attrs = meta.get("attributes", {}) or {}
        if "plate" in ome or "plate" in attrs:
            return True
        if "bioformats2raw.layout" in attrs or "bioformats2raw.layout" in ome:
            return True
        if _has_child_ngff_images(root):
            return True
        return False
    return _has_sibling_ngff_images(root)


def _detect_layout(root: Path) -> str:
    """Detect NGFF layout: 'flat', 'hcs', or 'sibling'. Raises ValueError otherwise."""
    meta = _read_group_metadata(root)
    if meta is not None:
        ome = _get_ome_attrs(meta)
        attrs = meta.get("attributes", {}) or {}
        if "plate" in ome or "plate" in attrs:
            return "hcs"
        if _has_child_ngff_images(root):
            return "flat"
        if "multiscales" in ome:
            raise ValueError(
                f"{root} is a single-image NGFF, not a multi-tile dataset. "
                "TileFusion requires multiple tiles to stitch."
            )
        raise ValueError(f"{root}/zarr.json does not describe an NGFF multi-tile dataset")
    if _has_sibling_ngff_images(root):
        return "sibling"
    raise ValueError(f"{root} does not look like an NGFF ome-zarr dataset")


def _sort_key(name: str):
    """Sort key that prefers numeric ordering for names like '0', '1', '10'."""
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)


def _discover_flat_tiles(root: Path) -> List[Tuple[Tuple[str], Path]]:
    """Find child directories under ``root`` that are NGFF image groups."""
    tiles = []
    for sub in root.iterdir():
        if sub.is_dir() and _has_multiscales(sub):
            tiles.append((sub.name, sub))
    tiles.sort(key=lambda item: _sort_key(item[0]))
    # Normalize tile ids to single-element tuples so downstream code can treat
    # them uniformly. Tile id shape is ('name',) for flat layouts.
    return [((name,), path) for (name, path) in tiles]


def _discover_hcs_plate_tiles(root: Path) -> List[Tuple[Tuple[str, str, int], Path]]:
    """Enumerate per-field tiles from an NGFF HCS plate layout."""
    meta = _read_group_metadata(root)
    ome = _get_ome_attrs(meta)
    plate = ome.get("plate") or meta.get("attributes", {}).get("plate")
    if plate is None:
        raise ValueError(f"No plate metadata at {root}")

    wells = plate.get("wells", []) or []
    tiles: List[Tuple[Tuple[str, str, int], Path]] = []
    for well in wells:
        well_path_str = well.get("path")
        if not well_path_str:
            continue
        parts = well_path_str.split("/", 1)
        if len(parts) != 2:
            continue
        row_label, col_label = parts
        well_dir = root / well_path_str
        well_meta = _read_group_metadata(well_dir)
        if well_meta is None:
            continue
        well_ome = _get_ome_attrs(well_meta)
        well_section = well_ome.get("well") or well_meta.get("attributes", {}).get("well", {})
        images = well_section.get("images", []) if isinstance(well_section, dict) else []
        for img in images:
            field_rel = img.get("path")
            if field_rel is None:
                continue
            field_path = well_dir / str(field_rel)
            if not _has_multiscales(field_path):
                continue
            try:
                field_idx = int(field_rel)
            except (TypeError, ValueError):
                field_idx = str(field_rel)
            tiles.append(((row_label, col_label, field_idx), field_path))
    return tiles


def _discover_sibling_tiles(root: Path) -> List[Tuple[Tuple[str], Path]]:
    """Find sibling NGFF image directories directly under ``root``."""
    tiles = []
    for sub in root.iterdir():
        if sub.is_dir() and _has_multiscales(sub):
            tiles.append((sub.name, sub))
    tiles.sort(key=lambda item: _sort_key(item[0]))
    return [((name,), path) for (name, path) in tiles]


def _normalize_axes(axes: Sequence[Dict[str, Any]]) -> Tuple[str, List[float]]:
    """Return (axis_string, unit_factors_to_um) from an NGFF axes list."""
    axis_str = ""
    unit_factors: List[float] = []
    for ax in axes:
        name = (ax.get("name") or "").lower()
        if name not in "tczyx":
            raise ValueError(f"Unknown NGFF axis name: {ax.get('name')!r}")
        axis_str += name
        if ax.get("type") == "space":
            unit = (ax.get("unit") or "").lower()
            unit_factors.append(_UNIT_TO_UM.get(unit, 1.0))
        else:
            unit_factors.append(1.0)
    if len(set(axis_str)) != len(axis_str):
        raise ValueError(f"Duplicate NGFF axes: {axis_str!r}")
    for req in ("y", "x"):
        if req not in axis_str:
            raise ValueError(f"NGFF multiscales missing required spatial axis {req!r}")
    return axis_str, unit_factors


def _compose_transforms(
    multiscales: Dict[str, Any],
    dataset_idx: int,
    n_axes: int,
) -> Tuple[List[float], List[float]]:
    """Compose multiscales-level and dataset-level coordinateTransformations.

    Applies the NGFF rule: outer (multiscales) transforms are applied before
    inner (dataset) transforms. ``scale`` composes by element-wise product;
    ``translation`` composes additively, with the current scale applied to
    subsequent translations.
    """
    scale = [1.0] * n_axes
    translation = [0.0] * n_axes
    datasets = multiscales.get("datasets") or []
    outer = multiscales.get("coordinateTransformations", []) or []
    inner = datasets[dataset_idx].get("coordinateTransformations", []) if datasets else []
    for xforms in (outer, inner):
        for xf in xforms:
            xtype = xf.get("type")
            if xtype == "scale":
                s = xf.get("scale") or []
                for i in range(min(n_axes, len(s))):
                    scale[i] *= float(s[i])
            elif xtype == "translation":
                t = xf.get("translation") or []
                for i in range(min(n_axes, len(t))):
                    translation[i] += float(t[i])
    return scale, translation


def _open_array(arr_path: Path) -> ts.TensorStore:
    """Open a tensorstore handle for an NGFF array, handling v2 and v3."""
    if (arr_path / "zarr.json").exists():
        driver = "zarr3"
    elif (arr_path / ".zarray").exists():
        driver = "zarr"
    else:
        raise ValueError(f"No zarr array found at {arr_path}")
    spec = {
        "driver": driver,
        "kvstore": {"driver": "file", "path": str(arr_path)},
    }
    return ts.open(spec, create=False, open=True).result()


def _parse_image(tile_path: Path) -> Dict[str, Any]:
    """Parse one NGFF image group: axes, scale, translation, open handle."""
    meta = _read_group_metadata(tile_path)
    if meta is None:
        raise ValueError(f"Missing NGFF metadata at {tile_path}")
    ome = _get_ome_attrs(meta)
    multiscales = ome.get("multiscales")
    if not multiscales:
        raise ValueError(f"No multiscales at {tile_path}")
    mult = multiscales[0]
    axes = mult.get("axes") or []
    axis_str, unit_factors = _normalize_axes(axes)
    scale, translation = _compose_transforms(mult, 0, len(axes))
    scale_um = [scale[i] * unit_factors[i] for i in range(len(axes))]
    translation_um = [translation[i] * unit_factors[i] for i in range(len(axes))]
    datasets = mult.get("datasets") or []
    if not datasets:
        raise ValueError(f"NGFF multiscales has no datasets at {tile_path}")
    scale0_rel = datasets[0].get("path")
    if not scale0_rel:
        raise ValueError(f"NGFF scale-0 dataset has no path at {tile_path}")
    ts_handle = _open_array(tile_path / scale0_rel)

    channel_names: List[str] = []
    omero = ome.get("omero") or {}
    for ch in omero.get("channels", []) or []:
        label = ch.get("label") or ch.get("name")
        if label:
            channel_names.append(str(label))

    return {
        "axes": axis_str,
        "scale_um": scale_um,
        "translation_um": translation_um,
        "handle": ts_handle,
        "channel_names": channel_names,
        "shape": tuple(ts_handle.shape),
    }


def load_ngff_ome_zarr_metadata(root_path: Path) -> Dict[str, Any]:
    """Load metadata from a standard NGFF OME-Zarr multi-tile dataset.

    Parameters
    ----------
    root_path : Path
        Root of the dataset. Layout is auto-detected (flat / hcs / sibling).

    Returns
    -------
    dict
        Metadata dictionary matching the contract used by ``TileFusion``:
        ``n_tiles``, ``n_series``, ``shape`` (Y, X), ``channels``,
        ``channel_names``, ``n_z``, ``n_t``, ``dz_um``, ``time_dim``,
        ``position_dim``, ``pixel_size`` (py_um, px_um), ``tile_positions``
        (list of (y_um, x_um) tuples), ``tile_identifiers``, ``unique_regions``,
        plus ``ngff_root``, ``tile_stores`` (tile_id -> tensorstore), and
        ``tile_axes`` (normalized axis string).
    """
    root_path = Path(root_path)
    layout = _detect_layout(root_path)

    if layout == "hcs":
        tile_groups = _discover_hcs_plate_tiles(root_path)
    elif layout == "flat":
        tile_groups = _discover_flat_tiles(root_path)
    else:  # sibling
        tile_groups = _discover_sibling_tiles(root_path)

    if not tile_groups:
        raise ValueError(f"No NGFF image tiles found under {root_path}")

    tile_stores: Dict[Any, ts.TensorStore] = {}
    tile_positions: List[Tuple[float, float]] = []
    tile_identifiers: List[Any] = []
    reference: Optional[Dict[str, Any]] = None

    for tile_id, tile_path in tile_groups:
        info = _parse_image(tile_path)
        axes = info["axes"]
        shape = info["shape"]

        y_idx = axes.index("y")
        x_idx = axes.index("x")
        Y = int(shape[y_idx])
        X = int(shape[x_idx])
        C = int(shape[axes.index("c")]) if "c" in axes else 1
        Z = int(shape[axes.index("z")]) if "z" in axes else 1
        T = int(shape[axes.index("t")]) if "t" in axes else 1
        py_um = info["scale_um"][y_idx]
        px_um = info["scale_um"][x_idx]
        dz_um = info["scale_um"][axes.index("z")] if "z" in axes else 1.0
        y_pos = info["translation_um"][y_idx]
        x_pos = info["translation_um"][x_idx]

        if reference is None:
            reference = {
                "axes": axes,
                "Y": Y,
                "X": X,
                "C": C,
                "Z": Z,
                "T": T,
                "py_um": py_um,
                "px_um": px_um,
                "dz_um": dz_um,
                "channel_names": info["channel_names"],
            }
        else:
            if (Y, X, C) != (reference["Y"], reference["X"], reference["C"]):
                raise ValueError(
                    f"Tile {tile_id} shape (Y={Y}, X={X}, C={C}) differs from "
                    f"reference (Y={reference['Y']}, X={reference['X']}, C={reference['C']})"
                )
            if Z != reference["Z"] or T != reference["T"]:
                raise ValueError(
                    f"Tile {tile_id} has Z={Z}, T={T} but reference has "
                    f"Z={reference['Z']}, T={reference['T']}"
                )
            if axes != reference["axes"]:
                raise ValueError(
                    f"Tile {tile_id} axes {axes!r} differ from reference {reference['axes']!r}"
                )

        tile_stores[tile_id] = info["handle"]
        tile_positions.append((float(y_pos), float(x_pos)))
        tile_identifiers.append(tile_id)

    assert reference is not None  # populated on first loop iteration

    channels = reference["C"]
    channel_names = list(reference["channel_names"]) if reference["channel_names"] else []
    if len(channel_names) < channels:
        channel_names.extend(f"ch{i}" for i in range(len(channel_names), channels))

    unique_regions: List[str] = []
    if layout == "hcs":
        seen = set()
        for tid in tile_identifiers:
            row = tid[0] if isinstance(tid, tuple) and tid else None
            if row is not None and row not in seen:
                unique_regions.append(row)
                seen.add(row)

    return {
        "n_tiles": len(tile_identifiers),
        "n_series": len(tile_identifiers),
        "shape": (reference["Y"], reference["X"]),
        "channels": channels,
        "channel_names": channel_names,
        "n_z": reference["Z"],
        "n_t": reference["T"],
        "dz_um": float(reference["dz_um"]),
        "time_dim": reference["T"],
        "position_dim": len(tile_identifiers),
        "pixel_size": (float(reference["py_um"]), float(reference["px_um"])),
        "tile_positions": tile_positions,
        "tile_identifiers": tile_identifiers,
        "unique_regions": unique_regions,
        "ngff_root": root_path,
        "tile_stores": tile_stores,
        "tile_axes": reference["axes"],
        "ngff_layout": layout,
    }


def _read_with_axes(
    ts_handle: ts.TensorStore,
    tile_axes: str,
    t: int,
    c: Optional[int],
    z: int,
    y: Optional[slice],
    x: Optional[slice],
    target_axes: Sequence[str],
) -> np.ndarray:
    """Read from a tensorstore and reorder/expand to ``target_axes``.

    Scalar indices (t, z, optionally c) drop those dims; slice indices keep
    them. Any target axis not present in ``tile_axes`` is added as a unit dim.
    """
    sel: List[Any] = []
    remaining: List[str] = []
    for ax in tile_axes:
        if ax == "t":
            sel.append(int(t))
        elif ax == "z":
            sel.append(int(z))
        elif ax == "c":
            if c is None:
                sel.append(slice(None))
                remaining.append("c")
            else:
                sel.append(int(c))
        elif ax == "y":
            sel.append(y if y is not None else slice(None))
            remaining.append("y")
        elif ax == "x":
            sel.append(x if x is not None else slice(None))
            remaining.append("x")
    arr = ts_handle[tuple(sel)].read().result()

    target = list(target_axes)
    target_in_remaining = [a for a in target if a in remaining]
    if target_in_remaining:
        perm = [remaining.index(a) for a in target_in_remaining]
        arr = np.transpose(arr, perm)

    # Insert unit dims for any target axes not present in the array
    for i, ax in enumerate(target):
        if ax not in remaining:
            arr = np.expand_dims(arr, axis=i)

    return arr


def read_ngff_ome_zarr_tile(
    tile_stores: Dict[Any, ts.TensorStore],
    tile_identifiers: List[Any],
    tile_axes: str,
    tile_idx: int,
    z_level: int = 0,
    time_idx: int = 0,
) -> np.ndarray:
    """Read all channels of a tile at (time_idx, z_level). Returns (C, Y, X) float32."""
    tile_id = tile_identifiers[tile_idx]
    ts_handle = tile_stores[tile_id]
    arr = _read_with_axes(
        ts_handle, tile_axes, time_idx, None, z_level, None, None, ("c", "y", "x")
    )
    return np.ascontiguousarray(arr).astype(np.float32)


def read_ngff_ome_zarr_region(
    tile_stores: Dict[Any, ts.TensorStore],
    tile_identifiers: List[Any],
    tile_axes: str,
    tile_idx: int,
    y_slice: slice,
    x_slice: slice,
    channel_idx: int = 0,
    z_level: int = 0,
    time_idx: int = 0,
) -> np.ndarray:
    """Read a single-channel region of a tile. Returns (1, h, w) float32."""
    tile_id = tile_identifiers[tile_idx]
    ts_handle = tile_stores[tile_id]
    arr = _read_with_axes(
        ts_handle, tile_axes, time_idx, channel_idx, z_level, y_slice, x_slice, ("y", "x")
    )
    return arr[np.newaxis, :, :].astype(np.float32)
