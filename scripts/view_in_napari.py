#!/usr/bin/env python3
"""
Simple script to view fused OME-Zarr in napari.
Works around napari-ome-zarr plugin issues with Zarr v3.
"""

import sys
from pathlib import Path

import napari
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def main():
    if len(sys.argv) > 1:
        zarr_path = Path(sys.argv[1])
    else:
        zarr_path = Path("data/ashlar/COLNOR69MW2-cycle-1.ome_fused.ome.zarr")

    if not zarr_path.exists():
        print(f"Error: {zarr_path} does not exist")
        sys.exit(1)

    print(f"Opening {zarr_path}...")
    reader = Reader(parse_url(str(zarr_path)))
    nodes = list(reader())

    if not nodes:
        print("Error: No data found in zarr")
        sys.exit(1)

    node = nodes[0]
    data = node.data

    print(f"Found {len(data)} resolution levels")
    print(f"Full resolution: {data[0].shape}")

    viewer = napari.Viewer()
    viewer.add_image(
        data,
        multiscale=True,
        name=zarr_path.stem,
        contrast_limits=[0, 65535],
    )
    print("Image loaded successfully")
    napari.run()


if __name__ == "__main__":
    main()
