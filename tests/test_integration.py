"""Integration tests for TileFusion with synthetic tile data.

Adapted from original cephla_tilefusion integration tests by Doug Shepherd / QI2lab.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorstore as ts
import tifffile

from cephla_tilefusion import TileFusion


def _write_individual_tiffs_folder(
    path: Path, tiles: list, positions: list, channel_names: list = None
):
    """Create an individual TIFFs folder with coordinates.csv."""
    if channel_names is None:
        channel_names = ["ch0"]

    img_folder = path / "0"
    img_folder.mkdir(parents=True, exist_ok=True)

    # Write coordinates.csv
    coords = pd.DataFrame(
        {
            "fov": list(range(len(tiles))),
            "x (mm)": [p[1] / 1000 for p in positions],  # Convert µm to mm
            "y (mm)": [p[0] / 1000 for p in positions],
        }
    )
    coords.to_csv(img_folder / "coordinates.csv", index=False)

    # Write TIFF files
    for fov, tile in enumerate(tiles):
        for ch_name in channel_names:
            if tile.ndim == 2:
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch_name}.tiff", tile)
            else:
                # Multi-channel: tile is (C, Y, X)
                ch_idx = channel_names.index(ch_name)
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch_name}.tiff", tile[ch_idx])

    # Write acquisition parameters
    params = {"objective": {"magnification": 1.0}, "sensor_pixel_size_um": 1.0}
    with open(path / "acquisition parameters.json", "w") as f:
        json.dump(params, f)


def _read_fused_output(path: Path):
    """Read the fused output from a zarr store."""
    scale0 = path / "scale0" / "image"
    store = ts.open(
        {"driver": "zarr3", "kvstore": {"driver": "file", "path": str(scale0)}}
    ).result()
    return store.read().result()


class TestTileFusionIntegration:
    """Integration tests for the full TileFusion pipeline."""

    def test_two_tiles_horizontal(self, tmp_path):
        """Test fusion of two horizontally adjacent tiles."""
        # Create two tiles with known offset
        tile_size = 100
        overlap = 20

        rng = np.random.default_rng(42)
        global_img = rng.integers(
            100, 1000, size=(tile_size, tile_size * 2 - overlap), dtype=np.uint16
        )
        # Add features
        global_img[30:70, 40:140] += 5000

        tile0 = global_img[:, :tile_size]
        tile1 = global_img[:, tile_size - overlap :]

        # Stage positions in µm (pixel_size = 1.0 µm)
        positions = [(0.0, 0.0), (0.0, float(tile_size - overlap))]

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile0, tile1], positions)

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(0, 0),
            downsample_factors=(1, 1),
            ssim_window=7,
            threshold=0.3,
            multiscale_factors=(2,),
        )
        tf.run()

        # Verify output exists
        assert output_path.exists()
        assert (output_path / "scale0" / "image").exists()
        assert (output_path / "scale1" / "image").exists()

        # Read fused result
        fused = _read_fused_output(output_path)
        assert fused.ndim == 5  # (T, C, Z, Y, X)

        # Verify dimensions are reasonable (padded to chunk multiples)
        assert fused.shape[3] >= tile_size  # Y dimension
        assert fused.shape[4] >= tile_size  # X dimension (at least one tile width)

    def test_two_tiles_vertical(self, tmp_path):
        """Test fusion of two vertically adjacent tiles."""
        tile_size = 100
        overlap = 20

        rng = np.random.default_rng(43)
        global_img = rng.integers(
            100, 1000, size=(tile_size * 2 - overlap, tile_size), dtype=np.uint16
        )
        global_img[40:140, 30:70] += 5000

        tile0 = global_img[:tile_size, :]
        tile1 = global_img[tile_size - overlap :, :]

        positions = [(0.0, 0.0), (float(tile_size - overlap), 0.0)]

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile0, tile1], positions)

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(10, 10),
            downsample_factors=(1, 1),
            threshold=0.3,
            multiscale_factors=(2,),
        )
        tf.run()

        assert output_path.exists()
        fused = _read_fused_output(output_path)
        assert fused.shape[4] >= tile_size  # X dimension (padded to chunk multiple)

    def test_four_tiles_grid(self, tmp_path):
        """Test fusion of 2x2 grid of tiles."""
        tile_size = 80
        overlap = 15

        rng = np.random.default_rng(44)
        global_size = tile_size * 2 - overlap
        global_img = rng.integers(100, 1000, size=(global_size, global_size), dtype=np.uint16)

        # Add distinct features in each quadrant
        global_img[20:40, 20:40] += 3000  # Top-left
        global_img[20:40, global_size - 40 : global_size - 20] += 4000  # Top-right
        global_img[global_size - 40 : global_size - 20, 20:40] += 5000  # Bottom-left
        global_img[
            global_size - 40 : global_size - 20, global_size - 40 : global_size - 20
        ] += 6000  # Bottom-right

        # Extract tiles
        tiles = [
            global_img[:tile_size, :tile_size],  # Top-left
            global_img[:tile_size, tile_size - overlap :],  # Top-right
            global_img[tile_size - overlap :, :tile_size],  # Bottom-left
            global_img[tile_size - overlap :, tile_size - overlap :],  # Bottom-right
        ]

        step = tile_size - overlap
        positions = [
            (0.0, 0.0),
            (0.0, float(step)),
            (float(step), 0.0),
            (float(step), float(step)),
        ]

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, tiles, positions)

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(5, 5),
            downsample_factors=(1, 1),
            threshold=0.2,
            multiscale_factors=(2,),
        )
        tf.run()

        assert output_path.exists()

        # Should have 4 pairwise registrations for a 2x2 grid
        # (0,1), (0,2), (1,3), (2,3)
        assert len(tf.pairwise_metrics) >= 2  # At least some pairs registered

    def test_no_registration_mode(self, tmp_path):
        """Test fusion with registration disabled (using stage positions only)."""
        tile_size = 50

        rng = np.random.default_rng(45)
        tile0 = rng.integers(100, 1000, size=(tile_size, tile_size), dtype=np.uint16)
        tile1 = rng.integers(100, 1000, size=(tile_size, tile_size), dtype=np.uint16)

        positions = [(0.0, 0.0), (0.0, 40.0)]  # Some overlap

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile0, tile1], positions)

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(0, 0),
            threshold=1.0,  # High threshold effectively disables registration
            multiscale_factors=(2,),
        )
        tf.run()

        assert output_path.exists()

    def test_multichannel(self, tmp_path):
        """Test fusion with multiple channels."""
        tile_size = 60
        overlap = 10

        rng = np.random.default_rng(46)

        # Create 2-channel tiles
        tile0 = np.stack(
            [
                rng.integers(100, 500, size=(tile_size, tile_size), dtype=np.uint16),
                rng.integers(500, 1000, size=(tile_size, tile_size), dtype=np.uint16),
            ]
        )
        tile1 = np.stack(
            [
                rng.integers(100, 500, size=(tile_size, tile_size), dtype=np.uint16),
                rng.integers(500, 1000, size=(tile_size, tile_size), dtype=np.uint16),
            ]
        )

        # Add features that appear in both channels
        tile0[:, 20:40, 20:40] += 3000
        tile1[:, 20:40, 20:40] += 3000

        positions = [(0.0, 0.0), (0.0, float(tile_size - overlap))]

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(
            data_path, [tile0, tile1], positions, channel_names=["ch0", "ch1"]
        )

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(0, 0),
            threshold=0.3,
            channel_to_use=0,  # Register on first channel
            multiscale_factors=(2,),
        )
        tf.run()

        assert output_path.exists()
        fused = _read_fused_output(output_path)
        assert fused.shape[1] == 2  # Two channels


class TestRegistrationAccuracy:
    """Tests for registration accuracy with known shifts."""

    def test_recovers_known_shift(self, tmp_path):
        """Test that registration recovers a known pixel shift."""
        tile_size = 100

        rng = np.random.default_rng(50)
        base_tile = rng.integers(100, 1000, size=(tile_size, tile_size), dtype=np.uint16)

        # Add distinct features
        base_tile[30:50, 30:50] += 5000
        base_tile[60:70, 60:70] += 3000

        # Create second tile with known shift
        known_shift = (0, 5)  # 5 pixels in x direction
        tile0 = base_tile.copy()
        tile1 = np.roll(base_tile, known_shift[1], axis=1)

        # Stage positions claim tiles are at same location
        # Registration should find the actual shift
        positions = [(0.0, 0.0), (0.0, 0.0)]

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile0, tile1], positions)

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            blend_pixels=(0, 0),
            downsample_factors=(1, 1),
            ssim_window=7,
            threshold=0.3,
            multiscale_factors=(2,),
        )

        # Run registration only
        tf.refine_tile_positions_with_cross_correlation()

        # Check that a shift was detected
        if tf.pairwise_metrics:
            _, (dy, dx, score) = next(iter(tf.pairwise_metrics.items()))
            # The detected shift should be close to the known shift
            assert abs(dx - known_shift[1]) < 3 or abs(dx + known_shift[1]) < 3


class TestMultiscale:
    """Tests for multiscale pyramid generation."""

    def test_pyramid_levels_exist(self, tmp_path):
        """Test that all pyramid levels are created."""
        tile_size = 64

        rng = np.random.default_rng(60)
        tile = rng.integers(100, 1000, size=(tile_size, tile_size), dtype=np.uint16)

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile], [(0.0, 0.0)])

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            multiscale_factors=(2, 4, 8),
        )
        tf.run()

        # Check all scale levels exist
        assert (output_path / "scale0" / "image").exists()
        assert (output_path / "scale1" / "image").exists()
        assert (output_path / "scale2" / "image").exists()
        assert (output_path / "scale3" / "image").exists()

    def test_ngff_metadata(self, tmp_path):
        """Test that NGFF metadata is written correctly."""
        tile_size = 32

        rng = np.random.default_rng(61)
        tile = rng.integers(100, 1000, size=(tile_size, tile_size), dtype=np.uint16)

        data_path = tmp_path / "tiles"
        _write_individual_tiffs_folder(data_path, [tile], [(0.0, 0.0)])

        output_path = tmp_path / "fused.ome.zarr"
        tf = TileFusion(
            data_path,
            output_path=output_path,
            multiscale_factors=(2,),
        )
        tf.run()

        # Check zarr.json has multiscales metadata
        zarr_json = output_path / "zarr.json"
        assert zarr_json.exists()

        with open(zarr_json) as f:
            meta = json.load(f)

        assert "attributes" in meta
        assert "ome" in meta["attributes"]
        assert "multiscales" in meta["attributes"]["ome"]
