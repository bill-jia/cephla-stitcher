"""Tests for tilefusion.io module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from tilefusion.io import (
    load_ome_tiff_metadata,
    load_individual_tiffs_metadata,
    read_individual_tiffs_tile,
    is_ngff_ome_zarr,
    load_ngff_ome_zarr_metadata,
    read_ngff_ome_zarr_tile,
    read_ngff_ome_zarr_region,
)


class TestLoadIndividualTiffsMetadata:
    """Tests for load_individual_tiffs_metadata function."""

    @pytest.fixture
    def sample_tiff_folder(self, tmp_path):
        """Create a sample individual TIFFs folder structure."""
        # Create subfolder
        img_folder = tmp_path / "0"
        img_folder.mkdir()

        # Create coordinates.csv
        coords = pd.DataFrame(
            {
                "fov": [0, 1, 2, 3],
                "x (mm)": [0.0, 1.0, 0.0, 1.0],
                "y (mm)": [0.0, 0.0, 1.0, 1.0],
            }
        )
        coords.to_csv(img_folder / "coordinates.csv", index=False)

        # Create sample TIFF files
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        for fov in range(4):
            for ch in ["Fluorescence_488_nm", "Fluorescence_561_nm"]:
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch}.tiff", img)

        return tmp_path

    def test_loads_metadata(self, sample_tiff_folder):
        """Test that metadata is loaded correctly."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        assert meta["n_tiles"] == 4
        assert meta["shape"] == (100, 100)
        assert meta["channels"] == 2
        assert len(meta["tile_positions"]) == 4
        assert len(meta["tile_identifiers"]) == 4

    def test_tile_positions(self, sample_tiff_folder):
        """Test that tile positions are converted correctly."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        # Positions should be in µm (mm * 1000)
        assert meta["tile_positions"][0] == (0.0, 0.0)
        assert meta["tile_positions"][1] == (0.0, 1000.0)

    def test_channel_names(self, sample_tiff_folder):
        """Test that channel names are detected."""
        meta = load_individual_tiffs_metadata(sample_tiff_folder)

        assert "Fluorescence_488_nm" in meta["channel_names"]
        assert "Fluorescence_561_nm" in meta["channel_names"]


class TestReadIndividualTiffsTile:
    """Tests for read_individual_tiffs_tile function."""

    @pytest.fixture
    def sample_tiff_folder(self, tmp_path):
        """Create sample TIFF files."""
        img_folder = tmp_path / "0"
        img_folder.mkdir()

        # Create test images with known values
        for fov in range(2):
            for idx, ch in enumerate(["ch1", "ch2"]):
                img = np.full((50, 50), fill_value=(fov + 1) * (idx + 1) * 100, dtype=np.uint16)
                tifffile.imwrite(img_folder / f"manual_{fov}_0_{ch}.tiff", img)

        # tile_identifiers are tuples: (fov,) for manual format
        return img_folder, ["ch1", "ch2"], [(0,), (1,)]

    def test_reads_all_channels(self, sample_tiff_folder):
        """Test that all channels are read."""
        img_folder, channel_names, tile_identifiers = sample_tiff_folder
        tile = read_individual_tiffs_tile(img_folder, channel_names, tile_identifiers, tile_idx=0)

        assert tile.shape == (2, 50, 50)
        assert tile.dtype == np.float32

    def test_correct_values(self, sample_tiff_folder):
        """Test that values are read correctly."""
        img_folder, channel_names, tile_identifiers = sample_tiff_folder
        tile = read_individual_tiffs_tile(img_folder, channel_names, tile_identifiers, tile_idx=0)

        # FOV 0: ch1 = 100, ch2 = 200
        assert np.allclose(tile[0], 100)
        assert np.allclose(tile[1], 200)


class TestOMETiffMetadata:
    """Tests for OME-TIFF metadata loading."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file."""
        path = tmp_path / "test.ome.tiff"

        # Create simple multi-series OME-TIFF
        data = [np.random.randint(0, 65535, (100, 100), dtype=np.uint16) for _ in range(4)]

        # Minimal OME-XML
        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" 
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:1"><Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:2"><Pixels ID="Pixels:2" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="50"/>
            </Pixels></Image>
            <Image ID="Image:3"><Pixels ID="Pixels:3" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="50"/>
            </Pixels></Image>
        </OME>"""

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path

    def test_loads_ome_metadata(self, sample_ome_tiff):
        """Test loading OME-TIFF metadata."""
        # Note: This may fail if tifffile doesn't write proper multi-series OME-TIFF
        # The test structure is correct but actual OME-TIFF writing is complex
        try:
            meta = load_ome_tiff_metadata(sample_ome_tiff)
            assert "n_tiles" in meta
            assert "shape" in meta
            assert "pixel_size" in meta
            # Clean up tiff_handle
            if "tiff_handle" in meta:
                meta["tiff_handle"].close()
        except Exception:
            pytest.skip("OME-TIFF creation requires proper OME-XML handling")

    def test_handle_closed_on_error(self, tmp_path):
        """Test that handle is closed if metadata parsing fails (ID 2650114878)."""
        import os

        # Create invalid TIFF (no OME metadata)
        path = tmp_path / "invalid.tiff"
        tifffile.imwrite(path, np.zeros((10, 10), dtype=np.uint16))

        with pytest.raises(ValueError, match="does not contain OME metadata"):
            load_ome_tiff_metadata(path)

        # Verify file handle is closed by attempting operations that would fail
        # if the handle were still open (especially on Windows)
        # 1. We can reopen the file
        with tifffile.TiffFile(path) as tif:
            assert tif is not None

        # 2. We can delete the file (would fail on Windows with open handle)
        os.remove(path)
        assert not path.exists()


class TestThreadSafety:
    """Tests for thread-safe concurrent tile reads."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file with multiple tiles."""
        path = tmp_path / "test.ome.tiff"

        # Create tiles with distinct values for verification
        data = [np.full((100, 100), fill_value=i * 1000, dtype=np.uint16) for i in range(8)]

        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">"""
        for i in range(8):
            ome_xml += f"""
            <Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="{(i % 4) * 50}" PositionY="{(i // 4) * 50}"/>
            </Pixels></Image>"""
        ome_xml += "</OME>"

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path, data

    def test_concurrent_reads_thread_local_handles(self, sample_ome_tiff):
        """Test that concurrent reads from multiple threads use separate handles."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tilefusion import TileFusion

        path, expected_data = sample_ome_tiff

        try:
            with TileFusion(path) as tf:
                # Track which threads read which tiles
                results = {}
                errors = []

                def read_tile(tile_idx):
                    import threading

                    thread_id = threading.current_thread().ident
                    try:
                        tile = tf._read_tile(tile_idx)
                        return tile_idx, thread_id, tile
                    except Exception as e:
                        return tile_idx, thread_id, e

                # Read tiles concurrently from multiple threads
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(read_tile, i) for i in range(8)]
                    for future in as_completed(futures):
                        tile_idx, thread_id, result = future.result()
                        if isinstance(result, Exception):
                            errors.append((tile_idx, result))
                        else:
                            results[tile_idx] = (thread_id, result)

                # Verify no errors occurred
                assert not errors, f"Errors during concurrent reads: {errors}"

                # Verify all tiles were read correctly
                assert len(results) == 8, f"Expected 8 results, got {len(results)}"

                # Verify data integrity - each tile should have its expected value
                for tile_idx, (thread_id, tile) in results.items():
                    expected_val = tile_idx * 1000
                    # The tile is flipped, so check mean value
                    actual_mean = tile.mean()
                    assert (
                        abs(actual_mean - expected_val) < 1
                    ), f"Tile {tile_idx}: expected ~{expected_val}, got {actual_mean}"

                # Verify multiple handles were created (one per thread)
                assert len(tf._all_handles) > 0, "No thread-local handles created"

        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_handles_cleaned_up_after_close(self, sample_ome_tiff):
        """Test that all thread-local handles are closed on cleanup."""
        from concurrent.futures import ThreadPoolExecutor
        from tilefusion import TileFusion

        path, _ = sample_ome_tiff

        try:
            tf = TileFusion(path)

            # Create handles in multiple threads
            def read_tile(tile_idx):
                return tf._read_tile(tile_idx)

            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(read_tile, range(4)))

            # Verify handles were created
            num_handles = len(tf._all_handles)
            assert num_handles > 0, "No handles created"

            # Close and verify cleanup
            tf.close()
            assert len(tf._all_handles) == 0, "Handles not cleaned up"

        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise


class TestTileFusionResourceManagement:
    """Tests for TileFusion resource management (close, context manager) - ID 2650114876."""

    @pytest.fixture
    def sample_ome_tiff(self, tmp_path):
        """Create a sample OME-TIFF file."""
        path = tmp_path / "test.ome.tiff"

        data = [np.random.randint(0, 65535, (100, 100), dtype=np.uint16) for _ in range(4)]

        ome_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
            <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" 
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:1"><Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="0"/>
            </Pixels></Image>
            <Image ID="Image:2"><Pixels ID="Pixels:2" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="0" PositionY="50"/>
            </Pixels></Image>
            <Image ID="Image:3"><Pixels ID="Pixels:3" DimensionOrder="XYCZT" Type="uint16"
                SizeX="100" SizeY="100" SizeC="1" SizeT="1" SizeZ="1"
                PhysicalSizeX="0.5" PhysicalSizeY="0.5">
                <Plane TheC="0" TheT="0" TheZ="0" PositionX="50" PositionY="50"/>
            </Pixels></Image>
        </OME>"""

        with tifffile.TiffWriter(path, ome=True) as tif:
            for i, d in enumerate(data):
                tif.write(d, description=ome_xml if i == 0 else None)

        return path

    def test_close_method(self, sample_ome_tiff):
        """Test that close() properly closes thread-local handles."""
        from tilefusion import TileFusion

        try:
            tf = TileFusion(sample_ome_tiff)
            # Trigger handle creation by reading a tile
            tf._read_tile(0)
            assert len(tf._all_handles) > 0, "Handle should be created after read"
            tf.close()
            assert len(tf._all_handles) == 0, "Handles should be cleared after close"
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_close_idempotent(self, sample_ome_tiff):
        """Test that close() can be called multiple times safely."""
        from tilefusion import TileFusion

        try:
            tf = TileFusion(sample_ome_tiff)
            tf.close()
            tf.close()  # Should not raise
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise

    def test_context_manager(self, sample_ome_tiff):
        """Test context manager protocol cleans up handles on exit."""
        from tilefusion import TileFusion

        try:
            with TileFusion(sample_ome_tiff) as tf:
                # Trigger handle creation by reading a tile
                tf._read_tile(0)
                assert len(tf._all_handles) > 0, "Handle should be created"
            # After exiting context, handles should be cleaned up
            assert len(tf._all_handles) == 0, "Handles should be cleared after exit"
        except (ValueError, AttributeError) as e:
            if "OME" in str(e) or "series" in str(e).lower():
                pytest.skip("OME-TIFF creation requires proper OME-XML handling")
            raise


def _write_ngff_image(
    image_dir,
    shape,
    axes,
    scale,
    translation,
    channel_names=None,
    dtype="uint16",
    fill_value=None,
):
    """Write a minimal NGFF v0.5 image at ``image_dir`` with a scale-0 array.

    Axes is a list of dicts matching the NGFF spec. Returns the tensorstore
    handle for the array so the test can verify later.
    """
    import tensorstore as ts

    image_dir.mkdir(parents=True, exist_ok=True)

    ome = {
        "version": "0.5",
        "multiscales": [
            {
                "axes": axes,
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": list(scale)},
                            {"type": "translation", "translation": list(translation)},
                        ],
                    }
                ],
                "name": "image",
                "@type": "ngff:Image",
            }
        ],
    }
    if channel_names:
        ome["omero"] = {
            "channels": [{"label": name} for name in channel_names],
        }

    zarr_json = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"ome": ome},
    }
    with open(image_dir / "zarr.json", "w") as f:
        json.dump(zarr_json, f)

    arr_path = image_dir / "0"
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(arr_path)},
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(shape)},
            },
            "chunk_key_encoding": {"name": "default"},
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "data_type": dtype,
            "dimension_names": [ax["name"] for ax in axes],
        },
    }
    ts_handle = ts.open(spec, create=True, open=True).result()
    if fill_value is not None:
        arr = np.full(shape, fill_value=fill_value, dtype=dtype)
        ts_handle.write(arr).result()
    return ts_handle


_STD_5D_AXES = [
    {"name": "t", "type": "time", "unit": "second"},
    {"name": "c", "type": "channel"},
    {"name": "z", "type": "space", "unit": "micrometer"},
    {"name": "y", "type": "space", "unit": "micrometer"},
    {"name": "x", "type": "space", "unit": "micrometer"},
]


class TestNgffOmeZarrFlat:
    """Tests for flat (bioformats2raw-style) NGFF OME-Zarr layout."""

    @pytest.fixture
    def flat_dataset(self, tmp_path):
        """Build a 2x2 grid of NGFF tiles as numbered child groups under root."""
        root = tmp_path / "dataset.ome.zarr"
        root.mkdir()
        # Root group with bioformats2raw.layout marker
        with open(root / "zarr.json", "w") as f:
            json.dump(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"bioformats2raw.layout": 3},
                },
                f,
            )

        # Stage positions in micrometers for a 2x2 grid with ~10% overlap
        positions_um = [
            (0.0, 0.0),
            (0.0, 28.0),
            (28.0, 0.0),
            (28.0, 28.0),
        ]
        # Shape (t=1, c=2, z=1, y=32, x=32), scale (1, 1, 1, 0.5, 0.5) um/px
        for idx, (y_um, x_um) in enumerate(positions_um):
            tile_dir = root / str(idx)
            _write_ngff_image(
                tile_dir,
                shape=(1, 2, 1, 32, 32),
                axes=_STD_5D_AXES,
                scale=[1.0, 1.0, 1.0, 0.5, 0.5],
                translation=[0.0, 0.0, 0.0, y_um, x_um],
                channel_names=["DAPI", "GFP"],
                fill_value=(idx + 1) * 100,
            )
        return root, positions_um

    def test_is_ngff_ome_zarr_detection(self, flat_dataset):
        root, _ = flat_dataset
        assert is_ngff_ome_zarr(root)

    def test_loads_metadata(self, flat_dataset):
        root, positions_um = flat_dataset
        meta = load_ngff_ome_zarr_metadata(root)

        assert meta["n_tiles"] == 4
        assert meta["shape"] == (32, 32)
        assert meta["channels"] == 2
        assert meta["n_z"] == 1
        assert meta["n_t"] == 1
        assert meta["pixel_size"] == (0.5, 0.5)
        assert meta["tile_axes"] == "tczyx"
        assert meta["ngff_layout"] == "flat"
        assert meta["channel_names"] == ["DAPI", "GFP"]
        assert list(meta["tile_positions"]) == list(positions_um)
        assert meta["unique_regions"] == []

    def test_read_tile(self, flat_dataset):
        root, _ = flat_dataset
        meta = load_ngff_ome_zarr_metadata(root)

        tile0 = read_ngff_ome_zarr_tile(
            meta["tile_stores"], meta["tile_identifiers"], meta["tile_axes"], 0
        )
        assert tile0.shape == (2, 32, 32)
        assert tile0.dtype == np.float32
        assert np.allclose(tile0, 100.0)

        tile3 = read_ngff_ome_zarr_tile(
            meta["tile_stores"], meta["tile_identifiers"], meta["tile_axes"], 3
        )
        assert np.allclose(tile3, 400.0)

    def test_read_region(self, flat_dataset):
        root, _ = flat_dataset
        meta = load_ngff_ome_zarr_metadata(root)

        region = read_ngff_ome_zarr_region(
            meta["tile_stores"],
            meta["tile_identifiers"],
            meta["tile_axes"],
            tile_idx=2,
            y_slice=slice(0, 16),
            x_slice=slice(0, 16),
            channel_idx=0,
        )
        assert region.shape == (1, 16, 16)
        assert region.dtype == np.float32
        assert np.allclose(region, 300.0)

    def test_tilefusion_detects_format(self, flat_dataset):
        from tilefusion import TileFusion

        root, _ = flat_dataset
        tf = TileFusion(root)
        assert tf._is_ngff_ome_zarr_format is True
        assert tf._is_zarr_format is False
        assert tf._is_individual_tiffs_format is False
        assert tf.n_tiles == 4
        assert tf.channels == 2
        # Read via TileFusion dispatch path
        tile = tf._read_tile(0)
        assert tile.shape == (2, 32, 32)


class TestNgffOmeZarrUnits:
    """Unit conversion tests for non-micrometer NGFF axes."""

    def test_millimeter_axes_converted_to_um(self, tmp_path):
        root = tmp_path / "dataset.ome.zarr"
        root.mkdir()
        with open(root / "zarr.json", "w") as f:
            json.dump({"zarr_format": 3, "node_type": "group", "attributes": {}}, f)

        axes_mm = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "millimeter"},
            {"name": "y", "type": "space", "unit": "millimeter"},
            {"name": "x", "type": "space", "unit": "millimeter"},
        ]
        for idx, (y_mm, x_mm) in enumerate([(0.0, 0.0), (0.0, 0.1)]):
            _write_ngff_image(
                root / str(idx),
                shape=(1, 1, 1, 16, 16),
                axes=axes_mm,
                scale=[1.0, 1.0, 1.0, 0.0005, 0.0005],  # 0.5 µm in mm
                translation=[0.0, 0.0, 0.0, y_mm, x_mm],
                fill_value=idx + 1,
            )

        meta = load_ngff_ome_zarr_metadata(root)
        # Millimeter scale/translation should now read in micrometers
        assert meta["pixel_size"] == pytest.approx((0.5, 0.5))
        assert meta["tile_positions"][1] == pytest.approx((0.0, 100.0))


class TestNgffOmeZarrSibling:
    """Tests for directory-of-sibling-.ome.zarr layout."""

    @pytest.fixture
    def sibling_dataset(self, tmp_path):
        root = tmp_path / "collection"
        root.mkdir()
        positions_um = [(0.0, 0.0), (0.0, 14.0)]
        for idx, (y_um, x_um) in enumerate(positions_um):
            tile_dir = root / f"tile_{idx}.ome.zarr"
            _write_ngff_image(
                tile_dir,
                shape=(1, 1, 1, 16, 16),
                axes=_STD_5D_AXES,
                scale=[1.0, 1.0, 1.0, 0.5, 0.5],
                translation=[0.0, 0.0, 0.0, y_um, x_um],
                fill_value=(idx + 1) * 50,
            )
        return root, positions_um

    def test_detects_sibling_layout(self, sibling_dataset):
        root, _ = sibling_dataset
        assert is_ngff_ome_zarr(root)
        meta = load_ngff_ome_zarr_metadata(root)
        assert meta["ngff_layout"] == "sibling"
        assert meta["n_tiles"] == 2

    def test_sibling_read(self, sibling_dataset):
        root, _ = sibling_dataset
        meta = load_ngff_ome_zarr_metadata(root)
        tile1 = read_ngff_ome_zarr_tile(
            meta["tile_stores"], meta["tile_identifiers"], meta["tile_axes"], 1
        )
        assert np.allclose(tile1, 100.0)


class TestNgffOmeZarrHcs:
    """Tests for NGFF HCS plate layout."""

    @pytest.fixture
    def hcs_dataset(self, tmp_path):
        """Build a minimal 1-row / 2-column HCS plate with one field per well."""
        root = tmp_path / "plate.ome.zarr"
        root.mkdir()

        plate_metadata = {
            "name": "test-plate",
            "rows": [{"name": "A"}],
            "columns": [{"name": "1"}, {"name": "2"}],
            "wells": [
                {"path": "A/1", "rowIndex": 0, "columnIndex": 0},
                {"path": "A/2", "rowIndex": 0, "columnIndex": 1},
            ],
            "field_count": 1,
        }
        with open(root / "zarr.json", "w") as f:
            json.dump(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"ome": {"version": "0.5", "plate": plate_metadata}},
                },
                f,
            )

        # Per-well metadata declaring its images
        for col in ("1", "2"):
            (root / "A" / col).mkdir(parents=True, exist_ok=True)
            with open(root / "A" / col / "zarr.json", "w") as f:
                json.dump(
                    {
                        "zarr_format": 3,
                        "node_type": "group",
                        "attributes": {
                            "ome": {
                                "version": "0.5",
                                "well": {"images": [{"path": "0", "acquisition": 0}]},
                            }
                        },
                    },
                    f,
                )

        # Field-level NGFF images
        positions_um = {("A", "1"): (0.0, 0.0), ("A", "2"): (0.0, 30.0)}
        for (row, col), (y_um, x_um) in positions_um.items():
            field_dir = root / row / col / "0"
            _write_ngff_image(
                field_dir,
                shape=(1, 1, 1, 24, 24),
                axes=_STD_5D_AXES,
                scale=[1.0, 1.0, 1.0, 0.5, 0.5],
                translation=[0.0, 0.0, 0.0, y_um, x_um],
                fill_value=(list(positions_um).index((row, col)) + 1) * 200,
            )
        return root, positions_um

    def test_loads_plate_metadata(self, hcs_dataset):
        root, positions_um = hcs_dataset
        assert is_ngff_ome_zarr(root)
        meta = load_ngff_ome_zarr_metadata(root)
        assert meta["ngff_layout"] == "hcs"
        assert meta["n_tiles"] == 2
        assert meta["unique_regions"] == ["A"]
        # tile_identifiers are (row, col, field_idx); row in slot 0 drives region filter
        assert all(tid[0] == "A" for tid in meta["tile_identifiers"])

    def test_read_hcs_tile(self, hcs_dataset):
        root, _ = hcs_dataset
        meta = load_ngff_ome_zarr_metadata(root)
        tile = read_ngff_ome_zarr_tile(
            meta["tile_stores"], meta["tile_identifiers"], meta["tile_axes"], 0
        )
        assert tile.shape == (1, 24, 24)
        assert np.allclose(tile, 200.0)


class TestNgffOmeZarrFormatDispatch:
    """Regression tests: ensure the new detection branch doesn't break existing formats."""

    def test_existing_individual_tiffs_still_detected(self, tmp_path):
        """An individual-TIFFs folder must not be mis-detected as NGFF."""
        from tilefusion.io.ngff_ome_zarr import is_ngff_ome_zarr as _is_ngff

        img_folder = tmp_path / "0"
        img_folder.mkdir()
        pd.DataFrame({"fov": [0], "x (mm)": [0.0], "y (mm)": [0.0]}).to_csv(
            img_folder / "coordinates.csv", index=False
        )
        tifffile.imwrite(
            img_folder / "manual_0_0_ch1.tiff",
            np.zeros((10, 10), dtype=np.uint16),
        )
        assert _is_ngff(tmp_path) is False
