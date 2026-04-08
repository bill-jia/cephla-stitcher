#!/usr/bin/env python3
"""
Stitcher GUI - A simple interface for tile fusion of OME-TIFF files.
"""

import sys
import os
from pathlib import Path

# Fix Qt plugin path for conda environments on macOS
if sys.platform == "darwin" and "CONDA_PREFIX" in os.environ:
    conda_plugins = Path(os.environ["CONDA_PREFIX"]) / "plugins"
    if conda_plugins.exists() and "QT_PLUGIN_PATH" not in os.environ:
        os.environ["QT_PLUGIN_PATH"] = str(conda_plugins)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFileDialog,
    QProgressBar,
    QTextEdit,
    QFrame,
    QComboBox,
    QSlider,
    QRadioButton,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent

STYLE_SHEET = """
QGroupBox {
    font-weight: bold;
    margin-top: 12px;
    padding-top: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
}
QPushButton#runButton {
    background-color: #0071e3;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}
QPushButton#runButton:hover {
    background-color: #0077ed;
}
QPushButton#runButton:disabled {
    background-color: #c7c7cc;
}
QPushButton#napariButton {
    background-color: #34c759;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 10px 20px;
}
QPushButton#napariButton:hover {
    background-color: #30d158;
}
QPushButton#napariButton:disabled {
    background-color: #c7c7cc;
}
QPushButton#previewButton {
    background-color: #ff9500;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton#previewButton:hover {
    background-color: #ff9f0a;
}
QPushButton#previewButton:disabled {
    background-color: #c7c7cc;
}
QProgressBar {
    border: none;
    border-radius: 4px;
    height: 6px;
}
QProgressBar::chunk {
    background-color: #0071e3;
    border-radius: 4px;
}
QPushButton#calcFlatfieldButton {
    background-color: #5856d6;
    color: white;
    font-weight: bold;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
}
QPushButton#calcFlatfieldButton:hover {
    background-color: #6866e0;
}
QPushButton#calcFlatfieldButton:disabled {
    background-color: #c7c7cc;
}
"""


class PreviewWorker(QThread):
    """Worker thread for running preview stitching on subset of tiles."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)  # color_before, color_after, fused
    error = pyqtSignal(str)

    def __init__(
        self,
        tiff_path,
        preview_cols,
        preview_rows,
        downsample_factor,
        flatfield=None,
        darkfield=None,
        registration_z=None,
        registration_t=0,
        registration_channel=0,
    ):
        super().__init__()
        self.tiff_path = tiff_path
        self.preview_cols = preview_cols
        self.preview_rows = preview_rows
        self.downsample_factor = downsample_factor
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.registration_z = registration_z
        self.registration_t = registration_t
        self.registration_channel = registration_channel

    def run(self):
        try:
            import numpy as np
            from tilefusion import TileFusion

            self.progress.emit("Loading metadata...")

            # Create TileFusion instance - handles both OME-TIFF and SQUID formats
            tf_full = TileFusion(
                self.tiff_path,
                downsample_factors=(self.downsample_factor, self.downsample_factor),
                flatfield=self.flatfield,
                darkfield=self.darkfield,
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                channel_to_use=self.registration_channel,
            )

            positions = np.array(tf_full._tile_positions)

            # Build proper grid mapping for irregular grids
            unique_y = np.sort(np.unique(np.round(positions[:, 0], 0)))  # Y positions (rows)
            unique_x = np.sort(np.unique(np.round(positions[:, 1], 0)))  # X positions (cols)
            n_rows, n_cols = len(unique_y), len(unique_x)

            y_to_row = {y: i for i, y in enumerate(unique_y)}
            x_to_col = {x: i for i, x in enumerate(unique_x)}

            # Map (row, col) -> tile index
            grid = {}
            for idx, (y, x) in enumerate(positions):
                r = y_to_row[np.round(y, 0)]
                c = x_to_col[np.round(x, 0)]
                grid[(r, c)] = idx

            self.progress.emit(
                f"Grid: {n_cols}x{n_rows}, selecting center {self.preview_cols}x{self.preview_rows}"
            )

            center_row, center_col = n_rows // 2, n_cols // 2
            half_rows, half_cols = self.preview_rows // 2, self.preview_cols // 2

            selected_indices = []
            selected_grid_pos = []  # Track (row, col) for coloring
            for row in range(center_row - half_rows, center_row - half_rows + self.preview_rows):
                for col in range(
                    center_col - half_cols, center_col - half_cols + self.preview_cols
                ):
                    if (row, col) in grid:
                        selected_indices.append(grid[(row, col)])
                        selected_grid_pos.append(
                            (row - (center_row - half_rows), col - (center_col - half_cols))
                        )

            self.progress.emit(f"Selected {len(selected_indices)} tiles")

            original_positions = tf_full._tile_positions.copy()
            selected_positions = [original_positions[i] for i in selected_indices]

            # Create a new TileFusion for the subset
            tf = TileFusion(
                self.tiff_path,
                downsample_factors=(self.downsample_factor, self.downsample_factor),
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                channel_to_use=self.registration_channel,
            )
            tf._tile_positions = selected_positions
            tf.n_tiles = len(selected_indices)
            tf.position_dim = tf.n_tiles
            tf._tile_index_map = selected_indices

            # Store original read methods
            original_read_tile = tf._read_tile
            original_read_tile_region = tf._read_tile_region

            def patched_read_tile(tile_idx):
                real_idx = tf._tile_index_map[tile_idx]
                # Temporarily restore original method to read from full dataset
                return original_read_tile.__func__(tf_full, real_idx)

            def patched_read_tile_region(tile_idx, y_slice, x_slice):
                real_idx = tf._tile_index_map[tile_idx]
                return original_read_tile_region.__func__(tf_full, real_idx, y_slice, x_slice)

            tf._read_tile = patched_read_tile
            tf._read_tile_region = patched_read_tile_region

            self.progress.emit("Running registration...")
            tf.refine_tile_positions_with_cross_correlation()
            self.progress.emit(f"Found {len(tf.pairwise_metrics)} pairs")

            tf.optimize_shifts(
                method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True
            )
            global_offsets = tf.global_offsets

            pixel_size = tf._pixel_size
            min_y = min(p[0] for p in selected_positions)
            min_x = min(p[1] for p in selected_positions)
            max_y = max(p[0] for p in selected_positions) + tf.Y * pixel_size[0]
            max_x = max(p[1] for p in selected_positions) + tf.X * pixel_size[1]

            h = int((max_y - min_y) / pixel_size[0]) + 100
            w = int((max_x - min_x) / pixel_size[1]) + 100

            self.progress.emit(f"Creating preview images ({h}x{w})...")

            color_before = np.zeros((h, w, 3), dtype=np.uint8)
            color_after = np.zeros((h, w, 3), dtype=np.uint8)
            fused = np.zeros((h, w), dtype=np.float32)
            weight = np.zeros((h, w), dtype=np.float32)

            checkerboard_colors = [
                (255, 100, 100),
                (100, 255, 100),
                (100, 100, 255),
                (255, 255, 100),
                (255, 100, 255),
                (100, 255, 255),
            ]

            def get_color(row, col):
                return checkerboard_colors[((row % 2) * 3 + (col % 3)) % 6]

            # Read tiles using TileFusion's format-aware methods
            for i, (pos, orig_idx) in enumerate(zip(selected_positions, selected_indices)):
                arr = tf_full._read_tile(orig_idx)
                if arr.ndim == 3:
                    arr = arr[0]  # Take first channel for preview
                arr_raw = arr.astype(np.float32)

                p1, p99 = np.percentile(arr_raw, [2, 98])
                arr_norm = np.clip((arr_raw - p1) / (p99 - p1 + 1e-6), 0, 1)

                grid_row, grid_col = selected_grid_pos[i]
                color = get_color(grid_row, grid_col)

                oy_before = int(round((pos[0] - min_y) / pixel_size[0]))
                ox_before = int(round((pos[1] - min_x) / pixel_size[1]))
                oy_after = oy_before + int(global_offsets[i][0])
                ox_after = ox_before + int(global_offsets[i][1])

                th, tw = arr_norm.shape

                # BEFORE
                y1, y2 = max(0, oy_before), min(oy_before + th, h)
                x1, x2 = max(0, ox_before), min(ox_before + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_before[y1:y2, x1:x2, c] = (
                            arr_norm[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)

                # AFTER
                y1, y2 = max(0, oy_after), min(oy_after + th, h)
                x1, x2 = max(0, ox_after), min(ox_after + tw, w)
                if y2 > y1 and x2 > x1:
                    tile_h, tile_w = y2 - y1, x2 - x1
                    for c in range(3):
                        color_after[y1:y2, x1:x2, c] = (
                            arr_norm[:tile_h, :tile_w] * color[c]
                        ).astype(np.uint8)
                    fused[y1:y2, x1:x2] += arr_raw[:tile_h, :tile_w]
                    weight[y1:y2, x1:x2] += 1.0

            weight = np.maximum(weight, 1.0)
            fused = fused / weight

            self.progress.emit("Preview ready!")
            self.finished.emit(color_before, color_after, fused)

        except Exception as e:
            import traceback

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


def _run_fusion_pipeline(
    tiff_path,
    do_registration,
    blend_pixels,
    downsample_factor,
    fusion_mode,
    flatfield=None,
    darkfield=None,
    registration_z=None,
    registration_t=0,
    registration_channel=0,
    log_fn=None,
):
    """Shared stitching pipeline used by both single and batch workers.

    Returns the output path string. Raises on failure.
    """
    import gc
    import json
    import shutil
    import time

    import numpy as np
    from tilefusion import TileFusion

    def log(msg):
        if log_fn:
            log_fn(msg)

    p = Path(tiff_path)
    output_path = p.parent / f"{p.stem}_fused.ome.zarr"
    output_folder = p.parent / f"{p.stem}_fused"

    if output_path.exists():
        shutil.rmtree(output_path)
    if output_folder.exists():
        shutil.rmtree(output_folder)

    metrics_path = p.parent / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    for m in p.parent.glob("metrics_*.json"):
        m.unlink()

    step_start = time.time()
    tf = TileFusion(
        tiff_path,
        output_path=output_path,
        blend_pixels=blend_pixels,
        downsample_factors=(downsample_factor, downsample_factor),
        flatfield=flatfield,
        darkfield=darkfield,
        registration_z=registration_z,
        registration_t=registration_t,
        channel_to_use=registration_channel,
    )
    load_time = time.time() - step_start
    log(f"Loaded {tf.n_tiles} tiles ({tf.Y}x{tf.X}) [{load_time:.1f}s]")

    if len(tf._unique_regions) > 1:
        log(f"Multi-region dataset: {tf._unique_regions}")
        tf.stitch_all_regions()
        return str(output_folder)

    step_start = time.time()
    if do_registration:
        log("Computing registration...")
        tf.refine_tile_positions_with_cross_correlation()
        tf.save_pairwise_metrics(metrics_path)
        reg_time = time.time() - step_start
        log(f"Registration complete: {len(tf.pairwise_metrics)} pairs [{reg_time:.1f}s]")
    else:
        tf.threshold = 1.0
        log("Using stage positions (no registration)")

    step_start = time.time()
    log("Optimizing positions...")
    tf.optimize_shifts(method="TWO_ROUND_ITERATIVE", rel_thresh=0.5, abs_thresh=2.0, iterative=True)
    gc.collect()

    tf._tile_positions = [
        tuple(np.array(pos) + off * np.array(tf.pixel_size))
        for pos, off in zip(tf._tile_positions, tf.global_offsets)
    ]
    opt_time = time.time() - step_start
    log(f"Positions optimized [{opt_time:.1f}s]")

    step_start = time.time()
    log("Computing fused image space...")
    tf._compute_fused_image_space()
    tf._pad_to_chunk_multiple()
    log(f"Output size: {tf.padded_shape[0]} x {tf.padded_shape[1]}")

    scale0 = output_path / "scale0" / "image"
    scale0.parent.mkdir(parents=True, exist_ok=True)
    tf._create_fused_tensorstore(output_path=scale0)

    mode_label = "direct placement" if fusion_mode == "direct" else "blended"
    log(f"Fusing tiles ({mode_label})...")
    tf._fuse_tiles(mode=fusion_mode)
    fuse_time = time.time() - step_start
    log(f"Tiles fused [{fuse_time:.1f}s]")

    ngff = {
        "attributes": {"_ARRAY_DIMENSIONS": ["t", "c", "y", "x"]},
        "zarr_format": 3,
        "node_type": "group",
    }
    with open(output_path / "scale0" / "zarr.json", "w") as f:
        json.dump(ngff, f, indent=2)

    step_start = time.time()
    log("Building multiscale pyramid...")
    tf._create_multiscales(output_path, factors=tf.multiscale_factors)
    tf._generate_ngff_zarr3_json(output_path, resolution_multiples=tf.resolution_multiples)
    pyramid_time = time.time() - step_start
    log(f"Pyramid built [{pyramid_time:.1f}s]")

    return str(output_path)


class FusionWorker(QThread):
    """Worker thread for running tile fusion."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str, float)  # output_path, elapsed_time
    error = pyqtSignal(str)

    def __init__(
        self,
        tiff_path,
        do_registration,
        blend_pixels,
        downsample_factor,
        fusion_mode="blended",
        flatfield=None,
        darkfield=None,
        registration_z=None,
        registration_t=0,
        registration_channel=0,
    ):
        super().__init__()
        self.tiff_path = tiff_path
        self.do_registration = do_registration
        self.blend_pixels = blend_pixels
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.registration_z = registration_z
        self.registration_t = registration_t
        self.registration_channel = registration_channel
        self.output_path = None

    def run(self):
        try:
            import time

            start_time = time.time()
            self.progress.emit(f"Loading {self.tiff_path}...")

            self.output_path = _run_fusion_pipeline(
                self.tiff_path,
                self.do_registration,
                self.blend_pixels,
                self.downsample_factor,
                self.fusion_mode,
                flatfield=self.flatfield,
                darkfield=self.darkfield,
                registration_z=self.registration_z,
                registration_t=self.registration_t,
                registration_channel=self.registration_channel,
                log_fn=self.progress.emit,
            )

            elapsed_time = time.time() - start_time
            self.finished.emit(self.output_path, elapsed_time)

        except Exception as e:
            import traceback

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class BatchFusionWorker(QThread):
    """Worker thread for batch processing multiple folders/files."""

    progress = pyqtSignal(str)
    item_started = pyqtSignal(int, int, str)  # (current_index, total, item_name)
    item_finished = pyqtSignal(int, int)  # (current_index, total) for progress bar
    finished = pyqtSignal(int, int, float)  # (succeeded, failed, total_time)
    error = pyqtSignal(str)

    def __init__(
        self,
        paths,
        do_registration,
        blend_pixels,
        downsample_factor,
        fusion_mode="blended",
        flatfield=None,
        darkfield=None,
    ):
        super().__init__()
        self.paths = paths
        self.do_registration = do_registration
        self.blend_pixels = blend_pixels
        self.downsample_factor = downsample_factor
        self.fusion_mode = fusion_mode
        self.flatfield = flatfield
        self.darkfield = darkfield

    def _log(self, index, total, name, message):
        self.progress.emit(f"[{index + 1}/{total} {name}] {message}")

    def run(self):
        try:
            self._run_batch()
        except Exception as e:
            import traceback

            self.error.emit(f"Batch processing failed: {e}\n{traceback.format_exc()}")
            self.finished.emit(0, len(self.paths), 0.0)

    def _run_batch(self):
        import time

        total = len(self.paths)
        succeeded = 0
        failed = 0
        batch_start = time.time()

        for idx, tiff_path in enumerate(self.paths):
            name = Path(tiff_path).name
            self.item_started.emit(idx, total, name)

            try:

                def log_fn(msg, _idx=idx, _total=total, _name=name):
                    self._log(_idx, _total, _name, msg)

                _run_fusion_pipeline(
                    tiff_path,
                    self.do_registration,
                    self.blend_pixels,
                    self.downsample_factor,
                    self.fusion_mode,
                    flatfield=self.flatfield,
                    darkfield=self.darkfield,
                    log_fn=log_fn,
                )
                succeeded += 1
            except MemoryError:
                failed += 1
                self._log(idx, total, name, "FAILED: Out of memory. Stopping batch.")
                self.item_finished.emit(idx, total)
                break
            except Exception as e:
                import traceback

                failed += 1
                self._log(idx, total, name, f"FAILED: {e}")
                self._log(idx, total, name, traceback.format_exc())

            self.item_finished.emit(idx, total)

        total_time = time.time() - batch_start
        self.finished.emit(succeeded, failed, total_time)


class DropArea(QFrame):
    """Drag and drop area for files or folders. Supports single and multi-drop."""

    fileDropped = pyqtSignal(str)
    filesDropped = pyqtSignal(list)  # list of path strings (directories or .tif/.tiff files)
    _default_style = "border: 2px dashed #888; border-radius: 8px; background: #fafafa;"
    _hover_style = "border: 2px dashed #0071e3; border-radius: 8px; background: #e8f4ff;"
    _active_style = "border: 2px solid #34c759; border-radius: 8px; background: #f0fff4;"
    _warn_style = "border: 2px solid #ff9500; border-radius: 8px; background: #fff8f0;"

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(100)
        self.setStyleSheet(self._default_style)

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(12, 12, 12, 12)

        self.icon_label = QLabel("📂")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 28px; border: none; background: transparent;")
        layout.addWidget(self.icon_label)

        self.label = QLabel("Drop OME-TIFF or SQUID folder here\nor click to browse")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setWordWrap(True)
        self.label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.label)

        self.file_paths = []

    @property
    def file_path(self):
        return self.file_paths[0] if self.file_paths else None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        if self.file_path:
            self.setStyleSheet(self._active_style)
        else:
            self.setStyleSheet(self._default_style)

    def _is_valid_path(self, file_path):
        """Check if a path is a valid folder or TIFF file."""
        path = Path(file_path)
        return path.is_dir() or file_path.endswith((".tif", ".tiff"))

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            self.setStyleSheet(self._default_style)
            return

        valid_paths = []
        invalid_names = []
        for url in urls:
            file_path = url.toLocalFile()
            if self._is_valid_path(file_path):
                valid_paths.append(file_path)
            else:
                invalid_names.append(Path(file_path).name)

        if not valid_paths:
            self.setStyleSheet(self._default_style)
            return

        if len(valid_paths) == 1:
            self.setFile(valid_paths[0])
            self.fileDropped.emit(valid_paths[0])
        else:
            self.setFiles(valid_paths, invalid_names)
            self.filesDropped.emit(valid_paths)

    def mousePressEvent(self, event):
        from PyQt5.QtWidgets import QMenu

        menu = QMenu(self)
        file_action = menu.addAction("Select OME-TIFF file...")
        folder_action = menu.addAction("Select SQUID folder...")

        action = menu.exec_(self.mapToGlobal(event.pos()))

        if action == file_action:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select OME-TIFF file", "", "TIFF files (*.tif *.tiff);;All files (*.*)"
            )
            if file_path:
                self.setFile(file_path)
                self.fileDropped.emit(file_path)
        elif action == folder_action:
            folder_path = QFileDialog.getExistingDirectory(self, "Select SQUID folder")
            if folder_path:
                self.setFile(folder_path)
                self.fileDropped.emit(folder_path)

    def setFile(self, file_path):
        self.file_paths = [file_path]
        path = Path(file_path)
        self.setStyleSheet(self._active_style)
        self.icon_label.setText("✅")
        if path.is_dir():
            self.label.setText(f"📁 {path.name}")
        else:
            self.label.setText(path.name)

    def setFiles(self, paths, invalid_names=None):
        """Set multiple paths and update the display for batch mode."""
        self.file_paths = list(paths)
        names = [Path(p).name for p in paths]
        label_lines = f"📦 {len(paths)} items selected:\n" + "\n".join(f"  {n}" for n in names)
        if invalid_names:
            label_lines += f"\n⚠ Skipped: {', '.join(invalid_names)}"
            self.setStyleSheet(self._warn_style)
        else:
            self.setStyleSheet(self._active_style)
        self.icon_label.setText("✅")
        self.label.setText(label_lines)


class FlatfieldDropArea(QFrame):
    """Small drag and drop area for flatfield .npy files."""

    fileDropped = pyqtSignal(str)
    _default_style = "border: 2px dashed #888; border-radius: 8px; background: #fafafa;"
    _hover_style = "border: 2px dashed #5856d6; border-radius: 8px; background: #f0f0ff;"
    _active_style = "border: 2px solid #5856d6; border-radius: 8px; background: #f5f5ff;"

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self.setStyleSheet(self._default_style)

        layout = QHBoxLayout(self)
        layout.setSpacing(8)

        self.icon_label = QLabel("📄")
        self.icon_label.setStyleSheet("font-size: 20px; border: none; background: transparent;")
        layout.addWidget(self.icon_label)

        self.label = QLabel("Drop flatfield .npy here or click to browse")
        self.label.setStyleSheet("border: none; background: transparent;")
        layout.addWidget(self.label)
        layout.addStretch()

        self.file_path = None

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self._hover_style)

    def dragLeaveEvent(self, event):
        if self.file_path:
            self.setStyleSheet(self._active_style)
        else:
            self.setStyleSheet(self._default_style)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.endswith(".npy"):
                self.setFile(file_path)
                self.fileDropped.emit(file_path)
            else:
                self.setStyleSheet(self._default_style)
        else:
            self.setStyleSheet(self._default_style)

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select flatfield file", "", "NumPy files (*.npy);;All files (*.*)"
        )
        if file_path:
            self.setFile(file_path)
            self.fileDropped.emit(file_path)

    def setFile(self, file_path):
        self.file_path = file_path
        path = Path(file_path)
        self.setStyleSheet(self._active_style)
        self.icon_label.setText("✅")
        self.label.setText(path.name)

    def clear(self):
        self.file_path = None
        self.setStyleSheet(self._default_style)
        self.icon_label.setText("📄")
        self.label.setText("Drop flatfield .npy here or click to browse")


class FlatfieldWorker(QThread):
    """Worker thread for calculating flatfield using BaSiCPy."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object)  # flatfield, darkfield (or None)
    error = pyqtSignal(str)

    def __init__(self, file_path, n_samples=50, use_darkfield=False):
        super().__init__()
        self.file_path = file_path
        self.n_samples = n_samples
        self.use_darkfield = use_darkfield

    def run(self):
        try:
            import numpy as np
            from tilefusion import TileFusion, calculate_flatfield, HAS_BASICPY

            if not HAS_BASICPY:
                self.error.emit("BaSiCPy is not installed. Install with: pip install basicpy")
                return

            self.progress.emit("Loading metadata...")

            # Create TileFusion instance to read tiles.
            # NOTE: No flatfield/darkfield passed intentionally - flatfield estimation
            # must be performed on raw, uncorrected tiles.
            tf = TileFusion(self.file_path)

            # Determine how many tiles to sample
            n_tiles = tf.n_tiles
            n_samples = min(self.n_samples, n_tiles)

            self.progress.emit(f"Sampling {n_samples} tiles from {n_tiles} total...")

            # Random sample of tile indices
            rng = np.random.default_rng(42)
            sample_indices = rng.choice(n_tiles, size=n_samples, replace=False)
            sample_indices = sorted(sample_indices)

            # Read sampled tiles
            # NOTE: Using private method tf._read_tile intentionally.
            # FlatfieldWorker needs direct access to raw tile data for sampling.
            tiles = []
            for i, tile_idx in enumerate(sample_indices):
                self.progress.emit(f"Reading tile {i+1}/{n_samples}...")
                tile = tf._read_tile(tile_idx)
                tiles.append(tile)

            self.progress.emit("Calculating flatfield with BaSiCPy...")
            flatfield, darkfield = calculate_flatfield(
                tiles, use_darkfield=self.use_darkfield, constant_darkfield=True
            )

            self.progress.emit("Flatfield calculation complete!")
            self.finished.emit(flatfield, darkfield)

        except Exception as e:
            import traceback

            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")


class StitcherGUI(QMainWindow):
    """Main GUI window for the stitcher."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stitcher")
        self.setMinimumSize(580, 850)

        self.worker = None
        self.output_path = None
        self.regions = []  # List of region names for multi-region outputs
        self.is_multi_region = False

        # Batch processing state
        self.batch_paths = []

        # Flatfield correction state
        self.flatfield = None  # Shape (C, Y, X) or None
        self.darkfield = None  # Shape (C, Y, X) or None
        self.flatfield_worker = None

        # Dataset dimension state (for registration z/t selection)
        self.dataset_n_z = 1
        self.dataset_n_t = 1
        self.dataset_n_channels = 1
        self.dataset_channel_names = []

        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Input drop area (no wrapper group to avoid double border)
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        self.drop_area.filesDropped.connect(self.on_files_dropped)
        layout.addWidget(self.drop_area)

        # Preview section
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout(preview_group)

        preview_layout.addWidget(QLabel("Grid:"))

        self.preview_cols_spin = QSpinBox()
        self.preview_cols_spin.setRange(2, 15)
        self.preview_cols_spin.setValue(5)
        self.preview_cols_spin.setFixedWidth(55)
        preview_layout.addWidget(self.preview_cols_spin)

        preview_layout.addWidget(QLabel("x"))

        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(2, 15)
        self.preview_rows_spin.setValue(5)
        self.preview_rows_spin.setFixedWidth(55)
        preview_layout.addWidget(self.preview_rows_spin)

        preview_layout.addStretch()

        self.preview_button = QPushButton("👁 Preview")
        self.preview_button.setObjectName("previewButton")
        self.preview_button.setCursor(Qt.PointingHandCursor)
        self.preview_button.clicked.connect(self.run_preview)
        self.preview_button.setEnabled(False)
        preview_layout.addWidget(self.preview_button)

        layout.addWidget(preview_group)

        # Flatfield correction section
        flatfield_group = QGroupBox("Flatfield Correction")
        flatfield_layout = QVBoxLayout(flatfield_group)
        flatfield_layout.setSpacing(8)

        self.flatfield_checkbox = QCheckBox("Enable flatfield correction")
        self.flatfield_checkbox.setChecked(True)
        self.flatfield_checkbox.toggled.connect(self.on_flatfield_toggled)
        flatfield_layout.addWidget(self.flatfield_checkbox)

        # Container for flatfield options (shown when enabled)
        self.flatfield_options_widget = QWidget()
        flatfield_options_layout = QVBoxLayout(self.flatfield_options_widget)
        flatfield_options_layout.setContentsMargins(20, 0, 0, 0)
        flatfield_options_layout.setSpacing(8)

        # Radio buttons for Calculate vs Load
        self.flatfield_mode_group = QButtonGroup(self)
        radio_layout = QHBoxLayout()

        self.calc_radio = QRadioButton("Calculate from tiles")
        self.calc_radio.setChecked(True)
        self.flatfield_mode_group.addButton(self.calc_radio, 0)
        radio_layout.addWidget(self.calc_radio)

        self.load_radio = QRadioButton("Load from file")
        self.flatfield_mode_group.addButton(self.load_radio, 1)
        radio_layout.addWidget(self.load_radio)

        radio_layout.addStretch()
        flatfield_options_layout.addLayout(radio_layout)

        # Calculate options container
        self.calc_options_widget = QWidget()
        calc_options_layout = QVBoxLayout(self.calc_options_widget)
        calc_options_layout.setContentsMargins(0, 0, 0, 0)
        calc_options_layout.setSpacing(8)

        self.darkfield_checkbox = QCheckBox("Include darkfield correction")
        self.darkfield_checkbox.setChecked(False)
        calc_options_layout.addWidget(self.darkfield_checkbox)

        calc_btn_layout = QHBoxLayout()
        self.calc_flatfield_button = QPushButton("Calculate Flatfield")
        self.calc_flatfield_button.setObjectName("calcFlatfieldButton")
        self.calc_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.calc_flatfield_button.clicked.connect(self.calculate_flatfield)
        self.calc_flatfield_button.setEnabled(False)
        calc_btn_layout.addWidget(self.calc_flatfield_button)

        self.save_flatfield_button = QPushButton("Save")
        self.save_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.save_flatfield_button.clicked.connect(self.save_flatfield)
        self.save_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setToolTip("Save calculated flatfield to .npy file")
        calc_btn_layout.addWidget(self.save_flatfield_button)
        calc_btn_layout.addStretch()
        calc_options_layout.addLayout(calc_btn_layout)

        flatfield_options_layout.addWidget(self.calc_options_widget)

        # Load options container
        self.load_options_widget = QWidget()
        self.load_options_widget.setVisible(False)
        load_options_layout = QVBoxLayout(self.load_options_widget)
        load_options_layout.setContentsMargins(0, 0, 0, 0)

        self.flatfield_drop_area = FlatfieldDropArea()
        self.flatfield_drop_area.fileDropped.connect(self.on_flatfield_dropped)
        load_options_layout.addWidget(self.flatfield_drop_area)

        flatfield_options_layout.addWidget(self.load_options_widget)

        # Flatfield status and view button
        status_layout = QHBoxLayout()
        self.flatfield_status = QLabel("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        status_layout.addWidget(self.flatfield_status)

        self.view_flatfield_button = QPushButton("View")
        self.view_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.view_flatfield_button.clicked.connect(self.view_flatfield)
        self.view_flatfield_button.setEnabled(False)
        self.view_flatfield_button.setToolTip("View flatfield and darkfield")
        self.view_flatfield_button.setFixedWidth(60)
        status_layout.addWidget(self.view_flatfield_button)

        self.clear_flatfield_button = QPushButton("Clear")
        self.clear_flatfield_button.setCursor(Qt.PointingHandCursor)
        self.clear_flatfield_button.clicked.connect(self.clear_flatfield)
        self.clear_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setToolTip("Clear loaded flatfield")
        self.clear_flatfield_button.setFixedWidth(60)
        status_layout.addWidget(self.clear_flatfield_button)
        status_layout.addStretch()

        flatfield_options_layout.addLayout(status_layout)

        flatfield_layout.addWidget(self.flatfield_options_widget)

        self.flatfield_mode_group.buttonClicked.connect(self.on_flatfield_mode_changed)

        layout.addWidget(flatfield_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(8)

        self.registration_checkbox = QCheckBox("Enable registration refinement")
        self.registration_checkbox.setChecked(False)
        self.registration_checkbox.toggled.connect(self.on_registration_toggled)
        settings_layout.addWidget(self.registration_checkbox)

        # Downsample factor (shown when registration enabled)
        self.downsample_widget = QWidget()
        self.downsample_widget.setVisible(False)
        downsample_layout = QHBoxLayout(self.downsample_widget)
        downsample_layout.setContentsMargins(20, 0, 0, 0)
        downsample_layout.addWidget(QLabel("Downsample:"))
        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 8)
        self.downsample_spin.setValue(2)
        self.downsample_spin.setToolTip("Lower = slower but more accurate")
        downsample_layout.addWidget(self.downsample_spin)
        downsample_layout.addStretch()
        settings_layout.addWidget(self.downsample_widget)

        # Registration z/t selection (shown when registration enabled AND multi-z/t dataset)
        self.reg_zt_widget = QWidget()
        self.reg_zt_widget.setVisible(False)
        reg_zt_layout = QHBoxLayout(self.reg_zt_widget)
        reg_zt_layout.setContentsMargins(20, 0, 0, 0)
        self.reg_z_label = QLabel("Z-level:")
        reg_zt_layout.addWidget(self.reg_z_label)
        self.reg_z_spin = QSpinBox()
        self.reg_z_spin.setRange(0, 0)
        self.reg_z_spin.setValue(0)
        self.reg_z_spin.setToolTip("Z-level to use for registration")
        self.reg_z_spin.setFixedWidth(60)
        reg_zt_layout.addWidget(self.reg_z_spin)
        self.reg_t_label = QLabel("Timepoint:")
        reg_zt_layout.addWidget(self.reg_t_label)
        self.reg_t_spin = QSpinBox()
        self.reg_t_spin.setRange(0, 0)
        self.reg_t_spin.setValue(0)
        self.reg_t_spin.setToolTip("Timepoint to use for registration")
        self.reg_t_spin.setFixedWidth(60)
        reg_zt_layout.addWidget(self.reg_t_spin)
        self.reg_channel_label = QLabel("Channel:")
        reg_zt_layout.addWidget(self.reg_channel_label)
        self.reg_channel_combo = QComboBox()
        self.reg_channel_combo.setToolTip("Channel to use for registration")
        self.reg_channel_combo.setMinimumWidth(120)
        reg_zt_layout.addWidget(self.reg_channel_combo)
        reg_zt_layout.addStretch()
        settings_layout.addWidget(self.reg_zt_widget)

        self.blend_checkbox = QCheckBox("Enable blending")
        self.blend_checkbox.setChecked(False)
        self.blend_checkbox.toggled.connect(self.on_blend_toggled)
        settings_layout.addWidget(self.blend_checkbox)

        # Blend pixels (shown when blending enabled)
        self.blend_value_widget = QWidget()
        self.blend_value_widget.setVisible(False)
        blend_value_layout = QHBoxLayout(self.blend_value_widget)
        blend_value_layout.setContentsMargins(20, 0, 0, 0)
        blend_value_layout.addWidget(QLabel("Blend pixels:"))
        self.blend_spin = QSpinBox()
        self.blend_spin.setRange(1, 500)
        self.blend_spin.setValue(50)
        blend_value_layout.addWidget(self.blend_spin)
        blend_value_layout.addStretch()
        settings_layout.addWidget(self.blend_value_widget)

        layout.addWidget(settings_group)

        # Run button
        self.run_button = QPushButton("▶ Run Stitching")
        self.run_button.setObjectName("runButton")
        self.run_button.setMinimumHeight(40)
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self.run_stitching)
        self.run_button.setEnabled(False)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumHeight(6)
        layout.addWidget(self.progress_bar)

        # Log output
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(100)
        self.log_text.setMaximumHeight(140)
        self.log_text.setPlaceholderText("Log output...")
        layout.addWidget(self.log_text)

        # Region selection (for multi-region outputs)
        self.region_widget = QWidget()
        self.region_widget.setVisible(False)
        region_layout = QHBoxLayout(self.region_widget)
        region_layout.setContentsMargins(0, 0, 0, 0)
        region_layout.addWidget(QLabel("Region:"))
        self.region_combo = QComboBox()
        self.region_combo.setMinimumWidth(100)
        self.region_combo.currentIndexChanged.connect(self._on_region_combo_changed)
        region_layout.addWidget(self.region_combo)
        self.region_slider = QSlider(Qt.Horizontal)
        self.region_slider.setMinimum(0)
        self.region_slider.setMaximum(0)
        self.region_slider.valueChanged.connect(self._on_region_slider_changed)
        region_layout.addWidget(self.region_slider)
        layout.addWidget(self.region_widget)

        # Open in Napari button
        self.napari_button = QPushButton("🔬 Open in Napari")
        self.napari_button.setObjectName("napariButton")
        self.napari_button.setMinimumHeight(40)
        self.napari_button.setCursor(Qt.PointingHandCursor)
        self.napari_button.clicked.connect(self.open_in_napari)
        self.napari_button.setEnabled(False)
        layout.addWidget(self.napari_button)

        layout.addStretch()

    @property
    def is_batch_mode(self):
        return len(self.batch_paths) > 1

    def _update_batch_mode_ui(self):
        """Update UI to reflect batch vs single mode."""
        batch = self.is_batch_mode
        self.preview_button.setEnabled(not batch)
        self.calc_flatfield_button.setEnabled(not batch and self.drop_area.file_path is not None)
        self.reg_zt_widget.setEnabled(not batch)
        if batch:
            self.preview_button.setToolTip("Preview is not available in batch mode")
            self.calc_flatfield_button.setToolTip(
                "Calculate flatfield from a single dataset first, then load it for batch"
            )
            self.reg_zt_widget.setToolTip("Registration z/t/channel uses defaults in batch mode")
        else:
            self.preview_button.setToolTip("")
            self.calc_flatfield_button.setToolTip("")
            self.reg_zt_widget.setToolTip("")
            self.napari_button.setToolTip("")

    def on_file_dropped(self, file_path):
        """Handle single file/folder drop — exits batch mode."""
        self.batch_paths = []
        self._update_batch_mode_ui()

        path = Path(file_path)
        if path.is_dir():
            self.log(f"Selected SQUID folder: {file_path}")
        else:
            self.log(f"Selected OME-TIFF: {file_path}")
        self.run_button.setEnabled(True)
        self.preview_button.setEnabled(True)
        self.calc_flatfield_button.setEnabled(True)
        # Clear previous flatfield when new file is selected
        self.flatfield = None
        self.darkfield = None
        self.flatfield_status.setText("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        self.flatfield_drop_area.clear()
        self.view_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setEnabled(False)

        # Load dataset dimensions for registration z/t selection
        try:
            from tilefusion import TileFusion

            tf_temp = TileFusion(file_path)
            self.dataset_n_z = tf_temp.n_z
            self.dataset_n_t = tf_temp.n_t
            self.dataset_n_channels = tf_temp.channels
            if "channel_names" in tf_temp._metadata:
                self.dataset_channel_names = tf_temp._metadata["channel_names"]
            else:
                self.dataset_channel_names = [
                    f"Channel {i}" for i in range(self.dataset_n_channels)
                ]
            tf_temp.close()
            if self.dataset_n_z > 1 or self.dataset_n_t > 1:
                self.log(f"Dataset: {self.dataset_n_z} z-levels, {self.dataset_n_t} timepoints")
            self._update_reg_zt_controls()
        except Exception:
            self.dataset_n_z = 1
            self.dataset_n_t = 1
            self.dataset_n_channels = 1
            self.dataset_channel_names = []
            self._update_reg_zt_controls()

        # Auto-load existing flatfield if present, otherwise disable correction
        # For directories (SQUID folders), also check inside the directory
        if path.is_dir():
            flatfield_path = path / f"{path.name}_flatfield.npy"
            if not flatfield_path.exists():
                # Fallback: check next to the directory
                flatfield_path = path.parent / f"{path.name}_flatfield.npy"
        else:
            flatfield_path = path.parent / f"{path.stem}_flatfield.npy"

        if flatfield_path.exists():
            self.log(f"Found existing flatfield: {flatfield_path.name}")
            self.on_flatfield_dropped(str(flatfield_path))
            self.flatfield_drop_area.setFile(str(flatfield_path))
        else:
            self.flatfield_checkbox.setChecked(False)

    def on_files_dropped(self, paths):
        """Handle multi-drop — validate each path and enter batch mode."""
        from tilefusion import TileFusion

        self.log_text.clear()
        self.log(f"Validating {len(paths)} dropped items...")

        valid_paths = []
        invalid_names = []
        for p in paths:
            name = Path(p).name
            try:
                with TileFusion(p):
                    pass
                valid_paths.append(p)
                self.log(f"  ✓ {name}")
            except Exception as e:
                invalid_names.append(name)
                self.log(f"  ✗ {name}: {e}")

        if not valid_paths:
            self.log("No valid datasets found.")
            self.run_button.setEnabled(False)
            return

        if invalid_names:
            self.log(
                f"\n{len(valid_paths)} of {len(paths)} valid. "
                f"Skipped: {', '.join(invalid_names)}"
            )

        # Single valid item — fall back to normal single-item flow
        if len(valid_paths) == 1:
            self.log(f"\nOnly 1 valid item — using single mode.")
            self.drop_area.setFile(valid_paths[0])
            self.on_file_dropped(valid_paths[0])
            return

        # Multiple valid items — enter batch mode
        self.drop_area.setFiles(valid_paths, invalid_names)
        self.batch_paths = valid_paths
        self._update_batch_mode_ui()
        self.run_button.setEnabled(True)

        self.dataset_n_z = 1
        self.dataset_n_t = 1
        self.dataset_n_channels = 1
        self.dataset_channel_names = []

        if not invalid_names:
            self.log(f"\nAll {len(valid_paths)} items valid. Ready to run batch.")

    def on_registration_toggled(self, checked):
        self.downsample_widget.setVisible(checked)
        self._update_reg_zt_controls()

    def _update_reg_zt_controls(self):
        """Update visibility and ranges of registration z/t controls."""
        registration_enabled = self.registration_checkbox.isChecked()
        has_multi_z = self.dataset_n_z > 1
        has_multi_t = self.dataset_n_t > 1
        has_multi_channel = self.dataset_n_channels > 1

        # Show z/t widget only when registration is enabled AND dataset has multi-z or multi-t or multi-channel
        show_zt = registration_enabled and (has_multi_z or has_multi_t or has_multi_channel)
        self.reg_zt_widget.setVisible(show_zt)

        if show_zt:
            # Update z spinbox
            self.reg_z_label.setVisible(has_multi_z)
            self.reg_z_spin.setVisible(has_multi_z)
            if has_multi_z:
                self.reg_z_spin.setRange(0, self.dataset_n_z - 1)
                self.reg_z_spin.setValue(self.dataset_n_z // 2)  # Default to middle

            # Update t spinbox
            self.reg_t_label.setVisible(has_multi_t)
            self.reg_t_spin.setVisible(has_multi_t)
            if has_multi_t:
                self.reg_t_spin.setRange(0, self.dataset_n_t - 1)
                self.reg_t_spin.setValue(0)  # Default to first timepoint

            # Update channel combo
            self.reg_channel_label.setVisible(has_multi_channel)
            self.reg_channel_combo.setVisible(has_multi_channel)
            if has_multi_channel:
                self.reg_channel_combo.clear()
                self.reg_channel_combo.addItems(self.dataset_channel_names)
                self.reg_channel_combo.setCurrentIndex(0)

    def on_blend_toggled(self, checked):
        self.blend_value_widget.setVisible(checked)

    def on_flatfield_toggled(self, checked):
        # Only show/hide flatfield options; preserve any loaded/calculated data
        self.flatfield_options_widget.setVisible(checked)

    def on_flatfield_mode_changed(self, button):
        is_calculate = self.calc_radio.isChecked()
        self.calc_options_widget.setVisible(is_calculate)
        self.load_options_widget.setVisible(not is_calculate)

    def calculate_flatfield(self):
        if not self.drop_area.file_path:
            return

        self.calc_flatfield_button.setEnabled(False)
        self.flatfield_status.setText("Calculating flatfield...")
        self.flatfield_status.setStyleSheet("color: #ff9500; font-size: 11px;")

        self.flatfield_worker = FlatfieldWorker(
            self.drop_area.file_path,
            n_samples=50,
            use_darkfield=self.darkfield_checkbox.isChecked(),
        )
        self.flatfield_worker.progress.connect(self.log)
        self.flatfield_worker.finished.connect(self.on_flatfield_calculated)
        self.flatfield_worker.error.connect(self.on_flatfield_error)
        self.flatfield_worker.start()

    def on_flatfield_calculated(self, flatfield, darkfield):
        self.flatfield = flatfield
        self.darkfield = darkfield
        self.calc_flatfield_button.setEnabled(True)
        self.save_flatfield_button.setEnabled(True)
        self.view_flatfield_button.setEnabled(True)
        self.clear_flatfield_button.setEnabled(True)

        n_channels = flatfield.shape[0]
        status = f"Flatfield ready ({n_channels} channels)"
        if darkfield is not None:
            status += " + darkfield"
        self.flatfield_status.setText(status)
        self.flatfield_status.setStyleSheet("color: #34c759; font-size: 11px; font-weight: 600;")
        self.log(f"Flatfield calculation complete: {flatfield.shape}")

        # Auto-save flatfield next to input file
        if self.drop_area.file_path:
            try:
                from tilefusion import save_flatfield as save_ff

                input_path = Path(self.drop_area.file_path)
                # Use path.name for directories, path.stem for files (consistent with auto-load)
                if input_path.is_dir():
                    auto_save_path = input_path / f"{input_path.name}_flatfield.npy"
                else:
                    auto_save_path = input_path.parent / f"{input_path.stem}_flatfield.npy"
                save_ff(auto_save_path, self.flatfield, self.darkfield)
                self.log(f"Auto-saved flatfield to {auto_save_path}")
            except Exception as e:
                self.log(f"Warning: Could not auto-save flatfield: {e}")

    def save_flatfield(self):
        if self.flatfield is None:
            return

        # Default path based on input (consistent with auto-save/auto-load)
        default_path = "flatfield.npy"
        if self.drop_area.file_path:
            input_path = Path(self.drop_area.file_path)
            if input_path.is_dir():
                default_path = str(input_path / f"{input_path.name}_flatfield.npy")
            else:
                default_path = str(input_path.parent / f"{input_path.stem}_flatfield.npy")

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Flatfield",
            default_path,
            "NumPy files (*.npy);;All files (*.*)",
        )
        if file_path:
            try:
                from tilefusion import save_flatfield as save_ff

                save_ff(Path(file_path), self.flatfield, self.darkfield)
                self.log(f"Saved flatfield to {file_path}")
            except Exception as e:
                self.log(f"Error saving flatfield: {e}")

    def on_flatfield_error(self, error_msg):
        self.calc_flatfield_button.setEnabled(True)
        self.flatfield_status.setText("Calculation failed")
        self.flatfield_status.setStyleSheet("color: #ff3b30; font-size: 11px;")
        self.log(error_msg)

    def on_flatfield_dropped(self, file_path):
        import numpy as np

        try:
            from tilefusion import load_flatfield

            self.flatfield, self.darkfield = load_flatfield(Path(file_path))
            n_channels = self.flatfield.shape[0]
            status = f"Loaded ({n_channels} channels)"
            if self.darkfield is not None:
                status += " + darkfield"
            self.flatfield_status.setText(status)
            self.flatfield_status.setStyleSheet(
                "color: #34c759; font-size: 11px; font-weight: 600;"
            )
            self.view_flatfield_button.setEnabled(True)
            self.clear_flatfield_button.setEnabled(True)
            self.save_flatfield_button.setEnabled(True)
            # Enable flatfield correction when successfully loaded
            self.flatfield_checkbox.setChecked(True)
            self.log(f"Loaded flatfield from {file_path}: {self.flatfield.shape}")
        except Exception as e:
            # Clear any stale flatfield data on load failure
            self.flatfield = None
            self.darkfield = None
            self.flatfield_status.setText(f"Load failed: {e}")
            self.flatfield_status.setStyleSheet("color: #ff3b30; font-size: 11px;")
            self.view_flatfield_button.setEnabled(False)
            self.log(f"Error loading flatfield: {e}")

    def view_flatfield(self):
        if self.flatfield is None:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            import tempfile
            import subprocess

            n_channels = self.flatfield.shape[0]
            has_darkfield = self.darkfield is not None
            n_rows = 2 if has_darkfield else 1

            fig, axes = plt.subplots(n_rows, n_channels, figsize=(4 * n_channels, 4 * n_rows))

            # Handle single channel case (axes not 2D)
            if n_channels == 1 and n_rows == 1:
                axes = [[axes]]
            elif n_channels == 1:
                axes = [[ax] for ax in axes]
            elif n_rows == 1:
                axes = [axes]

            # First row: flatfield
            for ch in range(n_channels):
                ax = axes[0][ch]
                im = ax.imshow(self.flatfield[ch], cmap="viridis", vmin=0)
                ax.set_title(f"Flatfield Ch{ch}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Second row: darkfield (if available)
            if has_darkfield:
                for ch in range(n_channels):
                    ax = axes[1][ch]
                    im = ax.imshow(self.darkfield[ch], cmap="magma", vmin=0)
                    # Show constant value in title if darkfield is uniform
                    df_val = self.darkfield[ch].ravel()[0]
                    if np.allclose(self.darkfield[ch], df_val):
                        ax.set_title(f"Darkfield Ch{ch} (={df_val:.1f})")
                    else:
                        ax.set_title(f"Darkfield Ch{ch}")
                    ax.axis("off")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()

            # Save to temp file and open with system viewer
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                fig.savefig(f.name, dpi=150, bbox_inches="tight")
                plt.close(fig)
                # Open with default image viewer
                if sys.platform == "darwin":
                    subprocess.Popen(["open", f.name])
                elif sys.platform == "win32":
                    subprocess.Popen(["cmd", "/c", "start", "", f.name])
                else:
                    subprocess.Popen(["xdg-open", f.name])

            self.log("Opened flatfield viewer")
        except Exception as e:
            self.log(f"Error opening viewer: {e}")

    def clear_flatfield(self):
        """Clear loaded/calculated flatfield."""
        self.flatfield = None
        self.darkfield = None
        self.flatfield_status.setText("No flatfield")
        self.flatfield_status.setStyleSheet("color: #86868b; font-size: 11px;")
        self.view_flatfield_button.setEnabled(False)
        self.clear_flatfield_button.setEnabled(False)
        self.save_flatfield_button.setEnabled(False)
        self.flatfield_drop_area.clear()
        self.log("Flatfield cleared")

    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def run_stitching(self):
        if not self.drop_area.file_path:
            return

        self.run_button.setEnabled(False)
        self.napari_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()

        if self.blend_checkbox.isChecked():
            blend_val = self.blend_spin.value()
            blend_pixels = (blend_val, blend_val)
            fusion_mode = "blended"
        else:
            blend_pixels = (0, 0)
            fusion_mode = "direct"

        # Get flatfield if enabled
        flatfield = self.flatfield if self.flatfield_checkbox.isChecked() else None
        darkfield = self.darkfield if self.flatfield_checkbox.isChecked() else None

        if self.is_batch_mode:
            self._run_batch(blend_pixels, fusion_mode, flatfield, darkfield)
        else:
            self._run_single(blend_pixels, fusion_mode, flatfield, darkfield)

    def _run_single(self, blend_pixels, fusion_mode, flatfield, darkfield):
        # Get registration z/t values (None means use default middle z)
        registration_z = self.reg_z_spin.value() if self.dataset_n_z > 1 else None
        registration_t = self.reg_t_spin.value() if self.dataset_n_t > 1 else 0
        registration_channel = (
            self.reg_channel_combo.currentIndex() if self.dataset_n_channels > 1 else 0
        )

        self.worker = FusionWorker(
            self.drop_area.file_path,
            self.registration_checkbox.isChecked(),
            blend_pixels,
            self.downsample_spin.value(),
            fusion_mode,
            flatfield=flatfield,
            darkfield=darkfield,
            registration_z=registration_z,
            registration_t=registration_t,
            registration_channel=registration_channel,
        )
        self.worker.progress.connect(self.log)
        self.worker.finished.connect(self.on_fusion_finished)
        self.worker.error.connect(self.on_fusion_error)
        self.worker.start()

    def _run_batch(self, blend_pixels, fusion_mode, flatfield, darkfield):
        total = len(self.batch_paths)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        self.log(f"Starting batch processing: {total} items\n")

        self.worker = BatchFusionWorker(
            self.batch_paths,
            self.registration_checkbox.isChecked(),
            blend_pixels,
            self.downsample_spin.value(),
            fusion_mode,
            flatfield=flatfield,
            darkfield=darkfield,
        )
        self.worker.progress.connect(self.log)
        self.worker.error.connect(self.on_fusion_error)
        self.worker.item_started.connect(self._on_batch_item_started)
        self.worker.item_finished.connect(self._on_batch_item_finished)
        self.worker.finished.connect(self._on_batch_finished)
        self.worker.start()

    def _on_batch_item_started(self, index, total, name):
        self.log(f"\n{'='*40}")
        self.log(f"Processing {index + 1}/{total}: {name}")
        self.log(f"{'='*40}")

    def _on_batch_item_finished(self, index, total):
        self.progress_bar.setValue(index + 1)

    def _on_batch_finished(self, succeeded, failed, total_time):
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)  # Reset to indeterminate for next run
        self.batch_paths = []
        self.run_button.setEnabled(True)
        self.napari_button.setEnabled(True)
        self._update_batch_mode_ui()

        minutes = int(total_time // 60)
        seconds = total_time % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

        self.log(f"\n{'='*40}")
        self.log(f"Batch complete! {succeeded} succeeded, {failed} failed. Total time: {time_str}")
        self.log(f"{'='*40}")

    def on_fusion_finished(self, output_path, elapsed_time):
        self.output_path = output_path
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.napari_button.setEnabled(True)

        # Check if this is a multi-region output folder
        output_dir = Path(output_path)
        zarr_subdirs = sorted(output_dir.glob("*.ome.zarr"))
        if zarr_subdirs:
            # Multi-region output
            self.is_multi_region = True
            self.regions = [d.stem.replace(".ome", "") for d in zarr_subdirs]
            self.region_combo.blockSignals(True)
            self.region_combo.clear()
            self.region_combo.addItems(self.regions)
            self.region_combo.blockSignals(False)
            self.region_slider.setMaximum(len(self.regions) - 1)
            self.region_slider.setValue(0)
            self.region_widget.setVisible(True)
            self.log(f"Found {len(self.regions)} regions: {', '.join(self.regions)}")
        else:
            # Single output
            self.is_multi_region = False
            self.regions = []
            self.region_widget.setVisible(False)

        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        time_str = f"{minutes}m {seconds:.1f}s" if minutes > 0 else f"{seconds:.1f}s"

        self.log(f"\n✓ Fusion complete! Time: {time_str}\nOutput: {output_path}")

    def on_fusion_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.log(f"\n✗ {error_msg}")

    def run_preview(self):
        if not self.drop_area.file_path:
            return

        self.preview_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.log_text.clear()
        self.log("Starting preview...")

        # Get flatfield if enabled
        flatfield = self.flatfield if self.flatfield_checkbox.isChecked() else None
        darkfield = self.darkfield if self.flatfield_checkbox.isChecked() else None

        # Get registration z/t values (None means use default middle z)
        registration_z = self.reg_z_spin.value() if self.dataset_n_z > 1 else None
        registration_t = self.reg_t_spin.value() if self.dataset_n_t > 1 else 0
        registration_channel = (
            self.reg_channel_combo.currentIndex() if self.dataset_n_channels > 1 else 0
        )

        self.preview_worker = PreviewWorker(
            self.drop_area.file_path,
            self.preview_cols_spin.value(),
            self.preview_rows_spin.value(),
            self.downsample_spin.value(),
            flatfield=flatfield,
            darkfield=darkfield,
            registration_z=registration_z,
            registration_t=registration_t,
            registration_channel=registration_channel,
        )
        self.preview_worker.progress.connect(self.log)
        self.preview_worker.finished.connect(self.on_preview_finished)
        self.preview_worker.error.connect(self.on_preview_error)
        self.preview_worker.start()

    def on_preview_finished(self, color_before, color_after, fused):
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)

        self.log("Opening napari with before/after comparison...")

        try:
            import napari

            viewer = napari.Viewer()
            viewer.add_image(color_before, name="BEFORE registration (colored)", rgb=True)
            viewer.add_image(
                color_after, name="AFTER registration (colored)", rgb=True, visible=False
            )
            if fused is not None:
                viewer.add_image(fused, name="Fused result", colormap="gray", visible=False)
            napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}")

    def on_preview_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.preview_button.setEnabled(True)
        self.run_button.setEnabled(True)
        self.log(f"\n✗ {error_msg}")

    def _on_region_combo_changed(self, index):
        """Sync slider when dropdown changes."""
        self.region_slider.blockSignals(True)
        self.region_slider.setValue(index)
        self.region_slider.blockSignals(False)

    def _on_region_slider_changed(self, value):
        """Sync dropdown when slider changes."""
        self.region_combo.blockSignals(True)
        self.region_combo.setCurrentIndex(value)
        self.region_combo.blockSignals(False)

    def open_in_napari(self):
        if not self.output_path:
            try:
                import napari

                napari.Viewer()
                napari.run()
            except Exception as e:
                self.log(f"Error opening Napari: {e}")
            return

        # Determine the actual zarr path to open
        if self.is_multi_region and self.regions:
            selected_region = self.region_combo.currentText()
            zarr_path = Path(self.output_path) / f"{selected_region}.ome.zarr"
            self.log(f"Opening region '{selected_region}' in Napari: {zarr_path}")
        else:
            zarr_path = Path(self.output_path)
            self.log(f"Opening in Napari: {self.output_path}")

        try:
            import napari
            import tensorstore as ts
            import numpy as np

            viewer = napari.Viewer()
            output_path = zarr_path

            # Find all scale levels
            scale_dirs = sorted(output_path.glob("scale*"))
            pyramid_data = []

            for scale_dir in scale_dirs:
                image_path = scale_dir / "image"
                if image_path.exists():
                    store = ts.open(
                        {
                            "driver": "zarr3",
                            "kvstore": {"driver": "file", "path": str(image_path)},
                        }
                    ).result()
                    pyramid_data.append(store)

            if not pyramid_data:
                self.log("No image data found in output")
                return

            # Get shape from first level: (t, c, z, y, x) or (t, c, y, x)
            shape = pyramid_data[0].shape
            is_5d = len(shape) == 5
            n_channels = shape[1] if len(shape) >= 4 else 1
            n_z = shape[2] if is_5d else 1
            middle_z = n_z // 2

            # Get channel names if available
            channel_names = None
            try:
                from tilefusion import TileFusion

                tf = TileFusion(self.drop_area.file_path)
                if "channel_names" in tf._metadata:
                    channel_names = [ch.replace("_", " ") for ch in tf._metadata["channel_names"]]
            except:
                pass

            channel_colors = ["blue", "green", "yellow", "red", "magenta", "cyan"]

            def auto_contrast(data, pmax=99.9):
                """Compute contrast limits optimized for fluorescence microscopy.

                Uses mode-based background detection: finds the histogram peak
                (background) and sets minimum above it. This effectively
                suppresses background while preserving signal.
                """
                # Estimate background using histogram mode
                # Sample data for speed if large
                flat = data.ravel()
                if len(flat) > 100000:
                    flat = np.random.choice(flat, 100000, replace=False)

                # Find histogram peak (mode) - this is the background
                hist, bin_edges = np.histogram(flat, bins=256)
                mode_idx = np.argmax(hist)
                mode_val = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2

                # Estimate background noise (std of values below median)
                background_pixels = flat[flat <= np.median(flat)]
                if len(background_pixels) > 0:
                    bg_std = np.std(background_pixels)
                else:
                    bg_std = mode_val * 0.1

                # Set min to mode + 2*std (above background noise)
                lo = mode_val + 2 * bg_std
                hi = np.percentile(data, pmax)

                # Ensure minimum range
                if hi - lo < 10:
                    hi = lo + 100
                return [float(lo), float(hi)]

            def dtype_range(dtype):
                """Get valid range for a numpy dtype."""
                if np.issubdtype(dtype, np.integer):
                    info = np.iinfo(dtype)
                    return [info.min, info.max]
                elif np.issubdtype(dtype, np.floating):
                    return [0.0, 1.0]
                return [0, 65535]

            # Use lowest resolution level for fast auto-contrast computation
            lowest_res = pyramid_data[-1]

            # Check if we have multiple z or t
            has_zt_dims = is_5d and (n_z > 1 or shape[0] > 1)  # shape[0] is T

            if has_zt_dims:
                # Load full 5D data for z/t sliders (use only scale0 for memory)
                store = pyramid_data[0]
                self.log(f"Loading full volume: T={shape[0]}, C={n_channels}, Z={n_z}")

                for c in range(n_channels):
                    # Read full t, z for this channel: (T, Z, Y, X)
                    data = store[:, c, :, :, :].read().result()
                    data = np.asarray(data)

                    # Auto-contrast from middle slice
                    mid_t, mid_z = data.shape[0] // 2, data.shape[1] // 2
                    contrast = auto_contrast(data[mid_t, mid_z])

                    name = (
                        channel_names[c]
                        if channel_names and c < len(channel_names)
                        else f"Channel {c}"
                    )
                    layer = viewer.add_image(
                        data,
                        name=name,
                        colormap=channel_colors[c % len(channel_colors)],
                        blending="additive",
                        contrast_limits=contrast,
                    )
                    layer.contrast_limits_range = dtype_range(data.dtype)

                # Set axis labels for sliders after adding layers
                viewer.dims.axis_labels = ("t", "z", "y", "x")
            elif n_channels > 1:
                for c in range(n_channels):
                    # Read channel data from each pyramid level
                    channel_pyramid = []
                    for store in pyramid_data:
                        if is_5d:
                            data = store[0, c, middle_z, :, :].read().result()
                        else:
                            data = store[0, c, :, :].read().result()
                        channel_pyramid.append(np.asarray(data))

                    # Auto-contrast from lowest res level
                    contrast = auto_contrast(channel_pyramid[-1])

                    name = (
                        channel_names[c]
                        if channel_names and c < len(channel_names)
                        else f"Channel {c}"
                    )
                    layer = viewer.add_image(
                        channel_pyramid,
                        multiscale=True,
                        name=name,
                        colormap=channel_colors[c % len(channel_colors)],
                        blending="additive",
                        contrast_limits=contrast,
                    )
                    layer.contrast_limits_range = dtype_range(channel_pyramid[-1].dtype)
            else:
                # Single channel
                single_pyramid = []
                for store in pyramid_data:
                    if is_5d:
                        data = store[0, 0, middle_z, :, :].read().result()
                    else:
                        data = store[0, 0, :, :].read().result()
                    single_pyramid.append(np.asarray(data))

                contrast = auto_contrast(single_pyramid[-1])

                layer = viewer.add_image(
                    single_pyramid,
                    multiscale=True,
                    name=output_path.stem,
                    contrast_limits=contrast,
                )
                layer.contrast_limits_range = dtype_range(single_pyramid[-1].dtype)

            napari.run()
        except Exception as e:
            self.log(f"Error opening Napari: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = StitcherGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
