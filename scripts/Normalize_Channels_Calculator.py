# Normalize Channels Calculator
# Version: 2.3.0
# Author: Rodrigo Zeba + ChatGPT
# License: MIT
#
# v2.3.0:
# - Added percentile comparison for P90, P95 and P98.
# - Added Use P90 / Use P95 / Use P98 buttons.
# - Apply now saves automatic percentile-suffixed outputs such as normalized_SHO_result_p95.fit.
# - Report now includes percentile comparison and interpretation.
#
# v2.2.3:
# - Compact visual adjustment: reduced main input/button heights.
# - Reduced Calculate/Apply width so they no longer touch each other.
# - Kept workflow buttons compact.
#
# v2.2.2:
# - Added Copy Formulas, Reset Formulas and Save Report TXT buttons.
# - Added starless-channel recommendation in the report.
# - Removed visible output filename field from v2.2 to preserve layout.
#
# v2.1.0:
# - Added preview zoom controls: Zoom -, Fit, 1:1 and Zoom +.
# - Added mouse wheel zoom centered under the cursor.
# - Improved pan behavior using QGraphicsView ScrollHandDrag.
# - Preview only auto-fits on resize while in Fit mode.
#
# v2.0.8:
# - Reworked layout for usability: left column for Input/Info, right area for Formulas/Preview.
# - Formula fields now sit in a horizontal row above the preview.
# - Calculate and Apply buttons are no longer clipped.
# - Apply now gives a clearer warning if formulas are empty.
#
# v2.0.7:
# - Increased main window height to give Preview and Info more vertical space.
# - Added minimum heights for Info and Preview panels.
# - Kept Input/Formulas controls at safe heights to avoid text clipping on Windows.
#
# v2.0.6:
# - Reworked layout according to mockup: Input and Formulas on top; Info and Preview below.
# - Input and Formulas are flat panels without QGroupBox titles, closer to the proposed design.
# - Preview is wide but lower, leaving room for the top controls.
#
# v2.0.5:
# - Reworked layout: Input and Formulas are now side by side.
# - Info panel now spans below both Input and Formulas.
# - Preview panel is slightly smaller, giving more room to controls and Info.
#
# v2.0.4:
# - Increased main window height and left panel width.
# - Added fixed/minimum heights to Input, Formulas and Info panels.
# - Added extra spacing in the Formulas panel to prevent overlap with Info.
# - Reduced QGroupBox inner padding slightly to improve usable space.
#
# v2.0.3:
# - Rebuilt the Formulas panel with a vertical layout to prevent button/input overlap.
# - Calculate and Apply now have enough vertical space.
# - Formula fields are easier to read.
#
# v2.0.2:
# - Moved file selection buttons to the right side of each file input.
# - Reduced vertical height of the input panel.
# - Added blue primary action buttons inspired by Siril's Scripts button.
#
# v2.0.1:
# - Fixed compressed/truncated inputs and buttons in the left panel.
# - Increased left panel width and minimum widget heights.
# - Added clearer section spacing.
#
# v2.0.0:
# - Migrated GUI from Tkinter to PyQt6.
# - Keeps the same core functionality as v1.0.5:
#   - Select Ha / SII / OIII mono FITS files
#   - Choose percentile
#   - Calculate medians, percentiles, signals and suggested SHO weights
#   - Edit R/G/B formulas manually
#   - Apply current formulas
#   - Save normalized_SHO_result.fit in Siril's working directory
#   - Open result in Siril
#   - Show internal simple autostretched preview
#
# Notes:
# - Preview is display-only and uses a simple linked autostretch.
# - The saved FITS is the composed/rescaled RGB result, not the preview image.

import os
import sys
import traceback

import numpy as np
import sirilpy as s

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "Normalize Channels Calculator"
VERSION = "2.3.0"
DEFAULT_OUTPUT = "normalized_SHO_result.fit"


def finite_1d(data):
    arr = np.asarray(data, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = arr.reshape(-1)
    else:
        arr = arr.ravel()
    arr = arr[np.isfinite(arr)]
    return arr


def as_2d_float(data, label):
    arr = np.asarray(data, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a mono / single-channel image. Got shape: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def calc_stats(data, percentile):
    arr = finite_1d(data)
    med = float(np.median(arr))
    pct = float(np.percentile(arr, percentile))
    signal = float(pct - med)
    return {
        "median": med,
        "percentile": pct,
        "signal": signal,
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def calculate_percentile_result(ha_data, sii_data, oiii_data, percentile, ha_green_factor=0.70):
    """Calculate medians, percentile signals and suggested SHO weights for one percentile."""
    ha_stats = calc_stats(ha_data, percentile)
    sii_stats = calc_stats(sii_data, percentile)
    oiii_stats = calc_stats(oiii_data, percentile)

    ha_sig = ha_stats["signal"]
    sii_sig = sii_stats["signal"]
    oiii_sig = oiii_stats["signal"]

    if ha_sig <= 0 or sii_sig <= 0 or oiii_sig <= 0:
        raise ValueError(
            f"One or more channels have non-positive signal estimate for P{percentile:g}. "
            "Try another percentile or inspect the channels."
        )

    weight_sii = safe_div(ha_sig, sii_sig)
    weight_ha = ha_green_factor
    weight_oiii = safe_div(ha_sig, oiii_sig)

    return {
        "percentile": float(percentile),
        "ha_stats": ha_stats,
        "sii_stats": sii_stats,
        "oiii_stats": oiii_stats,
        "weight_sii": weight_sii,
        "weight_ha": weight_ha,
        "weight_oiii": weight_oiii,
        "r_formula": f"(SII - med(SII)) * {weight_sii:.2f}",
        "g_formula": f"(Ha - med(Ha)) * {weight_ha:.2f}",
        "b_formula": f"(OIII - med(OIII)) * {weight_oiii:.2f}",
    }


def percentile_label(percentile):
    """Return a safe label like p90, p95, p98 or p92_5."""
    value = float(percentile)
    if value.is_integer():
        return f"p{int(value)}"
    return "p" + str(value).replace(".", "_")


def safe_div(numerator, denominator):
    if denominator <= 0:
        return None
    return numerator / denominator


def fmt(x, n=8):
    if x is None:
        return "N/A"
    return f"{x:.{n}f}"


def global_rescale_rgb(rgb):
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros_like(rgb, dtype=np.float32)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(rgb, dtype=np.float32)
    out = (rgb - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def clipped_zero_rgb(rgb):
    out = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def evaluate_formula(expr, ha, sii, oiii, label):
    if expr is None or not str(expr).strip():
        raise ValueError(f"{label} formula is empty.")

    expr = str(expr).strip().replace(",", ".")

    env = {
        "Ha": ha,
        "SII": sii,
        "OIII": oiii,
        "med": np.median,
        "median": np.median,
        "mean": np.mean,
        "min": np.min,
        "max": np.max,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "log": np.log,
        "log10": np.log10,
        "asinh": np.arcsinh,
        "clip": np.clip,
        "np": np,
    }

    try:
        result = eval(expr, {"__builtins__": {}}, env)
    except Exception as exc:
        raise ValueError(f"Error evaluating {label} formula:\n{expr}\n\n{exc}") from exc

    arr = np.asarray(result, dtype=np.float64)

    if arr.ndim == 0:
        raise ValueError(f"{label} formula returned a scalar value, not an image:\n{expr}")

    arr = np.squeeze(arr)

    if arr.shape != ha.shape:
        raise ValueError(
            f"{label} formula returned an image with wrong shape.\n"
            f"Expected: {ha.shape}\nGot: {arr.shape}\nFormula: {expr}"
        )

    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    return arr


def preview_qpixmap(rgb):
    arr = np.asarray(rgb, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None

    lo = float(np.percentile(finite, 0.5))
    hi = float(np.percentile(finite, 99.7))
    if hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return None

    disp = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    disp = np.power(disp, 1.0 / 2.2)
    disp8 = (disp * 255.0).clip(0, 255).astype(np.uint8)

    if disp8.ndim == 3 and disp8.shape[0] == 3:
        disp8 = np.transpose(disp8, (1, 2, 0))

    if disp8.ndim != 3 or disp8.shape[2] != 3:
        return None

    h, w, _ = disp8.shape
    disp8 = np.ascontiguousarray(disp8)
    bytes_per_line = 3 * w
    qimg = QImage(disp8.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class PreviewView(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item = None
        self.fit_mode = True
        self.zoom_step = 1.25
        self.current_zoom = 1.0

        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)

        # Better zoom behavior: mouse wheel zooms around cursor position.
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fit_to_view()

    def has_image(self):
        return self.pixmap_item is not None

    def fit_to_view(self):
        if not self.has_image():
            return

        self.fit_mode = True
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self.current_zoom = 1.0

    def actual_size(self):
        """Show the preview at 1 image pixel = 1 screen pixel."""
        if not self.has_image():
            return

        self.fit_mode = False
        self.resetTransform()
        self.current_zoom = 1.0
        self.centerOn(self.pixmap_item)

    def zoom_in(self):
        self._zoom(self.zoom_step)

    def zoom_out(self):
        self._zoom(1.0 / self.zoom_step)

    def _zoom(self, factor):
        if not self.has_image():
            return

        self.fit_mode = False
        self.scale(factor, factor)
        self.current_zoom *= factor

    def wheelEvent(self, event):
        if not self.has_image():
            super().wheelEvent(event)
            return

        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return

        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # Keep fitting only while in Fit mode. If the user zoomed manually,
        # do not reset their zoom when the window is resized.
        if self.fit_mode:
            self.fit_to_view()


class NormalizeChannelsWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.siril = s.SirilInterface()
        self.siril.connect()

        self.ha_path = ""
        self.sii_path = ""
        self.oiii_path = ""

        self.ha_fit = None
        self.sii_fit = None
        self.oiii_fit = None

        self.ha_stats = None
        self.sii_stats = None
        self.oiii_stats = None

        self.weight_sii = None
        self.weight_oiii = None
        self.weight_ha = 0.70
        self.last_calculated_formulas = {"r": "", "g": "", "b": ""}
        self.percentile_results = {}
        self.selected_percentile_label = "custom"
        self.selected_percentile_value = None

        self.last_rgb = None
        self.last_output_path = ""

        self.setWindowTitle(f"{APP_NAME} - v{VERSION}")
        self.resize(1240, 760)
        self.setMinimumSize(1120, 700)

        self._build_ui()
        self._apply_dark_style()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(24, 16, 24, 16)
        main_layout.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setObjectName("TitleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        instructions = QLabel(
            "Select Ha, SII and OIII mono FITS files, choose a percentile, then click Calculate. "
            "The script estimates background, signal and starting SHO weights. "
            "Apply creates an RGB FITS in Siril's working directory and shows a simple autostretched preview."
        )
        instructions.setObjectName("InstructionLabel")
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(title)
        main_layout.addWidget(instructions)

        # Usability-oriented layout:
        # left column  = Input + Info
        # right column = Formulas + Preview
        body = QHBoxLayout()
        body.setSpacing(18)
        main_layout.addLayout(body, stretch=1)

        # ============================================================
        # LEFT COLUMN: Input + Info
        # ============================================================
        left_col = QVBoxLayout()
        left_col.setSpacing(14)
        body.addLayout(left_col, stretch=0)

        input_panel = QWidget()
        input_panel.setObjectName("Panel")
        input_panel.setMinimumWidth(420)
        input_panel.setMaximumWidth(450)

        input_layout = QGridLayout(input_panel)
        input_layout.setContentsMargins(14, 12, 14, 12)
        input_layout.setHorizontalSpacing(10)
        input_layout.setVerticalSpacing(8)
        input_layout.setColumnStretch(1, 1)
        input_layout.setColumnMinimumWidth(2, 105)

        self.percentile_edit = QLineEdit("95")
        self.percentile_edit.setMinimumHeight(28)
        input_layout.addWidget(QLabel("Define the Percentile"), 0, 0, 1, 3)
        input_layout.addWidget(self.percentile_edit, 1, 0, 1, 3)

        self.ha_edit = QLineEdit()
        self.ha_edit.setReadOnly(True)
        self.ha_edit.setMinimumHeight(28)
        self.ha_btn = QPushButton("Select Ha File")
        self.ha_btn.setObjectName("SecondaryButton")
        self.ha_btn.setMinimumHeight(28)
        self.ha_btn.clicked.connect(self.select_ha)
        input_layout.addWidget(QLabel("Ha File"), 2, 0, 1, 3)
        input_layout.addWidget(self.ha_edit, 3, 0, 1, 2)
        input_layout.addWidget(self.ha_btn, 3, 2)

        self.sii_edit = QLineEdit()
        self.sii_edit.setReadOnly(True)
        self.sii_edit.setMinimumHeight(28)
        self.sii_btn = QPushButton("Select SII File")
        self.sii_btn.setObjectName("SecondaryButton")
        self.sii_btn.setMinimumHeight(28)
        self.sii_btn.clicked.connect(self.select_sii)
        input_layout.addWidget(QLabel("SII File"), 4, 0, 1, 3)
        input_layout.addWidget(self.sii_edit, 5, 0, 1, 2)
        input_layout.addWidget(self.sii_btn, 5, 2)

        self.oiii_edit = QLineEdit()
        self.oiii_edit.setReadOnly(True)
        self.oiii_edit.setMinimumHeight(28)
        self.oiii_btn = QPushButton("Select OIII File")
        self.oiii_btn.setObjectName("SecondaryButton")
        self.oiii_btn.setMinimumHeight(28)
        self.oiii_btn.clicked.connect(self.select_oiii)
        input_layout.addWidget(QLabel("OIII File"), 6, 0, 1, 3)
        input_layout.addWidget(self.oiii_edit, 7, 0, 1, 2)
        input_layout.addWidget(self.oiii_btn, 7, 2)

        left_col.addWidget(input_panel, stretch=0)

        info_panel = QWidget()
        info_panel.setObjectName("Panel")
        info_panel.setMinimumWidth(420)
        info_panel.setMaximumWidth(450)

        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(8)

        info_label = QLabel("Info")
        info_layout.addWidget(info_label)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(300)
        info_layout.addWidget(self.info_text, stretch=1)

        left_col.addWidget(info_panel, stretch=1)

        # ============================================================
        # RIGHT COLUMN: Formulas + Preview
        # ============================================================
        right_col = QVBoxLayout()
        right_col.setSpacing(14)
        body.addLayout(right_col, stretch=1)

        formula_panel = QWidget()
        formula_panel.setObjectName("Panel")
        formula_panel.setMinimumHeight(175)
        formula_panel.setMaximumHeight(205)

        formula_layout = QGridLayout(formula_panel)
        formula_layout.setContentsMargins(14, 12, 14, 12)
        formula_layout.setHorizontalSpacing(12)
        formula_layout.setVerticalSpacing(8)
        formula_layout.setColumnStretch(0, 1)
        formula_layout.setColumnStretch(1, 1)
        formula_layout.setColumnStretch(2, 1)

        self.r_formula = QLineEdit()
        self.g_formula = QLineEdit()
        self.b_formula = QLineEdit()
        self.r_formula.setMinimumHeight(28)
        self.g_formula.setMinimumHeight(28)
        self.b_formula.setMinimumHeight(28)

        formula_layout.addWidget(QLabel("R / SII Formula"), 0, 0)
        formula_layout.addWidget(QLabel("G / Ha Formula"), 0, 1)
        formula_layout.addWidget(QLabel("B / OIII Formula"), 0, 2)

        formula_layout.addWidget(self.r_formula, 1, 0)
        formula_layout.addWidget(self.g_formula, 1, 1)
        formula_layout.addWidget(self.b_formula, 1, 2)

        self.rescale_check = QCheckBox("Global rescale output")
        self.rescale_check.setChecked(True)
        formula_layout.addWidget(self.rescale_check, 2, 0)

        self.use_p90_btn = QPushButton("Use P90")
        self.use_p90_btn.setObjectName("PercentileButton")
        self.use_p90_btn.setMinimumHeight(24)
        self.use_p90_btn.setMinimumWidth(72)
        self.use_p90_btn.clicked.connect(lambda: self.use_percentile_result(90))

        self.use_p95_btn = QPushButton("Use P95")
        self.use_p95_btn.setObjectName("PercentileButton")
        self.use_p95_btn.setMinimumHeight(24)
        self.use_p95_btn.setMinimumWidth(72)
        self.use_p95_btn.clicked.connect(lambda: self.use_percentile_result(95))

        self.use_p98_btn = QPushButton("Use P98")
        self.use_p98_btn.setObjectName("PercentileButton")
        self.use_p98_btn.setMinimumHeight(24)
        self.use_p98_btn.setMinimumWidth(72)
        self.use_p98_btn.clicked.connect(lambda: self.use_percentile_result(98))

        self.copy_formulas_btn = QPushButton("Copy formulas")
        self.copy_formulas_btn.setObjectName("WorkflowButton")
        self.copy_formulas_btn.setMinimumHeight(26)
        self.copy_formulas_btn.setMinimumWidth(102)
        self.copy_formulas_btn.clicked.connect(self.copy_formulas)

        self.reset_formulas_btn = QPushButton("Reset formulas")
        self.reset_formulas_btn.setObjectName("WorkflowButton")
        self.reset_formulas_btn.setMinimumHeight(26)
        self.reset_formulas_btn.setMinimumWidth(106)
        self.reset_formulas_btn.clicked.connect(self.reset_formulas)

        self.save_report_btn = QPushButton("Save report TXT")
        self.save_report_btn.setObjectName("WorkflowButton")
        self.save_report_btn.setMinimumHeight(26)
        self.save_report_btn.setMinimumWidth(112)
        self.save_report_btn.clicked.connect(self.save_report_txt)

        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.setObjectName("PrimaryButton")
        self.calculate_btn.setMinimumHeight(26)
        self.calculate_btn.setMinimumWidth(92)
        self.calculate_btn.clicked.connect(self.calculate)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.setMinimumHeight(26)
        self.apply_btn.setMinimumWidth(82)
        self.apply_btn.clicked.connect(self.apply)

        percentile_row = QHBoxLayout()
        percentile_row.setSpacing(8)
        percentile_row.addWidget(QLabel("Use weights:"))
        percentile_row.addWidget(self.use_p90_btn)
        percentile_row.addWidget(self.use_p95_btn)
        percentile_row.addWidget(self.use_p98_btn)
        percentile_row.addStretch(1)

        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch(1)
        button_row.addWidget(self.copy_formulas_btn)
        button_row.addWidget(self.reset_formulas_btn)
        button_row.addWidget(self.save_report_btn)
        button_row.addWidget(self.calculate_btn)
        button_row.addWidget(self.apply_btn)

        formula_layout.addLayout(percentile_row, 2, 1, 1, 2)
        formula_layout.addLayout(button_row, 3, 0, 1, 3)

        right_col.addWidget(formula_panel, stretch=0)

        preview_panel = QWidget()
        preview_panel.setObjectName("Panel")

        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 12, 14, 12)
        preview_layout.setSpacing(8)

        preview_header = QHBoxLayout()

        preview_label = QLabel("Preview")
        preview_header.addWidget(preview_label)
        preview_header.addStretch(1)

        self.zoom_out_btn = QPushButton("-")
        self.zoom_out_btn.setObjectName("ToolButton")
        self.zoom_out_btn.setFixedWidth(42)
        self.zoom_out_btn.clicked.connect(lambda: self.preview.zoom_out())

        self.fit_btn = QPushButton("Fit")
        self.fit_btn.setObjectName("ToolButton")
        self.fit_btn.setFixedWidth(56)
        self.fit_btn.clicked.connect(lambda: self.preview.fit_to_view())

        self.actual_size_btn = QPushButton("1:1")
        self.actual_size_btn.setObjectName("ToolButton")
        self.actual_size_btn.setFixedWidth(56)
        self.actual_size_btn.clicked.connect(lambda: self.preview.actual_size())

        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setObjectName("ToolButton")
        self.zoom_in_btn.setFixedWidth(42)
        self.zoom_in_btn.clicked.connect(lambda: self.preview.zoom_in())

        preview_header.addWidget(self.zoom_out_btn)
        preview_header.addWidget(self.fit_btn)
        preview_header.addWidget(self.actual_size_btn)
        preview_header.addWidget(self.zoom_in_btn)

        self.pan_note = QLabel("(Mouse wheel zooms. Drag to pan when zoomed.)")
        self.pan_note.setObjectName("ZoomNote")
        preview_header.addWidget(self.pan_note)

        preview_layout.addLayout(preview_header)

        self.preview = PreviewView()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview.setMinimumHeight(420)
        preview_layout.addWidget(self.preview, stretch=1)

        self.preview_note = QLabel("Preview uses simple linked autostretch for display only.")
        self.preview_note.setObjectName("SmallNote")
        preview_layout.addWidget(self.preview_note)

        self.output_label = QLabel("")
        self.output_label.setObjectName("SmallNote")
        preview_layout.addWidget(self.output_label)

        right_col.addWidget(preview_panel, stretch=1)


    def _apply_dark_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #151515;
                color: #f0f0f0;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 10pt;
            }

            QLabel#TitleLabel {
                font-size: 19pt;
                color: #ffffff;
                padding: 2px;
            }

            QLabel#InstructionLabel {
                color: #d6d6d6;
                font-size: 9pt;
            }

            QLabel#SmallNote {
                color: #c8c8c8;
                font-size: 8pt;
            }

            QLabel#ZoomNote {
                color: #f0b642;
                font-size: 8pt;
                font-style: italic;
            }


            QWidget#Panel {
                background-color: #151515;
                border: 1px solid #444444;
                border-radius: 8px;
            }

            QGroupBox {
                border: 1px solid #444444;
                border-radius: 8px;
                margin-top: 14px;
                padding: 8px;
                color: #ffffff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
                color: #ffffff;
            }

            QLineEdit, QTextEdit {
                background-color: #3f3f3f;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 4px 6px;
                selection-background-color: #0078d4;
            }

            QLineEdit {
                min-height: 22px;
            }

            QLineEdit:read-only {
                color: #dddddd;
            }

            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 5px 7px;
                min-height: 22px;
            }

            QPushButton:hover {
                background-color: #4a4a4a;
            }

            QPushButton:pressed {
                background-color: #2d2d2d;
            }

            QPushButton#PrimaryButton {
                background-color: #0d6fd6;
                border: 1px solid #0d6fd6;
                color: #ffffff;
                font-weight: 600;
                padding-left: 8px;
                padding-right: 8px;
            }

            QPushButton#PrimaryButton:hover {
                background-color: #167eea;
                border: 1px solid #167eea;
            }

            QPushButton#PrimaryButton:pressed {
                background-color: #0a58aa;
                border: 1px solid #0a58aa;
            }

            QPushButton#SecondaryButton {
                background-color: #2f5f9f;
                border: 1px solid #2f5f9f;
                color: #ffffff;
            }

            QPushButton#SecondaryButton:hover {
                background-color: #3670bd;
                border: 1px solid #3670bd;
            }

            QPushButton#SecondaryButton:pressed {
                background-color: #254d83;
                border: 1px solid #254d83;
            }

            QPushButton#ToolButton {
                background-color: #3a3a3a;
                border: 1px solid #666666;
                color: #ffffff;
                font-weight: 600;
                padding: 5px;
            }

            QPushButton#ToolButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #777777;
            }

            QPushButton#ToolButton:pressed {
                background-color: #2d2d2d;
                border: 1px solid #555555;
            }

            QPushButton#WorkflowButton {
                background-color: #3a3a3a;
                border: 1px solid #666666;
                color: #ffffff;
                font-size: 8pt;
                font-weight: 600;
                padding: 3px;
                min-height: 20px;
            }

            QPushButton#WorkflowButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #777777;
            }

            QPushButton#WorkflowButton:pressed {
                background-color: #2d2d2d;
                border: 1px solid #555555;
            }

            QPushButton#PercentileButton {
                background-color: #2f5f9f;
                border: 1px solid #2f5f9f;
                color: #ffffff;
                font-size: 8pt;
                font-weight: 600;
                padding: 3px;
                min-height: 20px;
            }

            QPushButton#PercentileButton:hover {
                background-color: #3670bd;
                border: 1px solid #3670bd;
            }

            QPushButton#PercentileButton:pressed {
                background-color: #254d83;
                border: 1px solid #254d83;
            }

            QCheckBox {
                spacing: 6px;
            }

            QGraphicsView {
                background-color: #303030;
                border: 1px solid #444444;
                border-radius: 8px;
            }
        """)

    def _select_file(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "FITS files (*.fit *.fits *.fts);;All files (*.*)"
        )
        return path

    def _set_file_entry(self, edit, path):
        edit.setText(os.path.basename(path))
        edit.setToolTip(path)

    def select_ha(self):
        path = self._select_file("Select Ha mono FITS")
        if path:
            self.ha_path = path
            self._set_file_entry(self.ha_edit, path)

    def select_sii(self):
        path = self._select_file("Select SII mono FITS")
        if path:
            self.sii_path = path
            self._set_file_entry(self.sii_edit, path)

    def select_oiii(self):
        path = self._select_file("Select OIII mono FITS")
        if path:
            self.oiii_path = path
            self._set_file_entry(self.oiii_edit, path)

    def _log(self, message, level="info"):
        for method_name in ("log_info", "log", "info"):
            try:
                method = getattr(self.siril, method_name)
                method(str(message))
                return
            except Exception:
                pass
        return

    def _load_fit(self, path, label):
        if not path:
            raise ValueError(f"Select {label} file first.")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        fit = self.siril.load_image_from_file(path, with_pixels=True, preview=False)

        if fit is None or fit.data is None:
            raise ValueError(f"Could not load pixel data from {label}.")
        return fit

    def _get_percentile(self):
        try:
            pct = float(self.percentile_edit.text().strip().replace(",", "."))
        except Exception:
            raise ValueError("Percentile must be a number, e.g. 90, 95 or 98.")
        if pct <= 0 or pct >= 100:
            raise ValueError("Percentile must be greater than 0 and lower than 100.")
        return pct

    def _get_output_path(self):
        # v2.3 uses percentile-suffixed outputs to make comparison easier.
        label = self.selected_percentile_label or "custom"
        base_name = os.path.splitext(DEFAULT_OUTPUT)[0]
        output_filename = f"{base_name}_{label}.fit"

        try:
            wd = self.siril.get_siril_wd()
            if wd and os.path.isdir(wd):
                return os.path.join(wd, output_filename)
        except Exception:
            pass

        folder = os.path.dirname(self.ha_path) if self.ha_path else os.getcwd()
        return os.path.join(folder, output_filename)

    def calculate(self):
        try:
            pct = self._get_percentile()

            self.ha_fit = self._load_fit(self.ha_path, "Ha")
            self.sii_fit = self._load_fit(self.sii_path, "SII")
            self.oiii_fit = self._load_fit(self.oiii_path, "OIII")

            ha_data = as_2d_float(self.ha_fit.data, "Ha")
            sii_data = as_2d_float(self.sii_fit.data, "SII")
            oiii_data = as_2d_float(self.oiii_fit.data, "OIII")

            if ha_data.shape != sii_data.shape or ha_data.shape != oiii_data.shape:
                raise ValueError(
                    "All images must have the same dimensions.\n"
                    f"Ha: {ha_data.shape}\nSII: {sii_data.shape}\nOIII: {oiii_data.shape}"
                )

            # Calculate the user-selected percentile plus comparison presets.
            self.percentile_results = {}

            percentiles_to_compute = [90, 95, 98]
            if pct not in percentiles_to_compute:
                percentiles_to_compute.append(pct)

            for p in percentiles_to_compute:
                self.percentile_results[float(p)] = calculate_percentile_result(
                    ha_data, sii_data, oiii_data, p, self.weight_ha
                )

            selected_result = self.percentile_results[float(pct)]

            self.ha_stats = selected_result["ha_stats"]
            self.sii_stats = selected_result["sii_stats"]
            self.oiii_stats = selected_result["oiii_stats"]

            self.weight_sii = selected_result["weight_sii"]
            self.weight_ha = selected_result["weight_ha"]
            self.weight_oiii = selected_result["weight_oiii"]

            self.r_formula.setText(selected_result["r_formula"])
            self.g_formula.setText(selected_result["g_formula"])
            self.b_formula.setText(selected_result["b_formula"])

            self.selected_percentile_value = float(pct)
            self.selected_percentile_label = percentile_label(pct)

            self.last_calculated_formulas = {
                "r": self.r_formula.text(),
                "g": self.g_formula.text(),
                "b": self.b_formula.text(),
            }

            report = self._make_report(pct)
            self.info_text.setPlainText(report)
            self._log("Normalize Channels Calculator: calculation complete.")

        except Exception as e:
            self._show_error("Calculate error", e)

    def _make_report(self, pct):
        selected = self.percentile_results.get(float(pct))
        if selected is None:
            raise ValueError(f"No percentile result found for P{pct:g}.")

        ha = selected["ha_stats"]
        sii = selected["sii_stats"]
        oiii = selected["oiii_stats"]

        comparison_lines = []
        for p in [90, 95, 98]:
            result = self.percentile_results.get(float(p))
            if result is None:
                continue
            comparison_lines.append(
                f"P{p} suggested weights:\n"
                f"R / SII:  {result['weight_sii']:.3f}\n"
                f"G / Ha:   {result['weight_ha']:.3f}  (artistic green reduction)\n"
                f"B / OIII: {result['weight_oiii']:.3f}\n"
            )

        comparison_text = "\n".join(comparison_lines)

        return f"""Normalize Channels Calculator v{VERSION}

Selected files:
Ha:   {self.ha_path}
SII:  {self.sii_path}
OIII: {self.oiii_path}

Selected percentile: P{pct:.1f}
Selected output suffix: _{self.selected_percentile_label}

=== Ha ===
Median: {fmt(ha['median'], 10)}
P{pct:.1f}:   {fmt(ha['percentile'], 10)}
Signal estimate P{pct:.1f} - median: {fmt(ha['signal'], 10)}
Mean:   {fmt(ha['mean'], 10)}
Min:    {fmt(ha['min'], 10)}
Max:    {fmt(ha['max'], 10)}

=== SII ===
Median: {fmt(sii['median'], 10)}
P{pct:.1f}:   {fmt(sii['percentile'], 10)}
Signal estimate P{pct:.1f} - median: {fmt(sii['signal'], 10)}
Mean:   {fmt(sii['mean'], 10)}
Min:    {fmt(sii['min'], 10)}
Max:    {fmt(sii['max'], 10)}

=== OIII ===
Median: {fmt(oiii['median'], 10)}
P{pct:.1f}:   {fmt(oiii['percentile'], 10)}
Signal estimate P{pct:.1f} - median: {fmt(oiii['signal'], 10)}
Mean:   {fmt(oiii['mean'], 10)}
Min:    {fmt(oiii['min'], 10)}
Max:    {fmt(oiii['max'], 10)}

=== Selected suggested SHO weights ===
R / SII:  {selected['weight_sii']:.3f}
G / Ha:   {selected['weight_ha']:.3f}  (artistic green reduction)
B / OIII: {selected['weight_oiii']:.3f}

=== Selected Pixel Math formulas ===
R = {selected['r_formula']}
G = {selected['g_formula']}
B = {selected['b_formula']}

=== Percentile comparison ===
{comparison_text}
Interpretation:
- P90 is more conservative and less affected by bright structures/residual stars.
- P95 is the balanced default starting point.
- P98 is more aggressive, but can be biased by bright stars, highlights or star-removal artifacts.

Notes:
- Median is used as an estimated background pedestal.
- P{pct:.1f} - median is used as an estimated useful signal above background.
- Weights are a starting point, not a final truth.
- Ha is reduced because in SHO it maps to green and often dominates visually.
- Apply uses the current text in the R/G/B formula fields, so manual edits are used.
- Apply uses a simple global rescale after composition by default.
- Preview uses a simple linked autostretch for display only.
- Recommended: use starless mono channels when possible. Bright stars can bias percentile-based signal estimates, especially P95/P98.
"""

    def use_percentile_result(self, percentile):
        if not self.percentile_results:
            self._show_error("Use percentile", ValueError("No percentile results available. Click Calculate first."))
            return

        result = self.percentile_results.get(float(percentile))
        if result is None:
            self._show_error("Use percentile", ValueError(f"P{percentile:g} was not calculated. Click Calculate first."))
            return

        self.r_formula.setText(result["r_formula"])
        self.g_formula.setText(result["g_formula"])
        self.b_formula.setText(result["b_formula"])

        self.ha_stats = result["ha_stats"]
        self.sii_stats = result["sii_stats"]
        self.oiii_stats = result["oiii_stats"]

        self.weight_sii = result["weight_sii"]
        self.weight_ha = result["weight_ha"]
        self.weight_oiii = result["weight_oiii"]

        self.selected_percentile_value = float(percentile)
        self.selected_percentile_label = percentile_label(percentile)

        self.last_calculated_formulas = {
            "r": self.r_formula.text(),
            "g": self.g_formula.text(),
            "b": self.b_formula.text(),
        }

        self.info_text.setPlainText(self._make_report(float(percentile)))
        self.output_label.setText(f"Using P{percentile:g} weights. Apply will save with suffix _{self.selected_percentile_label}.")

    def _formula_text_block(self):
        return (
            f"R = {self.r_formula.text().strip()}\n"
            f"G = {self.g_formula.text().strip()}\n"
            f"B = {self.b_formula.text().strip()}"
        )

    def copy_formulas(self):
        if not self.r_formula.text().strip() or not self.g_formula.text().strip() or not self.b_formula.text().strip():
            self._show_error("Copy formulas", ValueError("Formula fields are empty. Click Calculate first or type formulas manually."))
            return

        QApplication.clipboard().setText(self._formula_text_block())
        self.output_label.setText("Formulas copied to clipboard.")

    def reset_formulas(self):
        if not any(self.last_calculated_formulas.values()):
            self._show_error("Reset formulas", ValueError("No calculated formulas available. Click Calculate first."))
            return

        self.r_formula.setText(self.last_calculated_formulas.get("r", ""))
        self.g_formula.setText(self.last_calculated_formulas.get("g", ""))
        self.b_formula.setText(self.last_calculated_formulas.get("b", ""))
        self.output_label.setText("Formulas reset to last calculated values.")

    def save_report_txt(self):
        report = self.info_text.toPlainText().strip()

        if not report:
            self._show_error("Save report TXT", ValueError("Report is empty. Click Calculate first."))
            return

        default_name = "normalize_channels_report.txt"
        try:
            wd = self.siril.get_siril_wd()
            if wd and os.path.isdir(wd):
                default_path = os.path.join(wd, default_name)
            else:
                default_path = os.path.join(os.path.dirname(self.ha_path), default_name) if self.ha_path else default_name
        except Exception:
            default_path = default_name

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save report TXT",
            default_path,
            "Text files (*.txt);;All files (*.*)"
        )

        if not path:
            return

        if not path.lower().endswith(".txt"):
            path += ".txt"

        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
            f.write("\n\n")
            f.write("Current formulas:\n")
            f.write(self._formula_text_block())
            f.write("\n")

        self.output_label.setText(f"Report saved: {path}")


    def _compose_rgb(self):
        if self.ha_fit is None or self.sii_fit is None or self.oiii_fit is None:
            self.calculate()

        ha = as_2d_float(self.ha_fit.data, "Ha")
        sii = as_2d_float(self.sii_fit.data, "SII")
        oiii = as_2d_float(self.oiii_fit.data, "OIII")

        r = evaluate_formula(self.r_formula.text(), ha, sii, oiii, "R / SII")
        g = evaluate_formula(self.g_formula.text(), ha, sii, oiii, "G / Ha")
        b = evaluate_formula(self.b_formula.text(), ha, sii, oiii, "B / OIII")

        rgb = np.stack([r, g, b], axis=0)

        if self.rescale_check.isChecked():
            rgb = global_rescale_rgb(rgb)
        else:
            rgb = clipped_zero_rgb(rgb)

        return rgb.astype(np.float32)

    def apply(self):
        try:
            if not self.r_formula.text().strip() or not self.g_formula.text().strip() or not self.b_formula.text().strip():
                raise ValueError("Formula fields are empty. Click Calculate first or type R/G/B formulas manually.")

            if self.weight_sii is None or self.weight_oiii is None:
                self.calculate()

            output_path = self._get_output_path()

            if os.path.exists(output_path):
                response = QMessageBox.question(
                    self,
                    "Overwrite output?",
                    f"The file already exists and will be overwritten:\n\n{output_path}\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if response != QMessageBox.StandardButton.Yes:
                    return

            rgb = self._compose_rgb()
            self.last_rgb = rgb
            self.last_output_path = output_path

            header = getattr(self.ha_fit, "header", None)
            if not header:
                header = ""

            self.siril.save_image_file(rgb, header=header, filename=output_path)

            try:
                quoted_output_path = f'"{output_path}"'
                self.siril.cmd("load", quoted_output_path)
            except Exception as load_error:
                self._log(f"Could not automatically open output FITS in Siril: {load_error}", level="warning")

            pixmap = preview_qpixmap(rgb)
            if pixmap is not None:
                self.preview.set_pixmap(pixmap)
            else:
                self.output_label.setText("Result saved. Preview unavailable, but FITS was created.")

            self.output_label.setText(
                f"Saved: {output_path}  |  Used current formulas. Preview uses simple linked autostretch."
            )
            self._log(f"Normalize Channels Calculator: saved {output_path}")

        except Exception as e:
            self._show_error("Apply error", e)

    def _show_error(self, title, err):
        msg = str(err)
        self._log(f"{title}: {msg}", level="error")
        self._log(traceback.format_exc(), level="error")
        QMessageBox.critical(self, title, msg)


def main():
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    window = NormalizeChannelsWindow()
    window.show()

    if created_app:
        app.exec()


if __name__ == "__main__":
    main()
