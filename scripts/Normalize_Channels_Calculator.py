# Normalize Channels Calculator
# Version: 2.0.8
# Author: Rodrigo Zeba + ChatGPT
# License: MIT
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
VERSION = "2.0.8"

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
VERSION = "2.0.8"
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
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_pixmap(self, pixmap):
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self.fit_to_view()

    def fit_to_view(self):
        if self.pixmap_item is not None:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
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
        input_layout.setColumnMinimumWidth(2, 112)

        self.percentile_edit = QLineEdit("95")
        self.percentile_edit.setMinimumHeight(30)
        input_layout.addWidget(QLabel("Define the Percentile"), 0, 0, 1, 3)
        input_layout.addWidget(self.percentile_edit, 1, 0, 1, 3)

        self.ha_edit = QLineEdit()
        self.ha_edit.setReadOnly(True)
        self.ha_edit.setMinimumHeight(30)
        self.ha_btn = QPushButton("Select Ha File")
        self.ha_btn.setObjectName("SecondaryButton")
        self.ha_btn.setMinimumHeight(30)
        self.ha_btn.clicked.connect(self.select_ha)
        input_layout.addWidget(QLabel("Ha File"), 2, 0, 1, 3)
        input_layout.addWidget(self.ha_edit, 3, 0, 1, 2)
        input_layout.addWidget(self.ha_btn, 3, 2)

        self.sii_edit = QLineEdit()
        self.sii_edit.setReadOnly(True)
        self.sii_edit.setMinimumHeight(30)
        self.sii_btn = QPushButton("Select SII File")
        self.sii_btn.setObjectName("SecondaryButton")
        self.sii_btn.setMinimumHeight(30)
        self.sii_btn.clicked.connect(self.select_sii)
        input_layout.addWidget(QLabel("SII File"), 4, 0, 1, 3)
        input_layout.addWidget(self.sii_edit, 5, 0, 1, 2)
        input_layout.addWidget(self.sii_btn, 5, 2)

        self.oiii_edit = QLineEdit()
        self.oiii_edit.setReadOnly(True)
        self.oiii_edit.setMinimumHeight(30)
        self.oiii_btn = QPushButton("Select OIII File")
        self.oiii_btn.setObjectName("SecondaryButton")
        self.oiii_btn.setMinimumHeight(30)
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
        formula_panel.setMinimumHeight(145)
        formula_panel.setMaximumHeight(170)

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
        self.r_formula.setMinimumHeight(30)
        self.g_formula.setMinimumHeight(30)
        self.b_formula.setMinimumHeight(30)

        formula_layout.addWidget(QLabel("R / SII Formula"), 0, 0)
        formula_layout.addWidget(QLabel("G / Ha Formula"), 0, 1)
        formula_layout.addWidget(QLabel("B / OIII Formula"), 0, 2)

        formula_layout.addWidget(self.r_formula, 1, 0)
        formula_layout.addWidget(self.g_formula, 1, 1)
        formula_layout.addWidget(self.b_formula, 1, 2)

        self.rescale_check = QCheckBox("Global rescale output")
        self.rescale_check.setChecked(True)
        formula_layout.addWidget(self.rescale_check, 2, 0)

        self.calculate_btn = QPushButton("Calculate")
        self.calculate_btn.setObjectName("PrimaryButton")
        self.calculate_btn.setMinimumHeight(30)
        self.calculate_btn.setMinimumWidth(160)
        self.calculate_btn.clicked.connect(self.calculate)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setObjectName("PrimaryButton")
        self.apply_btn.setMinimumHeight(30)
        self.apply_btn.setMinimumWidth(160)
        self.apply_btn.clicked.connect(self.apply)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        button_row.addStretch(1)
        button_row.addWidget(self.calculate_btn)
        button_row.addWidget(self.apply_btn)

        formula_layout.addLayout(button_row, 2, 1, 1, 2)

        right_col.addWidget(formula_panel, stretch=0)

        preview_panel = QWidget()
        preview_panel.setObjectName("Panel")

        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 12, 14, 12)
        preview_layout.setSpacing(8)

        preview_label = QLabel("Preview")
        preview_layout.addWidget(preview_label)

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
                padding: 6px;
                selection-background-color: #0078d4;
            }

            QLineEdit {
                min-height: 24px;
            }

            QLineEdit:read-only {
                color: #dddddd;
            }

            QPushButton {
                background-color: #3a3a3a;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 6px;
                padding: 7px;
                min-height: 24px;
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
        try:
            wd = self.siril.get_siril_wd()
            if wd and os.path.isdir(wd):
                return os.path.join(wd, DEFAULT_OUTPUT)
        except Exception:
            pass

        folder = os.path.dirname(self.ha_path) if self.ha_path else os.getcwd()
        return os.path.join(folder, DEFAULT_OUTPUT)

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

            self.ha_stats = calc_stats(ha_data, pct)
            self.sii_stats = calc_stats(sii_data, pct)
            self.oiii_stats = calc_stats(oiii_data, pct)

            ha_sig = self.ha_stats["signal"]
            sii_sig = self.sii_stats["signal"]
            oiii_sig = self.oiii_stats["signal"]

            if ha_sig <= 0 or sii_sig <= 0 or oiii_sig <= 0:
                raise ValueError(
                    "One or more channels have non-positive signal estimate. "
                    "Try a different percentile or inspect the channels."
                )

            self.weight_sii = safe_div(ha_sig, sii_sig)
            self.weight_oiii = safe_div(ha_sig, oiii_sig)
            self.weight_ha = 0.70

            self.r_formula.setText(f"(SII - med(SII)) * {self.weight_sii:.2f}")
            self.g_formula.setText(f"(Ha - med(Ha)) * {self.weight_ha:.2f}")
            self.b_formula.setText(f"(OIII - med(OIII)) * {self.weight_oiii:.2f}")

            report = self._make_report(pct)
            self.info_text.setPlainText(report)
            self._log("Normalize Channels Calculator: calculation complete.")

        except Exception as e:
            self._show_error("Calculate error", e)

    def _make_report(self, pct):
        ha = self.ha_stats
        sii = self.sii_stats
        oiii = self.oiii_stats

        return f"""Normalize Channels Calculator v{VERSION}

Selected files:
Ha:   {self.ha_path}
SII:  {self.sii_path}
OIII: {self.oiii_path}

Percentile: P{pct:.1f}

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

=== Suggested SHO weights ===
R / SII:  {self.weight_sii:.3f}
G / Ha:   {self.weight_ha:.3f}  (artistic green reduction)
B / OIII: {self.weight_oiii:.3f}

=== Suggested Pixel Math formulas ===
R = (SII - med(SII)) * {self.weight_sii:.2f}
G = (Ha - med(Ha)) * {self.weight_ha:.2f}
B = (OIII - med(OIII)) * {self.weight_oiii:.2f}

Notes:
- Median is used as an estimated background pedestal.
- P{pct:.1f} - median is used as an estimated useful signal above background.
- Weights are a starting point, not a final truth.
- Ha is reduced because in SHO it maps to green and often dominates visually.
- Apply uses the current text in the R/G/B formula fields, so manual edits are used.
- Apply uses a simple global rescale after composition by default.
- Preview uses a simple linked autostretch for display only.
"""

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
