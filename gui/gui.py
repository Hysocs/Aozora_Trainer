
import json
import html
import os
import re
import math
import copy
from . import gui_math
import sys
import shutil
import ctypes
import zlib
import importlib.util
import signal
import textwrap
from dataclasses import replace
from pathlib import Path
from collections import deque
from datetime import datetime
from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import QFileIconProvider
from PyQt6.QtCore import QFileInfo
import subprocess

from .gui_theme import THEME, make_stylesheet, set_role

try:
    from training_utils.config import config as default_config
except ImportError:
    class default_config:
        RAVEN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "momentum_dtype": "bfloat16"}
        PAGED_ADAMW_8BIT_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}
        TITAN_PARAMS = {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01, "debias_strength": 1.0, "momentum_dtype": "bfloat16"}
        OPTIMIZER_TYPE = "Raven"
        TIMESTEP_ALLOCATION = {"bin_size": 100, "counts": []}

# Compatibility aliases keep feature code readable while every value comes from
# the single semantic palette in gui_theme.py.
DARK_BG = THEME.window
PANEL_BG = THEME.surface_raised
NESTED_GROUP_BG = THEME.chart
GRAPH_BG = THEME.canvas
BORDER = THEME.border
ACCENT = THEME.accent
ACCENT2 = THEME.accent_alt
TEXT_PRI = THEME.text
TEXT_SEC = THEME.text_muted
DANGER = THEME.danger
SUCCESS = THEME.success
WARN = THEME.warning
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

TRAINING_MODE_SDXL = "SDXL"
TRAINING_MODE_ANIMA_DIT = "Anima DiT"
DIT_AVAILABLE = importlib.util.find_spec("training_utils.anima") is not None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")


def platform_path(value):
    """Normalize a user/config path for the host OS without resolving symlinks."""
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        return ""
    if not IS_WINDOWS:
        text = text.replace("\\", "/")
    return os.path.normpath(text)


def usable_dialog_start(value, fallback=None):
    candidate = platform_path(value)
    if candidate and os.path.exists(candidate):
        return candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
    fallback = platform_path(fallback)
    if fallback and os.path.isdir(fallback):
        return fallback
    return QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.HomeLocation) or str(Path.home())


def report_gui_exception(context, exc):
    print(f"GUI warning: {context}: {type(exc).__name__}: {exc}", file=sys.stderr)
SDXL_PATH_KEYS = {
    "SINGLE_FILE_CHECKPOINT_PATH",
    "VAE_PATH",
    "RESUME_MODEL_PATH",
    "RESUME_STATE_PATH",
}
ANIMA_PATH_KEYS = {
    "DIT_PATH",
    "DIT_VAE_PATH",
    "TEXT_ENCODER_PATH",
    "TOKENIZER_PATH",
    "TOKENIZER_T5XXL_PATH",
    "ANIMA_RESUME_MODEL_PATH",
    "ANIMA_RESUME_STATE_PATH",
}
ANIMA_LOCKED_WIDGET_KEYS = {
    "PREDICTION_TYPE",
    "MEMORY_EFFICIENT_ATTENTION",
    "CAPTION_CHUNKING_ENABLED",
    "VAE_NORMALIZATION_MODE",
    "VAE_SHIFT_FACTOR",
    "VAE_SCALING_FACTOR",
    "VAE_LATENT_CHANNELS",
}
CAPTION_JSON_TYPES = ("tags", "nl", "tags_nl", "nl_tags")

TEXT_MUTED = THEME.text_disabled
BORDER_MUTED = THEME.border_muted

STYLESHEET = make_stylesheet()

DEFAULT_THEME_COLORS = {
    "primary": "#7c6af7",
    "secondary": "#5839b0",
}
CLAY_CHART_COLORS = {
    "loss_ema": "#c2ad55",
    "sigma_samples": "#c1845b",
    "mean_loss_by_sigma": "#c2ad55",
    "learning_rate": "#c1845b",
    "gradient_norm": "#c1845b",
    "clipped_gradient": "#c2ad55",
    "lr_scheduler": "#c1845b",
    "ticket_allocator": "#c2ad55",
    "loss_weight": "#c2ad55",
}
PURPLE_CHART_COLORS = {
    "loss_ema": "#3d8943",
    "sigma_samples": "#7c6af7",
    "mean_loss_by_sigma": "#7c6af7",
    "learning_rate": "#45aeb4",
    "gradient_norm": "#c1845b",
    "clipped_gradient": "#c2ad55",
    "lr_scheduler": "#45aeb4",
    "ticket_allocator": "#7c6af7",
    "loss_weight": "#3d8943",
}
DEFAULT_CHART_COLORS = PURPLE_CHART_COLORS
THEME_COLOR_PRESETS = [
    ("Clay & Ochre", "#c1845b", "#c2ad55", CLAY_CHART_COLORS),
    ("Violet & Amethyst", "#7c6af7", "#5839b0", PURPLE_CHART_COLORS),
]

_sleep_inhibitor_process = None


def prevent_sleep(enable=True):
    """Toggle OS sleep inhibition while a training process is active."""
    global _sleep_inhibitor_process
    try:
        if IS_WINDOWS:
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            kernel32 = ctypes.windll.kernel32
            state = (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) if enable else ES_CONTINUOUS
            return bool(kernel32.SetThreadExecutionState(state))

        if IS_LINUX:
            if not enable:
                if _sleep_inhibitor_process and _sleep_inhibitor_process.poll() is None:
                    _sleep_inhibitor_process.terminate()
                    try:
                        _sleep_inhibitor_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        _sleep_inhibitor_process.kill()
                        _sleep_inhibitor_process.wait()
                _sleep_inhibitor_process = None
                return True

            if _sleep_inhibitor_process and _sleep_inhibitor_process.poll() is None:
                return True
            inhibitor = shutil.which("systemd-inhibit")
            sleeper = shutil.which("sleep")
            if not inhibitor or not sleeper:
                return False
            _sleep_inhibitor_process = subprocess.Popen(
                [inhibitor, "--what=sleep:idle", "--mode=block",
                 "--why=Aozora model training is active", sleeper, "infinity"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return _sleep_inhibitor_process.poll() is None
        return False
    except Exception as exc:
        report_gui_exception("could not change OS sleep inhibition", exc)
        _sleep_inhibitor_process = None
        return False


def fixed_width_font(point_size=9, bold=False):
    font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
    font.setPointSize(point_size)
    if bold:
        font.setWeight(QtGui.QFont.Weight.Bold)
    return font


class NoScrollSpinBox(QtWidgets.QSpinBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def wheelEvent(self, e): e.ignore()

class CommitOnPressComboBox(QtWidgets.QComboBox):
    """Combo box that reliably commits popup rows on the first mouse press."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.view().viewport().installEventFilter(self)

    def eventFilter(self, watched, event):
        if (
            watched is self.view().viewport()
            and event.type() == QtCore.QEvent.Type.MouseButtonPress
            and event.button() == QtCore.Qt.MouseButton.LeftButton
        ):
            index = self.view().indexAt(event.position().toPoint())
            if index.isValid():
                row = index.row()
                self.setCurrentIndex(row)
                self.hidePopup()
                self.activated.emit(row)
                return True
        return super().eventFilter(watched, event)

class NoScrollComboBox(CommitOnPressComboBox):
    def wheelEvent(self, e): e.ignore()

class NoScrollSlider(QtWidgets.QSlider):
    def wheelEvent(self, e): e.ignore()


class ResponsivePixmapLabel(QtWidgets.QLabel):
    """A preview label that fits its source pixmap to the available layout space."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = QtGui.QPixmap()
        self._fit_timer = QtCore.QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.setInterval(60)
        self._fit_timer.timeout.connect(self._fit_pixmap)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(160, 160)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def set_source_pixmap(self, pixmap):
        self._source_pixmap = QtGui.QPixmap(pixmap)
        self._schedule_fit()

    def clear(self):
        self._fit_timer.stop()
        self._source_pixmap = QtGui.QPixmap()
        super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_fit()

    def sizeHint(self):
        # A displayed pixmap must not feed its dimensions back into the layout.
        return QtCore.QSize(640, 480)

    def minimumSizeHint(self):
        return QtCore.QSize(160, 160)

    def _schedule_fit(self):
        if not self._source_pixmap.isNull():
            self._fit_timer.start()

    def _fit_pixmap(self):
        if self._source_pixmap.isNull() or self.width() <= 0 or self.height() <= 0:
            super().clear()
            return
        fitted = self._source_pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(fitted)


class ResizeSplitterHandle(QtWidgets.QSplitterHandle):
    """Distinct resize gutter with an accent grip, unlike a scrollbar thumb."""
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self._hovered = False

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), THEME.color("surface_hover" if self._hovered else "nested_group"))
        painter.setPen(QtGui.QPen(THEME.color("border"), 1))
        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            painter.drawLine(0, 0, 0, self.height())
            painter.drawLine(self.width() - 1, 0, self.width() - 1, self.height())
            center = self.rect().center()
            painter.setBrush(THEME.color("accent_hover" if self._hovered else "accent"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for offset in (-9, 0, 9):
                painter.drawEllipse(QtCore.QPointF(center.x(), center.y() + offset), 2.0, 2.0)
        else:
            painter.drawLine(0, 0, self.width(), 0)
            painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
            center = self.rect().center()
            painter.setBrush(THEME.color("accent_hover" if self._hovered else "accent"))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for offset in (-9, 0, 9):
                painter.drawEllipse(QtCore.QPointF(center.x() + offset, center.y()), 2.0, 2.0)


class ThemedSplitter(QtWidgets.QSplitter):
    def __init__(self, orientation=QtCore.Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setHandleWidth(10)

    def createHandle(self):
        return ResizeSplitterHandle(self.orientation(), self)


class EmptyStateListWidget(QtWidgets.QListWidget):
    def __init__(self, empty_text="", parent=None):
        super().__init__(parent)
        self.empty_text = empty_text

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() or not self.empty_text:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setPen(THEME.color("text_muted"))
        text_rect = self.viewport().rect().adjusted(16, 18, -16, -16)
        painter.drawText(
            text_rect,
            QtCore.Qt.AlignmentFlag.AlignTop |
            QtCore.Qt.AlignmentFlag.AlignHCenter |
            QtCore.Qt.TextFlag.TextWordWrap,
            self.empty_text,
        )


class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __lt__(self, other):
        try: return float(self.text()) < float(other.text())
        except ValueError: return super().__lt__(other)

class DateTableWidgetItem(QtWidgets.QTableWidgetItem):
    def __init__(self, display_text, timestamp):
        super().__init__(display_text)
        self.timestamp = timestamp
    def __lt__(self, other):
        return self.timestamp < other.timestamp


def make_spin(lo, hi, val=None, *, scroll=False):
    w = QtWidgets.QSpinBox() if scroll else NoScrollSpinBox()
    w.setRange(lo, hi)
    if val is not None: w.setValue(val)
    return w

def make_dspin(lo, hi, val=None, step=0.1, decimals=2, *, scroll=False):
    w = QtWidgets.QDoubleSpinBox() if scroll else NoScrollDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(decimals)
    if val is not None: w.setValue(val)
    return w

def make_combo(items, *, scroll=False):
    w = CommitOnPressComboBox() if scroll else NoScrollComboBox()
    w.addItems(items)
    return w

def make_btn(text, callback=None, style=None):
    b = QtWidgets.QPushButton(text)
    if callback: b.clicked.connect(callback)
    if style: b.setStyleSheet(style)
    return b

def style_role(widget, role):
    """Use a global theme variant; avoids per-widget QSS parsing."""
    return set_role(widget, role)

def make_label(text, color=None, bold=False, size=None):
    lbl = QtWidgets.QLabel(text)
    role = None
    for candidate, value in (
        ("accent", ACCENT),
        ("accent_alt", ACCENT2),
        ("warning", WARN),
        ("danger", DANGER),
        ("success", SUCCESS),
    ):
        if color and QtGui.QColor(color) == QtGui.QColor(value):
            role = candidate
            break
    if role:
        lbl.setProperty("themeColorRole", role)
        lbl.setProperty("themeBold", bool(bold))
        lbl.setProperty("themePointSize", int(size or 0))
    parts = []
    if color: parts.append(f"color: {color};")
    if bold: parts.append("font-weight: bold;")
    if size: parts.append(f"font-size: {size}pt;")
    if parts: lbl.setStyleSheet(" ".join(parts))
    return lbl


class ThemeSwatchButton(QtWidgets.QAbstractButton):
    """Compact stacked theme swatches attached to the tab strip."""

    def __init__(self, primary, secondary, parent=None):
        super().__init__(parent)
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.setFixedSize(20, 20)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Choose GUI colors")

    def set_colors(self, primary, secondary):
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(THEME.color("accent") if self.underMouse() else THEME.color("border"), 1))
        painter.setBrush(THEME.color("window"))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 4, 4)
        swatch = QtCore.QRectF(2.5, 2.5, 15, 15)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(swatch, 2, 2)
        painter.save()
        painter.setClipPath(clip)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(self.primary)
        painter.drawPolygon(QtGui.QPolygonF([
            swatch.topLeft(), swatch.topRight(), swatch.bottomLeft(),
        ]))
        painter.setBrush(self.secondary)
        painter.drawPolygon(QtGui.QPolygonF([
            swatch.topRight(), swatch.bottomRight(), swatch.bottomLeft(),
        ]))
        painter.restore()


class ThemePresetButton(QtWidgets.QAbstractButton):
    """Popup row showing a preset name and its two colors."""

    def __init__(self, text, primary, secondary, parent=None):
        super().__init__(parent)
        self.text = text
        self.primary = QtGui.QColor(primary)
        self.secondary = QtGui.QColor(secondary)
        self.setFixedHeight(34)
        self.setMinimumWidth(235)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self.underMouse():
            painter.fillRect(self.rect(), THEME.color("surface_hover"))
        painter.setPen(THEME.color("text"))
        painter.drawText(
            self.rect().adjusted(10, 0, -58, 0),
            QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
            self.text,
        )
        painter.setPen(QtGui.QPen(THEME.color("border"), 1))
        painter.setBrush(self.primary)
        painter.drawRoundedRect(QtCore.QRectF(self.width() - 50, 8, 18, 18), 3, 3)
        painter.setBrush(self.secondary)
        painter.drawRoundedRect(QtCore.QRectF(self.width() - 27, 8, 18, 18), 3, 3)

def make_separator(horizontal=True):
    f = QtWidgets.QFrame()
    f.setFrameShape(QtWidgets.QFrame.Shape.HLine if horizontal else QtWidgets.QFrame.Shape.VLine)
    f.setStyleSheet(f"border: 1px solid {BORDER}; margin: 4px 0;")
    return f

RAW_GROUP_TITLES = {
    "Dataset List", "Dataset Preview", "Image Preview", "Caption Source",
    "File & Directory Paths", "Model Files", "Tokenizer Files",
    "VAE Configuration", "Batching & DataLoaders",
}
TRANSFORMED_GROUP_TITLES = {
    "Conditioning Regularization", "Caption Cache Options",
    "Aspect Ratio Bucketing", "Image Scheduling", "Loss Configuration",
    "Loss Settings", "Timestep Loss Weight Curve", "Loss Weight Presets",
    "Gradient Centralization", "Dataset Settings",
    "Timestep Ticket Allocation", "Ticket Allocation Chart",
    "Distribution Settings", "Distribution Presets", "Timestep Coverage",
    "UNet Layer Exclusion", "DiT Layer Exclusion",
}
TRANSFORMED_CHECKBOX_KEYS = {
    "UNCONDITIONAL_DROPOUT", "TEXT_CONDITIONING_SCALE_ENABLED",
    "T5_TOKEN_DROPOUT_ENABLED", "CAPTION_CHUNKING_ENABLED", "SHOULD_UPSCALE",
    "MULTI_BUCKET_ENABLED", "TIMESTEP_FORCE_IMAGE_BIN_SPREAD",
    "TIMESTEP_STRATIFIED_SAMPLING",
}
RAW_CHECKBOX_KEYS = {
    "ANIMA_STREAMING_SAVE",
}

def set_semantic_color(widget, semantic):
    if widget.property("semanticColor") == semantic:
        return widget
    widget.setProperty("semanticColor", semantic)
    if widget.testAttribute(QtCore.Qt.WidgetAttribute.WA_WState_Polished):
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
    return widget

def inherit_semantic_colors(root):
    """Give controls the semantic color of their nearest enclosing group."""
    control_types = (
        QtWidgets.QPushButton, QtWidgets.QLineEdit, QtWidgets.QPlainTextEdit,
        QtWidgets.QTextEdit, QtWidgets.QComboBox, QtWidgets.QSpinBox,
        QtWidgets.QDoubleSpinBox, QtWidgets.QCheckBox, QtWidgets.QSlider,
        QtWidgets.QListWidget, QtWidgets.QTableWidget,
    )
    for widget in root.findChildren(QtWidgets.QWidget):
        if not isinstance(widget, control_types) or widget.property("semanticColor"):
            continue
        parent = widget.parentWidget()
        while parent is not None and parent is not root:
            if isinstance(parent, QtWidgets.QGroupBox):
                semantic = parent.property("semanticColor")
                if semantic:
                    set_semantic_color(widget, semantic)
                    break
            parent = parent.parentWidget()

def group_box(title, layout_type=QtWidgets.QVBoxLayout, role="nested"):
    gb = QtWidgets.QGroupBox(title)
    set_role(gb, role)
    if title in RAW_GROUP_TITLES:
        set_semantic_color(gb, "raw")
    elif title in TRANSFORMED_GROUP_TITLES:
        set_semantic_color(gb, "transformed")
    else:
        set_semantic_color(gb, "raw")
    lay = layout_type(gb)
    return gb, lay

def form_row(layout, label_text, widget, tooltip=None):
    lbl = QtWidgets.QLabel(label_text)
    if tooltip:
        lbl.setToolTip(tooltip)
        widget.setToolTip(tooltip)
    layout.addRow(lbl, widget)

class CompressedLogBuffer:
    def __init__(self, block_size=128, compression_level=6, max_active_bytes=64 * 1024):
        self.block_size = max(16, int(block_size))
        self.compression_level = max(1, min(9, int(compression_level)))
        self.max_active_bytes = max(4096, int(max_active_bytes))
        self.blocks = []
        self.current_lines = []
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def clear(self):
        self.blocks.clear()
        self.current_lines.clear()
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def append(self, text, replace_last=False):
        if replace_last:
            self.remove_last_line()
        lines = str(text).rstrip('\n').splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            line_bytes = len(line.encode('utf-8', errors='replace')) + 1
            self.current_lines.append(line)
            self.current_bytes += line_bytes
            self.line_count += 1
            self.uncompressed_bytes += line_bytes
            if len(self.current_lines) >= self.block_size or self.current_bytes >= self.max_active_bytes:
                self._seal_current_block()

    def remove_last_line(self):
        if self.line_count <= 0:
            return
        if self.current_lines:
            removed = self.current_lines.pop()
            self.current_bytes = max(0, self.current_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
            self.line_count -= 1
            self.uncompressed_bytes = max(0, self.uncompressed_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
            return
        if not self.blocks:
            return
        lines = self._decode_block(len(self.blocks) - 1)
        old_payload_size = len(self.blocks[-1][1])
        if lines:
            removed = lines.pop()
            self.line_count -= 1
            self.uncompressed_bytes = max(0, self.uncompressed_bytes - len(removed.encode('utf-8', errors='replace')) - 1)
        self.blocks.pop()
        self.compressed_bytes = max(0, self.compressed_bytes - old_payload_size)
        if lines:
            self.current_lines = lines
            self.current_bytes = sum(len(line.encode('utf-8', errors='replace')) + 1 for line in self.current_lines)
            if len(self.current_lines) >= self.block_size or self.current_bytes >= self.max_active_bytes:
                self._seal_current_block()

    def get_lines(self, start_line, count):
        if count <= 0 or self.line_count <= 0:
            return []
        start_line = max(0, min(int(start_line), self.line_count - 1))
        end_line = min(self.line_count, start_line + int(count))
        output = []
        cursor = 0
        for line_count, payload in self.blocks:
            block_start = cursor
            block_end = cursor + line_count
            if block_end > start_line and block_start < end_line:
                lines = zlib.decompress(payload).decode('utf-8', errors='replace').split('\n')
                local_start = max(0, start_line - block_start)
                local_end = min(line_count, end_line - block_start)
                output.extend(lines[local_start:local_end])
            cursor = block_end
            if cursor >= end_line:
                break
        if cursor < end_line and self.current_lines:
            block_start = cursor
            block_end = cursor + len(self.current_lines)
            if block_end > start_line and block_start < end_line:
                local_start = max(0, start_line - block_start)
                local_end = min(len(self.current_lines), end_line - block_start)
                output.extend(self.current_lines[local_start:local_end])
        return output

    def memory_summary(self):
        stored = self.compressed_bytes + self.current_bytes
        ratio = 1.0 if self.uncompressed_bytes <= 0 else stored / self.uncompressed_bytes
        return stored, self.uncompressed_bytes, ratio

    def get_all_text(self):
        if self.line_count <= 0:
            return ""
        return "\n".join(self.get_lines(0, self.line_count))

    def _seal_current_block(self):
        if not self.current_lines:
            return
        raw = '\n'.join(self.current_lines).encode('utf-8', errors='replace')
        payload = zlib.compress(raw, self.compression_level)
        self.blocks.append((len(self.current_lines), payload))
        self.compressed_bytes += len(payload)
        self.current_lines = []
        self.current_bytes = 0

    def _decode_block(self, index):
        if not (0 <= index < len(self.blocks)):
            return []
        line_count, payload = self.blocks[index]
        lines = zlib.decompress(payload).decode('utf-8', errors='replace').split('\n')
        return lines[:line_count]


class VirtualConsoleWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, visible_lines=900, clear_callback=None):
        super().__init__(parent)
        self.visible_lines = max(100, int(visible_lines))
        self.buffer = CompressedLogBuffer(block_size=128, compression_level=6)
        self.pending_render = False
        self.follow_output = True
        self.clear_callback = clear_callback
        self._internal_scroll_update = False
        self._internal_text_update = False
        self._build_ui()
        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self._render_now)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.textbox = QtWidgets.QPlainTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setMinimumHeight(200)
        self.textbox.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.textbox.setUndoRedoEnabled(False)
        set_role(self.textbox, "consoleText")
        self.textbox.viewport().installEventFilter(self)
        self.textbox.verticalScrollBar().valueChanged.connect(self._on_inner_scrollbar_changed)

        self.scrollbar = QtWidgets.QScrollBar(QtCore.Qt.Orientation.Vertical)
        self.scrollbar.valueChanged.connect(self._on_scrollbar_changed)

        row.addWidget(self.textbox, 1)
        row.addWidget(self.scrollbar)
        root.addLayout(row, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        self.status_label = make_label("Lines: 0 | Buffer: empty", color=TEXT_SEC)
        self.follow_button = QtWidgets.QPushButton("Following Output")
        self.follow_button.setObjectName("FollowOutputButton")
        self.follow_button.setCheckable(True)
        self.follow_button.setChecked(True)
        self.follow_button.setToolTip("Toggle auto-scroll to the latest console output.")
        self.follow_button.toggled.connect(self._on_follow_toggled)
        self.clear_button = QtWidgets.QPushButton("Clear Console")
        self.clear_button.setToolTip("Clear the console output.")
        if self.clear_callback:
            self.clear_button.clicked.connect(self.clear_callback)
        self.copy_button = QtWidgets.QPushButton("Copy Full Logs")
        self.copy_button.setToolTip("Copy the complete log buffer, including lines not currently visible.")
        self.copy_button.clicked.connect(self.copy_full_logs)
        footer.addWidget(self.status_label)
        footer.addStretch()
        footer.addWidget(self.follow_button)
        footer.addWidget(self.copy_button)
        footer.addWidget(self.clear_button)
        root.addLayout(footer)

    def copy_full_logs(self):
        QtWidgets.QApplication.clipboard().setText(self.buffer.get_all_text())

    def eventFilter(self, obj, event):
        if obj is self.textbox.viewport() and event.type() == QtCore.QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            if delta:
                inner_sb = self.textbox.verticalScrollBar()
                inner_can_scroll = (
                    (delta > 0 and inner_sb.value() > inner_sb.minimum()) or
                    (delta < 0 and inner_sb.value() < inner_sb.maximum())
                )
                if inner_can_scroll or self.scrollbar.maximum() <= 0:
                    return False
                step = max(1, self.scrollbar.singleStep())
                self.scrollbar.setValue(self.scrollbar.value() - int(delta / 120) * step)
                self._set_follow_output(self._at_console_bottom())
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def append_line(self, text, replace_last=False):
        was_at_bottom = self._at_console_bottom()
        self.buffer.append(text, replace_last=replace_last)
        if self.follow_output and was_at_bottom:
            self.follow_output = True
        self._schedule_render()

    def clear(self):
        self.buffer.clear()
        self.textbox.clear()
        self.scrollbar.setRange(0, 0)
        self.status_label.setText("Lines: 0 | Buffer: empty")
        self.follow_output = True
        self._sync_follow_button()

    def _schedule_render(self):
        if not self.pending_render:
            self.pending_render = True
            self.render_timer.start(33)

    def _on_follow_toggled(self, checked):
        self.follow_output = bool(checked)
        self._sync_follow_button()
        if self.follow_output:
            self.scrollbar.setValue(self.scrollbar.maximum())
            self._schedule_render()

    def _on_scrollbar_changed(self, value):
        if self._internal_scroll_update:
            return
        self._set_follow_output(self._at_console_bottom())
        self._schedule_render()

    def _on_inner_scrollbar_changed(self, value):
        if self._internal_text_update:
            return
        self._set_follow_output(self._at_console_bottom())

    def _at_console_bottom(self):
        inner_sb = self.textbox.verticalScrollBar()
        outer_bottom = self.scrollbar.value() >= self.scrollbar.maximum() - 1
        inner_bottom = inner_sb.value() >= inner_sb.maximum() - 1
        return outer_bottom and inner_bottom

    def _set_follow_output(self, follow):
        self.follow_output = bool(follow)
        self._sync_follow_button()

    def _sync_follow_button(self):
        self.follow_button.blockSignals(True)
        self.follow_button.setChecked(self.follow_output)
        self.follow_button.setText("Following Output" if self.follow_output else "Jump to Bottom")
        self.follow_button.blockSignals(False)

    def _render_now(self):
        self.pending_render = False
        total = self.buffer.line_count
        max_start = max(0, total - self.visible_lines)
        self._internal_scroll_update = True
        self.scrollbar.setRange(0, max_start)
        self.scrollbar.setPageStep(self.visible_lines)
        self.scrollbar.setSingleStep(max(1, self.visible_lines // 20))
        if self.follow_output:
            self.scrollbar.setValue(max_start)
        elif self.scrollbar.value() > max_start:
            self.scrollbar.setValue(max_start)
        start = self.scrollbar.value()
        self._internal_scroll_update = False

        lines = self.buffer.get_lines(start, self.visible_lines)
        inner_sb = self.textbox.verticalScrollBar()
        inner_value = inner_sb.value()
        self._internal_text_update = True
        self.textbox.setPlainText('\n'.join(lines))
        if self.follow_output:
            inner_sb.setValue(inner_sb.maximum())
        else:
            inner_sb.setValue(min(inner_value, inner_sb.maximum()))
        self._internal_text_update = False

        stored, uncompressed, ratio = self.buffer.memory_summary()
        shown_start = 0 if total == 0 else start + 1
        shown_end = min(total, start + len(lines))
        self.status_label.setText(
            f"Lines: {total:,} | Showing: {shown_start:,}-{shown_end:,} | "
            f"Memory: {self._fmt_bytes(stored)} compressed from {self._fmt_bytes(uncompressed)} ({ratio:.2%})"
        )

    def _fmt_bytes(self, value):
        value = float(value)
        for unit in ["B", "KB", "MB", "GB"]:
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024.0




class CustomFolderDialog(QtWidgets.QDialog):
    def __init__(self, start_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Dataset Folder")
        self.resize(1100, 700)
        self.selected_path = None
        self.history = []
        self.history_idx = -1
        self.is_navigating_history = False
        self.icon_provider = QFileIconProvider()
        self.current_path = usable_dialog_start(start_path)
        self._build_ui()
        self.load_directory(self.current_path)

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(5)
        def nav_btn(icon, tip, cb):
            b = QtWidgets.QPushButton()
            b.setIcon(self.style().standardIcon(icon))
            b.setFixedWidth(35)
            b.setToolTip(tip)
            b.clicked.connect(cb)
            return b

        self.btn_back    = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowBack, "Back", self.go_back)
        self.btn_fwd     = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowForward, "Forward", self.go_forward)
        self.btn_up      = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_ArrowUp, "Up", self.go_up)
        self.btn_refresh = nav_btn(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload, "Refresh",
                                   lambda: self.load_directory(self.current_path))
        self.btn_back.setEnabled(False)
        self.btn_fwd.setEnabled(False)

        self.path_edit = QtWidgets.QLineEdit(self.current_path)
        self.path_edit.returnPressed.connect(self._on_path_entered)

        for w in [self.btn_back, self.btn_fwd, self.btn_up, self.path_edit, self.btn_refresh]:
            nav.addWidget(w) if w is not self.path_edit else nav.addWidget(w, 1)
        layout.addLayout(nav)

        splitter = ThemedSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {BORDER}; }}")

        self.sidebar = QtWidgets.QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setIconSize(QtCore.QSize(24, 24))
        self.sidebar.setStyleSheet(f"""
            QListWidget {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; outline: none; }}
            QListWidget::item {{ padding: 6px; color: {TEXT_PRI}; margin: 2px; }}
            QListWidget::item:hover {{ background: {BORDER}; border-radius: 4px; }}
            QListWidget::item:selected {{ background: {ACCENT}; color: white; border-radius: 4px; }}
        """)
        self.sidebar.itemClicked.connect(lambda item: item.data(QtCore.Qt.ItemDataRole.UserRole) and self.load_directory(item.data(QtCore.Qt.ItemDataRole.UserRole)))
        self._populate_sidebar()
        splitter.addWidget(self.sidebar)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Name", "Images", "Date Modified", "HiddenPath"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnHidden(3, True)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setIconSize(QtCore.QSize(20, 20))
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(lambda r, _: self.load_directory(self.table.item(r, 3).text()))
        self.table.setStyleSheet(f"""
            QTableWidget {{ background: {PANEL_BG}; alternate-background-color: {DARK_BG}; border: 1px solid {BORDER}; border-radius: 4px; }}
            QTableWidget::item {{ padding: 4px; }}
            QHeaderView::section {{ background: {BORDER}; padding: 6px; border: none; border-right: 1px solid {BORDER}; font-weight: bold; }}
        """)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        bottom = QtWidgets.QHBoxLayout()
        self.status_label = make_label("", color=TEXT_SEC)
        cancel_btn = make_btn("Cancel", self.reject)
        select_btn = make_btn("Select Current Folder", self.select_current)
        set_role(select_btn, "accent")
        bottom.addWidget(self.status_label)
        bottom.addStretch()
        bottom.addWidget(cancel_btn)
        bottom.addWidget(select_btn)
        layout.addLayout(bottom)

    def _populate_sidebar(self):
        paths = QtCore.QStandardPaths
        for text, loc in [("Desktop", paths.StandardLocation.DesktopLocation),
                          ("Documents", paths.StandardLocation.DocumentsLocation),
                          ("Pictures", paths.StandardLocation.PicturesLocation),
                          ("Downloads", paths.StandardLocation.DownloadLocation)]:
            p = paths.writableLocation(loc)
            if os.path.exists(p):
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(p)), text)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, p)
                self.sidebar.addItem(item)
        sep = QtWidgets.QListWidgetItem("───── Drives ─────")
        sep.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        sep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.sidebar.addItem(sep)
        for vol in QtCore.QStorageInfo.mountedVolumes():
            if vol.isValid() and vol.isReady():
                name = vol.name() or vol.rootPath()
                item = QtWidgets.QListWidgetItem(self.icon_provider.icon(QFileInfo(vol.rootPath())), name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, vol.rootPath())
                self.sidebar.addItem(item)

    def _on_path_entered(self):
        p = platform_path(self.path_edit.text())
        if os.path.isdir(p): self.load_directory(p)
        else: QtWidgets.QMessageBox.warning(self, "Invalid Path", "Path does not exist or is not a directory.")

    def load_directory(self, path):
        if not self.is_navigating_history:
            if self.history_idx == -1 or self.history[self.history_idx] != path:
                self.history = self.history[:self.history_idx + 1]
                self.history.append(path)
                self.history_idx += 1
        self.btn_back.setEnabled(self.history_idx > 0)
        self.btn_fwd.setEnabled(self.history_idx < len(self.history) - 1)
        self.is_navigating_history = False
        self.current_path = path
        self.path_edit.setText(path)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.status_label.setText("Scanning...")
        QtWidgets.QApplication.processEvents()

        img_exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}
        try:
            entries = [e for e in os.scandir(path) if e.is_dir()]
            self.table.setRowCount(len(entries))
            for row, entry in enumerate(entries):
                info = QFileInfo(entry.path)
                self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(self.icon_provider.icon(info), entry.name))
                try:
                    count = sum(1 for f in os.scandir(entry.path)
                                if f.is_file() and os.path.splitext(f.name)[1].lower() in img_exts)
                    ci = NumericTableWidgetItem(str(count))
                    ci.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    ci.setForeground(QtGui.QColor(ACCENT2 if count > 0 else TEXT_SEC))
                    if count > 0: ci.setFont(fixed_width_font(9, bold=True))
                    self.table.setItem(row, 1, ci)
                except PermissionError:
                    self.table.setItem(row, 1, NumericTableWidgetItem("-1"))
                try:
                    ts = entry.stat().st_mtime
                    di = DateTableWidgetItem(datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M"), ts)
                    di.setForeground(QtGui.QColor(TEXT_SEC))
                    self.table.setItem(row, 2, di)
                except Exception:
                    self.table.setItem(row, 2, DateTableWidgetItem("N/A", 0))
                self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(entry.path))
            self.status_label.setText(f"{len(entries)} folders found.")
        except PermissionError:
            QtWidgets.QMessageBox.warning(self, "Access Denied", f"Cannot access {path}")
            self.go_back()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))
        self.table.setSortingEnabled(True)

    def go_up(self):
        parent = os.path.dirname(self.current_path)
        if parent and parent != self.current_path: self.load_directory(parent)

    def go_back(self):
        if self.history_idx > 0:
            self.history_idx -= 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def go_forward(self):
        if self.history_idx < len(self.history) - 1:
            self.history_idx += 1
            self.is_navigating_history = True
            self.load_directory(self.history[self.history_idx])

    def select_current(self):
        rows = self.table.selectionModel().selectedRows()
        self.selected_path = (self.table.item(rows[0].row(), 3).text() if rows else self.current_path)
        self.accept()


class GraphPanel(QtWidgets.QWidget):
    def __init__(self, title, y_label, parent=None):
        super().__init__(parent)
        self.title = title
        self.y_label = y_label
        self.lines = []
        # The top band is a compact, Chart.js-style series legend.
        # It replaces the centered title and stays outside the plot area.
        self.padding = {'top': 42, 'bottom': 40, 'left': 70, 'right': 20}
        self.bg_color = THEME.color("canvas")
        self.graph_bg_color = THEME.color("canvas")
        self.grid_color = THEME.color("border")
        self.text_color = THEME.color("text")
        self.title_color = THEME.color("accent")
        self.x_min, self.x_max = 0, 100
        self.y_min, self.y_max = 0, 1
        self.data_x_min, self.data_x_max = 0, 100
        self.fill_enabled = False
        self.view_x_min = None
        self.view_x_max = None
        self.render_x_min = None
        self.render_x_max = None
        self.target_y_min = 0
        self.target_y_max = 1
        self.render_y_min = None
        self.render_y_max = None
        self.drag_start_pos = None
        self.drag_start_range = None
        self.hover_point = None
        self._dirty_bounds = True
        self.anim_timer = QtCore.QTimer(self)
        self.anim_timer.timeout.connect(self._animate_view)
        self.repaint_timer = QtCore.QTimer(self)
        self.repaint_timer.setSingleShot(True)
        self.repaint_timer.timeout.connect(self.update)
        self.setMouseTracking(True)
        self.setMinimumHeight(220)

    def _graph_rect(self):
        return QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])

    def add_line(self, color, label, max_points=2000, linewidth=2, line_style="solid"):
        self.lines.append({
            'data': [],
            'x_values': [],
            'max_points': max_points,
            'version': 0,
            'color': QtGui.QColor(color),
            'label': label,
            'linewidth': linewidth,
            'line_style': line_style,
            'visible': True,
        })
        return len(self.lines) - 1

    def set_line_visible(self, line_index, visible):
        if 0 <= line_index < len(self.lines):
            self.lines[line_index]['visible'] = bool(visible)
            self._dirty_bounds = True
            self.update()

    def append_data(self, line_index, x, y):
        if 0 <= line_index < len(self.lines):
            line = self.lines[line_index]
            if line['x_values'] and x <= line['x_values'][-1]:
                pos = bisect_left(line['x_values'], x)
                if pos < len(line['x_values']) and line['x_values'][pos] == x:
                    line['data'][pos] = (x, y)
                else:
                    line['data'].insert(pos, (x, y))
                    line['x_values'].insert(pos, x)
            else:
                line['data'].append((x, y))
                line['x_values'].append(x)
            line['version'] += 1
            if len(line['data']) > line['max_points']:
                self._compact_line(line)
            self._refresh_data_range()
            self._fit_full_range(False)
            self._dirty_bounds = True
            self._schedule_repaint()

    def clear_all_data(self):
        for line in self.lines:
            line['data'].clear()
            line['x_values'].clear()
            line['version'] += 1
        self.view_x_min = None
        self.view_x_max = None
        self.render_x_min = None
        self.render_x_max = None
        self.render_y_min = None
        self.render_y_max = None
        self.hover_point = None
        self._dirty_bounds = True
        self._update_bounds()
        self.update()

    def _refresh_data_range(self):
        firsts = [line['data'][0][0] for line in self.lines if line['data']]
        lasts = [line['data'][-1][0] for line in self.lines if line['data']]
        if not firsts or not lasts:
            self.data_x_min, self.data_x_max = 0, 100
            self.view_x_min, self.view_x_max = None, None
            return
        self.data_x_min = min(firsts)
        self.data_x_max = max(lasts)
        if self.data_x_min == self.data_x_max:
            self.data_x_max = self.data_x_min + 1

    def _fit_full_range(self, animate=True):
        self.view_x_min = self.data_x_min
        self.view_x_max = self.data_x_max
        if not animate:
            self.render_x_min = self.view_x_min
            self.render_x_max = self.view_x_max
            self.render_y_min = None
            self.render_y_max = None
        self._dirty_bounds = True
        if animate:
            self._start_view_animation()

    def _compact_line(self, line):
        data = line['data']
        target = max(256, line['max_points'] // 2)
        if len(data) <= target:
            return
        bucket_count = max(2, (target - 2) // 2)
        middle = data[1:-1]
        bucket_size = len(middle) / bucket_count
        compacted = [data[0]]
        for bucket in range(bucket_count):
            start = int(bucket * bucket_size)
            end = int((bucket + 1) * bucket_size)
            if bucket == bucket_count - 1:
                end = len(middle)
            segment = middle[start:end]
            if not segment:
                continue
            min_i = min(range(len(segment)), key=lambda i: segment[i][1])
            max_i = max(range(len(segment)), key=lambda i: segment[i][1])
            for local_i in sorted({min_i, max_i}):
                compacted.append(segment[local_i])
        compacted.append(data[-1])
        line['data'] = compacted
        line['x_values'] = [x for x, _ in compacted]
        line['version'] += 1

    def _get_visible_slice(self, line):
        data = line['data']
        if not data: return []
        view_min = self.x_min
        view_max = self.x_max
        if len(data) <= 2:
            return data[:]
        x_values = line['x_values']
        start = bisect_left(x_values, view_min)
        end = bisect_right(x_values, view_max)
        start = max(0, start - 1)
        end = min(len(data), end + 1)
        if start >= end:
            if start >= len(data):
                return data[-1:]
            return data[start:start + 1]
        return data[start:end]

    def _sample_visible_points(self, raw, max_points):
        count = len(raw)
        if count <= max_points:
            return raw[:count]

        bucket_count = max(2, max_points // 2)
        bucket_size = count / bucket_count
        sampled = []

        for bucket in range(bucket_count):
            start = int(bucket * bucket_size)
            end = int((bucket + 1) * bucket_size)
            if bucket == bucket_count - 1:
                end = count
            if end <= start:
                continue

            segment = raw[start:end]
            if not segment:
                continue
            min_i = min(range(len(segment)), key=lambda i: segment[i][1])
            max_i = max(range(len(segment)), key=lambda i: segment[i][1])
            for local_i in sorted({min_i, max_i}):
                sampled.append(raw[start + local_i])

        return sampled

    def _update_bounds(self):
        all_y, all_x = [], []
        target_x_min = self.view_x_min if self.view_x_min is not None else self.data_x_min
        target_x_max = self.view_x_max if self.view_x_max is not None else self.data_x_max
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min, self.render_x_max = target_x_min, target_x_max
        self.x_min, self.x_max = self.render_x_min, self.render_x_max
        for line in self.lines:
            if not line.get('visible', True): continue
            raw = self._get_visible_slice(line)
            if raw:
                all_x.extend(x for x, _ in raw)
                all_y.extend(y for _, y in raw)
        if all_x:
            self.x_min = self.render_x_min
            self.x_max = self.render_x_max
            if self.x_min == self.x_max:
                self.x_max = self.x_min + 1
            if all_y:
                yr = max(all_y) - min(all_y) or 1
                self.target_y_min = min(all_y) - yr * 0.08
                self.target_y_max = max(all_y) + yr * 0.08
            else:
                self.target_y_min, self.target_y_max = 0, 1
        else:
            self.x_min, self.x_max = 0, 100
            self.target_y_min, self.target_y_max = 0, 1
        if self.render_y_min is None or self.render_y_max is None:
            self.render_y_min, self.render_y_max = self.target_y_min, self.target_y_max
        self.y_min, self.y_max = self.render_y_min, self.render_y_max
        self._dirty_bounds = False

    def _to_screen(self, x, y):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        xr = self.x_max - self.x_min or 1
        yr = self.y_max - self.y_min or 1
        sx = self.padding['left'] + ((x - self.x_min) / xr) * gw
        sy = self.padding['top'] + gh - ((y - self.y_min) / yr) * gh
        return QtCore.QPointF(sx, sy)

    def _from_screen_x(self, px):
        gr = self._graph_rect()
        xr = self.x_max - self.x_min or 1
        return self.x_min + ((px - gr.left()) / max(1, gr.width())) * xr

    def paintEvent(self, event):
        if self._dirty_bounds:
            self._update_bounds()
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        gr = self._graph_rect()
        painter.fillRect(gr, self.graph_bg_color)
        self._draw_grid(painter, gr)
        self._draw_lines(painter, gr)
        self._draw_legend(painter)
        self._draw_hover(painter, gr)

    def _draw_grid(self, painter, rect):
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            y_val = self.y_max - (i / 4) * (self.y_max - self.y_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self._fmt(y_val))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            x_val = self.x_min + (i / 5) * (self.x_max - self.x_min)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(x_val)))
            painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setPen(self.text_color)
        f = painter.font(); f.setPixelSize(12); painter.setFont(f)
        painter.save()
        painter.translate(15, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-50, -10, 100, 20), QtCore.Qt.AlignmentFlag.AlignCenter, self.y_label)
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Step")

    def _draw_lines(self, painter, rect):
        painter.save()
        painter.setClipRect(rect)
        max_points = max(128, rect.width() * 2)
        for line in self.lines:
            if not line.get('visible', True): continue
            raw = self._get_visible_slice(line)
            if len(raw) < 2: continue
            sampled = self._sample_visible_points(raw, max_points)
            pts = [self._to_screen(x, y) for x, y in sampled]
            if len(pts) < 2:
                continue
            if self.fill_enabled:
                poly = QtGui.QPolygonF(pts)
                poly.append(QtCore.QPointF(pts[-1].x(), rect.bottom()))
                poly.append(QtCore.QPointF(pts[0].x(), rect.bottom()))
                fc = QtGui.QColor(line['color']); fc.setAlpha(self._fill_alpha(len(raw)))
                painter.setBrush(fc); painter.setPen(QtCore.Qt.PenStyle.NoPen)
                painter.drawPolygon(poly)
            width = self._line_width(line)
            painter.setPen(self._line_pen(line, width))
            painter.drawPolyline(QtGui.QPolygonF(pts))
        painter.restore()

    def _line_pen(self, line, width=None):
        pen = QtGui.QPen(line['color'], width if width is not None else self._line_width(line))
        style = line.get('line_style', 'solid')
        if style == 'dotted':
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([1.0, 3.0])
            pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        elif style == 'dashed':
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([6.0, 3.0])
        return pen

    def _line_width(self, line):
        count = len(line['data'])
        extra = min(1.4, math.log10(max(1, count)) * 0.25)
        return line['linewidth'] + extra

    def _fill_alpha(self, visible_count):
        if visible_count <= 16:
            return 64
        if visible_count <= 128:
            return 52
        return 38


    def _draw_legend(self, painter):
        lx = self.padding['left']
        ly = 13
        f = painter.font(); f.setPixelSize(12); f.setBold(False); painter.setFont(f)
        for line in self.lines:
            if not line.get('visible', True): continue
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(line['color'])
            painter.drawRoundedRect(QtCore.QRectF(lx, ly, 10, 10), 2, 2)
            painter.setPen(self.text_color)
            label_width = painter.fontMetrics().horizontalAdvance(line['label'])
            painter.drawText(QtCore.QRect(lx + 16, ly - 3, label_width + 2, 16),
                             QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, line['label'])
            lx += 16 + label_width + 22

    def _draw_hover(self, painter, rect):
        if not self.hover_point:
            return
        line, x, y, pt = self.hover_point
        hc = QtGui.QColor(line['color'])
        painter.setPen(QtGui.QPen(hc, 1))
        painter.drawLine(QtCore.QPointF(pt.x(), rect.top()), QtCore.QPointF(pt.x(), rect.bottom()))
        painter.drawLine(QtCore.QPointF(rect.left(), pt.y()), QtCore.QPointF(rect.right(), pt.y()))
        painter.setBrush(hc)
        painter.drawEllipse(pt, 4, 4)
        text = f"{line['label']}  Step {int(x)}  {self._fmt(y)}"
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 16
        h = 24
        tx = min(max(rect.left() + 6, int(pt.x()) + 10), rect.right() - w - 6)
        ty = max(rect.top() + 6, int(pt.y()) - h - 10)
        box = QtCore.QRect(tx, ty, w, h)
        painter.setPen(QtGui.QPen(self.grid_color, 1))
        painter.setBrush(self.bg_color)
        painter.drawRoundedRect(box, 4, 4)
        painter.setPen(self.text_color)
        painter.drawText(box.adjusted(8, 0, -8, 0), QtCore.Qt.AlignmentFlag.AlignVCenter, text)

    def _fmt(self, v):
        if abs(v) < 0.01 or abs(v) > 10000: return f"{v:.1e}"
        if abs(v) < 1: return f"{v:.4f}"
        return f"{v:.2f}"

    def set_fill(self, e): self.fill_enabled = e; self.update()

    def wheelEvent(self, event):
        gr = self._graph_rect()
        if not gr.contains(event.position().toPoint()):
            event.ignore()
            return
        if self.view_x_min is None or self.view_x_max is None:
            self._fit_full_range()
        steps = max(-4, min(4, event.angleDelta().y() / 120))
        factor = 0.94 ** steps
        center = self._from_screen_x(event.position().x())
        span = max(1e-9, (self.view_x_max - self.view_x_min) * factor)
        full_span = max(1e-9, self.data_x_max - self.data_x_min)
        min_span = max(1, full_span / 1000000)
        span = max(min_span, min(span, full_span))
        rel = (center - self.view_x_min) / max(1e-9, self.view_x_max - self.view_x_min)
        self.view_x_min = center - span * rel
        self.view_x_max = self.view_x_min + span
        self._clamp_view()
        self._dirty_bounds = True
        self._start_view_animation()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._graph_rect().contains(event.position().toPoint()):
            self.drag_start_pos = event.position()
            self.drag_start_range = (self.view_x_min, self.view_x_max)
            self.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_start_pos is not None and self.drag_start_range[0] is not None:
            dx = event.position().x() - self.drag_start_pos.x()
            span = self.drag_start_range[1] - self.drag_start_range[0]
            shift = -dx / max(1, self._graph_rect().width()) * span
            self.view_x_min = self.drag_start_range[0] + shift
            self.view_x_max = self.drag_start_range[1] + shift
            self._clamp_view()
            self._dirty_bounds = True
            self._start_view_animation()
            event.accept()
            return
        self.hover_point = self._nearest_point(event.position())
        self._schedule_repaint()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drag_start_pos = None
            self.drag_start_range = None
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            event.accept()

    def leaveEvent(self, event):
        self.hover_point = None
        self._schedule_repaint()

    def _nearest_point(self, pos):
        gr = self._graph_rect()
        if not gr.contains(pos.toPoint()):
            return None
        nearest = None
        nearest_dist = 12 * 12
        for line in self.lines:
            for x, y in self._get_visible_slice(line):
                pt = self._to_screen(x, y)
                dist = (pt.x() - pos.x()) ** 2 + (pt.y() - pos.y()) ** 2
                if dist < nearest_dist:
                    nearest = (line, x, y, pt)
                    nearest_dist = dist
        return nearest

    def _clamp_view(self):
        if self.view_x_min is None or self.view_x_max is None:
            return
        span = self.view_x_max - self.view_x_min
        full_span = self.data_x_max - self.data_x_min
        if span >= full_span:
            self.view_x_min = self.data_x_min
            self.view_x_max = self.data_x_max
        elif self.view_x_min < self.data_x_min:
            self.view_x_min = self.data_x_min
            self.view_x_max = self.data_x_min + span
        elif self.view_x_max > self.data_x_max:
            self.view_x_max = self.data_x_max
            self.view_x_min = self.data_x_max - span

    def _start_view_animation(self):
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min = self.view_x_min
            self.render_x_max = self.view_x_max
        if not self.anim_timer.isActive():
            self.anim_timer.start(16)
        self._schedule_repaint()

    def _animate_view(self):
        if self.view_x_min is None or self.view_x_max is None:
            self.anim_timer.stop()
            return
        if self.render_x_min is None or self.render_x_max is None:
            self.render_x_min, self.render_x_max = self.view_x_min, self.view_x_max
        self._dirty_bounds = True
        self._update_bounds()
        targets = [
            (self.render_x_min, self.view_x_min),
            (self.render_x_max, self.view_x_max),
            (self.render_y_min, self.target_y_min),
            (self.render_y_max, self.target_y_max),
        ]
        done = True
        next_values = []
        for current, target in targets:
            if current is None:
                current = target
            delta = target - current
            scale = max(1.0, abs(target))
            if abs(delta) > scale * 0.001:
                done = False
            next_values.append(current + delta * 0.28)
        self.render_x_min, self.render_x_max, self.render_y_min, self.render_y_max = next_values
        if done:
            self.render_x_min, self.render_x_max = self.view_x_min, self.view_x_max
            self.render_y_min, self.render_y_max = self.target_y_min, self.target_y_max
            self.anim_timer.stop()
        self._dirty_bounds = True
        self._schedule_repaint()

    def _schedule_repaint(self):
        if not self.repaint_timer.isActive():
            self.repaint_timer.start(0)


class LiveTimestepHistogram(QtWidgets.QWidget):
    def __init__(self, bin_count=20, parent=None):
        super().__init__(parent)
        self.counts = [0] * bin_count
        self.y_axis_label = "Samples"
        self.value_max = 1000.0
        self.x_axis_label = "Timestep"
        self.value_name = "Timestep"
        self.bar_color = QtGui.QColor(ACCENT2)
        self.bar_color_alt = THEME.color("accent_deep")
        self.padding = {'top': 42, 'bottom': 40, 'left': 70, 'right': 20}
        self.setMinimumHeight(220)

    def append_value(self, value):
        index = min(len(self.counts) - 1, max(0, int(float(value) / self.value_max * len(self.counts))))
        self.counts[index] += 1
        self.update()

    def append_values(self, values):
        for value in values:
            index = min(len(self.counts) - 1, max(0, int(float(value) / self.value_max * len(self.counts))))
            self.counts[index] += 1
        self.update()

    def set_bin_count(self, bin_count):
        bin_count = max(1, int(bin_count))
        if len(self.counts) != bin_count:
            self.counts = [0] * bin_count
            self.update()

    def set_value_mode(self, mode):
        value_max, label = (1.0, "Sigma") if mode == "sigma" else (1000.0, "Timestep")
        if self.value_max != value_max or self.x_axis_label != label:
            self.value_max = value_max
            self.x_axis_label = label
            self.value_name = label
            self.clear_all_data()

    def clear_all_data(self):
        self.counts = [0] * len(self.counts)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), THEME.color("canvas"))
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        painter.fillRect(rect, THEME.color("canvas"))
        maximum = max(1, max(self.counts, default=0))
        painter.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1))
        for value in self.y_grid_values(maximum):
            y = rect.bottom() - (value / maximum) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            painter.setPen(QtGui.QColor(TEXT_PRI))
            painter.drawText(QtCore.QRect(5, int(y - 10), self.padding['left'] - 10, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             self.format_y_tick(value))
            painter.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1))
        width = rect.width() / len(self.counts)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for index, count in enumerate(self.counts):
            height = count / maximum * rect.height()
            bar = QtCore.QRectF(rect.left() + index * width + 1, rect.bottom() - height,
                                max(1, width - 2), height)
            color = QtGui.QColor(self.bar_color if index % 2 == 0 else self.bar_color_alt)
            color.setAlpha(210)
            painter.setBrush(color)
            painter.drawRoundedRect(bar, 2, 2)
        painter.setPen(QtGui.QColor(TEXT_PRI))
        for i in range(6):
            x = rect.left() + i / 5 * rect.width()
            axis_value = i / 5 * self.value_max
            axis_text = f"{axis_value:.1f}" if self.value_max == 1.0 else str(round(axis_value))
            painter.drawText(QtCore.QRect(int(x - 30), rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, axis_text)
        painter.save(); painter.translate(15, self.height() / 2); painter.rotate(-90)
        painter.drawText(QtCore.QRect(-50, -10, 100, 20), QtCore.Qt.AlignmentFlag.AlignCenter, self.y_axis_label)
        painter.restore()
        painter.drawText(QtCore.QRect(0, self.height() - 20, self.width(), 20),
                         QtCore.Qt.AlignmentFlag.AlignCenter, self.x_axis_label)
        swatch = QtCore.QRectF(self.padding['left'], 13, 10, 10)
        clip_path = QtGui.QPainterPath()
        clip_path.addRoundedRect(swatch, 2, 2)
        painter.save()
        painter.setClipPath(clip_path)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        top_right = QtGui.QPolygonF([
            swatch.topLeft(), swatch.topRight(), swatch.bottomRight()
        ])
        bottom_left = QtGui.QPolygonF([
            swatch.topLeft(), swatch.bottomRight(), swatch.bottomLeft()
        ])
        painter.setBrush(self.bar_color)
        painter.drawPolygon(top_right)
        painter.setBrush(self.bar_color_alt)
        painter.drawPolygon(bottom_left)
        painter.restore()
        painter.setPen(QtGui.QColor(TEXT_PRI))
        painter.drawText(QtCore.QRect(self.padding['left'] + 16, 10, 280, 16),
                         QtCore.Qt.AlignmentFlag.AlignVCenter,
                         self.legend_text())

    def y_grid_values(self, maximum):
        if maximum <= 4:
            return list(range(int(maximum) + 1))
        return [maximum * i / 4 for i in range(5)]

    def format_y_tick(self, value):
        return str(round(value))

    def legend_text(self):
        return f"{self.value_name} Samples ({sum(self.counts):,} total)"


class LiveTimestepMeanLossHistogram(LiveTimestepHistogram):
    def __init__(self, bin_count=20, parent=None):
        super().__init__(bin_count, parent)
        self.loss_sums = [0.0] * bin_count
        self.sample_counts = [0] * bin_count
        self.y_axis_label = "Mean Loss"
        self.bar_color = QtGui.QColor(WARN)
        self.bar_color_alt = THEME.color("warning_deep")

    def set_bin_count(self, bin_count):
        bin_count = max(1, int(bin_count))
        if len(self.counts) != bin_count:
            self.counts = [0.0] * bin_count
            self.loss_sums = [0.0] * bin_count
            self.sample_counts = [0] * bin_count
            self.update()

    def append_sample(self, timestep, loss):
        index = min(len(self.counts) - 1, max(0, int(float(timestep) / self.value_max * len(self.counts))))
        self.loss_sums[index] += float(loss)
        self.sample_counts[index] += 1
        self.counts[index] = self.loss_sums[index] / self.sample_counts[index]
        self.update()

    def append_samples(self, values, loss):
        for value in values:
            index = min(len(self.counts) - 1, max(0, int(float(value) / self.value_max * len(self.counts))))
            self.loss_sums[index] += float(loss)
            self.sample_counts[index] += 1
            self.counts[index] = self.loss_sums[index] / self.sample_counts[index]
        self.update()

    def clear_all_data(self):
        size = len(self.counts)
        self.counts = [0.0] * size
        self.loss_sums = [0.0] * size
        self.sample_counts = [0] * size
        self.update()

    def y_grid_values(self, maximum):
        return [maximum * i / 4 for i in range(5)]

    def format_y_tick(self, value):
        return f"{value:.4f}" if value < 1 else f"{value:.3f}"

    def legend_text(self):
        return f"Mean Loss by {self.value_name} ({sum(self.sample_counts):,} samples)"


class LiveMetricsWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.max_points = 60000
        self.pending_update = False
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_update)
        self.pending_data = deque()
        self.max_pending_per_tick = 2000
        self.graphs = {}
        self._setup_ui()

    def _make_graph_container(self, name, title, y_label):
        gb, lay = group_box("")
        set_role(gb, "flat")
        if name == "timestep":
            graph = LiveTimestepHistogram()
        elif name == "timestep_loss":
            graph = LiveTimestepMeanLossHistogram()
        else:
            graph = GraphPanel(title, y_label)
        self.graphs[name] = {'widget': graph, 'lines': {}}
        lay.addWidget(graph, 1)

        ctrl = QtWidgets.QHBoxLayout()
        fill_chk = QtWidgets.QCheckBox("Fill")
        fill_chk.setChecked(True)
        if name not in ("timestep", "timestep_loss"):
            fill_chk.stateChanged.connect(lambda s, g=graph: g.set_fill(s == QtCore.Qt.CheckState.Checked.value))
            graph.set_fill(True)
            ctrl.addWidget(fill_chk)
        else:
            fill_chk.hide()
        if name in ("step_loss", "optim_loss"):
            raw_chk = QtWidgets.QCheckBox("Show raw loss")
            set_semantic_color(raw_chk, "raw")
            raw_chk.toggled.connect(lambda checked, n=name: self._set_loss_raw_mode(n, checked))
            ctrl.addWidget(raw_chk)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        return gb

    def _add_line(self, graph_name, line_name, color, linewidth=2, line_style="solid"):
        g = self.graphs[graph_name]
        idx = g['widget'].add_line(color, line_name, self.max_points, linewidth, line_style)
        g['lines'][line_name] = idx

    def _set_loss_raw_mode(self, graph_name, show_raw):
        graph = self.graphs[graph_name]
        raw_name = "Step Loss" if graph_name == "step_loss" else "Optimizer Loss"
        ema_name = "Loss EMA" if graph_name == "step_loss" else "Optimizer Loss EMA"
        graph['widget'].set_line_visible(graph['lines'][raw_name], show_raw)
        graph['widget'].set_line_visible(graph['lines'][ema_name], not show_raw)

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)

        ctrl = QtWidgets.QHBoxLayout()
        self.clear_btn = make_btn("Clear Data", self.clear_data)
        self.pause_btn = QtWidgets.QPushButton("Pause Updates")
        self.pause_btn.setCheckable(True)
        self.pause_btn.toggled.connect(self._on_pause_toggled)
        self.stats_label = make_label("No data yet", color=ACCENT2, bold=True)
        for w in [self.clear_btn, self.pause_btn, None, self.stats_label]:
            (ctrl.addStretch() if w is None else ctrl.addWidget(w))
        main.addLayout(ctrl)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(10)
        for name, title, y_label, row, col, row_span, col_span in [
            ("step_loss", "Per-Step Loss", "Loss", 0, 0, 1, 1),
            ("timestep", "Timestep", "Value", 0, 1, 1, 1),
            ("optim_loss", "Optimizer Loss", "Loss", 1, 0, 1, 1),
            ("lr", "Learning Rate", "LR", 1, 1, 1, 1),
            ("timestep_loss", "Mean Loss by Timestep", "Mean Loss", 2, 0, 1, 1),
            ("grad_norm", "Gradient Norms", "Norm", 2, 1, 1, 1),
        ]:
            grid.addWidget(self._make_graph_container(name, title, y_label), row, col, row_span, col_span)
        main.addLayout(grid)

        for graph_name, line_name, color, width, line_style in [
            ("step_loss", "Step Loss", ACCENT2, 2, "solid"),
            ("step_loss", "Loss EMA", WARN, 3, "solid"),
            ("optim_loss", "Optimizer Loss", ACCENT2, 2, "solid"),
            ("optim_loss", "Optimizer Loss EMA", WARN, 3, "solid"),
            ("lr", "LR", ACCENT2, 2, "solid"),
            ("grad_norm", "Clipped", WARN, 2, "dotted"),
            ("grad_norm", "Raw", ACCENT2, 4, "solid"),
        ]:
            self._add_line(graph_name, line_name, color, width, line_style)
        self.graphs['step_loss']['widget'].set_line_visible(self.graphs['step_loss']['lines']['Step Loss'], False)
        self.graphs['optim_loss']['widget'].set_line_visible(self.graphs['optim_loss']['lines']['Optimizer Loss'], False)
        self.loss_ema_beta = 0.98
        self.step_loss_ema = self.optim_loss_ema = None

        self.latest_global_step = self.latest_optim_step = self.latest_timestep = 0
        self.latest_sigma = None
        self.latest_lr = self.latest_step_loss = self.latest_optim_loss = self.latest_grad = 0.0

    def _on_pause_toggled(self, paused):
        if paused:
            self.update_timer.stop()
        elif self.pending_update:
            self._queue_update()

    def set_timestep_bucket_size(self, bucket_size):
        bucket_size = max(1, int(bucket_size))
        bucket_count = math.ceil(1000 / bucket_size)
        self.graphs['timestep']['widget'].set_bin_count(bucket_count)
        self.graphs['timestep_loss']['widget'].set_bin_count(bucket_count)

    def parse_and_update(self, text):
        if self.pause_btn.isChecked(): return
        added = False
        anima_match = re.search(r'Training\s*\|.*\|\s*(\d+)/(\d+)\s*\[.*?\]\s*\[Loss:\s*([\d.e+-]+),\s*Ticket:\s*(\d+),\s*Sigma:\s*([\d.e+-]+)\]', text)
        timestep_match = re.search(r'Training\s*\|.*\|\s*(\d+)/(\d+)\s*\[.*?\]\s*\[Loss:\s*([\d.e+-]+),\s*Timestep:\s*(\d+)\]', text)
        m = anima_match or timestep_match
        if m:
            step, loss, ticket = int(m.group(1)) - 1, float(m.group(3)), int(m.group(4))
            sigma = float(m.group(5)) if anima_match else None
            # Anima stratification is defined in exact integer ticket space.
            # Its BF16 training sigma can round across a visual bucket boundary
            # (for example 0.08 -> 0.080078), so use the ticket-bin center for
            # histogram placement while continuing to label the axis as Sigma.
            graph_values = [((ticket + 0.5) / 1000.0) if sigma is not None else ticket]
            graph_mode = 'sigma' if sigma is not None else 'timestep'
            self.pending_data.append(('progress_step', step, loss, ticket, graph_values, graph_mode))
            self.latest_global_step, self.latest_step_loss, self.latest_timestep = step, loss, ticket
            self.latest_sigma = sigma
            added = True
        m = re.search(r'--- Optimizer Step:\s*(\d+)\s*\|\s*Loss:\s*([\d.e+-]+)\s*\|\s*LR:\s*([\d.e+-]+)\s*---', text)
        if m:
            step, avg_loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
            self.pending_data.append(('optim_step', step, avg_loss, lr))
            self.latest_optim_step, self.latest_lr, self.latest_optim_loss = step, lr, avg_loss
            added = True
        m = re.search(r'Grad Norm \(Raw/Clipped\):\s*([\d.]+)\s*/\s*([\d.]+)', text)
        if m:
            self.pending_data.append(('grad', float(m.group(1)), float(m.group(2))))
            self.latest_grad = float(m.group(1))
            added = True
        if added:
            self.pending_update = True
            self._queue_update()

    def _queue_update(self):
        if not self.pause_btn.isChecked() and not self.update_timer.isActive():
            self.update_timer.start(0)

    def _perform_update(self):
        if not self.pending_update or not self.pending_data:
            self.update_timer.stop(); return
        last_optim_step = self.latest_optim_step
        processed = 0
        while self.pending_data and processed < self.max_pending_per_tick:
            data = self.pending_data.popleft()
            processed += 1
            t = data[0]
            if t == 'progress_step':
                _, step, loss, ticket, graph_values, graph_mode = data
                self.graphs['step_loss']['widget'].append_data(self.graphs['step_loss']['lines']['Step Loss'], step, loss)
                self.step_loss_ema = loss if self.step_loss_ema is None else self.loss_ema_beta * self.step_loss_ema + (1.0 - self.loss_ema_beta) * loss
                self.graphs['step_loss']['widget'].append_data(self.graphs['step_loss']['lines']['Loss EMA'], step, self.step_loss_ema)
                self.graphs['timestep']['widget'].set_value_mode(graph_mode)
                self.graphs['timestep_loss']['widget'].set_value_mode(graph_mode)
                self.graphs['timestep']['widget'].append_values(graph_values)
                self.graphs['timestep_loss']['widget'].append_samples(graph_values, loss)
            elif t == 'optim_step':
                _, step, avg_loss, lr = data
                last_optim_step = step
                self.graphs['optim_loss']['widget'].append_data(self.graphs['optim_loss']['lines']['Optimizer Loss'], step, avg_loss)
                self.optim_loss_ema = avg_loss if self.optim_loss_ema is None else self.loss_ema_beta * self.optim_loss_ema + (1.0 - self.loss_ema_beta) * avg_loss
                self.graphs['optim_loss']['widget'].append_data(self.graphs['optim_loss']['lines']['Optimizer Loss EMA'], step, self.optim_loss_ema)
                self.graphs['lr']['widget'].append_data(self.graphs['lr']['lines']['LR'], step, lr)
            elif t == 'grad' and last_optim_step is not None:
                _, raw, clipped = data
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Raw'], last_optim_step, raw)
                self.graphs['grad_norm']['widget'].append_data(self.graphs['grad_norm']['lines']['Clipped'], last_optim_step, clipped)
        self.latest_optim_step = last_optim_step
        sampling_status = (
            f"Ticket: {self.latest_timestep} | Sigma: {self.latest_sigma:.6f}"
            if self.latest_sigma is not None else f"Timestep: {self.latest_timestep}"
        )
        self.stats_label.setText(
            f"Step: {self.latest_global_step} | Loss: {self.latest_step_loss:.4f} | "
            f"{sampling_status} | Optimizer Loss: {self.latest_optim_loss:.4f} | "
            f"LR: {self.latest_lr:.2e} | Grad: {self.latest_grad:.4f}")
        self.pending_update = bool(self.pending_data)
        if self.pending_update:
            self._queue_update()

    def clear_data(self):
        self.update_timer.stop()
        self.pending_update = False
        self.pending_data.clear()
        for gd in self.graphs.values():
            gd['widget'].clear_all_data()
        self.latest_global_step = self.latest_optim_step = self.latest_timestep = 0
        self.latest_sigma = None
        self.latest_lr = self.latest_step_loss = self.latest_optim_loss = self.latest_grad = 0.0
        self.step_loss_ema = self.optim_loss_ema = None
        self.stats_label.setText("No data yet")

    def showEvent(self, e):
        super().showEvent(e)
        if self.pending_update and not self.pause_btn.isChecked():
            self._queue_update()

    def hideEvent(self, e):
        super().hideEvent(e)
        if not self.pending_update:
            self.update_timer.stop()


class LRCurveWidget(QtWidgets.QWidget):
    pointsChanged = QtCore.pyqtSignal(list)
    selectionChanged = QtCore.pyqtSignal(int)
    LOG_FLOOR_DIVISOR = 10000.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self._points = []
        self._visual_points = []
        self.max_steps = 10000
        self.min_lr_bound = 0.0
        self.max_lr_bound = 1.0e-6
        self.epoch_data = []
        self.epoch_step_interval = 0
        self.epoch_marker_count = 0
        self.padding = {'top': 40, 'bottom': 78, 'left': 78, 'right': 20}
        self.point_radius = 8
        self._dragging_point_index = -1
        self._selected_point_index = -1
        self.bg_color = THEME.color("canvas")
        self.grid_color = THEME.color("border")
        self.epoch_grid_color = THEME.color("text_muted")
        self.line_color = THEME.color("accent_alt")
        self.point_color = THEME.color("text")
        self.point_fill_color = THEME.color("accent_alt")
        self.selected_point_color = THEME.color("warning")
        self.text_color = THEME.color("text")
        self.setMouseTracking(True)

    def set_epoch_data(self, d):
        self.epoch_data = d
        self.epoch_step_interval = 0
        self.epoch_marker_count = len(d)
        self.update()

    def set_epoch_interval(self, step_interval, marker_count):
        self.epoch_data = []
        self.epoch_step_interval = max(0, int(step_interval))
        self.epoch_marker_count = max(0, int(marker_count))
        self.update()

    def set_bounds(self, max_steps, min_lr, max_lr):
        self.max_steps = max(max_steps, 1)
        self.min_lr_bound = min_lr
        self.max_lr_bound = max_lr if max_lr > min_lr else min_lr + 1e-9
        changed = False
        for p in self._points:
            clamped = max(self.min_lr_bound, min(self.max_lr_bound, p[1]))
            if clamped != p[1]: p[1] = clamped; changed = True
        if changed: self.pointsChanged.emit(self._points)
        self._update_visual(); self.update()

    def set_points(self, points):
        self._points = sorted(points, key=lambda p: p[0])
        self._update_visual(); self.update()

    def _update_visual(self):
        self._visual_points = [[p[0], max(self.min_lr_bound, min(self.max_lr_bound, p[1]))] for p in self._points]

    def get_points(self): return self._points

    def _log_range(self):
        safe_max = max(self.max_lr_bound, 1e-12)
        log_max = math.log(safe_max)
        eff_min = max(self.min_lr_bound if self.min_lr_bound > 0 else safe_max / self.LOG_FLOOR_DIVISOR, 1e-12)
        return log_max, math.log(eff_min)

    def _to_pixel(self, nx, lr):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        px = self.padding['left'] + nx * gw
        if lr <= self.min_lr_bound: return QtCore.QPointF(px, self.padding['top'] + gh)
        log_max, log_min = self._log_range()
        lr_range = log_max - log_min
        if lr_range <= 0: return QtCore.QPointF(px, self.padding['top'])
        ny = (math.log(lr) - log_min) / lr_range
        return QtCore.QPointF(px, self.padding['top'] + (1 - ny) * gh)

    def _to_data(self, px, py):
        gw = self.width() - self.padding['left'] - self.padding['right']
        gh = self.height() - self.padding['top'] - self.padding['bottom']
        nx = (px - self.padding['left']) / gw
        ny = 1 - ((max(self.padding['top'], min(py, self.padding['top'] + gh)) - self.padding['top']) / gh)
        log_max, log_min = self._log_range()
        lr_range = log_max - log_min
        lr = math.exp(log_min + ny * lr_range) if lr_range > 0 else self.min_lr_bound
        return max(0.0, min(1.0, nx)), max(self.min_lr_bound, min(self.max_lr_bound, lr))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        gr = QtCore.QRect(self.padding['left'], self.padding['top'],
                          self.width() - self.padding['left'] - self.padding['right'],
                          self.height() - self.padding['top'] - self.padding['bottom'])
        self._draw_grid(painter, gr)
        self._draw_curve(painter)
        self._draw_points(painter)

    def _draw_grid(self, painter, rect):
        painter.setPen(self.grid_color)
        log_max, log_min = self._log_range()
        log_range = log_max - log_min
        f = self.font(); f.setPixelSize(11); painter.setFont(f)
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            painter.setPen(self.grid_color)
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            ny = 1.0 - i / 4
            lr_val = (self.max_lr_bound if i == 0 else self.min_lr_bound if i == 4
                      else math.exp(log_min + ny * log_range) if log_range > 0 else self.max_lr_bound)
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(0, int(y - 10), self.padding['left'] - 5, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, f"{lr_val:.1e}")
            x_step = rect.left() + (i / 4) * rect.width()
            painter.drawText(QtCore.QRect(int(x_step - 50), rect.bottom() + 5, 100, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(self.max_steps * i / 4)))

        self._draw_epoch_markers(painter, rect)

        painter.setPen(self.text_color)
        axis_font = self.font(); axis_font.setPixelSize(11); painter.setFont(axis_font)
        painter.save()
        painter.translate(14, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-70, -9, 140, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Learning Rate")
        painter.restore()
        painter.drawText(QtCore.QRect(rect.left(), rect.bottom() + 44, rect.width(), 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Training Step")

        painter.setPen(self.text_color)
        f3 = self.font(); f3.setBold(True); f3.setPixelSize(12); painter.setFont(f3)
        painter.drawText(QtCore.QRect(rect.left(), 5, rect.width(), 25),
                         QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                         "Learning Rate Schedule")

    def _draw_epoch_markers(self, painter, rect):
        marker_count = self.epoch_marker_count or len(self.epoch_data)
        if marker_count <= 0:
            return

        ep_pen = QtGui.QPen(self.epoch_grid_color)
        ep_pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setPen(ep_pen)

        label_font = self.font()
        label_font.setPixelSize(11)
        painter.setFont(label_font)
        min_label_px = 78

        if self.epoch_step_interval > 0:
            spacing_px = rect.width() * self.epoch_step_interval / max(self.max_steps, 1)
            draw_every = max(1, math.ceil(1.0 / max(spacing_px, 0.001)))
            label_every = max(draw_every, math.ceil(min_label_px / max(spacing_px, 0.001)))

            lines = []
            for i in range(draw_every, marker_count + 1, draw_every):
                step = i * self.epoch_step_interval
                if step >= self.max_steps:
                    break
                x = rect.left() + (step / self.max_steps) * rect.width()
                ix = int(round(x))
                lines.append(QtCore.QLine(ix, rect.top(), ix, rect.bottom()))
            if lines:
                painter.drawLines(lines)

            painter.setPen(self.text_color)
            for i in range(label_every, marker_count + 1, label_every):
                step = i * self.epoch_step_interval
                if step >= self.max_steps:
                    break
                x = rect.left() + (step / self.max_steps) * rect.width()
                painter.drawText(QtCore.QRect(int(x - 40), rect.bottom() + 25, 80, 15),
                                 QtCore.Qt.AlignmentFlag.AlignCenter, str(step))
            return

        previous_label_x = -10_000
        for nx, steps in self.epoch_data:
            x = rect.left() + nx * rect.width()
            painter.setPen(ep_pen)
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            if x - previous_label_x >= min_label_px:
                painter.setPen(self.text_color)
                painter.drawText(QtCore.QRect(int(x - 40), rect.bottom() + 25, 80, 15),
                                 QtCore.Qt.AlignmentFlag.AlignCenter, str(steps))
                previous_label_x = x

    def _draw_curve(self, painter):
        if not self._visual_points: return
        pts = [self._to_pixel(p[0], p[1]) for p in self._visual_points]
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(QtGui.QPolygonF(pts))
        fill = QtGui.QPolygonF(pts)
        fill.append(self._to_pixel(self._visual_points[-1][0], self.min_lr_bound))
        fill.append(self._to_pixel(self._visual_points[0][0], self.min_lr_bound))
        fc = QtGui.QColor(self.point_fill_color); fc.setAlpha(50)
        painter.setBrush(fc); painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(fill)

    def _draw_points(self, painter):
        for i, p in enumerate(self._visual_points):
            pos = self._to_pixel(p[0], p[1])
            is_sel = i == self._selected_point_index
            painter.setBrush(self.selected_point_color if is_sel else self.point_fill_color)
            painter.setPen(self.point_color)
            painter.drawEllipse(pos, self.point_radius, self.point_radius)
            if is_sel or i == self._dragging_point_index:
                op = self._points[i]
                painter.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 30, 100, 20),
                                 QtCore.Qt.AlignmentFlag.AlignCenter,
                                 f"({int(op[0] * self.max_steps)}, {op[1]:.1e})")

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        new_sel = -1
        for i, p in enumerate(self._visual_points):
            if (QtCore.QPointF(event.pos()) - self._to_pixel(p[0], p[1])).manhattanLength() < self.point_radius * 1.5:
                self._dragging_point_index = i; new_sel = i; break
        if self._selected_point_index != new_sel:
            self._selected_point_index = new_sel
            self.selectionChanged.emit(new_sel)
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_point_index != -1:
            nx, lr = self._to_data(event.pos().x(), event.pos().y())
            i = self._dragging_point_index
            if i == 0 or i == len(self._points) - 1:
                nx = 0.0 if i == 0 else 1.0
            else:
                nx = max(self._points[i-1][0], min(self._points[i+1][0], nx))
            self._points[i] = [nx, lr]
            self._update_visual(); self.pointsChanged.emit(self._points); self.update()
        else:
            on_pt = any((QtCore.QPointF(event.pos()) - self._to_pixel(p[0], p[1])).manhattanLength() < self.point_radius * 1.5 for p in self._visual_points)
            self.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor if on_pt else QtCore.Qt.CursorShape.ArrowCursor))

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton: return
        self._dragging_point_index = -1
        self.set_points(self._points); self.pointsChanged.emit(self._points)

    def add_point(self):
        if len(self._points) < 2: return
        max_gap, insert_idx = 0, -1
        for i in range(len(self._points) - 1):
            gap = self._points[i+1][0] - self._points[i][0]
            if gap > max_gap: max_gap = gap; insert_idx = i + 1
        if insert_idx != -1:
            prev, nxt = self._points[insert_idx-1], self._points[insert_idx]
            _, log_min = self._log_range()
            new_lr = math.exp(max(log_min, (math.log(max(prev[1], 1e-12)) + math.log(max(nxt[1], 1e-12))) / 2))
            self._points.insert(insert_idx, [(prev[0] + nxt[0]) / 2, new_lr])
            self.set_points(self._points); self.pointsChanged.emit(self._points)

    def remove_selected_point(self):
        i = self._selected_point_index
        if 0 < i < len(self._points) - 1:
            self._points.pop(i)
            self._selected_point_index = -1
            self.selectionChanged.emit(-1)
            self.set_points(self._points); self.pointsChanged.emit(self._points)

    def apply_preset(self, pts): self.set_points(pts); self.pointsChanged.emit(pts)

    def set_standard_preset(self, mode):
        min_lr, max_lr = self.min_lr_bound, self.max_lr_bound
        warmup_end = 0.05

        if mode == "Constant":
            points = [
                [0.0, min_lr],
                [warmup_end, max_lr],
                [0.95, max_lr],
                [1.0, min_lr],
            ]
        elif mode == "Linear":
            points = [
                [0.0, min_lr],
                [warmup_end, max_lr],
                [1.0, min_lr],
            ]
        elif mode == "Cosine":
            points = [[0.0, min_lr], [warmup_end, max_lr]]
            for index in range(1, 21):
                progress = index / 20
                x = warmup_end + progress * (1.0 - warmup_end)
                y = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                points.append([x, y])
        else:
            raise ValueError(f"Unknown learning-rate preset: {mode}")

        self.apply_preset(points)


class TimestepHistogramWidget(QtWidgets.QWidget):
    allocationChanged = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(240)
        self.padding = {'top': 32, 'bottom': 52, 'left': 58, 'right': 20}
        self.bin_size = 100
        self.max_tickets = 1000
        self.counts = []
        self._dragging_bin_index = -1
        self.bg_color = THEME.color("canvas")
        self.bar_color_even = THEME.color("warning")
        self.bar_color_odd = THEME.color("warning_deep")
        self.bar_hover_color = THEME.color("warning_hover")
        self.disabled_bar_color = THEME.color("border")
        self.text_color = THEME.color("text")
        self.grid_color = THEME.color("border")
        self.alert_color = THEME.color("danger")
        self.disabled_text_color = THEME.color("text_muted")
        self.setMouseTracking(True)
        self._init_bins()

    def set_total_steps(self, steps):
        steps = max(int(steps), 1)
        self.max_tickets = steps
        cur = sum(self.counts)
        if not self.counts or cur == 0: self._init_bins(); return
        raw = [(c / cur) * steps for c in self.counts]
        new = [int(x) for x in raw]
        diff = steps - sum(new)
        fracs = sorted(enumerate(raw[i] - new[i] for i in range(len(new))), key=lambda x: x[1], reverse=True)
        for i in range(diff): new[fracs[i][0]] += 1
        self.counts = new
        self.update(); self._emit_change()

    def set_bin_size(self, size):
        if size <= 0: return
        self.bin_size = size
        self._init_bins(); self.update(); self._emit_change()

    def set_allocation(self, alloc):
        if not alloc or "bin_size" not in alloc or "counts" not in alloc:
            self._init_bins(); return
        self.bin_size = alloc["bin_size"]
        expected = math.ceil(1000 / self.bin_size)
        if len(alloc["counts"]) != expected:
            self._init_bins()
        else:
            self.counts = alloc["counts"]
            s = sum(self.counts)
            if s > 0: self.max_tickets = s
        self.update()

    def get_allocation(self): return {"bin_size": self.bin_size, "counts": self.counts}

    def generate_from_weights(self, weights):
        n = len(self.counts)
        if n == 0 or not weights: return
        tw = sum(weights) or 1
        raw = [(w / tw) * self.max_tickets for w in weights]
        new = [int(c) for c in raw]
        diff = self.max_tickets - sum(new)
        fracs = sorted(enumerate(raw[i] - new[i] for i in range(n)), key=lambda x: x[1], reverse=True)
        for i in range(diff): new[fracs[i][0]] += 1
        self.counts = new; self.update(); self._emit_change()

    def _init_bins(self):
        self.bin_size = max(self.bin_size, 1)
        n = max(math.ceil(1000 / self.bin_size), 1)
        base, rem = divmod(self.max_tickets, n)
        self.counts = [base + (1 if i < rem else 0) for i in range(n)]
        self._emit_change()

    def _emit_change(self): self.allocationChanged.emit(self.get_allocation())

    def _bin_rect(self, idx, rect):
        n = len(self.counts)
        if n == 0: return QtCore.QRectF()
        bw = rect.width() / n
        x = rect.left() + idx * bw
        y_scale = max((max(self.counts) if self.counts else 1) * 1.35, 1.0)
        h = (self.counts[idx] / y_scale) * rect.height()
        return QtCore.QRectF(x, rect.bottom() - h, bw, h)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        enabled = self.isEnabled()
        self._paint_grid(painter, rect, enabled)
        self._paint_bars(painter, rect, enabled)
        self._paint_labels(painter, rect, enabled)
        self._paint_header(painter, rect, enabled)

    def _paint_grid(self, painter, rect, enabled):
        gc = self.grid_color if enabled else self.disabled_text_color
        tc = self.text_color if enabled else self.disabled_text_color
        painter.setPen(QtGui.QPen(gc, 1, QtCore.Qt.PenStyle.DashLine))
        for i in range(5):
            y = rect.bottom() - (i / 4) * rect.height()
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.setPen(tc)
            painter.drawText(QtCore.QRectF(x - 30, rect.bottom() + 5, 60, 20),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(i / 5 * 1000)))
            painter.setPen(QtGui.QPen(gc, 1, QtCore.Qt.PenStyle.DashLine))

    def _paint_bars(self, painter, rect, enabled):
        f = painter.font(); f.setPixelSize(11); painter.setFont(f)
        for i, count in enumerate(self.counts):
            br = self._bin_rect(i, rect)
            color = (self.bar_hover_color if i == self._dragging_bin_index else
                     (self.bar_color_odd if i % 2 else self.bar_color_even)) if enabled else self.disabled_bar_color
            painter.setBrush(color); painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRect(br)
            if enabled:
                painter.setPen(self.text_color)
                ty = br.top() - 20 - (15 if i % 2 else 0)
                if ty < rect.top(): ty = br.top() + 5 if br.height() > 40 else rect.top() + 5
                painter.drawText(QtCore.QRectF(br.left(), ty, br.width(), 20), QtCore.Qt.AlignmentFlag.AlignCenter, str(count))

    def _paint_labels(self, painter, rect, enabled):
        painter.setPen(self.text_color if enabled else self.disabled_text_color)
        label_font = painter.font()
        label_font.setPixelSize(11)
        painter.setFont(label_font)
        painter.save()
        painter.translate(14, self.height() / 2)
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-70, -9, 140, 18), QtCore.Qt.AlignmentFlag.AlignCenter, "Tickets")
        painter.restore()
        painter.drawText(QtCore.QRect(rect.left(), rect.bottom() + 30, rect.width(), 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter, "Timestep (0-1000)")

    def _paint_header(self, painter, rect, enabled):
        used, total = sum(self.counts), self.max_tickets
        f = painter.font(); f.setBold(True); f.setPixelSize(12); painter.setFont(f)
        painter.setPen(QtGui.QColor(WARN if used == total else DANGER))
        painter.drawText(QtCore.QRect(rect.left(), 5, rect.width(), 25),
                         QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                         f"Tickets Used: {used} / {total}")

    def mousePressEvent(self, event):
        if not self.isEnabled() or event.button() != QtCore.Qt.MouseButton.LeftButton: return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        rel_x = event.pos().x() - rect.left()
        n = len(self.counts)
        if n and 0 <= rel_x <= rect.width():
            self._dragging_bin_index = int(rel_x / (rect.width() / n)); self.update()

    def mouseMoveEvent(self, event):
        if not self.isEnabled(): return
        rect = QtCore.QRect(self.padding['left'], self.padding['top'],
                            self.width() - self.padding['left'] - self.padding['right'],
                            self.height() - self.padding['top'] - self.padding['bottom'])
        n = len(self.counts)
        rel_x = event.pos().x() - rect.left()
        if self._dragging_bin_index == -1:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if n and 0 <= rel_x <= rect.width() else QtCore.Qt.CursorShape.ArrowCursor)
            return
        idx = self._dragging_bin_index
        if not (0 <= idx < n): return
        y_scale = max((max(self.counts) if self.counts else 1) * 1.35, 1.0)
        target = int(((rect.bottom() - event.pos().y()) / rect.height()) * y_scale)
        avail = self.max_tickets - (sum(self.counts) - self.counts[idx])
        new_val = max(0, min(target, avail))
        if new_val != self.counts[idx]:
            self.counts[idx] = new_val; self._emit_change(); self.update()

    def mouseReleaseEvent(self, event):
        self._dragging_bin_index = -1; self.update()


class TimestepLossWeightCurveWidget(QtWidgets.QWidget):
    pointsChanged = QtCore.pyqtSignal(list)
    selectionChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.padding = {'top': 30, 'bottom': 56, 'left': 74, 'right': 18}
        self.point_radius = 7
        self.min_weight = 0.0
        self.max_weight = 2.0
        self._points = [[0.0, 1.0], [1.0, 1.0]]
        self._preset = None
        self._dragging_point_index = -1
        self._drag_had_motion = False
        self._selected_point_index = -1
        self.bg_color = THEME.color("canvas")
        self.grid_color = THEME.color("border")
        self.line_color = THEME.color("warning")
        self.point_color = THEME.color("text")
        self.point_fill_color = THEME.color("warning")
        self.selected_point_color = THEME.color("accent_alt")
        self.text_color = THEME.color("text")
        self.setMouseTracking(True)

    def set_points(self, points):
        self._preset = None
        if isinstance(points, dict) and str(points.get("preset", "")).lower() == "bell":
            self._preset = "bell"
            points = self._bell_preview_points()
        cleaned = []
        for p in points or []:
            try:
                x = max(0.0, min(1.0, float(p[0])))
                y = max(self.min_weight, min(self.max_weight, float(p[1])))
                cleaned.append([x, y])
            except (TypeError, ValueError, IndexError):
                continue
        if len(cleaned) < 2:
            cleaned = [[0.0, 1.0], [1.0, 1.0]]
        cleaned.sort(key=lambda p: p[0])
        cleaned[0][0] = 0.0
        cleaned[-1][0] = 1.0
        self._points = cleaned
        self._selected_point_index = -1
        self.selectionChanged.emit(-1)
        self.update()

    def get_points(self):
        if self._preset == "bell":
            return {"preset": "bell"}
        return [[round(p[0], 8), round(p[1], 4)] for p in self._points]

    def apply_preset(self, points):
        self._preset = None
        self.set_points(points)
        self.pointsChanged.emit(self.get_points())

    def apply_bell_preset(self):
        self.set_points({"preset": "bell"})
        self.pointsChanged.emit(self.get_points())

    def apply_min_snr_like_preset(self):
        self.apply_preset([
            [0.0, 0.0043],
            [0.025025, 0.1198],
            [0.05005, 0.2544],
            [0.075075, 0.4107],
            [0.1001, 0.5914],
            [0.125125, 0.7999],
            [0.15015, 1.0],
            [1.0, 1.0],
        ])

    @staticmethod
    def _bell_preview_points():
        steps = 1000
        values = [math.exp(-2.0 * ((i - steps / 2) / steps) ** 2) for i in range(steps)]
        y_min = min(values)
        denom = sum(v - y_min for v in values) or 1.0
        scale = steps / denom
        sample_indices = [0, 125, 250, 375, 500, 625, 750, 875, 999]
        return [[i / (steps - 1), (values[i] - y_min) * scale] for i in sample_indices]

    def add_point(self):
        self._preset = None
        if len(self._points) < 2:
            return
        max_gap, insert_idx = 0.0, -1
        for i in range(len(self._points) - 1):
            gap = self._points[i + 1][0] - self._points[i][0]
            if gap > max_gap:
                max_gap, insert_idx = gap, i + 1
        if insert_idx == -1:
            return
        prev, nxt = self._points[insert_idx - 1], self._points[insert_idx]
        self._points.insert(insert_idx, [(prev[0] + nxt[0]) / 2, (prev[1] + nxt[1]) / 2])
        self.set_points(self._points)
        self.pointsChanged.emit(self.get_points())

    def remove_selected_point(self):
        self._preset = None
        i = self._selected_point_index
        if 0 < i < len(self._points) - 1:
            self._points.pop(i)
            self.set_points(self._points)
            self.pointsChanged.emit(self.get_points())

    def _graph_rect(self):
        return QtCore.QRect(
            self.padding['left'],
            self.padding['top'],
            self.width() - self.padding['left'] - self.padding['right'],
            self.height() - self.padding['top'] - self.padding['bottom'],
        )

    def _to_pixel(self, x, y):
        rect = self._graph_rect()
        px = rect.left() + x * rect.width()
        ny = (y - self.min_weight) / max(self.max_weight - self.min_weight, 1e-9)
        py = rect.bottom() - ny * rect.height()
        return QtCore.QPointF(px, py)

    def _to_data(self, px, py):
        rect = self._graph_rect()
        x = (px - rect.left()) / max(rect.width(), 1)
        ny = 1.0 - ((max(rect.top(), min(py, rect.bottom())) - rect.top()) / max(rect.height(), 1))
        y = self.min_weight + ny * (self.max_weight - self.min_weight)
        return max(0.0, min(1.0, x)), max(self.min_weight, min(self.max_weight, y))

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.bg_color)
        rect = self._graph_rect()
        font = painter.font()
        font.setPixelSize(11)
        painter.setFont(font)
        for i in range(5):
            y = rect.top() + (i / 4) * rect.height()
            value = self.max_weight - (i / 4) * (self.max_weight - self.min_weight)
            painter.setPen(self.grid_color)
            painter.drawLine(rect.left(), int(y), rect.right(), int(y))
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(0, int(y - 10), self.padding['left'] - 5, 20),
                             QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                             f"{value:.1f}")
        for i in range(6):
            x = rect.left() + (i / 5) * rect.width()
            painter.setPen(self.grid_color)
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
            painter.setPen(self.text_color)
            painter.drawText(QtCore.QRect(int(x - 30), rect.bottom() + 4, 60, 18),
                             QtCore.Qt.AlignmentFlag.AlignCenter, str(int(i / 5 * 1000)))
        painter.setPen(self.text_color)
        axis_font = painter.font()
        axis_font.setPixelSize(11)
        painter.setFont(axis_font)
        painter.save()
        painter.translate(14, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QtCore.QRect(-70, -9, 140, 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter,
                         "Loss Weight")
        painter.restore()
        painter.drawText(QtCore.QRect(rect.left(), rect.bottom() + 24, rect.width(), 18),
                         QtCore.Qt.AlignmentFlag.AlignCenter,
                         "Timestep (0-1000)")
        painter.setPen(self.text_color)
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPixelSize(12)
        painter.setFont(title_font)
        painter.drawText(QtCore.QRect(rect.left(), 4, rect.width(), 22),
                         QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop,
                         "Loss Weight")

        pts = [self._to_pixel(p[0], p[1]) for p in self._points]
        if len(pts) >= 2:
            fill_path = QtGui.QPainterPath()
            fill_path.moveTo(pts[0].x(), rect.bottom())
            fill_path.lineTo(pts[0])
            for pos in pts[1:]:
                fill_path.lineTo(pos)
            fill_path.lineTo(pts[-1].x(), rect.bottom())
            fill_path.closeSubpath()
            fill_color = QtGui.QColor(self.line_color)
            fill_color.setAlpha(45)
            painter.save()
            painter.setClipRect(rect)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawPath(fill_path)
            painter.restore()
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(QtGui.QPolygonF(pts))
        for i, point in enumerate(self._points):
            pos = self._to_pixel(point[0], point[1])
            painter.setBrush(self.selected_point_color if i == self._selected_point_index else self.point_fill_color)
            painter.setPen(self.point_color)
            painter.drawEllipse(pos, self.point_radius, self.point_radius)
            if i == self._selected_point_index or i == self._dragging_point_index:
                painter.drawText(QtCore.QRectF(pos.x() - 42, pos.y() - 27, 84, 18),
                                 QtCore.Qt.AlignmentFlag.AlignCenter,
                                 f"{point[1]:.2f}")

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        new_sel = -1
        for i, point in enumerate(self._points):
            if (QtCore.QPointF(event.pos()) - self._to_pixel(point[0], point[1])).manhattanLength() < self.point_radius * 1.8:
                self._dragging_point_index = i
                self._drag_had_motion = False
                new_sel = i
                break
        if self._selected_point_index != new_sel:
            self._selected_point_index = new_sel
            self.selectionChanged.emit(new_sel)
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging_point_index != -1:
            self._preset = None
            self._drag_had_motion = True
            x, y = self._to_data(event.pos().x(), event.pos().y())
            i = self._dragging_point_index
            if i == 0 or i == len(self._points) - 1:
                x = 0.0 if i == 0 else 1.0
            else:
                x = max(self._points[i - 1][0] + 1e-4, min(self._points[i + 1][0] - 1e-4, x))
            self._points[i] = [x, y]
            self.pointsChanged.emit(self.get_points())
            self.update()
            return
        on_point = any(
            (QtCore.QPointF(event.pos()) - self._to_pixel(p[0], p[1])).manhattanLength() < self.point_radius * 1.8
            for p in self._points
        )
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor if on_point else QtCore.Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        self._dragging_point_index = -1
        if self._drag_had_motion:
            self.set_points(self._points)
            self.pointsChanged.emit(self.get_points())
        self._drag_had_motion = False


class ProcessRunner(QThread):
    logSignal = pyqtSignal(str)
    paramInfoSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(str, bool)
    finishedSignal = pyqtSignal(int)
    errorSignal = pyqtSignal(str)
    metricsSignal = pyqtSignal(str)
    cacheCreatedSignal = pyqtSignal()

    def __init__(self, executable, args, working_dir, env=None, creation_flags=0):
        super().__init__()
        self.executable = executable
        self.args = args
        self.working_dir = working_dir
        self.env = env
        self.creation_flags = creation_flags
        self.process = None
        self.stop_requested = False

    @staticmethod
    def _clean_output_line(line):
        return ANSI_ESCAPE_RE.sub("", line)

    def run(self):
        try:
            flags = self.creation_flags
            popen_options = {}
            if IS_WINDOWS:
                flags |= (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.HIGH_PRIORITY_CLASS)
                popen_options["creationflags"] = flags
            else:
                # Keep the trainer and any data-loader children in one stoppable group.
                popen_options["start_new_session"] = True
            self.process = subprocess.Popen(
                [self.executable] + self.args,
                cwd=self.working_dir, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                universal_newlines=True, bufsize=1, **popen_options)
            self.logSignal.emit(f"INFO: Started subprocess (PID: {self.process.pid})")
            for line in iter(self.process.stdout.readline, ''):
                line = self._clean_output_line(line.strip())
                if not line or "NOTE: Redirects are currently not supported" in line: continue
                if line.startswith("GUI_PARAM_INFO::"):
                    self.paramInfoSignal.emit(line.replace('GUI_PARAM_INFO::', '').strip())
                else:
                    is_progress = '\r' in line or bool(re.match(r'^\s*\d+%\|\S*\|', line))
                    if any(kw in line.lower() for kw in ["memory inaccessible", "cuda out of memory", "access violation", "nan/inf"]):
                        self.logSignal.emit(f"*** ERROR DETECTED: {line} ***")
                    else:
                        self.progressSignal.emit(line.split('\r')[-1], is_progress)
                    self.metricsSignal.emit(line)
                    if "saved latents cache" in line.lower() or "caching complete" in line.lower() or "anima dit items" in line.lower():
                        self.cacheCreatedSignal.emit()
            self.finishedSignal.emit(self.process.wait())
        except Exception as e:
            self.errorSignal.emit(f"Subprocess error: {str(e)}")
            self.finishedSignal.emit(-1)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stop_requested = True
            if IS_WINDOWS:
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if IS_WINDOWS:
                    self.process.kill()
                else:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
            self.logSignal.emit("Process stopped.")


class DatasetLoaderThread(QThread):
    finished = pyqtSignal(list, str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.repeats = 1

    def run(self):
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        images_data = []
        try:
            for file_path in Path(self.path).rglob("*"):
                if file_path.suffix.lower() in exts:
                    cap_path = file_path.with_suffix('.txt')
                    json_path = file_path.with_suffix('.json')
                    images_data.append({"image_path": str(file_path),
                                        "caption_path": str(cap_path) if cap_path.exists() else None,
                                        "json_caption_path": str(json_path) if json_path.exists() else None,
                                        "caption_loaded": False, "caption": ""})
        except Exception as e:
            print(f"Error scanning {self.path}: {e}")
        self.finished.emit(images_data, self.path)


class DatasetManagerWidget(QtWidgets.QWidget):
    datasetsChanged = QtCore.pyqtSignal()

    def __init__(self, parent_gui):
        super().__init__()
        self.parent_gui = parent_gui
        self.datasets = []
        self.dataset_widgets = []
        self.dataset_nav_entries = []
        self.active_dataset_index = 0
        self.loader_threads = []
        self._previews_active = True
        self._preview_queue = []
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(False)
        self._preview_timer.setInterval(15)
        self._preview_timer.timeout.connect(self._load_next_queued_preview)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        dataset_tools = QtWidgets.QVBoxLayout()
        dataset_tools.setContentsMargins(0, 0, 0, 0)
        dataset_tools.setSpacing(5)
        add_row = QtWidgets.QHBoxLayout()
        add_row.setSpacing(5)
        standard_btn = make_btn("Add Folder", self.add_dataset_folder_native)
        smart_btn = make_btn("⌕", self.add_dataset_folder_custom)
        set_role(standard_btn, "segmentLeft")
        set_role(smart_btn, "segmentRight")
        smart_btn.setToolTip("Smart folder browser")
        standard_btn.setFixedHeight(24)
        standard_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        smart_btn.setFixedSize(34, 24)
        add_folder_control = QtWidgets.QWidget()
        add_folder_layout = QtWidgets.QHBoxLayout(add_folder_control)
        add_folder_layout.setContentsMargins(0, 0, 0, 0)
        add_folder_layout.setSpacing(0)
        add_folder_layout.addWidget(standard_btn, 1)
        add_folder_layout.addWidget(smart_btn, 0)
        add_folder_control.setMinimumWidth(100)
        add_folder_control.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        add_row.addWidget(add_folder_control, 1)
        self.sort_combo = make_combo(["Default (Order Added)", "Name (A-Z)", "Name (Z-A)",
                                      "Image Count (High → Low)", "Image Count (Low → High)"])
        self.sort_combo.clear()
        for label, key in [
            ("Sort: Added", "default"),
            ("Sort: Name A–Z", "name_asc"),
            ("Sort: Name Z–A", "name_desc"),
            ("Sort: Images ↓", "images_desc"),
            ("Sort: Images ↑", "images_asc"),
        ]:
            self.sort_combo.addItem(label, key)
        self.sort_combo.setMinimumWidth(100)
        self.sort_combo.setFixedHeight(24)
        self.sort_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.sort_combo.currentIndexChanged.connect(self.sort_datasets)
        add_row.addWidget(self.sort_combo, 1)
        dataset_tools.addLayout(add_row)
        self.loading_label = make_label("", color=WARN, bold=True)
        self.loading_label.setVisible(False)

        content_row = QtWidgets.QHBoxLayout()
        content_row.setContentsMargins(0, 0, 0, 0)
        content_row.setSpacing(0)

        self.dataset_content_splitter = ThemedSplitter(QtCore.Qt.Orientation.Horizontal)
        self.dataset_content_splitter.setChildrenCollapsible(False)

        focus_group, focus_lay = group_box("Dataset List")
        set_role(focus_group, "datasetNavigator")
        focus_group.setMinimumWidth(220)
        focus_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        focus_lay.setContentsMargins(6, 10, 6, 6)
        focus_lay.setSpacing(5)
        focus_lay.addLayout(dataset_tools)
        self.dataset_search = QtWidgets.QLineEdit()
        self.dataset_search.setMinimumWidth(120)
        self.dataset_search.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.dataset_search.setPlaceholderText("Search datasets...")
        self.dataset_search.setClearButtonEnabled(True)
        self.dataset_search.textChanged.connect(self._search_dataset_navigator)
        focus_lay.addWidget(self.dataset_search)
        focus_lay.addWidget(self.loading_label)
        self.quick_focus_list = EmptyStateListWidget(
            "No datasets detected. Click Add Folder to add one."
        )
        set_role(self.quick_focus_list, "quickFocus")
        self.quick_focus_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.quick_focus_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.quick_focus_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.quick_focus_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.quick_focus_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.quick_focus_list.itemClicked.connect(self._select_dataset_item)
        self.quick_focus_list.customContextMenuRequested.connect(self._show_dataset_context_menu)
        self.remove_selected_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence(QtCore.Qt.Key.Key_Delete), self.quick_focus_list
        )
        self.remove_selected_shortcut.activated.connect(self._remove_selected_datasets)
        focus_lay.addWidget(self.quick_focus_list)

        list_pane = QtWidgets.QWidget()
        list_pane_layout = QtWidgets.QVBoxLayout(list_pane)
        list_pane_layout.setContentsMargins(10, 15, 10, 15)
        dataset_heading = make_label("Datasets", color=ACCENT2, bold=True, size=12)
        list_pane_layout.addWidget(dataset_heading)
        list_pane_layout.addWidget(focus_group, 1)
        self.dataset_content_splitter.addWidget(list_pane)

        self.grid_container = QtWidgets.QWidget()
        self.dataset_grid = QtWidgets.QGridLayout(self.grid_container)
        self.dataset_grid.setSpacing(15)
        self.dataset_grid.setContentsMargins(0, 0, 0, 0)
        self.dataset_grid.setRowStretch(0, 1)
        self.grid_container.setMinimumWidth(260)

        detail_panel = QtWidgets.QWidget()
        detail_layout = QtWidgets.QGridLayout(detail_panel)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(0)
        detail_layout.addWidget(self.grid_container, 0, 0)

        detail_pane = QtWidgets.QWidget()
        detail_pane_layout = QtWidgets.QVBoxLayout(detail_pane)
        detail_pane_layout.setContentsMargins(10, 15, 15, 15)
        detail_heading_spacer = QtWidgets.QWidget()
        detail_heading_spacer.setFixedHeight(dataset_heading.sizeHint().height())
        detail_pane_layout.addWidget(detail_heading_spacer)
        detail_pane_layout.addWidget(detail_panel, 1)
        self.dataset_content_splitter.addWidget(detail_pane)
        self.dataset_content_splitter.setStretchFactor(0, 0)
        self.dataset_content_splitter.setStretchFactor(1, 1)
        self.dataset_content_splitter.setSizes([300, 885])
        content_row.addWidget(self.dataset_content_splitter, 1)
        layout.addLayout(content_row, 1)

    def get_total_repeats(self): return gui_math.repeated_image_count(self.datasets)
    def get_datasets_config(self): return [{"path": d["path"], "repeats": d["repeats"]} for d in self.datasets]

    def _select_dataset_item(self, item):
        index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(index, int) and 0 <= index < len(self.datasets):
            selected = self._selected_dataset_indices()
            self.active_dataset_index = index if index in selected else (selected[0] if selected else index)
            self.repopulate_dataset_grid(selected)

    def _selected_dataset_indices(self):
        return sorted({
            index
            for item in self.quick_focus_list.selectedItems()
            for index in [item.data(QtCore.Qt.ItemDataRole.UserRole)]
            if isinstance(index, int) and 0 <= index < len(self.datasets)
        })

    def _show_dataset_context_menu(self, position):
        item = self.quick_focus_list.itemAt(position)
        if item is None:
            return
        index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if index not in self._selected_dataset_indices():
            self.quick_focus_list.clearSelection()
            item.setSelected(True)
            self.quick_focus_list.setCurrentItem(item)
        count = len(self._selected_dataset_indices())
        menu = QtWidgets.QMenu(self.quick_focus_list)

        action_panel = QtWidgets.QGroupBox("Dataset Actions")
        action_panel.setProperty("density", "compact")
        action_panel.setFixedWidth(250)
        panel_layout = QtWidgets.QVBoxLayout(action_panel)
        panel_layout.setContentsMargins(8, 10, 8, 8)
        panel_layout.setSpacing(7)
        selection_text = "1 dataset selected" if count == 1 else f"{count} datasets selected"
        selection_label = make_label(selection_text, color=TEXT_SEC, size=8)
        panel_layout.addWidget(selection_label)
        remove_button = make_btn("Remove Dataset" if count == 1 else f"Remove {count} Datasets")
        set_role(remove_button, "danger")
        remove_button.setToolTip("Remove the selected dataset entries from this training configuration.")
        remove_button.clicked.connect(menu.close)
        remove_button.clicked.connect(self._remove_selected_datasets)
        panel_layout.addWidget(remove_button)

        panel_action = QtWidgets.QWidgetAction(menu)
        panel_action.setDefaultWidget(action_panel)
        menu.addAction(panel_action)
        menu.exec(self.quick_focus_list.viewport().mapToGlobal(position))

    def _remove_selected_datasets(self):
        indices = self._selected_dataset_indices()
        if not indices:
            return
        for index in reversed(indices):
            del self.datasets[index]
        self.active_dataset_index = min(indices[0], max(0, len(self.datasets) - 1))
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def _search_dataset_navigator(self, text):
        """Temporarily rank navigator entries without changing dataset order."""
        if not hasattr(self, "quick_focus_list") or self.quick_focus_list.count() < 2:
            return
        query = str(text).strip().casefold()
        active_item = None
        for row in range(self.quick_focus_list.count()):
            item = self.quick_focus_list.item(row)
            index = item.data(QtCore.Qt.ItemDataRole.UserRole)
            name = (Path(self.datasets[index]["path"]).name or self.datasets[index]["path"]).casefold()
            if not query:
                tier, position, distance = 0, 0, 0
            else:
                similarity = SequenceMatcher(None, query, name).ratio()
                distance = int(round((1.0 - similarity) * 1_000_000))
                if name == query:
                    tier, position = 0, 0
                elif name.startswith(query):
                    tier, position = 1, 0
                else:
                    position = name.find(query)
                    tier = 2 if position >= 0 else 3
                    position = max(0, position)
            # The custom entry widget covers this text; it exists only as a
            # stable sort key, avoiding destructive item removal/reinsertion.
            item.setText(f"{tier}:{position:06d}:{distance:07d}:{index:09d}")
            if index == self.active_dataset_index:
                active_item = item
        self.quick_focus_list.sortItems(QtCore.Qt.SortOrder.AscendingOrder)
        if active_item is not None:
            self.quick_focus_list.setCurrentItem(active_item)

    def _start_loading_path(self, path, repeats=1):
        self.loading_label.setText(f"Scanning: {Path(path).name}...")
        self.loading_label.setVisible(True)
        loader = DatasetLoaderThread(path)
        loader.repeats = repeats
        loader.finished.connect(self._on_loader_finished)
        self.loader_threads.append(loader)
        loader.start()

    def _on_loader_finished(self, images_data, path):
        sender = self.sender()
        repeats = getattr(sender, 'repeats', 1)
        if sender in self.loader_threads: self.loader_threads.remove(sender)
        if not self.loader_threads:
            self.loading_label.clear()
            self.loading_label.setVisible(False)
        if not images_data:
            if self.isVisible(): QtWidgets.QMessageBox.warning(self, "No Images", f"No valid images found in {path}.")
            return
        self.datasets.append({"path": path, "images_data": images_data, "image_count": len(images_data),
                               "current_preview_idx": 0, "repeats": repeats})
        self.sort_datasets(); self.update_dataset_totals()

    def sort_datasets(self):
        active_path = None
        if 0 <= self.active_dataset_index < len(self.datasets):
            active_path = self.datasets[self.active_dataset_index]['path']
        mode = self.sort_combo.currentData()
        if mode == "name_asc":
            self.datasets.sort(key=lambda x: Path(x['path']).name.lower())
        elif mode == "name_desc":
            self.datasets.sort(key=lambda x: Path(x['path']).name.lower(), reverse=True)
        elif mode == "images_desc":
            self.datasets.sort(key=lambda x: x['image_count'], reverse=True)
        elif mode == "images_asc":
            self.datasets.sort(key=lambda x: x['image_count'])
        if active_path is not None:
            self.active_dataset_index = next(
                (i for i, ds in enumerate(self.datasets) if ds['path'] == active_path), 0
            )
        self.repopulate_dataset_grid()

    def load_datasets_from_config(self, datasets_config):
        self.datasets = []
        self.active_dataset_index = 0
        for t in self.loader_threads: t.quit()
        self.loader_threads = []
        self.repopulate_dataset_grid()
        for d in datasets_config:
            p = platform_path(d.get("path"))
            if p and os.path.exists(p): self._start_loading_path(p, d.get("repeats", 1))

    def add_dataset_folder_native(self):
        start = usable_dialog_start(self.parent_gui.last_browsed_path)
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Dataset Folder", start)
        if p:
            p = platform_path(p)
            self.parent_gui.last_browsed_path = p
            self._start_loading_path(p, 1)

    def add_dataset_folder_custom(self):
        dialog = CustomFolderDialog(self.parent_gui.last_browsed_path, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted and dialog.selected_path:
            self.parent_gui.last_browsed_path = dialog.selected_path
            self._start_loading_path(dialog.selected_path, 1)

    def _cycle_preview(self, idx, direction):
        ds = self.datasets[idx]
        ds["current_preview_idx"] = (ds["current_preview_idx"] + direction) % len(ds["images_data"])
        self._update_preview_for_card(idx)

    def _set_preview_placeholder(self, idx, text=""):
        if idx >= len(self.dataset_widgets) or idx >= len(self.datasets) or self.dataset_widgets[idx] is None: return
        ds = self.datasets[idx]
        wdg = self.dataset_widgets[idx]
        wdg["preview_label"].clear()
        wdg["caption_text"].setPlainText(text)
        wdg["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")

    def _queue_preview_refresh(self, indices=None):
        if not self._previews_active:
            return
        if indices is None:
            indices = range(len(self.dataset_widgets))
        self._preview_queue = [idx for idx in indices if idx < len(self.dataset_widgets) and self.dataset_widgets[idx] is not None]
        for idx in self._preview_queue:
            self._set_preview_placeholder(idx, "Loading preview...")
        if self._preview_queue and not self._preview_timer.isActive():
            self._preview_timer.start()

    def _load_next_queued_preview(self):
        if not self._previews_active or not self._preview_queue:
            self._preview_timer.stop()
            return
        idx = self._preview_queue.pop(0)
        if idx < len(self.dataset_widgets) and self.dataset_widgets[idx] is not None:
            self._update_preview_for_card(idx)
        if not self._preview_queue:
            self._preview_timer.stop()

    def _update_preview_for_card(self, idx):
        if idx >= len(self.dataset_widgets) or self.dataset_widgets[idx] is None: return
        ds = self.datasets[idx]
        wdg = self.dataset_widgets[idx]
        if not self._previews_active:
            wdg["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")
            return
        data = ds["images_data"][ds["current_preview_idx"]]
        if not data["caption_loaded"]:
            caption_mode = "txt"
            if "CAPTION_SOURCE_TYPE" in self.parent_gui.widgets:
                caption_mode = self.parent_gui.widgets["CAPTION_SOURCE_TYPE"].currentText()
            if caption_mode == "json":
                if data.get("json_caption_path"):
                    try:
                        with open(data["json_caption_path"], 'r', encoding='utf-8') as f:
                            payload = json.load(f)
                        parts = []
                        for key in CAPTION_JSON_TYPES:
                            value = payload.get(key, "") if isinstance(payload, dict) else ""
                            value = str(value).strip()
                            if value:
                                parts.append(f"{key}: {value}")
                        data["caption"] = "\n\n".join(parts) or "[No caption fields found]"
                    except Exception as e:
                        data["caption"] = f"[Error reading JSON caption: {e}]"
                else:
                    data["caption"] = "[No JSON caption file]"
            elif data["caption_path"]:
                try:
                    with open(data["caption_path"], 'r', encoding='utf-8') as f: data["caption"] = f.read().strip()
                except Exception: data["caption"] = "[Error reading caption]"
            else: data["caption"] = "[No caption file]"
            data["caption_loaded"] = True
        px = QtGui.QPixmap(data["image_path"])
        wdg["preview_label"].set_source_pixmap(px)
        wdg["caption_text"].setPlainText(data["caption"])
        wdg["counter_label"].setText(f"{ds['current_preview_idx'] + 1}/{len(ds['images_data'])}")

    def set_preview_active(self, active):
        active = bool(active)
        if self._previews_active == active:
            return
        self._previews_active = active
        if active:
            # Let the tab, navigator, and card layouts reach their final sizes
            # before decoding and displaying previews.
            QtCore.QTimer.singleShot(75, self._queue_preview_refresh)
        else:
            self._preview_timer.stop()
            self._preview_queue = []
            self.release_preview_resources(clear_caption_cache=True)

    def release_preview_resources(self, clear_caption_cache=False):
        for wdg in self.dataset_widgets:
            if wdg is None:
                continue
            wdg["preview_label"].clear()
            wdg["caption_text"].clear()
        if clear_caption_cache:
            for ds in self.datasets:
                for data in ds["images_data"]:
                    if data.get("caption_loaded"):
                        data["caption"] = ""
                        data["caption_loaded"] = False

    def repopulate_dataset_grid(self, selected_indices=None):
        while self.dataset_grid.count():
            item = self.dataset_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.dataset_widgets = []
        self.dataset_nav_entries = []
        self.quick_focus_list.clear()
        n = len(self.datasets)
        self.dataset_widgets = [None] * n
        if n:
            self.active_dataset_index = max(0, min(self.active_dataset_index, n - 1))
        else:
            self.active_dataset_index = 0
        selected_indices = {
            idx for idx in (selected_indices or [self.active_dataset_index]) if 0 <= idx < n
        }

        for nav_idx, nav_ds in enumerate(self.datasets):
            title = Path(nav_ds['path']).name or nav_ds['path']
            focus_item = QtWidgets.QListWidgetItem()
            focus_item.setData(QtCore.Qt.ItemDataRole.UserRole, nav_idx)
            focus_item.setToolTip(nav_ds['path'])
            focus_item.setSizeHint(QtCore.QSize(0, 76))
            self.quick_focus_list.addItem(focus_item)

            entry = QtWidgets.QWidget()
            set_role(entry, "datasetEntry")
            entry.setProperty("selected", nav_idx in selected_indices)
            entry_lay = QtWidgets.QVBoxLayout(entry)
            entry_lay.setContentsMargins(9, 7, 9, 7)
            entry_lay.setSpacing(3)
            title_lbl = make_label(title, color=ACCENT2 if nav_idx == self.active_dataset_index else TEXT_PRI, bold=True)
            title_lbl.setToolTip(nav_ds['path'])
            entry_lay.addWidget(title_lbl)
            stats_lay = QtWidgets.QHBoxLayout()
            stats_lay.setContentsMargins(0, 0, 0, 0)
            stats_lay.setSpacing(10)
            stats_lay.addWidget(make_label(f"Images: {nav_ds['image_count']}", color=TEXT_SEC, size=8))
            effective_lbl = make_label(
                f"With repeats: {nav_ds['image_count'] * nav_ds['repeats']}", color=TEXT_SEC, size=8
            )
            stats_lay.addWidget(effective_lbl)
            stats_lay.addStretch(1)
            entry_lay.addLayout(stats_lay)
            self.quick_focus_list.setItemWidget(focus_item, entry)
            self.dataset_nav_entries.append({"effective_label": effective_lbl})

        if n:
            for row in range(self.quick_focus_list.count()):
                item = self.quick_focus_list.item(row)
                item.setSelected(item.data(QtCore.Qt.ItemDataRole.UserRole) in selected_indices)
            active_item = next(
                (self.quick_focus_list.item(row) for row in range(self.quick_focus_list.count())
                 if self.quick_focus_list.item(row).data(QtCore.Qt.ItemDataRole.UserRole) == self.active_dataset_index),
                None,
            )
            if active_item is not None:
                self.quick_focus_list.setCurrentItem(
                    active_item, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate
                )
            if self.dataset_search.text().strip():
                self._search_dataset_navigator(self.dataset_search.text())

        compact = False

        if not n:
            empty_preview_group = QtWidgets.QGroupBox("Dataset Preview")
            set_role(empty_preview_group, "nested")
            set_semantic_color(empty_preview_group, "raw")
            empty_preview_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            empty_preview_layout = QtWidgets.QVBoxLayout(empty_preview_group)
            empty_preview_message = make_label(
                "No datasets detected. Add a dataset folder to preview its images and settings.",
                color=TEXT_SEC,
                size=10,
            )
            empty_preview_message.setWordWrap(True)
            empty_preview_message.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            empty_preview_layout.addStretch(1)
            empty_preview_layout.addWidget(empty_preview_message)
            empty_preview_layout.addStretch(1)
            self.dataset_grid.addWidget(empty_preview_group, 0, 0)

        active_indices = [self.active_dataset_index] if n else []
        for idx in active_indices:
            ds = self.datasets[idx]
            card = QtWidgets.QGroupBox(Path(ds['path']).name or ds['path'])
            set_role(card, "nested")
            set_semantic_color(card, "raw")
            card.setProperty("density", "compact" if compact else "normal")
            card.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            cl = QtWidgets.QVBoxLayout(card)
            cl.setContentsMargins(*(8, 10, 8, 8) if compact else (12, 14, 12, 12))
            cl.setSpacing(8 if compact else 12)

            preview_group, preview_sec = group_box("Image Preview")
            preview_group.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
            preview_sec.setContentsMargins(*(5, 7, 5, 5) if compact else (8, 10, 8, 8))
            preview_sec.setSpacing(3 if compact else 5)
            preview_lbl = ResponsivePixmapLabel()
            set_role(preview_lbl, "preview")
            preview_sec.addWidget(preview_lbl, 1)

            nav_h = QtWidgets.QHBoxLayout()
            nav_h.setSpacing(8)
            left_btn = make_btn("◄")
            left_btn.setFixedHeight(22); left_btn.setMinimumWidth(35)
            left_btn.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 0;")
            left_btn.clicked.connect(lambda _, i=idx: self._cycle_preview(i, -1))
            counter_lbl = make_label(f"1/{len(ds['images_data'])}", color=ACCENT2, bold=True)
            counter_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            right_btn = make_btn("►")
            right_btn.setFixedHeight(22); right_btn.setMinimumWidth(35)
            right_btn.setStyleSheet("font-size: 12pt; font-weight: bold; padding: 0;")
            right_btn.clicked.connect(lambda _, i=idx: self._cycle_preview(i, 1))
            nav_h.addWidget(left_btn); nav_h.addWidget(counter_lbl, 1); nav_h.addWidget(right_btn)
            preview_sec.addLayout(nav_h)
            cl.addWidget(preview_group)

            details_panel = QtWidgets.QWidget()
            set_role(details_panel, "transparent")
            details_lay = QtWidgets.QHBoxLayout(details_panel)
            details_lay.setContentsMargins(0, 0, 0, 0)
            details_lay.setSpacing(6 if compact else 10)

            cap_container = QtWidgets.QWidget()
            set_role(cap_container, "panel")
            cap_container.setMinimumWidth(130 if compact else 240)
            cap_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            cap_layout = QtWidgets.QVBoxLayout(cap_container)
            cap_layout.setContentsMargins(*(5, 5, 5, 5) if compact else (8, 8, 8, 8))
            cap_layout.addWidget(make_label("<b>Caption Preview:</b>", color=ACCENT2, size=9 if compact else 11))
            caption_text = QtWidgets.QTextEdit("Loading...")
            caption_text.setReadOnly(True)
            caption_text.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            caption_text.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
            caption_text.setMinimumHeight(70 if compact else 90)
            caption_text.setProperty("uiRole", "caption")
            cap_layout.addWidget(caption_text, 1)
            details_lay.addWidget(cap_container, 2)

            settings_group, settings_lay = group_box("Dataset Settings")
            settings_lay.setContentsMargins(*(6, 7, 6, 6) if compact else (10, 12, 10, 10))
            settings_lay.setSpacing(5 if compact else 8)

            path_short = (Path(ds['path']).name[:27] + "...") if len(Path(ds['path']).name) > 30 else Path(ds['path']).name
            path_lbl = make_label(f"<b>Folder:</b> {path_short}")
            path_lbl.setToolTip(ds['path'])
            path_lbl.setWordWrap(True)
            settings_lay.addWidget(path_lbl)
            settings_lay.addWidget(make_separator())
            settings_lay.addWidget(make_label(f"<b>Images:</b> {ds['image_count']}"))
            repeats_total_lbl = make_label(f"<b>Total (with repeats):</b> {ds['image_count'] * ds['repeats']}", color=WARN)
            settings_lay.addWidget(repeats_total_lbl)

            rep_h = QtWidgets.QHBoxLayout()
            rep_h.setContentsMargins(0, 5, 0, 0)
            rep_h.addWidget(make_label("Repeats:"))
            rep_spin = NoScrollSpinBox()
            set_role(rep_spin, "compact")
            rep_spin.setRange(1, 10000); rep_spin.setValue(ds["repeats"])
            rep_spin.valueChanged.connect(lambda v, i=idx: self.update_repeats(i, v))
            rep_h.addWidget(rep_spin, 1)
            settings_lay.addLayout(rep_h)
            settings_lay.addStretch(1)

            btn_h = QtWidgets.QHBoxLayout()
            btn_h.setSpacing(5)
            rm_btn = make_btn("Remove Dataset")
            set_role(rm_btn, "danger")
            rm_btn.clicked.connect(lambda _, i=idx: self.remove_dataset(i))
            clear_btn = make_btn("Clear Cache"); clear_btn.setStyleSheet("min-height: 24px; max-height: 24px; padding: 4px 15px;")
            clear_btn.clicked.connect(lambda _, p=ds["path"]: self.confirm_clear_cache(p))
            cache_exists = self._cache_exists(ds["path"])
            clear_btn.setEnabled(cache_exists)
            if not cache_exists: clear_btn.setToolTip("No cache found")
            btn_h.addWidget(clear_btn); btn_h.addWidget(rm_btn)
            settings_lay.addLayout(btn_h)
            details_lay.addWidget(settings_group, 1)
            cl.addWidget(details_panel, 0)
            cl.addSpacing(28)

            self.dataset_grid.addWidget(card, 0, 0)
            inherit_semantic_colors(card)
            self.dataset_widgets[idx] = {
                "card": card,
                "preview_label": preview_lbl, "caption_text": caption_text,
                "counter_label": counter_lbl, "repeats_total_label": repeats_total_lbl,
                "clear_btn": clear_btn
            }
            self._update_preview_for_card(idx)

        self.dataset_grid.setColumnStretch(0, 1)

    def _cache_exists(self, path):
        return any((Path(path) / name).is_dir() for name in self.parent_gui.get_current_cache_folder_names())

    def update_repeats(self, idx, val):
        self.datasets[idx]["repeats"] = val
        if idx < len(self.dataset_widgets) and self.dataset_widgets[idx] is not None:
            ds = self.datasets[idx]
            self.dataset_widgets[idx]["repeats_total_label"].setText(f"<b>Total (with repeats):</b> {ds['image_count'] * ds['repeats']}")
        if idx < len(self.dataset_nav_entries):
            ds = self.datasets[idx]
            self.dataset_nav_entries[idx]["effective_label"].setText(
                f"With repeats: {ds['image_count'] * ds['repeats']}"
            )
        self.update_dataset_totals()

    def remove_dataset(self, idx):
        del self.datasets[idx]
        if self.active_dataset_index > idx:
            self.active_dataset_index -= 1
        self.active_dataset_index = min(self.active_dataset_index, max(0, len(self.datasets) - 1))
        self.repopulate_dataset_grid()
        self.update_dataset_totals()

    def update_dataset_totals(self):
        self.datasetsChanged.emit()

    def confirm_clear_cache(self, path):
        names = self.parent_gui.get_current_cache_folder_names()
        existing = [name for name in names if (Path(path) / name).is_dir()]
        target_text = "', '".join(existing or names)
        if QtWidgets.QMessageBox.question(self, "Confirm", f"Delete cached latents in '{target_text}' for this dataset?",
                                          QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) == QtWidgets.QMessageBox.StandardButton.Yes:
            self.clear_cached_latents(path)

    def clear_cached_latents(self, path):
        names = self.parent_gui.get_current_cache_folder_names()
        deleted = False
        root = Path(path)
        for name in names:
            candidates = [root / name]
            candidates.extend(root.rglob(name))
            for d in candidates:
                if d.is_dir():
                    try: shutil.rmtree(d); self.parent_gui.log(f"Deleted cache: {d}"); deleted = True
                    except Exception as e: self.parent_gui.log(f"Error deleting {d}: {e}")
        if deleted: self.refresh_cache_buttons()
        else: self.parent_gui.log(f"No cache directories found matching: {', '.join(names)}.")

    def refresh_cache_buttons(self):
        for idx, ds in enumerate(self.datasets):
            if idx < len(self.dataset_widgets) and self.dataset_widgets[idx] is not None:
                exists = self._cache_exists(ds["path"])
                btn = self.dataset_widgets[idx]["clear_btn"]
                btn.setEnabled(exists)
                btn.setToolTip("" if exists else "No cache found")


UI_DEFS = {
    "SINGLE_FILE_CHECKPOINT_PATH": ("Base Model (.safetensors)", "Path to the base SDXL model.", "path", "file_safetensors"),
    "DIT_PATH":                    ("DiT Model (.safetensors)", "Path to the Anima DiT diffusion model.", "path", "file_safetensors"),
    "DIT_VAE_PATH":                ("DiT VAE (.safetensors)", "Path to the VAE used by the Anima DiT model.", "path", "file_safetensors"),
    "VAE_PATH":                    ("Separate VAE (Optional)", "Leave empty to use the VAE from the base model.", "path", "file_safetensors"),
    "TEXT_ENCODER_PATH":           ("Text Encoder (.safetensors)", "Path to the Qwen text encoder for Anima DiT.", "path", "file_safetensors"),
    "TOKENIZER_PATH":              ("Qwen Tokenizer Folder", "Local folder containing the Qwen tokenizer files for Anima.", "path", "folder"),
    "TOKENIZER_T5XXL_PATH":        ("T5XXL Tokenizer Folder", "Local folder containing the T5XXL tokenizer files for Anima.", "path", "folder"),
    "OUTPUT_DIR":                  ("Output Directory", "Folder where checkpoints will be saved.", "path", "folder"),
    "OUTPUT_NAME":                 ("Output Filename", "Base filename without .safetensors. Insert {uuid} for one six-character lowercase ID per training run.", "line"),
    "ANIMA_STREAMING_SAVE":        ("Low-RAM Anima Save", "Save Anima DiT safetensors one tensor at a time to reduce peak system RAM during checkpoint saves.", "check"),
    "CACHING_BATCH_SIZE":          ("Caching Batch Size", "Adjust based on VRAM (e.g., 2-8).", "spin", 1, 64),
    "TEXT_CACHE_PRECISION":        ("Text Cache Precision", "Floating-point dtype used for cached text embeddings on disk.", "combo", ["float32", "bfloat16", "float16"]),
    "VAE_CACHE_PRECISION":         ("VAE Cache Precision", "Floating-point dtype used for VAE latent caching. Anima VAE encoding also uses this dtype.", "combo", ["float32", "bfloat16", "float16"]),
    "NUM_WORKERS":                 ("Dataloader Workers", "Set to 0 on Windows if you have issues.", "spin", 0, 16),
    "UNCONDITIONAL_DROPOUT":       ("Use Null Conditioning Dropout", "At random, train a sample with empty-prompt conditioning instead of its caption.", "check"),
    "UNCONDITIONAL_DROPOUT_CHANCE":("Null Conditioning Chance", "Probability (0.0-1.0) of using empty-prompt conditioning for a sample.", "dspin", 0.0, 1.0, 0.05, 2),
    "QWEN_NULL_DROPOUT_CHANCE":    ("Null Qwen Chance", "Per-sample probability (0.0-1.0) of replacing Qwen prompt embeddings with empty-prompt embeddings.", "dspin", 0.0, 1.0, 0.05, 2),
    "T5_NULL_DROPOUT_CHANCE":      ("Null T5 Chance", "Per-sample probability (0.0-1.0) of replacing T5 token IDs with empty-prompt token IDs.", "dspin", 0.0, 1.0, 0.05, 2),
    "TEXT_CONDITIONING_SCALE_ENABLED": ("Use Soft Text Conditioning", "Randomly dull caption conditioning by interpolating caption embeddings toward empty-prompt embeddings.", "check"),
    "TEXT_CONDITIONING_SCALE_MIN": ("Text Conditioning Min", "Lowest caption strength to train. Values below 1 interpolate caption embeddings toward empty-prompt embeddings.", "dspin", 0.0, 1.0, 0.05, 2),
    "TEXT_CONDITIONING_SCALE_MAX": ("Text Conditioning Max", "Highest caption strength to train. Values above 1 extrapolate caption conditioning past full strength.", "dspin", 0.0, 2.0, 0.05, 2),
    "T5_TOKEN_DROPOUT_ENABLED":    ("Use T5 Token Dropout", "During training, randomly replace part of a caption's T5 tokens with pad tokens.", "check"),
    "T5_TOKEN_DROPOUT_CHANCE":     ("T5 Dropout Chance", "Per-sample probability (0.0-1.0) of applying T5 token dropout during training.", "dspin", 0.0, 1.0, 0.05, 2),
    "T5_TOKEN_DROPOUT_MIN":        ("T5 Dropout Min", "Minimum fraction (0.0-1.0) of eligible T5 tokens to replace when dropout triggers.", "dspin", 0.0, 1.0, 0.05, 2),
    "T5_TOKEN_DROPOUT_MAX":        ("T5 Dropout Max", "Maximum fraction (0.0-1.0) of eligible T5 tokens to replace when dropout triggers.", "dspin", 0.0, 1.0, 0.05, 2),
    "CAPTION_CHUNKING_ENABLED":    ("Allow Caption Chunking", "Encode full caption text in 77-token CLIP chunks and concatenate the cached embeddings.", "check"),
    "CAPTION_SOURCE_TYPE":         ("Caption Type", "Use .txt sidecars or exact-format .json sidecars.", "combo", ["txt", "json"]),
    "CAPTION_TAGS_PERCENT":        ("Tags %", "Training-time chance to load the cached tags caption variant.", "spin", 0, 100),
    "CAPTION_NL_PERCENT":          ("NL %", "Training-time chance to load the cached natural-language caption variant.", "spin", 0, 100),
    "CAPTION_TAGS_NL_PERCENT":     ("Tags+NL %", "Training-time chance to load tags followed by natural language.", "spin", 0, 100),
    "CAPTION_NL_TAGS_PERCENT":     ("NL+Tags %", "Training-time chance to load natural language followed by tags.", "spin", 0, 100),
    "MAX_BUCKET_RESOLUTION":       ("Max Bucket Size", "Largest square bucket tier. Aspect buckets are chosen from the preset ladder up to this size.", "combo", ["896", "1024", "1152", "1536"]),
    "SHOULD_UPSCALE":              ("Upscale Images", "Upscale small images closer to bucket limit.", "check"),
    "MULTI_BUCKET_ENABLED":        ("Use Multi-Bucket Cache", "Cache each image into nearby bucket resolutions so concepts are less tied to one aspect ratio.", "check"),
    "MULTI_BUCKET_EXTRA_BUCKETS":  ("Extra Buckets Per Image", "Maximum number of additional nearby buckets to cache per image. Higher values increase cache time and disk use.", "spin", 0, 8),
    "PREDICTION_TYPE":             ("Prediction Type", "v_prediction, epsilon, or rectified_flow.", "combo", ["epsilon", "v_prediction", "rectified_flow"]),
    "MAX_TRAIN_STEPS":             ("Max Training Steps", "Total number of training steps.", "line"),
    "BATCH_SIZE":                  ("Batch Size", "Number of samples per batch.", "spin", 1, 32),
    "SAVE_EVERY_N_STEPS":          ("Save Every N (Optimizer Steps)", "When to save a checkpoint.", "line"),
    "GRADIENT_ACCUMULATION_STEPS": ("Gradient Accumulation", "Simulates a larger batch size.", "line"),
    "MIXED_PRECISION":             ("Mixed Precision", "bfloat16 for modern GPUs, float16 for older.", "combo", ["bfloat16", "float16"]),
    "CLIP_GRAD_NORM":              ("Gradient Clipping", "Maximum gradient norm. 0 to disable.", "line"),
    "ANIMA_GRADIENT_CHECKPOINTING_MODE": (
        "Grad Checkpointing",
        "Full uses the least VRAM and recomputes complete blocks. Conservative uses more VRAM to cache Anima MLP down projections and reduce training step time.",
        "combo",
        ["Full", "Conservative"],
    ),
    "SEED":                        ("Seed", "Ensures reproducible training.", "line"),
    "RESUME_MODEL_PATH":           ("SDXL Resume Model", "The SDXL .safetensors checkpoint file.", "path", "file_safetensors"),
    "RESUME_STATE_PATH":           ("SDXL Resume State", "The SDXL .pt optimizer state file.", "path", "file_pt"),
    "ANIMA_RESUME_MODEL_PATH":     ("Anima Resume Model", "The Anima DiT .safetensors checkpoint file.", "path", "file_safetensors"),
    "ANIMA_RESUME_STATE_PATH":     ("Anima Resume State", "The Anima DiT .pt optimizer state file.", "path", "file_pt"),
    "UNET_EXCLUDE_TARGETS":        ("Exclude Layers (Keywords)", "Keywords for layers to exclude (comma-separated).", "textedit", 100),
    "DIT_EXCLUDE_TARGETS":         ("Exclude DiT Layers", "Keywords or fnmatch patterns for DiT layers to exclude.", "textedit", 100),
    "LR_GRAPH_MIN":                ("Graph Min LR", "Minimum learning rate displayed on the Y-axis.", "line"),
    "LR_GRAPH_MAX":                ("Graph Max LR", "Maximum learning rate displayed on the Y-axis.", "line"),
    "TIMESTEP_STRATIFIED_SAMPLING": ("Stratified Timestep Coverage", "Balance bin order locally and draw each bin's timestep values from shuffled no-repeat decks.", "check"),
    "TIMESTEP_FORCE_IMAGE_BIN_SPREAD": ("Force Image-Bin Spread", "Preplan batch-1 image order so images avoid repeating recent timestep bins while timestep sampling stays unchanged.", "check"),
    "MEMORY_EFFICIENT_ATTENTION":  ("Attention Backend", "Select the attention mechanism to use.", "combo", ["sdpa", "cudnn", "xformers (Only if no Flash)", "pytorch29_optimized"]),
    "LOSS_TYPE":                   ("Loss Type", "Select the loss function strategy.", "combo", ["MSE"]),
    "ANIMA_SEMANTIC_LOSS_ENABLED": (
        "Use Semantic Detail Loss (Experimental)",
        "Pre-cache a detail mask per image and multiply Anima's spatial flow-matching loss by 1.0x to 2.0x before applying timestep loss weighting.",
        "check",
    ),
    "VAE_NORMALIZATION_MODE":      ("VAE Normalization", "scalar uses shift/scale, flux_bn32 uses the ComfyUI Flux 32ch BN layout.", "combo", ["scalar", "flux_bn32 (Comfy Flux BN)"]),
    "VAE_SHIFT_FACTOR":            ("VAE Shift Factor", "Latent shift mean.", "dspin", -10.0, 10.0, 0.0001, 4),
    "VAE_SCALING_FACTOR":          ("VAE Scaling Factor", "Latent scaling factor.", "dspin", 0.0, 10.0, 0.0001, 5),
    "VAE_LATENT_CHANNELS":         ("Latent Channels", "4 for Standard/EQ, 32 for Flux/NoobAI.", "spin", 4, 128),

}

OPTIMIZER_INFO = {
    "raven": {
        "name": "RavenAdamW",
        "vram": "12 GB+",
        "brief": "AdamW with CPU-offloaded BF16/FP32 moments and FP32 update math.",
        "details": (
            "RavenAdamW keeps the first and second Adam moments in system RAM, then transfers one "
            "parameter's state to a reusable GPU scratch buffer for FP32 update math. It supports "
            "Raven's partial bias correction and selectable momentum precision.\n\n"
            "Best for: conservative full-parameter training where matching Raven's optimizer behavior matters.\n"
            "Tradeoff: large CPU↔GPU transfers make optimizer steps slower than paged 8-bit AdamW."
        ),
    },
    "paged_adamw_8bit": {
        "name": "PagedAdamW8bit",
        "vram": "12 GB+",
        "brief": "Standard AdamW with blockwise 8-bit paged optimizer state from bitsandbytes.",
        "details": (
            "PagedAdamW8bit keeps model parameters and gradients at their configured training precision, "
            "while storing Adam's moment history in blockwise-quantized 8-bit paged memory. Moment values "
            "are dequantized for the optimizer calculation and requantized between updates.\n\n"
            "Best for: faster long runs with substantially less optimizer-state transfer and memory.\n"
            "Tradeoff: it uses standard AdamW bias correction and does not apply Raven's Debias Strength "
            "or Momentum Precision controls."
        ),
    },
    "titan": {
        "name": "TitanAdamW",
        "vram": "6 GB+",
        "brief": "AdamW with CPU momentum and immediate post-backward gradient offload.",
        "details": (
            "TitanAdamW extends the CPU-offloaded AdamW design by moving each completed gradient to system "
            "RAM after accumulation. This reduces persistent gradient VRAM and uses reusable FP32 GPU scratch "
            "space for optimizer calculations.\n\n"
            "Best for: configurations that cannot fit with Raven.\n"
            "Tradeoff: extra gradient and momentum transfers make it the slower AdamW option."
        ),
    },
}

def optimizer_tooltip(info, width=62):
    """Build a compact themed tooltip with predictable line lengths."""
    text = f"{info.get('brief', '')}\n\n{info.get('details', '')}".strip()
    wrapped_paragraphs = []
    for paragraph in text.split("\n\n"):
        wrapped_lines = [
            textwrap.fill(line, width=width, break_long_words=False, break_on_hyphens=False)
            for line in paragraph.splitlines()
        ]
        wrapped_paragraphs.append("<br>".join(html.escape(line) for line in wrapped_lines))
    body = "<br><br>".join(wrapped_paragraphs)
    vram = html.escape(info.get("vram", ""))
    vram_line = (
        f'<br><br><span style="color:{THEME.text_muted};">Recommended GPU VRAM: </span>'
        f'<span style="color:{THEME.success}; font-weight:600;">{vram}</span>'
        if vram else ""
    )
    return f'<html><div style="color:{THEME.text};">{body}{vram_line}</div></html>'

class TrainingGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("TrainingGUI")
        self.setWindowTitle("AOZORA Trainer")
        self.setMinimumSize(1000, 800)
        self.resize(1500, 1000)
        self.config_dir = str(PROJECT_ROOT / "configs")
        self.state_file = os.path.join(self.config_dir, "gui_state.json")
        saved_state = self._read_gui_state()
        saved_colors = saved_state.get("theme_colors", {})
        self.theme_colors = {
            "primary": saved_colors.get("primary", DEFAULT_THEME_COLORS["primary"]),
            "secondary": saved_colors.get("secondary", DEFAULT_THEME_COLORS["secondary"]),
        }
        saved_chart_colors = saved_state.get("chart_colors", {})
        self.chart_colors = {
            role: saved_chart_colors.get(role, fallback)
            for role, fallback in DEFAULT_CHART_COLORS.items()
        }
        self._apply_theme_colors(
            self.theme_colors["primary"],
            self.theme_colors["secondary"],
            save=False,
        )
        self.widgets = {}
        self.process_runner = None
        self.current_config = {}
        self.current_preset = default_config.default_preset()
        self.current_mode_key = default_config.MODE_SDXL
        self._applying_config = False
        self.keyed_groups = []
        self.widget_rows = {}
        self.last_line_is_progress = False
        self.default_preset = default_config.default_preset()
        self.default_config = default_config.flatten_preset(self.default_preset, default_config.MODE_SDXL)
        self.presets = {}
        self.last_browsed_path = usable_dialog_start("")

        self._initialize_configs()
        self._setup_ui()
        self._refresh_runtime_theme_colors()
        self._mark_nested_groups()

        self.config_dropdown.blockSignals(True)
        last = saved_state.get("last_config")
        if last and last in self.presets:
            idx = self.config_dropdown.findData(last)
            if idx != -1:
                self.config_dropdown.setCurrentIndex(idx)
                self.load_selected_config(idx)
            else:
                self._load_first_config()
        else:
            self._load_first_config()
        self.config_dropdown.blockSignals(False)

    def _mark_nested_groups(self):
        """Mark group hierarchy and terminal groups for reliable depth theming."""
        for group in self.findChildren(QtWidgets.QGroupBox):
            parent = group.parentWidget()
            while parent is not None and not isinstance(parent, QtWidgets.QGroupBox):
                parent = parent.parentWidget()
            group.setProperty("nested", parent is not None)
            is_leaf = not bool(group.findChildren(QtWidgets.QGroupBox))
            group.setProperty("leaf", is_leaf)
            if group.property("uiRole") == "nested" and is_leaf:
                group.setProperty("groupSurface", "inner")
                group.style().unpolish(group)
                group.style().polish(group)
        inherit_semantic_colors(self)

    def _create_training_mode_combo(self):
        self.training_mode_combo = make_combo([TRAINING_MODE_SDXL, TRAINING_MODE_ANIMA_DIT])
        self.training_mode_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.training_mode_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        if not DIT_AVAILABLE:
            self._set_combo_item_enabled(
                self.training_mode_combo,
                TRAINING_MODE_ANIMA_DIT,
                False,
                "The bundled training_utils package is required for Anima DiT training.",
            )
        self.training_mode_combo.currentTextChanged.connect(self._on_training_mode_changed)

    def _initialize_configs(self):
        os.makedirs(self.config_dir, exist_ok=True)
        if not any(f.endswith(".json") and f != "gui_state.json" for f in os.listdir(self.config_dir)):
            with open(os.path.join(self.config_dir, "default.json"), 'w') as f:
                json.dump(self.default_preset, f, indent=4)
        self.presets = {}
        for fn in os.listdir(self.config_dir):
            if fn.endswith(".json") and fn != "gui_state.json":
                name = os.path.splitext(fn)[0]
                try:
                    with open(os.path.join(self.config_dir, fn), 'r') as f:
                        self.presets[name] = json.load(f)
                except Exception as e:
                    self.log(f"Warning: Could not load '{fn}': {e}")

    def _read_gui_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return state if isinstance(state, dict) else {}
        except Exception: pass
        return {}

    def _load_gui_state(self):
        return self._read_gui_state().get("last_config")

    def _save_gui_state(self):
        try:
            state = self._read_gui_state()
            idx = self.config_dropdown.currentIndex()
            if idx >= 0:
                key = self.config_dropdown.itemData(idx) or self.config_dropdown.itemText(idx).replace(" ", "_").lower()
                state["last_config"] = key
            state["theme_colors"] = dict(self.theme_colors)
            state["chart_colors"] = dict(self.chart_colors)
            os.makedirs(self.config_dir, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e: self.log(f"Warning: Could not save GUI state: {e}")

    @staticmethod
    def _normalized_theme_color(value, fallback):
        color = QtGui.QColor(str(value))
        return color.name() if color.isValid() else fallback

    def _apply_theme_colors(self, primary, secondary, save=True):
        global THEME, STYLESHEET, ACCENT, ACCENT2, WARN

        primary = self._normalized_theme_color(primary, DEFAULT_THEME_COLORS["primary"])
        secondary = self._normalized_theme_color(secondary, DEFAULT_THEME_COLORS["secondary"])
        primary_color = QtGui.QColor(primary)
        secondary_color = QtGui.QColor(secondary)
        THEME = replace(
            THEME,
            accent=primary,
            accent_alt=primary,
            accent_hover=primary_color.lighter(116).name(),
            accent_deep=primary_color.darker(150).name(),
            warning=secondary,
            warning_hover=secondary_color.lighter(112).name(),
            warning_deep=secondary_color.darker(150).name(),
        )
        ACCENT = THEME.accent
        ACCENT2 = THEME.accent_alt
        WARN = THEME.warning
        STYLESHEET = make_stylesheet(THEME)
        self.theme_colors = {"primary": primary, "secondary": secondary}

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(STYLESHEET)
        if hasattr(self, "theme_swatch_button"):
            self.theme_swatch_button.set_colors(primary, secondary)
        if hasattr(self, "widgets"):
            self._refresh_runtime_theme_colors()
        if save and hasattr(self, "config_dropdown"):
            self._save_gui_state()

    def _apply_chart_colors(self, colors, save=True):
        updated = dict(self.chart_colors)
        for role, fallback in DEFAULT_CHART_COLORS.items():
            if role in colors:
                updated[role] = self._normalized_theme_color(colors[role], fallback)
        self.chart_colors = updated
        if hasattr(self, "widgets"):
            self._refresh_runtime_theme_colors()
        if save and hasattr(self, "config_dropdown"):
            self._save_gui_state()

    def _apply_theme_preset(self, primary, secondary, chart_colors):
        charts = chart_colors or {
            "loss_ema": secondary,
            "sigma_samples": primary,
            "mean_loss_by_sigma": secondary,
            "learning_rate": primary,
            "gradient_norm": primary,
            "clipped_gradient": secondary,
            "lr_scheduler": primary,
            "ticket_allocator": secondary,
            "loss_weight": secondary,
        }
        self._apply_theme_colors(primary, secondary, save=False)
        self._apply_chart_colors(charts)

    def _refresh_runtime_theme_colors(self):
        for label in self.findChildren(QtWidgets.QLabel):
            role = label.property("themeColorRole")
            if not role:
                continue
            parts = [f"color: {getattr(THEME, role)};"]
            if label.property("themeBold"):
                parts.append("font-weight: bold;")
            point_size = int(label.property("themePointSize") or 0)
            if point_size:
                parts.append(f"font-size: {point_size}pt;")
            label.setStyleSheet(" ".join(parts))

        loss_ema = QtGui.QColor(self.chart_colors["loss_ema"])
        sigma_samples = QtGui.QColor(self.chart_colors["sigma_samples"])
        mean_loss = QtGui.QColor(self.chart_colors["mean_loss_by_sigma"])
        learning_rate = QtGui.QColor(self.chart_colors["learning_rate"])
        gradient_norm = QtGui.QColor(self.chart_colors["gradient_norm"])
        clipped_gradient = QtGui.QColor(self.chart_colors["clipped_gradient"])
        lr_scheduler = QtGui.QColor(self.chart_colors["lr_scheduler"])
        ticket_allocator = QtGui.QColor(self.chart_colors["ticket_allocator"])
        loss_weight = QtGui.QColor(self.chart_colors["loss_weight"])

        if hasattr(self, "lr_curve_widget"):
            self.lr_curve_widget.line_color = lr_scheduler
            self.lr_curve_widget.point_fill_color = lr_scheduler
            self.lr_curve_widget.selected_point_color = THEME.color("warning")
            self.lr_curve_widget.update()
        if hasattr(self, "timestep_histogram"):
            self.timestep_histogram.bar_color_even = ticket_allocator
            self.timestep_histogram.bar_color_odd = ticket_allocator.darker(150)
            self.timestep_histogram.bar_hover_color = ticket_allocator.lighter(112)
            self.timestep_histogram.update()
        if hasattr(self, "timestep_loss_curve"):
            self.timestep_loss_curve.line_color = loss_weight
            self.timestep_loss_curve.point_fill_color = loss_weight
            self.timestep_loss_curve.selected_point_color = THEME.color("accent_alt")
            self.timestep_loss_curve.update()
        if hasattr(self, "live_metrics_widget"):
            metrics = self.live_metrics_widget
            sample_histogram = metrics.graphs["timestep"]["widget"]
            sample_histogram.bar_color = sigma_samples
            sample_histogram.bar_color_alt = sigma_samples.darker(150)
            sample_histogram.update()
            mean_histogram = metrics.graphs["timestep_loss"]["widget"]
            mean_histogram.bar_color = mean_loss
            mean_histogram.bar_color_alt = mean_loss.darker(150)
            mean_histogram.update()
            for graph_name, line_names, color in (
                ("step_loss", ("Loss EMA",), loss_ema),
                ("optim_loss", ("Optimizer Loss EMA",), loss_ema),
                ("lr", ("LR",), learning_rate),
                ("grad_norm", ("Raw",), gradient_norm),
                ("grad_norm", ("Clipped",), clipped_gradient),
            ):
                graph_data = metrics.graphs[graph_name]
                for line_name in line_names:
                    line_index = graph_data["lines"][line_name]
                    graph_data["widget"].lines[line_index]["color"] = color
            for graph in self.live_metrics_widget.findChildren(GraphPanel):
                graph.title_color = THEME.color("accent")
                graph.update()
        self.update()

    def _show_theme_menu(self):
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)

        preset_container = QtWidgets.QWidget()
        preset_container_layout = QtWidgets.QVBoxLayout(preset_container)
        preset_container_layout.setContentsMargins(9, 7, 9, 7)
        preset_group = QtWidgets.QGroupBox("Color Presets")
        preset_layout = QtWidgets.QVBoxLayout(preset_group)
        preset_layout.setContentsMargins(8, 8, 8, 8)
        preset_layout.setSpacing(4)
        for name, primary, secondary, chart_colors in THEME_COLOR_PRESETS:
            button = ThemePresetButton(name, primary, secondary)
            preset_layout.addWidget(button)
            button.clicked.connect(
                lambda _=False, p=primary, s=secondary, c=chart_colors, m=menu: (
                    self._apply_theme_preset(p, s, c),
                    m.close(),
                )
            )
        preset_container_layout.addWidget(preset_group)
        preset_action = QtWidgets.QWidgetAction(menu)
        preset_action.setDefaultWidget(preset_container)
        menu.addAction(preset_action)

        menu.addSeparator()
        custom = QtWidgets.QWidget()
        custom_layout = QtWidgets.QVBoxLayout(custom)
        custom_layout.setContentsMargins(9, 7, 9, 9)
        custom_layout.setSpacing(8)

        interface_group = QtWidgets.QGroupBox("Interface")
        interface_row = QtWidgets.QHBoxLayout(interface_group)
        interface_row.setContentsMargins(8, 8, 8, 8)
        interface_row.setSpacing(8)
        primary_button = QtWidgets.QPushButton("Primary")
        secondary_button = QtWidgets.QPushButton("Transformed")
        for button in (primary_button, secondary_button):
            button.setMinimumSize(105, 42)
        primary_button.setStyleSheet(
            f"background:{self.theme_colors['primary']}; color:{THEME.text}; border:1px solid {THEME.border};"
        )
        secondary_button.setStyleSheet(
            f"background:{self.theme_colors['secondary']}; color:{THEME.window}; border:1px solid {THEME.border};"
        )
        primary_button.clicked.connect(lambda: self._choose_custom_theme_color("primary", primary_button, secondary_button))
        secondary_button.clicked.connect(lambda: self._choose_custom_theme_color("secondary", primary_button, secondary_button))
        interface_row.addWidget(primary_button)
        interface_row.addWidget(secondary_button)
        custom_layout.addWidget(interface_group)

        def add_chart_group(title, entries):
            group = QtWidgets.QGroupBox(title)
            row = QtWidgets.QHBoxLayout(group)
            row.setContentsMargins(8, 8, 8, 8)
            row.setSpacing(8)
            for role, label in entries:
                button = QtWidgets.QPushButton(label)
                button.setMinimumSize(105, 42)
                self._style_color_button(button, self.chart_colors[role])
                button.clicked.connect(
                    lambda _=False, r=role, b=button: self._choose_custom_chart_color(r, b)
                )
                row.addWidget(button)
            if len(entries) == 1:
                row.addStretch(1)
            custom_layout.addWidget(group)

        add_chart_group("Loss", (
            ("loss_ema", "Loss EMA"),
            ("loss_weight", "Loss Weight"),
        ))
        add_chart_group("Sigma", (
            ("sigma_samples", "Samples"),
            ("mean_loss_by_sigma", "Mean Loss"),
        ))
        add_chart_group("Learning Rate", (
            ("learning_rate", "Live"),
            ("lr_scheduler", "Scheduler"),
        ))
        add_chart_group("Gradient Norm", (
            ("gradient_norm", "Raw"),
            ("clipped_gradient", "Clipped"),
        ))
        add_chart_group("Allocation", (
            ("ticket_allocator", "Ticket Allocator"),
        ))

        custom_action = QtWidgets.QWidgetAction(menu)
        custom_action.setDefaultWidget(custom)
        menu.addAction(custom_action)

        position = self.theme_swatch_button.mapToGlobal(
            QtCore.QPoint(0, self.theme_swatch_button.height() + 3)
        )
        menu.exec(position)

    @staticmethod
    def _style_color_button(button, color):
        foreground = "#ffffff" if QtGui.QColor(color).lightness() < 145 else "#11151c"
        button.setStyleSheet(
            f"background:{color}; color:{foreground}; border:1px solid {THEME.border};"
        )

    def _choose_custom_theme_color(self, role, primary_button, secondary_button):
        current = self.theme_colors[role]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Choose {role.title()} Color")
        if not color.isValid():
            return
        updated = dict(self.theme_colors)
        updated[role] = color.name()
        self._apply_theme_colors(updated["primary"], updated["secondary"])
        primary_button.setStyleSheet(
            f"background:{updated['primary']}; color:{THEME.text}; border:1px solid {THEME.border};"
        )
        secondary_button.setStyleSheet(
            f"background:{updated['secondary']}; color:{THEME.window}; border:1px solid {THEME.border};"
        )

    def _choose_custom_chart_color(self, role, button):
        current = self.chart_colors[role]
        title = role.replace("_", " ").title()
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), self, f"Choose {title} Color")
        if not color.isValid():
            return
        self._apply_chart_colors({role: color.name()})
        self._style_color_button(button, color.name())

    def _load_first_config(self):
        if self.config_dropdown.count() > 0:
            self.config_dropdown.setCurrentIndex(0)
            self.load_selected_config(0)
        else:
            self.current_preset = copy.deepcopy(self.default_preset)
            self.current_mode_key = self.current_preset["active_mode"]
            self.current_config = default_config.flatten_preset(self.current_preset, self.current_mode_key)
            self._apply_config_to_widgets()

    def load_selected_config(self, index):
        key = self.config_dropdown.itemData(index) or self.config_dropdown.itemText(index).replace(" ", "_").lower()
        if key in self.presets:
            self.current_preset = default_config.normalize_preset(self.presets[key])
            self.current_mode_key = self.current_preset["active_mode"]
            self.current_config = default_config.flatten_preset(self.current_preset, self.current_mode_key)
            self.log(f"Loaded config: '{key}.json'")
        else:
            self.current_preset = copy.deepcopy(self.default_preset)
            self.current_mode_key = self.current_preset["active_mode"]
            self.current_config = default_config.flatten_preset(self.current_preset, self.current_mode_key)
            self.log(f"Warning: preset '{key}' not found, using defaults.")
        self._apply_config_to_widgets()
        self._save_gui_state()

    def save_config(self):
        idx = self.config_dropdown.currentIndex()
        if idx < 0: self.log("Error: No configuration selected."); return
        key = self.config_dropdown.itemData(idx) or self.config_dropdown.itemText(idx).replace(" ", "_").lower()
        cfg = self._collect_config()
        try:
            with open(os.path.join(self.config_dir, f"{key}.json"), 'w') as f: json.dump(cfg, f, indent=4)
            self.presets[key] = cfg
            self.log(f"Saved config: '{key}.json'")
        except Exception as e: self.log(f"Error saving config: {e}")

    def save_as_config(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset As", "Enter preset name (alphanumeric, underscores):")
        if not ok or not name: return
        if not re.match(r'^[a-zA-Z0-9_]+$', name): self.log("Error: Invalid preset name."); return
        save_path = os.path.join(self.config_dir, f"{name}.json")
        if os.path.exists(save_path):
            if QtWidgets.QMessageBox.question(self, "Overwrite?", f"Preset '{name}' exists. Overwrite?",
                                              QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        cfg = self._collect_config()
        try:
            with open(save_path, 'w') as f: json.dump(cfg, f, indent=4)
            self.presets[name] = cfg
            self.config_dropdown.blockSignals(True)
            self.config_dropdown.clear()
            for pn in sorted(self.presets.keys()):
                self.config_dropdown.addItem(pn.replace("_", " ").title(), pn)
            self.config_dropdown.setCurrentIndex(self.config_dropdown.findData(name))
            self.config_dropdown.blockSignals(False)
            self.log(f"Saved preset: '{name}.json'")
        except Exception as e: self.log(f"Error saving preset: {e}")

    def _make_widget(self, key):
        if key not in UI_DEFS: return None, None
        d = UI_DEFS[key]
        label_text, tooltip, wtype = d[0], d[1], d[2]
        extra = d[3:]
        lbl = QtWidgets.QLabel(label_text)
        lbl.setToolTip(tooltip)

        if wtype == "line":
            w = QtWidgets.QLineEdit()
            w.textChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "textedit":
            w = QtWidgets.QPlainTextEdit()
            if extra: w.setFixedHeight(extra[0])
            w.textChanged.connect(lambda: self._sync_widget(key))
        elif wtype == "spin":
            lo, hi = extra[0], extra[1]
            w = NoScrollSpinBox(); w.setRange(lo, hi)
            w.valueChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "dspin":
            lo, hi, step, dec = extra[0], extra[1], extra[2], extra[3]
            w = NoScrollDoubleSpinBox(); w.setRange(lo, hi); w.setSingleStep(step); w.setDecimals(dec)
            w.valueChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "combo":
            w = NoScrollComboBox(); w.addItems(extra[0])
            w.currentTextChanged.connect(lambda _, k=key: self._sync_widget(k))
        elif wtype == "check":
            w = QtWidgets.QCheckBox(label_text); w.setToolTip(tooltip)
            if key in TRANSFORMED_CHECKBOX_KEYS:
                set_semantic_color(w, "transformed")
            elif key in RAW_CHECKBOX_KEYS:
                set_semantic_color(w, "raw")
            w.stateChanged.connect(lambda _, k=key: self._sync_widget(k))
            self.widgets[key] = w
            return None, w
        elif wtype == "percent_slider":
            w = NoScrollSlider(QtCore.Qt.Orientation.Horizontal)
            w.setRange(0, 100); w.setSingleStep(1); w.setPageStep(5)
            set_semantic_color(w, "transformed")
            value_label = make_label("0%", color=ACCENT2, bold=True)
            value_label.setMinimumWidth(42)
            value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            w.valueChanged.connect(lambda v, k=key: self._sync_widget(k))
            w.valueChanged.connect(lambda v, label=value_label: label.setText(f"{v}%"))
            container = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(container); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(8)
            row.addWidget(w, 1); row.addWidget(value_label)
            w.setProperty("percentLabel", value_label)
            w.setToolTip(tooltip); container.setToolTip(tooltip)
            self.widgets[key] = w
            return lbl, container
        elif wtype == "path":
            file_type = extra[0]
            line = QtWidgets.QLineEdit()
            line.textChanged.connect(lambda _, k=key: self._sync_widget(k))
            self.widgets[key] = line
            container = QtWidgets.QWidget()
            vb = QtWidgets.QVBoxLayout(container); vb.setContentsMargins(0,0,0,0); vb.setSpacing(2)
            lb = make_label(label_text, color=ACCENT2, bold=True); lb.setToolTip(tooltip)
            vb.addWidget(lb); vb.addWidget(line)
            browse = make_btn("Browse...", lambda _, ft=file_type, le=line: self._browse_path(le, ft))
            vb.addWidget(browse)
            return None, container
        else:
            return None, None

        w.setToolTip(tooltip)
        self.widgets[key] = w
        return lbl, w

    def _add_to_form(self, form_layout, key):
        lbl, w = self._make_widget(key)
        if w:
            if lbl:
                form_layout.addRow(lbl, w)
                self.widget_rows[key] = (lbl, w)
            else:
                form_layout.addRow(w)
                self.widget_rows[key] = (w,)

    def _add_form_keys(self, form_layout, keys):
        for key in keys:
            self._add_to_form(form_layout, key)

    def _set_widget_row_visible(self, key, visible):
        row = self.widget_rows.get(key)
        if not row:
            if key in self.widgets:
                self.widgets[key].setVisible(visible)
            return
        for widget in row:
            widget.setVisible(visible)

    def _connect_widget_signal(self, key, signal_name, callback):
        if key in self.widgets:
            getattr(self.widgets[key], signal_name).connect(callback)

    def _is_dit_mode(self):
        return getattr(self, "training_mode_combo", None) and self.training_mode_combo.currentText() == TRAINING_MODE_ANIMA_DIT

    def _set_combo_item_enabled(self, combo, text, enabled, tooltip=""):
        idx = combo.findText(text)
        if idx < 0:
            return
        item = combo.model().item(idx)
        if not item:
            return
        flags = item.flags()
        if enabled:
            item.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEnabled)
        else:
            item.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEnabled)
        item.setToolTip(tooltip)

    def _update_architecture_controls(self):
        is_dit = self._is_dit_mode()
        if is_dit and not DIT_AVAILABLE:
            self.training_mode_combo.blockSignals(True)
            self.training_mode_combo.setCurrentText(TRAINING_MODE_SDXL)
            self.training_mode_combo.blockSignals(False)
            is_dit = False
            self.log("Anima DiT mode is unavailable because the bundled training_utils package could not be loaded.")

        self._update_path_mode_controls()
        if hasattr(self, "unet_exclusion_group"):
            self.unet_exclusion_group.setVisible(not is_dit)
        if hasattr(self, "dit_exclusion_group"):
            self.dit_exclusion_group.setVisible(is_dit)
        if hasattr(self, "advanced_group"):
            self.advanced_group.setVisible(not is_dit)
        self._update_loss_controls()

        self._update_mode_locked_group_states()

        if "UNCONDITIONAL_DROPOUT" in self.widgets:
            self._update_null_conditioning_dropout_controls()
        if "TEXT_CONDITIONING_SCALE_ENABLED" in self.widgets:
            self._update_text_conditioning_scale_controls()
        if "T5_TOKEN_DROPOUT_ENABLED" in self.widgets:
            self._set_widget_row_visible("T5_TOKEN_DROPOUT_ENABLED", is_dit)
            self._update_t5_token_dropout_controls()
        self._set_widget_row_visible("ANIMA_GRADIENT_CHECKPOINTING_MODE", is_dit)
        self._set_widget_row_visible("ANIMA_SEMANTIC_LOSS_ENABLED", is_dit)
        if not is_dit:
            self._update_vae_normalization_controls()
        if hasattr(self, "dataset_manager"):
            self.dataset_manager.refresh_cache_buttons()

    def _update_path_mode_controls(self):
        is_dit = self._is_dit_mode()
        is_resume = (
            hasattr(self, "model_load_strategy_combo")
            and self.model_load_strategy_combo.currentIndex() == 1
        )

        if hasattr(self, "model_load_strategy_combo"):
            self.model_load_strategy_combo.setVisible(True)
        if hasattr(self, "model_load_mode_label"):
            self.model_load_mode_label.setVisible(True)
        if hasattr(self, "path_stacked_widget"):
            self.path_stacked_widget.setCurrentIndex(1 if is_resume else 0)
            self.path_stacked_widget.setVisible(not is_dit)
            self._sync_path_stack_height()
        if hasattr(self, "dit_paths_widget"):
            self.dit_paths_widget.setVisible(is_dit)
        if hasattr(self, "tokenizer_paths_group"):
            self.tokenizer_paths_group.setVisible(is_dit)
        if hasattr(self, "model_paths_group"):
            self.model_paths_group.setTitle("Anima Model Files" if is_dit else "SDXL Model Files")
        if hasattr(self, "dit_model_path_row"):
            self.dit_model_path_row.setVisible(is_dit and not is_resume)
        if hasattr(self, "anima_resume_paths_widget"):
            self.anima_resume_paths_widget.setVisible(is_dit and is_resume)
        if "ANIMA_STREAMING_SAVE" in self.widgets:
            self.widgets["ANIMA_STREAMING_SAVE"].setVisible(is_dit)

    def _add_vertical_field(self, layout, label, widget, tooltip=None):
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        layout.addWidget(label)
        layout.addWidget(widget)

    def _add_vertical_widget_key(self, layout, key):
        lbl, widget = self._make_widget(key)
        if widget:
            if lbl:
                self._add_vertical_field(layout, lbl, widget, lbl.toolTip())
            else:
                layout.addWidget(widget)

    def _sync_widget(self, key):
        w = self.widgets.get(key)
        if not w: return
        if isinstance(w, QtWidgets.QLineEdit): self.current_config[key] = w.text().strip()
        elif isinstance(w, QtWidgets.QPlainTextEdit): self.current_config[key] = w.toPlainText().strip()
        elif isinstance(w, QtWidgets.QCheckBox): self.current_config[key] = w.isChecked()
        elif isinstance(w, QtWidgets.QComboBox): self.current_config[key] = w.currentText()
        elif isinstance(w, QtWidgets.QSlider): self.current_config[key] = w.value()
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)): self.current_config[key] = w.value()
        elif isinstance(w, LRCurveWidget): self.current_config[key] = w.get_points()
        elif isinstance(w, TimestepLossWeightCurveWidget): self.current_config[key] = w.get_points()
        elif isinstance(w, TimestepHistogramWidget): self.current_config[key] = w.get_allocation()
        if key == "VAE_NORMALIZATION_MODE":
            self._update_vae_normalization_controls()

    def _set_widget(self, key, value):
        w = self.widgets.get(key)
        if not w or value is None: return
        if UI_DEFS.get(key, (None, None, None))[2] == "path":
            value = platform_path(value)
        w.blockSignals(True)
        if isinstance(w, QtWidgets.QLineEdit): w.setText(str(value))
        elif isinstance(w, QtWidgets.QPlainTextEdit): w.setPlainText(str(value))
        elif isinstance(w, QtWidgets.QCheckBox): w.setChecked(bool(value))
        elif isinstance(w, QtWidgets.QComboBox):
            text = str(value)
            idx = w.findText(text)
            if idx < 0:
                idx = next((i for i in range(w.count()) if w.itemText(i).split()[0] == text), -1)
            if idx >= 0:
                w.setCurrentIndex(idx)
        elif isinstance(w, QtWidgets.QSlider):
            w.setValue(int(value))
            percent_label = w.property("percentLabel")
            if percent_label:
                percent_label.setText(f"{int(value)}%")
            if hasattr(self, "caption_value_labels") and key in self.caption_value_labels:
                self.caption_value_labels[key].setText(f"{int(value)}%")
        elif isinstance(w, QtWidgets.QDoubleSpinBox): w.setValue(float(value))
        elif isinstance(w, QtWidgets.QSpinBox): w.setValue(int(value))
        w.blockSignals(False)

    def _browse_path(self, entry, file_type):
        start = usable_dialog_start(entry.text(), self.last_browsed_path)
        if file_type == "folder": path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", start)
        elif file_type == "file_safetensors": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model", start, "Safetensors (*.safetensors)")
        elif file_type == "file_pt": path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select State", start, "PyTorch State (*.pt)")
        else: path = ""
        if path:
            path = platform_path(path)
            entry.setText(path)
            self.last_browsed_path = os.path.dirname(path) if os.path.isfile(path) else path

    def _form_group(self, title, keys, layout_cls=QtWidgets.QFormLayout):
        gb, _ = group_box(title, layout_cls)
        self._add_form_keys(gb.layout(), keys)
        self._register_keyed_group(gb, keys)
        return gb

    def _vertical_form_group(self, title, keys):
        gb, lay = group_box(title, QtWidgets.QVBoxLayout)
        lay.setSpacing(8)
        for key in keys:
            self._add_vertical_widget_key(lay, key)
        self._register_keyed_group(gb, keys)
        return gb

    def _register_keyed_group(self, group, keys):
        self.keyed_groups.append((group, set(keys)))

    def _update_mode_locked_group_states(self):
        hidden_keys = set(ANIMA_LOCKED_WIDGET_KEYS) if self._is_dit_mode() else set()
        if self._is_dit_mode():
            hidden_keys.add("UNET_EXCLUDE_TARGETS")
        else:
            hidden_keys.add("DIT_EXCLUDE_TARGETS")
        for group, keys in self.keyed_groups:
            if not keys:
                continue
            group.setVisible(not keys.issubset(hidden_keys))
        for key in ANIMA_LOCKED_WIDGET_KEYS:
            self._set_widget_row_visible(key, key not in hidden_keys)

    def _make_scroll_panel(self, widget, min_width=None, max_width=None):
        # Make form-based sidebar content responsive instead of allowing wide
        # label/field rows to paint beyond the viewport.
        for form in widget.findChildren(QtWidgets.QFormLayout):
            form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        responsive_fields = (
            QtWidgets.QLineEdit,
            QtWidgets.QPlainTextEdit,
            QtWidgets.QTextEdit,
            QtWidgets.QComboBox,
            QtWidgets.QSpinBox,
            QtWidgets.QDoubleSpinBox,
        )
        for field_type in responsive_fields:
            for field in widget.findChildren(field_type):
                field.setMinimumWidth(0)
                field.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Ignored,
                    field.sizePolicy().verticalPolicy(),
                )
        for label in widget.findChildren(QtWidgets.QLabel):
            label.setWordWrap(True)

        scroll = QtWidgets.QScrollArea()
        set_role(scroll, "settingsSidebar")
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        if min_width: scroll.setMinimumWidth(min_width)
        if max_width: scroll.setMaximumWidth(max_width)
        scroll.setWidget(widget)
        return scroll

    def _setup_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(5, 5, 5, 5)
        main.setSpacing(0)

        self.log_textbox = VirtualConsoleWidget(visible_lines=900, clear_callback=self.clear_console_log)
        self._create_training_mode_combo()

        # Shell layer 1: navigation owns only tabs and global configuration controls.
        self.tabs_group = QtWidgets.QWidget()
        set_role(self.tabs_group, "navigation")
        tabs_layout = QtWidgets.QHBoxLayout(self.tabs_group)
        tabs_layout.setContentsMargins(0, 0, 0, 0)
        tabs_layout.setSpacing(0)
        self.tab_bar = QtWidgets.QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        tabs_layout.addWidget(self.tab_bar, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        self.theme_swatch_button = ThemeSwatchButton(
            self.theme_colors["primary"],
            self.theme_colors["secondary"],
        )
        self.theme_swatch_button.clicked.connect(self._show_theme_menu)
        tabs_layout.addWidget(self.theme_swatch_button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        tabs_layout.addStretch(1)
        tabs_layout.addWidget(self._build_corner_widget(), 0)
        main.addWidget(self.tabs_group, 0)

        # Shell layer 2: one stable frame owns every page and all page splitters.
        self.main_frame = QtWidgets.QFrame()
        set_role(self.main_frame, "mainFrame")
        frame_layout = QtWidgets.QVBoxLayout(self.main_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)
        self.content_stack = QtWidgets.QStackedWidget()
        self.tab_view = self.content_stack  # Compatibility for callers using currentIndex().
        frame_layout.addWidget(self.content_stack, 1)

        for title_text, builder in [("Dataset", self._build_dataset_tab),
                                    ("Model && Training Parameters", self._build_model_training_tab)]:
            page = QtWidgets.QWidget()
            builder(page)
            self.tab_bar.addTab(title_text)
            self.content_stack.addWidget(page)

        self.live_metrics_widget = LiveMetricsWidget()
        self.tab_bar.addTab("Live Metrics")
        self.content_stack.addWidget(self.live_metrics_widget)

        console_widget = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_widget)
        console_layout.setContentsMargins(15, 15, 15, 15)
        console_layout.addWidget(self.log_textbox, stretch=1)
        self.tab_bar.addTab("Training Console")
        self.content_stack.addWidget(console_widget)

        self.tab_bar.currentChanged.connect(self.content_stack.setCurrentIndex)
        self.tab_bar.currentChanged.connect(self._on_tab_changed)
        self.content_stack.setCurrentIndex(self.tab_bar.currentIndex())
        main.addWidget(self.main_frame, 1)

        # Shell layer 3: footer is a sibling of navigation/content, never a page child.
        self.footer_group = self._build_bottom_bar()
        set_role(self.footer_group, "footer")
        main.addWidget(self.footer_group, 0)

    def _on_tab_changed(self, index):
        if hasattr(self, "dataset_manager"):
            self.dataset_manager.set_preview_active(index == 0)

    def _build_corner_widget(self):
        corner = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(corner)
        lay.setContentsMargins(10, 5, 10, 5)
        lay.setSpacing(8)

        lay.addWidget(make_label("Architecture:"))
        lay.addWidget(self.training_mode_combo)

        lay.addWidget(make_label("Config:"))
        self.config_dropdown = NoScrollComboBox()
        for name in sorted(self.presets.keys()):
            self.config_dropdown.addItem(name.replace("_", " ").title(), name)
        self.config_dropdown.currentIndexChanged.connect(self.load_selected_config)
        lay.addWidget(self.config_dropdown)
        lay.addWidget(make_btn("Save Config", self.save_config))
        lay.addWidget(make_btn("Save As...", self.save_as_config))
        return corner

    def _build_bottom_bar(self):
        footer = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(footer)
        lay.setContentsMargins(0, 5, 5, 5)

        calc_gb = QtWidgets.QGroupBox()
        set_role(calc_gb, "flat")
        calc_lay = QtWidgets.QHBoxLayout(calc_gb)
        calc_lay.setContentsMargins(10, 0, 10, 0)
        calc_lay.setSpacing(10)
        self.total_images_label = make_label("0", color=ACCENT2, bold=True, size=12)
        self.total_repeats_label = make_label("0", color=WARN, bold=True, size=12)
        self.epochs_label = make_label("N/A", color=WARN, bold=True, size=12)
        self.raw_image_epochs_label = make_label("N/A", color=ACCENT2, bold=True, size=12)
        self.optimizer_steps_label = make_label("N/A", color=ACCENT2, bold=True, size=12)
        for label, value in [
            ("Raw Images:", self.total_images_label),
            ("Images After Augmentation:", self.total_repeats_label),
        ]:
            calc_lay.addWidget(make_label(label))
            calc_lay.addWidget(value)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        sep.setFixedHeight(22)
        sep.setStyleSheet(f"border: 1px solid {BORDER};")
        calc_lay.addWidget(sep)
        for label, value in [
            ("Raw-Image Epochs:", self.raw_image_epochs_label),
            ("Augmented Epochs:", self.epochs_label),
            ("Optimizer Steps:", self.optimizer_steps_label),
        ]:
            calc_lay.addWidget(make_label(label))
            calc_lay.addWidget(value)
        lay.addWidget(calc_gb)
        lay.addStretch()

        self.start_button = make_btn("Start Training", self.start_training)
        self.start_button.setObjectName("StartButton")
        
        self.force_save_button = make_btn("Emergency Checkpoint", self.force_save_checkpoint)
        self.force_save_button.setObjectName("ForceSaveButton")
        self.force_save_button.setVisible(False)
        set_role(self.force_save_button, "warning")
        
        self.stop_button = make_btn("Stop Training", self.stop_training)
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setVisible(False)
        
        lay.addWidget(self.start_button)
        lay.addWidget(self.force_save_button)
        lay.addWidget(self.stop_button)
        return footer

    def _build_dataset_tab(self, parent):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        split = ThemedSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        settings_panel = QtWidgets.QWidget()
        settings_lay = QtWidgets.QVBoxLayout(settings_panel)
        settings_lay.setContentsMargins(15, 15, 10, 15)
        settings_lay.setSpacing(10)
        settings_lay.addWidget(make_label("Dataset Settings", color=ACCENT2, bold=True, size=12))
        settings_lay.addWidget(self._build_caption_source_group())
        settings_lay.addWidget(self._build_vae_group())
        for title, keys in [
            ("Batching & DataLoaders", ["CACHING_BATCH_SIZE", "TEXT_CACHE_PRECISION", "VAE_CACHE_PRECISION", "NUM_WORKERS"]),
            ("Conditioning Regularization", ["UNCONDITIONAL_DROPOUT", "UNCONDITIONAL_DROPOUT_CHANCE", "QWEN_NULL_DROPOUT_CHANCE", "T5_NULL_DROPOUT_CHANCE", "TEXT_CONDITIONING_SCALE_ENABLED", "TEXT_CONDITIONING_SCALE_MIN", "TEXT_CONDITIONING_SCALE_MAX", "T5_TOKEN_DROPOUT_ENABLED", "T5_TOKEN_DROPOUT_CHANCE", "T5_TOKEN_DROPOUT_MIN", "T5_TOKEN_DROPOUT_MAX"]),
            ("Caption Cache Options", ["CAPTION_CHUNKING_ENABLED"]),
            ("Aspect Ratio Bucketing", ["MAX_BUCKET_RESOLUTION", "SHOULD_UPSCALE", "MULTI_BUCKET_ENABLED", "MULTI_BUCKET_EXTRA_BUCKETS"]),
            ("Image Scheduling", ["TIMESTEP_FORCE_IMAGE_BIN_SPREAD"]),
        ]:
            settings_lay.addWidget(self._form_group(title, keys))
        settings_lay.addStretch(1)
        settings_scroll = self._make_scroll_panel(settings_panel, 280, 700)
        split.addWidget(settings_scroll)

        self.dataset_manager = DatasetManagerWidget(self)
        self.dataset_manager.setMinimumWidth(320)
        self.dataset_manager.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self.dataset_manager.datasetsChanged.connect(self._update_training_calculations)
        self.dataset_manager.datasetsChanged.connect(self._update_epoch_markers_on_graph)
        split.addWidget(self.dataset_manager)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 920])
        layout.addWidget(split, 1)

        if "MULTI_BUCKET_EXTRA_BUCKETS" in self.widgets:
            self._connect_widget_signal("MULTI_BUCKET_ENABLED", "stateChanged",
                                        lambda s: self.widgets["MULTI_BUCKET_EXTRA_BUCKETS"].setEnabled(bool(s)))
            self._connect_widget_signal("MULTI_BUCKET_ENABLED", "stateChanged",
                                        lambda _: self._update_training_calculations())
            self._connect_widget_signal("MULTI_BUCKET_ENABLED", "stateChanged",
                                        lambda _: self._update_epoch_markers_on_graph())
            self._connect_widget_signal("MULTI_BUCKET_EXTRA_BUCKETS", "valueChanged",
                                        lambda _: self._update_training_calculations())
            self._connect_widget_signal("MULTI_BUCKET_EXTRA_BUCKETS", "valueChanged",
                                        lambda _: self._update_epoch_markers_on_graph())
        if "UNCONDITIONAL_DROPOUT_CHANCE" in self.widgets:
            self._connect_widget_signal("UNCONDITIONAL_DROPOUT", "stateChanged",
                                        lambda s: self.widgets["UNCONDITIONAL_DROPOUT_CHANCE"].setEnabled(bool(s)))
        if "QWEN_NULL_DROPOUT_CHANCE" in self.widgets:
            self._connect_widget_signal("UNCONDITIONAL_DROPOUT", "stateChanged",
                                        lambda _: self._update_null_conditioning_dropout_controls())
            self._update_null_conditioning_dropout_controls()
        if "TEXT_CONDITIONING_SCALE_ENABLED" in self.widgets:
            self._connect_widget_signal("TEXT_CONDITIONING_SCALE_ENABLED", "stateChanged",
                                        lambda _: self._update_text_conditioning_scale_controls())
            self._update_text_conditioning_scale_controls()
        if "T5_TOKEN_DROPOUT_ENABLED" in self.widgets:
            self._connect_widget_signal("T5_TOKEN_DROPOUT_ENABLED", "stateChanged",
                                        lambda _: self._update_t5_token_dropout_controls())
            self._update_t5_token_dropout_controls()

    def _build_caption_source_group(self):
        gb, lay = group_box("Caption Source", QtWidgets.QVBoxLayout)
        lay.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(make_label("Caption Type:"))
        combo = make_combo(["txt", "json"])
        combo.setToolTip("txt uses image.txt. json uses image.json with tags, nl, tags_nl, and nl_tags keys.")
        combo.currentTextChanged.connect(lambda _, k="CAPTION_SOURCE_TYPE": self._sync_widget(k))
        combo.currentTextChanged.connect(lambda _: self._update_caption_json_controls())
        combo.currentTextChanged.connect(lambda _: self.dataset_manager._queue_preview_refresh() if hasattr(self, "dataset_manager") else None)
        self.widgets["CAPTION_SOURCE_TYPE"] = combo
        top.addWidget(combo, 1)
        help_btn = make_btn("?", self._show_json_caption_help)
        help_btn.setFixedSize(28, 28)
        help_btn.setToolTip("Show required JSON caption format")
        top.addWidget(help_btn)
        lay.addLayout(top)

        self.caption_json_rows = []
        self.caption_value_labels = {}
        for key, label in [
            ("CAPTION_TAGS_PERCENT", "Tags"),
            ("CAPTION_NL_PERCENT", "NL"),
            ("CAPTION_TAGS_NL_PERCENT", "Tags+NL"),
            ("CAPTION_NL_TAGS_PERCENT", "NL+Tags"),
        ]:
            row = self._make_caption_slider_row(key, label)
            self.caption_json_rows.append(row)
            lay.addWidget(row)

        self._register_keyed_group(
            gb,
            [
                "CAPTION_SOURCE_TYPE",
                "CAPTION_TAGS_PERCENT",
                "CAPTION_NL_PERCENT",
                "CAPTION_TAGS_NL_PERCENT",
                "CAPTION_NL_TAGS_PERCENT",
            ],
        )
        QtCore.QTimer.singleShot(0, self._update_caption_json_controls)
        return gb

    def _make_caption_slider_row(self, key, label):
        row = QtWidgets.QWidget()
        row_lay = QtWidgets.QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(8)
        row_lay.addWidget(make_label(label), 0)
        slider = NoScrollSlider(QtCore.Qt.Orientation.Horizontal)
        set_semantic_color(slider, "raw")
        slider.setRange(0, 100)
        slider.setSingleStep(1)
        slider.setPageStep(5)
        value_label = make_label("0%", color=ACCENT2, bold=True)
        value_label.setMinimumWidth(42)
        value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        slider.valueChanged.connect(lambda v, k=key: self._sync_widget(k))
        slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}%"))
        self.widgets[key] = slider
        self.caption_value_labels[key] = value_label
        row_lay.addWidget(slider, 1)
        row_lay.addWidget(value_label)
        return row

    def _update_caption_json_controls(self):
        is_json = self.widgets.get("CAPTION_SOURCE_TYPE") and self.widgets["CAPTION_SOURCE_TYPE"].currentText() == "json"
        for row in getattr(self, "caption_json_rows", []):
            row.setVisible(is_json)
        if hasattr(self, "dataset_manager"):
            self.dataset_manager.release_preview_resources(clear_caption_cache=True)

    def _show_json_caption_help(self):
        text = (
            "JSON caption mode expects one .json file next to each image with exact keys:\n\n"
            "{\n"
            "  \"tags\": \"best quality, score_7, safe, 1girl, solo, pink hair\",\n"
            "  \"nl\": \"An anime-style girl with pink hair sits in a bedroom.\",\n"
            "  \"tags_nl\": \"best quality, score_7, safe, 1girl, solo, pink hair. An anime-style girl with pink hair sits in a bedroom.\",\n"
            "  \"nl_tags\": \"An anime-style girl with pink hair sits in a bedroom. best quality, score_7, safe, 1girl, solo, pink hair\"\n"
            "}\n\n"
            "All four keys must exist and be non-empty. Extra keys are ignored. "
            "The four sliders are training-time weights; caching stores all four variants."
        )
        QtWidgets.QMessageBox.information(self, "JSON Caption Format", text)

    def _update_text_conditioning_scale_controls(self):
        enabled = (
            "TEXT_CONDITIONING_SCALE_ENABLED" in self.widgets and
            self.widgets["TEXT_CONDITIONING_SCALE_ENABLED"].isChecked()
        )
        for key in ["TEXT_CONDITIONING_SCALE_MIN", "TEXT_CONDITIONING_SCALE_MAX"]:
            if key in self.widgets:
                self._set_widget_row_visible(key, enabled)

    def _update_null_conditioning_dropout_controls(self):
        is_dit = self._is_dit_mode()
        enabled = (
            "UNCONDITIONAL_DROPOUT" in self.widgets and
            self.widgets["UNCONDITIONAL_DROPOUT"].isEnabled() and
            self.widgets["UNCONDITIONAL_DROPOUT"].isChecked()
        )
        self._set_widget_row_visible("UNCONDITIONAL_DROPOUT_CHANCE", not is_dit)
        self._set_widget_row_visible("QWEN_NULL_DROPOUT_CHANCE", is_dit)
        self._set_widget_row_visible("T5_NULL_DROPOUT_CHANCE", is_dit)
        for key in ["UNCONDITIONAL_DROPOUT_CHANCE", "QWEN_NULL_DROPOUT_CHANCE", "T5_NULL_DROPOUT_CHANCE"]:
            if key in self.widgets:
                mode_matches = (key == "UNCONDITIONAL_DROPOUT_CHANCE") != is_dit
                self._set_widget_row_visible(key, enabled and mode_matches)

    def _update_t5_token_dropout_controls(self):
        is_dit = self._is_dit_mode()
        enabled = (
            "T5_TOKEN_DROPOUT_ENABLED" in self.widgets and
            self.widgets["T5_TOKEN_DROPOUT_ENABLED"].isEnabled() and
            self.widgets["T5_TOKEN_DROPOUT_ENABLED"].isChecked()
        )
        for key in ["T5_TOKEN_DROPOUT_CHANCE", "T5_TOKEN_DROPOUT_MIN", "T5_TOKEN_DROPOUT_MAX"]:
            if key in self.widgets:
                self._set_widget_row_visible(key, enabled and is_dit)

    def _build_architecture_group(self):
        mode_gb, mode_lay = group_box("Prediction", QtWidgets.QVBoxLayout)
        mode_lay.setSpacing(4)

        self._add_vertical_widget_key(mode_lay, "PREDICTION_TYPE")
        self.widgets["PREDICTION_TYPE"].currentTextChanged.connect(self._on_prediction_type_changed)
        self._register_keyed_group(mode_gb, ["PREDICTION_TYPE"])
        return mode_gb

    def _build_model_training_tab(self, parent):
        layout = QtWidgets.QVBoxLayout(parent)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        split = ThemedSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        settings_panel = QtWidgets.QWidget()
        settings_lay = QtWidgets.QVBoxLayout(settings_panel)
        settings_lay.setContentsMargins(15, 15, 10, 15)
        settings_lay.setSpacing(10)
        settings_lay.addWidget(make_label("Model Setup", color=ACCENT2, bold=True, size=12))
        settings_lay.addWidget(self._build_architecture_group())
        settings_lay.addWidget(self._build_path_group())
        settings_lay.addWidget(self._build_training_length_group())
        settings_lay.addWidget(self._build_batching_group())
        settings_lay.addWidget(self._build_runtime_group())
        self.unet_exclusion_group = self._vertical_form_group("UNet Layer Exclusion", ["UNET_EXCLUDE_TARGETS"])
        self.dit_exclusion_group = self._vertical_form_group("DiT Layer Exclusion", ["DIT_EXCLUDE_TARGETS"])
        settings_lay.addWidget(self.unet_exclusion_group)
        settings_lay.addWidget(self.dit_exclusion_group)
        self.advanced_group = self._build_advanced_group()
        settings_lay.addWidget(self.advanced_group)
        settings_lay.addStretch(1)
        settings_scroll = self._make_scroll_panel(settings_panel, 240, 700)
        split.addWidget(settings_scroll)

        main_panel = QtWidgets.QWidget()
        main_lay = QtWidgets.QVBoxLayout(main_panel)
        main_lay.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
        main_panel.setMinimumWidth(320)
        main_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        main_lay.setContentsMargins(10, 15, 15, 15)
        main_lay.setSpacing(14)
        main_lay.addWidget(make_label("Training Parameters", color=ACCENT2, bold=True, size=12))

        parameter_columns = QtWidgets.QHBoxLayout()
        parameter_columns.setSpacing(20)

        left_column = QtWidgets.QVBoxLayout()
        left_column.setSpacing(14)
        left_column.addWidget(self._build_lr_scheduler_group())
        self.loss_group = self._build_loss_group()
        left_column.addWidget(self.loss_group)
        left_column.addStretch(1)
        parameter_columns.addLayout(left_column, 1)

        right_column = QtWidgets.QVBoxLayout()
        right_column.setSpacing(14)
        self.timestep_group = self._build_timestep_group()
        right_column.addWidget(self.timestep_group)
        right_column.addWidget(self._build_optimizer_group())
        right_column.addStretch(1)
        parameter_columns.addLayout(right_column, 1)

        main_lay.addLayout(parameter_columns)
        main_lay.addStretch(1)
        main_scroll = QtWidgets.QScrollArea()
        set_role(main_scroll, "mainContent")
        main_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_scroll.setWidgetResizable(True)
        main_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_scroll.setMinimumWidth(320)
        main_scroll.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        main_scroll.setWidget(main_panel)
        split.addWidget(main_scroll)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 1100])
        layout.addWidget(split, 1)

        for args in [
            ("MAX_TRAIN_STEPS", "textChanged", self._update_and_clamp_lr_graph),
            ("MAX_TRAIN_STEPS", "textChanged", self._update_training_calculations),
            ("GRADIENT_ACCUMULATION_STEPS", "textChanged", self._update_epoch_markers_on_graph),
            ("GRADIENT_ACCUMULATION_STEPS", "textChanged", self._update_training_calculations),
            ("BATCH_SIZE", "valueChanged", self._update_epoch_markers_on_graph),
            ("BATCH_SIZE", "valueChanged", self._update_training_calculations),
        ]:
            self._connect_widget_signal(*args)
        self._update_lr_button_states(-1)

    def _build_path_group(self):
        gb, lay = group_box("File & Directory Paths", QtWidgets.QVBoxLayout)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(3)
        self.model_load_strategy_combo = make_combo(["Load Base Model", "Resume from Checkpoint"])
        self.model_load_mode_label = make_label("Mode:")
        self._add_vertical_field(lay, self.model_load_mode_label, self.model_load_strategy_combo)

        self.model_paths_group, model_paths_lay = self._make_path_subgroup("Model Files")
        self.path_stacked_widget = QtWidgets.QStackedWidget()
        self.path_stacked_widget.setContentsMargins(0, 0, 0, 0)
        self.path_stacked_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        base_page = QtWidgets.QWidget()
        base_lay = QtWidgets.QVBoxLayout(base_page); base_lay.setContentsMargins(0, 0, 0, 0); base_lay.setSpacing(2)
        self.add_vae_path_btn = make_btn("+ Separate VAE", lambda _=False: self._set_optional_vae_visible(True))
        self.add_vae_path_btn.setFixedSize(118, 24)
        self.add_vae_path_btn.setToolTip("Add a separate VAE path")
        self.remove_vae_path_btn = make_btn("- Separate VAE", lambda _=False: self._set_optional_vae_visible(False))
        self.remove_vae_path_btn.setFixedSize(118, 24)
        self._style_optional_vae_buttons()
        self.remove_vae_path_btn.setToolTip("Hide separate VAE path")
        self.remove_vae_path_btn.setVisible(False)
        base_lay.addWidget(self._make_compact_path_row(
            "SINGLE_FILE_CHECKPOINT_PATH",
            [self.add_vae_path_btn, self.remove_vae_path_btn]
        ))
        self.optional_vae_row = self._make_compact_path_row("VAE_PATH")
        self.optional_vae_row.setVisible(False)
        self.optional_vae_row.setMaximumHeight(0)
        base_lay.addWidget(self.optional_vae_row)
        self.path_stacked_widget.addWidget(base_page)

        resume_page = QtWidgets.QWidget()
        resume_lay = QtWidgets.QVBoxLayout(resume_page); resume_lay.setContentsMargins(0, 0, 0, 0); resume_lay.setSpacing(2)
        for key in ["RESUME_MODEL_PATH", "RESUME_STATE_PATH"]:
            resume_lay.addWidget(self._make_compact_path_row(key))
        self.path_stacked_widget.addWidget(resume_page)
        model_paths_lay.addWidget(self.path_stacked_widget)

        self.dit_paths_widget = QtWidgets.QWidget()
        dit_lay = QtWidgets.QVBoxLayout(self.dit_paths_widget)
        dit_lay.setContentsMargins(0, 0, 0, 0)
        dit_lay.setSpacing(2)
        self.dit_model_path_row = self._make_compact_path_row("DIT_PATH")
        dit_lay.addWidget(self.dit_model_path_row)
        self.anima_resume_paths_widget = QtWidgets.QWidget()
        anima_resume_lay = QtWidgets.QVBoxLayout(self.anima_resume_paths_widget)
        anima_resume_lay.setContentsMargins(0, 0, 0, 0)
        anima_resume_lay.setSpacing(2)
        for key in ["ANIMA_RESUME_MODEL_PATH", "ANIMA_RESUME_STATE_PATH"]:
            anima_resume_lay.addWidget(self._make_compact_path_row(key))
        dit_lay.addWidget(self.anima_resume_paths_widget)
        dit_lay.addWidget(self._make_compact_path_row("DIT_VAE_PATH"))
        dit_lay.addWidget(self._make_compact_path_row("TEXT_ENCODER_PATH"))
        model_paths_lay.addWidget(self.dit_paths_widget)
        lay.addWidget(self.model_paths_group)

        self.tokenizer_paths_group, tokenizer_paths_lay = self._make_path_subgroup("Tokenizer Folders")
        tokenizer_help_btn = make_btn("!", self._show_anima_tokenizer_help)
        tokenizer_help_btn.setFixedSize(24, 24)
        set_role(tokenizer_help_btn, "icon")
        tokenizer_help_btn.setToolTip("Show Anima tokenizer download links")
        tokenizer_paths_lay.addWidget(self._make_compact_path_row("TOKENIZER_PATH", [tokenizer_help_btn]))
        tokenizer_paths_lay.addWidget(self._make_compact_path_row("TOKENIZER_T5XXL_PATH"))
        lay.addWidget(self.tokenizer_paths_group)

        self.output_paths_group, output_paths_lay = self._make_path_subgroup("Output")
        output_paths_lay.addWidget(self._make_compact_path_row("OUTPUT_DIR"))
        self.output_name_mode_combo = make_combo([
            "Auto (selected model + _trained_{uuid})",
            "Custom filename",
        ])
        self._add_vertical_field(
            output_paths_lay,
            make_label("Filename Mode"),
            self.output_name_mode_combo,
            "Auto names the file from the selected model and adds _trained_{uuid}.\n"
            "Custom enables the Output Filename field so you can choose the name.\n\n"
            "Place {uuid} anywhere in a custom name to generate one random "
            "six-character lowercase ID for that training run. The same ID is "
            "reused by its checkpoints and final model.\n\n"
            "Example: portrait_{uuid}_v2 -> portrait_a7k3x9_v2.safetensors",
        )
        output_name_label, output_name_edit = self._make_widget("OUTPUT_NAME")
        self._add_vertical_field(
            output_paths_lay,
            output_name_label,
            output_name_edit,
            output_name_label.toolTip(),
        )
        self._add_vertical_widget_key(output_paths_lay, "ANIMA_STREAMING_SAVE")
        lay.addWidget(self.output_paths_group)

        self.model_load_strategy_combo.currentIndexChanged.connect(lambda _: self._update_path_mode_controls())
        self.output_name_mode_combo.currentIndexChanged.connect(self._update_output_name_controls)
        for source_key in ("SINGLE_FILE_CHECKPOINT_PATH", "RESUME_MODEL_PATH", "DIT_PATH", "ANIMA_RESUME_MODEL_PATH"):
            self.widgets[source_key].textChanged.connect(lambda _: self._update_output_name_controls())
        self.training_mode_combo.currentIndexChanged.connect(lambda _: self._update_output_name_controls())
        self.model_load_strategy_combo.currentIndexChanged.connect(lambda _: self._update_output_name_controls())
        self.path_stacked_widget.currentChanged.connect(lambda _: self._sync_path_stack_height())
        self._update_path_mode_controls()
        self._update_output_name_controls()
        self._sync_path_stack_height()
        return gb

    def _suggest_output_name(self):
        is_anima = default_config.mode_key_from_label(self.training_mode_combo.currentText()) == default_config.MODE_ANIMA
        is_resume = self.model_load_strategy_combo.currentIndex() == 1
        if is_anima:
            source_key = "ANIMA_RESUME_MODEL_PATH" if is_resume else "DIT_PATH"
        else:
            source_key = "RESUME_MODEL_PATH" if is_resume else "SINGLE_FILE_CHECKPOINT_PATH"
        source = self.widgets[source_key].text().strip()
        stem = Path(source).stem if source else "model"
        return f"{stem}_trained_{{uuid}}"

    def _update_output_name_controls(self, _index=None):
        if not hasattr(self, "output_name_mode_combo") or "OUTPUT_NAME" not in self.widgets:
            return
        name_edit = self.widgets["OUTPUT_NAME"]
        automatic = self.output_name_mode_combo.currentIndex() == 0
        previous = name_edit.blockSignals(True)
        try:
            if automatic:
                name_edit.setText(self._suggest_output_name())
            elif not name_edit.text().strip() or name_edit.text().strip().lower() == "auto":
                name_edit.setText(self._suggest_output_name())
            name_edit.setEnabled(not automatic)
        finally:
            name_edit.blockSignals(previous)

    def _make_path_subgroup(self, title):
        gb = QtWidgets.QGroupBox(title)
        set_role(gb, "nested")
        set_semantic_color(gb, "raw")
        gb.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        lay = QtWidgets.QVBoxLayout(gb)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(3)
        return gb, lay

    def _show_anima_tokenizer_help(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Anima Tokenizer Files")
        dialog.setMinimumWidth(520)
        lay = QtWidgets.QVBoxLayout(dialog)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        text = QtWidgets.QLabel(
            "Anima needs two tokenizer folders in addition to the DiT, VAE, and Qwen text encoder weights.\n\n"
            "Select local folders that contain tokenizer files such as tokenizer.json, tokenizer.model, "
            "spiece.model, or vocab.json. Training will use only the selected local folders and will not "
            "download them automatically."
        )
        text.setWordWrap(True)
        lay.addWidget(text)

        links = QtWidgets.QLabel(
            '<b>Qwen tokenizer:</b><br>'
            '<a href="https://huggingface.co/Qwen/Qwen3-0.6B">https://huggingface.co/Qwen/Qwen3-0.6B</a><br><br>'
            '<b>T5XXL tokenizer:</b><br>'
            '<a href="https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/tokenizer_3">'
            'https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main/tokenizer_3</a><br><br>'
            '<b>T5XXL fallback:</b><br>'
            '<a href="https://huggingface.co/google/t5-v1_1-xxl">https://huggingface.co/google/t5-v1_1-xxl</a>'
        )
        links.setOpenExternalLinks(True)
        links.setWordWrap(True)
        links.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextBrowserInteraction |
            QtCore.Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        lay.addWidget(links)

        close_btn = make_btn("Close", dialog.accept)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_btn)
        lay.addLayout(row)
        dialog.exec()

    def _set_optional_vae_visible(self, visible):
        if hasattr(self, "optional_vae_row"):
            self.optional_vae_row.setMaximumHeight(16777215 if visible else 0)
            self.optional_vae_row.setVisible(visible)
        if hasattr(self, "add_vae_path_btn"):
            self.add_vae_path_btn.setEnabled(not visible)
            self.add_vae_path_btn.setToolTip("Separate VAE path is shown" if visible else "Add a separate VAE path")
            self._style_optional_vae_buttons()
        if hasattr(self, "remove_vae_path_btn"):
            self.remove_vae_path_btn.setVisible(visible)
        if hasattr(self, "path_stacked_widget"):
            self.path_stacked_widget.updateGeometry()
            self._sync_path_stack_height()

    def _sync_path_stack_height(self):
        if not hasattr(self, "path_stacked_widget"):
            return
        page = self.path_stacked_widget.currentWidget()
        if page:
            self.path_stacked_widget.setFixedHeight(page.sizeHint().height())

    def _style_optional_vae_buttons(self):
        if hasattr(self, "add_vae_path_btn"):
            self.add_vae_path_btn.setStyleSheet(
                "padding: 0; "
                "min-width: 118px; max-width: 118px; "
                "min-height: 24px; max-height: 24px;"
            )
        if hasattr(self, "remove_vae_path_btn"):
            self.remove_vae_path_btn.setStyleSheet(
                "padding: 0; "
                "min-width: 118px; max-width: 118px; "
                "min-height: 24px; max-height: 24px; font-weight: bold;"
            )

    def _make_compact_path_row(self, key, label_actions=None):
        label_text, tooltip, _, file_type = UI_DEFS[key]
        row = QtWidgets.QWidget()
        row.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        lay = QtWidgets.QVBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        label = make_label(label_text, color=ACCENT2, bold=True)
        label.setToolTip(tooltip)
        if label_actions:
            label_row = QtWidgets.QHBoxLayout()
            label_row.setContentsMargins(0, 0, 0, 0)
            label_row.setSpacing(4)
            label_row.addWidget(label)
            label_row.addStretch(1)
            for action in label_actions:
                label_row.addWidget(action)
            lay.addLayout(label_row)
        else:
            lay.addWidget(label)

        picker = QtWidgets.QWidget()
        picker_lay = QtWidgets.QHBoxLayout(picker)
        picker_lay.setContentsMargins(0, 0, 0, 0)
        picker_lay.setSpacing(4)

        line = QtWidgets.QLineEdit()
        line.setToolTip(tooltip)
        line.textChanged.connect(lambda _, k=key: self._sync_widget(k))
        self.widgets[key] = line
        picker_lay.addWidget(line, 1)

        browse = make_btn("+", lambda _, ft=file_type, le=line: self._browse_path(le, ft))
        browse.setFixedSize(24, 24)
        set_role(browse, "icon")
        browse.setToolTip(f"Select {label_text}")
        picker_lay.addWidget(browse)
        lay.addWidget(picker)
        return row

    def _build_training_length_group(self):
        gb, lay = group_box("Training Length", QtWidgets.QFormLayout)
        self._add_form_keys(lay, ["MAX_TRAIN_STEPS", "SAVE_EVERY_N_STEPS", "SEED"])
        return gb

    def _build_batching_group(self):
        gb, lay = group_box("Batching", QtWidgets.QFormLayout)
        self._add_form_keys(lay, ["BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS"])
        return gb

    def _build_runtime_group(self):
        gb, lay = group_box("Runtime", QtWidgets.QFormLayout)
        self._add_form_keys(lay, ["MIXED_PRECISION", "CLIP_GRAD_NORM", "ANIMA_GRADIENT_CHECKPOINTING_MODE"])
        return gb
    
    def _build_lr_scheduler_group(self):
        gb, lay = group_box("Learning Rate Scheduler")
        self.lr_curve_widget = LRCurveWidget()
        self.widgets['LR_CUSTOM_CURVE'] = self.lr_curve_widget
        self.lr_curve_widget.pointsChanged.connect(lambda pts: self._sync_widget("LR_CUSTOM_CURVE"))
        self.lr_curve_widget.selectionChanged.connect(self._update_lr_button_states)
        curve_group, curve_lay = group_box("Learning Rate Curve")
        chart_stage = QtWidgets.QWidget()
        chart_stack = QtWidgets.QGridLayout(chart_stage)
        chart_stack.setContentsMargins(0, 0, 0, 0)
        chart_stack.addWidget(self.lr_curve_widget, 0, 0)
        button_overlay = QtWidgets.QWidget()
        button_overlay.setObjectName("learningRatePointPill")
        set_role(button_overlay, "chartOverlay")
        btn_row = QtWidgets.QHBoxLayout(button_overlay)
        btn_row.setContentsMargins(3, 3, 3, 3)
        btn_row.setSpacing(2)
        self.add_point_btn = make_btn("+", self.lr_curve_widget.add_point)
        self.remove_point_btn = make_btn("−", self.lr_curve_widget.remove_selected_point)
        self.add_point_btn.setToolTip("Add a point to the learning-rate curve.")
        self.remove_point_btn.setToolTip("Remove the selected learning-rate point.")
        for button in (self.add_point_btn, self.remove_point_btn):
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            button.setMinimumSize(32, 32)
            button.setMaximumSize(32, 32)
            button.resize(32, 32)
            set_role(button, "icon")
        btn_row.addWidget(self.add_point_btn); btn_row.addWidget(self.remove_point_btn)
        chart_stack.addWidget(
            button_overlay, 0, 0,
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom,
        )
        curve_lay.addWidget(chart_stage)
        curve_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        lay.addWidget(curve_group, 0)

        curve_settings, settings_lay = group_box("Curve Settings")
        bounds_row = QtWidgets.QHBoxLayout()
        bounds_row.setContentsMargins(0, 0, 0, 0)
        bounds_row.setSpacing(10)
        for key in ["LR_GRAPH_MIN", "LR_GRAPH_MAX"]:
            col = QtWidgets.QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(3)
            self._add_vertical_widget_key(col, key)
            bounds_row.addLayout(col, 1)
        settings_lay.addLayout(bounds_row)
        curve_settings.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        lay.addWidget(curve_settings, 0)

        presets_group, preset_lay = group_box("Schedule Presets", QtWidgets.QHBoxLayout)
        preset_lay.setSpacing(8)
        for label, mode in (("Constant (Default)", "Constant"), ("Cosine", "Cosine"), ("Linear", "Linear")):
            button = make_btn(label, lambda _checked=False, name=mode: self.lr_curve_widget.set_standard_preset(name))
            if mode == "Constant":
                button.setToolTip("5% warmup, 90% constant learning rate, then 5% decay.")
            else:
                button.setToolTip(f"5% warmup followed by {mode.lower()} decay.")
            preset_lay.addWidget(button, 1)
        presets_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        lay.addWidget(presets_group, 0)
        lay.addStretch(1)
        self.widgets["LR_GRAPH_MIN"].textChanged.connect(self._update_and_clamp_lr_graph)
        self.widgets["LR_GRAPH_MAX"].textChanged.connect(self._update_and_clamp_lr_graph)
        return gb

    def _update_loss_controls(self):
        if "LOSS_TYPE" not in self.widgets:
            return
        combo = self.widgets["LOSS_TYPE"]
        allowed = ["MSE"]
        current = combo.currentText()
        desired = self.current_config.get("LOSS_TYPE", current)
        if desired not in allowed:
            desired = "MSE"

        combo.blockSignals(True)
        if [combo.itemText(i) for i in range(combo.count())] != allowed:
            combo.clear()
            combo.addItems(allowed)
        combo.setCurrentText(desired)
        combo.blockSignals(False)
        self.current_config["LOSS_TYPE"] = desired

    def _build_loss_group(self):
        gb, lay = group_box("Loss Configuration")
        settings_group, settings_lay = group_box("Loss Settings")
        loss_form = QtWidgets.QFormLayout()
        loss_form.setContentsMargins(0, 0, 0, 0)
        self._add_form_keys(loss_form, ["LOSS_TYPE", "ANIMA_SEMANTIC_LOSS_ENABLED"])
        settings_lay.addLayout(loss_form)
        settings_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        lay.addWidget(settings_group, 0)

        self.timestep_loss_curve = TimestepLossWeightCurveWidget()
        self.widgets["TIMESTEP_LOSS_WEIGHT_CURVE"] = self.timestep_loss_curve
        self.timestep_loss_curve.pointsChanged.connect(lambda _: self._sync_widget("TIMESTEP_LOSS_WEIGHT_CURVE"))
        self.timestep_loss_curve.selectionChanged.connect(self._update_timestep_loss_button_states)
        curve_group, curve_lay = group_box("Timestep Loss Weight Curve")
        chart_stage = QtWidgets.QWidget()
        chart_stack = QtWidgets.QGridLayout(chart_stage)
        chart_stack.setContentsMargins(0, 0, 0, 0)
        chart_stack.addWidget(self.timestep_loss_curve, 0, 0)
        button_overlay = QtWidgets.QWidget()
        button_overlay.setObjectName("timestepLossPointPill")
        set_role(button_overlay, "chartOverlay")
        loss_btn_row = QtWidgets.QHBoxLayout(button_overlay)
        loss_btn_row.setContentsMargins(3, 3, 3, 3)
        loss_btn_row.setSpacing(2)
        self.loss_weight_add_btn = make_btn("+", self.timestep_loss_curve.add_point)
        self.loss_weight_remove_btn = make_btn("−", self.timestep_loss_curve.remove_selected_point)
        self.loss_weight_add_btn.setToolTip("Add a point to the timestep loss-weight curve.")
        self.loss_weight_remove_btn.setToolTip("Remove the selected timestep loss-weight point.")
        for button in (self.loss_weight_add_btn, self.loss_weight_remove_btn):
            button.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
            button.setMinimumSize(32, 32)
            button.setMaximumSize(32, 32)
            button.resize(32, 32)
            set_role(button, "icon")
        self.loss_weight_remove_btn.setEnabled(False)
        loss_btn_row.addWidget(self.loss_weight_add_btn)
        loss_btn_row.addWidget(self.loss_weight_remove_btn)
        chart_stack.addWidget(
            button_overlay, 0, 0,
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignBottom,
        )
        curve_lay.addWidget(chart_stage)
        curve_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        lay.addWidget(curve_group, 1)

        presets_group, presets_lay = group_box("Loss Weight Presets")
        preset_row = QtWidgets.QGridLayout()
        preset_row.setSpacing(8)
        hysocs_flat_learning = [
            [0.0, 0.14], [0.02, 0.25], [0.03, 0.55], [0.05, 0.7],
            [0.075, 0.85], [0.1, 0.95], [0.125, 1.1222],
            [0.175, 1.3309], [0.225, 1.3926], [0.275, 1.3916],
            [0.325, 1.4054], [0.375, 1.4146], [0.425, 1.3843],
            [0.475, 1.3103], [0.525, 1.2181], [0.575, 1.1336],
            [0.625, 1.0737], [0.675, 1.0093], [0.725, 0.9109],
            [0.775, 0.8013], [0.825, 0.6956], [0.875, 0.6021],
            [0.925, 0.5174], [0.975, 0.4583], [1.0, 0.4583],
        ]
        for idx, (label, points) in enumerate([
            ("Uniform (Default)", [[0.0, 1.0], [1.0, 1.0]]),
            ("Mid Boost", [[0.0, 1.0], [0.5, 2.0], [1.0, 1.0]]),
            ("Soft Bell", None),
            ("Early Suppress", [[0.0, 0.5], [0.2, 1.0], [1.0, 1.0]]),
            ("Late Suppress", [[0.0, 1.0], [0.8, 1.0], [1.0, 0.5]]),
            ("Min-SNR-like", "min_snr_like"),
            ("Timestep Balance", hysocs_flat_learning),
        ]):
            if label == "Soft Bell":
                b = make_btn(label, lambda _=False: self.timestep_loss_curve.apply_bell_preset())
            elif points == "min_snr_like":
                b = make_btn(label, lambda _=False: self.timestep_loss_curve.apply_min_snr_like_preset())
            else:
                b = make_btn(label, lambda _, pts=points: self.timestep_loss_curve.apply_preset(pts))
            if label == "Timestep Balance":
                b.setToolTip(
                    "Apply the Hysocs 25-point equalization curve for flatter learning across all timesteps."
                )
            set_role(b, "compact")
            preset_row.addWidget(b, idx // 3, idx % 3)
        presets_lay.addLayout(preset_row)
        presets_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        lay.addWidget(presets_group, 0)
        return gb

    def _build_optimizer_group(self):
        gb, lay = group_box("Optimizer")
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(make_label("Optimizer Type:"))
        self.widgets["OPTIMIZER_TYPE"] = NoScrollComboBox()
        for optimizer_key in ("raven", "paged_adamw_8bit", "titan"):
            info = OPTIMIZER_INFO[optimizer_key]
            self.widgets["OPTIMIZER_TYPE"].addItem(info["name"], optimizer_key)
            item_index = self.widgets["OPTIMIZER_TYPE"].count() - 1
            self.widgets["OPTIMIZER_TYPE"].setItemData(
                item_index, optimizer_tooltip(info), QtCore.Qt.ItemDataRole.ToolTipRole
            )
        self.widgets["OPTIMIZER_TYPE"].currentIndexChanged.connect(self._on_optimizer_selection)
        # ``activated`` also fires when the user explicitly picks the already
        # selected row, repairing the settings page if external config loading
        # ever left the combo and stack out of sync.
        self.widgets["OPTIMIZER_TYPE"].activated.connect(self._on_optimizer_selection)
        sel_row.addWidget(self.widgets["OPTIMIZER_TYPE"], 1)
        lay.addLayout(sel_row)

        self.optimizer_settings_group, opt_lay = group_box("Optimizer Settings")
        self.optimizer_stack = QtWidgets.QStackedWidget()
        self.optimizer_stack.addWidget(self._build_adam_optimizer_form("RAVEN"))
        self.optimizer_stack.addWidget(self._build_adam_optimizer_form("PAGED_ADAMW_8BIT", core_only=True))
        self.optimizer_stack.addWidget(self._build_adam_optimizer_form("TITAN"))
        opt_lay.addWidget(self.optimizer_stack)
        lay.addWidget(self.optimizer_settings_group)
        self._on_optimizer_selection()
        return gb

    def _on_optimizer_selection(self, index=-1):
        combo = self.widgets["OPTIMIZER_TYPE"]
        if index < 0 or index >= combo.count():
            index = combo.currentIndex()
        if 0 <= index < self.optimizer_stack.count():
            self.optimizer_stack.setCurrentIndex(index)
            if hasattr(self, "current_config"):
                self.current_config["OPTIMIZER_TYPE"] = combo.itemData(index)
        self._update_optimizer_tooltip(index)

    def _update_optimizer_tooltip(self, index=-1):
        combo = self.widgets["OPTIMIZER_TYPE"]
        if index < 0 or index >= combo.count():
            index = combo.currentIndex()
        optimizer_key = combo.itemData(index) if index >= 0 else None
        info = OPTIMIZER_INFO.get(optimizer_key, {})
        combo.setToolTip(optimizer_tooltip(info))

    def _build_adam_optimizer_form(self, prefix, core_only=False):
        container = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(container); lay.setContentsMargins(0, 5, 0, 0); lay.setSpacing(8)

        fields = [
            ("betas", QtWidgets.QLineEdit()),
            ("eps", QtWidgets.QLineEdit()),
            ("weight_decay", make_dspin(0.0, 1.0, step=0.00001, decimals=6)),
        ]
        if not core_only:
            fields.append(("debias_strength", make_dspin(0.0, 1.0, step=0.01, decimals=3)))
        for name, widget in fields:
            self.widgets[f"{prefix}_{name}"] = widget
        if prefix in {"RAVEN", "TITAN"}:
            self.widgets[f'{prefix}_momentum_dtype'] = make_combo(["bfloat16", "float32"])
            self.widgets[f'{prefix}_momentum_dtype'].setCurrentText("bfloat16") 

        core_group, core_lay = group_box("Core Optimizer Settings", QtWidgets.QFormLayout)
        core_fields = [("Betas (b1, b2):", f'{prefix}_betas'), ("Epsilon:", f'{prefix}_eps'),
                       ("Weight Decay:", f'{prefix}_weight_decay')]
        if not core_only:
            core_fields.append(("Debias Strength:", f'{prefix}_debias_strength'))
        for lbl, key in core_fields:
            core_lay.addRow(lbl, self.widgets[key])
        lay.addWidget(core_group)

        if prefix in {"RAVEN", "TITAN"}:
            momentum_group, momentum_lay = group_box("Momentum Precision", QtWidgets.QFormLayout)
            momentum_lay.addRow(self.widgets[f'{prefix}_momentum_dtype'])
            lay.addWidget(momentum_group)

        lay.addStretch(1)
        return container
    
    def _toggle_optimizer_widgets(self):
        self._on_optimizer_selection()

    def _build_vae_group(self):
        gb, lay = group_box("VAE Configuration")
        self._register_keyed_group(
            gb,
            ["VAE_NORMALIZATION_MODE", "VAE_LATENT_CHANNELS", "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"],
        )
        form = QtWidgets.QFormLayout()
        self._add_form_keys(form, ["VAE_NORMALIZATION_MODE", "VAE_LATENT_CHANNELS", "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"])
        self.widgets["VAE_NORMALIZATION_MODE"].currentTextChanged.connect(lambda _: self._update_vae_normalization_controls())
        self._update_vae_normalization_controls()
        lay.addLayout(form)
        lay.addWidget(make_label("Presets:", color=ACCENT2, bold=True))
        preset_grid = QtWidgets.QGridLayout()
        preset_grid.setSpacing(8)
        for idx, (label, args) in enumerate([
            ("Standard SDXL", (0.0, 0.13025, 4, "scalar")),
            ("Flux BN32", (0.0, 1.0, 32, "flux_bn32")),
            ("Flux/NoobAI Scalar", (0.0760, 0.6043, 32, "scalar")),
            ("EQ VAE", (0.1726, 0.1280, 4, "scalar")),
        ]):
            preset_grid.addWidget(make_btn(label, lambda _, a=args: self._apply_vae_preset(*a)), idx // 2, idx % 2)
        lay.addLayout(preset_grid)
        return gb

    def _apply_vae_preset(self, shift, scale, channels, mode="scalar"):
        mode_idx = next((i for i in range(self.widgets["VAE_NORMALIZATION_MODE"].count())
                         if self.widgets["VAE_NORMALIZATION_MODE"].itemText(i).split()[0] == mode), -1)
        if mode_idx >= 0:
            self.widgets["VAE_NORMALIZATION_MODE"].setCurrentIndex(mode_idx)
        self.widgets["VAE_SHIFT_FACTOR"].setValue(shift)
        self.widgets["VAE_SCALING_FACTOR"].setValue(scale)
        self.widgets["VAE_LATENT_CHANNELS"].setValue(channels)
        self._update_vae_normalization_controls()
        self.log(f"Applied VAE Preset: Mode={mode}, Shift={shift}, Scale={scale}, Ch={channels}")

    def _update_vae_normalization_controls(self):
        if not all(key in self.widgets for key in ["VAE_NORMALIZATION_MODE", "VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"]):
            return
        uses_scalar = self.widgets["VAE_NORMALIZATION_MODE"].currentText() == "scalar"
        for key in ["VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"]:
            self.widgets[key].setEnabled(uses_scalar)
            self.widgets[key].setToolTip(
                UI_DEFS[key][1] if uses_scalar else "Ignored unless VAE normalization is scalar."
            )

    def _build_advanced_group(self):
        return self._vertical_form_group("Miscellaneous", ["MEMORY_EFFICIENT_ATTENTION"])

    def _build_timestep_group(self):
        gb, lay = group_box("Timestep Ticket Allocation")
        self.timestep_histogram = TimestepHistogramWidget()
        self.widgets["TIMESTEP_ALLOCATION"] = self.timestep_histogram
        self.timestep_histogram.allocationChanged.connect(lambda _: self._sync_widget("TIMESTEP_ALLOCATION"))
        chart_group, chart_lay = group_box("Ticket Allocation Chart")
        chart_lay.addWidget(self.timestep_histogram)
        lay.addWidget(chart_group)

        distribution_group, distribution_lay = group_box("Distribution Settings")
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(make_label("Bin Size:"))
        self.bin_size_combo = make_combo(["20", "30", "40", "50", "100", "200", "250", "500"])
        self.bin_size_combo.setCurrentText("100")
        r1.addWidget(self.bin_size_combo)
        r1.addSpacing(20)
        r1.addWidget(make_label("Distribution Mode:"))
        self.ts_mode_combo = make_combo([
            "Wave",
            "Logit-Normal",
            "Beta",
            "Odds-Scaled (Z-Image)",
        ])
        r1.addWidget(self.ts_mode_combo, 1)
        distribution_lay.addLayout(r1)

        self.ts_slider_stack = QtWidgets.QStackedWidget()

        wave_page = QtWidgets.QWidget()
        wl = QtWidgets.QFormLayout(wave_page); wl.setContentsMargins(0, 0, 0, 0)
        self.slider_wave_freq, self.spin_wave_freq = self._add_slider_row(wl, "Frequency:", 0.0, 5.0, 1.0, 100.0)
        self.slider_wave_phase, self.spin_wave_phase = self._add_slider_row(wl, "Phase:", 0.0, 6.28, 0.0, 100.0)
        self.slider_wave_amp, self.spin_wave_amp = self._add_slider_row(wl, "Amplitude:", 0.0, 2.0, 0.0, 100.0)
        self.ts_slider_stack.addWidget(wave_page)

        logit_page = QtWidgets.QWidget()
        ll = QtWidgets.QFormLayout(logit_page); ll.setContentsMargins(0, 0, 0, 0)
        self.slider_ln_mu, self.spin_ln_mu = self._add_slider_row(ll, "Center (Mu):", -3.0, 3.0, 0.0, 100.0)
        self.slider_ln_sigma, self.spin_ln_sigma = self._add_slider_row(ll, "Spread (Sigma):", 0.1, 3.0, 1.0, 100.0)
        self.ts_slider_stack.addWidget(logit_page)

        beta_page = QtWidgets.QWidget()
        bl = QtWidgets.QFormLayout(beta_page); bl.setContentsMargins(0, 0, 0, 0)
        self.slider_beta_alpha, self.spin_beta_alpha = self._add_slider_row(bl, "Alpha:", 0.1, 10.0, 3.0, 100.0)
        self.slider_beta_beta, self.spin_beta_beta = self._add_slider_row(bl, "Beta:", 0.1, 10.0, 3.0, 100.0)
        self.ts_slider_stack.addWidget(beta_page)

        odds_page = QtWidgets.QWidget()
        odds_layout = QtWidgets.QFormLayout(odds_page)
        odds_layout.setContentsMargins(0, 0, 0, 0)
        odds_control = QtWidgets.QWidget()
        odds_control_layout = QtWidgets.QHBoxLayout(odds_control)
        odds_control_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_odds_scale = NoScrollSlider(QtCore.Qt.Orientation.Horizontal)
        set_semantic_color(self.slider_odds_scale, "transformed")
        self.slider_odds_scale.setRange(-1000, 1000)
        self.slider_odds_scale.setValue(300)
        self.spin_odds_scale = NoScrollDoubleSpinBox()
        self.spin_odds_scale.setRange(-10.0, 10.0)
        self.spin_odds_scale.setSingleStep(0.1)
        self.spin_odds_scale.setDecimals(2)
        self.spin_odds_scale.setValue(3.0)
        self.spin_odds_scale.setFixedWidth(120)
        self.slider_odds_scale.valueChanged.connect(
            lambda value: self.spin_odds_scale.setValue(value / 100.0)
        )
        self.spin_odds_scale.valueChanged.connect(
            lambda value: self.slider_odds_scale.setValue(round(value * 100))
        )
        self.slider_odds_scale.valueChanged.connect(lambda _: self._update_timestep_distribution())
        self.spin_odds_scale.valueChanged.connect(lambda _: self._update_timestep_distribution())
        odds_control_layout.addWidget(self.slider_odds_scale)
        odds_control_layout.addWidget(self.spin_odds_scale)
        odds_layout.addRow("Odds Scale:", odds_control)
        odds_note = make_label(
            "Directional Z-Image-style odds scaling: positive biases right, negative biases left; "
            "values from -1 to 1 are uniform."
        )
        odds_note.setWordWrap(True)
        odds_layout.addRow(odds_note)
        self.ts_slider_stack.addWidget(odds_page)
        distribution_lay.addWidget(self.ts_slider_stack)
        lay.addWidget(distribution_group)

        presets_group, presets_lay = group_box("Distribution Presets")
        self.timestep_presets_group = presets_group
        self.ts_button_stack = QtWidgets.QStackedWidget()
        presets_by_mode = [
            [("Uniform (Default)", "Uniform"), ("Peak Ends", "Peak Ends"), ("Peak Middle", "Peak Middle")],
            [("Bell Curve", "Bell Curve"), ("Detail (Early)", "Detail"), ("Structure (Late)", "Structure"),
             ("Logit-Normal (RF/SD3 Recommended)", "Logit-Normal (RF/SD3 Recommended)"),
             ("Anima Default (1.0)", "Anima Logit Default"),
             ("Anima Style LoRA (1.3)", "Anima Logit Style LoRA")],
            [("Symmetric (3,3)", "Beta Symmetric"), ("Right Skew (2,5)", "Beta Right Skew"),
             ("Left Skew (5,2)", "Beta Left Skew"), ("U-Shape (0.5,0.5)", "Beta U-Shape")],
        ]
        for page_presets in presets_by_mode:
            page = QtWidgets.QWidget()
            pgrid = QtWidgets.QGridLayout(page); pgrid.setSpacing(8); pgrid.setContentsMargins(0, 0, 0, 0)
            for i, (label, preset_name) in enumerate(page_presets):
                b = make_btn(label, lambda _, pn=preset_name: self._apply_timestep_preset(pn))
                set_role(b, "compact")
                pgrid.addWidget(b, i // 3, i % 3)
            self.ts_button_stack.addWidget(page)
        self.ts_button_stack.addWidget(QtWidgets.QWidget())
        presets_lay.addWidget(self.ts_button_stack)
        lay.addWidget(presets_group)

        self.bin_size_combo.currentTextChanged.connect(lambda t: (self.timestep_histogram.set_bin_size(int(t)), self._update_timestep_distribution()))
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_slider_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(self.ts_button_stack.setCurrentIndex)
        self.ts_mode_combo.currentIndexChanged.connect(lambda _: self._update_timestep_distribution())
        self.ts_mode_combo.currentTextChanged.connect(
            lambda mode: self.timestep_presets_group.setVisible(mode != "Odds-Scaled (Z-Image)")
        )
        self.timestep_coverage_controls = QtWidgets.QGroupBox("Timestep Coverage")
        set_semantic_color(self.timestep_coverage_controls, "transformed")
        coverage_form = QtWidgets.QFormLayout(self.timestep_coverage_controls)
        coverage_form.setContentsMargins(8, 6, 8, 6)
        self._add_form_keys(coverage_form, ["TIMESTEP_STRATIFIED_SAMPLING"])
        self.widgets["TIMESTEP_STRATIFIED_SAMPLING"].setText("Stratified Coverage")
        lay.addWidget(self.timestep_coverage_controls)
        return gb

    def _update_timestep_loss_button_states(self, idx):
        if hasattr(self, "loss_weight_remove_btn"):
            self.loss_weight_remove_btn.setEnabled(idx > 0 and idx < len(self.timestep_loss_curve._points) - 1)

    def _add_slider_row(self, layout, label_text, min_val, max_val, default_val, divider):
        container = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(container); h.setContentsMargins(0, 0, 0, 0)
        lbl = make_label(label_text); lbl.setFixedWidth(90)
        h.addWidget(lbl)
        sl = NoScrollSlider(QtCore.Qt.Orientation.Horizontal)
        set_semantic_color(sl, "transformed")
        sl.setRange(int(min_val * divider), int(max_val * divider))
        sl.setValue(int(default_val * divider))
        sp = NoScrollDoubleSpinBox(); sp.setRange(min_val, max_val); sp.setSingleStep(0.1)
        sp.setValue(default_val); sp.setDecimals(2); sp.setFixedWidth(120)
        sl.valueChanged.connect(lambda v: sp.setValue(v / divider) if abs(sp.value() - v / divider) > 0.5 / divider else None)
        sp.valueChanged.connect(lambda v: (sl.setValue(int(v * divider)), self._update_timestep_distribution()))
        sl.valueChanged.connect(lambda _: self._update_timestep_distribution())
        h.addWidget(sl); h.addWidget(sp)
        layout.addRow(container)
        return sl, sp

    def _apply_timestep_preset(self, name):
        self.ts_mode_combo.blockSignals(True)

        mode_map = {
            "Uniform":         "Wave",
            "Peak Ends":       "Wave",
            "Peak Middle":     "Wave",
            "Bell Curve":      "Logit-Normal",
            "Detail":          "Logit-Normal",
            "Structure":       "Logit-Normal",
            "Logit-Normal (RF/SD3 Recommended)": "Logit-Normal",
            "Anima Logit Default": "Logit-Normal",
            "Anima Logit Style LoRA": "Logit-Normal",
            "Beta Symmetric":  "Beta",
            "Beta Right Skew": "Beta",
            "Beta Left Skew":  "Beta",
            "Beta U-Shape":    "Beta",
        }
        values = {
            "Uniform":         dict(wave_amp=0.0, wave_freq=1.0, wave_phase=0.0),
            "Peak Ends":       dict(wave_freq=1.0, wave_phase=0.0, wave_amp=0.8),
            "Peak Middle":     dict(wave_freq=1.0, wave_phase=3.14, wave_amp=0.6),
            "Bell Curve":      dict(ln_mu=0.0, ln_sigma=1.0),
            "Detail":          dict(ln_mu=-1.0, ln_sigma=0.8),
            "Structure":       dict(ln_mu=1.0, ln_sigma=0.8),
            "Logit-Normal (RF/SD3 Recommended)": dict(ln_mu=-0.5, ln_sigma=1.0),
            "Anima Logit Default": dict(ln_mu=0.0, ln_sigma=1.0),
            "Anima Logit Style LoRA": dict(ln_mu=0.0, ln_sigma=1.3),
            "Beta Symmetric":  dict(beta_alpha=3.0, beta_beta=3.0),
            "Beta Right Skew": dict(beta_alpha=2.0, beta_beta=5.0),
            "Beta Left Skew":  dict(beta_alpha=5.0, beta_beta=2.0),
            "Beta U-Shape":    dict(beta_alpha=0.5, beta_beta=0.5),
        }
        if name not in mode_map:
            self.ts_mode_combo.blockSignals(False)
            return

        target_mode = mode_map[name]
        vals = values[name]

        mode_idx = self.ts_mode_combo.findText(target_mode)
        if mode_idx < 0:
            mode_idx = 0
        self.ts_mode_combo.setCurrentIndex(mode_idx)
        self.ts_slider_stack.setCurrentIndex(mode_idx)
        self.ts_button_stack.setCurrentIndex(mode_idx)
        self.ts_mode_combo.blockSignals(False)

        slider_spin_pairs = [
            (self.slider_wave_freq, self.spin_wave_freq, "wave_freq", 100.0),
            (self.slider_wave_phase, self.spin_wave_phase, "wave_phase", 100.0),
            (self.slider_wave_amp, self.spin_wave_amp, "wave_amp", 100.0),
            (self.slider_ln_mu, self.spin_ln_mu, "ln_mu", 100.0),
            (self.slider_ln_sigma, self.spin_ln_sigma, "ln_sigma", 100.0),
            (self.slider_beta_alpha, self.spin_beta_alpha, "beta_alpha", 100.0),
            (self.slider_beta_beta, self.spin_beta_beta, "beta_beta", 100.0),
        ]
        for sl, sp, key, divider in slider_spin_pairs:
            sl.blockSignals(True); sp.blockSignals(True)
            if key in vals:
                sp.setValue(vals[key])
                sl.setValue(int(vals[key] * divider))
            sl.blockSignals(False); sp.blockSignals(False)

        self._update_timestep_distribution()

    def _update_timestep_distribution(self):
        mode = self.ts_mode_combo.currentText()
        n = max(math.ceil(1000 / int(self.bin_size_combo.currentText())), 1)
        weights = []

        if mode == "Wave":
            freq, phase, amp = self.spin_wave_freq.value(), self.spin_wave_phase.value(), self.spin_wave_amp.value()
            for i in range(n):
                t = i / max(1, n - 1)
                weights.append(max(0.0, 1.0 + amp * math.cos(2 * math.pi * freq * t + phase)))
        elif mode == "Logit-Normal":
            mu, sigma = self.spin_ln_mu.value(), self.spin_ln_sigma.value()
            bin_size = int(self.bin_size_combo.currentText())
            def logit(p): return math.log(p / (1 - p))
            def ncdf(x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
            for i in range(n):
                t_s, t_e = i * bin_size, min((i + 1) * bin_size, 1000)
                eps = 1e-6
                w = ncdf((logit(min(t_e / 1000, 1 - eps)) - mu) / sigma) - ncdf((logit(max(t_s / 1000, eps)) - mu) / sigma)
                weights.append(max(0.0, w))
        elif mode == "Beta":
            alpha, beta = self.spin_beta_alpha.value(), self.spin_beta_beta.value()
            bin_size = int(self.bin_size_combo.currentText())
            for i in range(n):
                x = max(1e-4, min(1 - 1e-4, ((i * bin_size) + bin_size / 2) / 1000))
                weights.append(max(0.0, x ** (alpha - 1) * (1 - x) ** (beta - 1)))
        elif mode == "Odds-Scaled (Z-Image)":
            weights = gui_math.odds_scaled_ticket_weights(
                int(self.bin_size_combo.currentText()),
                self.spin_odds_scale.value(),
                1000,
            )
        self.timestep_histogram.generate_from_weights(weights)

    def _on_prediction_type_changed(self, text):
        if hasattr(self, 'dataset_manager'): 
            self.dataset_manager.refresh_cache_buttons()

    def _on_training_mode_changed(self, text):
        if self._applying_config:
            self._update_architecture_controls()
            return
        old_mode = self.current_mode_key
        new_mode = default_config.mode_key_from_label(text)
        if old_mode != new_mode:
            self._store_current_mode_config(old_mode)
            self.current_mode_key = new_mode
            self.current_preset["active_mode"] = new_mode
            self.current_config = default_config.flatten_preset(self.current_preset, new_mode)
            self._apply_config_to_widgets()
            return
        self._update_architecture_controls()
        if hasattr(self, 'dataset_manager'): 
            self.dataset_manager.refresh_cache_buttons()

    def _store_current_mode_config(self, mode_key=None):
        mode_key = mode_key or self.current_mode_key
        flat_cfg = self._collect_flat_config(mode_key)
        self.current_preset = default_config.nest_flat_config(flat_cfg, mode_key, self.current_preset)
        self.current_preset["active_mode"] = mode_key

    def _apply_config_to_widgets(self):
        self._applying_config = True
        for w in self.widgets.values(): w.blockSignals(True)
        try:
            mode = self.current_config.get("TRAINING_MODE", "SDXL")
            if mode == "SDXL":
                mode = TRAINING_MODE_SDXL
            if mode == TRAINING_MODE_ANIMA_DIT and not DIT_AVAILABLE:
                mode = TRAINING_MODE_SDXL
                self.current_config = default_config.flatten_preset(self.current_preset, default_config.MODE_SDXL)
            self.current_mode_key = default_config.mode_key_from_label(mode)
            self.current_preset["active_mode"] = self.current_mode_key
            self.training_mode_combo.blockSignals(True)
            self.training_mode_combo.setCurrentText(mode)
            self.training_mode_combo.blockSignals(False)

            is_resuming = self.current_config.get("RESUME_TRAINING", False)
            self.model_load_strategy_combo.setCurrentIndex(1 if is_resuming else 0)

            skip = {"OPTIMIZER_TYPE", "LR_CUSTOM_CURVE", "LOSS_TYPE", "TIMESTEP_ALLOCATION", "TIMESTEP_LOSS_WEIGHT_CURVE"}
            skip |= {k for k in self.widgets if k.startswith(("RAVEN_", "PAGED_ADAMW_8BIT_", "TITAN_"))}
            for key, w in self.widgets.items():
                if key in skip: continue
                self._set_widget(key, self.current_config.get(key))

            configured_output_name = str(self.current_config.get("OUTPUT_NAME", "auto") or "auto").strip()
            self.output_name_mode_combo.blockSignals(True)
            self.output_name_mode_combo.setCurrentIndex(0 if configured_output_name.lower() == "auto" else 1)
            self.output_name_mode_combo.blockSignals(False)
            self._update_output_name_controls()

            opt_type = self.current_config.get("OPTIMIZER_TYPE", default_config.OPTIMIZER_TYPE).lower()
            idx = self.widgets["OPTIMIZER_TYPE"].findData(opt_type)
            self.widgets["OPTIMIZER_TYPE"].setCurrentIndex(idx if idx >= 0 else 0)

            for prefix, def_attr in [("RAVEN", "RAVEN_PARAMS"), ("TITAN", "TITAN_PARAMS")]:
                defaults = getattr(default_config, def_attr, {})
                params = {**defaults, **self.current_config.get(def_attr, {})}
                self.widgets[f"{prefix}_betas"].setText(', '.join(map(str, params.get("betas", [0.9, 0.999]))))
                self.widgets[f"{prefix}_eps"].setText(str(params.get("eps", 1e-8)))
                self.widgets[f"{prefix}_weight_decay"].setValue(params.get("weight_decay", 0.01))
                self.widgets[f"{prefix}_debias_strength"].setValue(params.get("debias_strength", 1.0))
                if "momentum_dtype" in params:
                    self.widgets[f"{prefix}_momentum_dtype"].setCurrentText(params["momentum_dtype"])

            paged_defaults = getattr(default_config, "PAGED_ADAMW_8BIT_PARAMS", {})
            paged_params = {
                **paged_defaults,
                **self.current_config.get("PAGED_ADAMW_8BIT_PARAMS", {}),
            }
            self.widgets["PAGED_ADAMW_8BIT_betas"].setText(
                ', '.join(map(str, paged_params.get("betas", [0.9, 0.999])))
            )
            self.widgets["PAGED_ADAMW_8BIT_eps"].setText(str(paged_params.get("eps", 1e-8)))
            self.widgets["PAGED_ADAMW_8BIT_weight_decay"].setValue(
                paged_params.get("weight_decay", 0.01)
            )

            self._toggle_optimizer_widgets()

            loss_type = self.current_config.get("LOSS_TYPE", "MSE")
            self.widgets["LOSS_TYPE"].setCurrentText("MSE" if loss_type not in {"MSE"} else loss_type)

            if "MULTI_BUCKET_ENABLED" in self.widgets:
                self.widgets["MULTI_BUCKET_EXTRA_BUCKETS"].setEnabled(self.widgets["MULTI_BUCKET_ENABLED"].isChecked())
            if "UNCONDITIONAL_DROPOUT" in self.widgets:
                self._update_null_conditioning_dropout_controls()
            if "TEXT_CONDITIONING_SCALE_ENABLED" in self.widgets:
                self._update_text_conditioning_scale_controls()
            if "T5_TOKEN_DROPOUT_ENABLED" in self.widgets:
                self._update_t5_token_dropout_controls()
            if "CAPTION_SOURCE_TYPE" in self.widgets:
                self._update_caption_json_controls()
            if "VAE_PATH" in self.widgets:
                self._set_optional_vae_visible(bool(self.widgets["VAE_PATH"].text().strip()))
            self._update_vae_normalization_controls()

            self._update_and_clamp_lr_graph()

            alloc = self.current_config.get("TIMESTEP_ALLOCATION")
            if alloc:
                if "bin_size" in alloc:
                    self.bin_size_combo.blockSignals(True)
                    self.bin_size_combo.setCurrentText(str(alloc["bin_size"]))
                    self.bin_size_combo.blockSignals(False)
                self.timestep_histogram.set_allocation(alloc)
            loss_curve = self.current_config.get("TIMESTEP_LOSS_WEIGHT_CURVE")
            if hasattr(self, "timestep_loss_curve"):
                self.timestep_loss_curve.set_points(loss_curve or [[0.0, 1.0], [1.0, 1.0]])
            ts_mode = self.current_config.get("TIMESTEP_MODE", "Wave")
            if ts_mode == "Shift":
                ts_mode = "Odds-Scaled (Z-Image)"
            self.ts_mode_combo.blockSignals(True)
            self.ts_mode_combo.setCurrentText(ts_mode)
            self.ts_mode_combo.blockSignals(False)
            mode_idx = max(0, self.ts_mode_combo.currentIndex())
            self.ts_slider_stack.setCurrentIndex(mode_idx)
            self.ts_button_stack.setCurrentIndex(mode_idx)
            self.timestep_presets_group.setVisible(ts_mode != "Odds-Scaled (Z-Image)")
            self.spin_odds_scale.blockSignals(True)
            self.slider_odds_scale.blockSignals(True)
            odds_scale = float(self.current_config.get("TIMESTEP_ODDS_SCALE", 3.0))
            self.spin_odds_scale.setValue(odds_scale)
            self.slider_odds_scale.setValue(round(odds_scale * 100))
            self.slider_odds_scale.blockSignals(False)
            self.spin_odds_scale.blockSignals(False)

            if hasattr(self, 'dataset_manager'):
                self.dataset_manager.load_datasets_from_config(self.current_config.get("INSTANCE_DATASETS", []))
            self._update_training_calculations()
            self._update_architecture_controls()

            if "PREDICTION_TYPE" in self.widgets:
                self._on_prediction_type_changed(self.widgets["PREDICTION_TYPE"].currentText())

        finally:
            for w in self.widgets.values(): w.blockSignals(False)
            self._applying_config = False
            self._update_optimizer_tooltip()

    def _collect_flat_config(self, mode_key=None):
        cfg = {}
        skip_keys = {"RESUME_TRAINING", "INSTANCE_DATASETS", "OPTIMIZER_TYPE",
                     "RAVEN_PARAMS", "PAGED_ADAMW_8BIT_PARAMS", "TITAN_PARAMS",
                     "LOSS_TYPE", "TIMESTEP_ALLOCATION", "TIMESTEP_LOSS_WEIGHT_CURVE"}
        mode_key = mode_key or default_config.mode_key_from_label(self.training_mode_combo.currentText())
        cfg["TRAINING_MODE"] = default_config.MODE_LABELS[mode_key]
        is_dit = mode_key == default_config.MODE_ANIMA
        inactive_path_keys = SDXL_PATH_KEYS if is_dit else ANIMA_PATH_KEYS

        for key in [*UI_DEFS.keys(), "LR_CUSTOM_CURVE"]:
            val = self.current_config.get(key)
            if key in skip_keys: continue
            if key in inactive_path_keys: continue
            if val is None: continue
            cfg[key] = [[round(p[0], 8), round(p[1], 10)] for p in val] if key == "LR_CUSTOM_CURVE" else val

        cfg["OUTPUT_NAME"] = (
            "auto"
            if self.output_name_mode_combo.currentIndex() == 0
            else self.widgets["OUTPUT_NAME"].text().strip() or self._suggest_output_name()
        )

        cfg["RESUME_TRAINING"] = self.model_load_strategy_combo.currentIndex() == 1
        cfg["INSTANCE_DATASETS"] = self.dataset_manager.get_datasets_config()
        cfg["OPTIMIZER_TYPE"] = self.widgets["OPTIMIZER_TYPE"].currentData()
        cfg["LOSS_TYPE"] = "MSE"
        cfg["NOISE_MODE"] = "normal"
        cfg["TIMESTEP_MODE"] = self.ts_mode_combo.currentText()
        cfg["TIMESTEP_ODDS_SCALE"] = self.spin_odds_scale.value()
        if hasattr(self, 'timestep_histogram'): cfg["TIMESTEP_ALLOCATION"] = self.timestep_histogram.get_allocation()
        if hasattr(self, 'timestep_loss_curve'): cfg["TIMESTEP_LOSS_WEIGHT_CURVE"] = self.timestep_loss_curve.get_points()

        for prefix, key in [("RAVEN", "RAVEN_PARAMS"), ("TITAN", "TITAN_PARAMS")]:
            try:
                betas = [float(x) for x in self.widgets[f"{prefix}_betas"].text().split(',')]
            except Exception as exc:
                report_gui_exception(f"invalid {prefix} betas; using defaults", exc)
                betas = [0.9, 0.999]
            try: eps = float(self.widgets[f"{prefix}_eps"].text())
            except Exception as exc:
                report_gui_exception(f"invalid {prefix} epsilon; using default", exc)
                eps = 1e-8
            cfg[key] = {
                "betas": betas, "eps": eps,
                "weight_decay": self.widgets[f"{prefix}_weight_decay"].value(),
                "debias_strength": self.widgets[f"{prefix}_debias_strength"].value(),
            }
            

            cfg[key]["momentum_dtype"] = self.widgets[f"{prefix}_momentum_dtype"].currentText()

        try:
            paged_betas = [float(x) for x in self.widgets["PAGED_ADAMW_8BIT_betas"].text().split(',')]
        except Exception as exc:
            report_gui_exception("invalid paged AdamW 8-bit betas; using defaults", exc)
            paged_betas = [0.9, 0.999]
        try:
            paged_eps = float(self.widgets["PAGED_ADAMW_8BIT_eps"].text())
        except Exception as exc:
            report_gui_exception("invalid paged AdamW 8-bit epsilon; using default", exc)
            paged_eps = 1e-8
        cfg["PAGED_ADAMW_8BIT_PARAMS"] = {
            "betas": paged_betas,
            "eps": paged_eps,
            "weight_decay": self.widgets["PAGED_ADAMW_8BIT_weight_decay"].value(),
        }

        vae_norm_mode = self.widgets["VAE_NORMALIZATION_MODE"].currentText().split()[0]
        cfg["VAE_NORMALIZATION_MODE"] = vae_norm_mode
        for key in ["VAE_SHIFT_FACTOR", "VAE_SCALING_FACTOR"]:
            cfg[key] = None if vae_norm_mode == "flux_bn32" else self.widgets[key].value()
        cfg["VAE_LATENT_CHANNELS"] = self.widgets["VAE_LATENT_CHANNELS"].value()
        return cfg

    def _collect_config(self):
        self._store_current_mode_config(self.current_mode_key)
        self.current_preset["active_mode"] = self.current_mode_key
        return copy.deepcopy(self.current_preset)

    def _update_training_calculations(self):
        total_images_raw = sum(d["image_count"] for d in self.dataset_manager.datasets)
        total_images_with_repeats = self._effective_dataset_image_count()
        self.total_images_label.setText(f"{total_images_raw:,}")
        self.total_repeats_label.setText(f"{total_images_with_repeats:,}")
        try:
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
            grad_accum = int(self.widgets["GRADIENT_ACCUMULATION_STEPS"].text())
            batch = self.widgets["BATCH_SIZE"].value()
            opt_steps, _steps_per_epoch, epochs = gui_math.training_calculations(
                max_steps, grad_accum, batch, total_images_with_repeats
            )
            _, _, raw_image_epochs = gui_math.training_calculations(
                max_steps, grad_accum, batch, total_images_raw
            )
            self.optimizer_steps_label.setText(f"{opt_steps:,}")
            self.epochs_label.setText("∞" if epochs == float('inf') else f"{epochs:.2f}")
            self.raw_image_epochs_label.setText(
                "∞" if raw_image_epochs == float('inf') else f"{raw_image_epochs:.2f}"
            )
            if hasattr(self, 'timestep_histogram'): self.timestep_histogram.set_total_steps(max_steps)
        except (ValueError, KeyError):
            self.optimizer_steps_label.setText("Invalid")
            self.epochs_label.setText("Invalid")
            self.raw_image_epochs_label.setText("Invalid")

    def _effective_dataset_image_count(self):
        repeated_images = self.dataset_manager.get_total_repeats()
        multi_bucket_enabled = self.widgets["MULTI_BUCKET_ENABLED"].isChecked()
        extra_buckets = self.widgets["MULTI_BUCKET_EXTRA_BUCKETS"].value() if multi_bucket_enabled else 0
        return repeated_images * (1 + extra_buckets)

    def _update_and_clamp_lr_graph(self):
        if not hasattr(self, 'lr_curve_widget'): return
        try: steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
        except Exception as exc:
            report_gui_exception("invalid max training steps for LR graph; using 1", exc)
            steps = 1
        try: min_lr = float(self.widgets["LR_GRAPH_MIN"].text())
        except Exception as exc:
            report_gui_exception("invalid LR graph minimum; using 0", exc)
            min_lr = 0.0
        try: max_lr = float(self.widgets["LR_GRAPH_MAX"].text())
        except Exception as exc:
            report_gui_exception("invalid LR graph maximum; using 1e-6", exc)
            max_lr = 1e-6
        if "LR_CUSTOM_CURVE" in self.current_config:
            self.current_config["LR_CUSTOM_CURVE"] = [
                [p[0], max(min_lr, min(max_lr, p[1]))] for p in self.current_config["LR_CUSTOM_CURVE"]]
        self.lr_curve_widget.set_bounds(steps, min_lr, max_lr)
        self.lr_curve_widget.set_points(self.current_config.get("LR_CUSTOM_CURVE", []))
        self._update_epoch_markers_on_graph()

    def _update_epoch_markers_on_graph(self):
        if not hasattr(self, 'lr_curve_widget') or not hasattr(self, 'dataset_manager'): return
        try:
            total = self._effective_dataset_image_count()
            max_steps = int(self.widgets["MAX_TRAIN_STEPS"].text())
            batch = self.widgets["BATCH_SIZE"].value()
        except Exception as exc:
            report_gui_exception("could not update epoch markers; clearing them", exc)
            self.lr_curve_widget.set_epoch_data([])
            return
        if total > 0 and max_steps > 0:
            interval, count = gui_math.epoch_marker_interval(max_steps, batch, total)
            self.lr_curve_widget.set_epoch_interval(interval, count)
        else:
            self.lr_curve_widget.set_epoch_interval(0, 0)

    def _update_lr_button_states(self, sel_idx):
        if hasattr(self, 'remove_point_btn'):
            removable = 0 < sel_idx < len(self.lr_curve_widget.get_points()) - 1
            self.remove_point_btn.setEnabled(removable)

    def get_current_cache_folder_name(self):
        return self.get_current_cache_folder_names()[0]

    def get_current_cache_folder_names(self):
        if self._is_dit_mode():
            names = [
                self.current_config.get(
                    "ANIMA_CACHE_FOLDER_NAME", ".precomputed_anima_dit_cache"
                )
            ]
            if bool(self.current_config.get("ANIMA_SEMANTIC_LOSS_ENABLED", False)):
                names.append(".precomputed_anima_semantic_cache")
            return names
        pred_type = self.widgets["PREDICTION_TYPE"].currentText() if "PREDICTION_TYPE" in self.widgets else ""
        if pred_type == "rectified_flow":
            return [".precomputed_embeddings_cache_rf"]
        return [".precomputed_embeddings_cache_standard_sdxl"]

    def start_training(self):
        self.save_config()
        self.log("\n" + "=" * 50 + "\nStarting training process...\n" + "=" * 50)
        key = self.config_dropdown.itemData(self.config_dropdown.currentIndex()) or ""
        config_path = os.path.abspath(os.path.join(self.config_dir, f"{key}.json"))
        if not os.path.exists(config_path): self.log(f"CRITICAL: Config not found: {config_path}"); return
        is_dit = self.training_mode_combo.currentText() == TRAINING_MODE_ANIMA_DIT
        train_script = "train_anima.py" if is_dit else "train.py"
        train_py_path = str(PROJECT_ROOT / train_script)
        if not os.path.exists(train_py_path): self.log(f"CRITICAL: {train_script} not found."); return

        self.start_button.setVisible(False); self.stop_button.setVisible(True); self.force_save_button.setVisible(True)
        allocation = self.current_config.get("TIMESTEP_ALLOCATION") or {}
        self.live_metrics_widget.set_timestep_bucket_size(allocation.get("bin_size", 50))
        self.live_metrics_widget.clear_data()

        script_dir = str(PROJECT_ROOT)
        stale_force_save_flag = os.path.join(script_dir, "force_save.flag")
        if os.path.exists(stale_force_save_flag):
            try:
                os.remove(stale_force_save_flag)
            except Exception as e:
                self.log(f"Warning: Could not remove stale emergency checkpoint flag: {e}")

        env = os.environ.copy()
        python_dir = os.path.dirname(sys.executable)
        executable_dirs = [python_dir]
        windows_scripts = os.path.join(python_dir, "Scripts")
        if IS_WINDOWS and os.path.isdir(windows_scripts):
            executable_dirs.append(windows_scripts)
        env["PATH"] = os.pathsep.join([*executable_dirs, env.get("PATH", "")])
        env["PYTHONPATH"] = os.pathsep.join([script_dir, env.get("PYTHONPATH", "")])

        flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.process_runner = ProcessRunner(sys.executable, ["-u", train_py_path, "--config", config_path],
                                            script_dir, env, flags)
        self.process_runner.logSignal.connect(self.log)
        self.process_runner.progressSignal.connect(self.handle_process_output)
        self.process_runner.finishedSignal.connect(self.training_finished)
        self.process_runner.errorSignal.connect(self.log)
        self.process_runner.metricsSignal.connect(self.live_metrics_widget.parse_and_update)
        self.process_runner.cacheCreatedSignal.connect(self.dataset_manager.refresh_cache_buttons)
        if not prevent_sleep(True):
            self.log("WARNING: Could not inhibit system sleep on this platform.")
        self.process_runner.start()
        self.log(f"INFO: Starting {train_script} with config: {config_path}")

    def stop_training(self):
        if self.process_runner and self.process_runner.isRunning(): self.process_runner.stop()
        else: self.log("No active training process to stop.")

    def force_save_checkpoint(self):
        if self.process_runner and self.process_runner.isRunning():
            flag_path = str(PROJECT_ROOT / "force_save.flag")
            try:
                with open(flag_path, 'w') as f:
                    f.write("1")
                self.log("INFO: Requested emergency checkpoint. Will save at the end of the current optimizer step.")
            except Exception as e:
                self.log(f"ERROR: Could not create save flag: {e}")
        else:
            self.log("No active training process to save.")

    def training_finished(self, exit_code=0):
        stopped_by_user = bool(self.process_runner and self.process_runner.stop_requested)
        if self.process_runner: self.process_runner.quit(); self.process_runner.wait(); self.process_runner = None
        if stopped_by_user:
            result = "Training stopped by user."
        elif exit_code == 0:
            result = "Training finished successfully."
        else:
            result = f"Training finished with error (Code: {exit_code})."
        self.log("\n" + "=" * 50 + f"\n{result}\n" + "=" * 50)
        self.start_button.setVisible(True); self.stop_button.setVisible(False); self.force_save_button.setVisible(False)
        if hasattr(self, 'dataset_manager'): self.dataset_manager.refresh_cache_buttons()
        prevent_sleep(False)

    def log(self, message):
        self.append_log(str(message).strip(), replace=False)

    def append_log(self, text, replace=False):
        if not text:
            return
        self.log_textbox.append_line(text.rstrip(), replace_last=replace)

    def handle_process_output(self, text, is_progress):
        if text:
            self.append_log(text, replace=is_progress and self.last_line_is_progress)
            self.last_line_is_progress = is_progress

    def clear_console_log(self):
        self.log_textbox.clear()
        self.last_line_is_progress = False
        self.log("Console cleared.")
    def closeEvent(self, event):
        self._save_gui_state()
        if self.process_runner and self.process_runner.isRunning():
            reply = QtWidgets.QMessageBox.question(self, "Training in Progress",
                                                   "Training is running. Stop it and exit?",
                                                   QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self.stop_training()
                if self.process_runner: self.process_runner.quit(); self.process_runner.wait(2000)
                prevent_sleep(False)
                event.accept()
            else:
                event.ignore(); return
        if hasattr(self, 'dataset_manager'):
            for t in self.dataset_manager.loader_threads:
                if t.isRunning(): t.quit(); t.wait(1000)
        prevent_sleep(False)
        event.accept()

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        painter = QtGui.QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, painter, self)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    win = TrainingGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
