"""Film Lab main window."""
import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEasingCurve, QEvent, QObject, QPropertyAnimation, QRunnable, QSettings, Qt, QThread, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView,
                             QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QPushButton, QRadioButton, QScrollArea, QSplitter, QStackedWidget, QVBoxLayout,
                             QWidget)

from .. import export, geometry, imageio, looks, pipeline
from .. import __version__
from .. import auto as auto_mod
from .. import log as log_mod
from ..log import log
from .. import lens as lens_mod
from ..lut import Lut, LutError
from . import style
from .canvas import PreviewCanvas, Region
from .widgets import (EDITED_ROLE, KIND_ROLE, MARKED_ROLE, THUMB_ROLE, DropOverlay, ElidedLabel, FilmstripDelegate,
                      Histogram, LookDelegate, NoScrollComboBox, ParamSlider, Section, Toast)

APP_NAME = 'Film Lab'
DATA_DIR = log_mod.DATA_DIR
PREVIEW_SIDE = 1800
BUNDLED_LUT_DIR = Path(__file__).resolve().parent.parent / 'luts'


LUT_CATEGORIES = ('Color Negative', 'Slide', 'Black & White')


def bundled_luts():
    """[(category, film name, path)] of the film LUTs shipped with Film Lab."""
    out = []
    for cat in LUT_CATEGORIES:
        for p in sorted((BUNDLED_LUT_DIR / cat).glob('*')):
            if p.suffix.lower() in ('.png', '.tif', '.tiff', '.cube'):
                out.append((cat, p.stem, str(p)))
    return out


def lut_display_name(path):
    return Path(path).stem


LOOK_THUMB_SIDE = 260
EXPORT_SIZES = (('Full resolution', 0), ('4000 px long edge (large prints)', 4000),
                ('3000 px long edge', 3000), ('2048 px long edge (web, sharing)', 2048),
                ('1600 px long edge', 1600), ('1080 px long edge (Instagram, messages)', 1080))
FILTERS = (('all', 'All'), ('raw', 'RAW'), ('jpeg', 'Images'), ('marked', 'Marked'))
EXPORT_FORMATS = (('JPEG', 'jpeg'), ('PNG', 'png'), ('TIFF', 'tiff'), ('WebP', 'webp'))


def trash_file(path):
    """Move a file to the desktop Trash (restorable). Falls back to `gio trash` for drives where Qt
    can't find a Trash folder. Returns True on success; never deletes permanently."""
    from PyQt6.QtCore import QFile
    ok = QFile.moveToTrash(str(path))
    if isinstance(ok, tuple):
        ok = ok[0]
    if ok:
        return True
    import shutil
    import subprocess
    if shutil.which('gio'):
        try:
            return subprocess.run(['gio', 'trash', '--', str(path)], capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return False


# Background tasks

class _Signals(QObject):
    done = pyqtSignal(object)


class Task(QRunnable):
    """Runs fn(*args) in a thread pool and delivers the result (or exception) on the GUI thread."""
    _live = set()

    def __init__(self, fn, *args, callback=None):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _Signals()
        Task._live.add(self.signals)
        if callback:
            self.signals.done.connect(callback)
        self.signals.done.connect(lambda _: Task._live.discard(self.signals))

    def run(self):
        try:
            result = self.fn(*self.args)
        except Exception as e:
            log.exception('Background task %s failed', getattr(self.fn, '__name__', self.fn))
            result = e
        self.signals.done.emit(result)


def to_qimage(rgb):
    rgb = np.ascontiguousarray(pipeline.to_uint8(rgb))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def _small(lin, side):
    h, w = lin.shape[:2]
    k = max(1, int(max(h, w) / side))
    return lin[::k, ::k].copy() if k > 1 else lin


def _load_base(photo, source):
    """Preview-size source photo: (linear array, is_raw, full-resolution (h, w), lens info or None)."""
    t = time.time()
    lin, is_raw, full = imageio.load_linear(source, PREVIEW_SIDE)
    try:
        info = lens_mod.identify(imageio.read_exif(photo, source))
    except Exception:
        log.exception('Lens identification failed for %s', source)
        info = None
    ev = imageio.raw_brightness_match(source) if is_raw else 0.0
    log.info('Opened %s (%s, %dx%d) in %.2fs; brightness match %+.2f EV; lens: %s',
             Path(source).name, photo.kind, full[1], full[0], time.time() - t, ev,
             info.label if info else 'no profile')
    return lin, is_raw, full, info


def _histogram(img, bins=96):
    """(4, bins) normalised R, G, B, brightness histograms and (black, white) clipping fractions."""
    small = img[::3, ::3].reshape(-1, 3)
    L = small @ pipeline.LUMA
    hs = [np.histogram(c, bins=bins, range=(0, 1))[0] for c in (small[:, 0], small[:, 1], small[:, 2], L)]
    h = np.array(hs, np.float32)
    peak = max(1.0, float(h[:, 1:-1].max()))
    clip = (float((L <= 1 / 255).mean()), float((small >= 254 / 255).any(axis=1).mean()))
    return np.clip(np.sqrt(h / peak), 0, 1), clip


_warp_cache = {}
_warp_lock = threading.Lock()


def _render_preview(src, is_raw, settings, src_full, draft=False, lens_info=None, cache_key=None):
    """Whole cropped photo at preview size: (after QImage, before QImage, linear output, output size,
    histogram). A draft renders at half resolution for instant feedback while a slider is dragged."""
    if draft:
        src = src[::2, ::2]
    sp = src.shape[1] / src_full[1]
    geo, frame, shape = pipeline.output_frame(settings, src_full, sp)
    # Lens correction + crop only change when those settings change: reuse the result while
    # sliders like Exposure or Contrast are being moved
    # `cache_key` names the photo + file (memory addresses can be reused, so never key on id())
    key = None if cache_key is None else (cache_key, draft, src_full, lens_info, is_raw,
                                          tuple(getattr(settings, f) for f in pipeline.GEOMETRY_FIELDS[:6]
                                                + pipeline.LENS_FIELDS + ('lens_vig_amount',)))
    with _warp_lock:
        lin = _warp_cache.get(key) if key is not None else None
    if lin is None:
        corr = pipeline.lens_correction(settings, is_raw, lens_info, src_full)
        lin = pipeline.warp_source(src, src_full, sp, settings, frame, shape, corr)
        if key is not None:
            with _warp_lock:
                _warp_cache[key] = lin
                while len(_warp_cache) > 4:
                    _warp_cache.pop(next(iter(_warp_cache)))
    out = pipeline.render(lin, is_raw, settings, frame=frame)
    after = to_qimage(out)
    before = to_qimage(pipeline.render_original(lin, is_raw))
    return after, before, lin, geo.out_size, _histogram(out)


def _render_crop_view(src, is_raw, settings, src_full, lens_info=None):
    """The uncropped, unstraightened frame (quarter turns applied) with the look, for crop mode."""
    s = settings.without_geometry()
    s.rot90 = settings.rot90
    s.vignette = 0.0
    after, _, _, out_size, _ = _render_preview(src, is_raw, s, src_full, False, lens_info)
    return after, out_size


def _quick_preview(photo, source, settings):
    """A fast first look (~0.1 s) at a photo while the real preview loads: for RAW files the JPEG
    preview the camera embedded, for images a low-resolution decode. Settings are applied (no lens
    correction), so it looks close to the final result. Returns (after, before, output size) or None."""
    import io
    from PIL import Image as PILImage
    if imageio.is_raw(source):
        import rawpy
        with rawpy.imread(str(source)) as raw:
            w, h = raw.sizes.width, raw.sizes.height
            if raw.sizes.flip in (5, 6):
                w, h = h, w
            thumb = raw.extract_thumb()
            flip = raw.sizes.flip
        if thumb.format != rawpy.ThumbFormat.JPEG:
            return None
        im = PILImage.open(io.BytesIO(thumb.data)).convert('RGB')
        if flip in imageio._RAW_ROTATION:
            im = im.rotate(imageio._RAW_ROTATION[flip], expand=True)
        im.thumbnail((900, 900))
        lin = imageio.srgb_to_linear(np.asarray(im, np.float32) / 255.0)
        full = (h, w)
    else:
        lin, _, full = imageio.load_linear(source, 900)
    after, before, _, out_size, _ = _render_preview(lin, False, settings, full, False, None)
    return after, before, out_size


def _load_full(source):
    lin, is_raw, _ = imageio.load_linear(source)
    return lin, is_raw


def _render_region(key, full_src, is_raw, settings, rect, scale, context, lens_info=None):
    """Render output rect (x, y, w, h) of the cropped photo at `scale` (1 = 100%, or 1/k when zoomed out)."""
    k = 1 if scale >= 1 else max(1, int(1 / scale))
    res = pipeline.render_region(full_src, is_raw, settings, rect, lens_info, context, k, with_original=True)
    if res is None:
        return None
    after, before, real = res
    return Region(key, real, 1.0 / k, to_qimage(after), to_qimage(before))


def _render_look_thumbs(lin, is_raw, settings):
    small = _small(lin, LOOK_THUMB_SIDE)
    out = []
    for name in looks.names():
        s = settings.with_look_defaults(name)
        s.lut_path = ''
        s.grain = 0  # grain is invisible at this size anyway
        out.append(to_qimage(pipeline.render(small, is_raw, s)))
    return out


def _camera_line(photo, source):
    """'Sony ILCE-7M2 · FE 28-70mm F3.5-5.6 OSS · 1/250 s · f/8 · ISO 100 · 35 mm' (whatever is known)."""
    exif = imageio.read_exif(photo, source)
    x = exif.get_ifd(0x8769)
    make = str(exif.get(0x010F, '') or '').strip()
    model = str(exif.get(0x0110, '') or '').strip()
    brand = make.split()[0].title() if make else ''
    brand = {'Ricoh': 'Pentax'}.get(brand, brand) if 'PENTAX' in model.upper() else brand
    camera = model if brand and brand.lower() in model.lower() else ('%s %s' % (brand, model)).strip()
    parts = [camera] if camera else []
    lens = str(x.get(0xA434, '') or '').strip('\x00 ')
    if lens and lens not in ('----', 'None'):
        parts.append(lens)
    try:
        t = float(x.get(0x829A))
        parts.append(('1/%d s' % round(1 / t)) if 0 < t < 0.5 else ('%g s' % round(t, 1)))
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    try:
        parts.append('f/%s' % ('%.1f' % float(x.get(0x829D))).rstrip('0').rstrip('.'))
    except (TypeError, ValueError):
        pass
    iso = x.get(0x8827)
    if isinstance(iso, tuple):
        iso = iso[0] if iso else None
    if iso:
        parts.append('ISO %s' % iso)
    try:
        parts.append('%d mm' % round(float(x.get(0x920A))))
    except (TypeError, ValueError):
        pass
    return '  ·  '.join(parts)


def _load_thumb(photo):
    im = imageio.load_thumbnail(photo, 300)
    rgb = np.asarray(im.convert('RGB'))
    h, w = rgb.shape[:2]
    return QImage(np.ascontiguousarray(rgb).data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ExportThread(QThread):
    """Exports photos sequentially. Each photo is rendered in bands on a low-priority thread pool,
    which bounds memory use and lets Cancel stop between bands."""
    progress = pyqtSignal(int, int, str, object)   # done (thousandths of photos), total*1000, name, error
    finished_all = pyqtSignal(int, list, bool)     # ok count, errors, cancelled

    def __init__(self, jobs, quality, workers, long_edge=0, fmt='jpeg'):
        super().__init__()
        self.jobs, self.quality, self.workers, self.long_edge, self.fmt = jobs, quality, workers, long_edge, fmt
        self.cancelled = False

    def run(self):
        from concurrent.futures import ThreadPoolExecutor
        export.lower_thread_priority()
        errors, ok = [], 0
        total = len(self.jobs)
        with ThreadPoolExecutor(max_workers=self.workers, initializer=export.lower_thread_priority,
                                thread_name_prefix='export') as pool:
            for i, (photo, sd, out) in enumerate(self.jobs):
                if self.cancelled:
                    break
                name = Path(out).name

                def step(frac, i=i, name=name):
                    self.progress.emit(int((i + frac) * 1000), total * 1000, name, None)

                path, err = export.export_photo(photo, sd, str(out), self.quality, self.long_edge, self.fmt,
                                                submit=pool.submit, cancelled=lambda: self.cancelled,
                                                progress=step)
                if err == 'cancelled':
                    break
                log.info('Exported %s -> %s%s', photo.stem, Path(path).name, ' FAILED: %s' % err if err else '')
                if err:
                    errors.append('%s: %s' % (Path(path).name, err))
                else:
                    ok += 1
                self.progress.emit((i + 1) * 1000, total * 1000, name, err)
        self.finished_all.emit(ok, errors, self.cancelled)


# Dialogs

class ExportDialog(QDialog):
    def __init__(self, parent, out_dir, n_selected, n_marked, n_all, quality, suffix, filtered=False,
                 example=None, current_name=''):
        super().__init__(parent)
        self.setWindowTitle('Export')
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(12)

        title = QLabel('Export photos')
        title.setObjectName('title')
        lay.addWidget(title)

        self.rb_marked = QRadioButton('Marked photos (%d)' % n_marked)
        self.rb_selected = QRadioButton('Selected photos (%d)' % n_selected if n_selected > 1
                                        else 'This photo (%s)' % current_name)
        self.rb_all = QRadioButton(('All photos shown (%d)' if filtered else 'All photos in folder (%d)') % n_all)
        self.rb_marked.setEnabled(n_marked > 0)
        if n_marked == 0:
            self.rb_marked.setToolTip('Mark photos with the ✓ badge on their thumbnail, or press P')
        (self.rb_marked if n_marked else self.rb_selected).setChecked(True)
        for rb in (self.rb_marked, self.rb_selected, self.rb_all):
            lay.addWidget(rb)

        lay.addSpacing(6)
        lay.addWidget(QLabel('Save to'))
        row = QHBoxLayout()
        self.path = QLineEdit(str(out_dir))
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        self.fmt = NoScrollComboBox()
        for label, key in EXPORT_FORMATS:
            self.fmt.addItem(label, key)
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel('Format'))
        fmt_row.addWidget(self.fmt, 1)
        lay.addLayout(fmt_row)
        self.size = NoScrollComboBox()
        for label, px in EXPORT_SIZES:
            self.size.addItem(label, px)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel('Size'))
        size_row.addWidget(self.size, 1)
        lay.addLayout(size_row)
        self.quality = ParamSlider('JPEG quality', 70, 100, 95, 1, fmt=lambda v: '%d' % v)
        self.quality.set_value(quality)
        lay.addWidget(self.quality)
        self.fmt.currentIndexChanged.connect(
            lambda _: self.quality.setEnabled(self.fmt.currentData() in ('jpeg', 'webp')))
        self.suffix = QCheckBox('Add the look name to file names')
        self.suffix.setChecked(suffix)
        lay.addWidget(self.suffix)
        self.example_label = QLabel('')
        self.example_label.setObjectName('faint')
        lay.addWidget(self.example_label)
        self._example = example          # function(fmt, with_suffix) -> file name of the first photo
        self.suffix.toggled.connect(self._update_example)
        self.fmt.currentIndexChanged.connect(self._update_example)
        self._update_example()

        note = QLabel('Originals are never modified. Existing files with the same name are replaced.')
        note.setObjectName('faint')
        note.setWordWrap(True)
        lay.addWidget(note)

        buttons = QDialogButtonBox()
        go = buttons.addButton('Export', QDialogButtonBox.ButtonRole.AcceptRole)
        go.setObjectName('primary')
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addSpacing(6)
        lay.addWidget(buttons)

    def _update_example(self, *_):
        if self._example is not None:
            self.example_label.setText('File name: ' + self._example(self.fmt.currentData() or 'jpeg',
                                                                      self.suffix.isChecked()))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, 'Export folder', self.path.text())
        if d:
            self.path.setText(d)


# Main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setAcceptDrops(True)
        self.qsettings = QSettings('film-lab', 'film-lab')

        self.folder = None
        self.photos = []
        self.photo_settings = {}          # photo key -> Settings (photos you've touched)
        self.marked = set()               # photo keys marked for export
        self.settings = pipeline.Settings()
        self.bases = OrderedDict()        # photo key + source -> (preview-size lin, is_raw, full size)
        self.preview_context = None       # (view key, linear preview of the cropped photo) for regions
        self.crop_mode = False
        self.history = {}                 # photo key -> [Settings snapshots] for undo
        self.future = {}                  # photo key -> [Settings] for redo
        self._restoring = False
        self.clipboard = None
        self.filter_mode = 'all'
        self.current_key = None
        self.look_thumb_key = None

        self.preview_pool = QThreadPool()
        self.preview_pool.setMaxThreadCount(1)
        self.load_pool = QThreadPool()
        self.load_pool.setMaxThreadCount(2)
        self.thumb_pool = QThreadPool()
        self.thumb_pool.setMaxThreadCount(4)
        self.quick_pool = QThreadPool()      # quick preview for a newly opened photo
        self.quick_pool.setMaxThreadCount(1)
        self.look_pool = QThreadPool()       # look-gallery thumbnails
        self.look_pool.setMaxThreadCount(1)
        self.look_timer = QTimer(self, singleShot=True, interval=350, timeout=self._start_look_thumbs)
        self.render_running = False
        self.render_pending = False
        self.export_thread = None
        self._preloading = set()
        self._session_current = None
        self._controls = {}

        # Full-resolution data for 100% zoom and the loupe (current photo only)
        self.full_base = None             # (base key, lin, is_raw)
        self.full_loading = None          # base key being loaded
        self.region_pools = {'detail': QThreadPool(), 'loupe': QThreadPool()}
        for pool in self.region_pools.values():
            pool.setMaxThreadCount(1)
        self.region_running = {'detail': False, 'loupe': False}
        self.region_pending = {'detail': None, 'loupe': None}

        self.render_timer = QTimer(self, singleShot=True, interval=25, timeout=self._start_render)
        self.save_timer = QTimer(self, singleShot=True, interval=1500, timeout=self._save_session)
        self.undo_timer = QTimer(self, singleShot=True, interval=400, timeout=self._snapshot)
        self.final_timer = QTimer(self, singleShot=True, interval=280, timeout=self._final_render)
        self._interactive = False
        self._last_request = 0.0
        self._last_render_draft = False
        self._fade_next = False

        self._build_ui()
        self._update_mark_count()
        self._restore_window()
        last = self.qsettings.value('last_folder', '')
        if last and Path(last).is_dir():
            QTimer.singleShot(0, lambda: self.open_folder(last))

    # Layout

    def _build_ui(self):
        self.setStyleSheet(style.QSS)
        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_toolbar())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_empty_state())
        self.stack.addWidget(self._build_editor())
        rl.addWidget(self.stack, 1)
        self.setCentralWidget(root)

        self.toast = Toast(root)
        self.toast.anchor = self.canvas
        self.drop_overlay = DropOverlay(root)
        self.statusBar().hide()
        QShortcut(QKeySequence('F1'), self, activated=self.show_shortcuts)
        QShortcut(QKeySequence('?'), self, activated=self.show_shortcuts)

        QShortcut(QKeySequence('Ctrl+O'), self, activated=self.choose_folder)
        QShortcut(QKeySequence('Ctrl+E'), self, activated=self.export_dialog)
        QShortcut(QKeySequence('Ctrl+R'), self, activated=self.revert)
        QShortcut(QKeySequence('A'), self, activated=self.toggle_auto)
        QShortcut(QKeySequence('C'), self, activated=self.compare_btn.toggle)
        QShortcut(QKeySequence('Ctrl+Shift+V'), self, activated=self.apply_to_all)
        for key in ('0', 'F'):
            QShortcut(QKeySequence(key), self, activated=lambda: self.canvas.zoom_fit())
        QShortcut(QKeySequence('1'), self, activated=lambda: self.canvas.zoom_to(1.0, self._zoom_anchor()))
        QShortcut(QKeySequence('2'), self, activated=lambda: self.canvas.zoom_to(2.0, self._zoom_anchor()))
        for key in ('+', '='):
            QShortcut(QKeySequence(key), self, activated=lambda: self.canvas.zoom_by(1.25, self._zoom_anchor()))
        QShortcut(QKeySequence('-'), self, activated=lambda: self.canvas.zoom_by(1 / 1.25, self._zoom_anchor()))
        QShortcut(QKeySequence('L'), self, activated=lambda: self.canvas.toggle_loupe())
        QShortcut(QKeySequence('P'), self, activated=self.toggle_marks)
        QShortcut(QKeySequence('R'), self, activated=self.toggle_crop)
        for key in ('Return', 'Enter', 'Escape'):
            QShortcut(QKeySequence(key), self, activated=lambda: self.crop_mode and self.toggle_crop(False))
        QShortcut(QKeySequence('Ctrl+B'), self, activated=lambda: self._set_strip_visible(not self.strip_visible))
        QShortcut(QKeySequence('Delete'), self, activated=self.delete_photos)
        QShortcut(QKeySequence('Ctrl+Z'), self, activated=self.undo)
        for key in ('Ctrl+Shift+Z', 'Ctrl+Y'):
            QShortcut(QKeySequence(key), self, activated=self.redo)
        QShortcut(QKeySequence('Ctrl+C'), self, activated=self.copy_settings)
        QShortcut(QKeySequence('Ctrl+V'), self, activated=self.paste_settings)

    def _build_toolbar(self):
        bar = QWidget()
        bar.setObjectName('toolbar')
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(8)

        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(22, 22))
        name = QLabel('FILM LAB')
        name.setToolTip('Film Lab by Fishbone')
        name.setObjectName('appName')
        lay.addWidget(logo)
        lay.addWidget(name)
        lay.addSpacing(14)

        self.open_btn = self._button('Open Folder', self.choose_folder, tip='Open a folder or memory card (Ctrl+O)')
        lay.addWidget(self.open_btn)
        self.folder_label = ElidedLabel('')
        self.folder_label.setObjectName('dim')
        lay.addWidget(self.folder_label, 1)


        self.undo_btn = self._button('↶', self.undo, name='flat', tip='Undo (Ctrl+Z)')
        self.redo_btn = self._button('↷', self.redo, name='flat', tip='Redo (Ctrl+Shift+Z)')
        for b in (self.undo_btn, self.redo_btn):
            b.setStyleSheet('font-size: 15pt; padding: 0 8px;')
            lay.addWidget(b)
        self.compare_btn = self._button('Before / After', None, checkable=True,
                                        tip='Split view: drag to compare (C). Hold Space to see the original.')
        self.compare_btn.toggled.connect(lambda on: self.canvas.set_compare(on))
        lay.addWidget(self.compare_btn)

        self.recipes_btn = self._button('Recipes', None, tip='Save and reuse your favourite settings')
        self.recipes_menu = QMenu(self)
        self.recipes_menu.aboutToShow.connect(self._fill_recipes_menu)
        self.recipes_btn.setMenu(self.recipes_menu)
        lay.addWidget(self.recipes_btn)

        self.export_btn = self._button('Export', self.export_dialog, name='primary', tip='Export photos (Ctrl+E)')
        lay.addWidget(self.export_btn)
        help_btn = self._button('?', None, name='round', tip='Help: keyboard shortcuts, about Film Lab')
        help_menu = QMenu(self)
        help_menu.addAction('Keyboard Shortcuts\tF1', self.show_shortcuts)
        help_menu.addSeparator()
        help_menu.addAction('Copy Diagnostic Info', self.copy_diagnostics)
        help_menu.addAction('Open Log Folder', self.open_log_folder)
        help_menu.addSeparator()
        help_menu.addAction('About Film Lab', self.show_about)
        help_btn.setMenu(help_menu)
        lay.addWidget(help_btn)
        for b in (self.compare_btn, self.recipes_btn, self.export_btn):
            b.setEnabled(False)
        return bar

    def _button(self, text, slot, checkable=False, name=None, tip=''):
        b = QPushButton(text)
        b.setCheckable(checkable)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        if name:
            b.setObjectName(name)
        if tip:
            b.setToolTip(tip)
        if slot:
            b.clicked.connect(slot)
        return b

    def _build_empty_state(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addStretch(2)
        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(96, 96))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        byline = QLabel('Film Lab  ·  by Fishbone')
        byline.setObjectName('dim')
        byline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel('Your darkroom is ready')
        title.setObjectName('emptyTitle')
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text = QLabel('Open a folder of photos or your camera\'s memory card.\n'
                      'RAW (.ARW) and JPEG files are both supported.')
        text.setObjectName('emptyText')
        text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn = self._button('Open Folder…', self.choose_folder, name='primary')
        btn.setMinimumWidth(180)
        hint = QLabel('or drop a folder onto this window')
        hint.setObjectName('faint')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        for wd in (icon, byline, title, text):
            lay.addWidget(wd)
            lay.addSpacing(6)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(btn)
        row.addStretch()
        lay.addSpacing(10)
        lay.addLayout(row)
        lay.addWidget(hint)
        lay.addStretch(3)
        return w

    def _build_editor(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        self.canvas = PreviewCanvas()
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.canvas.detailNeeded.connect(lambda rect, scale: self._request_region('detail', rect, scale))
        self.canvas.loupeNeeded.connect(lambda rect, scale: self._request_region('loupe', rect, scale))
        self.canvas.cropChanged.connect(self._on_canvas_crop)
        self.canvas.cropFinished.connect(self._on_canvas_crop)
        self.canvas.angleRequested.connect(self.set_angle)
        split.addWidget(self.canvas)
        split.addWidget(self._build_panel())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, 330])
        self.splitter = split
        lay.addWidget(split, 1)

        strip_bar = QWidget()
        strip_bar.setObjectName('filmstripBar')
        sl = QVBoxLayout(strip_bar)
        sl.setContentsMargins(8, 6, 8, 4)
        sl.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(6, 0, 6, 0)
        self.strip_count = QLabel('')
        self.strip_count.setObjectName('faint')
        self.filter_btns = {}
        filt = QHBoxLayout()
        filt.setSpacing(0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, (key, label) in enumerate(FILTERS):
            name = 'segmentLeft' if i == 0 else ('segmentRight' if i == len(FILTERS) - 1 else 'segment')
            b = self._button(label, lambda _=False, k=key: self.set_filter(k), checkable=True, name=name,
                             tip={'all': 'Show every photo', 'raw': 'Only photos that have a RAW file',
                                  'jpeg': 'Only photos that have a JPEG, PNG, TIFF, HEIF or other image file',
                                  'marked': 'Only photos marked for export'}[key])
            b.setProperty('small', True)
            group.addButton(b)
            filt.addWidget(b)
            self.filter_btns[key] = b
        self.filter_btns['all'].setChecked(True)
        self.delete_btn = self._button('Delete…', self.delete_photos, name='danger',
                                       tip='Move the selected photos to the Trash (Delete)')
        self.mark_count = QLabel('')
        self.mark_count.setObjectName('dim')
        self.mark_btn = self._button('Mark / Unmark', self.toggle_marks, name='flat',
                                     tip='Mark the selected photos for export (P).\nOr click the ✓ badge on a thumbnail.')
        self.clear_marks_btn = self._button('Clear Marks', self.clear_marks, name='flat')
        head.addLayout(filt)
        head.addSpacing(12)
        head.addWidget(self.strip_count)
        head.addStretch()
        head.addWidget(self.mark_count)
        head.addWidget(self.mark_btn)
        head.addWidget(self.clear_marks_btn)
        head.addSpacing(10)
        head.addWidget(self.delete_btn)
        sl.addLayout(head)
        self.filmstrip = QListWidget()
        self.filmstrip.setMouseTracking(True)
        self.filmstrip.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.filmstrip.viewport().installEventFilter(self)
        self.filmstrip.setViewMode(QListView.ViewMode.IconMode)
        self.filmstrip.setFlow(QListView.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setMovement(QListView.Movement.Static)
        self.filmstrip.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.filmstrip.setHorizontalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip.setItemDelegate(FilmstripDelegate(self.filmstrip))
        self.filmstrip.setFixedHeight(FilmstripDelegate.SIZE.height() + 18)
        self.filmstrip.setUniformItemSizes(True)
        self.filmstrip.currentItemChanged.connect(self._on_current_photo)
        sl.addWidget(self.filmstrip)
        # Drawer-style tab on the filmstrip's top edge: hides/shows it, and stays put when it's hidden
        tab_row = QWidget()
        tab_row.setObjectName('stripTabRow')
        tr = QHBoxLayout(tab_row)
        tr.setContentsMargins(0, 0, 0, 0)
        tr.addStretch()
        self.strip_tab = QPushButton()
        self.strip_tab.setObjectName('stripTab')
        self.strip_tab.setCursor(Qt.CursorShape.PointingHandCursor)
        self.strip_tab.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.strip_tab.setToolTip('Show or hide the filmstrip (Ctrl+B)')
        self.strip_tab.clicked.connect(lambda: self._set_strip_visible(not self.strip_visible))
        tr.addWidget(self.strip_tab)
        tr.addStretch()
        lay.addWidget(tab_row)
        lay.addWidget(strip_bar)
        self.strip_bar = strip_bar
        self._strip_anim = QPropertyAnimation(strip_bar, b'maximumHeight', self)
        self._strip_anim.setDuration(200)
        self._strip_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._strip_anim.finished.connect(lambda: self.strip_bar.setVisible(self.strip_visible))
        self.strip_visible = self.qsettings.value('strip_visible', True, type=bool)
        strip_bar.setVisible(self.strip_visible)
        self._update_strip_tab()
        return w

    def _build_panel(self):
        panel = QWidget()
        panel.setObjectName('panel')
        panel.setMinimumWidth(300)
        panel.setMaximumWidth(420)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(18, 14, 18, 18)
        lay.setSpacing(4)

        body.setMinimumWidth(0)
        self.photo_title = ElidedLabel('')
        self.photo_title.setObjectName('title')
        self.photo_info = ElidedLabel('', Qt.TextElideMode.ElideRight)
        self.photo_info.setObjectName('dim')
        lay.addWidget(self.photo_title)
        lay.addWidget(self.photo_info)
        self.photo_camera = ElidedLabel('', Qt.TextElideMode.ElideRight)
        self.photo_camera.setObjectName('faint')
        lay.addWidget(self.photo_camera)
        lay.addSpacing(8)
        self.histogram = Histogram()
        lay.addWidget(self.histogram)
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 8, 0, 0)
        actions.setSpacing(6)
        self.auto_btn = auto_btn = QPushButton('Auto')
        auto_btn.setCheckable(True)
        auto_btn.setObjectName('autoToggle')
        auto_btn.setToolTip('Auto sets Exposure, Highlights, Shadows, Contrast and white balance for this photo.\n'
                            'Lit up = Auto is on. Click again to turn it off and get your previous values back. (A)')
        auto_btn.clicked.connect(lambda _=False: self.toggle_auto())
        actions.addWidget(auto_btn, 1)
        revert = QPushButton('Revert')
        revert.setToolTip('Back to the photo as shot: removes the look, every adjustment and the crop.\n'
                          'Ctrl+Z brings your edits back. (Ctrl+R)')
        revert.clicked.connect(self.revert)
        to_all = QPushButton('Apply to All')
        to_all.setToolTip('Use this photo\'s look and adjustments for every photo in the folder.\n'
                          'Each photo keeps its own crop. (Ctrl+Shift+V)')
        to_all.clicked.connect(self.apply_to_all)
        for b in (revert, to_all):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            actions.addWidget(b, 1)
        lay.addLayout(actions)
        self.source_row = QWidget()
        sr = QHBoxLayout(self.source_row)
        sr.setContentsMargins(0, 6, 0, 0)
        sr.setSpacing(0)
        lab = QLabel('Develop from')
        lab.setObjectName('dim')
        sr.addWidget(lab)
        sr.addStretch()
        self.src_raw_btn = self._button('RAW', lambda: self.set_source('raw'), checkable=True, name='segmentLeft',
                                        tip='Develop from the RAW file: most detail and latitude')
        self.src_jpeg_btn = self._button('JPEG', lambda: self.set_source('jpeg'), checkable=True,
                                         name='segmentRight', tip="Develop from the camera's JPEG")
        g = QButtonGroup(self)
        g.setExclusive(True)
        g.addButton(self.src_raw_btn)
        g.addButton(self.src_jpeg_btn)
        sr.addWidget(self.src_raw_btn)
        sr.addWidget(self.src_jpeg_btn)
        lay.addWidget(self.source_row)

        # Look gallery
        sec = Section('Look', store=self.qsettings)
        self.look_list = QListWidget()
        self.look_list.setObjectName('looks')
        self.look_list.setViewMode(QListView.ViewMode.IconMode)
        self.look_list.setMovement(QListView.Movement.Static)
        self.look_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.look_list.setUniformItemSizes(True)
        self.look_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.look_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.look_list.setItemDelegate(LookDelegate(self.look_list))
        self.look_list.setMouseTracking(True)
        for look in looks.LOOKS:
            it = QListWidgetItem(look.name)
            it.setToolTip('<b>%s</b><br>%s<br><i>%s</i>' % (look.name, look.description, look.category))
            self.look_list.addItem(it)
        tabs = QHBoxLayout()
        tabs.setSpacing(0)
        tg = QButtonGroup(self)
        tg.setExclusive(True)
        self.look_tabs = {}
        cats = (('all', 'All'), ('Color Negative', 'Negative'), ('Slide', 'Slide'), ('Black & White', 'B&&W'))
        for i, (key, label) in enumerate(cats):
            name = 'segmentLeft' if i == 0 else ('segmentRight' if i == len(cats) - 1 else 'segment')
            b = self._button(label, lambda _=False, k=key: self._filter_looks(k), checkable=True, name=name)
            b.setProperty('small', True)
            tg.addButton(b)
            tabs.addWidget(b, 1)
            self.look_tabs[key] = b
        self.look_tabs['all'].setChecked(True)
        tab_row = QWidget()
        tab_row.setLayout(tabs)
        tabs.setContentsMargins(0, 0, 0, 4)
        sec.add(tab_row)
        self.look_list.currentRowChanged.connect(self._on_look_row)
        self._look_filter = 'all'
        sec.add(self.look_list)
        self.look_desc = QLabel('')
        self.look_desc.setObjectName('lookDescription')
        self.look_desc.setWordWrap(True)
        sec.add(self.look_desc)
        self.sl_strength = sec.add(self._slider('strength', 'Strength', 0, 1, 1, 0.01, pct=True,
                                                tip='Blend between the plain photo and the full look'))
        lay.addWidget(sec)

        sec = Section('Crop & Straighten', store=self.qsettings)
        self.crop_btn = QPushButton('Crop && Straighten')
        self.crop_btn.setCheckable(True)
        self.crop_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.crop_btn.setToolTip('Crop and level the photo (R). Enter or R again when done.')
        self.crop_btn.clicked.connect(lambda on: self.toggle_crop(on))
        sec.add(self.crop_btn)
        self.aspect = NoScrollComboBox()
        self.aspect.addItems(geometry.ASPECTS)
        self.aspect.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.aspect.currentTextChanged.connect(self.set_aspect)
        self.orient_btn = QPushButton('⇄')
        self.orient_btn.setObjectName('flat')
        self.orient_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.orient_btn.setToolTip('Switch the aspect ratio between landscape and portrait')
        self.orient_btn.clicked.connect(self.flip_aspect)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        lab = QLabel('Aspect')
        lab.setMinimumWidth(70)
        rl.addWidget(lab)
        rl.addWidget(self.aspect, 1)
        rl.addWidget(self.orient_btn)
        sec.add(row)
        self.sl_angle = sec.add(self._slider('angle', 'Straighten', -45, 45, 0, 0.1,
                                             fmt=lambda v: '%+.1f°' % v, handler=self.set_angle,
                                             tip='Level the horizon. In crop mode you can also Ctrl+drag\n'
                                                 'along something that should be level.'))
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        auto_level = QPushButton('Auto-straighten')
        auto_level.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        auto_level.setToolTip('Levels the horizon or the strongest vertical lines.\n'
                              'Works best with clear straight lines (horizons, buildings, shorelines).')
        auto_level.clicked.connect(self.auto_straighten)
        sec.add(auto_level)
        for text, tip, slot in (('⟲  Rotate', 'Rotate 90° counter-clockwise', lambda: self.rotate(-1)),
                                ('⟳  Rotate', 'Rotate 90° clockwise', lambda: self.rotate(1)),
                                ('Reset', 'Remove crop and straightening', self.reset_crop)):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            rl.addWidget(b)
        sec.add(row)
        lay.addWidget(sec)

        sec = Section('Lens Corrections', store=self.qsettings)
        self.lens_label = ElidedLabel('', Qt.TextElideMode.ElideRight)
        self.lens_label.setObjectName('dim')
        sec.add(self.lens_label)
        self.lens_mode = NoScrollComboBox()
        for label, key in (('Auto (RAW files)', 'auto'), ('On', 'on'), ('Off', 'off')):
            self.lens_mode.addItem(label, key)
        self.lens_mode.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lens_mode.setToolTip('Auto corrects RAW files only: camera JPEGs already have the\n'
                                  'camera\'s own lens corrections applied.')
        self.lens_mode.currentIndexChanged.connect(self._on_lens_mode)
        sec.add(self._labeled('Apply', self.lens_mode))
        self.lens_checks = {}
        for field, text, tip in (('lens_distortion', 'Distortion', 'Straightens lines bent by the lens'),
                                 ('lens_tca', 'Colour fringes', 'Lines up the colour channels (chromatic aberration)'),
                                 ('lens_vignetting', 'Vignetting', 'Brightens the darker corners')):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            cb.toggled.connect(lambda on, f=field: self._set_field(f, bool(on)))
            self.lens_checks[field] = cb
            sec.add(cb)
        sec.add(self._slider('lens_vig_amount', 'Vignetting amount', 0, 1, 1, 0.01, pct=True,
                             tip='How much of the corner darkening to remove. 100% = fully even.'))
        self.lens_section = sec
        lay.addWidget(sec)

        sec = Section('Light', store=self.qsettings,
                      action=('Auto', 'Set exposure, highlights, shadows and contrast automatically',
                              lambda: self.auto_adjust(True, False)))
        self.sl_exposure = sec.add(self._slider('exposure', 'Exposure', -3, 3, 0, 0.05,
                                                fmt=lambda v: '%+.2f EV' % v))
        sec.add(self._slider('contrast', 'Contrast', -1, 1, 0, 0.01))
        sec.add(self._slider('highlights', 'Highlights', -1, 1, 0, 0.01,
                             tip='Pull down to recover detail in bright skies and highlights'))
        sec.add(self._slider('shadows', 'Shadows', -1, 1, 0, 0.01,
                             tip='Push up to open up dark foregrounds and shadows'))
        sec.add(self._slider('fade', 'Fade', 0, 0.3, 0, 0.005, pct_of=0.3,
                             tip='Lifts the blacks for a matte, printed look'))
        lay.addWidget(sec)

        self.color_section = Section('Color', store=self.qsettings,
                                     action=('Auto', 'Neutralise a colour cast (gently, so warm or cool light keeps its mood)',
                                             lambda: self.auto_adjust(False, True)))
        self.color_section.add(self._slider('warmth', 'Warmth', -1, 1, 0, 0.01))
        self.color_section.add(self._slider('tint', 'Tint', -1, 1, 0, 0.01, tip='Green ← → magenta'))
        self.color_section.add(self._slider('saturation', 'Saturation', -1, 1, 0, 0.01))
        lay.addWidget(self.color_section)

        self.bw_section = Section('Black & White', store=self.qsettings)
        self.bw_filter = self._combo('bw_filter', list(pipeline.BW_FILTERS))
        self.bw_filter.setToolTip('Film: the film stock\'s own response to colour.\n'
                                  'Or simulate a coloured lens filter: red/orange darken blue skies and\n'
                                  'smooth skin; yellow is a gentle landscape filter; green lightens foliage.')
        self.toning = self._combo('toning', list(pipeline.TONINGS))
        self.bw_section.add(self._labeled('Filter', self.bw_filter))
        self.bw_section.add(self._labeled('Toning', self.toning))
        self.bw_section.add(self._slider('toning_amount', 'Toning amount', 0, 1, 0, 0.01, pct=True))
        lay.addWidget(self.bw_section)

        sec = Section('Detail', store=self.qsettings)
        sec.add(self._slider('sharpen', 'Sharpening', 0, 1, 0, 0.01, pct=True,
                             tip='Crisper fine detail. RAW files start softer than camera JPEGs;\n'
                                 'around 30-50% is typical. Check at 100%.'))
        sec.add(self._slider('nr_luma', 'Noise: luminance', 0, 1, 0, 0.01, pct=True,
                             tip='Smooths grainy speckle from high ISO while keeping edges.\n'
                                 'Check at 100% or with the loupe: too much looks waxy.'))
        sec.add(self._slider('nr_color', 'Noise: color', 0, 1, 0, 0.01, pct=True,
                             tip='Removes red/green/blue blotches in shadows and flat areas.\n'
                                 'Safe to use generously on high-ISO photos.'))
        lay.addWidget(sec)

        sec = Section('Film', store=self.qsettings)
        sec.add(self._slider('grain', 'Grain', 0, 1, 0, 0.01, pct=True))
        sec.add(self._slider('grain_size', 'Grain size', 0.5, 3, 1, 0.05, fmt=lambda v: '%.2f×' % v))
        sec.add(self._slider('halation', 'Halation', 0, 1, 0, 0.01, pct=True,
                             tip='Red glow around bright highlights, as on cinema film'))
        sec.add(self._slider('vignette', 'Vignette', -1, 1, 0, 0.01))
        lay.addWidget(sec)

        sec = Section('LUT', store=self.qsettings)
        row = QHBoxLayout()
        self.lut_label = ElidedLabel('None', Qt.TextElideMode.ElideRight)
        self.lut_label.setObjectName('dim')
        load = QPushButton('Choose ▾')
        load.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.lut_menu = QMenu(self)
        self.lut_menu.aboutToShow.connect(self._fill_lut_menu)
        load.setMenu(self.lut_menu)
        self.lut_clear = QPushButton('✕')
        self.lut_clear.setObjectName('flat')
        self.lut_clear.setToolTip('Remove LUT')
        self.lut_clear.clicked.connect(lambda: self._set_lut(''))
        row.addWidget(self.lut_label, 1)
        row.addWidget(self.lut_clear)
        row.addWidget(load)
        holder = QWidget()
        holder.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        sec.add(holder)
        self.sl_lut_amount = sec.add(self._slider('lut_amount', 'LUT amount', 0, 1, 1, 0.01, pct=True))
        lay.addWidget(sec)

        lay.addSpacing(10)
        credit = QLabel('Film Lab  ·  by Fishbone')
        credit.setObjectName('faint')
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(credit)
        lay.addStretch()

        scroll.setWidget(body)
        outer.addWidget(scroll)
        return panel

    def _slider(self, field, label, lo, hi, default, step, fmt=None, pct=False, pct_of=None, tip='', handler=None):
        if pct:
            fmt = lambda v: '%d%%' % round(v * 100)
        elif pct_of:
            fmt = lambda v, m=pct_of: '%d%%' % round(v / m * 100)
        s = ParamSlider(label, lo, hi, default, step, fmt=fmt, tooltip=tip)
        s.valueChanged.connect(handler if handler else (lambda v, f=field: self._set_field(f, v)))
        self._controls[field] = s
        return s

    def _combo(self, field, items):
        c = NoScrollComboBox()
        c.addItems(items)
        c.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        c.currentTextChanged.connect(lambda t, f=field: self._set_field(f, t))
        self._controls[field] = c
        return c

    def _labeled(self, text, widget):
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lab = QLabel(text)
        lab.setMinimumWidth(70)
        lay.addWidget(lab)
        lay.addWidget(widget, 1)
        return w

    def _filter_looks(self, category):
        """Show one film type in the gallery ('Natural' is always shown as the starting point)."""
        self._look_filter = category
        for i, look in enumerate(looks.LOOKS):
            self.look_list.setRowHidden(i, category != 'all' and look.category not in (category, 'Digital'))
        self._layout_look_grid()

    def _layout_look_grid(self):
        """Look gallery is a fixed grid (no inner scrolling): size it to fit all cards."""
        width = self.look_list.viewport().width()
        cell = self.look_list.itemDelegate().SIZE
        cols = max(1, width // (cell.width() + 4))
        shown = sum(not self.look_list.isRowHidden(i) for i in range(self.look_list.count()))
        rows = -(-shown // cols)
        self.look_list.setFixedHeight(rows * (cell.height() + 4) + 8)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, self._layout_look_grid)
        root = self.centralWidget()
        if root is not None:
            self.drop_overlay.setGeometry(root.rect())
            if self.toast.isVisible():
                self.toast._place()

    # Settings <-> controls

    def _sync_controls(self):
        s = self.settings
        for field, w in self._controls.items():
            v = getattr(s, field)
            w.blockSignals(True)
            if isinstance(w, QComboBox):
                w.setCurrentText(v)
            else:
                w.set_value(v)
            w.blockSignals(False)
        look = looks.get(s.look)
        self.look_list.blockSignals(True)
        self.look_list.setCurrentRow(looks.names().index(look.name))
        self.look_list.blockSignals(False)
        self.look_desc.setText(look.description)
        self.color_section.setEnabled(not look.mono)
        self.bw_section.setEnabled(look.mono)
        self.color_section.setToolTip('' if not look.mono else 'Not used by black & white looks')
        self.bw_section.setToolTip('' if look.mono else 'Choose a black & white look to use these')
        self.lut_label.setText(lut_display_name(s.lut_path) if s.lut_path else 'None')
        self.lut_clear.setVisible(bool(s.lut_path))
        self.sl_lut_amount.setEnabled(bool(s.lut_path))
        self.aspect.blockSignals(True)
        self.aspect.setCurrentText(s.crop_aspect)
        self.aspect.blockSignals(False)
        self.orient_btn.setEnabled(s.crop_aspect not in ('Free', '1:1'))
        self.crop_btn.blockSignals(True)
        self.crop_btn.setChecked(self.crop_mode)
        self.crop_btn.blockSignals(False)
        self.auto_btn.blockSignals(True)
        self.auto_btn.setChecked(bool(s.auto_on))
        self.auto_btn.blockSignals(False)
        self.lens_mode.blockSignals(True)
        self.lens_mode.setCurrentIndex(max(0, self.lens_mode.findData(s.lens_mode)))
        self.lens_mode.blockSignals(False)
        for field, cb in self.lens_checks.items():
            cb.blockSignals(True)
            cb.setChecked(bool(getattr(s, field)))
            cb.setEnabled(s.lens_mode != 'off')
            cb.blockSignals(False)
        self._sync_source()

    def _set_field(self, field, value):
        if getattr(self.settings, field) == value:
            return
        setattr(self.settings, field, value)
        self._settings_changed()

    def _settings_changed(self):
        if self.current_key:
            self.photo_settings[self.current_key] = self.settings
            self._mark_edited(self.current_key)
        self.save_timer.start()
        self.undo_timer.start()
        self.request_render()
        self.canvas.mark_detail_stale()

    def _on_look_row(self, row):
        if row < 0:
            return
        name = looks.names()[row]
        if name == self.settings.look:
            return
        self.settings = self.settings.with_look_defaults(name)
        self._sync_controls()
        self._fade_next = True
        self._interactive = False
        self._settings_changed()
        self._interactive = False

    def _analysis_source(self, cropped=True):
        """(linear image, is_raw) of the current photo for the Auto tools, or None."""
        ph = self.current_photo()
        if ph is None or self._base_key(ph) not in self.bases:
            return None
        lin, is_raw, full, info = self.bases[self._base_key(ph)]
        if cropped and self.preview_context is not None and self.preview_context[0] == self._view_key(ph):
            return self.preview_context[1], is_raw     # what's inside the crop, lens-corrected
        return lin, is_raw

    AUTO_FIELDS = ('exposure', 'highlights', 'shadows', 'contrast', 'warmth', 'tint')

    def toggle_auto(self):
        """Auto on: remember the current values, apply Auto. Auto off: put the remembered values back."""
        if self.current_key is None:
            return
        s = self.settings
        if s.auto_on:
            try:
                prev = json.loads(s.auto_prev or '{}')
            except ValueError:
                prev = {}
            for f in self.AUTO_FIELDS:
                if f in prev:
                    setattr(s, f, prev[f])
            s.auto_on, s.auto_prev = False, ''
            self._sync_controls()
            self._interactive = False
            self._settings_changed()
            self.flash('Auto off: previous values restored.')
            log.info('Auto turned off on %s', self.current_photo().stem)
            return
        prev = {f: getattr(s, f) for f in self.AUTO_FIELDS}
        if self.auto_adjust(True, True):
            self.settings.auto_on = True
            self.settings.auto_prev = json.dumps(prev)
            self._sync_controls()
            self.save_timer.start()
            self.undo_timer.start()

    def auto_adjust(self, light=True, color=True):
        src = self._analysis_source()
        if src is None:
            self._sync_controls()
            return False
        sug = auto_mod.auto_light_color(*src)
        log.info('Auto (%s%s) on %s: %s', 'light' if light else '', ' colour' if color else '',
                 self.current_photo().stem, sug)
        fields = (('exposure', 'highlights', 'shadows', 'contrast') if light else ()) + \
                 (('warmth', 'tint') if color else ())
        for f in fields:
            setattr(self.settings, f, sug[f])
        self._sync_controls()
        self._interactive = False
        self._settings_changed()
        names = {'exposure': 'exposure %+.2f EV', 'highlights': 'highlights %+d', 'shadows': 'shadows %+d',
                 'contrast': 'contrast %+d', 'warmth': 'warmth %+d', 'tint': 'tint %+d'}
        parts = [names[f] % (sug[f] if f == 'exposure' else round(sug[f] * 100)) for f in fields if sug[f]]
        how = 'Click Auto again to undo.' if light and color else 'Ctrl+Z to undo.'
        self.flash('Auto: %s.  %s' % (', '.join(parts) if parts else 'this photo already looks balanced', how), 5000)
        return True

    def auto_straighten(self):
        src = self._analysis_source(cropped=False)
        if src is None:
            return
        lin, is_raw = src
        angle = auto_mod.auto_straighten(np.rot90(lin, -(self.settings.rot90 % 4)), is_raw)
        log.info('Auto-straighten on %s: %s', self.current_photo().stem, angle)
        if angle is None:
            self.flash('No clear horizon or straight lines found. In crop mode (R), Ctrl+drag along '
                       'something that should be level.', 6000)
            return
        if abs(angle - self.settings.angle) < 0.05:
            self.flash('Already level.')
            return
        self.set_angle(angle)
        self.flash('Straightened by %+.1f°.  Ctrl+Z to undo.' % angle)

    def revert(self):
        """Everything back to the photo as shot (look, adjustments, crop). Undoable."""
        if self.current_key is None:
            return
        if self.crop_mode:
            self.toggle_crop(False)
        keep_source = self.settings.source
        self.settings = pipeline.Settings()
        self.settings.source = keep_source
        self._sync_controls()
        self._settings_changed()
        self._geometry_changed()
        self.flash('Reverted to the original. Ctrl+Z to undo.')

    def reset_settings(self):
        """Back to Natural with no adjustments. Crop and the RAW/JPEG choice are kept."""
        keep = self.settings
        self.settings = pipeline.Settings().with_geometry_of(keep)
        self.settings.source = keep.source
        self._sync_controls()
        self._settings_changed()

    def apply_to_all(self):
        if not self.photos:
            return
        for ph in self.photos:
            if ph.key == self.current_key:
                continue
            # Each photo keeps its own crop and RAW/JPEG choice
            own = self.photo_settings.get(ph.key, pipeline.Settings())
            new = self.settings.with_geometry_of(own)
            new.source = own.source
            new.auto_on, new.auto_prev = False, ''
            self.photo_settings[ph.key] = new
        self.photo_settings[self.current_key] = self.settings
        for i in range(self.filmstrip.count()):
            self.filmstrip.item(i).setData(EDITED_ROLE, True)
        self.save_timer.start()
        self.flash('Applied “%s” settings to all %d photos' % (self.settings.look, len(self.photos)))

    def _fill_lut_menu(self):
        m = self.lut_menu
        m.clear()
        a = m.addAction('None', lambda: self._set_lut(''))
        a.setCheckable(True)
        a.setChecked(not self.settings.lut_path)
        m.addSeparator()
        for cat in LUT_CATEGORIES:
            sub = m.addMenu(cat)
            for c, title, path in bundled_luts():
                if c != cat:
                    continue
                a = sub.addAction(title, lambda p=path: self._set_lut(p))
                a.setCheckable(True)
                a.setChecked(path == self.settings.lut_path)
        m.addSeparator()
        if self.settings.lut_path and not self.settings.lut_path.startswith(str(BUNDLED_LUT_DIR)):
            a = m.addAction(lut_display_name(self.settings.lut_path))
            a.setCheckable(True)
            a.setChecked(True)
        m.addAction('Load from File…', self.choose_lut)

    def choose_lut(self):
        start = str(Path(self.settings.lut_path).parent) if self.settings.lut_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load LUT', start,
            'LUTs (*.cube *.CUBE *.png *.PNG *.tif *.tiff);;Cube LUT (*.cube *.CUBE);;Hald CLUT image (*.png *.tif *.tiff)')
        if path:
            try:
                Lut.load(path)
            except (LutError, OSError, ValueError) as e:
                QMessageBox.warning(self, 'Could not load LUT', str(e))
                return
            self._set_lut(path)

    def _set_lut(self, path):
        self.settings.lut_path = path
        self._sync_controls()
        self._settings_changed()

    # Undo / redo, copy / paste

    def _snapshot(self):
        key = self.current_key
        if key is None or self._restoring:
            return
        h = self.history.setdefault(key, [])
        if not h or h[-1] != self.settings:
            h.append(self.settings.copy())
            del h[:-200]
            self.future[key] = []
        self._update_undo_buttons()

    def _update_undo_buttons(self):
        key = self.current_key
        self.undo_btn.setEnabled(key is not None and len(self.history.get(key, [])) > 1)
        self.redo_btn.setEnabled(key is not None and bool(self.future.get(key)))

    def _restore(self, s):
        old_source = self.settings.source
        self._restoring = True
        self.settings = s.copy()
        self.photo_settings[self.current_key] = self.settings
        self._mark_edited(self.current_key)
        self._sync_controls()
        self.save_timer.start()
        if self.settings.source != old_source:
            self._on_current_photo(None, None)
        else:
            if self.crop_mode:
                self._push_crop_state()
            self.request_render()
            self.canvas.mark_detail_stale()
        self._restoring = False
        self._update_undo_buttons()

    def undo(self):
        if self.current_key is None:
            return
        if self.undo_timer.isActive():
            self.undo_timer.stop()
            self._snapshot()
        h = self.history.get(self.current_key, [])
        if len(h) < 2:
            self.flash('Nothing to undo')
            return
        self.future.setdefault(self.current_key, []).append(h.pop())
        self._restore(h[-1])

    def redo(self):
        f = self.future.get(self.current_key) if self.current_key else None
        if not f:
            self.flash('Nothing to redo')
            return
        s = f.pop()
        self.history.setdefault(self.current_key, []).append(s)
        self._restore(s)

    def copy_settings(self):
        if self.current_key is None:
            return
        self.clipboard = self.settings.without_geometry()
        self.flash('Copied “%s” settings. Select photos and press Ctrl+V to paste.' % self.settings.look)

    def paste_settings(self):
        if self.clipboard is None or not self.photos:
            return
        rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()
                       if not self.filmstrip.isRowHidden(self.filmstrip.row(it))})
        if not rows and self.filmstrip.currentRow() >= 0:
            rows = [self.filmstrip.currentRow()]
        for r in rows:
            ph = self.photos[r]
            own = self.photo_settings.get(ph.key) or (self.settings if ph.key == self.current_key else pipeline.Settings())
            new = self.clipboard.with_geometry_of(own)   # crop and RAW/JPEG choice stay with each photo
            new.source = own.source
            self.photo_settings[ph.key] = new
            self.filmstrip.item(r).setData(EDITED_ROLE, True)
            if ph.key == self.current_key:
                self.settings = new
                self._sync_controls()
                self._settings_changed()
        self.save_timer.start()
        self.flash('Pasted settings to %d photo%s' % (len(rows), '' if len(rows) == 1 else 's'))

    # Crop & straighten

    def _frame_info(self):
        """(frame_w, frame_h, ratio or None) for the current photo, or None if it isn't loaded yet."""
        ph = self.current_photo()
        if ph is None or self._base_key(ph) not in self.bases:
            return None
        src_full = self.bases[self._base_key(ph)][2]
        fh, fw = self.settings.geometry(*src_full).frame_size
        return fw, fh, geometry.aspect_ratio(self.settings.crop_aspect, fw, fh, self.settings.crop_portrait)

    def toggle_crop(self, on=None):
        on = (not self.crop_mode) if on is None else bool(on)
        if on and self._frame_info() is None:
            on = False
        if on == self.crop_mode:
            self.crop_btn.setChecked(on)
            return
        self.crop_mode = on
        self.crop_btn.setChecked(on)
        if on:
            self.compare_btn.setChecked(False)
            self.canvas.zoom_fit(animate=False)
            self.canvas.set_crop_view(None, None)
        self.canvas.set_crop_mode(on)
        self._push_crop_state()
        self.request_render()
        self.flash('Crop: drag the edges, Ctrl+drag along the horizon to straighten, Enter when done' if on else '')

    def _push_crop_state(self):
        info = self._frame_info()
        if info is None:
            return
        s = self.settings
        self.canvas.set_crop_state(s.angle, (s.crop_cx, s.crop_cy, s.crop_w, s.crop_h), info[2])

    def _geometry_changed(self, render=True):
        self._sync_controls()
        if self.current_key:
            self.photo_settings[self.current_key] = self.settings
            self._mark_edited(self.current_key)
        self.save_timer.start()
        self.undo_timer.start()
        if self.crop_mode:
            self._push_crop_state()
        elif render:
            self.request_render()

    def set_angle(self, angle):
        info = self._frame_info()
        if info is None:
            return
        fw, fh, ratio = info
        s = self.settings
        angle = round(min(45.0, max(-45.0, float(angle))), 2)
        was_full = geometry.is_max_crop(fw, fh, s.angle, s.crop_cx, s.crop_cy, s.crop_w, s.crop_h, ratio)
        if was_full:
            s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = geometry.reset_crop_for_angle(fw, fh, angle, ratio)
        else:
            s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = geometry.fit_inside(fw, fh, angle, s.crop_cx, s.crop_cy,
                                                                           s.crop_w, s.crop_h)
        s.angle = angle
        self._geometry_changed()

    def set_aspect(self, name):
        info = self._frame_info()
        s = self.settings
        if info is None or name == s.crop_aspect:
            return
        fw, fh, _ = info
        s.crop_aspect = name
        s.crop_portrait = fh > fw   # start in the photo's own orientation
        self._apply_aspect()

    def flip_aspect(self):
        self.settings.crop_portrait = not self.settings.crop_portrait
        self._apply_aspect()

    def _apply_aspect(self):
        info = self._frame_info()
        if info is None:
            return
        fw, fh, ratio = info
        s = self.settings
        if ratio is not None:
            s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = geometry.reset_crop_for_angle(fw, fh, s.angle, ratio)
        self._geometry_changed()

    def rotate(self, direction):
        s = self.settings
        s.rot90 = (s.rot90 + direction) % 4
        s.crop_portrait = not s.crop_portrait if s.crop_aspect not in ('Free', '1:1') else s.crop_portrait
        info = self._frame_info()
        if info is not None:
            fw, fh, ratio = info
            s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = geometry.reset_crop_for_angle(fw, fh, s.angle, ratio)
        self._geometry_changed()
        if self.crop_mode:
            self.request_render()   # the frame itself changed shape

    def reset_crop(self):
        s = self.settings
        s.angle = 0.0
        s.crop_aspect = 'Free'
        s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = 0.5, 0.5, 1.0, 1.0
        self._geometry_changed()

    def _on_canvas_crop(self, rect):
        s = self.settings
        s.crop_cx, s.crop_cy, s.crop_w, s.crop_h = (float(v) for v in rect)
        self._geometry_changed(render=False)

    # Filmstrip: filter, hide, delete

    def set_filter(self, mode):
        self.filter_mode = mode
        self.filter_btns[mode].setChecked(True)
        self._apply_filter()

    def _matches_filter(self, ph):
        return {'all': True, 'raw': ph.raw is not None, 'jpeg': ph.jpeg is not None,
                'marked': ph.key in self.marked}[self.filter_mode]

    def _apply_filter(self):
        shown = 0
        for i, ph in enumerate(self.photos):
            hide = not self._matches_filter(ph)
            self.filmstrip.setRowHidden(i, hide)
            if hide and self.filmstrip.item(i).isSelected():
                self.filmstrip.item(i).setSelected(False)
            shown += not hide
        total = len(self.photos)
        self.strip_count.setText(('%d photo%s' % (total, '' if total == 1 else 's')) if shown == total
                                 else '%d of %d photos' % (shown, total))
        row = self.filmstrip.currentRow()
        if (row < 0 or self.filmstrip.isRowHidden(row)) and shown:
            first = next(i for i in range(total) if not self.filmstrip.isRowHidden(i))
            self.filmstrip.setCurrentRow(first)

    def _update_strip_tab(self):
        self.strip_tab.setText(('▾  Filmstrip' if self.strip_visible else '▴  Filmstrip'))

    def _set_strip_visible(self, on):
        if on == self.strip_visible:
            return
        self.strip_visible = on
        self._update_strip_tab()
        full = self.strip_bar.sizeHint().height()
        self._strip_anim.stop()
        if on:
            self.strip_bar.setMaximumHeight(0)
            self.strip_bar.show()
            self._strip_anim.setStartValue(0)
            self._strip_anim.setEndValue(full)
        else:
            self._strip_anim.setStartValue(self.strip_bar.height())
            self._strip_anim.setEndValue(0)
        self._strip_anim.start()
        self.qsettings.setValue('strip_visible', on)

    def delete_photos(self, rows=None):
        if not self.photos or self.export_thread is not None:
            return
        if rows is None:
            rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()
                           if not self.filmstrip.isRowHidden(self.filmstrip.row(it))})
        if not rows and self.filmstrip.currentRow() >= 0:
            rows = [self.filmstrip.currentRow()]
        if not rows:
            return
        targets = [self.photos[r] for r in rows]
        pairs = any(ph.raw and ph.jpeg for ph in targets)
        n = len(targets)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle('Move to Trash')
        what = 'this photo' if n == 1 else 'these %d photos' % n
        box.setText('Move %s to the Trash?' % what)
        box.setInformativeText('You can restore files from the Trash in your file manager.'
                               + ('\nSome photos are RAW + image pairs: choose which files to remove.' if pairs else ''))
        box.setDetailedText('\n'.join(p.name for ph in targets for p in (ph.raw, ph.jpeg) if p))
        choices = {}
        if pairs:
            labels = {imageio.image_label(ph.jpeg) for ph in targets if ph.raw and ph.jpeg}
            img = labels.pop() if len(labels) == 1 else 'image'
            choices[box.addButton('RAW + %s' % img, QMessageBox.ButtonRole.DestructiveRole)] = 'both'
            choices[box.addButton('RAW only', QMessageBox.ButtonRole.DestructiveRole)] = 'raw'
            choices[box.addButton('%s only' % img, QMessageBox.ButtonRole.DestructiveRole)] = 'jpeg'
        else:
            choices[box.addButton('Move to Trash', QMessageBox.ButtonRole.DestructiveRole)] = 'both'
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        default = {'raw': 'raw', 'jpeg': 'jpeg'}.get(self.filter_mode, 'both') if pairs else 'both'
        for b, c in choices.items():
            if c == default:
                box.setDefaultButton(b)
        box.setEscapeButton(cancel)
        box.exec()
        choice = choices.get(box.clickedButton())
        if choice is None:
            return

        files = []
        for ph in targets:
            if ph.raw and choice in ('both', 'raw'):
                files.append(ph.raw)
            if ph.jpeg and choice in ('both', 'jpeg'):
                files.append(ph.jpeg)
        failed = {str(f) for f in files if not trash_file(f)}
        self._remove_deleted(targets, choice, failed)
        moved = len(files) - len(failed)
        self.flash('Moved %d file%s to the Trash' % (moved, '' if moved == 1 else 's'), 6000)
        if failed:
            QMessageBox.warning(self, 'Some files were not moved',
                                'These files could not be moved to the Trash and were left untouched:\n\n'
                                + '\n'.join(str(f) for f in failed))

    def _remove_deleted(self, targets, choice, failed):
        """Update the filmstrip in place after files went to the Trash: thumbnails, previews and
        edits of the remaining photos are kept (no folder rescan)."""
        current = self.current_photo()
        cur_row = self.filmstrip.currentRow()
        removed_current = False
        for ph in targets:
            if ph.raw and choice in ('both', 'raw') and str(ph.raw) not in failed:
                ph.raw = None
            if ph.jpeg and choice in ('both', 'jpeg') and str(ph.jpeg) not in failed:
                ph.jpeg = None
            for bk in [k for k in self.bases if k.startswith(ph.key + '|')]:
                del self.bases[bk]
            if self.full_base is not None and self.full_base[0].startswith(ph.key + '|'):
                self.full_base = None
            row = self.photos.index(ph)
            if ph.raw is None and ph.jpeg is None:          # nothing left of this photo
                self.filmstrip.blockSignals(True)
                self.filmstrip.takeItem(row)
                self.filmstrip.blockSignals(False)
                self.photos.pop(row)
                for store in (self.photo_settings, self.history, self.future):
                    store.pop(ph.key, None)
                self.marked.discard(ph.key)
                removed_current = removed_current or ph is current
            else:                                           # one file of a pair is left
                it = self.filmstrip.item(row)
                it.setData(KIND_ROLE, ph.kind)
                it.setToolTip('%s\n%s' % (ph.stem, ' + '.join(p.name for p in (ph.raw, ph.jpeg) if p)))
                s = self.photo_settings.get(ph.key)
                if s is not None:
                    s.source = 'raw' if ph.raw else 'jpeg'
        self._update_mark_count()
        self.save_timer.start()
        if not self.photos:
            self.stack.setCurrentIndex(0)
            return
        if removed_current:
            row = min(cur_row, len(self.photos) - 1)
            while row > 0 and self.filmstrip.isRowHidden(row):
                row -= 1
            self.filmstrip.setCurrentRow(-1)
            self.filmstrip.setCurrentRow(row)
        elif current is not None and current in targets:
            self._on_current_photo(None, None)              # same photo, different file now
        self._apply_filter()

    # Marking photos for export

    def _set_marked(self, rows, on):
        for r in rows:
            ph = self.photos[r]
            if on:
                self.marked.add(ph.key)
            else:
                self.marked.discard(ph.key)
            self.filmstrip.item(r).setData(MARKED_ROLE, on)
        self._update_mark_count()
        self.save_timer.start()
        if self.filter_mode == 'marked':
            self._apply_filter()

    def toggle_marks(self):
        """Mark the selected photos; if they're all marked already, unmark them."""
        rows = sorted({self.filmstrip.row(it) for it in self.filmstrip.selectedItems()})
        if not rows and self.filmstrip.currentRow() >= 0:
            rows = [self.filmstrip.currentRow()]
        if not rows:
            return
        all_marked = all(self.photos[r].key in self.marked for r in rows)
        self._set_marked(rows, not all_marked)

    def clear_marks(self):
        self._set_marked([i for i, ph in enumerate(self.photos) if ph.key in self.marked], False)

    def _update_mark_count(self):
        n = len(self.marked)
        self.mark_count.setText('✓ %d marked for export' % n if n else '')
        self.clear_marks_btn.setVisible(n > 0)

    def eventFilter(self, obj, ev):
        # Clicking the ✓ badge on a filmstrip thumbnail toggles its mark without changing the selection
        if obj is self.filmstrip.viewport() and ev.type() == QEvent.Type.MouseButtonPress \
                and ev.button() == Qt.MouseButton.LeftButton:
            pos = ev.position()
            idx = self.filmstrip.indexAt(pos.toPoint())
            if idx.isValid() and FilmstripDelegate.trash_rect(self.filmstrip.visualRect(idx)).contains(pos):
                self.delete_photos([idx.row()])
                return True
            if idx.isValid():
                c = FilmstripDelegate.mark_center(self.filmstrip.visualRect(idx))
                if (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2 <= (FilmstripDelegate.MARK_RADIUS + 3) ** 2:
                    row = idx.row()
                    self._set_marked([row], self.photos[row].key not in self.marked)
                    return True
        return super().eventFilter(obj, ev)

    # Folders & photos

    def choose_folder(self):
        start = self.folder or self.qsettings.value('last_folder', str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, 'Open folder or memory card', start)
        if d:
            self.open_folder(d)

    def open_folder(self, folder, select=None, quiet=False):
        self._save_session()
        if self.crop_mode:
            self.toggle_crop(False)
        photos = imageio.scan_folder(folder)
        if not photos:
            if not quiet:
                QMessageBox.information(self, APP_NAME, 'No RAW or JPEG photos found in\n%s' % folder)
            else:
                self.photos = []
                self.filmstrip.clear()
                self.stack.setCurrentIndex(0)
            return
        self.folder = str(folder)
        self.qsettings.setValue('last_folder', self.folder)
        kinds = {}
        for ph in photos:
            kinds[ph.kind] = kinds.get(ph.kind, 0) + 1
        log.info('Opened folder with %d photos: %s', len(photos), ', '.join('%d %s' % (n, k) for k, n in sorted(kinds.items())))
        self.photos = photos
        self.bases.clear()
        self.photo_settings = {}
        self.marked = set()
        self.look_thumb_key = None
        self._load_session()
        self._update_mark_count()

        self.folder_label.setText(self._pretty_path(folder))
        self.filmstrip.blockSignals(True)
        self.filmstrip.clear()
        for ph in photos:
            it = QListWidgetItem(ph.stem)
            it.setData(KIND_ROLE, ph.kind)
            it.setData(EDITED_ROLE, ph.key in self.photo_settings)
            it.setData(MARKED_ROLE, ph.key in self.marked)
            it.setToolTip('%s\n%s' % (ph.stem, ' + '.join(p.name for p in (ph.raw, ph.jpeg) if p)))
            self.filmstrip.addItem(it)
        self.filmstrip.blockSignals(False)
        self._thumb_stats = {'ok': 0, 'failed': 0, 'total': len(photos), 'start': time.time()}
        for i, ph in enumerate(photos):
            self.thumb_pool.start(Task(_load_thumb, ph, callback=lambda r, i=i, k=ph.key: self._on_thumb(i, k, r)))

        for b in (self.compare_btn, self.recipes_btn, self.export_btn):
            b.setEnabled(True)
        self.stack.setCurrentIndex(1)
        QTimer.singleShot(0, self._layout_look_grid)

        start = 0
        current = select or self._session_current
        if current:
            for i, ph in enumerate(photos):
                if ph.key == current:
                    start = i
        self.filmstrip.setCurrentRow(start)
        self._apply_filter()
        self.canvas.setFocus()

    @staticmethod
    def _pretty_path(p):
        home = str(Path.home())
        p = str(p)
        return '~' + p[len(home):] if p.startswith(home) else p

    def _on_thumb(self, i, key, result):
        st = getattr(self, '_thumb_stats', None)
        if st is not None:
            st['failed' if isinstance(result, Exception) else 'ok'] += 1
            if st['ok'] + st['failed'] == st['total']:
                log.info('Thumbnails: %d loaded, %d failed, %.1fs', st['ok'], st['failed'], time.time() - st['start'])
        if isinstance(result, Exception):
            return
        if not (i < len(self.photos) and self.photos[i].key == key):     # rows shift after deletions
            i = next((r for r, ph in enumerate(self.photos) if ph.key == key), -1)
            if i < 0:
                return
        self.filmstrip.item(i).setData(THUMB_ROLE, QPixmap.fromImage(result))
        if key == self.current_key and self.canvas.after is None and self.canvas.before is None:
            self.canvas.set_images(None, result)

    def _mark_edited(self, key):
        for i, ph in enumerate(self.photos):
            if ph.key == key:
                it = self.filmstrip.item(i)
                if not it.data(EDITED_ROLE):
                    it.setData(EDITED_ROLE, True)
                return

    def current_photo(self):
        row = self.filmstrip.currentRow()
        return self.photos[row] if 0 <= row < len(self.photos) else None

    def _on_current_photo(self, item, _prev):
        ph = self.current_photo()
        if ph is None:
            return
        if self.crop_mode:
            self.toggle_crop(False)
        if self.undo_timer.isActive():   # record the last edit on the photo we're leaving
            self.undo_timer.stop()
            self._snapshot()
        self.current_key = ph.key
        if ph.key in self.photo_settings:
            self.settings = self.photo_settings[ph.key]
        else:
            # A new photo inherits the current look, but not the crop, source choice or Auto state
            self.settings = self.settings.without_geometry()
            self.settings.source = 'raw'
        self._sync_controls()
        self.photo_title.setText(ph.stem)
        self.history.setdefault(ph.key, [self.settings.copy()])
        self.future.setdefault(ph.key, [])
        self._update_undo_buttons()
        self.photo_camera.setText('')
        self.load_pool.start(Task(_camera_line, ph, self._source(ph),
                                  callback=lambda r, k=ph.key: k == self.current_key and not isinstance(r, Exception)
                                  and self.photo_camera.setText(r)))

        thumb = self.filmstrip.item(self.filmstrip.currentRow()).data(THUMB_ROLE)
        bk = self._base_key(ph)
        if self.full_base is not None and self.full_base[0] != bk:
            self.full_base = None  # free the previous photo's full-resolution data
        self.region_pending = {'detail': None, 'loupe': None}
        self.preview_context = None
        if self.canvas.photo_key is None or not self.canvas.photo_key.startswith(bk + '|'):
            # Show the thumbnail until the preview is developed
            self.canvas.set_photo(self._view_key(ph) + '|thumb', (thumb.height(), thumb.width()) if thumb else None)
            self.canvas.set_images(None, thumb.toImage() if thumb else None)
        self._ensure_base(ph)

    def _settings_of(self, ph):
        """The settings a photo will be developed with (untouched photos inherit the current look)."""
        if ph.key in self.photo_settings:
            return self.photo_settings[ph.key]
        if ph.key == self.current_key:
            return self.settings
        s = self.settings.without_geometry()
        s.source = 'raw'
        return s

    def _source(self, ph):
        """The file to develop: automatic from what exists; for RAW+JPEG pairs, the photo's choice."""
        return ph.source(self._settings_of(ph).source != 'jpeg')

    def _base_key(self, ph):
        return ph.key + '|' + str(self._source(ph))

    def _view_key(self, ph):
        """Identifies the cropped output: changes when the source or geometry changes."""
        s = self._settings_of(ph)
        return self._base_key(ph) + '|' + repr(tuple(getattr(s, f) for f in
                                                     pipeline.GEOMETRY_FIELDS[:6] + pipeline.LENS_FIELDS
                                                     + ('lens_vig_amount',)))

    def _on_lens_mode(self, _):
        self._set_field('lens_mode', self.lens_mode.currentData())
        ph = self.current_photo()
        if ph is not None and self._base_key(ph) in self.bases:
            _, is_raw, _, info = self.bases[self._base_key(ph)]
            self._update_lens_label(info, is_raw)

    def _update_lens_label(self, info, is_raw):
        if not lens_mod.available():
            text = 'Lens corrections unavailable (lensfunpy not installed)'
        elif info is None:
            text = 'No lens profile for this camera/lens'
        else:
            text = info.label
            if self.settings.lens_mode == 'auto' and not is_raw:
                text += '  ·  JPEG: already corrected by the camera'
            elif self.settings.lens_mode == 'on' and not is_raw:
                text = '⚠ Camera JPEGs are usually already corrected; On may over-correct'
                self.lens_label.setStyleSheet('color: %s;' % style.ACCENT)
                self.lens_label.setText(text)
                self.lens_label.setToolTip('This photo is a JPEG from the camera, which normally applies its own\n'
                                           'lens corrections. Use Auto unless you know it was saved uncorrected.')
                return
        self.lens_label.setStyleSheet('')
        self.lens_label.setText(text)
        self.lens_label.setToolTip(text if info is None else '%s on %s %s' % (info.label, info.cam_maker, info.cam_model))

    def _sync_source(self):
        ph = self.current_photo()
        if ph is None:
            return
        src = self._source(ph)
        is_raw = imageio.is_raw(src)
        if ph.raw and ph.jpeg:
            label = imageio.image_label(ph.jpeg)
            self.photo_info.setText('RAW + %s pair' % label)
            self.src_jpeg_btn.setText(label)
            self.source_row.show()
            self.src_raw_btn.setChecked(is_raw)
            self.src_jpeg_btn.setChecked(not is_raw)
        else:
            self.photo_info.setText('%s file  ·  %s' % (imageio.image_label(src), src.suffix.upper().lstrip('.')))
            self.source_row.hide()

    def set_source(self, which):
        ph = self.current_photo()
        if ph is None or self.settings.source == which or not (ph.raw and ph.jpeg):
            self._sync_source()
            return
        self.settings.source = which
        self._settings_changed()
        self._on_current_photo(None, None)

    def _ensure_base(self, ph):
        bk = self._base_key(ph)
        if bk in self.bases:
            self.bases.move_to_end(bk)
            self._base_ready(ph)
            return
        self.canvas.set_busy(True)
        self.status('Loading %s…' % self._source(ph).name)
        self.quick_pool.clear()
        self.quick_pool.start(Task(_quick_preview, ph, self._source(ph), self.settings.copy(),
                                   callback=lambda r, bk=bk, vk=self._view_key(ph): self._on_quick(bk, vk, r)))
        self.load_pool.start(Task(_load_base, ph, self._source(ph),
                                  callback=lambda r, ph=ph, bk=bk: self._on_base(ph, bk, r)))

    def _on_quick(self, bk, vk, result):
        """Show the instant preview, unless the real one is already here."""
        ph = self.current_photo()
        if isinstance(result, Exception) or result is None or ph is None or self._base_key(ph) != bk \
                or bk in self.bases or self._view_key(ph) != vk:
            return
        after, before, out_size = result
        self.canvas.set_photo(vk + '|quick', out_size)
        self.canvas.set_images(after, before)

    def _on_base(self, ph, bk, result):
        if isinstance(result, Exception):
            self.canvas.set_busy(False)
            self.status('')
            if ph.key == self.current_key:
                QMessageBox.warning(self, 'Could not open photo', '%s\n\n%s' % (self._source(ph), result))
            return
        self.bases[bk] = result
        while len(self.bases) > 6:
            self.bases.popitem(last=False)
        self.status('')
        if ph.key == self.current_key and bk == self._base_key(ph):
            self._base_ready(ph)
        self._preload_neighbours()

    def _preload_neighbours(self):
        row = self.filmstrip.currentRow()
        for r in (row + 1, row - 1):
            if 0 <= r < len(self.photos):
                ph = self.photos[r]
                bk = self._base_key(ph)
                if bk not in self.bases and bk not in self._preloading:
                    self._preloading.add(bk)
                    self.load_pool.start(Task(_load_base, ph, self._source(ph),
                                              callback=lambda res, bk=bk: self._on_preloaded(bk, res)))

    def _on_preloaded(self, bk, result):
        self._preloading.discard(bk)
        if not isinstance(result, Exception):
            self.bases[bk] = result
            while len(self.bases) > 6:
                self.bases.popitem(last=False)

    def _base_ready(self, ph):
        # Progressive: a fast half-resolution render first, full quality right after
        self._interactive = True
        self._progressive = True
        self.request_render()
        self._interactive = True
        if self.look_thumb_key != self._base_key(ph):
            self.look_timer.start()   # after the main preview has had its turn

    def _start_look_thumbs(self):
        ph = self.current_photo()
        if ph is None or self._base_key(ph) not in self.bases:
            return
        bk = self._base_key(ph)
        if self.look_thumb_key == bk:
            return
        lin, is_raw, _, _ = self.bases[bk]
        self.look_thumb_key = bk
        self.look_pool.clear()
        self.look_pool.start(Task(_render_look_thumbs, lin, is_raw, self.settings.copy(),
                                  callback=lambda r, k=bk: self._on_look_thumbs(k, r)))

    def _on_look_thumbs(self, key, result):
        if isinstance(result, Exception) or key != self.look_thumb_key:
            return
        for i, img in enumerate(result):
            self.look_list.item(i).setData(THUMB_ROLE, QPixmap.fromImage(img))

    # Rendering

    def request_render(self):
        """Re-render soon. Rapid requests (a slider being dragged) get fast half-resolution drafts,
        followed by a full-quality render once you pause."""
        now = time.monotonic()
        self._interactive = (now - self._last_request) < 0.35 or getattr(self, '_progressive', False)
        self._last_request = now
        self.final_timer.stop()
        self.render_timer.start()

    def _start_render(self):
        ph = self.current_photo()
        if ph is None or self._base_key(ph) not in self.bases:
            return
        if self.render_running:
            self.render_pending = True
            return
        self.render_running = True
        self.canvas.set_busy(True)
        self._render_started = time.time()
        lin, is_raw, full_size, info = self.bases[self._base_key(ph)]
        self._update_lens_label(info, is_raw)
        if self.crop_mode:
            self.preview_pool.start(Task(_render_crop_view, lin, is_raw, self.settings.copy(), full_size, info,
                                         callback=lambda r, k=self._view_key(ph): self._on_crop_view(k, r)))
            return
        self._last_render_draft = draft = self._interactive
        self.preview_pool.start(Task(_render_preview, lin, is_raw, self.settings.copy(), full_size, draft, info,
                                     self._base_key(ph),
                                     callback=lambda r, k=self._view_key(ph): self._on_rendered(k, r)))

    def _final_render(self):
        self._interactive = False
        self._start_render()

    def _on_crop_view(self, key, result):
        self.render_running = False
        if isinstance(result, Exception):
            self.status('Render error: %s' % result)
        elif self.crop_mode:
            image, frame_size = result
            self.canvas.set_crop_view(image, frame_size)
            self._push_crop_state()
        if self.render_pending:
            self.render_pending = False
            self._start_render()
        else:
            self.canvas.set_busy(False)

    def _on_rendered(self, key, result):
        self.render_running = False
        ph = self.current_photo()
        if isinstance(result, Exception):
            self.canvas.set_busy(False)
            self.status('Render error: %s' % result)
            log.error('Preview render failed: %r', result)
        elif ph is not None and key == self._view_key(ph) and not self.crop_mode:
            after, before, lin, out_size, (hist, clip) = result
            took = time.time() - getattr(self, '_render_started', time.time())
            if took > 1.0:
                log.warning('Slow preview render: %.2fs (%s, look %s)', took, ph.stem, self.settings.look)
            new_photo = self.canvas.photo_key not in (key, key + '|quick')
            if self.canvas.photo_key == key + '|quick':
                self._fade_next = True
            self.canvas.set_photo(key, out_size)
            self.canvas.set_images(after, before, fade=self._fade_next and not new_photo)
            self._fade_next = False
            self.preview_context = (key, lin)
            self.histogram.set_data(hist, clip)
        if self.render_pending:
            self.render_pending = False
            self._start_render()
        else:
            self.canvas.set_busy(False)
            if self._last_render_draft:
                self.final_timer.start(20 if getattr(self, '_progressive', False) else 280)
                self._progressive = False

    # Full-resolution regions (zoom & loupe)

    def _zoom_anchor(self):
        """Zoom around the mouse if it's over the photo, else the centre."""
        m = self.canvas.mouse
        return m if m is not None and self.canvas.rect().contains(m.toPoint()) else None

    def _request_region(self, kind, rect, scale):
        ph = self.current_photo()
        if ph is None:
            return
        bk = self._base_key(ph)
        self.region_pending[kind] = (bk, rect, scale)
        if self.full_base is None or self.full_base[0] != bk:
            if self.full_loading != bk:
                self.full_loading = bk
                self.canvas.set_loading_full(True)
                self.load_pool.start(Task(_load_full, self._source(ph),
                                          callback=lambda r, bk=bk: self._on_full_loaded(bk, r)))
            return
        self._start_region(kind)

    def _on_full_loaded(self, bk, result):
        if self.full_loading == bk:
            self.full_loading = None
        self.canvas.set_loading_full(False)
        if isinstance(result, Exception):
            self.status('Could not load full resolution: %s' % result)
            return
        ph = self.current_photo()
        if ph is None or self._base_key(ph) != bk:
            return  # user moved on
        self.full_base = (bk, result[0], result[1])
        for kind in ('detail', 'loupe'):
            self._start_region(kind)

    def _start_region(self, kind):
        req = self.region_pending[kind]
        if req is None or self.region_running[kind] or self.full_base is None:
            return
        bk, rect, scale = req
        ph = self.current_photo()
        if ph is None or bk != self.full_base[0] or bk != self._base_key(ph):
            return
        vk = self._view_key(ph)
        if self.preview_context is None or self.preview_context[0] != vk or self.canvas.photo_key != vk:
            return  # the developed preview for this crop isn't ready yet; it re-requests when it is
        self.region_pending[kind] = None
        self.region_running[kind] = True
        _, lin, is_raw = self.full_base
        self.region_pools[kind].start(Task(_render_region, vk, lin, is_raw, self.settings.copy(), rect, scale,
                                           self.preview_context[1], self.bases[bk][3],
                                           callback=lambda r, kind=kind: self._on_region(kind, r)))

    def _on_region(self, kind, result):
        self.region_running[kind] = False
        if isinstance(result, Exception):
            self.status('Render error: %s' % result)
        elif result is not None:
            if kind == 'detail':
                self.canvas.set_detail(result)
            else:
                self.canvas.set_loupe_region(result)
        self._start_region(kind)

    # Keyboard / drag & drop

    def keyPressEvent(self, e):
        if self.stack.currentIndex() == 1:
            if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
                self.canvas.set_show_original(True)
                return
            if e.key() in (Qt.Key.Key_Right, Qt.Key.Key_Left) and not self.crop_mode:
                step = 1 if e.key() == Qt.Key.Key_Right else -1
                row = self.filmstrip.currentRow() + step
                while 0 <= row < self.filmstrip.count() and self.filmstrip.isRowHidden(row):
                    row += step
                if 0 <= row < self.filmstrip.count():
                    self.filmstrip.setCurrentRow(row)
                return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space and not e.isAutoRepeat():
            self.canvas.set_show_original(False)
            return
        super().keyReleaseEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()
            self.drop_overlay.setGeometry(self.centralWidget().rect())
            self.drop_overlay.show()
            self.drop_overlay.raise_()

    def dragLeaveEvent(self, e):
        self.drop_overlay.hide()

    def dropEvent(self, e):
        self.drop_overlay.hide()
        for u in e.mimeData().urls():
            if u.isLocalFile():
                p = Path(u.toLocalFile())
                self.open_folder(p if p.is_dir() else p.parent)
                return

    # Export

    def export_dialog(self):
        if not self.photos or self.export_thread is not None:
            return
        selected = [self.photos[self.filmstrip.row(it)] for it in self.filmstrip.selectedItems()
                    if not self.filmstrip.isRowHidden(self.filmstrip.row(it))]
        shown = [ph for i, ph in enumerate(self.photos) if not self.filmstrip.isRowHidden(i)]
        if len(selected) <= 1 and self.current_photo():
            # With at most one highlighted photo, export the one on screen
            selected = [self.current_photo()]
        folder_id = hashlib.sha1(str(Path(self.folder).resolve()).encode()).hexdigest()[:12]
        default_out = self.qsettings.value('export_dir/' + folder_id, '') or str(Path(self.folder) / 'film-lab')
        marked = [ph for ph in self.photos if ph.key in self.marked]
        first = selected[0] if selected else self.current_photo()

        def example(fmt, with_suffix, ph=first):
            return export.output_path(ph, self._settings_of(ph), '', with_suffix, fmt).name

        dlg = ExportDialog(self, default_out, len(selected), len(marked), len(shown),
                           self.qsettings.value('quality', 95, type=int),
                           self.qsettings.value('suffix', True, type=bool), filtered=self.filter_mode != 'all',
                           example=example, current_name=first.stem if first else '')
        dlg.size.setCurrentIndex(self.qsettings.value('export_size', 0, type=int))
        dlg.fmt.setCurrentIndex(self.qsettings.value('export_format', 0, type=int))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        out_dir = dlg.path.text().strip()
        quality = int(dlg.quality.value())
        long_edge = int(dlg.size.currentData() or 0)
        fmt = dlg.fmt.currentData() or 'jpeg'
        self.qsettings.setValue('export_size', dlg.size.currentIndex())
        self.qsettings.setValue('export_format', dlg.fmt.currentIndex())
        suffix = dlg.suffix.isChecked()
        self.qsettings.setValue('export_dir/' + folder_id, out_dir)
        self.qsettings.setValue('quality', quality)
        self.qsettings.setValue('suffix', suffix)

        if dlg.rb_marked.isChecked():
            targets = marked
        elif dlg.rb_selected.isChecked():
            targets = selected
        else:
            targets = shown
        jobs = []
        for ph in targets:
            s = self._settings_of(ph)
            jobs.append((ph, s.to_dict(), export.output_path(ph, s, out_dir, suffix, fmt)))
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.warning(self, 'Cannot write there', str(e))
            return

        # Leave two cores for the UI; each band needs roughly 100-300 MB
        workers = max(1, min(4, (os.cpu_count() or 2) - 2, available_ram_gb()))
        self.export_thread = ExportThread(jobs, quality, workers, long_edge, fmt)
        self.export_thread.progress.connect(self._on_export_progress)
        self.export_thread.finished_all.connect(lambda ok, errs, c, d=out_dir: self._on_export_done(ok, errs, c, d))
        self._export_started = time.time()
        log.info('Export started: %d photos as %s, size %s, quality %d, %d worker(s), to %s',
                 len(jobs), fmt.upper(), ('%d px' % long_edge) if long_edge else 'full', quality, workers, out_dir)
        self.export_btn.setEnabled(False)
        self.toast.progress('Exporting %d photo%s…' % (len(jobs), '' if len(jobs) == 1 else 's'), 0,
                            len(jobs) * 1000, self._cancel_export)
        self.export_thread.start(QThread.Priority.LowPriority)

    def _on_export_progress(self, done, total, path, err):
        if self.export_thread is None or self.export_thread.cancelled:
            return
        self.toast.progress('Exporting  %d of %d' % (min(done // 1000 + 1, total // 1000), total // 1000),
                            done, total, self._cancel_export)

    def _cancel_export(self):
        if self.export_thread and not self.export_thread.cancelled:
            self.export_thread.cancelled = True
            log.info('Export cancel requested')
            self.toast.progress('Cancelling…', self.toast.bar.value(), self.toast.bar.maximum(), None)

    def _on_export_done(self, ok, errors, cancelled, out_dir):
        log.info('Export finished: %d ok, %d failed%s, %.1fs', ok, len(errors), ' (cancelled)' if cancelled else '',
                 time.time() - getattr(self, '_export_started', time.time()))
        for e in errors:
            log.error('Export failed: %s', e)
        self.export_thread.wait()
        self.export_thread = None
        self.export_btn.setEnabled(True)
        msg = '%s %d photo%s to %s' % ('Cancelled after exporting' if cancelled else 'Exported', ok,
                                       '' if ok == 1 else 's', self._pretty_path(out_dir))
        self.flash(msg, 8000)
        if errors:
            box = QMessageBox(self)
            box.setWindowTitle('Some photos failed')
            box.setText('%d photo%s could not be exported.' % (len(errors), '' if len(errors) == 1 else 's'))
            box.setDetailedText('\n'.join(errors))
            box.exec()

    # Recipes

    def _fill_recipes_menu(self):
        m = self.recipes_menu
        m.clear()
        m.addAction('Copy Settings\tCtrl+C', self.copy_settings)
        a = m.addAction('Paste Settings to Selected Photos\tCtrl+V', self.paste_settings)
        a.setEnabled(self.clipboard is not None)
        m.addSeparator()
        m.addAction('Save Current Settings as Recipe…', self.save_recipe)
        m.addSeparator()
        recipes = sorted((DATA_DIR / 'recipes').glob('*.json')) if (DATA_DIR / 'recipes').is_dir() else []
        if not recipes:
            a = m.addAction('No saved recipes yet')
            a.setEnabled(False)
        for p in recipes:
            m.addAction(p.stem, lambda p=p: self.load_recipe(p))
        if recipes:
            m.addSeparator()
            delete = m.addMenu('Delete Recipe')
            for p in recipes:
                delete.addAction(p.stem, lambda p=p: self.delete_recipe(p))

    def save_recipe(self):
        name, ok = QInputDialog.getText(self, 'Save recipe', 'Recipe name:', text=self.settings.look)
        name = name.strip().replace('/', '-')
        if not ok or not name:
            return
        d = DATA_DIR / 'recipes'
        d.mkdir(parents=True, exist_ok=True)
        recipe = self.settings.without_geometry().to_dict()
        recipe.pop('source', None)
        (d / (name + '.json')).write_text(json.dumps(recipe, indent=2))
        self.flash('Saved recipe “%s”' % name)

    def load_recipe(self, path):
        try:
            loaded = pipeline.Settings.from_dict(json.loads(Path(path).read_text()))
            loaded = loaded.with_geometry_of(self.settings)   # a recipe never changes the crop
            loaded.source = self.settings.source
            self.settings = loaded
        except (OSError, ValueError, TypeError) as e:
            QMessageBox.warning(self, 'Could not load recipe', str(e))
            return
        if self.settings.lut_path and not Path(self.settings.lut_path).is_file():
            self.settings.lut_path = ''
        self._sync_controls()
        self._settings_changed()
        self.flash('Applied recipe “%s”' % Path(path).stem)

    def delete_recipe(self, path):
        if QMessageBox.question(self, 'Delete recipe', 'Delete recipe “%s”?' % Path(path).stem) \
                == QMessageBox.StandardButton.Yes:
            Path(path).unlink(missing_ok=True)

    # Sessions (per-folder edits survive restarts)

    def _session_file(self):
        h = hashlib.sha1(str(Path(self.folder).resolve()).encode()).hexdigest()[:16]
        return DATA_DIR / 'sessions' / (h + '.json')

    def _save_session(self):
        if not self.folder:
            return
        data = {'folder': self.folder, 'current': self.current_key, 'last': self.settings.to_dict(),
                'photos': {k: s.to_dict() for k, s in self.photo_settings.items()},
                'marked': sorted(self.marked)}
        f = self._session_file()
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix('.tmp')
            tmp.write_text(json.dumps(data))
            tmp.replace(f)
        except OSError:
            pass

    def _load_session(self):
        """Restore this folder's saved edits. An unreadable file is renamed to .bad and logged;
        entries are read individually so one bad entry does not discard the rest."""
        self._session_current = None
        f = self._session_file()
        if not f.exists():
            return
        try:
            data = json.loads(f.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('not a session object')
        except Exception as e:
            log.warning('Could not read saved edits %s (%r); starting fresh for this folder', f.name, e)
            try:
                f.replace(f.with_suffix('.bad'))
            except OSError:
                pass
            return
        keys = {ph.key for ph in self.photos}
        bad = 0
        photos = data.get('photos', {})
        for k, v in (photos.items() if isinstance(photos, dict) else []):
            if k not in keys:
                continue
            try:
                self.photo_settings[k] = self._clean_settings(pipeline.Settings.from_dict(v))
            except Exception:
                bad += 1
        try:
            self.marked = {k for k in data.get('marked', []) if k in keys}
        except Exception:
            self.marked = set()
        try:
            self.settings = self._clean_settings(pipeline.Settings.from_dict(data.get('last', {})))
        except Exception:
            self.settings = pipeline.Settings()
        self._session_current = data.get('current') if isinstance(data.get('current'), str) else None
        log.info('Restored saved edits for %d photos%s', len(self.photo_settings),
                 ' (%d unreadable entries skipped)' % bad if bad else '')

    @staticmethod
    def _clean_settings(s):
        """Repair values a saved file may hold from older versions or other computers."""
        defaults = pipeline.Settings()
        for f in dataclasses.fields(pipeline.Settings):
            v, d = getattr(s, f.name), getattr(defaults, f.name)
            if isinstance(d, bool):
                ok = isinstance(v, bool)
            elif isinstance(d, (int, float)):
                ok = isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
            else:
                ok = isinstance(v, type(d))
            if not ok:
                setattr(s, f.name, d)
        if s.lut_path and not Path(s.lut_path).is_file():
            # e.g. a bundled LUT path from a previous run of the .exe (it unpacks to a new temp folder)
            name = Path(s.lut_path).name
            match = next((p for _, t, p in bundled_luts() if Path(p).name == name), '')
            s.lut_path = match
        if s.look not in looks.names():
            s.look = 'Natural'
        return s

    # Misc

    def status(self, text):
        """Transient progress text: shown on the photo itself ("DEVELOPING…"), so only errors need a toast."""
        if text and ('error' in text.lower() or 'could not' in text.lower()):
            self.toast.message(text, 6000)

    def flash(self, text, ms=3500):
        if text:
            self.toast.message(text, ms)

    def copy_diagnostics(self):
        ph = self.current_photo()
        current = ''
        if ph is not None:
            current = '\nCurrent photo: %s (%s)\n%s\nLens: %s\nSettings: %s' % (
                ph.stem, ph.kind, self.photo_camera.text(), self.lens_label.text(),
                {k: v for k, v in self.settings.to_dict().items() if v != getattr(pipeline.Settings, k, None)})
        text = '%s\n%s\n\n--- last log lines ---\n%s' % (log_mod.system_info(), current, log_mod.tail())
        QApplication.clipboard().setText(text)
        log.info('Diagnostic info copied to clipboard')
        self.flash('Diagnostic info copied. Paste it into a message (Ctrl+V).', 5000)

    def open_log_folder(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        log_mod.LOG_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(log_mod.LOG_DIR)))

    def show_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('About Film Lab')
        dlg.setMinimumWidth(560)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(30, 26, 30, 22)
        lay.setSpacing(6)
        icon = QLabel()
        icon.setPixmap(app_icon().pixmap(72, 72))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(icon)
        t = QLabel('Film Lab %s' % __version__)
        t.setObjectName('emptyTitle')
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(t)
        by = QLabel('Created by <b>Fishbone</b>')
        by.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(by)
        lay.addSpacing(12)
        credits = QLabel(
            'Film looks for RAW and image files.<br><br>'
            '<b>Built with</b><br>'
            'Film emulation profiles: RawTherapee Film Simulation Collection<br>'
            'by Pat David, Pavlov Dmitry and Michael Ezra (CC BY-SA 4.0)<br>'
            'Lens corrections: lensfun database (CC BY-SA 3.0, LGPL) via lensfunpy<br>'
            'RAW decoding: LibRaw via rawpy · Images: Pillow, pillow-heif<br>'
            'Interface: Qt via PyQt6 · Maths: NumPy<br><br>'
            '<span style="color:#9399a3">Film and camera names identify what each profile approximates;<br>'
            'there is no affiliation with their makers.</span><br><br>'
            '© 2026 Fishbone · Free software under the GNU GPL v3 or later')
        credits.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credits.setWordWrap(True)
        lay.addWidget(credits)
        lay.addSpacing(10)
        close = QPushButton('Close')
        close.setObjectName('primary')
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignCenter)
        dlg.exec()

    def show_shortcuts(self):
        rows = [('Open folder', 'Ctrl+O'), ('Export', 'Ctrl+E'), ('Previous / next photo', '← / →'),
                ('Zoom', 'Scroll, + / −'), ('Fit · 100% · 200%', '0 · 1 · 2'), ('Toggle Fit / 100%', 'Double-click'),
                ('Loupe (Shift+scroll: magnification)', 'L'), ('Before / after split', 'C'),
                ('Show original', 'Hold Space'), ('Crop & straighten', 'R  (Enter to finish)'),
                ('Straighten along a line', 'Ctrl+drag in crop'), ('Undo · Redo', 'Ctrl+Z · Ctrl+Shift+Z'),
                ('Copy · Paste settings', 'Ctrl+C · Ctrl+V'), ('Apply to all photos', 'Ctrl+Shift+V'),
                ('Revert to original', 'Ctrl+R'), ('Auto light & colour', 'A'), ('Mark for export', 'P'), ('Move to Trash', 'Delete'),
                ('Show / hide filmstrip', 'Ctrl+B'), ('Reset a slider', 'Double-click it'),
                ('Type an exact value', 'Click the number')]
        dlg = QDialog(self)
        dlg.setWindowTitle('Keyboard shortcuts')
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(26, 22, 26, 22)
        title = QLabel('Keyboard shortcuts')
        title.setObjectName('title')
        lay.addWidget(title)
        lay.addSpacing(8)
        from PyQt6.QtWidgets import QGridLayout
        grid = QGridLayout()
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(7)
        for i, (what, keys) in enumerate(rows):
            a = QLabel(what)
            b = QLabel(keys)
            b.setObjectName('kbd')
            grid.addWidget(a, i, 0)
            grid.addWidget(b, i, 1)
        lay.addLayout(grid)
        lay.addSpacing(10)
        close = QPushButton('Close')
        close.setObjectName('primary')
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _restore_window(self):
        geo = self.qsettings.value('geometry')
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1440, 920)

    def closeEvent(self, e):
        if self.export_thread is not None:
            if QMessageBox.question(self, 'Export in progress', 'An export is running. Quit anyway?') \
                    != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.export_thread.cancelled = True
            self.export_thread.wait()
        self._save_session()
        self.qsettings.setValue('geometry', self.saveGeometry())
        self._stop_background_work()
        log.info('Film Lab closed')
        super().closeEvent(e)

    def _stop_background_work(self):
        """Drop queued background tasks and wait for running ones. Qt aborts if the interpreter
        shuts down while a pool thread is still executing Python code."""
        for t in (self.render_timer, self.final_timer, self.look_timer, self.save_timer, self.undo_timer):
            t.stop()
        pools = [self.preview_pool, self.load_pool, self.thumb_pool, self.quick_pool, self.look_pool,
                 *self.region_pools.values()]
        for pool in pools:
            pool.clear()
        for pool in pools:
            pool.waitForDone(5000)


def available_ram_gb():
    """Available physical memory in GB (4 if it can't be determined)."""
    if os.name == 'nt':
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                        ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                        ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys // (1024 ** 3))
        return 4
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // (1024 * 1024)
    except (OSError, ValueError):
        pass
    return 4


def app_icon():
    """Amber rounded square with film perforations, drawn at runtime."""
    pm = QPixmap(256, 256)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(style.ACCENT))
    p.drawRoundedRect(8, 8, 240, 240, 52, 52)
    p.setBrush(QColor('#1a1306'))
    p.drawRoundedRect(36, 64, 184, 128, 14, 14)
    p.setBrush(QColor(style.ACCENT))
    for i in range(6):
        x = 48 + i * 28
        p.drawRoundedRect(x, 72, 16, 14, 3, 3)
        p.drawRoundedRect(x, 170, 16, 14, 3, 3)
    p.setBrush(QColor(style.ACCENT_HOVER))
    p.drawRoundedRect(58, 96, 140, 64, 8, 8)
    p.end()
    return QIcon(pm)


def _selftest(folder, out_dir):
    """Self-test for packaged builds: export every photo in `folder` with a film look using the same
    threaded band renderer as the app, and write a report. Returns an exit code."""
    lines, results = [], []
    try:
        photos = imageio.scan_folder(folder)
        s = pipeline.Settings().with_look_defaults('Kodak Portra 400')
        s.nr_color, s.lut_path = 0.3, bundled_luts()[0][2]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3, initializer=export.lower_thread_priority) as pool:
            results = []
            for ph in photos:
                t0 = time.time()
                results.append(export.export_photo(ph, s.to_dict(), str(Path(out_dir) / (ph.stem + '.jpg')), 90,
                                                   submit=pool.submit))
                log.info('Self-test exported %s in %.1fs', ph.stem, time.time() - t0)
        for ph, (p, e) in zip(photos, results):
            info = lens_mod.identify(imageio.read_exif(ph, ph.source(True)))
            lines.append('%s %s  [lens: %s]' % (Path(p).name, e or 'OK', info.label if info else 'none'))
        ok = bool(results) and all(e is None for _, e in results)
    except Exception as e:  # recorded in the report
        lines.append('FAILED: %r' % e)
        ok = False
    lines.append('RESULT: %s (%d photos, %d LUTs)' % ('PASS' if ok else 'FAIL', len(results), len(bundled_luts())))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / 'selftest.txt').write_text('\n'.join(lines), encoding='utf-8')
    return 0 if ok else 1


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--selftest':
        log_mod.setup()
        log.info('Self-test: %s -> %s', sys.argv[2], sys.argv[3])
        return _selftest(sys.argv[2], sys.argv[3])
    QApplication.setApplicationName(APP_NAME)
    QApplication.setDesktopFileName('film-lab')
    app = QApplication(sys.argv)
    log_mod.setup()
    log.info('==== Film Lab starting ====\n%s', log_mod.system_info())
    app.setStyle('Fusion')
    app.setPalette(style.palette())
    font = app.font()
    font.setFamilies(['Inter', 'Segoe UI Variable Text', 'Segoe UI', 'SF Pro Text', 'Cantarell', 'Noto Sans',
                      'DejaVu Sans'])
    font.setPointSizeF(10)
    font.setHintingPreference(font.HintingPreference.PreferNoHinting)
    app.setFont(font)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        w.open_folder(sys.argv[1])
    return app.exec()
