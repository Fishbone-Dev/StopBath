"""The photo canvas: fit / zoom / pan, before-after split, and a hover loupe.

Coordinates:
  full   - pixels of the full-resolution photo (what the export measures)
  widget - logical widget pixels
  device - physical screen pixels (widget * devicePixelRatio); "100%" means 1 full px = 1 device px

The canvas always has a whole-photo preview (about 1500-1800 px). When the view needs more detail
than that, it emits `detailNeeded` with a full-resolution region; the window renders it at full
quality and hands it back with `set_detail`. The loupe works the same way via `loupeNeeded`.
"""
import math

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from . import style
from .widgets import _chip, _rounded_path

MAX_ZOOM = 8.0          # device px per full px
LOUPE_RADIUS = 130      # widget px
LOUPE_STEPS = (1.0, 2.0, 3.0, 4.0)


class Region:
    """A rendered piece of the photo: `image` covers full-res rect (x, y, w, h) at `scale`."""

    def __init__(self, key, rect, scale, after, before):
        self.key, self.rect, self.scale, self.after, self.before = key, rect, scale, after, before

    def covers(self, rect, scale):
        x, y, w, h = self.rect
        rx, ry, rw, rh = rect
        return (self.scale >= scale * 0.999 and x <= rx + 0.5 and y <= ry + 0.5
                and x + w >= rx + rw - 0.5 and y + h >= ry + rh - 0.5)


class PreviewCanvas(QWidget):
    detailNeeded = pyqtSignal(tuple, float)   # full rect (x, y, w, h), scale (array px per full px)
    loupeNeeded = pyqtSignal(tuple, float)
    zoomChanged = pyqtSignal()
    cropChanged = pyqtSignal(tuple)           # normalized (cx, cy, w, h) while dragging
    cropFinished = pyqtSignal(tuple)          # after a drag ends
    angleRequested = pyqtSignal(float)        # from the straighten line

    def __init__(self, parent=None):
        super().__init__(parent)
        self.after = None            # whole-photo previews
        self.before = None
        self.full_size = None        # (h, w)
        self.photo_key = None
        self.detail = None           # Region
        self.loupe_region = None     # Region
        self.compare = False
        self.show_original = False
        self.split = 0.5
        self.busy = False
        self.loading_full = False

        self.zoom = None             # None = fit, else device px per full px
        self.center = None           # (cx, cy) in full px
        self.loupe_on = False
        self.loupe_mag = 1.0
        self.mouse = None            # widget QPointF while hovering
        self._drag = None            # 'split' | 'pan'
        self._drag_from = None

        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)
        self._detail_timer = QTimer(self, singleShot=True, interval=60, timeout=self._emit_detail)
        # Smooth zoom
        self._zoom_anim = QVariantAnimation(self, duration=170, startValue=0.0, endValue=1.0,
                                            easingCurve=QEasingCurve(QEasingCurve.Type.OutCubic))
        self._zoom_anim.valueChanged.connect(self._zoom_step)
        self._zoom_anim.finished.connect(self._zoom_done)
        self._zoom_from = self._zoom_target = None
        # Cross-fade when the picture changes (new look or photo)
        self._fade_old = None
        self._fade_t = 1.0
        self._fade_anim = QVariantAnimation(self, duration=200, startValue=0.0, endValue=1.0,
                                            easingCurve=QEasingCurve(QEasingCurve.Type.InOutQuad))
        self._fade_anim.valueChanged.connect(self._fade_step)
        self._fade_anim.finished.connect(self._fade_done)
        self.crop_mode = False
        self.crop = CropOverlay(self)
        self._build_overlay()

    # Overlay controls

    def _build_overlay(self):
        bar = QWidget(self)
        bar.setObjectName('overlayBar')
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        def btn(text, tip, slot, checkable=False, width=None):
            b = QPushButton(text)
            b.setObjectName('overlay')
            b.setToolTip(tip)
            b.setCheckable(checkable)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            if width:
                b.setFixedWidth(width)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        self.fit_btn = btn('Fit', 'Fit to window (0)', self.zoom_fit, checkable=True)
        self.one_btn = btn('100%', 'Actual pixels (1). Double-click the photo to toggle.',
                           lambda: self.zoom_to(1.0), checkable=True)
        btn('−', 'Zoom out (−)', lambda: self.zoom_by(1 / 1.25), width=30)
        self.zoom_label = QLabel('')
        self.zoom_label.setObjectName('overlayLabel')
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setFixedWidth(52)
        lay.addWidget(self.zoom_label)
        btn('+', 'Zoom in (+)', lambda: self.zoom_by(1.25), width=30)
        sep = QWidget()
        sep.setFixedSize(1, 18)
        sep.setStyleSheet('background: %s;' % style.BORDER)
        lay.addSpacing(4)
        lay.addWidget(sep)
        lay.addSpacing(4)
        self.loupe_btn = btn('Loupe', 'Magnifier under the cursor (L).\nShift+scroll changes its magnification.',
                             self.set_loupe, checkable=True)
        bar.adjustSize()
        bar.hide()
        self.overlay = bar
        self._sync_overlay()

    def _sync_overlay(self):
        self.fit_btn.setChecked(self.zoom is None)
        self.one_btn.setChecked(self.zoom is not None and abs(self.zoom - 1) < 1e-6)
        self.zoom_label.setText('%d%%' % round(self.device_scale() * 100) if self.full_size else '')
        self.loupe_btn.setChecked(self.loupe_on)
        self.loupe_btn.setText('Loupe %d%%' % round(self.loupe_mag * 100) if self.loupe_on else 'Loupe')
        self.overlay.adjustSize()
        self._place_overlay()

    def _place_overlay(self):
        o = self.overlay
        o.move(int((self.width() - o.width()) / 2), self.height() - o.height() - 14)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_overlay()
        self._clamp_center()
        self._schedule_detail()

    # Public API

    def set_photo(self, key, full_size):
        """Set the photo (and crop) being shown. Called after every re-render: the zoom level is
        kept, and the position only resets when the photo or output size changes."""
        changed = key != self.photo_key or full_size != self.full_size
        if key != self.photo_key:
            self.photo_key = key
            self.detail = None
            self.loupe_region = None
            self.after = self.before = None
        elif full_size != self.full_size:
            self.detail = None
            self.loupe_region = None
        self.full_size = full_size
        if full_size and (changed or self.center is None):
            self.center = (full_size[1] / 2, full_size[0] / 2)
        self._clamp_center()
        self.overlay.setVisible(full_size is not None and not self.crop_mode)
        self._sync_overlay()
        self._schedule_detail()
        self.update()

    def set_images(self, after, before, fade=False):
        if fade and self.after is not None and after is not None and self.zoom is None:
            self._fade_old = self.after
            self._fade_t = 0.0
            self._fade_anim.stop()
            self._fade_anim.start()
        self.after, self.before = after, before
        self.update()

    def _fade_step(self, t):
        self._fade_t = float(t)
        self.update()

    def _fade_done(self):
        self._fade_old = None
        self._fade_t = 1.0
        self.update()

    def set_detail(self, region):
        if region.key == self.photo_key:
            self.detail = region
            self.update()

    def set_loupe_region(self, region):
        if region.key == self.photo_key:
            self.loupe_region = region
            self.update()

    def invalidate_regions(self):
        """Settings changed: keep showing the old detail until the new one arrives, but re-request."""
        self._schedule_detail()
        self._request_loupe()

    # Crop mode

    def set_crop_mode(self, on):
        self.crop_mode = on
        self.crop.drag = self.crop.line = None
        self.overlay.setVisible(not on and self.full_size is not None)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def set_crop_view(self, image, frame_size):
        self.crop.image, self.crop.frame = image, frame_size
        self.update()

    def set_crop_state(self, angle, rect, ratio):
        self.crop.angle, self.crop.rect, self.crop.ratio = angle, rect, ratio
        self.update()

    def set_busy(self, busy):
        if busy != self.busy:
            self.busy = busy
            self.update()

    def set_loading_full(self, on):
        if on != self.loading_full:
            self.loading_full = on
            self.update()

    def set_compare(self, on):
        self.compare = on
        self.update()

    def set_show_original(self, on):
        self.show_original = on
        self.update()

    def set_loupe(self, on=None):
        self.loupe_on = (not self.loupe_on) if on is None else bool(on)
        if self.loupe_on:
            self.loupe_region = None
        self._sync_overlay()
        self._request_loupe()
        self.update()

    def toggle_loupe(self):
        self.set_loupe(not self.loupe_on)

    # Geometry

    def _dpr(self):
        return self.devicePixelRatioF() or 1.0

    def _fit_scale(self):
        """Widget px per full px when fitted."""
        if not self.full_size:
            return 1.0
        h, w = self.full_size
        margin = 28
        aw, ah = max(1, self.width() - 2 * margin), max(1, self.height() - 2 * margin)
        return min(aw / w, ah / h)

    def widget_scale(self):
        return self._fit_scale() if self.zoom is None else self.zoom / self._dpr()

    def device_scale(self):
        return self.widget_scale() * self._dpr()

    def _clamp_center(self):
        if not self.full_size or self.center is None:
            return
        h, w = self.full_size
        s = self.widget_scale()
        cx, cy = self.center
        half_w, half_h = self.width() / 2 / s, self.height() / 2 / s
        cx = w / 2 if half_w * 2 >= w else min(max(cx, half_w), w - half_w)
        cy = h / 2 if half_h * 2 >= h else min(max(cy, half_h), h - half_h)
        self.center = (cx, cy)

    def image_rect(self):
        """Where the whole photo is drawn, in widget coordinates (may extend beyond the widget)."""
        if not self.full_size:
            return QRectF()
        h, w = self.full_size
        s = self.widget_scale()
        if self.zoom is None:
            return QRectF((self.width() - w * s) / 2, (self.height() - h * s) / 2, w * s, h * s)
        cx, cy = self.center
        return QRectF(self.width() / 2 - cx * s, self.height() / 2 - cy * s, w * s, h * s)

    def widget_to_full(self, pt):
        r = self.image_rect()
        s = self.widget_scale()
        return (pt.x() - r.left()) / s, (pt.y() - r.top()) / s

    def visible_full_rect(self):
        r = self.image_rect()
        s = self.widget_scale()
        h, w = self.full_size
        x0 = max(0.0, -r.left() / s)
        y0 = max(0.0, -r.top() / s)
        x1 = min(float(w), (self.width() - r.left()) / s)
        y1 = min(float(h), (self.height() - r.top()) / s)
        return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)

    def _preview_scale(self):
        """Preview image px per full px."""
        if self.after is None and self.before is None or not self.full_size:
            return 0.0
        img = self.after if self.after is not None else self.before
        return img.width() / self.full_size[1]

    # Zooming

    def zoom_fit(self, animate=True):
        if not self.full_size:
            return
        self.zoom_to(0.0, None, animate)

    def zoom_to(self, z, anchor=None, animate=True):
        """Zoom to `z` device px per full px, keeping the full-res point under `anchor` still.
        z <= the fit scale means "fit to window". Animated unless animate=False."""
        if not self.full_size:
            return
        fit_dev = self._fit_scale() * self._dpr()
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        if self._zoom_anim.state() == QVariantAnimation.State.Running:
            self._zoom_anim.stop()
            self._apply_zoom_state(*self._zoom_target)
        fx, fy = self.widget_to_full(anchor)
        z = min(max(z, fit_dev), MAX_ZOOM)
        if z <= fit_dev * 1.001:
            target = (None, (self.full_size[1] / 2, self.full_size[0] / 2))
        else:
            s = z / self._dpr()
            target = (z, (fx - (anchor.x() - self.width() / 2) / s, fy - (anchor.y() - self.height() / 2) / s))
        start = (self.device_scale(), self.center or target[1])
        if not animate or abs(math.log((target[0] or fit_dev) / max(start[0], 1e-9))) < 0.01:
            self._apply_zoom_state(*target)
            self._after_zoom()
            return
        self._zoom_from, self._zoom_target = start, target
        self._zoom_anim.start()

    def _apply_zoom_state(self, zoom, center):
        self.zoom, self.center = zoom, center
        self._clamp_center()

    def _zoom_step(self, t):
        if self._zoom_from is None:
            return
        (z0, c0), (z1, c1) = self._zoom_from, self._zoom_target
        fit_dev = self._fit_scale() * self._dpr()
        z1v = z1 if z1 is not None else fit_dev
        z = math.exp(math.log(z0) + (math.log(z1v) - math.log(z0)) * t)
        self.zoom = max(z, fit_dev * 1.0005)
        self.center = (c0[0] + (c1[0] - c0[0]) * t, c0[1] + (c1[1] - c0[1]) * t)
        self._clamp_center()
        self._sync_overlay()
        self.update()

    def _zoom_done(self):
        if self._zoom_target is not None:
            self._apply_zoom_state(*self._zoom_target)
        self._zoom_from = self._zoom_target = None
        self._after_zoom()

    def zoom_by(self, factor, anchor=None, animate=True):
        z = self._zoom_target[0] if self._zoom_anim.state() == QVariantAnimation.State.Running and \
            self._zoom_target and self._zoom_target[0] else self.device_scale()
        self.zoom_to(z * factor, anchor, animate)

    def _after_zoom(self):
        if self.zoom is None and self.full_size:
            self.center = (self.full_size[1] / 2, self.full_size[0] / 2)
        self._clamp_center()
        self._sync_overlay()
        self._schedule_detail()
        self._request_loupe()
        self.zoomChanged.emit()
        self.update()

    def _needs_detail(self):
        return self.full_size is not None and self.device_scale() > self._preview_scale() * 1.05

    def _schedule_detail(self):
        if self._needs_detail():
            self._detail_timer.start()

    def _emit_detail(self):
        if not self._needs_detail():
            return
        x, y, w, h = self.visible_full_rect()
        if w <= 0 or h <= 0:
            return
        scale = min(1.0, self.device_scale())
        # Render a margin around the view so small pans don't need a new render
        mx, my = w * 0.2, h * 0.2
        H, W = self.full_size
        x0, y0 = max(0.0, x - mx), max(0.0, y - my)
        x1, y1 = min(float(W), x + w + mx), min(float(H), y + h + my)
        rect = (int(x0), int(y0), int(math.ceil(x1)) - int(x0), int(math.ceil(y1)) - int(y0))
        if self.detail is not None and self.detail.covers((x, y, w, h), scale) and not self._detail_stale:
            return
        self._detail_stale = False
        self.detailNeeded.emit(rect, scale)

    _detail_stale = False

    def mark_detail_stale(self):
        self._detail_stale = True
        self._loupe_stale = True
        self.invalidate_regions()

    # Loupe

    _loupe_stale = False

    def _loupe_active(self):
        return (self.loupe_on and self.mouse is not None and self.full_size is not None
                and self.device_scale() < self.loupe_mag * 0.999 and self._over_image(self.mouse))

    def _over_image(self, pt):
        return self.image_rect().contains(pt) and self.rect().contains(pt.toPoint())

    def _loupe_full_rect(self):
        """Full-res rect shown inside the loupe for the current mouse position."""
        fx, fy = self.widget_to_full(self.mouse)
        half = LOUPE_RADIUS * self._dpr() / self.loupe_mag
        return fx - half, fy - half, 2 * half, 2 * half

    def _request_loupe(self):
        if not self._loupe_active():
            return
        x, y, w, h = self._loupe_full_rect()
        H, W = self.full_size
        # Ask for a slightly larger area so small mouse moves are served from the same render
        pad = w * 0.35
        rx0, ry0 = max(0, int(x - pad)), max(0, int(y - pad))
        rx1, ry1 = min(W, int(math.ceil(x + w + pad))), min(H, int(math.ceil(y + h + pad)))
        if rx1 <= rx0 or ry1 <= ry0:
            return
        visible = (max(0.0, x), max(0.0, y), min(float(W), x + w) - max(0.0, x), min(float(H), y + h) - max(0.0, y))
        if self.loupe_region is not None and self.loupe_region.covers(visible, 1.0) and not self._loupe_stale:
            return
        self._loupe_stale = False
        self.loupeNeeded.emit((rx0, ry0, rx1 - rx0, ry1 - ry0), 1.0)

    # Painting

    @staticmethod
    def _draw_source(p, image, img_scale, origin, full_rect, target, smooth):
        """Draw the part of `image` (covering full-res from `origin` at `img_scale`) that shows
        full-res rect `full_rect` into widget rect `target`."""
        x, y, w, h = full_rect
        ox, oy = origin
        src = QRectF((x - ox) * img_scale, (y - oy) * img_scale, w * img_scale, h * img_scale)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        p.drawImage(target, image, src)

    def _paint_photo(self, p, which, clip):
        """Paint `which` ('after' or 'before') for the full-res area visible inside widget rect `clip`."""
        r = self.image_rect()
        s = self.widget_scale()
        area = clip.intersected(r)
        if area.isEmpty():
            return
        full = ((area.left() - r.left()) / s, (area.top() - r.top()) / s, area.width() / s, area.height() / s)
        smooth = self.device_scale() < 2.0
        preview = getattr(self, which)
        if preview is None:
            preview = self.after if self.after is not None else self.before
        if preview is not None:
            self._draw_source(p, preview, preview.width() / self.full_size[1], (0, 0), full, area, True)
        d = self.detail
        if d is not None and self._needs_detail():
            img = d.after if which == 'after' else d.before
            if img is not None:
                dx, dy, dw, dh = d.rect
                # Intersect the detail rect with what's visible
                ix0, iy0 = max(full[0], dx), max(full[1], dy)
                ix1, iy1 = min(full[0] + full[2], dx + dw), min(full[1] + full[3], dy + dh)
                if ix1 > ix0 and iy1 > iy0:
                    target = QRectF(r.left() + ix0 * s, r.top() + iy0 * s, (ix1 - ix0) * s, (iy1 - iy0) * s)
                    self._draw_source(p, img, d.scale, (dx, dy), (ix0, iy0, ix1 - ix0, iy1 - iy0), target, smooth)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(style.BG))
        if self.crop_mode:
            self.crop.paint(p)
            return
        if not self.full_size or (self.after is None and self.before is None):
            return
        r = self.image_rect()
        view = QRectF(self.rect())

        if self.zoom is None:
            for i, a in enumerate((28, 18, 10)):  # soft drop shadow when the whole photo is shown
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(0, 0, 0, a))
                d = 3 + i * 4
                p.drawRoundedRect(r.adjusted(-d, -d + 4, d, d + 4), 6 + d, 6 + d)

        p.save()
        if self.zoom is None:
            p.setClipPath(_rounded_path(r, 4))
        split_x = r.left() + r.width() * self.split if self.zoom is None else self.width() * self.split
        if self.show_original:
            self._paint_photo(p, 'before', view)
        elif self.compare and self.before is not None:
            self._paint_photo(p, 'before', QRectF(0, 0, split_x, self.height()))
            self._paint_photo(p, 'after', QRectF(split_x, 0, self.width() - split_x, self.height()))
        else:
            self._paint_photo(p, 'after', view)
            if self._fade_old is not None and self.zoom is None:
                new = self.after
                p.setOpacity(1.0 - self._fade_t)
                self.after = self._fade_old
                self._paint_photo(p, 'after', view)
                self.after = new
                p.setOpacity(1.0)
        p.restore()

        top = max(r.top(), 0) + 12
        left = max(r.left(), 0) + 12
        right = min(r.right(), self.width()) - 12
        if self.show_original:
            _chip(p, 'ORIGINAL', QPointF(left, top))
        elif self.compare and self.before is not None:
            self._paint_split_handle(p, split_x, max(r.top(), 0), min(r.bottom(), self.height()))
            _chip(p, 'BEFORE', QPointF(left, top))
            _chip(p, 'AFTER', QPointF(right, top), align_right=True)

        if self._loupe_active():
            self._paint_loupe(p, split_x)

        status = 'LOADING FULL RESOLUTION…' if self.loading_full and (self._needs_detail() or self._loupe_active()) \
            else ('DEVELOPING…' if self.busy else '')
        if status:
            _chip(p, status, QPointF(right, top + (34 if self.compare else 0)), align_right=True)

    def _paint_split_handle(self, p, x, top, bottom):
        p.setPen(QPen(QColor(255, 255, 255, 230), 2))
        p.drawLine(QPointF(x, top), QPointF(x, bottom))
        c = QPointF(x, (top + bottom) / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 240))
        p.drawEllipse(c, 15, 15)
        p.setPen(QPen(QColor(style.BG), 2))
        for sgn in (-1, 1):
            p.drawLine(c + QPointF(3 * sgn, -5), c + QPointF(8 * sgn, 0))
            p.drawLine(c + QPointF(8 * sgn, 0), c + QPointF(3 * sgn, 5))

    def _paint_loupe(self, p, split_x):
        m = self.mouse
        R = LOUPE_RADIUS
        circle = QRectF(m.x() - R, m.y() - R, 2 * R, 2 * R)
        which = 'before' if self.show_original or (self.compare and m.x() < split_x) else 'after'

        # Shadow + body
        p.setPen(Qt.PenStyle.NoPen)
        for i, a in enumerate((40, 24, 12)):
            p.setBrush(QColor(0, 0, 0, a))
            d = 2 + i * 3
            p.drawEllipse(circle.adjusted(-d, -d + 3, d, d + 3))
        p.save()
        path = QPainterPath()
        path.addEllipse(circle)
        p.setClipPath(path)
        p.fillRect(circle, QColor(style.BG))

        x, y, w, h = self._loupe_full_rect()
        s = 2 * R / w   # widget px per full px inside the loupe
        smooth = self.loupe_mag < 2.0
        preview = getattr(self, which)
        if preview is None:
            preview = self.after if self.after is not None else self.before
        if preview is not None:
            # Placeholder while the full-res render is on its way
            self._draw_source(p, preview, preview.width() / self.full_size[1], (0, 0), (x, y, w, h), circle, True)
        lr = self.loupe_region
        img = None if lr is None else (lr.after if which == 'after' else lr.before)
        if img is not None:
            dx, dy, dw, dh = lr.rect
            ix0, iy0 = max(x, dx), max(y, dy)
            ix1, iy1 = min(x + w, dx + dw), min(y + h, dy + dh)
            if ix1 > ix0 and iy1 > iy0:
                target = QRectF(circle.left() + (ix0 - x) * s, circle.top() + (iy0 - y) * s,
                                (ix1 - ix0) * s, (iy1 - iy0) * s)
                self._draw_source(p, img, lr.scale, (dx, dy), (ix0, iy0, ix1 - ix0, iy1 - iy0), target, smooth)
        p.restore()

        # Crosshair marking the exact point under the (hidden) cursor
        c = circle.center()
        for color, width in ((QColor(0, 0, 0, 120), 3), (QColor(255, 255, 255, 200), 1)):
            p.setPen(QPen(color, width))
            p.drawLine(c + QPointF(-9, 0), c + QPointF(-3, 0))
            p.drawLine(c + QPointF(3, 0), c + QPointF(9, 0))
            p.drawLine(c + QPointF(0, -9), c + QPointF(0, -3))
            p.drawLine(c + QPointF(0, 3), c + QPointF(0, 9))

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(255, 255, 255, 235), 2.5))
        p.drawEllipse(circle)
        p.setPen(QPen(QColor(0, 0, 0, 90), 1))
        p.drawEllipse(circle.adjusted(-1.8, -1.8, 1.8, 1.8))
        label = '%d%%' % round(self.loupe_mag * 100)
        if which == 'before' and self.compare:
            label = 'BEFORE · ' + label
        f = QFont(p.font())
        f.setPointSizeF(8)
        f.setBold(True)
        p.setFont(f)
        tw = p.fontMetrics().horizontalAdvance(label) + 14
        chip = QRectF(m.x() - tw / 2, circle.bottom() - 12, tw, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 235))
        p.drawRoundedRect(chip, 10, 10)
        p.setPen(QColor(style.BG))
        p.drawText(chip, Qt.AlignmentFlag.AlignCenter, label)

    # Input

    def wheelEvent(self, e):
        if not self.full_size or self.crop_mode:
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier and self.loupe_on:
            i = LOUPE_STEPS.index(self.loupe_mag) if self.loupe_mag in LOUPE_STEPS else 0
            i = min(max(i + (1 if steps > 0 else -1), 0), len(LOUPE_STEPS) - 1)
            self.loupe_mag = LOUPE_STEPS[i]
            self._sync_overlay()
            self.loupe_region = None
            self._request_loupe()
            self.update()
            return
        self.zoom_by(1.25 ** steps, e.position(), animate=abs(steps) >= 1)

    def _near_split(self, pos):
        r = self.image_rect()
        x = r.left() + r.width() * self.split if self.zoom is None else self.width() * self.split
        return abs(pos.x() - x) < 24

    def mousePressEvent(self, e):
        if self.crop_mode:
            if e.button() == Qt.MouseButton.LeftButton and self.crop.frame:
                ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                if self.crop.press(e.position(), ctrl):
                    self.update()
            return
        if e.button() == Qt.MouseButton.LeftButton:
            if self.compare and (self.zoom is None or self._near_split(e.position())):
                self._drag = 'split'
                self._set_split_from(e.position())
            elif self.zoom is not None:
                self._drag = 'pan'
                self._drag_from = (e.position(), self.center)
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif e.button() == Qt.MouseButton.MiddleButton and self.zoom is not None:
            self._drag = 'pan'
            self._drag_from = (e.position(), self.center)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        self.mouse = e.position()
        if self.crop_mode:
            if not self.crop.frame:
                return
            if self.crop.drag or self.crop.line:
                rect = self.crop.move(e.position())
                if rect is not None:
                    self.cropChanged.emit(tuple(rect))
            else:
                ctrl = bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier)
                self.setCursor(self.crop.cursor_for(e.position(), ctrl))
            self.update()
            return
        if self._drag == 'split':
            self._set_split_from(e.position())
        elif self._drag == 'pan':
            start, (cx, cy) = self._drag_from
            s = self.widget_scale()
            d = e.position() - start
            self.center = (cx - d.x() / s, cy - d.y() / s)
            self._clamp_center()
            self._schedule_detail()
        self._update_cursor()
        if self.loupe_on:
            self._request_loupe()
        self.update()

    def mouseReleaseEvent(self, e):
        if self.crop_mode:
            was_drag = self.crop.drag is not None
            result = self.crop.release()
            if result:
                self.angleRequested.emit(result[1])
            elif was_drag:
                self.cropFinished.emit(tuple(self.crop.rect))
            self.update()
            return
        if self._drag == 'pan':
            self._schedule_detail()
        self._drag = None
        self._update_cursor()

    def mouseDoubleClickEvent(self, e):
        if self.crop_mode or e.button() != Qt.MouseButton.LeftButton or not self.full_size:
            return
        if self.zoom is None:
            self.zoom_to(1.0, e.position())
        else:
            self.zoom_fit()

    def leaveEvent(self, e):
        self.mouse = None
        self.update()

    def _update_cursor(self):
        if self._drag == 'pan':
            return
        if self.mouse is not None and self.compare and (self.zoom is None or self._near_split(self.mouse)):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        elif self.loupe_on and self.mouse is not None and self._loupe_active():
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif self.zoom is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def _set_split_from(self, pos):
        if self.zoom is None:
            r = self.image_rect()
            if r.width() > 0:
                self.split = min(max((pos.x() - r.left()) / r.width(), 0.0), 1.0)
        else:
            self.split = min(max(pos.x() / max(1, self.width()), 0.0), 1.0)
        self.update()


class CropOverlay:
    """Crop & straighten interaction for PreviewCanvas.

    The whole frame (photo after quarter turns) is drawn rotated by the straighten angle, and the
    crop is an upright rectangle on top of it, as in other photo editors. Internally the rectangle
    lives in "q" space: straightened frame coordinates centred on the frame centre, so the screen
    is simply  O + s·q.
    """
    HANDLE = 14

    def __init__(self, canvas):
        self.c = canvas
        self.image = None           # QImage of the frame
        self.frame = None           # (fh, fw) full-res
        self.angle = 0.0
        self.rect = (0.5, 0.5, 1.0, 1.0)
        self.ratio = None           # locked width/height in pixels, or None
        self.drag = None
        self.line = None            # (start, end) widget points while straightening by line

    # Transforms
    def _rot(self, x, y, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return c * x - s * y, s * x + c * y

    def view(self):
        """(scale, origin) mapping q space to the widget."""
        fh, fw = self.frame
        a = math.radians(self.angle)
        c, s = abs(math.cos(a)), abs(math.sin(a))
        bw, bh = fw * c + fh * s, fw * s + fh * c
        margin = 46
        k = min((self.c.width() - 2 * margin) / bw, (self.c.height() - 2 * margin - 30) / bh)
        return max(k, 1e-6), QPointF(self.c.width() / 2, self.c.height() / 2 + 12)

    def q_rect(self, rect=None):
        """Crop as (qcx, qcy, half_w, half_h) in q space."""
        fh, fw = self.frame
        cx, cy, w, h = rect or self.rect
        qx, qy = self._rot(cx * fw - fw / 2, cy * fh - fh / 2, self.angle)
        return qx, qy, w * fw / 2, h * fh / 2

    def from_q(self, qx, qy, a, b):
        fh, fw = self.frame
        x, y = self._rot(qx, qy, -self.angle)
        return (x + fw / 2) / fw, (y + fh / 2) / fh, 2 * a / fw, 2 * b / fh

    def screen_rect(self):
        s, o = self.view()
        qx, qy, a, b = self.q_rect()
        return QRectF(o.x() + s * (qx - a), o.y() + s * (qy - b), 2 * s * a, 2 * s * b)

    def to_q(self, pt):
        s, o = self.view()
        return (pt.x() - o.x()) / s, (pt.y() - o.y()) / s

    def valid(self, rect):
        from ..geometry import MIN_SIZE, is_inside
        fh, fw = self.frame
        cx, cy, w, h = rect
        return w >= MIN_SIZE * 0.999 and h >= MIN_SIZE * 0.999 and is_inside(fw, fh, self.angle, cx, cy, w, h)

    # Painting
    def paint(self, p):
        if self.image is None or self.frame is None:
            return
        fh, fw = self.frame
        s, o = self.view()
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.translate(o)
        p.scale(s, s)
        p.rotate(self.angle)
        p.translate(-fw / 2, -fh / 2)
        p.drawImage(QRectF(0, 0, fw, fh), self.image)
        p.restore()

        r = self.screen_rect()
        shade = QPainterPath()
        shade.addRect(QRectF(self.c.rect()))
        shade.addRect(r)
        shade.setFillRule(Qt.FillRule.OddEvenFill)
        p.fillPath(shade, QColor(10, 11, 14, 175))

        # Rule-of-thirds grid (finer while dragging, to help levelling)
        lines = 9 if self.drag else 3
        p.setPen(QPen(QColor(255, 255, 255, 60 if self.drag else 80), 1))
        for i in range(1, lines):
            x = r.left() + r.width() * i / lines
            y = r.top() + r.height() * i / lines
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QPen(QColor(255, 255, 255, 220), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)

        # Corner and edge handles
        pen = QPen(QColor(255, 255, 255), 3)
        pen.setCapStyle(Qt.PenCapStyle.SquareCap)
        p.setPen(pen)
        L = 18
        for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x = r.left() if sx < 0 else r.right()
            y = r.top() if sy < 0 else r.bottom()
            p.drawLine(QPointF(x, y), QPointF(x - sx * L, y))
            p.drawLine(QPointF(x, y), QPointF(x, y - sy * L))
        for x, y, horiz in ((r.center().x(), r.top(), True), (r.center().x(), r.bottom(), True),
                            (r.left(), r.center().y(), False), (r.right(), r.center().y(), False)):
            if horiz:
                p.drawLine(QPointF(x - 10, y), QPointF(x + 10, y))
            else:
                p.drawLine(QPointF(x, y - 10), QPointF(x, y + 10))

        if self.line:
            a, b = self.line
            pen = QPen(QColor(style.ACCENT), 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(a, b)
            p.setBrush(QColor(style.ACCENT))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(a, 4, 4)
            p.drawEllipse(b, 4, 4)

        hint = ('Drag edges or corners to crop  ·  Drag inside to move  ·  '
                'Ctrl+drag along the horizon to straighten  ·  Enter when done')
        self._hint(p, hint)
        fh_, fw_ = self.frame
        cx, cy, w, h = self.rect
        info = '%d × %d px' % (round(w * fw_), round(h * fh_))
        if self.angle:
            info += '   ·   %+.1f°' % self.angle
        _chip(p, info, QPointF(r.center().x() - 60, r.bottom() + 8))

    def _hint(self, p, text):
        f = QFont(p.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 24
        box = QRectF((self.c.width() - w) / 2, 10, w, 24)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 150))
        p.drawRoundedRect(box, 12, 12)
        p.setPen(QColor(style.DIM))
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    # Hit testing
    def hit(self, pt):
        r = self.screen_rect()
        H = self.HANDLE
        near_l, near_r = abs(pt.x() - r.left()) < H, abs(pt.x() - r.right()) < H
        near_t, near_b = abs(pt.y() - r.top()) < H, abs(pt.y() - r.bottom()) < H
        in_x = r.left() - H < pt.x() < r.right() + H
        in_y = r.top() - H < pt.y() < r.bottom() + H
        sx = -1 if near_l else (1 if near_r else 0)
        sy = -1 if near_t else (1 if near_b else 0)
        if sx and sy:
            return ('corner', sx, sy)
        if sx and in_y:
            return ('edge', sx, 0)
        if sy and in_x:
            return ('edge', 0, sy)
        if r.contains(pt):
            return ('move', 0, 0)
        return None

    def cursor_for(self, pt, ctrl):
        if ctrl:
            return Qt.CursorShape.CrossCursor
        h = self.hit(pt)
        if h is None:
            return Qt.CursorShape.ArrowCursor
        kind, sx, sy = h
        if kind == 'move':
            return Qt.CursorShape.SizeAllCursor
        if kind == 'edge':
            return Qt.CursorShape.SizeHorCursor if sx else Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.SizeFDiagCursor if sx == sy else Qt.CursorShape.SizeBDiagCursor

    # Dragging
    def press(self, pt, ctrl):
        if ctrl:
            self.line = (pt, pt)
            self.drag = None
            return True
        h = self.hit(pt)
        if h is None:
            return False
        self.drag = (h, self.to_q(pt), self.q_rect(), self.rect)
        return True

    def move(self, pt):
        if self.line:
            self.line = (self.line[0], pt)
            return None
        if not self.drag:
            return None
        (kind, sx, sy), q0, (qx, qy, a, b), rect0 = self.drag
        mx, my = self.to_q(pt)
        from ..geometry import MIN_SIZE, fit_inside
        fh, fw = self.frame
        min_a, min_b = MIN_SIZE * fw / 2, MIN_SIZE * fh / 2
        if kind == 'move':
            cand = self.from_q(qx + mx - q0[0], qy + my - q0[1], a, b)
            cand = fit_inside(fw, fh, self.angle, *cand)
            self.rect = cand
            return cand
        # Resize with the opposite edge/corner fixed
        fx, fy = qx - sx * a, qy - sy * b           # fixed point (only the relevant axes matter)
        na, nb = a, b
        if sx:
            na = max(min_a, sx * (mx - fx) / 2)
        if sy:
            nb = max(min_b, sy * (my - fy) / 2)
        if self.ratio:
            r = self.ratio
            if sx and sy:
                if na / nb > r:
                    na = nb * r
                else:
                    nb = na / r
            elif sx:
                nb = na / r
            else:
                na = nb * r
        nqx = fx + sx * na if sx else qx
        nqy = fy + sy * nb if sy else qy
        target = (nqx, nqy, na, nb)
        start = (qx, qy, a, b)
        if not self.valid(self.from_q(*target)):
            # Largest step towards the target that stays inside the frame
            lo, hi = 0.0, 1.0
            for _ in range(24):
                mid = (lo + hi) / 2
                trial = tuple(s0 + (t - s0) * mid for s0, t in zip(start, target))
                if self.valid(self.from_q(*trial)):
                    lo = mid
                else:
                    hi = mid
            target = tuple(s0 + (t - s0) * lo for s0, t in zip(start, target))
        self.rect = self.from_q(*target)
        return self.rect

    def release(self):
        """Returns ('angle', degrees) after a straighten line, else None."""
        result = None
        if self.line:
            a, b = self.line
            dx, dy = b.x() - a.x(), b.y() - a.y()
            if math.hypot(dx, dy) > 20:
                ang = math.degrees(math.atan2(dy, dx))       # direction on screen
                # Fold to the nearest horizontal or vertical reference
                while ang > 90:
                    ang -= 180
                while ang <= -90:
                    ang += 180
                if ang > 45:
                    ang -= 90
                elif ang < -45:
                    ang += 90
                result = ('angle', self.angle - ang)
        self.line = None
        self.drag = None
        return result
