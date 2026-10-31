"""Reusable widgets: labelled sliders, panel sections and list delegates."""
from PyQt6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (QAbstractScrollArea, QApplication, QComboBox, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
                             QSizePolicy, QSlider, QStyle, QStyledItemDelegate,
                             QVBoxLayout, QWidget)

from . import style

THUMB_ROLE = Qt.ItemDataRole.UserRole + 10
KIND_ROLE = Qt.ItemDataRole.UserRole + 11
EDITED_ROLE = Qt.ItemDataRole.UserRole + 12
MARKED_ROLE = Qt.ItemDataRole.UserRole + 13


class ElidedLabel(QLabel):
    """A label that shortens its text with "…" to fit, instead of forcing its parent wider.
    The full text is shown as a tooltip when it doesn't fit."""

    def __init__(self, text='', mode=Qt.TextElideMode.ElideMiddle, parent=None):
        super().__init__(parent)
        self._full = ''
        self._mode = mode
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full = text or ''
        self._refresh()

    def text(self):
        return self._full

    def minimumSizeHint(self):
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self):
        return QSize(min(self.fontMetrics().horizontalAdvance(self._full) + 4, 400), super().sizeHint().height())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() in (e.Type.FontChange, e.Type.StyleChange):
            self._refresh()

    def _refresh(self):
        shown = self.fontMetrics().elidedText(self._full, self._mode, max(0, self.width()))
        super().setText(shown)
        self.setToolTip(self._full if shown != self._full else '')


class _Slider(QSlider):
    doubleClicked = pyqtSignal()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()

    def wheelEvent(self, e):
        # Don't hijack scrolling of the side panel unless the slider has focus
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class ParamSlider(QWidget):
    """Label + value readout above a slider. Double-click resets to the default."""
    valueChanged = pyqtSignal(float)

    def __init__(self, label, minimum, maximum, default=0.0, step=0.01, fmt=None, tooltip='', parent=None):
        super().__init__(parent)
        self.minimum, self.maximum, self.default, self.step = minimum, maximum, default, step
        self.fmt = fmt or (lambda v: '%+.2f' % v if minimum < 0 else '%.2f' % v)

        self.label = QLabel(label)
        self.readout = _ClickLabel()
        self.readout.setObjectName('value')
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.readout.setCursor(Qt.CursorShape.IBeamCursor)
        self.readout.setToolTip('Click to type a value')
        self.readout.clicked.connect(self._start_edit)
        self.editor = QLineEdit()
        self.editor.setObjectName('valueEdit')
        self.editor.setFixedWidth(70)
        self.editor.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.editor.hide()
        self.editor.editingFinished.connect(self._finish_edit)
        self.slider = _Slider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, int(round((maximum - minimum) / step)))
        self.slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.slider.valueChanged.connect(self._on_slider)
        self.slider.doubleClicked.connect(self.reset)
        if tooltip:
            self.setToolTip(tooltip + '\n\nDouble-click to reset.')

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.label)
        top.addStretch()
        top.addWidget(self.readout)
        top.addWidget(self.editor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        lay.addLayout(top)
        lay.addWidget(self.slider)
        self.set_value(default)

    def value(self):
        return round(self.minimum + self.slider.value() * self.step, 6)

    def _display_scale(self):
        """How displayed numbers relate to values (e.g. 100 for percentages)."""
        shown = self.fmt(self.maximum)
        if shown.strip().endswith('%') and self.maximum:
            try:
                return float(shown.strip().rstrip('%')) / self.maximum
            except ValueError:
                pass
        return 1.0

    def _start_edit(self):
        if not self.isEnabled():
            return
        v = self.value() * self._display_scale()
        self.editor.setText(('%g' % round(v, 2)))
        self.readout.hide()
        self.editor.show()
        self.editor.setFocus()
        self.editor.selectAll()

    def _finish_edit(self):
        if not self.editor.isVisible():
            return
        text = self.editor.text().strip()
        for junk in ('%', 'EV', '°', '×', 'x', '+'):
            text = text.replace(junk, '')
        self.editor.hide()
        self.readout.show()
        try:
            v = float(text.replace(',', '.')) / self._display_scale()
        except ValueError:
            return
        self.set_value(v)
        self.valueChanged.emit(self.value())

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.editor.isVisible():
            self.editor.hide()
            self.readout.show()
            return
        super().keyPressEvent(e)

    def set_value(self, v):
        """Set without emitting valueChanged."""
        v = min(max(float(v), self.minimum), self.maximum)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.minimum) / self.step)))
        self.slider.blockSignals(False)
        self._update_readout()

    def reset(self):
        self.set_value(self.default)
        self.valueChanged.emit(self.value())

    def _on_slider(self, _):
        self._update_readout()
        self.valueChanged.emit(self.value())

    def _update_readout(self):
        v = self.value()
        self.readout.setText(self.fmt(v))
        changed = abs(v - self.default) > self.step / 2
        color = style.FAINT if not self.isEnabled() else (style.ACCENT if changed else style.DIM)
        self.readout.setStyleSheet('color: %s;' % color)

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == e.Type.EnabledChange:
            self._update_readout()


class NoScrollComboBox(QComboBox):
    """A dropdown that ignores the mouse wheel, so scrolling the side panel never changes a setting
    by accident. Choose a value by clicking it, as usual."""

    def wheelEvent(self, e):
        # Hand the scroll to the panel this dropdown sits in (if any), as if it weren't here
        w = self.parentWidget()
        while w is not None and not isinstance(w, QAbstractScrollArea):
            w = w.parentWidget()
        if w is None:
            e.ignore()
            return
        QApplication.sendEvent(w.verticalScrollBar(), e)
        e.accept()


class _ClickLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class Section(QWidget):
    """A titled group in the side panel. Click the title to fold it away (animated); the state can
    be remembered through `store` (a QSettings)."""

    def __init__(self, title, parent=None, store=None, action=None):
        super().__init__(parent)
        self._name = title
        self._store = store
        self._header = QPushButton()
        self._header.setObjectName('sectionHeader')
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._header.clicked.connect(self.toggle)
        self._content = QWidget()
        self.body = QVBoxLayout(self._content)
        self.body.setContentsMargins(0, 2, 0, 2)
        self.body.setSpacing(6)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 2)
        lay.setSpacing(6)
        if action:
            text, tip, callback = action
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(self._header, 1)
            btn = QPushButton(text)
            btn.setObjectName('sectionAction')
            btn.setToolTip(tip)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(callback)
            row.addWidget(btn)
            lay.addLayout(row)
        else:
            lay.addWidget(self._header)
        lay.addWidget(self._content)
        self._anim = QPropertyAnimation(self._content, b'maximumHeight', self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._anim_done)
        self.expanded = True
        if store is not None and store.value('section/' + title, True, type=bool) is False:
            self.expanded = False
            self._content.setMaximumHeight(0)
            self._content.hide()
        self._update_header()

    def _update_header(self):
        self._header.setText(('▾  ' if self.expanded else '▸  ') + self._name.upper())

    def toggle(self):
        self.set_expanded(not self.expanded)

    def set_expanded(self, on):
        if on == self.expanded:
            return
        self.expanded = on
        self._update_header()
        if self._store is not None:
            self._store.setValue('section/' + self._name, on)
        full = self._content.sizeHint().height()
        self._anim.stop()
        if on:
            self._content.show()
            self._anim.setStartValue(0)
            self._anim.setEndValue(full)
        else:
            self._anim.setStartValue(self._content.height())
            self._anim.setEndValue(0)
        self._anim.start()

    def _anim_done(self):
        if self.expanded:
            self._content.setMaximumHeight(16777215)
        else:
            self._content.hide()

    def add(self, w):
        self.body.addWidget(w)
        return w


class Histogram(QWidget):
    """RGB + brightness histogram of the developed photo, with clipping warnings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.data = None          # (4, bins): R, G, B, luma, each normalised to 0..1
        self.clip = (0.0, 0.0)    # fraction of pixels crushed to black / blown to white

    def set_data(self, data, clip):
        self.data, self.clip = data, clip
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor('#121317'))
        p.drawRoundedRect(r, 8, 8)
        if self.data is None:
            return
        inner = r.adjusted(6, 8, -6, -6)
        n = self.data.shape[1]
        colors = (QColor(255, 80, 80, 95), QColor(80, 220, 120, 95), QColor(90, 140, 255, 95))
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        for ch, color in enumerate(colors):
            path = QPainterPath()
            path.moveTo(inner.left(), inner.bottom())
            for i in range(n):
                x = inner.left() + inner.width() * i / (n - 1)
                path.lineTo(x, inner.bottom() - inner.height() * float(self.data[ch, i]))
            path.lineTo(inner.right(), inner.bottom())
            path.closeSubpath()
            p.fillPath(path, color)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        path = QPainterPath()
        for i in range(n):
            x = inner.left() + inner.width() * i / (n - 1)
            y = inner.bottom() - inner.height() * float(self.data[3, i])
            path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
        p.setPen(QPen(QColor(235, 235, 240, 170), 1.2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        # Clipping warnings: little triangles in the corners
        for frac, left in ((self.clip[0], True), (self.clip[1], False)):
            if frac > 0.005:
                x = inner.left() if left else inner.right()
                d = 1 if left else -1
                tri = QPainterPath()
                tri.moveTo(x, inner.top())
                tri.lineTo(x + 9 * d, inner.top())
                tri.lineTo(x, inner.top() + 9)
                tri.closeSubpath()
                p.fillPath(tri, QColor(style.ACCENT))
                self.setToolTip('%.1f%% of the photo is pure %s' % (frac * 100, 'black' if left else 'white'))


class Toast(QWidget):
    """Small notification card that fades in/out over the photo. Can also show a progress bar
    with a Cancel button (used for exports)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName('toast')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 10, 10)
        lay.setSpacing(12)
        self.label = QLabel()
        self.label.setObjectName('toastText')
        self.bar = QProgressBar()
        self.bar.setFixedWidth(160)
        self.cancel = QPushButton('Cancel')
        self.cancel.setObjectName('flat')
        self.cancel.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.label)
        lay.addWidget(self.bar)
        lay.addWidget(self.cancel)
        self._fx = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._fx)
        self._anim = QPropertyAnimation(self._fx, b'opacity', self)
        self._anim.setDuration(180)
        self._anim.finished.connect(lambda: self._fx.opacity() < 0.01 and self.hide())
        self._timer = QTimer(self, singleShot=True, timeout=self.fade_out)
        self._cancel_cb = None
        self.cancel.clicked.connect(lambda: self._cancel_cb and self._cancel_cb())
        self.hide()

    anchor = None   # widget to float over (the photo); defaults to the parent

    def _place(self):
        self.adjustSize()
        par = self.parentWidget()
        a = self.anchor if self.anchor is not None and self.anchor.isVisible() else par
        pos = a.mapTo(par, a.rect().topLeft()) if a is not par else a.rect().topLeft()
        self.move(int(pos.x() + (a.width() - self.width()) / 2), int(pos.y() + a.height() - self.height() - 64))
        self.raise_()

    def _fade_in(self):
        if not self.isVisible() or self._fx.opacity() < 1:
            self.show()
            self._anim.stop()
            self._anim.setStartValue(self._fx.opacity() if self.isVisible() else 0.0)
            self._anim.setEndValue(1.0)
            self._anim.start()

    def fade_out(self):
        self._anim.stop()
        self._anim.setStartValue(self._fx.opacity())
        self._anim.setEndValue(0.0)
        self._anim.start()

    def message(self, text, ms=3500):
        self.label.setText(text)
        self.bar.hide()
        self.cancel.hide()
        self._place()
        self._fade_in()
        self._timer.start(ms)

    def progress(self, text, value, maximum, on_cancel):
        self._timer.stop()
        self.label.setText(text)
        self.bar.setRange(0, maximum)
        self.bar.setValue(value)
        self.bar.show()
        self.cancel.show()
        self._cancel_cb = on_cancel
        self._place()
        self._fade_in()


class DropOverlay(QWidget):
    """Shown while a folder is dragged over the window."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(14, 15, 18, 200))
        r = QRectF(self.rect()).adjusted(28, 28, -28, -28)
        pen = QPen(QColor(style.ACCENT), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(QColor(233, 162, 59, 18))
        p.drawRoundedRect(r, 18, 18)
        f = QFont(p.font())
        f.setPointSizeF(18)
        p.setFont(f)
        p.setPen(QColor(style.TEXT))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, 'Drop a folder to open it')


def _rounded_path(rect, radius):
    path = QPainterPath()
    path.addRoundedRect(QRectF(rect), radius, radius)
    return path


def _chip(p, text, anchor, align_right=False, bottom=False):
    font = QFont(p.font())
    font.setPointSizeF(8)
    font.setBold(True)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
    p.setFont(font)
    fm = p.fontMetrics()
    w, h = fm.horizontalAdvance(text) + 16, fm.height() + 8
    x = anchor.x() - w if align_right else anchor.x()
    y = anchor.y() - h if bottom else anchor.y()
    r = QRectF(x, y, w, h)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0, 150))
    p.drawRoundedRect(r, h / 2, h / 2)
    p.setPen(QColor(style.TEXT))
    p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


def _draw_cover(p, pix, rect):
    """Draw `pix` filling `rect`, cropping the overflow (like CSS object-fit: cover)."""
    if pix is None or pix.isNull():
        p.fillRect(rect, QColor(style.RAISED))
        return
    s = max(rect.width() / pix.width(), rect.height() / pix.height())
    w, h = pix.width() * s, pix.height() * s
    target = QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
    p.drawPixmap(target, pix, QRectF(pix.rect()))


def _draw_contain(p, pix, rect):
    s = min(rect.width() / pix.width(), rect.height() / pix.height())
    w, h = pix.width() * s, pix.height() * s
    target = QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
    p.drawPixmap(target, pix, QRectF(pix.rect()))
    return target


class FilmstripDelegate(QStyledItemDelegate):
    SIZE = QSize(150, 112)
    MARK_RADIUS = 9

    def sizeHint(self, option, index):
        return self.SIZE

    @classmethod
    def trash_rect(cls, item_rect):
        """Clickable trash icon in the thumbnail's top-right corner."""
        r = QRectF(item_rect).adjusted(6, 6, -6, -22)
        return QRectF(r.right() - 26, r.top() + 4, 22, 22)

    @classmethod
    def mark_center(cls, item_rect):
        """Centre of the clickable "mark for export" badge for an item at `item_rect`."""
        r = QRectF(item_rect).adjusted(6, 6, -6, -22)
        return QPointF(r.left() + 13, r.top() + 13)

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(option.rect).adjusted(6, 6, -6, -22)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        current = option.widget is not None and index == option.widget.currentIndex()
        pix = index.data(THUMB_ROLE)

        if pix is None:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.RAISED))
            p.drawRoundedRect(rect, 6, 6)
        else:
            p.setClipPath(_rounded_path(rect, 6))
            p.fillRect(rect, QColor('#0e0f12'))
            _draw_contain(p, pix, rect)
            p.setClipping(False)

        if current or selected:
            p.setPen(QPen(QColor(style.ACCENT if current else style.FAINT), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), 7, 7)

        # Badges
        kind = index.data(KIND_ROLE) or ''
        if kind:
            f = QFont(p.font())
            f.setPointSizeF(6.5)
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            bw = fm.horizontalAdvance(kind) + 8
            br = QRectF(rect.left() + 5, rect.bottom() - fm.height() - 7, bw, fm.height() + 2)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(br, 3, 3)
            p.setPen(QColor(style.TEXT))
            p.drawText(br, Qt.AlignmentFlag.AlignCenter, kind)
        if index.data(EDITED_ROLE):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.ACCENT))
            p.drawEllipse(QPointF(rect.right() - 9, rect.bottom() - 9), 4, 4)

        # Trash icon (always shown; brighter when the pointer is over this photo)
        hover_item = bool(option.state & QStyle.StateFlag.State_MouseOver)
        tr = self.trash_rect(option.rect)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 165 if hover_item else 120))
        p.drawRoundedRect(tr, 5, 5)
        ink = QColor(255, 255, 255, 235 if hover_item else 150)
        pen = QPen(ink, 1.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        cx, top = tr.center().x(), tr.top()
        p.drawLine(QPointF(cx - 5.5, top + 7), QPointF(cx + 5.5, top + 7))          # lid
        p.drawLine(QPointF(cx - 1.8, top + 5), QPointF(cx + 1.8, top + 5))          # handle
        body = QPainterPath()
        body.moveTo(cx - 4.2, top + 8.5)
        body.lineTo(cx - 3.4, top + 17)
        body.lineTo(cx + 3.4, top + 17)
        body.lineTo(cx + 4.2, top + 8.5)
        p.drawPath(body)
        p.drawLine(QPointF(cx - 1.2, top + 10.5), QPointF(cx - 1.0, top + 15))
        p.drawLine(QPointF(cx + 1.2, top + 10.5), QPointF(cx + 1.0, top + 15))

        # Mark-for-export badge: filled check when marked, empty ring on hover
        marked = bool(index.data(MARKED_ROLE))
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if marked or hover:
            c = self.mark_center(option.rect)
            rad = self.MARK_RADIUS
            if marked:
                p.setPen(QPen(QColor(0, 0, 0, 120), 1))
                p.setBrush(QColor(style.ACCENT))
                p.drawEllipse(c, rad, rad)
                pen = QPen(QColor('#1a1306'), 2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                p.setPen(pen)
                p.drawPolyline([c + QPointF(-4, 0), c + QPointF(-1, 3), c + QPointF(4.5, -3)])
            else:
                p.setPen(QPen(QColor(255, 255, 255, 220), 1.5))
                p.setBrush(QColor(0, 0, 0, 110))
                p.drawEllipse(c, rad - 1, rad - 1)

        # Name
        f = QFont(p.font())
        f.setPointSizeF(8)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QColor(style.TEXT if current else style.DIM))
        name_rect = QRectF(option.rect.left() + 4, rect.bottom() + 3, option.rect.width() - 8, 16)
        text = p.fontMetrics().elidedText(index.data(Qt.ItemDataRole.DisplayRole) or '',
                                          Qt.TextElideMode.ElideMiddle, int(name_rect.width()))
        p.drawText(name_rect, Qt.AlignmentFlag.AlignCenter, text)
        p.restore()


class LookDelegate(QStyledItemDelegate):
    SIZE = QSize(132, 112)

    def sizeHint(self, option, index):
        return self.SIZE

    def paint(self, p, option, index):
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hover = bool(option.state & QStyle.StateFlag.State_MouseOver)
        cell = QRectF(option.rect).adjusted(4, 4, -4, -4)
        img = QRectF(cell.left(), cell.top(), cell.width(), cell.height() - 22)

        if hover and not selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.RAISED))
            p.drawRoundedRect(cell.adjusted(-3, -3, 3, 3), 9, 9)

        p.save()
        p.setClipPath(_rounded_path(img, 7))
        _draw_cover(p, index.data(THUMB_ROLE), img)
        p.restore()

        if selected:
            p.setPen(QPen(QColor(style.ACCENT), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(img.adjusted(-1.5, -1.5, 1.5, 1.5), 8, 8)

        f = QFont(p.font())
        f.setPointSizeF(8.5)
        f.setBold(selected)
        p.setFont(f)
        p.setPen(QColor(style.ACCENT if selected else (style.TEXT if hover else style.DIM)))
        text_rect = QRectF(cell.left(), img.bottom() + 4, cell.width(), 18)
        p.drawText(text_rect, Qt.AlignmentFlag.AlignCenter,
                   p.fontMetrics().elidedText(index.data(Qt.ItemDataRole.DisplayRole), Qt.TextElideMode.ElideRight,
                                              int(text_rect.width())))
        p.restore()
