"""Dark theme."""
from pathlib import Path

from PyQt6.QtGui import QColor, QPalette

ASSETS = (Path(__file__).parent / 'assets').as_posix()

BG = '#141519'
PANEL = '#1b1d22'
RAISED = '#24272e'
HOVER = '#2c3038'
BORDER = '#30343c'
TEXT = '#e8e8ea'
DIM = '#9399a3'
FAINT = '#61666f'
ACCENT = '#e9a23b'
ACCENT_HOVER = '#f4b85c'
DANGER = '#e0655a'

QSS = f"""
* {{
    color: {TEXT};
    font-size: 10pt;
    outline: none;
}}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#panel {{ background: {PANEL}; }}
QWidget#toolbar {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QWidget#filmstripBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

QLabel#title {{ font-size: 13pt; font-weight: 600; }}
QLabel#appName {{ font-size: 12pt; font-weight: 700; color: {ACCENT}; letter-spacing: 1px; }}
QLabel#dim {{ color: {DIM}; }}
QLabel:disabled {{ color: {FAINT}; }}
QLabel#faint {{ color: {FAINT}; font-size: 9pt; }}
QLabel#sectionTitle {{ color: {DIM}; font-size: 8.5pt; font-weight: 700; letter-spacing: 1.5px; }}
QLabel#value {{ color: {DIM}; font-family: monospace; font-size: 9pt; }}
QLabel#lookDescription {{ color: {DIM}; font-style: italic; }}
QLabel#emptyTitle {{ font-size: 20pt; font-weight: 300; color: {TEXT}; }}
QLabel#emptyText {{ color: {DIM}; font-size: 11pt; }}

QPushButton {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FAINT}; background: {PANEL}; }}
QPushButton:checked {{ background: {ACCENT}; color: #1a1306; border-color: {ACCENT}; font-weight: 600; }}
QPushButton#primary {{ background: {ACCENT}; color: #1a1306; border: none; font-weight: 700; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: {RAISED}; color: {FAINT}; }}
QPushButton#flat {{ background: transparent; border: none; color: {DIM}; padding: 4px 8px; }}
QPushButton#flat:hover {{ color: {TEXT}; background: {RAISED}; }}
QPushButton#segment {{ border-radius: 0; padding: 5px 12px; }}
QPushButton#segmentLeft {{ border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 5px 12px; }}
QPushButton#segmentRight {{ border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: none; padding: 5px 12px; }}
QPushButton::menu-indicator {{ width: 0; }}
QPushButton#danger {{ background: transparent; border: none; color: {DIM}; padding: 4px 8px; }}
QPushButton#danger:hover {{ color: #ffffff; background: {DANGER}; }}
QPushButton[small="true"] {{ padding: 3px 10px; font-size: 9pt; }}

QComboBox {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 10px;
}}
QComboBox:hover {{ background: {HOVER}; }}
QComboBox:disabled {{ color: {FAINT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({ASSETS}/chevron.svg); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {RAISED}; border: 1px solid {BORDER}; selection-background-color: {ACCENT};
    selection-color: #1a1306; padding: 4px;
}}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #4a4f58; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: {BORDER}; }}
QSlider::handle:horizontal {{
    background: {TEXT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::handle:horizontal:disabled {{ background: {FAINT}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {FAINT}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; margin: 2px; }}
QScrollBar::handle:horizontal:hover {{ background: {FAINT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QListWidget {{ background: transparent; border: none; }}
QListWidget#looks::item {{ border-radius: 8px; padding: 4px; color: {DIM}; }}
QListWidget#looks::item:hover {{ background: {RAISED}; color: {TEXT}; }}
QListWidget#looks::item:selected {{ background: {RAISED}; color: {ACCENT}; }}

QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; padding: 6px; border-radius: 6px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #1a1306; }}
QMenu::item:disabled {{ color: {FAINT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 5px 8px; }}

QLineEdit, QSpinBox {{
    background: {RAISED}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {ACCENT};
}}
QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px; border: 1px solid {FAINT}; background: {RAISED};
}}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

QProgressBar {{
    background: {RAISED}; border: none; border-radius: 4px; height: 8px; text-align: center; color: transparent;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QWidget#overlayBar {{ background: rgba(20, 21, 25, 215); border: 1px solid {BORDER}; border-radius: 10px; }}
QPushButton#overlay {{
    background: transparent; border: none; border-radius: 6px; padding: 4px 10px; color: {DIM}; font-weight: 600;
}}
QPushButton#overlay:hover {{ background: {HOVER}; color: {TEXT}; }}
QPushButton#overlay:checked {{ background: {RAISED}; color: {ACCENT}; }}
QLabel#overlayLabel {{ color: {TEXT}; font-family: monospace; font-size: 9pt; }}
QPushButton#sectionHeader {{
    background: transparent; border: none; text-align: left; padding: 6px 0 4px 0;
    color: {DIM}; font-size: 8.5pt; font-weight: 700; letter-spacing: 1.5px;
}}
QPushButton#sectionHeader:hover {{ color: {TEXT}; }}
QPushButton#sectionAction {{ background: transparent; border: 1px solid {BORDER}; border-radius: 9px; color: {DIM};
    padding: 1px 10px; font-size: 8.5pt; font-weight: 600; }}
QPushButton#sectionAction:hover {{ color: #1a1306; background: {ACCENT}; border-color: {ACCENT}; }}
QPushButton#autoToggle {{ font-weight: 700; }}
QPushButton#autoToggle:checked {{ background: {ACCENT}; color: #1a1306; border-color: {ACCENT}; }}
QPushButton#autoToggle:checked:hover {{ background: {ACCENT_HOVER}; }}
QWidget#stripTabRow {{ background: transparent; }}
QPushButton#stripTab {{
    background: {PANEL}; color: {DIM}; border: 1px solid {BORDER}; border-bottom: none;
    border-top-left-radius: 8px; border-top-right-radius: 8px; border-bottom-left-radius: 0;
    border-bottom-right-radius: 0; padding: 2px 16px 3px 16px; font-size: 8.5pt; font-weight: 600;
}}
QPushButton#stripTab:hover {{ color: {TEXT}; background: {RAISED}; }}
QPushButton#round {{ border-radius: 14px; min-width: 28px; max-width: 28px; min-height: 28px; max-height: 28px;
    padding: 0; font-weight: 700; }}
QWidget#toast {{ background: rgba(28, 30, 36, 245); border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#toastText {{ color: {TEXT}; font-weight: 500; }}
QLineEdit#valueEdit {{ padding: 1px 6px; border-radius: 4px; border: 1px solid {ACCENT}; font-family: monospace;
    font-size: 9pt; }}
QLabel#value:hover {{ color: {TEXT}; background: {RAISED}; border-radius: 4px; }}
QLabel#kbd {{ color: {TEXT}; background: {RAISED}; border: 1px solid {BORDER}; border-radius: 5px;
    padding: 2px 8px; font-family: monospace; font-size: 9pt; }}
QSlider::handle:horizontal:pressed {{ background: {ACCENT}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; border-radius: 4px; }}
QStatusBar {{ background: {PANEL}; color: {DIM}; border-top: 1px solid {BORDER}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {BORDER}; }}
"""


def palette():
    p = QPalette()
    for role, color in [
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.Base, RAISED),
        (QPalette.ColorRole.AlternateBase, PANEL), (QPalette.ColorRole.Text, TEXT),
        (QPalette.ColorRole.WindowText, TEXT), (QPalette.ColorRole.Button, RAISED),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, ACCENT),
        (QPalette.ColorRole.HighlightedText, '#1a1306'), (QPalette.ColorRole.ToolTipBase, RAISED),
        (QPalette.ColorRole.ToolTipText, TEXT), (QPalette.ColorRole.PlaceholderText, FAINT),
    ]:
        p.setColor(role, QColor(color))
    return p
