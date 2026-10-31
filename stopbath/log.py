"""Logging to a file, so problems can be reported from the windowed Windows app (no console).

Log folder:  Windows  %APPDATA%\\Stop Bath\\logs      Linux  ~/.local/share/stop-bath/logs
  stopbath.log   what the app did, timings, and full error details (rotated, max ~3 x 2 MB)
  crash.log     low-level crash reports from native components (only if something hard-crashes)
"""
import faulthandler
import logging
import logging.handlers
import os
import platform
import shutil
import sys
import threading
from pathlib import Path

if os.name == 'nt':
    DATA_DIR = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming')) / 'Stop Bath'
    OLD_DATA_DIR = DATA_DIR.with_name('Film Lab')
else:
    DATA_DIR = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'stop-bath'
    OLD_DATA_DIR = DATA_DIR.with_name('film-lab')
LOG_DIR = DATA_DIR / 'logs'
LOG_FILE = LOG_DIR / 'stopbath.log'
log = logging.getLogger('stopbath')
_crash_file = None


def migrate_old_data():
    """Copy edits and recipes saved under the app's old name (Film Lab) on first start."""
    if DATA_DIR.exists() or not OLD_DATA_DIR.is_dir():
        return
    try:
        shutil.copytree(OLD_DATA_DIR, DATA_DIR, ignore=shutil.ignore_patterns('logs'))
    except OSError:
        pass


def setup():
    """Start logging (idempotent). Never raises."""
    global _crash_file
    if log.handlers:
        return log
    migrate_old_data()
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=2,
                                                       encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-7s %(threadName)-12s %(message)s'))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        _crash_file = open(LOG_DIR / 'crash.log', 'a', encoding='utf-8')
        faulthandler.enable(_crash_file)
    except Exception:
        log.addHandler(logging.NullHandler())

    def excepthook(kind, value, tb):
        log.critical('Unhandled error', exc_info=(kind, value, tb))
        sys.__excepthook__(kind, value, tb)

    def thread_excepthook(args):
        log.critical('Unhandled error in thread %s', args.thread.name if args.thread else '?',
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = excepthook
    threading.excepthook = thread_excepthook
    return log


def system_info():
    """One block of text describing the setup, for bug reports."""
    from . import __version__
    lines = ['Stop Bath %s' % __version__,
             'OS: %s %s (%s)' % (platform.system(), platform.release(), platform.version()),
             'Machine: %s, %s CPUs' % (platform.machine(), os.cpu_count()),
             'Python: %s%s' % (platform.python_version(), ' (packaged .exe)' if getattr(sys, 'frozen', False) else '')]
    for name, attr in (('numpy', '__version__'), ('PIL', '__version__'), ('rawpy', '__version__'),
                       ('lensfunpy', '__version__'), ('pillow_heif', '__version__')):
        try:
            mod = __import__(name)
            extra = ''
            if name == 'rawpy':
                extra = ' (LibRaw %s)' % '.'.join(str(v) for v in mod.libraw_version)
            lines.append('%s: %s%s' % (name, getattr(mod, attr, '?'), extra))
        except Exception as e:
            lines.append('%s: not available (%s)' % (name, e))
    try:
        from PyQt6.QtCore import QT_VERSION_STR
        from PyQt6.QtWidgets import QApplication
        lines.append('Qt: %s' % QT_VERSION_STR)
        app = QApplication.instance()
        if app is not None:
            for s in app.screens():
                g = s.geometry()
                lines.append('Screen: %dx%d at %.0f%% scaling' % (g.width(), g.height(), s.devicePixelRatio() * 100))
    except Exception:
        pass
    try:
        from .gui.app import available_ram_gb
        lines.append('Free memory: about %d GB' % available_ram_gb())
    except Exception:
        pass
    lines.append('Log folder: %s' % LOG_DIR)
    return '\n'.join(lines)


def tail(n=150):
    try:
        with open(LOG_FILE, encoding='utf-8', errors='replace') as f:
            return ''.join(f.readlines()[-n:])
    except OSError:
        return '(no log yet)'
