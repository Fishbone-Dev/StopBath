"""Full-resolution export."""
import os
import re
import threading
from pathlib import Path

import numpy as np

from . import imageio, lens, pipeline


def output_path(photo, settings, out_dir, suffix_look=True, fmt='jpeg'):
    look = re.sub(r'[^A-Za-z0-9]+', '-', settings.look).strip('-').lower()
    name = photo.stem + ('_' + look if suffix_look else '') + imageio.FORMATS[fmt][0]
    return Path(out_dir) / name


def lower_thread_priority():
    """Thread-pool initializer that lowers the calling thread's priority."""
    try:
        if os.name == 'nt':
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetThreadPriority(k32.GetCurrentThread(), -1)          # THREAD_PRIORITY_BELOW_NORMAL
        elif hasattr(os, 'setpriority'):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except Exception:
        pass


def export_photo(photo, settings_dict, out_path, quality=95, long_edge=0, fmt='jpeg',
                 submit=None, cancelled=None, progress=None):
    """Render one photo at full size and write it. The photo is rendered in bands, `cancelled()` is
    checked between bands, and the file is only written once rendering completes.
    Returns (out_path, error or None); a cancelled export returns the error 'cancelled'."""
    try:
        settings = pipeline.Settings.from_dict(settings_dict)
        source = photo.source(settings.source != 'jpeg')
        lin, raw, _ = imageio.load_linear(source)
        if cancelled is not None and cancelled():
            return str(out_path), 'cancelled'
        exif = imageio.read_exif(photo, source)
        rgb8 = pipeline.develop_in_bands(lin, raw, settings, lens.identify(exif), submit=submit,
                                         cancelled=cancelled, progress=progress)
        del lin
        if rgb8 is None:
            return str(out_path), 'cancelled'
        exif = imageio.read_exif(photo, source)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if long_edge and max(rgb8.shape[:2]) > long_edge:
            from PIL import Image
            im = Image.fromarray(rgb8)
            im.thumbnail((long_edge, long_edge), Image.LANCZOS)
            rgb8 = np.asarray(im)
        imageio.save_image(rgb8, out_path, exif, quality, fmt)
        return str(out_path), None
    except Exception as e:  # reported per photo; the batch continues
        return str(out_path), '%s: %s' % (type(e).__name__, e)


def export_one(photo, settings_dict, out_path, quality=95, long_edge=0, fmt='jpeg'):
    """Single-photo export without a thread pool (used by tests and the self-test)."""
    return export_photo(photo, settings_dict, out_path, quality, long_edge, fmt)
