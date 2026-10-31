"""Lens corrections from the lensfun database: distortion, chromatic aberration (colour fringes) and
vignetting (dark corners).

Identification is strict, since a wrong profile is worse than none: the camera model must match
exactly, and the lens must either be the camera's only lens (fixed-lens compacts) or have a name whose
focal lengths and aperture agree with the photo's metadata.

Corrections are precomputed as smooth maps on a coarse grid (cached per lens, focal length, aperture
and image size) and looked up for any pixel, so a preview, a 100% region and the export sample the
photo at exactly the same places.
"""
import re
import threading
from dataclasses import dataclass

import numpy as np

try:
    import lensfunpy
except ImportError:          # lens corrections disabled
    lensfunpy = None

_lock = threading.RLock()   # re-entrant: Correction() resolves the lens while holding it
_db = None
_cache = {}


@dataclass(frozen=True)
class LensInfo:
    cam_maker: str
    cam_model: str
    lens_maker: str
    lens_model: str
    crop: float
    focal: float
    aperture: float

    @property
    def label(self):
        return '%s  ·  %g mm  ·  f/%g' % (self.lens_model, round(self.focal, 1), round(self.aperture, 1))


def available():
    return lensfunpy is not None


def _database():
    global _db
    with _lock:
        if _db is None and lensfunpy is not None:
            _db = lensfunpy.Database()
        return _db


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def _numbers(s):
    return [float(n) for n in re.findall(r'\d+(?:\.\d+)?', s or '')]


def _numbers_agree(exif_name, profile_name):
    """Every number in the photo's lens name (focal lengths, f-number) must appear in the profile's."""
    a, b = _numbers(exif_name), _numbers(profile_name)
    return bool(a) and all(any(abs(x - y) < 0.06 for y in b) for x in a)


def identify(exif):
    """LensInfo for a photo's metadata (a PIL Exif), or None if there's no trustworthy profile."""
    db = _database()
    if db is None or exif is None:
        return None
    x = exif.get_ifd(0x8769)
    make = str(exif.get(0x010F, '') or '').strip('\x00 ')
    model = str(exif.get(0x0110, '') or '').strip('\x00 ')
    lens_name = str(x.get(0xA434, '') or '').strip('\x00 ')
    if lens_name in ('----', 'None'):
        lens_name = ''
    if not model:
        return None
    try:
        focal = float(x.get(0x920A) or 0)
        aperture = float(x.get(0x829D) or 0)
    except (TypeError, ValueError):
        focal, aperture = 0.0, 0.0
    with _lock:
        cams = [c for c in db.find_cameras(make, model, True)
                if _norm(c.model) in (_norm(model), _norm(model).replace(_norm(make), ''))]
        if not cams:
            return None
        cam = cams[0]
        mount_lenses = db.find_lenses(cam)
        lens = None
        if len(mount_lenses) == 1:
            lens = mount_lenses[0]          # fixed-lens camera
        elif lens_name:
            for cand in db.find_lenses(cam, None, lens_name, True)[:8]:
                if _numbers_agree(lens_name, cand.model):
                    lens = cand
                    break
        if lens is None:
            return None
        if focal <= 0:
            focal = float(lens.min_focal or 0)
        if aperture <= 0:
            aperture = float(lens.min_aperture or 8.0)
        if focal <= 0:
            return None
        return LensInfo(cam.maker, cam.model, lens.maker, lens.model, float(cam.crop_factor), focal, aperture)


def _resolve(info):
    db = _database()
    cams = db.find_cameras(info.cam_maker, info.cam_model, False)
    if not cams:
        return None, None
    lenses = [l for l in db.find_lenses(cams[0], info.lens_maker, info.lens_model, False)
              if l.model == info.lens_model]
    return cams[0], (lenses[0] if lenses else None)


class Correction:
    """Coordinate and vignetting maps for one lens + settings + image size (source pixels)."""

    def __init__(self, info, h, w, distortion=True, tca=True, vignetting=True, vig_amount=1.0):
        self.h, self.w = h, w
        self.geo = None
        self.vig = None
        with _lock:
            cam, lens = _resolve(info)
            if lens is None:
                return
            flags = 0
            if distortion:
                flags |= lensfunpy.ModifyFlags.DISTORTION | lensfunpy.ModifyFlags.SCALE
            if tca:
                flags |= lensfunpy.ModifyFlags.TCA
            if flags:
                mod = lensfunpy.Modifier(lens, info.crop, w, h)
                mod.initialize(info.focal, info.aperture, 1000.0, 0.0 if distortion else 1.0,
                               pixel_format=np.float32, flags=flags)
                self.rows = np.unique(np.round(np.linspace(0, h - 1, max(2, h // 24))).astype(int))
                self.cols = np.unique(np.concatenate([np.arange(0, w, 6), [w - 1]]))
                grid = []
                for y in self.rows:
                    c = mod.apply_subpixel_geometry_distortion(0, int(y), w, 1)
                    if c is None:
                        grid = None
                        break
                    grid.append(c[0, self.cols])
                if grid is not None:
                    # Store displacements (smooth, small) rather than absolute coordinates
                    g = np.asarray(grid, np.float32)                      # (rows, cols, 3, 2)
                    g[..., 0] -= self.cols[None, :, None].astype(np.float32)
                    g[..., 1] -= self.rows[:, None, None].astype(np.float32)
                    if np.abs(g).max() > 0.01:
                        self.geo = g
            if vignetting:
                k = max(1, max(h, w) // 600)
                vh, vw = max(2, h // k), max(2, w // k)
                mod = lensfunpy.Modifier(lens, info.crop, vw, vh)
                mod.initialize(info.focal, info.aperture, 1000.0, 1.0, pixel_format=np.float32,
                               flags=lensfunpy.ModifyFlags.VIGNETTING)
                img = np.ones((vh, vw, 3), np.float32)
                if mod.apply_color_modification(img) and abs(float(img.max()) - 1) > 1e-3:
                    gain = img[..., 1].copy()
                    self.vig = (1.0 + (gain - 1.0) * float(vig_amount)).astype(np.float32)
                    self.vig_scale = (vh / h, vw / w)

    @property
    def active(self):
        return self.geo is not None or self.vig is not None

    def _grid_lookup(self, grid, y, x):
        """Bilinear lookup of a (rows, cols, ...) grid at full-res lensfun pixel coords."""
        fy = np.interp(y, self.rows, np.arange(len(self.rows), dtype=np.float64))
        fx = np.interp(x, self.cols, np.arange(len(self.cols), dtype=np.float64))
        y0 = np.minimum(np.floor(fy).astype(np.int32), len(self.rows) - 2)
        x0 = np.minimum(np.floor(fx).astype(np.int32), len(self.cols) - 2)
        ty = (fy - y0).astype(np.float32)[..., None, None]
        tx = (fx - x0).astype(np.float32)[..., None, None]
        a = grid[y0, x0] * (1 - tx) + grid[y0, x0 + 1] * tx
        b = grid[y0 + 1, x0] * (1 - tx) + grid[y0 + 1, x0 + 1] * tx
        return a * (1 - ty) + b * ty

    def source_coords(self, sx, sy):
        """Corrected-image coords (continuous, pixel centres at +0.5) -> per-channel coords in the
        photo as shot: three (x, y) pairs for red, green and blue."""
        if self.geo is None:
            return [(sx, sy)] * 3
        d = self._grid_lookup(self.geo, sy - 0.5, sx - 0.5)          # (..., 3, 2)
        return [(sx + d[..., c, 0], sy + d[..., c, 1]) for c in range(3)]

    def vignette_gain(self, sx, sy):
        if self.vig is None:
            return None
        vh, vw = self.vig.shape
        gy = np.clip(sy * self.vig_scale[0] - 0.5, 0, vh - 1)
        gx = np.clip(sx * self.vig_scale[1] - 0.5, 0, vw - 1)
        y0 = np.minimum(np.floor(gy).astype(np.int32), vh - 2)
        x0 = np.minimum(np.floor(gx).astype(np.int32), vw - 2)
        ty, tx = (gy - y0).astype(np.float32), (gx - x0).astype(np.float32)
        v = self.vig
        a = v[y0, x0] * (1 - tx) + v[y0, x0 + 1] * tx
        b = v[y0 + 1, x0] * (1 - tx) + v[y0 + 1, x0 + 1] * tx
        return a * (1 - ty) + b * ty


def correction(info, h, w, distortion=True, tca=True, vignetting=True, vig_amount=1.0):
    """Cached Correction, or None when nothing applies."""
    vignetting = vignetting and vig_amount > 0
    if info is None or lensfunpy is None or not (distortion or tca or vignetting):
        return None
    key = (info, h, w, distortion, tca, vignetting, round(float(vig_amount), 3))
    with _lock:
        if key in _cache:
            return _cache[key]
    corr = Correction(info, h, w, distortion, tca, vignetting, vig_amount)
    corr = corr if corr.active else None
    with _lock:
        if len(_cache) > 8:
            _cache.clear()
        _cache[key] = corr
    return corr
