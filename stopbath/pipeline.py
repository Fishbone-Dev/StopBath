"""Image processing: turns a decoded photo plus Settings into the finished look.

All images are float32 numpy arrays shaped (H, W, 3). "Linear" means scene-linear light;
"display" means sRGB-encoded values in 0..1.

Order of operations:
  linear source -> exposure / white balance -> tone map (RAW) or re-encode (JPEG)
  -> noise reduction -> look (color/mono, contrast, split toning, fade) blended by strength
  -> LUT -> halation -> vignette -> grain -> clip

Halation, vignette and grain depend on where a pixel sits in the whole photo. A `Frame` describes
which part of the full-resolution photo an array covers and at what scale, so the screen preview,
a 100% zoomed region and the final export all produce the same result for the same spot.
"""
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np

from . import geometry as geometry_mod
from . import lens as lens_mod
from . import looks as looks_mod
from .lut import Lut

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

BW_FILTERS = {
    # 'Film': the film profile's own response to colour (film looks only).
    # Others: weights applied to display RGB, each summing to 1 so white stays white.
    'Film': None,
    'Neutral': (0.30, 0.59, 0.11),
    'Yellow': (0.45, 0.50, 0.05),
    'Orange': (0.65, 0.35, 0.00),
    'Red': (0.90, 0.15, -0.05),
    'Green': (0.20, 0.75, 0.05),
    'Blue': (0.10, 0.30, 0.60),
}

TONINGS = {
    # Midtone color of a gradient map black -> mid -> white (luma of each ~0.5).
    'None': None,
    'Sepia': (0.60, 0.48, 0.36),
    'Selenium': (0.52, 0.47, 0.50),
    'Platinum': (0.56, 0.50, 0.43),
    'Cyanotype': (0.32, 0.52, 0.66),
}


@dataclass
class Settings:
    look: str = 'Natural'
    strength: float = 1.0
    # Source for RAW+JPEG pairs: 'raw' or 'jpeg' (single-file photos always use what they have)
    source: str = 'raw'
    # Crop & straighten (see geometry.py)
    rot90: int = 0
    angle: float = 0.0
    crop_cx: float = 0.5
    crop_cy: float = 0.5
    crop_w: float = 1.0
    crop_h: float = 1.0
    crop_aspect: str = 'Free'
    crop_portrait: bool = False
    # Lens corrections (lensfun): 'auto' = RAW files only (camera JPEGs are already corrected)
    lens_mode: str = 'auto'
    lens_distortion: bool = True
    lens_tca: bool = True
    lens_vignetting: bool = True
    lens_vig_amount: float = 1.0   # 0..1, how much of the vignetting to remove
    # Auto: on/off, and the values Auto replaced (JSON) so turning it off restores them
    auto_on: bool = False
    auto_prev: str = ''
    # Detail
    sharpen: float = 0.0      # 0..1
    # Noise reduction
    nr_luma: float = 0.0      # 0..1
    nr_color: float = 0.0     # 0..1
    # Light
    exposure: float = 0.0     # stops, -3..3
    contrast: float = 0.0     # -1..1, added to the look's contrast
    highlights: float = 0.0   # -1..1 (negative recovers bright skies)
    shadows: float = 0.0      # -1..1 (positive opens up dark areas)
    fade: float = 0.0         # 0..0.3, added to the look's black lift
    # Color
    warmth: float = 0.0       # -1..1
    tint: float = 0.0         # -1..1 (negative = green, positive = magenta)
    saturation: float = 0.0   # -1..1, relative to the look
    # Black & white (used by mono looks)
    bw_filter: str = 'Neutral'
    toning: str = 'None'
    toning_amount: float = 0.0
    # Film effects
    grain: float = 0.0        # 0..1
    grain_size: float = 1.0   # 0.5..3
    halation: float = 0.0     # 0..1
    vignette: float = 0.0     # -1..1 (positive darkens edges)
    # LUT
    lut_path: str = ''
    lut_amount: float = 1.0

    def copy(self):
        return Settings(**asdict(self))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def without_geometry(self):
        """Copy with crop/rotation (and the Auto toggle, which belongs to one photo's analysis)
        reset: used when settings carry over to a different photo."""
        s = self.copy()
        for f in GEOMETRY_FIELDS + ('auto_on', 'auto_prev'):
            setattr(s, f, getattr(Settings, f))
        return s

    def with_geometry_of(self, other):
        s = self.copy()
        for f in GEOMETRY_FIELDS:
            setattr(s, f, getattr(other, f))
        return s

    def geometry(self, src_h, src_w):
        return geometry_mod.from_settings(self, src_h, src_w)

    def with_look_defaults(self, look_name):
        """New settings with the look's own effect defaults; personal light/color tweaks are kept."""
        s = self.copy()
        look = looks_mod.get(look_name)
        s.look = look.name
        s.grain = look.grain
        s.grain_size = look.grain_size
        s.halation = look.halation
        s.vignette = look.vignette
        s.bw_filter = look.bw_filter
        s.toning = look.toning
        s.toning_amount = look.toning_amount
        return s


GEOMETRY_FIELDS = ('rot90', 'angle', 'crop_cx', 'crop_cy', 'crop_w', 'crop_h', 'crop_aspect', 'crop_portrait')


# Color space helpers

def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, np.power((np.maximum(x, 0) + 0.055) / 1.055, 2.4)).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def luma(x):
    return x @ LUMA


def tonemap(lin):
    """Filmic curve for RAW (Narkowicz ACES fit), returns display-linear 0..1.
    The 1.6 input gain was calibrated so a7II RAWs match the brightness of the camera's own JPEGs."""
    x = np.maximum(lin, 0) * 1.6
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


# Blur (no scipy needed)

def _box_blur_axis(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode='edge'), axis=axis, dtype=np.float64)
    n = a.shape[axis]
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gaussian_blur(a, sigma):
    """Approximate Gaussian blur with three box passes along each axis (axes 0 and 1)."""
    if sigma < 0.5:
        return a
    # Box radius giving the requested sigma over three passes
    r = max(1, int(round((np.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    out = a
    for _ in range(3):
        out = _box_blur_axis(out, r, 0)
        out = _box_blur_axis(out, r, 1)
    return out


def _box2d(a, r):
    """Mean over a (2r+1) x (2r+1) window (edge-padded), on axes 0 and 1."""
    return _box_blur_axis(_box_blur_axis(a, r, 0), r, 1)


def _guided_filter(guide, src, r, eps):
    """Edge-preserving smoothing of `src` steered by `guide` (He et al., "Guided Image Filtering").
    Flat areas (local variance << eps) are averaged; edges (variance >> eps) are kept.
    `guide` is (H, W); `src` is (H, W) or (H, W, C)."""
    # 32-bit working arrays (the running sums inside _box2d are 64-bit, which is where precision matters)
    g = guide.astype(np.float32, copy=False)
    p = src.astype(np.float32, copy=False)
    if p.ndim == 3:
        g3 = g[..., None]
    else:
        g3 = g
    mean_g = _box2d(g, r)
    var_g = _box2d(g * g, r) - mean_g * mean_g
    mean_g3 = mean_g[..., None] if p.ndim == 3 else mean_g
    var_g3 = var_g[..., None] if p.ndim == 3 else var_g
    mean_p = _box2d(p, r)
    cov = _box2d(g3 * p, r) - mean_g3 * mean_p
    a = cov / (var_g3 + eps)
    b = mean_p - a * mean_g3
    return (_box2d(a, r) * g3 + _box2d(b, r)).astype(np.float32)


# Pixels of context a region render needs on each side for denoising to match a full render.
DENOISE_MARGIN = 48


def denoise(x, luma_amount, color_amount, frame):
    """High-ISO noise reduction on a display-space image.

    Luminance: self-guided filter on brightness, radius 2 px (full resolution); the amount sets how
    strong an edge must be to survive. Color: the two color-difference channels are smoothed over a
    wider radius, steered by brightness so colors don't bleed across edges. Radii scale with
    `frame.scale` so a downscaled preview gets a comparable, if approximate, result."""
    if luma_amount <= 0 and color_amount <= 0:
        return x
    sc = frame.scale
    Y = luma(x)
    cr = x[..., 0] - Y
    cb = x[..., 2] - Y
    if luma_amount > 0:
        r = max(1, int(round(2 * sc)))
        eps = (0.075 * luma_amount) ** 2
        Y = _guided_filter(Y, Y, r, eps)
    if color_amount > 0:
        r = max(1, int(round((3 + 9 * color_amount) * sc)))
        chroma = np.stack([cr, cb], axis=2)
        smooth = _guided_filter(Y, chroma, r, (0.03 + 0.05 * color_amount) ** 2)
        chroma = chroma + (smooth - chroma) * np.float32(min(1.0, 0.4 + color_amount))
        cr, cb = chroma[..., 0], chroma[..., 1]
    # Rebuild RGB: luma(rgb) must equal Y, which fixes green from red and blue
    wr, wg, wb = (float(v) for v in LUMA)
    R = Y + cr
    B = Y + cb
    G = Y - (wr * cr + wb * cb) / wg
    return np.stack([R, G, B], axis=2).astype(np.float32)


def highlights_shadows(x, highlights, shadows):
    """Brighten/darken the dark and bright ends of the tonal range without clipping.
    Both curves are monotonic for the full slider range, so tones never reverse."""
    if not highlights and not shadows:
        return x
    L = np.clip(luma(x), 0, 1)
    a = np.float32(0.4 * shadows)
    b = np.float32(-0.4 * highlights)
    L2 = L + a * 2.5 * L * (1 - L) ** 2 - b * 2.5 * L * L * (1 - L)
    return np.clip(x + (L2 - L)[..., None], 0, 1).astype(np.float32)


def sharpen(x, amount, frame):
    """Unsharp mask on brightness (no colour fringes), radius ~1 px at full resolution."""
    if amount <= 0:
        return x
    sigma = 1.0 * frame.scale
    if sigma < 0.5:
        return x   # finer than a screen pixel: nothing visible at this zoom
    L = luma(x)
    detail = L - gaussian_blur(L, sigma)
    return np.clip(x + (detail * np.float32(1.6 * amount))[..., None], 0, 1).astype(np.float32)


def _sample_bilinear(a, ys, xs):
    """Sample 2-D array `a` at fractional row coordinates `ys` and column coordinates `xs` (1-D each)."""
    h, w = a.shape
    ys = np.clip(ys, 0, h - 1)
    xs = np.clip(xs, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64); y1 = np.minimum(y0 + 1, h - 1); fy = (ys - y0).astype(np.float32)[:, None]
    x0 = np.floor(xs).astype(np.int64); x1 = np.minimum(x0 + 1, w - 1); fx = (xs - x0).astype(np.float32)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


@dataclass
class Frame:
    """Where an image array sits within the full-resolution photo.

    full_h, full_w: size of the full-resolution photo
    scale: array pixels per full-resolution pixel (1 = 100%, 0.3 = a downscaled preview)
    y0, x0: top-left corner of the array, in full-resolution pixels
    """
    full_h: int
    full_w: int
    scale: float = 1.0
    y0: float = 0.0
    x0: float = 0.0
    grain_ref: float = 0.0   # long side of the uncropped photo; grain is sized relative to it

    @property
    def grain_reference(self):
        return self.grain_ref or max(self.full_h, self.full_w)

    def full_coords(self, h, w):
        """Full-resolution coordinates of the centres of an h x w array's pixels."""
        ys = self.y0 + (np.arange(h, dtype=np.float64) + 0.5) / self.scale
        xs = self.x0 + (np.arange(w, dtype=np.float64) + 0.5) / self.scale
        return ys, xs


# Stages

def develop_base(lin, is_raw, s):
    """Exposure + white balance in linear light, then to display space."""
    gain = np.float32(2.0 ** s.exposure)
    wb = np.array([1 + 0.18 * s.warmth, 1 - 0.12 * s.tint, 1 - 0.18 * s.warmth], np.float32)
    wb = wb / (wb @ LUMA)  # keep brightness constant
    x = lin * (gain * wb)
    if is_raw:
        return linear_to_srgb(tonemap(x))
    return linear_to_srgb(x)


def s_curve(x, c):
    """Contrast around mid grey. c in -1..1; monotonic for the whole range."""
    c = float(np.clip(c, -1, 1))
    if c == 0:
        return x
    x = np.clip(x, 0, 1)
    return x + c * (x * x * (3 - 2 * x) - x)


def _apply_film_look(x, look, s):
    """Film looks: the stock's emulation profile, then the contrast / saturation / fade sliders."""
    mono = look.mono
    src = x
    w = BW_FILTERS.get(s.bw_filter)
    if mono and w is not None:
        # A coloured lens filter: decide the grey value ourselves, then let the film shape the tones
        L = np.clip(x @ np.asarray(w, np.float32), 0, 1)
        src = np.repeat(L[..., None], 3, axis=2)
    y = _get_lut(look.lut_path).apply(src)
    if mono:
        y = np.repeat(np.clip(luma(y), 0, 1)[..., None], 3, axis=2)
    elif s.saturation:
        L = luma(y)[..., None]
        y = L + (y - L) * np.float32(1 + s.saturation)
    y = s_curve(np.clip(y, 0, 1), s.contrast)
    fade = float(np.clip(s.fade, 0, 0.4))
    if fade:
        y = fade + np.clip(y, 0, 1) * (1 - fade)
    if mono:
        y = _tone(y, s)
    return np.clip(y, 0, 1).astype(np.float32)


def _tone(y, s):
    tone = TONINGS.get(s.toning)
    if tone is None or s.toning_amount <= 0:
        return y
    L = y[..., :1]
    mid = np.asarray(tone, np.float32)
    mapped = np.where(L < 0.5, 2 * L * mid, mid + (2 * L - 1) * (1 - mid))
    return y + (mapped - y) * np.float32(s.toning_amount)


def apply_look(x, look, s):
    if look.lut and Path(look.lut_path).is_file():
        return _apply_film_look(x, look, s)
    mono = look.mono
    if mono:
        w = BW_FILTERS.get(s.bw_filter) or BW_FILTERS['Neutral']
        L = np.clip(x @ np.asarray(w, np.float32), 0, 1)
        y = np.repeat(L[..., None], 3, axis=2)
    else:
        y = x @ np.asarray(look.matrix, np.float32).T
        sat = look.saturation * (1 + s.saturation)
        L = luma(y)[..., None]
        y = L + (y - L) * sat

    y = s_curve(y, look.contrast + s.contrast)

    if not mono:
        L = np.clip(luma(y), 0, 1)[..., None]
        y = y + (1 - L) ** 2 * np.asarray(look.shadow_tint, np.float32) \
              + L ** 2 * np.asarray(look.highlight_tint, np.float32)

    fade = float(np.clip(look.fade + s.fade, 0, 0.4))
    y = fade + np.clip(y, 0, 1) * (1 - fade - look.roll)

    if mono:
        y = _tone(y, s)
    return np.clip(y, 0, 1).astype(np.float32)


def _block_mean(x, max_side):
    h, w = x.shape[:2]
    k = max(1, int(np.ceil(max(h, w) / max_side)))
    if k == 1:
        return x
    return x[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k, -1).mean(axis=(1, 3), dtype=np.float32)


def glow_map(glow_source):
    """Low-resolution, blurred highlight map of the whole (styled) photo, used for halation."""
    small = _block_mean(glow_source, 800)
    hl = np.maximum(luma(srgb_to_linear(small)) - 0.55, 0) / 0.45
    return gaussian_blur(hl, sigma=0.012 * max(small.shape[:2]))


def apply_halation(x, amount, frame, glow_source=None, glow=None):
    """Red glow around highlights. The glow comes from the whole photo (`glow_source` at any
    resolution, or a precomputed `glow_map`) and is sampled at this array's position, so a
    zoomed-in region glows exactly like the same spot in the export."""
    if amount <= 0:
        return x
    if glow is None:
        glow = glow_map(glow_source)
    gh, gw = glow.shape
    ys, xs = frame.full_coords(*x.shape[:2])
    glow = _sample_bilinear(glow, ys * gh / frame.full_h - 0.5, xs * gw / frame.full_w - 0.5)[..., None]
    color = np.array([1.0, 0.28, 0.10], np.float32)
    out = srgb_to_linear(x) + glow * color * np.float32(0.6 * amount)
    return linear_to_srgb(out)


def apply_vignette(x, amount, frame):
    if amount == 0:
        return x
    H, W = frame.full_h, frame.full_w
    m = max(H, W) / 2
    ys, xs = frame.full_coords(*x.shape[:2])
    yy = ((ys - H / 2) / m).astype(np.float32)
    xx = ((xs - W / 2) / m).astype(np.float32)
    corner = np.hypot(W / 2 / m, H / 2 / m)
    r = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2) / np.float32(corner)  # 0 at centre, 1 in the corners
    t = np.clip((r - 0.35) / 0.9, 0, 1)
    fall = t * t * (3 - 2 * t)
    if amount > 0:
        factor = 1 - 0.75 * amount * fall
        return (x * factor[..., None]).astype(np.float32)
    return (x + (1 - x) * (-amount * 0.6 * fall)[..., None]).astype(np.float32)


def _hash_gauss(iy, ix, seed):
    """Deterministic standard-normal value for each integer lattice point (iy, ix): the same point
    always gets the same value, whichever region of the photo is being rendered."""
    iy = (np.asarray(iy, np.int64) & 0xFFFFFFFF).astype(np.uint32)
    ix = (np.asarray(ix, np.int64) & 0xFFFFFFFF).astype(np.uint32)

    def mix(salt):
        h = iy * np.uint32(0x8DA6B343) ^ ix * np.uint32(0xD8163841) ^ np.uint32((seed * 0x9E3779B9 + salt) & 0xFFFFFFFF)
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
        return ((h >> np.uint32(8)).astype(np.float32) + 0.5) / np.float32(1 << 24)

    u1, u2 = mix(0x1234567), mix(0x7654321)
    return (np.sqrt(-2 * np.log(u1)) * np.cos(np.float32(2 * np.pi) * u2)).astype(np.float32)


def apply_grain(x, amount, size, frame, seed=7):
    """Monochrome film grain, strongest in the midtones. Grain size is relative to a 6000 px frame.

    Grain lives on a lattice fixed to the full-resolution photo, so any region rendered at 100%
    shows exactly the grains the export will have. When grains are smaller than a screen pixel
    (zoomed-out previews) they are drawn per pixel at reduced strength, as they'd average out."""
    if amount <= 0:
        return x
    h, w = x.shape[:2]
    px_full = size * 1.6 * frame.grain_reference / 6000.0             # grain size in full-res pixels
    px_here = px_full * frame.scale                                     # ... in this array's pixels
    if px_here > 1:
        ys, xs = frame.full_coords(h, w)
        u, v = ys / px_full, xs / px_full
        iy0, ix0 = int(np.floor(u.min())), int(np.floor(v.min()))
        iy1, ix1 = int(np.floor(u.max())) + 1, int(np.floor(v.max())) + 1
        lattice = _hash_gauss(np.arange(iy0, iy1 + 1)[:, None], np.arange(ix0, ix1 + 1)[None, :], seed)
        # Bilinear interpolation of unit-variance noise has a mean variance of 4/9: rescale to 1
        noise = _sample_bilinear(lattice, u - iy0, v - ix0) * np.float32(1.5)
        amp = 1.0
    else:
        oy = int(round(frame.y0 * frame.scale))
        ox = int(round(frame.x0 * frame.scale))
        noise = _hash_gauss(np.arange(oy, oy + h)[:, None], np.arange(ox, ox + w)[None, :], seed + 1)
        amp = px_here
    L = np.clip(luma(x), 0, 1)
    weight = (4 * L * (1 - L)) ** 0.7 + 0.12
    return (x + (noise * weight * np.float32(0.11 * amount * amp))[..., None]).astype(np.float32)


_lut_cache = OrderedDict()
_lut_lock = threading.Lock()


def _get_lut(path):
    """Loaded LUTs, most recently used kept (a film look and a LUT are often active together).
    Thread-safe: previews, zoomed regions and the loupe render in parallel."""
    with _lut_lock:
        if path in _lut_cache:
            _lut_cache.move_to_end(path)
            return _lut_cache[path]
    lut = Lut.load(path)
    with _lut_lock:
        _lut_cache[path] = lut
        while len(_lut_cache) > 6:
            _lut_cache.popitem(last=False)
    return lut


def _style(lin, is_raw, s, frame=None):
    """Base development, noise reduction, look, strength and LUT."""
    base = develop_base(lin, is_raw, s)
    base = highlights_shadows(base, s.highlights, s.shadows)
    if frame is None:
        frame = Frame(lin.shape[0], lin.shape[1])
    if s.nr_luma > 0 or s.nr_color > 0:
        base = np.clip(denoise(base, s.nr_luma, s.nr_color, frame), 0, 1)
    base = sharpen(base, s.sharpen, frame)
    styled = apply_look(base, looks_mod.get(s.look), s)
    if s.strength < 1:
        styled = base + (styled - base) * np.float32(max(0.0, s.strength))
    if s.lut_path and Path(s.lut_path).is_file():   # a LUT file that has since moved is skipped
        lut = _get_lut(s.lut_path)
        styled = styled + (lut.apply(styled) - styled) * np.float32(s.lut_amount)
    return styled


_glow_cache = {}


def _context_glow(context, is_raw, s):
    """Glow map for region renders. Cached: it only depends on the photo and the per-pixel settings,
    and the loupe asks for it on every mouse move."""
    per_pixel = s.copy()
    per_pixel.grain = per_pixel.grain_size = per_pixel.halation = per_pixel.vignette = 0
    per_pixel.nr_luma = per_pixel.nr_color = 0  # irrelevant to a heavily blurred glow
    probe = context[:: max(1, context.shape[0] // 7), :: max(1, context.shape[1] // 7)]
    key = (id(context), context.shape, probe.tobytes(), is_raw, tuple(sorted(per_pixel.to_dict().items())))
    with _lut_lock:
        glow = _glow_cache.get(key)
    if glow is None:
        glow = glow_map(_style(_block_mean(context, 800), is_raw, per_pixel))
        with _lut_lock:
            _glow_cache.clear()
            _glow_cache[key] = glow
    return glow


def render(lin, is_raw, s, frame=None, context=None, grain_seed=7):
    """Full pipeline. `lin` is the decoded image (or a region of it) in linear light.

    frame:   where `lin` sits in the full photo (default: `lin` is the whole photo at 100%)
    context: the whole photo at any resolution, needed for halation when `lin` is only a region
    """
    if frame is None:
        frame = Frame(lin.shape[0], lin.shape[1])
    styled = _style(lin, is_raw, s, frame)
    if s.halation > 0:
        if context is None:
            glow = glow_map(styled)
        else:
            glow = _context_glow(context, is_raw, s)
        out = apply_halation(styled, s.halation, frame, glow=glow)
    else:
        out = styled
    out = apply_vignette(out, s.vignette, frame)
    out = apply_grain(out, s.grain, s.grain_size, frame, grain_seed)
    return np.clip(out, 0, 1)


def output_frame(s, src_full, scale=1.0):
    """Frame for the whole cropped output at `scale`, and the array shape that goes with it."""
    geo = s.geometry(*src_full)
    oh, ow = geo.out_size
    frame = Frame(oh, ow, scale, grain_ref=max(src_full))
    shape = (max(1, int(round(oh * scale))), max(1, int(round(ow * scale))))
    return geo, frame, shape


LENS_FIELDS = ('lens_mode', 'lens_distortion', 'lens_tca', 'lens_vignetting')


def lens_correction(s, is_raw, lens_info, src_full):
    """The lens.Correction to apply for these settings, or None."""
    if lens_info is None or s.lens_mode == 'off' or (s.lens_mode == 'auto' and not is_raw):
        return None
    return lens_mod.correction(lens_info, src_full[0], src_full[1],
                               s.lens_distortion, s.lens_tca, s.lens_vignetting, s.lens_vig_amount)


def warp_source(src, src_full, src_scale, s, frame, shape, lens=None):
    """Apply lens corrections and crop/rotation: the output region `frame` sampled from `src`
    (the source at `src_scale`)."""
    geo = s.geometry(*src_full)
    if lens is None and geo.is_identity and abs(src_scale - frame.scale) < 1e-9 and frame.x0 == 0 \
            and frame.y0 == 0 and src.shape[:2] == tuple(shape):
        return src
    return geometry_mod.warp(src, src_scale, geo, frame, shape, lens)


def develop(src, is_raw, s, lens_info=None):
    """Export path: full-resolution source in, finished full-resolution (cropped) image out."""
    src_full = src.shape[:2]
    geo, frame, shape = output_frame(s, src_full)
    lin = warp_source(src, src_full, 1.0, s, frame, shape, lens_correction(s, is_raw, lens_info, src_full))
    return render(lin, is_raw, s, frame=frame)


def render_region(full_src, is_raw, s, rect, lens_info=None, context=None, k=1, with_original=False):
    """Render output rect (x, y, w, h) of the finished (cropped) photo from the full-resolution source,
    at 1/k scale. Exactly matches the same area of a whole-photo render with the same `context`.
    Returns (after, before or None, (x, y, w, h) actually rendered), or None if the rect is empty."""
    src_full = full_src.shape[:2]
    geo, _, _ = output_frame(s, src_full)
    H, W = geo.out_size
    x, y, w, h = rect
    x, y = (x // k) * k, (y // k) * k
    w = min(W - x, int(math.ceil(w / k)) * k) // k * k
    h = min(H - y, int(math.ceil(h / k)) * k) // k * k
    if w <= 0 or h <= 0:
        return None
    # Denoising and sharpening look at neighbouring pixels: render with a margin, then trim it off
    m = 0
    if s.nr_luma > 0 or s.nr_color > 0 or s.sharpen > 0:
        m = int(math.ceil(DENOISE_MARGIN / k)) * k
    px0, py0 = max(0, x - m), max(0, y - m)
    px1, py1 = min(W, x + w + m), min(H, y + h + m)
    px1 = px0 + (px1 - px0) // k * k
    py1 = py0 + (py1 - py0) // k * k
    ref = max(src_full)
    lin = warp_source(full_src, src_full, 1.0, s, Frame(H, W, 1.0, py0, px0, grain_ref=ref),
                      (py1 - py0, px1 - px0), lens_correction(s, is_raw, lens_info, src_full))
    if k > 1:
        lin = lin.reshape((py1 - py0) // k, k, (px1 - px0) // k, k, 3).mean(axis=(1, 3), dtype=np.float32)
    frame = Frame(H, W, 1.0 / k, py0, px0, grain_ref=ref)
    ty, tx = (y - py0) // k, (x - px0) // k
    after = render(lin, is_raw, s, frame=frame, context=context)[ty:ty + h // k, tx:tx + w // k]
    before = render_original(lin, is_raw)[ty:ty + h // k, tx:tx + w // k] if with_original else None
    return after, before, (x, y, w, h)


def glow_context(full_src, is_raw, s, lens_info=None, side=900):
    """A small render-ready view of the whole cropped photo, used for halation by region renders."""
    src_full = full_src.shape[:2]
    geo, _, _ = output_frame(s, src_full)
    oh, ow = geo.out_size
    scale = min(1.0, side / max(oh, ow))
    frame = Frame(oh, ow, scale, grain_ref=max(src_full))
    shape = (max(1, int(round(oh * scale))), max(1, int(round(ow * scale))))
    k = max(1, int(1 / scale))
    small_src = full_src[::k, ::k]
    return warp_source(small_src, src_full, small_src.shape[1] / src_full[1], s, frame, shape,
                       lens_correction(s, is_raw, lens_info, src_full))


def develop_in_bands(full_src, is_raw, s, lens_info=None, band_rows=256, workers=1, cancelled=None,
                     progress=None, submit=None):
    """Export path with bounded memory: renders the finished photo in horizontal bands (in parallel
    when a `submit` function from a thread pool is given) and assembles 8-bit output.
    Returns the uint8 image, or None if `cancelled()` became true."""
    src_full = full_src.shape[:2]
    geo, _, _ = output_frame(s, src_full)
    H, W = geo.out_size
    context = glow_context(full_src, is_raw, s, lens_info) if s.halation > 0 else None
    out = np.empty((H, W, 3), np.uint8)
    rows = list(range(0, H, band_rows))
    done = [0]

    def band(y):
        if cancelled is not None and cancelled():
            return False
        after, _, (bx, by, bw, bh) = render_region(full_src, is_raw, s, (0, y, W, band_rows), lens_info, context)
        out[by:by + bh, bx:bx + bw] = to_uint8(after)
        done[0] += 1
        if progress is not None:
            progress(done[0] / len(rows))
        return True

    if submit is None:
        for y in rows:
            if not band(y):
                return None
    else:
        futures = [submit(band, y) for y in rows]
        ok = all(f.result() for f in futures)
        if not ok or (cancelled is not None and cancelled()):
            return None
    return out


def render_original(lin, is_raw):
    """The photo with no look or adjustments, for before/after comparison."""
    return develop_base(lin, is_raw, Settings())


def to_uint8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
