"""Crop, straighten and 90° rotation.

Coordinate spaces (all in full-resolution pixels, y pointing down):
  source - the decoded, upright photo (src_w x src_h)
  frame  - the source after 90° rotations (frame_w x frame_h)
  output - the crop: an axis-aligned rectangle in the *straightened* frame, out_w x out_h

The crop is stored normalized to the frame: centre (cx, cy) and size (w, h) as fractions of
frame_w / frame_h, plus the straighten angle in degrees. A positive angle turns the photo
clockwise: an output point at offset d from the crop centre comes from frame point C + R(-angle)·d.
Crops are always kept fully inside the frame, so no empty corners can appear.
"""
import math
from dataclasses import dataclass

import numpy as np

ASPECTS = ['Free', 'Original', '1:1', '4:5', '3:2', '4:3', '16:9', '65:24 XPan']
_RATIOS = {'1:1': 1.0, '4:5': 4 / 5, '3:2': 3 / 2, '4:3': 4 / 3, '16:9': 16 / 9, '65:24 XPan': 65 / 24}
MIN_SIZE = 0.03   # smallest crop, as a fraction of the frame


def aspect_ratio(name, frame_w, frame_h, portrait=False):
    """Width/height ratio for an aspect preset, or None for Free."""
    if name == 'Original':
        r = frame_w / frame_h
        # "Original" follows the photo; the portrait toggle flips it like the others
        r = max(r, 1 / r)
    else:
        r = _RATIOS.get(name)
        if r is None:
            return None
    r = max(r, 1 / r)
    return 1 / r if portrait else r


@dataclass(frozen=True)
class Geometry:
    src_h: int
    src_w: int
    rot90: int = 0          # quarter turns clockwise
    angle: float = 0.0      # degrees
    cx: float = 0.5
    cy: float = 0.5
    w: float = 1.0
    h: float = 1.0

    @property
    def frame_size(self):
        """(h, w) of the frame after 90° rotations."""
        return (self.src_h, self.src_w) if self.rot90 % 2 == 0 else (self.src_w, self.src_h)

    @property
    def out_size(self):
        fh, fw = self.frame_size
        return max(1, int(round(self.h * fh))), max(1, int(round(self.w * fw)))

    @property
    def is_identity(self):
        return (self.rot90 % 4 == 0 and self.angle == 0 and self.w >= 1 and self.h >= 1
                and self.cx == 0.5 and self.cy == 0.5)

    def crop_center(self):
        """Crop centre in frame px, snapped so an unrotated crop starts on a whole pixel."""
        fh, fw = self.frame_size
        oh, ow = self.out_size
        x0 = round(self.cx * fw - ow / 2)
        y0 = round(self.cy * fh - oh / 2)
        return x0 + ow / 2, y0 + oh / 2

    def output_to_frame(self, u, v):
        """Output coordinates -> frame coordinates (arrays broadcast)."""
        oh, ow = self.out_size
        cx, cy = self.crop_center()
        du, dv = u - ow / 2, v - oh / 2
        if self.angle == 0:
            return cx + du, cy + dv
        a = math.radians(self.angle)
        c, s = math.cos(a), math.sin(a)
        return cx + c * du + s * dv, cy - s * du + c * dv

    def frame_to_source(self, xf, yf):
        k = self.rot90 % 4
        if k == 0:
            return xf, yf
        if k == 1:      # clockwise: source (x, y) -> frame (H - y, x)
            return yf, self.src_h - xf
        if k == 2:
            return self.src_w - xf, self.src_h - yf
        return self.src_w - yf, xf   # counter-clockwise


def from_settings(s, src_h, src_w):
    return Geometry(src_h, src_w, int(s.rot90) % 4, float(s.angle), float(s.crop_cx), float(s.crop_cy),
                    float(s.crop_w), float(s.crop_h))


# Keeping crops valid

def _extents(w_px, h_px, angle):
    """Half-extents along frame axes of a w x h rectangle rotated by `angle` degrees."""
    a = math.radians(angle)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return (w_px / 2) * c + (h_px / 2) * s, (w_px / 2) * s + (h_px / 2) * c


def is_inside(fw, fh, angle, cx, cy, w, h, tol=1e-6):
    """Is the rotated crop (normalized cx, cy, w, h) fully inside the frame?"""
    ex, ey = _extents(w * fw, h * fh, angle)
    X, Y = cx * fw, cy * fh
    return X - ex >= -tol * fw and X + ex <= fw * (1 + tol) and Y - ey >= -tol * fh and Y + ey <= fh * (1 + tol)


def max_crop(fw, fh, angle, ratio=None):
    """Largest centred crop at `angle` with width/height `ratio` (None = the frame's own ratio).
    Returns normalized (w, h)."""
    if ratio is None:
        ratio = fw / fh
    a = math.radians(angle)
    c, s = abs(math.cos(a)), abs(math.sin(a))
    # half-height b, half-width r*b must satisfy r*b*c + b*s <= fw/2 and r*b*s + b*c <= fh/2
    b = min((fw / 2) / (ratio * c + s), (fh / 2) / (ratio * s + c))
    return min(1.0, 2 * ratio * b / fw), min(1.0, 2 * b / fh)


def fit_inside(fw, fh, angle, cx, cy, w, h):
    """Make a crop valid: shrink it (keeping its aspect) only as far as needed to fit in the frame
    at all, then slide it the minimum distance to lie inside. Returns normalized (cx, cy, w, h)."""
    ex, ey = _extents(w * fw, h * fh, angle)
    k = min(1.0, (fw / 2) / ex if ex > 0 else 1.0, (fh / 2) / ey if ey > 0 else 1.0)
    w, h = w * k, h * k
    ex, ey = ex * k, ey * k
    X = min(max(cx * fw, ex), fw - ex)
    Y = min(max(cy * fh, ey), fh - ey)
    return X / fw, Y / fh, w, h


def reset_crop_for_angle(fw, fh, angle, ratio=None):
    """Full-size crop for a new angle: the largest centred rectangle with the given aspect."""
    w, h = max_crop(fw, fh, angle, ratio)
    return 0.5, 0.5, w, h


def is_max_crop(fw, fh, angle, cx, cy, w, h, ratio=None):
    mw, mh = max_crop(fw, fh, angle, ratio)
    return abs(cx - 0.5) < 1e-4 and abs(cy - 0.5) < 1e-4 and abs(w - mw) < 1e-4 and abs(h - mh) < 1e-4


# Resampling

def _bilinear_rows(src, ay, ax):
    """Bilinear sample of (H, W, 3) `src` at array coordinates ay, ax (same 2-D shape)."""
    H, W = src.shape[:2]
    ay = np.clip(ay, 0, H - 1)
    ax = np.clip(ax, 0, W - 1)
    y0 = np.floor(ay).astype(np.int32)
    x0 = np.floor(ax).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ay - y0).astype(np.float32)[..., None]
    fx = (ax - x0).astype(np.float32)[..., None]
    top = src[y0, x0] * (1 - fx) + src[y0, x1] * fx
    bot = src[y1, x0] * (1 - fx) + src[y1, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)


def _bilinear_channel(ch, ay, ax):
    H, W = ch.shape
    ay = np.clip(ay, 0, H - 1)
    ax = np.clip(ax, 0, W - 1)
    y0 = np.floor(ay).astype(np.int32)
    x0 = np.floor(ax).astype(np.int32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    fy = (ay - y0).astype(np.float32)
    fx = (ax - x0).astype(np.float32)
    top = ch[y0, x0] * (1 - fx) + ch[y0, x1] * fx
    bot = ch[y1, x0] * (1 - fx) + ch[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def warp(src, src_scale, geo, frame, out_shape, lens=None):
    """Sample the output region described by `frame` (output full-res coords) from `src`, which is
    the upright source photo at `src_scale` array px per source px. Returns (h, w, 3) float32.
    `lens` (a lens.Correction) additionally corrects distortion, colour fringes and vignetting."""
    h, w = out_shape
    ys = frame.y0 + (np.arange(h, dtype=np.float64) + 0.5) / frame.scale
    xs = frame.x0 + (np.arange(w, dtype=np.float64) + 0.5) / frame.scale

    # Fast path: no straightening and source pixels line up 1:1 -> rotate + slice, exact and cheap
    if lens is None and geo.angle == 0 and abs(frame.scale - src_scale) < 1e-9:
        xf0, yf0 = geo.output_to_frame(xs[0], ys[0])
        fx0, fy0 = xf0 * src_scale - 0.5, yf0 * src_scale - 0.5
        if abs(fx0 - round(fx0)) < 1e-6 and abs(fy0 - round(fy0)) < 1e-6:
            rotated = np.rot90(src, -(geo.rot90 % 4))
            r0, c0 = int(round(fy0)), int(round(fx0))
            if r0 >= 0 and c0 >= 0 and r0 + h <= rotated.shape[0] and c0 + w <= rotated.shape[1]:
                return np.ascontiguousarray(rotated[r0:r0 + h, c0:c0 + w], dtype=np.float32)

    out = np.empty((h, w, 3), np.float32)
    step = max(1, 4_000_000 // max(1, w))   # rows per strip, bounds memory on full-size exports
    for r in range(0, h, step):
        yy = ys[r:r + step][:, None]
        xx = xs[None, :]
        xf, yf = geo.output_to_frame(xx, yy)
        xf, yf = np.broadcast_arrays(xf, yf)
        sx, sy = geo.frame_to_source(xf, yf)
        if lens is None:
            out[r:r + step] = _bilinear_rows(src, sy * src_scale - 0.5, sx * src_scale - 0.5)
            continue
        coords = lens.source_coords(sx, sy)
        for c in range(3):
            cx, cy = coords[c]
            out[r:r + step, :, c] = _bilinear_channel(src[..., c], cy * src_scale - 0.5, cx * src_scale - 0.5)
        gain = lens.vignette_gain(*coords[1])
        if gain is not None:
            out[r:r + step] *= gain[..., None]
    return out
