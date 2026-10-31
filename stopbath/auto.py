"""Automatic adjustments: light & colour from the histogram, and horizon/vertical straightening.

Both analyse the plain developed photo (no film look) and return slider values, so the result stays
editable. Corrections are partial and capped so low-key scenes and warm light are not neutralised.
"""
import math

import numpy as np

from .pipeline import LUMA, Settings, develop_base, gaussian_blur, highlights_shadows


def _small(lin, side=640):
    h, w = lin.shape[:2]
    k = max(1, int(math.ceil(max(h, w) / side)))
    return lin[::k, ::k]


def auto_light_color(lin, is_raw):
    """Suggested settings for Exposure, Highlights, Shadows, Contrast, Warmth and Tint as a dict."""
    lin = _small(lin)
    s = Settings()

    # White balance: grey-world on the most neutral mid-tone pixels
    disp = develop_base(lin, is_raw, s)
    L = disp @ LUMA
    chroma = np.abs(disp - L[..., None]).max(axis=2)
    mid = (L > 0.15) & (L < 0.85)
    warmth = tint = 0.0
    if mid.sum() > 500:
        # Use only pixels that are close to neutral. In scenes dominated by one colour (foliage, sea)
        # the least saturated pixels are not grey, so white balance is left unchanged.
        thresh = min(np.percentile(chroma[mid], 30), 0.06)
        grey = mid & (chroma <= thresh)
        if grey.sum() > max(200, 0.02 * mid.sum()):
            r, g, b = (lin[grey][:, c].mean() for c in range(3))
            if r > 0 and g > 0 and b > 0:
                warmth = (b - r) / (0.18 * (b + r))
                tint = (1 - (r + b) / (2 * g)) / 0.12
                # Half strength, capped, so warm or cool light is reduced rather than removed
                warmth = float(np.clip(warmth * 0.5, -0.4, 0.4))
                # Tint is easily misjudged on foliage; keep the correction small
                tint = float(np.clip(tint * 0.4, -0.15, 0.15))
    s.warmth, s.tint = round(warmth, 2), round(tint, 2)

    # Exposure: move the median towards mid-grey (bisection; the tone curve is monotonic)
    def median_at(ev):
        s.exposure = ev
        return float(np.median(develop_base(lin, is_raw, s) @ LUMA))
    lo, hi = -2.0, 2.0
    for _ in range(18):
        m = (lo + hi) / 2
        if median_at(m) < 0.44:
            lo = m
        else:
            hi = m
    ev = (lo + hi) / 2
    s.exposure = round(float(np.clip(ev * 0.75, -1.5, 1.5)), 2)   # partial correction

    disp = develop_base(lin, is_raw, s)
    L = np.clip(disp @ LUMA, 0, 1)
    p05, p99 = np.percentile(L, [5, 99])

    # Highlights: pull down when the brightest areas are near clipping
    s.highlights = round(float(-np.clip((p99 - 0.90) * 6, 0, 0.8)), 2)
    # Shadows: lift when much of the frame is nearly black
    s.shadows = round(float(np.clip((0.08 - p05) * 5, 0, 0.6)), 2)

    L2 = np.clip(highlights_shadows(disp, s.highlights, s.shadows) @ LUMA, 0, 1)
    p05, p95 = np.percentile(L2, [5, 95])
    spread = p95 - p05
    # Contrast: add a little to flat photos, remove a little from harsh ones
    s.contrast = round(float(np.clip((0.72 - spread) * 1.4, -0.3, 0.4)), 2)
    if abs(s.contrast) < 0.03:
        s.contrast = 0.0
    return {f: getattr(s, f) for f in ('exposure', 'highlights', 'shadows', 'contrast', 'warmth', 'tint')}


def auto_straighten(lin, is_raw, max_angle=12.0, debug=False):
    """Angle (degrees, same convention as the Straighten slider) that levels the strongest
    near-horizontal or near-vertical lines, or None if there's no clear line to go by.

    Edges are weighted by structure-tensor coherence, so long straight features (horizons, walls)
    dominate and texture such as foliage contributes little."""
    lin = _small(lin, 1000)
    disp = develop_base(lin, is_raw, Settings())
    L = (disp @ LUMA).astype(np.float32)
    gy, gx = np.gradient(gaussian_blur(L, 1.0))
    jxx = gaussian_blur(gx * gx, 4.0)
    jyy = gaussian_blur(gy * gy, 4.0)
    jxy = gaussian_blur(gx * gy, 4.0)
    tr = jxx + jyy
    disc = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coherence = np.where(tr > 1e-9, disc / (tr + 1e-12), 0) ** 2
    # Dominant gradient direction; the edge (line) runs perpendicular to it
    theta = 0.5 * np.degrees(np.arctan2(2 * jxy, jxx - jyy))
    edge = (theta + 90.0 + 90.0) % 180.0 - 90.0
    weight = coherence * tr
    b = 6
    weight[:b] = weight[-b:] = 0
    weight[:, :b] = weight[:, -b:] = 0
    keep = weight > np.percentile(weight, 90)
    edge, weight = edge[keep], weight[keep]
    total = weight.sum() + 1e-12
    dev_h = edge
    dev_v = np.where(edge > 0, edge - 90.0, edge + 90.0)
    bins = np.linspace(-max_angle, max_angle, int(max_angle * 20) + 1)   # 0.1° bins
    centres = (bins[:-1] + bins[1:]) / 2
    best = None
    for name, dev, prefer in (('horizontal', dev_h, 1.15), ('vertical', dev_v, 1.0)):
        sel = np.abs(dev) < max_angle
        hist, _ = np.histogram(dev[sel], bins=bins, weights=weight[sel])
        hist = np.convolve(hist, np.ones(9) / 9, mode='same')
        i = int(np.argmax(hist))
        # How much of all line evidence sits within ±0.5° of the peak, versus a uniform spread
        near = hist[max(0, i - 5):i + 6].sum()
        share = near / total
        uniform = 11.0 / len(hist) * (hist.sum() / total)
        score = (share / (uniform + 1e-9)) * prefer
        if debug:
            print('  %s peak %+.1f share %.3f score %.1f' % (name, centres[i], share, score))
        if best is None or score > best[0]:
            best = (score, centres[i], share)
    score, deviation, share = best
    if score < 3.0 or share < 0.04:
        return None
    # A line at screen angle `deviation` becomes level after turning the photo by -deviation
    return round(float(-deviation), 1)
