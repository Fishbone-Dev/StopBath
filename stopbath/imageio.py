"""Finding photos, decoding RAW/JPEG, thumbnails, and writing JPEGs with metadata."""
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, TiffImagePlugin

from .pipeline import srgb_to_linear

# Everything LibRaw reads that cameras commonly produce: Sony, Canon, Nikon, Fujifilm, Olympus/OM System,
# Panasonic/Leica, Pentax/Ricoh, Samsung, Hasselblad, Phase One, Sigma, Minolta, Kodak, Epson, Mamiya
# and DNG (Leica, Pentax, Ricoh, DJI, phones, Adobe converter).
RAW_EXTS = {'.arw', '.srf', '.sr2', '.dng', '.nef', '.nrw', '.cr2', '.cr3', '.crw', '.raf', '.orf', '.rw2',
            '.raw', '.rwl', '.pef', '.ptx', '.srw', '.3fr', '.fff', '.iiq', '.x3f', '.mrw', '.kdc', '.dcr',
            '.erf', '.mef', '.mos'}
# Ready-made images. In a RAW+image pair the image is the camera's own rendering.
IMAGE_EXTS = {'.jpg', '.jpeg', '.jpe', '.png', '.tif', '.tiff', '.webp', '.heic', '.heif', '.hif', '.avif',
              '.bmp'}
JPEG_EXTS = IMAGE_EXTS   # older name, kept for compatibility
_IMAGE_PREFERENCE = ['.jpg', '.jpeg', '.jpe', '.heic', '.heif', '.hif', '.tif', '.tiff', '.png', '.webp',
                     '.avif', '.bmp']

try:   # HEIC/HEIF (iPhone, recent Sony/Canon/Fujifilm "HIF") if pillow-heif is installed
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    IMAGE_EXTS -= {'.heic', '.heif', '.hif'}


def image_label(path):
    """Short name of a file's type for the interface: RAW, JPEG, PNG, TIFF, HEIF..."""
    ext = Path(path).suffix.lower()
    if ext in RAW_EXTS:
        return 'RAW'
    return {'.jpg': 'JPEG', '.jpeg': 'JPEG', '.jpe': 'JPEG', '.tif': 'TIFF', '.tiff': 'TIFF', '.heic': 'HEIF',
            '.heif': 'HEIF', '.hif': 'HEIF'}.get(ext, ext.lstrip('.').upper())


@dataclass
class Photo:
    stem: str
    folder: Path
    raw: Path = None
    jpeg: Path = None

    @property
    def key(self):
        return str(self.folder / self.stem)

    def source(self, prefer_raw=True):
        if prefer_raw and self.raw:
            return self.raw
        return self.jpeg or self.raw

    @property
    def kind(self):
        if self.raw and self.jpeg:
            return 'RAW+' + image_label(self.jpeg)
        return 'RAW' if self.raw else image_label(self.jpeg)


# Folders cameras and operating systems keep on memory cards that hold no photos to edit
# (e.g. Sony's M4ROOT/THMBNL video thumbnails, AVCHD structures, Windows/macOS system folders).
_SKIP_DIRS = {'M4ROOT', 'PRIVATE', 'AVF_INFO', 'MP_ROOT', 'THMBNL', 'AVCHD', 'CLIP', 'SUB', 'MISC',
              'SYSTEM VOLUME INFORMATION', '$RECYCLE.BIN', 'STOP-BATH', 'FILM-LAB'}


def scan_folder(folder):
    """All photos under `folder` (including subfolders like DCIM/100MSDCF); RAW+JPEG pairs are merged."""
    photos = {}
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith('.') and d.upper() not in _SKIP_DIRS)
        for name in filenames:
            if name.startswith('.'):
                continue
            p = Path(dirpath) / name
            ext = p.suffix.lower()
            if ext not in RAW_EXTS and ext not in IMAGE_EXTS:
                continue
            key = (dirpath, p.stem.lower())
            photo = photos.get(key)
            if photo is None:
                photo = photos[key] = Photo(p.stem, Path(dirpath))
            if ext in RAW_EXTS:
                photo.raw = p
            elif photo.jpeg is None or (_IMAGE_PREFERENCE.index(ext) if ext in _IMAGE_PREFERENCE else 99) < \
                    (_IMAGE_PREFERENCE.index(photo.jpeg.suffix.lower())
                     if photo.jpeg.suffix.lower() in _IMAGE_PREFERENCE else 99):
                photo.jpeg = p   # e.g. prefer the JPEG over a PNG copy with the same name
    return sorted(photos.values(), key=lambda ph: (str(ph.folder), ph.stem.lower()))


def is_raw(path):
    return Path(path).suffix.lower() in RAW_EXTS


# Decoding

def _block_downsample(a, max_side):
    """Shrink by an integer factor (area average) so the long side is at most ~max_side."""
    h, w = a.shape[:2]
    k = int(np.ceil(max(h, w) / max_side)) if max_side else 1
    if k <= 1:
        return a
    h2, w2 = h // k, w // k
    return a[: h2 * k, : w2 * k].reshape(h2, k, w2, k, 3).mean(axis=(1, 3), dtype=np.float32)


_ev_cache = {}


def _median_luma(rgb_display):
    """Median brightness of a display-space image, ignoring near-black pixels (letterbox bars)."""
    L = rgb_display @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    L = L[L > 0.02]
    return float(np.median(L)) if L.size > 100 else None


def raw_brightness_match(path):
    """Exposure offset (EV) that matches the median brightness of the rendered RAW to the camera's
    embedded JPEG preview. Deterministic and cached per file, so previews and exports agree."""
    import io
    import rawpy
    from .pipeline import linear_to_srgb, tonemap
    key = (str(path), os.path.getmtime(path))
    if key in _ev_cache:
        return _ev_cache[key]
    ev = 0.0
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            if thumb.format != rawpy.ThumbFormat.JPEG:
                raise ValueError('no JPEG preview')
            cam = Image.open(io.BytesIO(thumb.data)).convert('RGB')
            cam.thumbnail((400, 400))
            target = _median_luma(np.asarray(cam, np.float32) / 255.0)
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=True,
                                  output_bps=16, gamma=(1, 1), output_color=rawpy.ColorSpace.sRGB)
        lin = _block_downsample(rgb.astype(np.float32) / 65535.0, 400)
        if target is not None:
            lo, hi = -1.5, 1.5
            for _ in range(24):   # brightness rises with exposure, so bisect
                mid = (lo + hi) / 2
                m = _median_luma(linear_to_srgb(tonemap(lin * np.float32(2.0 ** mid))))
                if m is None or m < target:
                    lo = mid
                else:
                    hi = mid
            ev = round((lo + hi) / 2, 3)
            if abs(ev) >= 1.49:   # hit the limit: preview likely unrepresentative, don't trust it
                ev = 0.0
    except Exception:
        ev = 0.0
    _ev_cache[key] = ev
    return ev


def load_linear(path, max_side=None):
    """Decode to linear-light float32 RGB, optionally shrunk so the long side is about max_side.
    Returns (array, is_raw, (full_height, full_width)) where the size is that of the upright
    full-resolution image, i.e. what an export of this file will measure."""
    if is_raw(path):
        import rawpy
        with rawpy.imread(str(path)) as raw:
            w, h = raw.sizes.width, raw.sizes.height
            if raw.sizes.flip in (5, 6):
                w, h = h, w
            half = bool(max_side) and max(w, h) / 2 >= max_side
            rgb = raw.postprocess(half_size=half, use_camera_wb=True, no_auto_bright=True,
                                  output_bps=16, gamma=(1, 1), output_color=rawpy.ColorSpace.sRGB)
        lin = rgb.astype(np.float32) / 65535.0
        ev = raw_brightness_match(path)
        if ev:
            lin *= np.float32(2.0 ** ev)
        if not max_side:
            h, w = lin.shape[:2]  # exact size of the full decode
        return _block_downsample(lin, max_side), True, (h, w)

    with Image.open(path) as im:
        full_w, full_h = im.size
        if im.getexif().get(0x0112) in (5, 6, 7, 8):
            full_w, full_h = full_h, full_w
        if max_side:
            im.draft('RGB', (max_side, max_side))
        im = ImageOps.exif_transpose(im)
        if max_side and im.mode in ('RGB', 'L', 'CMYK', 'YCbCr') and max(im.size) > max_side:
            # 8-bit input: resize with Pillow before converting to float
            icc = im.info.get('icc_profile')
            im = im.copy()
            im.thumbnail((max_side, max_side), Image.LANCZOS)
            if icc:
                im.info['icc_profile'] = icc
        arr = _to_srgb_array(im)
        if arr.dtype == np.uint8:
            # 8-bit input: linearise with a 256-entry table (one float copy)
            return _SRGB_TO_LINEAR_8BIT[arr], False, (full_h, full_w)
        if max_side and max(arr.shape[:2]) > max_side:
            k = max_side / max(arr.shape[:2])
            size = (max(1, round(arr.shape[1] * k)), max(1, round(arr.shape[0] * k)))
            # resize each channel in float to keep 16-bit precision
            arr = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(arr[..., c]), 'F')
                                       .resize(size, Image.LANCZOS)) for c in range(3)], axis=2).clip(0, 1)
    return srgb_to_linear(arr), False, (full_h, full_w)


_SRGB_PROFILE = None
_SRGB_TO_LINEAR_8BIT = srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0).astype(np.float32)


def _to_srgb_array(im):
    """Any Pillow image -> float32 sRGB array in 0..1. Handles embedded colour profiles (Adobe RGB,
    Display P3...), 16-bit greyscale, CMYK and transparency (flattened onto white)."""
    global _SRGB_PROFILE
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I'):
        a = np.asarray(im, np.float32)
        a = a / (65535.0 if a.max() > 255 else 255.0)
        return np.repeat(np.clip(a, 0, 1)[..., None], 3, axis=2)
    if im.mode == 'F':
        a = np.clip(np.asarray(im, np.float32), 0, 1)
        return np.repeat(a[..., None], 3, axis=2)
    if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
        rgba = im.convert('RGBA')
        bg = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        im2 = Image.alpha_composite(bg, rgba).convert('RGB')
        im2.info['icc_profile'] = im.info.get('icc_profile')
        im = im2
    icc = im.info.get('icc_profile')
    rgb = im.convert('RGB') if im.mode != 'RGB' else im
    if icc and im.mode in ('RGB', 'RGBA', 'CMYK', 'P', 'L', 'LA'):
        try:
            import io
            if _SRGB_PROFILE is None:
                _SRGB_PROFILE = ImageCms.createProfile('sRGB')
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            if 'srgb' not in (ImageCms.getProfileDescription(src) or '').lower():
                rgb = ImageCms.profileToProfile(im if im.mode in ('RGB', 'CMYK') else rgb, src, _SRGB_PROFILE,
                                                outputMode='RGB')
        except Exception:
            pass   # unreadable profile: treat as sRGB
    return np.asarray(rgb, np.uint8)


_RAW_ROTATION = {3: 180, 5: 90, 6: 270}  # LibRaw flip -> PIL counter-clockwise degrees


def load_thumbnail(photo, size=320):
    """Fast thumbnail: the paired JPEG, or the preview embedded in the RAW."""
    if photo.jpeg:
        with Image.open(photo.jpeg) as im:
            im.draft('RGB', (size, size))
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((size, size), Image.LANCZOS)
            return im
    import io
    import rawpy
    with rawpy.imread(str(photo.raw)) as raw:
        flip = raw.sizes.flip
        try:
            thumb = raw.extract_thumb()
        except Exception:
            thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            im = Image.open(io.BytesIO(thumb.data)).convert('RGB')
            if flip in _RAW_ROTATION:
                im = im.rotate(_RAW_ROTATION[flip], expand=True)
        else:
            rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
            im = Image.fromarray(rgb)
    im.thumbnail((size, size), Image.LANCZOS)
    return im


# Metadata

_TIFF_TYPES = {1: ('B', 1), 2: ('s', 1), 3: ('H', 2), 4: ('L', 4), 5: ('LL', 8), 7: ('B', 1),
               9: ('l', 4), 10: ('ll', 8)}
_IFD0_TAGS = {0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298}  # make, model, software, datetime, artist, copyright
_EXIF_TAGS = {0x829A, 0x829D, 0x8822, 0x8827, 0x9003, 0x9004, 0x9201, 0x9202, 0x9204, 0x9207,
              0x9209, 0x920A, 0xA405, 0xA432, 0xA433, 0xA434}


def _read_tiff_ifd(f, offset, endian, wanted):
    tags = {}
    f.seek(offset)
    count = struct.unpack(endian + 'H', f.read(2))[0]
    entries = [struct.unpack(endian + 'HHLL', f.read(12)) for _ in range(min(count, 512))]
    for tag, typ, n, value in entries:
        if tag not in wanted or typ not in _TIFF_TYPES:
            continue
        fmt, size = _TIFF_TYPES[typ]
        total = size * n
        if total <= 4:
            raw = struct.pack(endian + 'L', value)[:total]
        else:
            f.seek(value)
            raw = f.read(total)
        if typ == 2:
            tags[tag] = raw.split(b'\0', 1)[0].decode('latin-1').strip()
        elif typ in (5, 10):
            vals = struct.unpack(endian + fmt[0] * (2 * n), raw)
            rats = [TiffImagePlugin.IFDRational(vals[i], vals[i + 1]) for i in range(0, 2 * n, 2)]
            tags[tag] = rats[0] if n == 1 else tuple(rats)
        else:
            vals = struct.unpack(endian + fmt * n, raw)
            tags[tag] = vals[0] if n == 1 else vals
    return tags


def _raw_exif(path):
    """Basic camera/exposure metadata from a TIFF-based RAW (ARW, DNG, NEF, ...)."""
    exif = Image.Exif()
    try:
        with open(path, 'rb') as f:
            order = f.read(2)
            endian = {b'II': '<', b'MM': '>'}.get(order)
            if endian is None:
                return exif
            magic, ifd0 = struct.unpack(endian + 'HL', f.read(6))
            if magic not in (42, 0x4F52, 0x5352, 0x55):   # TIFF, Olympus ORF variants, Panasonic RW2
                return exif
            tags = _IFD0_TAGS | {0x8769} | ({0x0017} if magic == 0x55 else set())
            for tag, value in _read_tiff_ifd(f, ifd0, endian, tags).items():
                if tag == 0x0017 and magic == 0x55:   # Panasonic RW2 keeps ISO in IFD0
                    exif.get_ifd(0x8769)[0x8827] = value
                elif tag == 0x8769:
                    exif_ifd = exif.get_ifd(0x8769)
                    exif_ifd.update(_read_tiff_ifd(f, value, endian, _EXIF_TAGS))
                else:
                    exif[tag] = value
    except (OSError, struct.error, ValueError):
        pass
    return exif


def _thumbnail_exif(path):
    """Metadata from the JPEG preview inside a RAW (works for Fujifilm, Canon CR3, Olympus, Panasonic...)."""
    import io
    import rawpy
    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(thumb.data)) as im:
                exif = im.getexif()
                exif.get_ifd(0x8769)
                exif.pop(0x0201, None)   # drop the preview's own thumbnail pointers
                exif.pop(0x0202, None)
                return exif
    except Exception:
        pass
    return Image.Exif()


def read_exif(photo, source):
    """Metadata for the exported file. Orientation is reset because pixels are already upright."""
    exif = Image.Exif()
    if photo.jpeg:
        try:
            with Image.open(photo.jpeg) as im:
                exif = im.getexif()
                exif.get_ifd(0x8769)  # load the Exif sub-IFD so it is written back
        except OSError:
            pass
    if 0x010F not in exif and photo.raw:
        # No camera JPEG (or it has no metadata): read the basics from the RAW itself,
        # or from the JPEG preview embedded in it for formats that aren't TIFF-based
        exif = _raw_exif(photo.raw)
        if 0x010F not in exif:
            exif = _thumbnail_exif(photo.raw)
    exif[0x0112] = 1
    exif[0x0131] = 'Stop Bath by Fishbone'
    return exif


def _exif_bytes(exif):
    data = exif.tobytes()
    if len(data) > 65000:  # JPEG APP1 limit; Sony maker notes can be large
        exif.get_ifd(0x8769).pop(0x927C, None)
        exif.pop(0x8825, None)
        data = exif.tobytes()
    return data if len(data) <= 65000 else b''


_SRGB = None


FORMATS = {'jpeg': ('.jpg', 'JPEG'), 'png': ('.png', 'PNG'), 'tiff': ('.tif', 'TIFF'), 'webp': ('.webp', 'WEBP')}


def save_image(rgb8, path, exif=None, quality=95, fmt='jpeg'):
    """Write an 8-bit RGB image as JPEG, PNG, TIFF or WebP, with metadata and an sRGB profile."""
    if fmt == 'jpeg':
        return save_jpeg(rgb8, path, exif, quality)
    global _SRGB
    if _SRGB is None:
        _SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    im = Image.fromarray(rgb8, 'RGB')
    kwargs = {'icc_profile': _SRGB}
    if fmt == 'png':
        kwargs['optimize'] = False
        kwargs['compress_level'] = 6
    elif fmt == 'tiff':
        kwargs['compression'] = 'tiff_lzw'
    elif fmt == 'webp':
        kwargs['quality'] = int(quality)
        kwargs['method'] = 4
    if exif is not None and fmt == 'tiff':
        # TIFF can't embed an Exif block from bytes reliably: write the main tags as TIFF tags
        info = TiffImagePlugin.ImageFileDirectory_v2()
        for tag in (0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298):
            if tag in exif:
                info[tag] = str(exif[tag])
        kwargs['tiffinfo'] = info
    elif exif is not None:
        data = _exif_bytes(exif)
        if data:
            kwargs['exif'] = data
    tmp = Path(str(path) + '.part')
    im.save(tmp, FORMATS[fmt][1], **kwargs)
    os.replace(tmp, path)


def save_jpeg(rgb8, path, exif=None, quality=95):
    global _SRGB
    if _SRGB is None:
        _SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    im = Image.fromarray(rgb8, 'RGB')
    kwargs = dict(quality=int(quality), optimize=True, icc_profile=_SRGB,
                  subsampling=0 if quality >= 90 else 2)
    if exif is not None:
        data = _exif_bytes(exif)
        if data:
            kwargs['exif'] = data
    tmp = Path(str(path) + '.part')
    im.save(tmp, 'JPEG', **kwargs)
    os.replace(tmp, path)
