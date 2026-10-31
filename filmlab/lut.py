"""Loading and applying 3D LUTs: Adobe/Resolve .cube files and Hald CLUT images (.png/.tif)."""
from pathlib import Path

import numpy as np


class LutError(Exception):
    pass


class Lut:
    def __init__(self, table, domain_min, domain_max, title=''):
        self.table = table            # (N, N, N, 3) indexed [b][g][r] as stored in the file
        self.size = table.shape[0]
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.title = title

    HALD_EXTS = ('.png', '.tif', '.tiff')

    @classmethod
    def load(cls, path):
        if str(path).lower().endswith(cls.HALD_EXTS):
            return cls.load_hald(path)
        return cls.load_cube(path)

    @classmethod
    def load_hald(cls, path):
        """A Hald CLUT image: a square image of side L^3 holding an (L^2)^3 lookup table,
        red changing fastest (the format used by RawTherapee, G'MIC and darktable)."""
        from PIL import Image
        with Image.open(path) as im:
            if im.mode not in ('RGB', 'RGBA', 'I;16', 'I;16B', 'RGB;16'):
                im = im.convert('RGB')
            arr = np.asarray(im.convert('RGB'), np.float32) / 255.0
        h, w = arr.shape[:2]
        n = round((h * w) ** (1 / 3))
        if h != w or n ** 3 != h * w or n < 2:
            raise LutError('Not a Hald CLUT image (must be square with a side of L³ pixels)')
        title = Path(path).stem
        return cls(arr.reshape(n, n, n, 3), np.zeros(3, np.float32), np.ones(3, np.float32), title)

    @classmethod
    def load_cube(cls, path):
        size = None
        dmin = np.zeros(3, np.float32)
        dmax = np.ones(3, np.float32)
        title = ''
        rows = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                key = line.split()[0].upper()
                if key == 'TITLE':
                    title = line[5:].strip().strip('"')
                elif key == 'LUT_3D_SIZE':
                    size = int(line.split()[1])
                elif key == 'LUT_1D_SIZE':
                    raise LutError('1D LUTs are not supported, only 3D .cube LUTs')
                elif key == 'DOMAIN_MIN':
                    dmin = np.array([float(v) for v in line.split()[1:4]], np.float32)
                elif key == 'DOMAIN_MAX':
                    dmax = np.array([float(v) for v in line.split()[1:4]], np.float32)
                elif key[0].isdigit() or key[0] in '-.':
                    rows.append([float(v) for v in line.split()[:3]])
        if size is None:
            raise LutError('Not a 3D .cube file (no LUT_3D_SIZE)')
        if size < 2:
            raise LutError('LUT_3D_SIZE must be at least 2')
        if len(rows) != size ** 3:
            raise LutError('Expected %d entries, found %d' % (size ** 3, len(rows)))
        if np.any(dmax <= dmin):
            raise LutError('Invalid DOMAIN_MIN/DOMAIN_MAX')
        # In .cube files red changes fastest, so the reshape gives [b][g][r]
        table = np.array(rows, np.float32).reshape(size, size, size, 3)
        return cls(table, dmin, dmax, title)

    def apply(self, img):
        """Trilinear interpolation of an (H, W, 3) float image.

        Uses one flat index per pixel and np.take on a contiguous (N³, 3) table, which is several
        times faster than 4-D fancy indexing for the large (144³) film tables. Works in row strips
        to keep the temporary arrays cache-sized."""
        n = self.size
        flat = self._flat_table()
        out = np.empty(img.shape, np.float32)
        h = img.shape[0]
        step = max(1, 262144 // max(1, img.shape[1]))
        span = (self.domain_max - self.domain_min).astype(np.float32)
        for r in range(0, h, step):
            x = (np.clip(img[r:r + step], self.domain_min, self.domain_max) - self.domain_min) / span
            x = x * np.float32(n - 1)
            i0 = np.minimum(x.astype(np.int32), n - 2)
            f = x - i0
            base = (i0[..., 2] * n + i0[..., 1]) * n + i0[..., 0]       # [b][g][r] order
            fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
            c000 = np.take(flat, base, axis=0)
            c100 = np.take(flat, base + 1, axis=0)
            c010 = np.take(flat, base + n, axis=0)
            c110 = np.take(flat, base + n + 1, axis=0)
            c001 = np.take(flat, base + n * n, axis=0)
            c101 = np.take(flat, base + n * n + 1, axis=0)
            c011 = np.take(flat, base + n * n + n, axis=0)
            c111 = np.take(flat, base + n * n + n + 1, axis=0)
            c00 = c000 + (c100 - c000) * fr
            c10 = c010 + (c110 - c010) * fr
            c01 = c001 + (c101 - c001) * fr
            c11 = c011 + (c111 - c011) * fr
            c0 = c00 + (c10 - c00) * fg
            c1 = c01 + (c11 - c01) * fg
            out[r:r + step] = c0 + (c1 - c0) * fb
        return out

    def _flat_table(self):
        if getattr(self, '_flat', None) is None:
            self._flat = np.ascontiguousarray(self.table.reshape(-1, 3), dtype=np.float32)
        return self._flat
