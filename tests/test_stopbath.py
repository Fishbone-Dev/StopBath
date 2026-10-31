"""Tests for the processing engine, LUTs, file handling and export.

Run:  .venv/bin/python -m unittest discover -s tests -v
Set STOPBATH_SAMPLE_RAW=/path/to/file.ARW to include a real RAW in the export test.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stopbath import export, geometry, imageio, looks, pipeline  # noqa: E402
from stopbath.lut import Lut, LutError  # noqa: E402


def gradient_image(h=64, w=96):
    """Smooth colorful test image in linear light, including pure primaries and greys."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / (w - 1), y / (h - 1), 1 - x / (w - 1)], axis=2)
    img[:8, :8] = (1, 0, 0)
    img[:8, 8:16] = (0, 0, 1)
    img[8:16, :16] = 0.18
    return pipeline.srgb_to_linear(img)


class PipelineTests(unittest.TestCase):
    def test_srgb_roundtrip(self):
        x = np.linspace(0, 1, 1001, dtype=np.float32)
        np.testing.assert_allclose(pipeline.linear_to_srgb(pipeline.srgb_to_linear(x)), x, atol=1e-5)

    def test_look_matrices_keep_grey_neutral(self):
        for look in looks.LOOKS:
            for row in look.matrix:
                self.assertAlmostEqual(sum(row), 1.0, places=6, msg=look.name)

    def test_every_look_renders_valid_output(self):
        lin = gradient_image()
        for look in looks.LOOKS:
            for is_raw in (True, False):
                s = pipeline.Settings().with_look_defaults(look.name)
                out = pipeline.render(lin, is_raw, s)
                self.assertEqual(out.shape, lin.shape, look.name)
                self.assertEqual(out.dtype, np.float32, look.name)
                self.assertTrue(np.isfinite(out).all(), look.name)
                self.assertGreaterEqual(out.min(), 0, look.name)
                self.assertLessEqual(out.max(), 1, look.name)

    def test_natural_jpeg_is_unchanged(self):
        disp = np.random.default_rng(1).random((40, 50, 3), dtype=np.float32)
        out = pipeline.render(pipeline.srgb_to_linear(disp), False, pipeline.Settings())
        np.testing.assert_allclose(out, disp, atol=1 / 512)

    def test_mono_looks_are_grey_unless_toned(self):
        lin = gradient_image()
        for look in looks.LOOKS:
            if not look.mono:
                continue
            s = pipeline.Settings().with_look_defaults(look.name)
            s.toning = 'None'
            s.grain = 0
            out = pipeline.render(lin, True, s)
            np.testing.assert_allclose(out[..., 0], out[..., 1], atol=1e-5, err_msg=look.name)
            np.testing.assert_allclose(out[..., 1], out[..., 2], atol=1e-5, err_msg=look.name)

    def test_red_filter_darkens_blue(self):
        lin = gradient_image()
        s = pipeline.Settings().with_look_defaults('Kodak Tri-X 400')
        s.grain = 0
        s.bw_filter = 'Red'
        red = pipeline.render(lin, True, s)
        s.bw_filter = 'Blue'
        blue = pipeline.render(lin, True, s)
        self.assertLess(red[:8, 8:16].mean(), blue[:8, 8:16].mean())  # blue patch
        self.assertGreater(red[:8, :8].mean(), blue[:8, :8].mean())   # red patch

    def test_sepia_is_warm(self):
        lin = gradient_image()
        s = pipeline.Settings().with_look_defaults('Kodak Tri-X 400')
        s.grain, s.toning, s.toning_amount = 0, 'Sepia', 0.8
        out = pipeline.render(lin, True, s)
        mid = out[8:16, :16].mean(axis=(0, 1))
        self.assertGreater(mid[0], mid[2])

    def test_film_looks_use_their_profiles(self):
        lin = gradient_image()
        natural = pipeline.render(lin, True, pipeline.Settings())
        for look in looks.LOOKS[1:]:
            self.assertTrue(Path(look.lut_path).is_file(), look.name)
            s = pipeline.Settings().with_look_defaults(look.name)
            s.grain = 0
            out = pipeline.render(lin, True, s)
            self.assertGreater(np.abs(out - natural).mean(), 0.004, look.name)   # differs from Natural
            if look.mono:
                self.assertEqual(s.bw_filter, 'Film')

    def test_bundled_luts_load(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from stopbath.gui.app import bundled_luts
        luts = bundled_luts()
        self.assertGreaterEqual(len(luts), 25)
        grey = np.linspace(0, 1, 50, dtype=np.float32)[None, :, None].repeat(3, axis=2)
        for cat, name, path in luts:
            lut = Lut.load(path)
            out = lut.apply(grey)
            self.assertTrue(np.isfinite(out).all(), name)
            self.assertGreater(out[0, -1].mean(), out[0, 0].mean() + 0.5, name)   # dark stays dark, light light
            if cat == 'Black & White':
                self.assertLess(np.abs(out - out.mean(axis=2, keepdims=True)).max(), 0.02, name)

    def test_s_curve_is_monotonic(self):
        x = np.linspace(0, 1, 2001, dtype=np.float32)
        for c in np.linspace(-1, 1, 21):
            self.assertTrue((np.diff(pipeline.s_curve(x, c)) >= -1e-7).all(), c)

    def test_exposure_brightens(self):
        lin = gradient_image()
        s = pipeline.Settings()
        dark = pipeline.render(lin, True, s).mean()
        s.exposure = 1
        self.assertGreater(pipeline.render(lin, True, s).mean(), dark + 0.03)

    def test_raw_mid_grey(self):
        out = pipeline.render(np.full((4, 4, 3), 0.18, np.float32), True, pipeline.Settings())
        self.assertTrue(0.5 < out.mean() < 0.7)

    def test_grain_is_deterministic_and_unbiased(self):
        img = np.full((300, 400, 3), 0.5, np.float32)
        f = pipeline.Frame(300, 400)
        a = pipeline.apply_grain(img, 0.6, 1.0, f)
        b = pipeline.apply_grain(img, 0.6, 1.0, f)
        np.testing.assert_array_equal(a, b)
        self.assertLess(abs(a.mean() - 0.5), 0.01)
        self.assertGreater(a.std(), 0.005)

    def test_effects_off_are_identity(self):
        img = np.random.default_rng(2).random((30, 40, 3), dtype=np.float32)
        f = pipeline.Frame(30, 40)
        np.testing.assert_array_equal(pipeline.apply_grain(img, 0, 1, f), img)
        np.testing.assert_array_equal(pipeline.apply_halation(img, 0, f, img), img)
        np.testing.assert_array_equal(pipeline.apply_vignette(img, 0, f), img)

    def test_vignette_darkens_corners_only(self):
        img = np.full((200, 300, 3), 0.5, np.float32)
        out = pipeline.apply_vignette(img, 1.0, pipeline.Frame(200, 300))
        self.assertAlmostEqual(float(out[100, 150, 0]), 0.5, places=3)
        self.assertLess(out[0, 0, 0], 0.4)

    def test_halation_glows_red_around_highlights(self):
        img = np.full((200, 200, 3), 0.1, np.float32)
        img[90:110, 90:110] = 1.0
        out = pipeline.apply_halation(img, 1.0, pipeline.Frame(200, 200), img)
        ring = out[80:85, 100]
        self.assertGreater(ring[:, 0].mean(), ring[:, 2].mean())

    def _photo(self, h=600, w=900):
        rng = np.random.default_rng(9)
        img = gradient_image(h, w)
        img[200:260, 300:380] = 3.0  # bright highlight for halation
        return img * np.float32(0.8) + rng.random((h, w, 3), dtype=np.float32) * np.float32(0.05)

    def test_region_render_matches_full_render(self):
        """What the 100% zoom and loupe show must equal the same area of the export."""
        full = self._photo()
        for grain_size in (0.5, 1.0, 2.5):   # covers per-pixel and lattice grain
            s = pipeline.Settings().with_look_defaults('Kodak Ektar 100')
            s.grain, s.grain_size, s.vignette, s.halation = 0.8, grain_size, 0.6, 0.0
            whole = pipeline.render(full, True, s)
            y0, x0, h, w = 123, 457, 150, 211
            region = pipeline.render(full[y0:y0 + h, x0:x0 + w], True, s,
                                     frame=pipeline.Frame(600, 900, 1.0, y0, x0))
            np.testing.assert_allclose(region, whole[y0:y0 + h, x0:x0 + w], atol=1e-5, err_msg=str(grain_size))

    def test_region_render_with_denoise_matches_full_render(self):
        """Same as above with noise reduction, which needs the margin the app adds around regions."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from stopbath.gui.app import _render_region  # the function the zoom view and loupe use
        rng = np.random.default_rng(11)
        full = self._photo() + rng.normal(0, 0.02, (600, 900, 3)).astype(np.float32)
        s = pipeline.Settings().with_look_defaults('Kodak Portra 400')
        s.nr_luma, s.nr_color, s.grain, s.sharpen, s.shadows, s.highlights = 0.8, 1.0, 0.3, 0.6, 0.4, -0.5
        whole = pipeline.render(full, True, s)
        for rect in ((457, 123, 211, 150), (0, 0, 200, 120), (700, 480, 200, 120)):   # middle and corners
            region = _render_region('k', full, True, s, rect, 1.0, full[::3, ::3])
            x, y, w, h = rect
            img = region.after
            got = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3]
            want = pipeline.to_uint8(whole[y:y + h, x:x + w]).reshape(h, w * 3)
            self.assertLessEqual(int(np.abs(got.astype(int) - want).max()), 1, rect)

    def test_cropped_straightened_region_matches_export(self):
        """Zoom/loupe with a crop and straighten angle must show exactly what the export contains."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from stopbath.gui.app import _render_region
        full = self._photo()
        for extra in ({'angle': 7.3, 'crop_w': 0.6, 'crop_h': 0.55, 'crop_cx': 0.45, 'crop_cy': 0.52},
                      {'rot90': 1, 'crop_w': 0.7, 'crop_h': 0.8},
                      {'rot90': 3, 'angle': -12.0, 'crop_w': 0.5, 'crop_h': 0.5, 'nr_luma': 0.5, 'nr_color': 0.6}):
            s = pipeline.Settings().with_look_defaults('Kodak Portra 400')
            s.grain, s.vignette = 0.5, 0.4
            for k, v in extra.items():
                setattr(s, k, v)
            whole = pipeline.develop(full, True, s)
            H, W = whole.shape[:2]
            rect = (W // 3, H // 4, W // 3, H // 3)
            region = _render_region('k', full, True, s, rect, 1.0, full[::3, ::3])
            x, y, w, h = rect
            img = region.after
            got = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3]
            want = pipeline.to_uint8(whole[y:y + h, x:x + w]).reshape(h, w * 3)
            self.assertLessEqual(int(np.abs(got.astype(int) - want).max()), 1, extra)

    def test_highlights_shadows_are_monotonic_and_directional(self):
        x = np.linspace(0, 1, 501, dtype=np.float32)[None, :, None].repeat(3, axis=2)
        for h in (-1, -0.5, 0.5, 1):
            for sh in (-1, 0, 1):
                y = pipeline.highlights_shadows(x, h, sh)[0, :, 0]
                self.assertTrue((np.diff(y) >= -1e-6).all(), (h, sh))
        self.assertLess(pipeline.highlights_shadows(x, -1, 0)[0, 400, 0], x[0, 400, 0])   # recovers highlights
        self.assertGreater(pipeline.highlights_shadows(x, 0, 1)[0, 100, 0], x[0, 100, 0])  # opens shadows

    def test_sharpen_increases_edge_contrast(self):
        img = np.full((40, 40, 3), 0.3, np.float32)
        img[:, 20:] = 0.6
        out = pipeline.sharpen(img, 1.0, pipeline.Frame(40, 40))
        self.assertGreater(out[:, 20].mean() - out[:, 19].mean(), 0.3 + 0.05)
        np.testing.assert_allclose(out[:, :10], img[:, :10], atol=1e-5)   # flat areas untouched

    def test_denoise_reduces_noise_and_keeps_edges(self):
        clean = np.zeros((200, 200, 3), np.float32)
        clean[:, 100:] = 0.8
        clean[:, :100] = 0.2
        rng = np.random.default_rng(12)
        noisy = clean + rng.normal(0, 0.04, clean.shape).astype(np.float32)
        out = pipeline.denoise(noisy, 1.0, 1.0, pipeline.Frame(200, 200))
        self.assertLess(out[20:180, 10:90].std(), noisy[20:180, 10:90].std() / 3)
        self.assertGreater(out[:, 102].mean() - out[:, 97].mean(), 0.55)   # edge still sharp
        self.assertAlmostEqual(float(pipeline.luma(out).mean()), float(pipeline.luma(noisy).mean()), places=2)
        np.testing.assert_array_equal(pipeline.denoise(noisy, 0, 0, pipeline.Frame(200, 200)), noisy)

    def test_region_halation_close_to_full(self):
        full = self._photo()
        s = pipeline.Settings()
        s.halation = 1.0
        whole = pipeline.render(full, True, s)
        y0, x0 = 180, 260
        context = full[::3, ::3]  # what the app has: a downscaled preview of the whole photo
        region = pipeline.render(full[y0:y0 + 120, x0:x0 + 160], True, s,
                                 frame=pipeline.Frame(600, 900, 1.0, y0, x0), context=context)
        self.assertLess(np.abs(region - whole[y0:y0 + 120, x0:x0 + 160]).max(), 0.03)

    def test_preview_scale_grain_strength_matches_export_when_viewed_small(self):
        s = pipeline.Settings()
        s.grain = 1.0
        flat = np.full((400, 600, 3), 0.18, np.float32)
        export_img = pipeline.render(flat, True, s)
        preview = pipeline.render(flat[::4, ::4], True, s, frame=pipeline.Frame(400, 600, 0.25))
        # Downscaling the export 4x averages its grain: its std should be close to the preview's
        shrunk = export_img.reshape(100, 4, 150, 4, 3).mean(axis=(1, 3))
        self.assertLess(abs(shrunk.std() - preview.std()) / shrunk.std(), 0.5)

    def test_blur_preserves_mean(self):
        a = np.random.default_rng(3).random((120, 90), dtype=np.float32)
        self.assertAlmostEqual(float(pipeline.gaussian_blur(a, 4).mean()), float(a.mean()), places=2)

    def test_settings_roundtrip_and_look_defaults(self):
        s = pipeline.Settings(exposure=0.7, warmth=0.3)
        d = s.to_dict()
        d['unknown_future_key'] = 1
        self.assertEqual(pipeline.Settings.from_dict(d), s)
        t = s.with_look_defaults('Kodak Tri-X 400')
        self.assertEqual((t.look, t.exposure, t.warmth), ('Kodak Tri-X 400', 0.7, 0.3))
        self.assertEqual(t.grain, looks.get('Kodak Tri-X 400').grain)
        self.assertEqual(pipeline.Settings().with_look_defaults('nope').look, 'Natural')


class GeometryTests(unittest.TestCase):
    def src(self, h=40, w=60):
        return np.random.default_rng(13).random((h, w, 3), dtype=np.float32)

    def warp(self, src, **kw):
        s = pipeline.Settings(**kw)
        geo, frame, shape = pipeline.output_frame(s, src.shape[:2])
        return pipeline.warp_source(src, src.shape[:2], 1.0, s, frame, shape)

    def test_identity_is_untouched(self):
        src = self.src()
        self.assertIs(self.warp(src), src)

    def test_quarter_turns_match_numpy(self):
        src = self.src()
        for k in range(4):
            np.testing.assert_array_equal(self.warp(src, rot90=k), np.rot90(src, -k), err_msg=str(k))

    def test_unrotated_crop_is_exact_slice(self):
        src = self.src()
        out = self.warp(src, crop_cx=0.25, crop_cy=0.5, crop_w=0.5, crop_h=0.5)
        np.testing.assert_array_equal(out, src[10:30, 0:30])

    def test_positive_angle_turns_clockwise(self):
        src = self.src(50, 50)
        geo = geometry.Geometry(50, 50, 0, 90.0)
        out = geometry.warp(src, 1.0, geo, pipeline.Frame(50, 50), (50, 50))
        np.testing.assert_allclose(out, np.rot90(src, -1), atol=1e-5)

    def test_straightened_crop_samples_inside_frame(self):
        fw, fh = 6000, 4000
        for angle in (-30, -7.5, 0.3, 12, 45):
            for ratio in (None, 1.0, 16 / 9, 4 / 5):
                w, h = geometry.max_crop(fw, fh, angle, ratio)
                self.assertTrue(geometry.is_inside(fw, fh, angle, 0.5, 0.5, w, h), (angle, ratio))
                self.assertFalse(geometry.is_inside(fw, fh, angle, 0.5, 0.5, w * 1.01, h * 1.01), (angle, ratio))
                if ratio:
                    self.assertAlmostEqual(w * fw / (h * fh), ratio, places=6)
        self.assertEqual(geometry.max_crop(fw, fh, 0), (1.0, 1.0))

    def test_fit_inside_repairs_any_crop(self):
        rng = np.random.default_rng(14)
        for _ in range(500):
            angle = rng.uniform(-45, 45)
            cx, cy, w, h = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0.05, 1.2), rng.uniform(0.05, 1.2)
            fx = geometry.fit_inside(3000, 2000, angle, cx, cy, w, h)
            self.assertTrue(geometry.is_inside(3000, 2000, angle, *fx))
            self.assertAlmostEqual(fx[2] / fx[3], w / h, places=6)   # aspect kept
            if geometry.is_inside(3000, 2000, angle, cx, cy, w, h):
                np.testing.assert_allclose(fx, (cx, cy, w, h), atol=1e-9)  # valid crops untouched

    def test_output_size_and_develop(self):
        src = np.full((400, 600, 3), 0.18, np.float32)
        s = pipeline.Settings(rot90=1, angle=5, crop_w=0.5, crop_h=0.5)
        out = pipeline.develop(src, True, s)
        self.assertEqual(out.shape, (300, 200, 3))      # frame is 600x400 after the quarter turn
        self.assertTrue(np.isfinite(out).all())

    def test_geometry_not_inherited(self):
        s = pipeline.Settings(look='Kodak Tri-X 400', angle=3, crop_w=0.5, rot90=2, exposure=1)
        t = s.without_geometry()
        self.assertEqual((t.look, t.exposure, t.angle, t.crop_w, t.rot90), ('Kodak Tri-X 400', 1, 0.0, 1.0, 0))


class LensTests(unittest.TestCase):
    def exif(self, make, model, lens_model='', focal=None, fnum=None):
        from PIL import Image as _I
        e = _I.Exif()
        e[0x010F], e[0x0110] = make, model
        x = e.get_ifd(0x8769)
        if lens_model:
            x[0xA434] = lens_model
        if focal:
            x[0x920A] = focal
        if fnum:
            x[0x829D] = fnum
        return e

    @unittest.skipUnless(__import__('stopbath.lens', fromlist=['x']).available(), 'lensfunpy not installed')
    def test_identification_is_strict(self):
        from stopbath import lens
        ok = lens.identify(self.exif('SONY', 'ILCE-7M2', 'E 25mm F2', 25.0, 8.0))
        self.assertIsNotNone(ok)
        self.assertEqual(ok.lens_model, 'E 25mm f/2')
        rx = lens.identify(self.exif('SONY', 'DSC-RX10M4', '', 24.0, 4.0))   # fixed lens: no lens tag needed
        self.assertIsNotNone(rx)
        # Unknown lens on an interchangeable-lens body: refuse rather than guess
        self.assertIsNone(lens.identify(self.exif('Canon', 'Canon EOS R6', '', 50.0, 4.0)))
        # A lens whose numbers don't match must not be accepted
        self.assertIsNone(lens.identify(self.exif('SONY', 'ILCE-7M2', 'E 99mm F7', 99.0, 7.0)))
        # Similar-but-different camera models must not match
        self.assertIsNone(lens.identify(self.exif('Apple', 'iPhone 12 Pro', 'iPhone 12 Pro back camera', 4.2, 1.6)))

    @unittest.skipUnless(__import__('stopbath.lens', fromlist=['x']).available(), 'lensfunpy not installed')
    def test_correction_maps_and_consistency(self):
        from stopbath import lens
        from stopbath.gui.app import _render_region
        info = lens.identify(self.exif('SONY', 'ILCE-7M2', 'E 25mm F2', 25.0, 4.0))
        rng = np.random.default_rng(21)
        full = (rng.random((600, 900, 3), dtype=np.float32) * 0.6 + 0.1)
        s = pipeline.Settings().with_look_defaults('Kodak Portra 400')
        s.angle, s.crop_w, s.crop_h, s.grain = 3.0, 0.8, 0.8, 0.3
        whole = pipeline.develop(full, True, s, info)
        s_off = s.copy()
        s_off.lens_mode = 'off'
        self.assertGreater(np.abs(whole - pipeline.develop(full, True, s_off, info)).mean(), 0.005)
        H, W = whole.shape[:2]
        rect = (W // 4, H // 5, W // 3, H // 3)
        region = _render_region('k', full, True, s, rect, 1.0, full[::3, ::3], info)
        x, y, w, h = rect
        img = region.after
        got = np.frombuffer(img.constBits().asstring(img.sizeInBytes()), np.uint8).reshape(h, img.bytesPerLine())[:, :w * 3]
        want = pipeline.to_uint8(whole[y:y + h, x:x + w]).reshape(h, w * 3)
        self.assertLessEqual(int(np.abs(got.astype(int) - want).max()), 1)
        # JPEG files are left alone in Auto mode (the camera already corrected them)
        self.assertIsNone(pipeline.lens_correction(pipeline.Settings(), False, info, (600, 900)))
        self.assertIsNotNone(pipeline.lens_correction(pipeline.Settings(lens_mode='on'), False, info, (600, 900)))


class AutoTests(unittest.TestCase):
    def test_auto_brightens_dark_photos_and_neutralises_casts(self):
        from stopbath import auto
        rng = np.random.default_rng(31)
        base = rng.random((300, 450, 3), dtype=np.float32) * 0.18 + 0.02          # dull, dark scene
        sug = auto.auto_light_color(base * 0.25, True)
        self.assertGreater(sug['exposure'], 0.3)
        warm = base * np.array([1.3, 1.0, 0.7], np.float32)
        self.assertLess(auto.auto_light_color(warm, True)['warmth'], -0.05)          # cools a warm cast
        cool = base * np.array([0.7, 1.0, 1.3], np.float32)
        self.assertGreater(auto.auto_light_color(cool, True)['warmth'], 0.05)
        for k, v in sug.items():
            self.assertTrue(-2 <= v <= 2, (k, v))

    def test_auto_straighten_levels_lines_and_refuses_noise(self):
        from stopbath import auto, geometry
        h, w = 600, 900
        img = np.full((h, w, 3), 0.25, np.float32)
        for y in range(80, h, 110):                                                # level stripes
            img[y:y + 30] = 0.7
        img[:, 400:430] = 0.05
        for tilt in (-3.0, 2.0):
            g = geometry.Geometry(h, w, 0, tilt, 0.5, 0.5, *geometry.max_crop(w, h, tilt))
            oh, ow = g.out_size
            tilted = geometry.warp(img, 1.0, g, pipeline.Frame(oh, ow), (oh, ow))
            angle = auto.auto_straighten(tilted, False)
            self.assertIsNotNone(angle)
            self.assertAlmostEqual(angle, -tilt, delta=0.4)
        noise = np.random.default_rng(32).random((h, w, 3), dtype=np.float32)
        self.assertIsNone(auto.auto_straighten(noise, False))


class LutTests(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile('w', suffix='.cube', delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def identity_cube(self, n):
        lines = ['TITLE "id"', 'LUT_3D_SIZE %d' % n]
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    lines.append('%f %f %f' % (r / (n - 1), g / (n - 1), b / (n - 1)))
        return '\n'.join(lines)

    def test_identity_lut(self):
        lut = Lut.load(self.write(self.identity_cube(9)))
        img = np.random.default_rng(4).random((20, 30, 3), dtype=np.float32)
        np.testing.assert_allclose(lut.apply(img), img, atol=1e-5)

    def test_channel_order(self):
        # LUT that swaps red and blue: detects a wrong [r][g][b] vs [b][g][r] reading
        n = 5
        lines = ['LUT_3D_SIZE %d' % n]
        for b in range(n):
            for g in range(n):
                for r in range(n):
                    lines.append('%f %f %f' % (b / (n - 1), g / (n - 1), r / (n - 1)))
        lut = Lut.load(self.write('\n'.join(lines)))
        out = lut.apply(np.array([[[1.0, 0.25, 0.0]]], np.float32))
        np.testing.assert_allclose(out[0, 0], [0.0, 0.25, 1.0], atol=1e-5)

    def test_bad_luts_rejected(self):
        with self.assertRaises(LutError):
            Lut.load(self.write('LUT_1D_SIZE 4\n0 0 0\n'))
        with self.assertRaises(LutError):
            Lut.load(self.write('LUT_3D_SIZE 2\n0 0 0\n'))
        with self.assertRaises(LutError):
            Lut.load(self.write('just text\n'))


class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__('shutil').rmtree(self.tmp))

    def make_jpeg(self, path, size=(320, 200), orientation=None):
        rgb = (np.random.default_rng(5).random((size[1], size[0], 3)) * 255).astype(np.uint8)
        im = Image.fromarray(rgb)
        exif = Image.Exif()
        exif[0x010F] = 'SONY'
        exif[0x0110] = 'ILCE-7M2'
        if orientation:
            exif[0x0112] = orientation
        im.save(path, quality=95, exif=exif.tobytes())

    def test_scan_pairs_raw_and_jpeg(self):
        sub = self.tmp / 'DCIM' / '100MSDCF'
        sub.mkdir(parents=True)
        (sub / 'DSC1.ARW').write_bytes(b'x')
        self.make_jpeg(sub / 'DSC1.JPG')
        self.make_jpeg(sub / 'DSC2.jpg')
        (sub / 'notes.txt').write_text('x')
        photos = imageio.scan_folder(self.tmp)
        self.assertEqual([(p.stem, p.kind) for p in photos], [('DSC1', 'RAW+JPEG'), ('DSC2', 'JPEG')])
        self.assertEqual(photos[0].source(True).suffix, '.ARW')
        self.assertEqual(photos[0].source(False).suffix, '.JPG')

    def test_export_jpeg_keeps_size_rotation_and_metadata(self):
        src = self.tmp / 'DSC9.JPG'
        self.make_jpeg(src, (320, 200), orientation=6)  # rotated 90°: upright image is 200x320
        photo = imageio.scan_folder(self.tmp)[0]
        s = pipeline.Settings().with_look_defaults('Kodak Portra 400')
        out = export.output_path(photo, s, self.tmp / 'out')
        path, err = export.export_one(photo, s.to_dict(), str(out))
        self.assertIsNone(err)
        self.assertEqual(Path(path).name, 'DSC9_kodak-portra-400.jpg')
        with Image.open(path) as im:
            self.assertEqual(im.size, (200, 320))
            exif = im.getexif()
            self.assertEqual(exif.get(0x0110), 'ILCE-7M2')
            self.assertEqual(exif.get(0x0112), 1)
            self.assertIn('icc_profile', im.info)

    def test_export_reports_errors_instead_of_raising(self):
        bad = self.tmp / 'BAD.JPG'
        bad.write_bytes(b'not a jpeg')
        photo = imageio.scan_folder(self.tmp)[0]
        path, err = export.export_one(photo, pipeline.Settings().to_dict(), str(self.tmp / 'o.jpg'))
        self.assertIsNotNone(err)
        self.assertFalse((self.tmp / 'o.jpg').exists())

    @unittest.skipUnless(os.environ.get('STOPBATH_SAMPLE_RAW'), 'set STOPBATH_SAMPLE_RAW to test a real RAW')
    def test_export_real_raw(self):
        import shutil
        src = Path(os.environ['STOPBATH_SAMPLE_RAW'])
        shutil.copy(src, self.tmp / src.name)
        photo = imageio.scan_folder(self.tmp)[0]
        s = pipeline.Settings().with_look_defaults('Kodak Tri-X 400')
        path, err = export.export_one(photo, s.to_dict(), str(self.tmp / 'raw.jpg'))
        self.assertIsNone(err)
        import rawpy
        with rawpy.imread(str(src)) as raw:
            w, h = raw.sizes.width, raw.sizes.height
        with Image.open(path) as im:
            self.assertEqual(max(im.size), max(w, h))
            exif = im.getexif()
            self.assertEqual(exif.get(0x010F), 'SONY')
            self.assertIn(0x829A, exif.get_ifd(0x8769))  # exposure time


if __name__ == '__main__':
    unittest.main()
