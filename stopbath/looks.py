"""Built-in looks: real film stocks.

Each film look combines a film emulation profile (a Hald CLUT from the RawTherapee Film Simulation
Collection, CC BY-SA 4.0, see luts/CREDITS.txt) with grain that suits the stock: fine for ISO 50-100
films, more visible for 400-800. Film names only identify the stock each profile approximates;
there's no affiliation with Kodak, Fujifilm or Ilford.
"""
from dataclasses import dataclass
from pathlib import Path

LUT_DIR = Path(__file__).resolve().parent / 'luts'
IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


@dataclass(frozen=True)
class Look:
    name: str
    description: str
    category: str = 'Color Negative'
    lut: str = ''              # path relative to luts/, '' for none
    mono: bool = False
    # Parametric look (used when there's no LUT; Natural uses the neutral defaults)
    matrix: tuple = IDENTITY
    saturation: float = 1.0
    shadow_tint: tuple = (0, 0, 0)
    highlight_tint: tuple = (0, 0, 0)
    contrast: float = 0.0
    fade: float = 0.0
    roll: float = 0.0
    # Defaults for the adjustable effects
    grain: float = 0.0
    grain_size: float = 1.0
    halation: float = 0.0
    vignette: float = 0.0
    bw_filter: str = 'Neutral'
    toning: str = 'None'
    toning_amount: float = 0.0

    @property
    def lut_path(self):
        return str(LUT_DIR / self.lut) if self.lut else ''


def _film(name, category, description, grain, grain_size, mono=False, **kw):
    return Look(name, description, category=category, lut='%s/%s.png' % (category, name), mono=mono,
                grain=grain, grain_size=grain_size, bw_filter='Film' if mono else 'Neutral', **kw)


NEG, SLIDE, BW = 'Color Negative', 'Slide', 'Black & White'

LOOKS = [
    Look('Natural', 'The photo as shot, without a film look. A clean starting point.', category='Digital'),

    # Colour negative: soft contrast, forgiving highlights, flattering skin
    _film('Kodak Portra 160', NEG, 'Fine-grained portrait film: natural, warm skin and soft, low-contrast tones. '
          'Great for portraits and weddings in good light.', 0.08, 0.8),
    _film('Kodak Portra 400', NEG, 'The classic all-rounder: warm, gentle skin tones and creamy highlights. '
          'Portraits, travel and everyday.', 0.15, 1.0),
    _film('Kodak Portra 800', NEG, 'Faster Portra: a little more saturation and grain. Indoors, dusk, available light.',
          0.24, 1.2),
    _film('Kodak Ektar 100', NEG, 'Vivid, ultra-fine-grain negative with punchy reds and deep blues. '
          'A favourite for landscapes and travel.', 0.05, 0.7),
    _film('Fujifilm Pro 400H', NEG, 'Airy and pastel, with soft greens and cool, clean highlights. '
          'Weddings, portraits in shade, foliage.', 0.15, 1.0),
    _film('Fujicolor Superia 400', NEG, 'Consumer film with crisp greens and a slightly cool cast. '
          'Street, travel and snapshots.', 0.22, 1.1),
    _film('Kodak Elite Color 200', NEG, 'Warm, saturated consumer colour: sunny holidays and family snapshots.',
          0.18, 1.0),

    # Slide (reversal): rich colour, deep blacks, little latitude
    _film('Fujifilm Velvia 50', SLIDE, 'Legendary landscape slide film: intensely saturated greens and blues, '
          'deep shadows. Use on landscapes, not faces.', 0.05, 0.6),
    _film('Fujifilm Provia 100F', SLIDE, 'Accurate, neutral slide film with clean colour and fine grain. '
          'Landscapes, nature and architecture.', 0.06, 0.7),
    _film('Fujifilm Astia 100F', SLIDE, 'Soft, low-contrast slide film made for skin tones. '
          'Portraits and fashion with slide-film clarity.', 0.06, 0.7),
    _film('Kodak Ektachrome 100 VS', SLIDE, 'Vivid saturation slide film with cool, clear blues. Landscapes, sky and water.',
          0.06, 0.7),
    _film('Kodak Kodachrome 64', SLIDE, 'The iconic warm, contrasty slide film with rich reds and yellows. '
          'Travel, street and summer scenes.', 0.08, 0.8),

    # Black & white
    _film('Kodak Tri-X 400', BW, 'The classic photojournalism film: punchy contrast and visible, gritty grain. '
          'Street, documentary and portraits with character.', 0.34, 1.2, mono=True),
    _film('Ilford HP5 Plus 400', BW, 'Versatile, slightly softer than Tri-X, with wide latitude. '
          'Portraits, street and overcast landscapes.', 0.30, 1.25, mono=True),
    _film('Kodak T-Max 100', BW, 'Very fine grain and crisp detail with a long, smooth tonal scale. '
          'Landscapes and architecture.', 0.09, 0.7, mono=True),
    _film('Ilford Delta 100', BW, 'Fine-grained and sharp with clean midtones. Landscapes and studio portraits.',
          0.10, 0.7, mono=True),
    _film('Ilford FP4 Plus 125', BW, 'Traditional medium-speed film: classic tonality with moderate grain. '
          'Landscapes and portraits.', 0.14, 0.9, mono=True),
    _film('Ilford Pan F Plus 50', BW, 'Slow, very fine grain with rich, contrasty blacks. Bright landscapes and still life.',
          0.06, 0.6, mono=True),
    _film('Fujifilm Neopan Acros 100', BW, 'Ultra-fine grain and smooth, silvery tones. '
          'Landscapes, long exposures and architecture.', 0.07, 0.7, mono=True),
]

CATEGORIES = ('Digital', NEG, SLIDE, BW)
_BY_NAME = {look.name: look for look in LOOKS}


def get(name):
    return _BY_NAME.get(name, LOOKS[0])


def names():
    return [look.name for look in LOOKS]
