"""Copies a curated set of film emulations from the RawTherapee Film Simulation Collection
(CC BY-SA 4.0, by Pat David, Pavlov Dmitry and Michael Ezra) into filmlab/luts/.

    curl -LO https://rawtherapee.com/shared/HaldCLUT.zip      # 421 MB, md5 4742e362a70c1a1c0fb9042a17d285e1
    .venv/bin/python tools/import_haldclut.py HaldCLUT.zip

The images are copied unchanged (lossless Hald CLUTs). Only the "normal" development of each film
is included; the collection also has pushed/pulled variants if you want to load those yourself.
"""
import sys
import zipfile
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / 'filmlab' / 'luts'

# category -> [(name shown in Film Lab, file in the collection)]
CURATED = {
    'Color Negative': [
        ('Kodak Portra 160', 'Color/Kodak/Kodak Portra 160 2.png'),
        ('Kodak Portra 160 NC', 'Color/Kodak/Kodak Portra 160 NC 2.png'),
        ('Kodak Portra 400', 'Color/Kodak/Kodak Portra 400 2.png'),
        ('Kodak Portra 400 NC', 'Color/Kodak/Kodak Portra 400 NC 2.png'),
        ('Kodak Portra 400 VC', 'Color/Kodak/Kodak Portra 400 VC 2.png'),
        ('Kodak Portra 800', 'Color/Kodak/Kodak Portra 800 2.png'),
        ('Kodak Ektar 100', 'Color/Kodak/Kodak Ektar 100.png'),
        ('Kodak Elite Color 200', 'Color/Kodak/Kodak Elite Color 200.png'),
        ('Fujifilm Pro 400H', 'Color/Fuji/Fuji 400H 2.png'),
        ('Fujifilm Pro 160C', 'Color/Fuji/Fuji 160C 2.png'),
        ('Fujicolor Superia 200', 'Color/Fuji/Fuji Superia 200.png'),
        ('Fujicolor Superia 400', 'Color/Fuji/Fuji Superia 400 2.png'),
        ('Fujicolor Superia Reala 100', 'Color/Fuji/Fuji Superia Reala 100.png'),
    ],
    'Slide': [
        ('Fujifilm Velvia 50', 'Color/Fuji/Fuji Velvia 50.png'),
        ('Fujifilm Provia 100F', 'Color/Fuji/Fuji Provia 100F.png'),
        ('Fujifilm Astia 100F', 'Color/Fuji/Fuji Astia 100F.png'),
        ('Kodak Ektachrome 100 VS', 'Color/Kodak/Kodak Ektachrome 100 VS.png'),
        ('Kodak Ektachrome E100GX', 'Color/Kodak/Kodak E-100 GX Ektachrome 100.png'),
        ('Kodak Kodachrome 64', 'Color/Kodak/Kodak Kodachrome 64.png'),
    ],
    'Black & White': [
        ('Kodak Tri-X 400', 'Black-and-White/Kodak/Kodak TRI-X 400 2.png'),
        ('Kodak T-Max 100', 'Black-and-White/Kodak/Kodak T-Max 100.png'),
        ('Kodak T-Max 400', 'Black-and-White/Kodak/Kodak T-Max 400.png'),
        ('Ilford HP5 Plus 400', 'Black-and-White/Ilford/Ilford HP5 Plus 400.png'),
        ('Ilford Delta 100', 'Black-and-White/Ilford/Ilford Delta 100.png'),
        ('Ilford FP4 Plus 125', 'Black-and-White/Ilford/Ilford FP4 Plus 125.png'),
        ('Ilford Pan F Plus 50', 'Black-and-White/Ilford/Ilford Pan F Plus 50.png'),
        ('Ilford XP2 Super', 'Black-and-White/Ilford/Ilford XP2.png'),
        ('Fujifilm Neopan Acros 100', 'Black-and-White/Fuji/Fuji Neopan Acros 100.png'),
    ],
}

CREDITS = """Film emulation LUTs in this folder come from the RawTherapee Film Simulation Collection
(version 2015-09-20), https://rawtherapee.com/shared/HaldCLUT.zip

License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
         https://creativecommons.org/licenses/by-sa/4.0/
Authors: Pat David, Pavlov Dmitry, Michael Ezra

The files are unmodified Hald CLUT images, renamed for readability. Film names are used only to
say which film stock each profile is designed to approximate; neither the authors of the collection
nor Film Lab are affiliated with or endorsed by Kodak, Fujifilm, Ilford or other trademark owners.
"""


def main(zip_path):
    z = zipfile.ZipFile(zip_path)
    for old in OUT.rglob('*'):
        if old.is_file() and old.suffix.lower() in ('.cube', '.png'):
            old.unlink()
    for category, films in CURATED.items():
        d = OUT / category
        d.mkdir(parents=True, exist_ok=True)
        for name, member in films:
            data = z.read('HaldCLUT/' + member)
            (d / (name + '.png')).write_bytes(data)
            print('%-15s %-30s %5.1f MB' % (category, name, len(data) / 1e6))
    (OUT / 'CREDITS.txt').write_text(CREDITS)


if __name__ == '__main__':
    main(sys.argv[1])
