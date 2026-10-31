# Film Lab

Film looks for RAW and JPEG photos, for Windows and Linux.

Copyright (C) 2026 Fishbone. Free software under the GNU GPL v3 or later (see `LICENSE` and `NOTICE`).

Downloads: https://github.com/Fishbone-Dev/FilmLab/releases

## Install

### Windows 10/11

Download `FilmLab-1.0.3-Windows-x64.zip`, unzip it anywhere and run `Film Lab.exe`. Nothing is
installed. The program isn't code-signed, so Windows may show "Windows protected your PC" the first
time: choose *More info* and *Run anyway*, or right-click the zip, open *Properties* and tick *Unblock*
before extracting.

### Linux

Requires Python 3.10 or newer. On Debian/Ubuntu also install `python3-venv` and `libxcb-cursor0`.

```
tar xf FilmLab-1.0.3-Linux.tar.gz
cd "Film Lab"
./film-lab              # first run sets up its Python environment in .venv
./film-lab --install    # optional: add Film Lab to the application menu
```

To uninstall, delete the folder (and `~/.local/share/applications/film-lab.desktop` if you used
`--install`).

## Features

- Opens a folder or a camera card (Ctrl+O, or drop it on the window). RAW+JPEG pairs are shown as one
  photo; camera system folders such as video thumbnails are skipped.
- 19 film looks based on real stocks: colour negative (Portra 160/400/800, Ektar 100, Pro 400H,
  Superia 400, Elite Color 200), slide (Velvia 50, Provia 100F, Astia 100F, Ektachrome 100 VS,
  Kodachrome 64) and black & white (Tri-X 400, HP5 Plus, T-Max 100, Delta 100, FP4 Plus, Pan F Plus,
  Acros 100). Each look sets grain to suit the film.
- Adjustments: exposure, contrast, highlights, shadows, fade, warmth, tint, saturation, B&W filter and
  toning, sharpening, noise reduction, grain, halation and vignette.
- Auto sets exposure, highlights, shadows, contrast and white balance. It is a toggle: turning it off
  restores the previous values.
- Lens corrections (distortion, chromatic aberration, vignetting) from the lensfun database. Applied to
  RAW files by default; camera JPEGs are already corrected.
- Crop and straighten, with aspect ratios, 90° rotation and auto-straighten.
- 28 bundled film LUTs, plus any `.cube` file or Hald CLUT image.
- Zoom to 100% and a loupe, both rendered from the full-resolution file, so they match the export.
- Before/after split view, histogram, undo/redo, copy/paste of settings, recipes.
- Edits are stored per photo and restored when the folder is reopened. Originals are never modified.
- Delete moves files to the Trash/Recycle Bin.

## Keyboard shortcuts

| Key | Action |
|---|---|
| Left / Right | Previous / next photo |
| Ctrl+O | Open folder |
| Ctrl+E | Export |
| A | Auto on/off |
| Ctrl+R | Revert to the original |
| Ctrl+Z, Ctrl+Shift+Z | Undo, redo |
| Ctrl+C, Ctrl+V | Copy settings, paste to selected photos |
| Ctrl+Shift+V | Apply settings to all photos |
| R | Crop & straighten (Enter to finish) |
| Ctrl+drag in crop mode | Straighten along a line |
| Scroll, + / - | Zoom |
| 0 or F, 1, 2 | Fit, 100%, 200% |
| L | Loupe (Shift+scroll changes magnification) |
| C | Before/after split |
| Space (hold) | Show the original |
| P | Mark selected photos for export |
| Delete | Move selected photos to the Trash |
| Ctrl+B | Show/hide the filmstrip |
| F1 | List of shortcuts |

## Export

JPEG, PNG, TIFF or WebP, at full resolution or a reduced long edge. Exports go to a `film-lab` folder
inside the photo folder unless another folder is chosen. Camera metadata is copied and an sRGB profile
is embedded. Photos are rendered in bands on background threads, so memory use stays moderate and an
export can be cancelled at any time.

## Supported files

- RAW: everything LibRaw reads, including Sony ARW, Canon CR3/CR2, Nikon NEF, Fujifilm RAF, Olympus ORF,
  Panasonic RW2, Pentax PEF and DNG.
- Images: JPEG, HEIC/HEIF, PNG, TIFF, WebP, AVIF and BMP. Embedded colour profiles are converted to sRGB.

RAW files are matched to the brightness of the camera's embedded JPEG preview, so different cameras
start from a similar rendering.

## Reporting problems

Use *Help (?) > Copy Diagnostic Info* and paste the result into a message. Log files are in
`%APPDATA%\Film Lab\logs` on Windows and `~/.local/share/film-lab/logs` on Linux.

## Building

- Run from source: `./film-lab` (Linux), or `python -m filmlab` with the packages in `requirements.txt`.
- Tests: `./film-lab --test`. Set `FILMLAB_SAMPLE_RAW=/path/to/file.ARW` to include a real RAW file.
- Windows executable: `tools/build-windows.sh` builds `Film Lab.exe` on Linux using Wine and PyInstaller.
  `tools/windows-installer.iss` is an optional Inno Setup script.
- Bundled film profiles: `tools/import_haldclut.py` imports them from the RawTherapee collection.

## Source layout

| Path | Contents |
|---|---|
| `filmlab/pipeline.py` | Image processing |
| `filmlab/looks.py` | Film look definitions |
| `filmlab/geometry.py` | Crop, straighten, rotation |
| `filmlab/lens.py` | Lens corrections |
| `filmlab/auto.py` | Auto adjustments and auto-straighten |
| `filmlab/lut.py` | LUT loading and application |
| `filmlab/imageio.py` | Decoding, thumbnails, metadata, saving |
| `filmlab/export.py` | Export |
| `filmlab/gui/` | User interface |
| `filmlab/luts/` | Bundled film profiles and their credits |
| `tests/` | Tests |
| `tools/` | Build and import scripts |

## Credits and licence

Film Lab is copyright (C) 2026 Fishbone and licensed under the GNU General Public License v3 or later.
See `LICENSE` (full licence text) and `NOTICE` (copyright and bundled components).

- Film profiles: RawTherapee Film Simulation Collection by Pat David, Pavlov Dmitry and Michael Ezra
  (CC BY-SA 4.0), see `filmlab/luts/CREDITS.txt`. Film names only identify the stock each profile
  approximates; there is no affiliation with Kodak, Fujifilm or Ilford.
- Lens data: lensfun (CC BY-SA 3.0) via lensfunpy.
- RAW decoding: LibRaw via rawpy. Images: Pillow and pillow-heif. Interface: Qt via PyQt6. NumPy.
