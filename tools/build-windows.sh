#!/usr/bin/env bash
# Builds the Windows app on Linux with Wine + PyInstaller (how dist/windows/Film Lab.exe was made).
#   tools/build-windows.sh
# Needs: wine, curl, unzip. Downloads Windows Python 3.12 (NuGet package) into build-win/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
B="$HERE/build-win"
export WINEPREFIX="$B/prefix" WINEDEBUG=-all
mkdir -p "$B"
if [[ ! -f "$B/py/tools/python.exe" ]]; then
  curl -fL -o "$B/python.nupkg" https://www.nuget.org/api/v2/package/python/3.12.10
  mkdir -p "$B/py" && unzip -q -o "$B/python.nupkg" 'tools/*' -d "$B/py"
  wineboot -i >/dev/null 2>&1 || true
  wine "$B/py/tools/python.exe" -m pip install -q pyinstaller -r "$HERE/requirements.txt"
fi
cd "$HERE"
wine "$B/py/tools/python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed \
  --name "Film Lab" --icon film-lab.ico --version-file tools/version_info.txt --paths . \
  --add-data "filmlab/luts;filmlab/luts" --add-data "filmlab/gui/assets;filmlab/gui/assets" \
  --collect-binaries rawpy --collect-all pillow_heif --collect-all lensfunpy --exclude-module tkinter --exclude-module unittest \
  --distpath "$B/dist" --workpath "$B/work" --specpath "$B" tools/filmlab_entry.py
echo "Built: $B/dist/Film Lab.exe"
