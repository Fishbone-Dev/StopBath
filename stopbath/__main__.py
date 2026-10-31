import multiprocessing
import sys

from .gui.app import main

if __name__ == '__main__':
    multiprocessing.freeze_support()   # required for PyInstaller builds
    sys.exit(main())
