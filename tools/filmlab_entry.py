"""Entry point for the packaged Windows build (PyInstaller)."""
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    from filmlab.gui.app import main
    sys.exit(main())
