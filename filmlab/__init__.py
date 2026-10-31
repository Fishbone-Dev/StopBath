"""Film Lab: film looks for RAW and image files.

Copyright (C) 2026 Fishbone. Licensed under the GNU General Public License v3 or later
(see LICENSE and COPYING).
"""
import os as _os

# Keep BLAS single-threaded. The array operations here are small and memory-bound, and the app
# parallelises with its own thread pools; per-call BLAS threads only oversubscribe the CPU.
# Must run before numpy is imported.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    _os.environ.setdefault(_var, '1')

__version__ = '1.0.3'
__author__ = 'Fishbone'
__license__ = 'GPL-3.0-or-later'
