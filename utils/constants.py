# Basic Grid
"""
constants.py

Defines shared constants used across modules.

Constants:
- INTERP_GRID: 1D NumPy array of wavelengths from 300 nm to 1100 nm (inclusive), in 1 nm steps.
"""

import numpy as np

INTERP_GRID = np.arange(300, 1101, 1)  # 300–1100 nm, step 5
