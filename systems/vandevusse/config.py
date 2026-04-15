from pathlib import Path

import numpy as np


VANDEVUSSE_ROOT = Path("VanDeVusse")
VANDEVUSSE_DATA_SUBDIR = VANDEVUSSE_ROOT / "Data"
VANDEVUSSE_RESULT_SUBDIR = VANDEVUSSE_ROOT / "Results"

# Plant-only defaults for the first Van de Vusse case-study phase.
VANDEVUSSE_DELTA_T_HOURS = 0.05

VANDEVUSSE_SYSTEM_PARAMS = np.array(
    [
        1.287e12,
        1.287e12,
        9.043e9,
        -9758.3,
        -9758.3,
        -8560.0,
        4.2,
        -11.0,
        -41.85,
        0.9342,
        3.01,
        2.00,
        4032.0,
        0.215,
        0.01,
        5.0,
    ],
    dtype=float,
)

VANDEVUSSE_DESIGN_PARAMS = np.array([5.10, 378.1], dtype=float)
VANDEVUSSE_SS_INPUTS = np.array([14.19, -1113.5], dtype=float)
