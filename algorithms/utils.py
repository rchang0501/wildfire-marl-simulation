import math

import numpy as np


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r2 - r1) + abs(c2 - c1)


def step_toward(cur_r: int, cur_c: int, tgt_r: int, tgt_c: int, m: int) -> tuple[int, int]:
    dr = tgt_r - cur_r
    dc = tgt_c - cur_c

    dx = int(np.clip(dc, -m, m))
    rem = m - abs(dx)
    dy = int(np.clip(dr, -rem, rem))
    return dx, dy


def must_return_to_base(jenv) -> np.ndarray:
    """Return a (num_units,) bool mask indicating which units must head back to base.

    A unit must return when its fuel is at or below the number of steps needed
    to reach the center cell (base).  Returns all-False when fuel is disabled
    (``jenv.unit_fuel is None``).
    """
    if jenv.unit_fuel is None:
        return np.zeros(jenv.num_units, dtype=bool)

    m = jenv.movement_per_step
    center_r = jenv.center_cell_row
    center_c = jenv.center_cell_col

    unit_r = jenv.cell_row[jenv.unit_positions]
    unit_c = jenv.cell_col[jenv.unit_positions]
    dist = np.abs(unit_r - center_r) + np.abs(unit_c - center_c)
    steps_needed = np.ceil(dist / m).astype(int)

    return jenv.unit_fuel <= steps_needed
