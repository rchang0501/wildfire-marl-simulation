import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.utils import must_return_to_base, step_toward
from environment.jurisdiction_env import JurisdictionEnv


def lp_assign(jenv: JurisdictionEnv) -> np.ndarray:
    """LP-based suppression assignment for a single jurisdiction.

    Formulates unit-to-fire-cell assignment as a linear program that minimizes
    total Manhattan distance.  The LP relaxation is guaranteed to produce
    integer solutions because the constraint matrix is totally unimodular
    (standard assignment / transportation structure).

    Targets are burning cells (capacity 1 each).  When there are more units
    than burning cells, extra "idle" targets at the grid center absorb the
    surplus, so every unit gets an assignment.  When there are more fires than
    units, some fires remain unassigned.

    Returns:
        actions: (num_units, 2) int array of (dx, dy)
    """
    n = jenv.num_units
    C = jenv.cols
    m = int(jenv.movement_per_step)

    actions = np.zeros((n, 2), dtype=int)

    # Fuel: units that must return to base get step_toward(center) immediately
    returning = must_return_to_base(jenv)
    for i in np.nonzero(returning)[0]:
        cur_cell = int(jenv.unit_positions[i])
        cur_r = int(jenv.cell_row[cur_cell])
        cur_c = int(jenv.cell_col[cur_cell])
        dx, dy = step_toward(cur_r, cur_c, jenv.center_cell_row, jenv.center_cell_col, m)
        actions[i] = (dx, dy)

    # Active (non-returning) unit indices
    active_mask = ~returning
    active_indices = np.nonzero(active_mask)[0]
    n_active = len(active_indices)

    burning_rc = np.argwhere(jenv.burning_map)  # (K, 2)
    if burning_rc.size == 0:
        # No fires -- move active units toward center
        for i in active_indices:
            cur_cell = int(jenv.unit_positions[i])
            cur_r = int(jenv.cell_row[cur_cell])
            cur_c = int(jenv.cell_col[cur_cell])
            dx, dy = step_toward(
                cur_r, cur_c, jenv.center_cell_row, jenv.center_cell_col, m
            )
            actions[i] = (dx, dy)
        return actions

    if n_active == 0:
        return actions

    # ---- Build target list ------------------------------------------------
    # Each burning cell is a target with capacity 1.
    # If units > fires, pad with center-cell targets so every unit is assigned.
    fire_rows = burning_rc[:, 0]
    fire_cols = burning_rc[:, 1]
    n_fires = len(fire_rows)

    n_idle = max(0, n_active - n_fires)
    n_targets = n_fires + n_idle

    tgt_rows = np.empty(n_targets, dtype=int)
    tgt_cols = np.empty(n_targets, dtype=int)
    tgt_rows[:n_fires] = fire_rows
    tgt_cols[:n_fires] = fire_cols
    if n_idle > 0:
        tgt_rows[n_fires:] = jenv.center_cell_row
        tgt_cols[n_fires:] = jenv.center_cell_col

    # ---- Unit positions as (row, col) for active units only ---------------
    active_positions = jenv.unit_positions[active_indices]
    unit_rows = jenv.cell_row[active_positions]
    unit_cols = jenv.cell_col[active_positions]

    # ---- Cost vector c (flattened n_active x n_targets) -------------------
    # c[i * n_targets + j] = manhattan distance from active unit i to target j
    dr = np.abs(unit_rows[:, None] - tgt_rows[None, :])  # (n_active, n_targets)
    dc = np.abs(unit_cols[:, None] - tgt_cols[None, :])
    cost_matrix = (dr + dc).astype(np.float64)
    c = cost_matrix.ravel()

    num_vars = n_active * n_targets

    # ---- Equality constraints: each unit assigned to exactly one target ---
    # sum_j x[i, j] = 1  for each unit i
    A_eq = lil_matrix((n_active, num_vars), dtype=np.float64)
    for i in range(n_active):
        start = i * n_targets
        A_eq[i, start : start + n_targets] = 1.0
    b_eq = np.ones(n_active)

    # ---- Inequality constraints: each target gets at most one unit --------
    # sum_i x[i, j] <= 1  for each target j
    A_ub = lil_matrix((n_targets, num_vars), dtype=np.float64)
    for j in range(n_targets):
        for i in range(n_active):
            A_ub[j, i * n_targets + j] = 1.0
    b_ub = np.ones(n_targets)

    # ---- Solve LP ---------------------------------------------------------
    bounds = [(0.0, 1.0)] * num_vars

    result = linprog(
        c,
        A_ub=A_ub.tocsc(),
        b_ub=b_ub,
        A_eq=A_eq.tocsc(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        # Fallback: move all active units toward center
        for idx, i in enumerate(active_indices):
            cur_r = int(unit_rows[idx])
            cur_c = int(unit_cols[idx])
            dx, dy = step_toward(
                cur_r, cur_c, jenv.center_cell_row, jenv.center_cell_col, m
            )
            actions[i] = (dx, dy)
        return actions

    # ---- Extract assignment from LP solution ------------------------------
    x = result.x.reshape(n_active, n_targets)

    for idx in range(n_active):
        orig_i = active_indices[idx]
        j = int(np.argmax(x[idx]))  # assigned target
        tgt_r = int(tgt_rows[j])
        tgt_c = int(tgt_cols[j])
        cur_r = int(unit_rows[idx])
        cur_c = int(unit_cols[idx])

        dist = abs(tgt_r - cur_r) + abs(tgt_c - cur_c)
        if dist == 0:
            continue
        elif dist <= m:
            actions[orig_i] = (tgt_c - cur_c, tgt_r - cur_r)
        else:
            dx, dy = step_toward(cur_r, cur_c, tgt_r, tgt_c, m)
            actions[orig_i] = (dx, dy)

    return actions


class LPSuppressionAlgorithm(SuppressionAlgorithm):
    """LP-based optimal suppression assignment."""

    name = "lp_suppression"

    def actions(self, jenv: JurisdictionEnv, rng: np.random.Generator) -> np.ndarray:
        return lp_assign(jenv)
