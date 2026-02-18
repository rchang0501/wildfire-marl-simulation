import numpy as np

from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.utils import manhattan_distance, must_return_to_base, step_toward
from environment.jurisdiction_env import JurisdictionEnv


def greedy(jenv: JurisdictionEnv) -> np.ndarray:
    """Greedy suppression for a single jurisdiction.

    For every unit (sorted by current cell index), pick the closest burning cell
    (Manhattan distance). If it can step onto the fire this turn, do so; else
    take the biggest step toward it. After assigning a target, that cell is
    removed from candidates for other units this step.

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

    burning_rc = np.argwhere(jenv.burning_map)  # (K, 2)
    if burning_rc.size == 0:
        # No fires: move idle (non-returning) units toward center
        for i in range(n):
            if returning[i]:
                continue
            cur_cell = int(jenv.unit_positions[i])
            cur_r = int(jenv.cell_row[cur_cell])
            cur_c = int(jenv.cell_col[cur_cell])
            dx, dy = step_toward(cur_r, cur_c, jenv.center_cell_row, jenv.center_cell_col, m)
            actions[i] = (dx, dy)
        return actions

    burning_flat = (burning_rc[:, 0] * C + burning_rc[:, 1]).astype(int)
    available_targets = set(burning_flat.tolist())

    # Sort units by cell index for deterministic ordering
    unit_order = np.argsort(jenv.unit_positions, kind="stable")

    for i in unit_order:
        if returning[i]:
            continue

        cur_cell = int(jenv.unit_positions[i])

        if not available_targets:
            # No targets left -- stay
            continue

        cur_r = int(jenv.cell_row[cur_cell])
        cur_c = int(jenv.cell_col[cur_cell])

        best_tgt_flat = None
        best_dist = None

        for tgt_flat in available_targets:
            tgt_r = tgt_flat // C
            tgt_c = tgt_flat % C
            d = manhattan_distance(cur_r, cur_c, tgt_r, tgt_c)
            if best_dist is None or d < best_dist or (d == best_dist and tgt_flat < best_tgt_flat):
                best_dist = d
                best_tgt_flat = tgt_flat

        available_targets.remove(best_tgt_flat)

        tgt_r = best_tgt_flat // C
        tgt_c = best_tgt_flat % C

        if best_dist == 0:
            # Already on fire cell -- stay
            continue
        elif best_dist <= m:
            dx = tgt_c - cur_c
            dy = tgt_r - cur_r
            actions[i] = (int(dx), int(dy))
        else:
            dx, dy = step_toward(cur_r, cur_c, tgt_r, tgt_c, m)
            actions[i] = (int(dx), int(dy))

    # Idle units (no fire target assigned) move toward center, skip returning
    for i in range(n):
        if returning[i]:
            continue
        if actions[i, 0] == 0 and actions[i, 1] == 0:
            cur_cell = int(jenv.unit_positions[i])
            cur_r = int(jenv.cell_row[cur_cell])
            cur_c = int(jenv.cell_col[cur_cell])
            dx, dy = step_toward(cur_r, cur_c, jenv.center_cell_row, jenv.center_cell_col, m)
            if dx != 0 or dy != 0:
                actions[i] = (int(dx), int(dy))

    return actions


class GreedyAlgorithm(SuppressionAlgorithm):
    """Greedy suppression heuristic."""

    name = "greedy"

    def actions(self, jenv: JurisdictionEnv, rng: np.random.Generator) -> np.ndarray:
        return greedy(jenv)
