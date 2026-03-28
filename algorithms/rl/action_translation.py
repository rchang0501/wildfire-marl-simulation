"""Translate RL agent actions to (dx, dy) displacements for JurisdictionEnv.

The RL agent selects from K nearest fires + 1 idle action per drone.
This module converts those selections into movement commands.
"""

import numpy as np

from algorithms.utils import must_return_to_base, step_toward
from environment.jurisdiction_env import JurisdictionEnv

from algorithms.rl.observation import K_NEAREST


def translate_actions(
    jenv: JurisdictionEnv,
    agent_actions: np.ndarray,
    k_nearest_targets: np.ndarray,
) -> np.ndarray:
    """Convert per-drone fire selections to (dx, dy) displacements.

    Args:
        jenv: The jurisdiction environment.
        agent_actions: (num_units,) int array, each in {0, ..., K}.
            0..K-1 select one of the K nearest fires.
            K = idle / return to center.
        k_nearest_targets: (num_units, K, 2) array of (row, col) target positions.

    Returns:
        actions: (num_units, 2) int array of (dx, dy) displacements.
    """
    n = jenv.num_units
    m = jenv.movement_per_step
    actions = np.zeros((n, 2), dtype=int)

    center_r = jenv.center_cell_row
    center_c = jenv.center_cell_col

    # Fuel override: units that must return get forced to center
    returning = must_return_to_base(jenv)

    unit_r = jenv.cell_row[jenv.unit_positions]
    unit_c = jenv.cell_col[jenv.unit_positions]

    for i in range(n):
        cur_r = int(unit_r[i])
        cur_c = int(unit_c[i])

        if returning[i]:
            dx, dy = step_toward(cur_r, cur_c, center_r, center_c, m)
            actions[i] = (dx, dy)
            continue

        action_idx = int(agent_actions[i])

        if action_idx >= K_NEAREST:
            # Idle: move toward center
            dx, dy = step_toward(cur_r, cur_c, center_r, center_c, m)
            actions[i] = (dx, dy)
        else:
            tgt_r = int(k_nearest_targets[i, action_idx, 0])
            tgt_c = int(k_nearest_targets[i, action_idx, 1])
            dx, dy = step_toward(cur_r, cur_c, tgt_r, tgt_c, m)
            actions[i] = (dx, dy)

    return actions
