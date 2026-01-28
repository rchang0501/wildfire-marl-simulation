import numpy as np
from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r2 - r1) + abs(c2 - c1)


def step_toward(cur_r: int, cur_c: int, tgt_r: int, tgt_c: int, m: int) -> tuple[int, int]:
    dr = tgt_r - cur_r
    dc = tgt_c - cur_c

    dx = int(np.clip(dc, -m, m))
    rem = m - abs(dx)
    dy = int(np.clip(dr, -rem, rem))
    return dx, dy


def greedy(env) -> np.ndarray:
        """
        Greedy suppression. For each jurisdiction:
            - For every unit (sorted by current flattened cell index),
                pick the closest burning cell (Manhattan distance).
            - If it can step onto the fire this turn, do so; else take the biggest step toward it.
            - After assigning a target to a unit, that burning cell is considered "claimed" and is
                removed from candidates for other units this step.

        Returns:
            actions: (N,3) int array of (tag,a,b)
                tag=0 move by (dx,dy)=(a,b)
                tag=2 stay
        """
        J = env.num_juris
        C = env.per_juris_cols
        m = int(env.movement_per_step)

        actions = np.zeros((env.num_units_total, 3), dtype=int)
        actions[:, 0] = 2  # default: stay

        for j in range(J):
            burning_rc = np.argwhere(env.burning_map[j])  # (K,2)
            if burning_rc.size == 0:
                continue

            burning_flat = (burning_rc[:, 0] * C + burning_rc[:, 1]).astype(int)
            available_targets = set(burning_flat.tolist())

            unit_indices = np.nonzero(env.unit_positions[:, 0] == j)[0]
            if unit_indices.size == 0:
                continue

            unit_cells = env.unit_positions[unit_indices, 1].astype(int)
            unit_order = unit_indices[np.argsort(unit_cells, kind="stable")]

            for u_idx in unit_order:
                cur_cell = int(env.unit_positions[u_idx, 1])

                if cur_cell < 0 or not available_targets:
                    actions[u_idx] = (2, 0, 0)
                    continue

                cur_r = int(env.per_juris_cell_row[cur_cell])
                cur_c = int(env.per_juris_cell_col[cur_cell])

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
                    actions[u_idx] = (2, 0, 0)
                elif best_dist <= m:
                    dx = tgt_c - cur_c
                    dy = tgt_r - cur_r
                    actions[u_idx] = (0, int(dx), int(dy))
                else:
                    dx, dy = step_toward(cur_r, cur_c, tgt_r, tgt_c, m)
                    actions[u_idx] = (0, int(dx), int(dy))

        idle = (actions[:, 0] == 2)
        not_in_transit = env.unit_positions[:, 1] >= 0
        remaining = np.nonzero(idle & not_in_transit)[0]
        for u_idx in remaining:
            cur_cell = int(env.unit_positions[u_idx, 1])
            if cur_cell < 0:
                continue
            cur_r = int(env.per_juris_cell_row[cur_cell])
            cur_c = int(env.per_juris_cell_col[cur_cell])
            dx, dy = step_toward(cur_r, cur_c, env.center_cell_row, env.center_cell_col, m)
            if dx != 0 or dy != 0:
                actions[u_idx] = (0, int(dx), int(dy))

        return actions


class GreedyAlgorithm(SuppressionAlgorithm):
    """
    Greedy suppression heuristic.
    """

    name = "greedy"

    def actions(
        self,
        env,
        rng: np.random.Generator,
        assigned_mask: np.ndarray,
        assigned_actions: np.ndarray,
    ) -> np.ndarray:
        # Heuristic is deterministic and does not use rng.
        actions = greedy(env)
        actions[assigned_mask] = assigned_actions[assigned_mask]
        return actions
