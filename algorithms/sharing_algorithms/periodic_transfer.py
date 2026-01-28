import numpy as np
from algorithms.sharing_algorithms.algorithm_base import SharingAlgorithm


def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r2 - r1) + abs(c2 - c1)


def step_toward(cur_r: int, cur_c: int, tgt_r: int, tgt_c: int, m: int) -> tuple[int, int]:
    dr = tgt_r - cur_r
    dc = tgt_c - cur_c

    dx = int(np.clip(dc, -m, m))
    rem = m - abs(dx)
    dy = int(np.clip(dr, -rem, rem))
    return dx, dy


class PeriodicTransferSharingAlgorithm(SharingAlgorithm):
    """
    Every s steps, select the best (least burning) and worst (most burning)
    jurisdictions, then move one unit from the best to the worst. If they are
    not adjacent, the unit is routed hop-by-hop through adjacent jurisdictions
    and remains under this algorithm's control until it reaches the destination.
    """

    name = "periodic_transfer"

    def __init__(self, param_dir: str | None = None, params: dict | None = None):
        super().__init__(param_dir=param_dir, params=params)
        self.period_s = int(self.params.get("period_s", 10))
        if self.period_s <= 0:
            raise ValueError("period_s must be positive.")
        self.total_steps = self.params.get("total_steps", None)
        if self.total_steps is not None:
            self.total_steps = int(self.total_steps)
        self.disabled = self.total_steps is not None and self.period_s > self.total_steps
        self.cooldown = 0
        self.active_unit_idx: int | None = None
        self.active_target_juris: int | None = None

    def assignments(self, env, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        assigned_mask = np.zeros((env.num_units_total,), dtype=bool)
        actions = self._default_actions(env)

        if self.disabled:
            return assigned_mask, actions

        if self.cooldown > 0:
            self.cooldown -= 1
            return assigned_mask, actions

        if self.active_unit_idx is None:
            burning_counts = np.sum(env.burning_map, axis=(1, 2)).astype(int)
            worst_juris = int(np.argmax(burning_counts))
            best_juris = int(np.argmin(burning_counts))

            if worst_juris == best_juris:
                return assigned_mask, actions

            if burning_counts[worst_juris] <= burning_counts[best_juris]:
                return assigned_mask, actions

            source_juris = best_juris
            target_juris = worst_juris

            unit_indices = np.nonzero(env.unit_positions[:, 0] == source_juris)[0]
            if unit_indices.size == 0:
                return assigned_mask, actions

            cells = env.unit_positions[unit_indices, 1].astype(int)
            mask = cells >= 0
            if not np.any(mask):
                return assigned_mask, actions

            unit_indices = unit_indices[mask]
            cells = cells[mask]

            cur_rs = env.per_juris_cell_row[cells]
            cur_cs = env.per_juris_cell_col[cells]
            dists = np.abs(cur_rs - env.center_cell_row) + np.abs(cur_cs - env.center_cell_col)
            self.active_unit_idx = int(unit_indices[np.argmin(dists)])
            self.active_target_juris = int(target_juris)

        if self.active_unit_idx is None or self.active_target_juris is None:
            return assigned_mask, actions

        chosen_idx = self.active_unit_idx
        chosen_cell = int(env.unit_positions[chosen_idx, 1])

        if chosen_cell < 0:
            assigned_mask[chosen_idx] = True
            return assigned_mask, actions

        if chosen_cell == env.center_cell_index:
            cur_j = int(env.unit_positions[chosen_idx, 0])
            tgt_j = int(self.active_target_juris)
            if cur_j != tgt_j:
                cur_r = int(env.juris_row[cur_j])
                cur_c = int(env.juris_col[cur_j])
                tgt_r = int(env.juris_row[tgt_j])
                tgt_c = int(env.juris_col[tgt_j])
                dr = tgt_r - cur_r
                dc = tgt_c - cur_c

                if abs(dr) + abs(dc) == 1:
                    next_j = tgt_j
                else:
                    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
                    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
                    if step_r != 0:
                        next_j = cur_j + step_r * env.num_juris_cols
                    else:
                        next_j = cur_j + step_c

                actions[chosen_idx] = (1, int(next_j), 0)
                assigned_mask[chosen_idx] = True
                return assigned_mask, actions

            actions[chosen_idx] = (2, 0, 0)
            assigned_mask[chosen_idx] = True
            self.active_unit_idx = None
            self.active_target_juris = None
            self.cooldown = self.period_s
            return assigned_mask, actions

        cur_r = int(env.per_juris_cell_row[chosen_cell])
        cur_c = int(env.per_juris_cell_col[chosen_cell])
        dx, dy = step_toward(cur_r, cur_c, env.center_cell_row, env.center_cell_col, env.movement_per_step)
        actions[chosen_idx] = (0, int(dx), int(dy))
        assigned_mask[chosen_idx] = True
        return assigned_mask, actions
