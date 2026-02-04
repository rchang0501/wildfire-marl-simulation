import numpy as np

from algorithms.sharing_algorithms.algorithm_base import SharingAlgorithm
from algorithms.utils import step_toward


class PeriodicTransferSharingAlgorithm(SharingAlgorithm):
    """Every s steps, select the best (least burning) and worst (most burning)
    jurisdictions, then move one unit from the best to the worst. If they are
    not adjacent, the unit is routed hop-by-hop through adjacent jurisdictions.

    Three-phase state machine:
    1. Select: pick worst/best jurisdictions, choose unit closest to center in best
    2. Steer: return steering (dx, dy) to move unit toward center
    3. Hop: once at center, return (unit_id, next_hop) transfer; repeat for multi-hop
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
        self.active_unit_id: int | None = None
        self.active_target_juris: int | None = None

    def decide_transfers(self, multi_env, rng) -> list[tuple[int, int]]:
        if self.disabled:
            return []

        if self.active_unit_id is None:
            return []

        uid = self.active_unit_id
        cur_j = int(multi_env.unit_jurisdiction[uid])

        # Unit is in transit -- nothing to do
        if cur_j < 0:
            return []

        jenv = multi_env.jurisdictions[cur_j]
        local_idx = int(multi_env.unit_local_index[uid])
        cur_cell = int(jenv.unit_positions[local_idx])

        # Not at center yet -- steering will handle it
        if cur_cell != jenv.center_cell:
            return []

        tgt_j = self.active_target_juris

        # Already at destination
        if cur_j == tgt_j:
            self.active_unit_id = None
            self.active_target_juris = None
            self.cooldown = self.period_s
            return []

        # Compute next hop toward target
        cur_r = int(multi_env.juris_row[cur_j])
        cur_c = int(multi_env.juris_col[cur_j])
        tgt_r = int(multi_env.juris_row[tgt_j])
        tgt_c = int(multi_env.juris_col[tgt_j])
        dr = tgt_r - cur_r
        dc = tgt_c - cur_c

        if abs(dr) + abs(dc) == 1:
            next_j = tgt_j
        else:
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
            if step_r != 0:
                next_j = cur_j + step_r * multi_env.num_juris_cols
            else:
                next_j = cur_j + step_c

        return [(uid, next_j)]

    def get_steering_actions(self, multi_env, rng) -> dict[int, tuple[int, int]]:
        if self.disabled:
            return {}

        if self.cooldown > 0:
            self.cooldown -= 1
            return {}

        # Phase 1: Select unit if none active
        if self.active_unit_id is None:
            burning_counts = np.array(multi_env.burning_counts, dtype=int)
            worst_juris = int(np.argmax(burning_counts))
            best_juris = int(np.argmin(burning_counts))

            if worst_juris == best_juris:
                return {}
            if burning_counts[worst_juris] <= burning_counts[best_juris]:
                return {}

            source_juris = best_juris
            target_juris = worst_juris

            jenv = multi_env.jurisdictions[source_juris]
            if jenv.num_units == 0:
                return {}

            # Find global unit IDs in source jurisdiction
            global_mask = multi_env.unit_jurisdiction == source_juris
            global_ids = np.nonzero(global_mask)[0]
            if global_ids.size == 0:
                return {}

            # Find the one closest to center
            local_indices = multi_env.unit_local_index[global_ids]
            cells = jenv.unit_positions[local_indices]
            cur_rs = jenv.cell_row[cells]
            cur_cs = jenv.cell_col[cells]
            dists = np.abs(cur_rs - jenv.center_cell_row) + np.abs(cur_cs - jenv.center_cell_col)
            best_idx = int(np.argmin(dists))
            self.active_unit_id = int(global_ids[best_idx])
            self.active_target_juris = int(target_juris)

        if self.active_unit_id is None:
            return {}

        uid = self.active_unit_id
        cur_j = int(multi_env.unit_jurisdiction[uid])

        # In transit -- no steering needed
        if cur_j < 0:
            return {}

        jenv = multi_env.jurisdictions[cur_j]
        local_idx = int(multi_env.unit_local_index[uid])
        cur_cell = int(jenv.unit_positions[local_idx])

        # At center -- transfer will be handled by decide_transfers
        if cur_cell == jenv.center_cell:
            return {}

        # Steer toward center
        cur_r = int(jenv.cell_row[cur_cell])
        cur_c = int(jenv.cell_col[cur_cell])
        dx, dy = step_toward(
            cur_r, cur_c,
            jenv.center_cell_row, jenv.center_cell_col,
            jenv.movement_per_step,
        )
        return {uid: (dx, dy)}
