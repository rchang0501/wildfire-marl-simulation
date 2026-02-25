from dataclasses import dataclass

import numpy as np

from environment.jurisdiction_env import JurisdictionEnv


@dataclass
class TransitUnit:
    unit_id: int
    from_juris: int
    to_juris: int
    remaining_steps: int
    fuel: int | None = None


class MultiJurisdictionEnv:
    """Composes multiple JurisdictionEnv instances and manages inter-jurisdiction transit."""

    def __init__(
        self,
        *,
        num_juris_rows: int,
        num_juris_cols: int,
        per_juris_rows: int,
        per_juris_cols: int,
        base_spread_prob: float,
        num_units_per_juris: int,
        suppression_success_prob: float,
        movement_per_step: int,
        juris_travel_time: int,
        lightning_mu_log: float,
        lightning_sigma_log: float,
        max_fuel: int | None = None,
        fuel_refuel_rate: int = 1,
    ):
        self.num_juris_rows = int(num_juris_rows)
        self.num_juris_cols = int(num_juris_cols)
        if self.num_juris_rows <= 0 or self.num_juris_cols <= 0:
            raise ValueError("num_juris_rows and num_juris_cols must be positive.")
        self.num_juris = self.num_juris_rows * self.num_juris_cols

        self.per_juris_rows = int(per_juris_rows)
        self.per_juris_cols = int(per_juris_cols)
        self.num_units_per_juris = int(num_units_per_juris)
        self.movement_per_step = int(movement_per_step)
        self.juris_travel_time = int(juris_travel_time)

        # Jurisdiction grid coordinates
        self.juris_row = np.arange(self.num_juris, dtype=int) // self.num_juris_cols
        self.juris_col = np.arange(self.num_juris, dtype=int) % self.num_juris_cols

        # Adjacency matrix
        self.adj_matrix: list[list[int]] = [
            [-1] * self.num_juris for _ in range(self.num_juris)
        ]
        for j in range(self.num_juris):
            self.adj_matrix[j][j] = 0
            r, c = int(self.juris_row[j]), int(self.juris_col[j])
            if r > 0:
                self.adj_matrix[j][j - self.num_juris_cols] = self.juris_travel_time
            if r < self.num_juris_rows - 1:
                self.adj_matrix[j][j + self.num_juris_cols] = self.juris_travel_time
            if c > 0:
                self.adj_matrix[j][j - 1] = self.juris_travel_time
            if c < self.num_juris_cols - 1:
                self.adj_matrix[j][j + 1] = self.juris_travel_time

        self.max_fuel = max_fuel
        self.fuel_refuel_rate = int(fuel_refuel_rate)

        # Create jurisdiction environments
        common_kwargs = dict(
            rows=per_juris_rows,
            cols=per_juris_cols,
            base_spread_prob=base_spread_prob,
            suppression_success_prob=suppression_success_prob,
            movement_per_step=movement_per_step,
            lightning_mu_log=lightning_mu_log,
            lightning_sigma_log=lightning_sigma_log,
            max_fuel=max_fuel,
            fuel_refuel_rate=fuel_refuel_rate,
        )
        self.jurisdictions: list[JurisdictionEnv] = [
            JurisdictionEnv(num_units=num_units_per_juris, **common_kwargs)
            for _ in range(self.num_juris)
        ]

        # Expose params that algorithms / main.py may need
        self.base_spread_prob = float(base_spread_prob)
        self.suppression_success_prob = float(suppression_success_prob)
        self.lightning_mu_log = float(lightning_mu_log)
        self.lightning_sigma_log = float(lightning_sigma_log)

        # Global unit tracking
        self.num_units_total = self.num_juris * num_units_per_juris
        # unit_jurisdiction[uid] = jurisdiction index (-1 if in transit)
        self.unit_jurisdiction = np.repeat(
            np.arange(self.num_juris, dtype=int), num_units_per_juris
        )
        # unit_local_index[uid] = index within that jurisdiction's unit_positions (-1 if in transit)
        self.unit_local_index = np.tile(
            np.arange(num_units_per_juris, dtype=int), self.num_juris
        )

        self.transit_units: list[TransitUnit] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def burning_counts(self) -> list[int]:
        return [j.burning_count for j in self.jurisdictions]

    @property
    def total_burning(self) -> int:
        return sum(j.burning_count for j in self.jurisdictions)

    @property
    def unit_count_per_juris(self) -> list[int]:
        return [j.num_units for j in self.jurisdictions]

    @property
    def num_transit_units(self) -> int:
        return len(self.transit_units)

    # ------------------------------------------------------------------
    # Transfer management
    # ------------------------------------------------------------------

    def initiate_transfer(self, unit_id: int, target_juris: int) -> None:
        """Begin transferring a unit from its current jurisdiction to target_juris."""
        cur_j = int(self.unit_jurisdiction[unit_id])
        if cur_j < 0:
            raise ValueError(f"Unit {unit_id} is already in transit.")

        local_idx = int(self.unit_local_index[unit_id])
        jenv = self.jurisdictions[cur_j]

        # Must be at center cell
        if int(jenv.unit_positions[local_idx]) != jenv.center_cell:
            raise ValueError(f"Unit {unit_id} must be at center cell to transfer.")

        travel_time = self.adj_matrix[cur_j][target_juris]
        if travel_time < 0:
            raise ValueError(
                f"No direct connection from jurisdiction {cur_j} to {target_juris}."
            )

        # Save fuel before removing from source jurisdiction
        saved_fuel = None
        if jenv.unit_fuel is not None:
            saved_fuel = int(jenv.unit_fuel[local_idx])

        # Remove from source jurisdiction
        jenv.remove_units([local_idx])

        # Fix local indices for units whose local index shifted
        mask = (self.unit_jurisdiction == cur_j) & (self.unit_local_index > local_idx)
        self.unit_local_index[mask] -= 1

        # Mark unit as in transit
        self.unit_jurisdiction[unit_id] = -1
        self.unit_local_index[unit_id] = -1

        self.transit_units.append(
            TransitUnit(
                unit_id=unit_id,
                from_juris=cur_j,
                to_juris=target_juris,
                remaining_steps=travel_time,
                fuel=saved_fuel,
            )
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def advance_transit(self) -> None:
        """Advance transit timers and deliver arrived units to their destinations.

        Call this before computing suppression actions so that newly arrived
        units are included in the jurisdiction's unit count.
        """
        arrivals: list[TransitUnit] = []
        still_in_transit: list[TransitUnit] = []
        for tu in self.transit_units:
            tu.remaining_steps -= 1
            if tu.remaining_steps <= 0:
                arrivals.append(tu)
            else:
                still_in_transit.append(tu)
        self.transit_units = still_in_transit

        for tu in arrivals:
            dest = self.jurisdictions[tu.to_juris]
            new_local_idx = dest.num_units
            fuel_arg = [tu.fuel] if tu.fuel is not None else None
            dest.add_units([dest.center_cell], fuel_levels=fuel_arg)
            self.unit_jurisdiction[tu.unit_id] = tu.to_juris
            self.unit_local_index[tu.unit_id] = new_local_idx

    def step(
        self,
        suppression_actions: dict[int, np.ndarray],
        rng_spread: np.random.Generator,
        rng_lightning: np.random.Generator,
    ):
        """Step each jurisdiction independently with its suppression actions.

        Call advance_transit() before this method so that arrived units are
        already present in their destination jurisdictions.

        Args:
            suppression_actions: {juris_index: (num_units_in_juris, 2) array of (dx, dy)}
            rng_spread: RNG for fire spread and suppression
            rng_lightning: RNG for lightning ignitions

        Returns:
            (rewards_per_juris, counts_per_juris)
        """
        rewards: list[float] = []
        counts: list[int] = []
        for j_idx, jenv in enumerate(self.jurisdictions):
            actions = suppression_actions.get(j_idx)
            if actions is None:
                actions = np.zeros((jenv.num_units, 2), dtype=int)
            _, _, _, reward, count = jenv.step(
                actions, rng_spread=rng_spread, rng_lightning=rng_lightning
            )
            rewards.append(reward)
            counts.append(count)

        return rewards, counts

    # ------------------------------------------------------------------
    # Snapshot (for animator compatibility)
    # ------------------------------------------------------------------

    def get_snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Return (J, R, C) burning map, (N, 2) unit positions, and (N,) unit fuel.

        unit_positions[:, 0] = jurisdiction index
        unit_positions[:, 1] = flat cell index (or negative if in transit)
        unit_fuel is None when fuel is disabled.
        """
        J = self.num_juris
        R = self.per_juris_rows
        C = self.per_juris_cols

        burning = np.zeros((J, R, C), dtype=bool)
        for j_idx, jenv in enumerate(self.jurisdictions):
            burning[j_idx] = jenv.burning_map

        unit_positions = np.zeros((self.num_units_total, 2), dtype=int)
        unit_fuel: np.ndarray | None = None
        if self.max_fuel is not None:
            unit_fuel = np.zeros(self.num_units_total, dtype=int)

        for uid in range(self.num_units_total):
            j = int(self.unit_jurisdiction[uid])
            if j < 0:
                # In transit -- encode as (to_juris, -remaining_steps)
                tu = next(t for t in self.transit_units if t.unit_id == uid)
                unit_positions[uid, 0] = tu.to_juris
                unit_positions[uid, 1] = -tu.remaining_steps
                if unit_fuel is not None and tu.fuel is not None:
                    unit_fuel[uid] = tu.fuel
            else:
                local_idx = int(self.unit_local_index[uid])
                jenv = self.jurisdictions[j]
                unit_positions[uid, 0] = j
                unit_positions[uid, 1] = int(jenv.unit_positions[local_idx])
                if unit_fuel is not None and jenv.unit_fuel is not None:
                    unit_fuel[uid] = int(jenv.unit_fuel[local_idx])

        return burning, unit_positions, unit_fuel
