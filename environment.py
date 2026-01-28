import numpy as np

class Fire_Environment:
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
    ):
        self.num_juris_rows = int(num_juris_rows)
        self.num_juris_cols = int(num_juris_cols)
        if self.num_juris_rows <= 0 or self.num_juris_cols <= 0:
            raise ValueError("num_juris_rows and num_juris_cols must be positive.")
        self.num_juris = self.num_juris_rows * self.num_juris_cols

        self.per_juris_rows = int(per_juris_rows)
        self.per_juris_cols = int(per_juris_cols)
        self.per_juris_shape = (self.per_juris_rows, self.per_juris_cols)

        self.juris_travel_time = int(juris_travel_time)
        if self.juris_travel_time < 0:
            raise ValueError("juris_travel_time must be >= 0.")

        self.juris_indices = np.arange(self.num_juris, dtype=int)
        self.juris_row = self.juris_indices // self.num_juris_cols
        self.juris_col = self.juris_indices % self.num_juris_cols

        self.juris_neighbors: list[list[int]] = []
        for j in range(self.num_juris):
            r = int(self.juris_row[j])
            c = int(self.juris_col[j])
            neighbors: list[int] = []
            if r > 0:
                neighbors.append(j - self.num_juris_cols)
            if r < self.num_juris_rows - 1:
                neighbors.append(j + self.num_juris_cols)
            if c > 0:
                neighbors.append(j - 1)
            if c < self.num_juris_cols - 1:
                neighbors.append(j + 1)
            self.juris_neighbors.append(neighbors)

        self.adj_matrix: list[list[int]] = [[-1 for _ in range(self.num_juris)] for _ in range(self.num_juris)]
        for j in range(self.num_juris):
            self.adj_matrix[j][j] = 0
            for k in self.juris_neighbors[j]:
                self.adj_matrix[j][k] = self.juris_travel_time

        self.movement_per_step = int(movement_per_step)

        self.burning_map = np.zeros((self.num_juris, self.per_juris_rows, self.per_juris_cols), dtype=bool)

        self.num_units_per_juris = int(num_units_per_juris)
        self.num_units_total = self.num_juris * self.num_units_per_juris

        self.per_juris_cell_indices = np.arange(self.per_juris_rows * self.per_juris_cols, dtype=int)
        self.per_juris_cell_row = self.per_juris_cell_indices // self.per_juris_cols
        self.per_juris_cell_col = self.per_juris_cell_indices % self.per_juris_cols

        self.center_cell_row = self.per_juris_rows // 2
        self.center_cell_col = self.per_juris_cols // 2
        self.center_cell_index = self.center_cell_row * self.per_juris_cols + self.center_cell_col

        self.unit_positions = np.zeros((self.num_units_total, 2), dtype=int)
        unit_juris = np.repeat(np.arange(self.num_juris, dtype=int), self.num_units_per_juris)
        self.unit_positions[:, 0] = unit_juris
        self.unit_positions[:, 1] = self.center_cell_index

        self.suppression_success_prob = float(suppression_success_prob)
        self.base_spread_prob = float(base_spread_prob)
        self.lightning_mu_log = float(lightning_mu_log)
        self.lightning_sigma_log = float(lightning_sigma_log)


    def probability_of_extinguishing_fire(self, num_units: int) -> float:
        return 1.0 - (1.0 - self.suppression_success_prob) ** num_units

    def _units_per_cell(
        self,
        unit_positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Vectorized version.
        units_per_cell[loc_j, r, c] = number of units physically present at that cell
        in jurisdiction-map loc_j. Units in transit (cell < 0) contribute 0.
        """
        units_per_cell = np.zeros(
            (self.num_juris, self.per_juris_rows, self.per_juris_cols),
            dtype=np.int16,
        )

        if unit_positions is None:
            unit_positions = self.unit_positions

        loc_j = unit_positions[:, 0].astype(int, copy=False)
        cell = unit_positions[:, 1].astype(int, copy=False)

        # Keep only units not in transit
        mask = cell >= 0
        if not np.any(mask):
            return units_per_cell

        loc_j = loc_j[mask]
        cell  = cell[mask]

        # Map flat cell index -> (row, col) using your precomputed arrays
        r = self.per_juris_cell_row[cell]
        c = self.per_juris_cell_col[cell]

        # Scatter-add 1 unit to each (loc_j, r, c)
        np.add.at(units_per_cell, (loc_j, r, c), 1)

        return units_per_cell

    def _lightning_ignitions(self, orig_burning: np.ndarray, rng_lightning: np.random.Generator) -> np.ndarray:
        """
        Sample exogenous lightning ignitions independently per jurisdiction.

        For each jurisdiction j:
        lam_j ~ LogNormal(mu_log, sigma_log)
        K_j   ~ Poisson(lam_j)
        choose K_j target cells uniformly from that jurisdiction's R*C cells
        (with replacement; multiple hits to the same cell are fine)
        Returns:
        lightning mask (J,R,C), restricted to cells not burning at start of step.
        """
        J, R, C = orig_burning.shape
        lightning = np.zeros_like(orig_burning, dtype=bool)

        # Per-juris lambda and counts
        lam = rng_lightning.lognormal(mean=self.lightning_mu_log, sigma=self.lightning_sigma_log, size=J)
        K = rng_lightning.poisson(lam).astype(int)  # shape (J,)

        RC = R * C
        for j in range(J):
            kj = int(K[j])
            if kj <= 0:
                continue

            targets = rng_lightning.integers(0, RC, size=kj, endpoint=False)
            r = targets // C
            c = targets % C
            lightning[j, r, c] = True

        return lightning & (~orig_burning)


    def spread_probabilities(self, fire_state: np.ndarray) -> np.ndarray:
        b = fire_state.astype(np.int8, copy=False)  # (J,R,C)
        nb = np.zeros_like(b, dtype=np.int16)

        nb[:, 1:,  :] += b[:, :-1, :]   # up
        nb[:, :-1, :] += b[:, 1:,  :]   # down
        nb[:, :, 1:]  += b[:, :, :-1]   # left
        nb[:, :, :-1] += b[:, :, 1:]    # right

        return 1.0 - (1.0 - self.base_spread_prob) ** nb

    def virtual_step(
        self,
        actions,
        rng_spread: np.random.Generator,
        rng_lightning: np.random.Generator,
        burning_map: np.ndarray | None = None,
        unit_positions: np.ndarray | None = None,
    ):
        """
        Stateless step. Does not mutate the environment.

        actions: shape (N,3), each entry (tag,a,b):
          tag=0 -> move by (dx,dy)=(a,b)
          tag=1 -> jump to jurisdiction target=a (b ignored)
          tag=2 -> stay
        """
        if burning_map is None:
            burning_map = self.burning_map
        if unit_positions is None:
            unit_positions = self.unit_positions

        # -------------------------------
        # A) Movement of units
        # -------------------------------
        new_unit_positions = np.copy(unit_positions)

        for idx in range(self.num_units_total):
            cur_j = int(unit_positions[idx, 0])
            cur_cell = int(unit_positions[idx, 1])

            # In transit
            if cur_cell < 0:
                new_unit_positions[idx, 1] = cur_cell + 1
                new_unit_positions[idx, 0] = cur_j
                if new_unit_positions[idx, 1] == 0:
                    new_unit_positions[idx, 1] = self.center_cell_index
                continue

            tag, a, b = actions[idx]

            if tag == 2:
                continue

            if tag == 1:
                if cur_cell != self.center_cell_index:
                    raise ValueError("Units can only jump jurisdictions from the center cell.")

                target_juris = int(a)
                if not (0 <= target_juris < self.num_juris):
                    raise ValueError("Invalid target_juris.")

                if target_juris == cur_j:
                    continue

                travel_time = self.adj_matrix[cur_j][target_juris]
                if travel_time < 0:
                    raise ValueError("Units can only jump to adjacent jurisdictions.")

                new_unit_positions[idx, 0] = target_juris
                if travel_time == 0:
                    new_unit_positions[idx, 1] = self.center_cell_index
                else:
                    new_unit_positions[idx, 1] = -travel_time
                continue

            if tag == 0:
                dx = int(a)
                dy = int(b)

                if abs(dx) + abs(dy) > self.movement_per_step:
                    raise ValueError("Movement exceeds allowed movement per step.")

                cur_r = self.per_juris_cell_row[cur_cell]
                cur_c = self.per_juris_cell_col[cur_cell]

                new_r = max(0, min(self.per_juris_rows - 1, cur_r + dy))
                new_c = max(0, min(self.per_juris_cols - 1, cur_c + dx))

                new_unit_positions[idx, 1] = new_r * self.per_juris_cols + new_c
                new_unit_positions[idx, 0] = cur_j
                continue

            raise ValueError(f"Invalid action tag {tag}.")

        # -------------------------------
        # B) Fire dynamics
        # -------------------------------
        orig_burning = burning_map.copy()  # snapshot

        units_per_cell = self._units_per_cell(unit_positions=new_unit_positions)

        burning_mask = orig_burning
        num_units_burning = units_per_cell[burning_mask]

        extinguish_prob = 1.0 - (1.0 - self.suppression_success_prob) ** num_units_burning
        extinguish = rng_spread.random(extinguish_prob.shape) < extinguish_prob

        still_burning = orig_burning.copy()
        still_burning[burning_mask] = ~extinguish

        spread_prob = self.spread_probabilities(orig_burning)
        spread_draw = rng_spread.random(orig_burning.shape)

        new_fires = (spread_draw < spread_prob) & (~orig_burning)
        lightning_new = self._lightning_ignitions(orig_burning, rng_lightning)

        next_burning = still_burning | new_fires | lightning_new

        # -------------------------------
        # C) Rewards and counts
        # -------------------------------
        persisting = orig_burning & still_burning
        persisting_counts = np.sum(persisting, axis=(1, 2))
        rewards = (-persisting_counts.astype(float)).tolist()

        counts_burning = np.sum(next_burning, axis=(1, 2)).astype(int).tolist()
        return next_burning, new_unit_positions, rewards, counts_burning

    def step(self, actions, rng_spread: np.random.Generator, rng_lightning: np.random.Generator):
        """
        Stateful step. Updates the environment using the virtual step output.
        """
        burning_map, unit_positions, rewards, counts_burning = self.virtual_step(
            actions=actions,
            rng_spread=rng_spread,
            rng_lightning=rng_lightning,
            burning_map=self.burning_map,
            unit_positions=self.unit_positions,
        )

        self.burning_map = burning_map
        self.unit_positions = unit_positions

        return burning_map, unit_positions, rewards, counts_burning
