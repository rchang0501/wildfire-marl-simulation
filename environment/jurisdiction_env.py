import numpy as np


class JurisdictionEnv:
    """Single-jurisdiction fire environment building block.

    Manages a single fire grid and a variable number of local suppression units.
    No concept of inter-jurisdiction transfers -- those are handled by
    MultiJurisdictionEnv.
    """

    def __init__(
        self,
        *,
        rows: int,
        cols: int,
        base_spread_prob: float,
        suppression_success_prob: float,
        movement_per_step: int,
        lightning_mu_log: float,
        lightning_sigma_log: float,
        num_units: int = 0,
    ):
        self.rows = int(rows)
        self.cols = int(cols)

        self.base_spread_prob = float(base_spread_prob)
        self.suppression_success_prob = float(suppression_success_prob)
        self.movement_per_step = int(movement_per_step)
        self.lightning_mu_log = float(lightning_mu_log)
        self.lightning_sigma_log = float(lightning_sigma_log)

        # Precomputed cell coordinate arrays
        num_cells = self.rows * self.cols
        self.cell_indices = np.arange(num_cells, dtype=int)
        self.cell_row = self.cell_indices // self.cols
        self.cell_col = self.cell_indices % self.cols

        self.center_cell_row = self.rows // 2
        self.center_cell_col = self.cols // 2
        self.center_cell = self.center_cell_row * self.cols + self.center_cell_col

        # State
        self.burning_map = np.zeros((self.rows, self.cols), dtype=bool)
        # unit_positions: 1-D array of flat cell indices, variable length
        self.unit_positions = np.full(num_units, self.center_cell, dtype=int)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_units(self) -> int:
        return len(self.unit_positions)

    @property
    def burning_count(self) -> int:
        return int(np.sum(self.burning_map))

    # ------------------------------------------------------------------
    # Unit management (called by MultiJurisdictionEnv)
    # ------------------------------------------------------------------

    def add_units(self, cell_indices: np.ndarray | list[int]) -> None:
        """Add units at given cell positions."""
        new = np.asarray(cell_indices, dtype=int)
        self.unit_positions = np.concatenate([self.unit_positions, new])

    def remove_units(self, local_indices: np.ndarray | list[int]) -> None:
        """Remove units by their local index."""
        self.unit_positions = np.delete(self.unit_positions, local_indices)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def units_per_cell(self) -> np.ndarray:
        """Return (rows, cols) count of units at each cell."""
        counts = np.zeros((self.rows, self.cols), dtype=np.int16)
        if self.num_units == 0:
            return counts
        r = self.cell_row[self.unit_positions]
        c = self.cell_col[self.unit_positions]
        np.add.at(counts, (r, c), 1)
        return counts

    def spread_probabilities(self, fire_state: np.ndarray) -> np.ndarray:
        """Per-cell probability of catching fire from neighbors."""
        b = fire_state.astype(np.int8, copy=False)  # (R, C)
        nb = np.zeros_like(b, dtype=np.int16)
        nb[1:, :] += b[:-1, :]   # up
        nb[:-1, :] += b[1:, :]   # down
        nb[:, 1:] += b[:, :-1]   # left
        nb[:, :-1] += b[:, 1:]   # right
        return 1.0 - (1.0 - self.base_spread_prob) ** nb

    # ------------------------------------------------------------------
    # Lightning
    # ------------------------------------------------------------------

    def _lightning_ignitions(
        self, orig_burning: np.ndarray, rng_lightning: np.random.Generator
    ) -> np.ndarray:
        R, C = orig_burning.shape
        lightning = np.zeros_like(orig_burning, dtype=bool)

        lam = rng_lightning.lognormal(
            mean=self.lightning_mu_log, sigma=self.lightning_sigma_log
        )
        k = int(rng_lightning.poisson(lam))
        if k <= 0:
            return lightning

        RC = R * C
        targets = rng_lightning.integers(0, RC, size=k, endpoint=False)
        r = targets // C
        c = targets % C
        lightning[r, c] = True
        return lightning & (~orig_burning)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def virtual_step(
        self,
        actions: np.ndarray,
        rng_spread: np.random.Generator,
        rng_lightning: np.random.Generator,
        burning_map: np.ndarray | None = None,
        unit_positions: np.ndarray | None = None,
    ):
        """Stateless step. Returns (next_burning, new_unit_positions, reward, burning_count)."""
        if burning_map is None:
            burning_map = self.burning_map
        if unit_positions is None:
            unit_positions = self.unit_positions

        n = len(unit_positions)

        # A) Movement
        new_unit_positions = unit_positions.copy()
        for i in range(n):
            cur_cell = int(unit_positions[i])
            dx, dy = int(actions[i, 0]), int(actions[i, 1])

            if dx == 0 and dy == 0:
                continue

            if abs(dx) + abs(dy) > self.movement_per_step:
                raise ValueError("Movement exceeds allowed movement per step.")

            cur_r = self.cell_row[cur_cell]
            cur_c = self.cell_col[cur_cell]
            new_r = max(0, min(self.rows - 1, cur_r + dy))
            new_c = max(0, min(self.cols - 1, cur_c + dx))
            new_unit_positions[i] = new_r * self.cols + new_c

        # B) Fire dynamics
        orig_burning = burning_map.copy()

        # Suppression
        upc = np.zeros((self.rows, self.cols), dtype=np.int16)
        if n > 0:
            r = self.cell_row[new_unit_positions]
            c = self.cell_col[new_unit_positions]
            np.add.at(upc, (r, c), 1)

        num_units_burning = upc[orig_burning]
        extinguish_prob = 1.0 - (1.0 - self.suppression_success_prob) ** num_units_burning
        extinguish = rng_spread.random(extinguish_prob.shape) < extinguish_prob

        still_burning = orig_burning.copy()
        still_burning[orig_burning] = ~extinguish

        # Spread
        spread_prob = self.spread_probabilities(orig_burning)
        spread_draw = rng_spread.random(orig_burning.shape)
        new_fires = (spread_draw < spread_prob) & (~orig_burning)

        # Lightning
        lightning_new = self._lightning_ignitions(orig_burning, rng_lightning)

        next_burning = still_burning | new_fires | lightning_new

        # Reward
        persisting = orig_burning & still_burning
        reward = -float(np.sum(persisting))
        count = int(np.sum(next_burning))

        return next_burning, new_unit_positions, reward, count

    def step(
        self,
        actions: np.ndarray,
        rng_spread: np.random.Generator,
        rng_lightning: np.random.Generator,
    ):
        """Stateful step. Mutates internal state."""
        burning_map, unit_positions, reward, count = self.virtual_step(
            actions=actions,
            rng_spread=rng_spread,
            rng_lightning=rng_lightning,
        )
        self.burning_map = burning_map
        self.unit_positions = unit_positions
        return burning_map, unit_positions, reward, count
