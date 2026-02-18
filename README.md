# Wildfire MARL Simulation

A multi-agent reinforcement learning environment for studying wildfire suppression and inter-jurisdiction resource sharing. The simulation models fire spread across grid-based jurisdictions where autonomous agents (suppression units) must decide how to move and fight fires, and a higher-level sharing policy decides when to transfer units between jurisdictions.

## How It Works

### The Fire Grid

Each jurisdiction is a 2D grid of cells. At each timestep, three things happen to the fire:

1. **Suppression** -- Units standing on burning cells have a chance to extinguish them. More units on a cell means a higher chance of putting the fire out.
2. **Spread** -- Fire spreads to neighboring cells (up/down/left/right). The more burning neighbors a cell has, the more likely it catches fire.
3. **Lightning** -- Random new fires ignite via a stochastic process (log-normal rate, Poisson count), simulating exogenous ignitions.

### Suppression Units

Units live on the grid and move each step by a `(dx, dy)` offset, clamped to grid bounds and limited by `movement_per_step` (Manhattan distance). The **suppression algorithm** decides where each unit moves. The current implementation is a greedy heuristic: each unit targets the nearest burning cell, claims it so other units pick different targets, and moves toward it. Idle units drift back toward the grid center.

### Fuel System (Optional)

When `max_fuel` is set, each unit starts with a full fuel tank and consumes 1 fuel per step. Units at the base (center cell) regain `fuel_refuel_rate` fuel per step (capped at `max_fuel`). Units that reach 0 fuel are immobilized and cannot move until refueled. Fuel-aware suppression algorithms automatically route low-fuel units back to base before they run out. When `max_fuel` is `None` (the default), fuel is disabled and the simulation behaves as before. Fuel is frozen during inter-jurisdiction transit (not consumed or refueled).

### Resource Sharing (Multi-Jurisdiction Only)

When multiple jurisdictions are composed together, a **sharing algorithm** can transfer units between them. Transfers work in three phases:

1. **Select** -- The algorithm picks a source jurisdiction (least fire) and a destination (most fire), then selects the unit closest to the center in the source.
2. **Steer** -- The algorithm overrides that unit's movement to walk it toward the center cell (the transfer departure point).
3. **Hop** -- Once at center, the unit enters transit to an adjacent jurisdiction. Multi-hop routes repeat this for non-adjacent destinations.

Units in transit are removed from their source jurisdiction and cannot suppress fires. After `juris_travel_time` steps they arrive at the destination's center cell.

### Two-Layer Architecture

The system is split into independent layers so each can be studied separately:

- **`JurisdictionEnv`** is the building block. It handles one fire grid with local units. It knows nothing about other jurisdictions or transfers. You can instantiate and step it alone for pure suppression research.
- **`MultiJurisdictionEnv`** composes multiple `JurisdictionEnv` instances and manages the transit system. It provides `initiate_transfer()` to move units between jurisdictions and `advance_transit()` / `step()` to tick the simulation forward.

Suppression algorithms operate on a single `JurisdictionEnv`. Sharing algorithms operate on a `MultiJurisdictionEnv`. The orchestration loop in `main.py` connects them.

## Project Structure

```
wildfire-marl-simulation/
├── environment/
│   ├── __init__.py                     # exports JurisdictionEnv, MultiJurisdictionEnv
│   ├── jurisdiction_env.py             # single-jurisdiction fire grid + units
│   └── multi_jurisdiction_env.py       # composes jurisdictions + transit system
├── algorithms/
│   ├── __init__.py                     # re-exports registries
│   ├── utils.py                        # shared helpers (manhattan_distance, step_toward)
│   ├── suppression_algorithms/
│   │   ├── __init__.py                 # SUPPRESSION_ALGORITHM_REGISTRY
│   │   ├── algorithm_base.py           # SuppressionAlgorithm ABC
│   │   └── greedy.py                   # greedy nearest-fire heuristic
│   └── sharing_algorithms/
│       ├── __init__.py                 # SHARING_ALGORITHM_REGISTRY
│       ├── algorithm_base.py           # SharingAlgorithm ABC
│       ├── none.py                     # no-op (no transfers)
│       └── periodic_transfer.py        # periodic best-to-worst transfer
├── main.py                             # CLI entry point (single / multi modes)
└── fire_animator.py                    # renders snapshot .npz files to GIF/MP4
```

## Setup

Requires Python 3.10+. Create a conda environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate sim-wildfire-marl
```

Or install dependencies manually:

```bash
pip install numpy matplotlib pillow
```

## Running the Simulation

### Single-Jurisdiction Mode (Suppression Only)

```bash
python main.py --mode single --suppression-algorithm greedy --verbose --steps 200
```

Optional flags for single mode:
```bash
python main.py --mode single --rows 16 --cols 16 --num-units 8 --save-snapshots --output-dir results
```

### Multi-Jurisdiction Mode (Suppression + Sharing)

```bash
# No sharing (baseline):
python main.py --mode multi --sharing-algorithm none --suppression-algorithm greedy --verbose --steps 200

# Periodic transfer:
python main.py --mode multi --sharing-algorithm periodic_transfer --suppression-algorithm greedy --period-s 10 --verbose --steps 200

# Custom grid layout:
python main.py --mode multi --num-juris-rows 3 --num-juris-cols 3 --per-juris-rows 20 --per-juris-cols 20 --save-snapshots
```

### Fuel-Constrained Mode

```bash
# Single jurisdiction with fuel
python main.py --mode single --suppression-algorithm greedy --max-fuel 30 --fuel-refuel-rate 2 --verbose --steps 100

# Multi jurisdiction with fuel and sharing
python main.py --mode multi --sharing-algorithm periodic_transfer --suppression-algorithm greedy --max-fuel 30 --fuel-refuel-rate 2 --period-s 10 --verbose --steps 100
```

### Generating Animations

```bash
python main.py --mode multi --sharing-algorithm periodic_transfer --save-snapshots --output-dir snapshots
python fire_animator.py --snapshots-dir snapshots --output-dir animations --fps 4
```

### Using JurisdictionEnv Standalone

```python
from environment import JurisdictionEnv
import numpy as np

jenv = JurisdictionEnv(
    rows=16, cols=16, base_spread_prob=0.06,
    suppression_success_prob=0.8, movement_per_step=4,
    lightning_mu_log=-2.0, lightning_sigma_log=2.0, num_units=8,
)
rng_s = np.random.default_rng(0)
rng_l = np.random.default_rng(1)

actions = np.zeros((jenv.num_units, 2), dtype=int)  # all stay
burning, positions, reward, count = jenv.step(actions, rng_s, rng_l)
```

## Reference for AI Agents

This section provides the technical context needed to extend this codebase.

### Architecture invariants

- Fire does not spread across jurisdiction boundaries. Jurisdictions are coupled only through unit transfers.
- `JurisdictionEnv` has no knowledge of multi-jurisdiction concepts. It must remain importable and usable without `MultiJurisdictionEnv`.
- `SuppressionAlgorithm.actions(jenv, rng)` receives a single `JurisdictionEnv`, returns `(num_units, 2)` int array of `(dx, dy)`. No masks, no tags, no global indices.
- `SharingAlgorithm` has two methods: `decide_transfers(multi_env, rng) -> list[(unit_id, target_juris)]` and `get_steering_actions(multi_env, rng) -> dict[unit_id, (dx, dy)]`. Steering overrides are applied after suppression actions are computed.
- The main loop order is: `decide_transfers` -> `initiate_transfer` -> `advance_transit` -> `get_steering_actions` -> `get_actions` (per jurisdiction) -> apply steering overrides -> `step`. This order matters because `advance_transit` delivers arrived units before actions are computed, preventing shape mismatches.

### Key data formats

- `JurisdictionEnv.burning_map`: `(rows, cols)` bool, 2D.
- `JurisdictionEnv.unit_positions`: 1D int array of flat cell indices (length = current num_units, variable due to add/remove).
- `MultiJurisdictionEnv.unit_jurisdiction`: `(num_units_total,)` int, global ID -> jurisdiction index (-1 if transit).
- `MultiJurisdictionEnv.unit_local_index`: `(num_units_total,)` int, global ID -> index within `jenv.unit_positions` (-1 if transit).
- Snapshot format for animator: `burning_map` is `(steps+1, J, R, C)` bool, `unit_positions` is `(steps+1, N, 2)` int where col 0 = jurisdiction, col 1 = flat cell index (negative = in transit with remaining steps encoded as `-remaining`). Single mode uses J=1.

### Adding a new suppression algorithm

1. Create `algorithms/suppression_algorithms/my_algo.py`.
2. Subclass `SuppressionAlgorithm`, implement `actions(self, jenv, rng) -> np.ndarray` returning `(jenv.num_units, 2)`.
3. Set `name = "my_algo"` class attribute.
4. Register in `algorithms/suppression_algorithms/__init__.py` by importing and adding to `SUPPRESSION_ALGORITHM_REGISTRY`.
5. The algorithm receives a `JurisdictionEnv` with these useful attributes: `burning_map`, `unit_positions`, `cell_row`, `cell_col`, `center_cell_row`, `center_cell_col`, `rows`, `cols`, `movement_per_step`, `num_units`, `unit_fuel` (None when fuel disabled), `max_fuel`. Use `jenv.units_per_cell()` and `jenv.spread_probabilities(fire_state)` for planning. Call `must_return_to_base(jenv)` from `algorithms.utils` to get a bool mask of units that should return to base for refueling.

### Adding a new sharing algorithm

1. Create `algorithms/sharing_algorithms/my_algo.py`.
2. Subclass `SharingAlgorithm`, implement `decide_transfers(self, multi_env, rng)` and optionally override `get_steering_actions(self, multi_env, rng)`.
3. `decide_transfers` returns `[(unit_id, target_juris), ...]`. Only return transfers for units at their jurisdiction's center cell. `initiate_transfer` will validate this.
4. `get_steering_actions` returns `{unit_id: (dx, dy)}` to override suppression actions for specific units (e.g., to walk them toward center before transfer).
5. Register in `algorithms/sharing_algorithms/__init__.py`.
6. Useful `multi_env` attributes: `jurisdictions` (list of `JurisdictionEnv`), `unit_jurisdiction`, `unit_local_index`, `burning_counts`, `juris_row`, `juris_col`, `adj_matrix`, `num_juris_rows`, `num_juris_cols`, `transit_units`.

### `virtual_step` for planning

`JurisdictionEnv.virtual_step(actions, rng_spread, rng_lightning, burning_map=, unit_positions=)` is stateless -- it returns `(next_burning, new_positions, reward, count)` without mutating the environment. Use this for lookahead / tree search in RL algorithms. Note: it still consumes RNG state, so fork the RNG if you need repeatable rollouts.

### Transit mechanics

- `initiate_transfer(unit_id, target_juris)` removes the unit from its jurisdiction (`jenv.remove_units`), shifts `unit_local_index` for remaining units in that jurisdiction, sets `unit_jurisdiction[uid] = -1`, and appends a `TransitUnit(unit_id, from_juris, to_juris, remaining_steps)`.
- `advance_transit()` decrements `remaining_steps` for all transit units. Those reaching 0 are delivered: `jenv.add_units([center_cell])` is called on the destination, and global tracking arrays are updated.
- `get_snapshot()` encodes transit units as `(to_juris, -remaining_steps)` in the unit_positions array, matching the old format the animator expects.

### Environment parameters

| Parameter | Default | Description |
|---|---|---|
| `rows` / `cols` | 16 | Grid dimensions per jurisdiction |
| `base_spread_prob` | 0.06 | Per-neighbor fire spread probability |
| `suppression_success_prob` | 0.8 | Per-unit chance of extinguishing a fire cell |
| `movement_per_step` | 4 | Max Manhattan distance a unit can move per step |
| `lightning_mu_log` | -2.0 | Log-normal mean for lightning rate |
| `lightning_sigma_log` | 2.0 | Log-normal std for lightning rate |
| `juris_travel_time` | 4 | Steps to transit between adjacent jurisdictions |
| `num_juris_rows` / `num_juris_cols` | 2 | Grid layout of jurisdictions (multi mode) |
| `max_fuel` | None | Max fuel per unit (None = unlimited / disabled) |
| `fuel_refuel_rate` | 1 | Fuel gained per step when at base (center cell) |
