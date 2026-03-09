# Wildfire MARL Simulation

A multi-agent reinforcement learning environment for studying wildfire suppression and inter-jurisdiction resource sharing. The simulation models fire spread across grid-based jurisdictions where autonomous agents (suppression units) must decide how to move and fight fires, and a higher-level sharing policy decides when to transfer units between jurisdictions.

## How It Works

### The Fire Grid

Each jurisdiction is a 2D grid of cells. At each timestep, three things happen to the fire:

1. **Suppression** -- Units standing on burning cells have a chance to extinguish them. More units on a cell means a higher chance of putting the fire out.
2. **Spread** -- Fire spreads to neighboring cells (up/down/left/right). The more burning neighbors a cell has, the more likely it catches fire.
3. **Lightning** -- Random new fires ignite via a stochastic process (log-normal rate, Poisson count), simulating exogenous ignitions.

### Suppression Units

Units live on the grid and move each step by a `(dx, dy)` offset, clamped to grid bounds and limited by `movement_per_step` (Manhattan distance). The **suppression algorithm** decides where each unit moves. Three algorithms are available:

- **`greedy`** — Each unit targets the nearest burning cell, claims it so other units pick different targets, and moves toward it. Idle units drift toward center.
- **`lp_suppression`** — Optimal unit-to-fire assignment via linear programming (minimizes total Manhattan distance).
- **`rl`** — Learned policy via PPO (see [RL Architecture](#rl-architecture)). A CNN+MLP actor-critic observes the full grid state, per-drone features (including drone index), and global scalars, then selects from each drone's K=10 nearest fires using sequential spatial masking to prevent drones from targeting the same cell. Requires a trained model (see [RL Training](#rl-training)).

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
│   │   ├── greedy.py                   # greedy nearest-fire heuristic
│   │   ├── lp_suppression.py           # LP-based optimal assignment
│   │   └── rl_suppression.py           # RL inference wrapper (loads trained PPO model)
│   ├── rl/                             # RL training module
│   │   ├── observation.py              # state space: grid channels, unit/global features, K-nearest
│   │   ├── action_translation.py       # action space: K-nearest fire selection → (dx,dy)
│   │   ├── reward.py                   # reward: -burning/total + 0.5*extinguished/total
│   │   ├── network.py                  # WildfireActorCritic (CNN + MLP, per-drone policy heads)
│   │   ├── gym_wrapper.py              # Gymnasium env wrapping JurisdictionEnv
│   │   └── train.py                    # PPO training script (python -m algorithms.rl.train)
│   └── sharing_algorithms/
│       ├── __init__.py                 # SHARING_ALGORITHM_REGISTRY
│       ├── algorithm_base.py           # SharingAlgorithm ABC
│       ├── none.py                     # no-op (no transfers)
│       └── periodic_transfer.py        # periodic best-to-worst transfer
├── main.py                             # CLI entry point (single / multi modes)
├── compare.py                          # algorithm comparison runner + CLI
├── compare_plots.py                    # matplotlib plots for comparison results
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
pip install -r requirements.txt
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

### RL Training

Train a PPO suppression agent, then evaluate it alongside baselines:

```bash
# Train (saves to trained_models/rl_v3/)
python -m algorithms.rl.train --total-timesteps 500000 --output-dir trained_models/rl_v3

# Resume from latest checkpoint (if training was interrupted)
python -m algorithms.rl.train --total-timesteps 500000 --output-dir trained_models/rl_v3 --resume

# Evaluate trained model in single mode
python main.py --mode single --suppression-algorithm rl --suppression-param-dir trained_models/rl_v3 --verbose --steps 200 --save-snapshots --output-dir results

# Generate animation from saved snapshots
python fire_animator.py --snapshots-dir results --output-dir animations --fps 2.0

# Compare against baselines
python compare.py --suppression greedy lp_suppression rl --sharing none --suppression-param-dir trained_models/rl_v3 --num-seeds 5 --steps 200 --label rl_v3_comparison --verbose
```

Key training flags: `--lr` (default 3e-4), `--rollout-steps` (default 2048), `--num-epochs` (default 4), `--gamma` (default 0.99), `--episode-steps` (default 200), `--seed`, `--max-fuel`. The training script saves `best_model.pt`, `final_model.pt`, periodic checkpoints (with optimizer state for resumability), and `params.json` for inference.

### Comparing Algorithms

`compare.py` runs the cartesian product of suppression x sharing algorithms across multiple random seeds, collects per-step metrics, computes summary statistics, and generates comparison plots.

```bash
# Default: all algorithms, 5 seeds, 200 steps
python compare.py

# Specific algorithms and more seeds
python compare.py --suppression greedy lp_suppression --sharing none periodic_transfer --seeds 0 1 2 3 4 5 6 7 8 9

# Include RL (requires --suppression-param-dir pointing to trained model)
python compare.py --suppression greedy lp_suppression rl --sharing none --suppression-param-dir trained_models/rl_v3 --num-seeds 5

# Quick test (fewer steps and seeds)
python compare.py --steps 50 --num-seeds 3

# Custom environment layout
python compare.py --num-juris-rows 3 --num-juris-cols 3 --label "3x3_experiment"

# Skip plot generation
python compare.py --no-plots
```

Results are saved to `comparisons/<timestamp>_<label>/` with the following structure:

```
comparisons/20250101_120000_comparison/
├── config.json          # environment and algorithm parameters used
├── summary.json         # aggregate statistics (mean/std across seeds)
├── raw_metrics.json     # full per-step data for every seed
└── plots/
    ├── timeseries_total_burning.png
    ├── timeseries_burning_gini.png
    ├── timeseries_units_in_transit.png
    ├── timeseries_burning_per_juris.png
    ├── summary_final_total_burning.png
    ├── summary_cumulative_fire_steps.png
    ├── summary_mean_burning_gini.png
    └── summary_peak_total_burning.png
```

**Metrics collected per step:** total burning cells, burning per jurisdiction, units per jurisdiction, units in transit, rewards per jurisdiction.

**Derived time-series:** cumulative burning (area under curve), Gini coefficient of per-jurisdiction burning (inequality), coefficient of variation of per-jurisdiction burning (disparity).

**Summary statistics (per seed, then aggregated as mean +/- std):** final/peak/mean total burning, cumulative fire-steps, final/peak burning per jurisdiction, mean/peak Gini, mean CV.

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

### RL Architecture

The RL module (`algorithms/rl/`) implements a centralized PPO agent for single-jurisdiction suppression. The architecture has gone through three iterations to solve a drone clustering problem.

#### State Space (`observation.py`)

Dict observation with 4 components:
- `grid` — `(4, 16, 16)` float32: burning map, spread probabilities, units-per-cell, recently extinguished.
- `units` — `(8, 5)` float32: normalized row, col, fuel, must_return, **drone index** (`i/(N-1)`, range [0,1]).
- `global_features` — `(3,)` float32: burning fraction, time progress, delta burning.
- `k_nearest` — `(8, 10, 2)` int: row/col of K=10 nearest fires per drone (padded with center cell).

#### Action Space (`action_translation.py`)

`MultiDiscrete([11]*8)` — each drone selects one of K=10 nearest fires or idle (index 10). Translated to `(dx, dy)` via `step_toward()`. Drones with `must_return_to_base=True` are overridden to return to center.

#### Reward (`reward.py`)

`-burning_after/total_cells + 0.5 * extinguished/total_cells`. Range ~[-1, 0.5]. Dense, aligned with minimizing cumulative fire. Includes an overlap penalty for drones stacking on the same cell.

#### Network (`network.py`)

`WildfireActorCritic` processes observations through four parallel encoders, combines them, then produces per-drone policy logits and a state value:

```
Grid (4,16,16) ──► GridEncoder (Conv3x3→32→Conv3x3→64→AvgPool→FC128) ──► (128,)
                                                                             │
Units (8,5) ──► UnitEncoder (shared MLP: 5→32→32) ──► per_unit (8,32)      │
                       │                                    │                │
                       ├── mean-pool ──► unit_summary (32,) ─┤               │
                       │                                     │               │
Global (3,) ──► GlobalEncoder (MLP: 3→16) ──► (16,) ────────┘               │
                                                             │               │
                                              cat [128, 32, 16] = (176,)    │
                                                             │               │
                                              SharedMLP (176→128→128) ──► shared (128,)
                                                             │
                    ┌────────────────────────────────────────┤
                    │                                        │
             ┌──────┴──────┐                          ValueHead (128→64→1)
             │  Per-drone  │
             │  Policy     │
             └──────┬──────┘
                    │
    For each drone i:
      cat [shared(128), unit_embed_i(32), kn_embed_i(32)] = (192,)
         │
      PolicyHead (192→64→11) ──► logits_i
```

The K-Nearest Encoder converts absolute fire coords to relative coords `(fire - drone) / grid_size`, flattens K*2=20 floats, and processes through MLP (20→32→32).

#### Sequential Spatial Masking

The key architectural innovation that prevents drone clustering. `forward()` produces raw `(B, N, 11)` logits in one pass. Then `get_action_and_value()` samples actions sequentially:

1. For each drone `i` in order `0..N-1`:
   - Clone drone `i`'s logits
   - For each fire action `a` in `0..K-1`: resolve the spatial cell from `k_nearest[i, a]`. If that cell is already claimed by a previous drone, set the logit to `-inf`
   - Idle (index K=10) is **never masked** — always available as fallback
   - Sample from the masked distribution (training) or argmax (inference via `get_greedy_action_masked()`)
   - Claim the chosen cell
2. Claimed cells tracked via `(B, rows*cols)` bool tensor — negligible overhead

This is **cell-based masking**, not action-index-based: drone 1's action 3 and drone 2's action 5 may target the same physical cell, and the masking catches this. The masking is PPO-consistent — during the eval pass, the same sequential order with stored actions reconstructs correct log probabilities.

#### Design History

| Version | Key Changes | Outcome |
|---------|------------|---------|
| v1 | Base PPO, no k_nearest encoder | All 8 drones cluster — identical observations from same starting position produce identical actions |
| v2 | + k_nearest encoder, + overlap penalty in reward | Still 100% identical actions — reward signal alone can't break the symmetry when observations are identical |
| v3 | + drone index feature (5th unit feat), + sequential spatial masking | Drone index breaks input symmetry so the model *can* differentiate drones; masking guarantees they *must* pick different targets |

The core insight: all drones start co-located at center with identical observations → identical logits → identical actions. This is a chicken-and-egg problem that soft incentives (reward penalties) cannot solve. The fix requires both (1) giving the model a way to distinguish drones (drone index) and (2) hard-constraining outputs so collisions are architecturally impossible (spatial masking).

#### Inference (`rl_suppression.py`)

`RLSuppressionAlgorithm` subclasses `SuppressionAlgorithm`, loads `param_dir/best_model.pt`, and uses `get_greedy_action_masked()` for deterministic action selection with spatial masking. Tracks `prev_burning` and `timestep` across calls. Reads `unit_features` from `params.json` for backward compatibility. Registered as `"rl"` in `SUPPRESSION_ALGORITHM_REGISTRY`.

#### Gym Wrapper (`gym_wrapper.py`)

`WildfireEnv` wraps `JurisdictionEnv` as a Gymnasium env. Handles reset/seed/episode tracking. Builds observations, translates actions, computes reward.

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
