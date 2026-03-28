"""Observation construction for the MARL suppression agent.

Builds a structured observation from a JurisdictionEnv consisting of:
- Grid channels: (4, rows, cols) — burning, spread probs, unit density, recently extinguished
- Per-unit features: (num_units, 5) — normalized position, fuel, must_return flag, drone index
- Global features: (4,) — burning fraction, time progress, delta burning, active drone ratio
- K-nearest fires: (num_units, K, 2) — target candidates for action space

Supports optional padding to max_units for training with variable drone counts.
"""

import numpy as np

from algorithms.utils import must_return_to_base
from environment.jurisdiction_env import JurisdictionEnv

K_NEAREST = 10


def build_grid_channels(
    jenv: JurisdictionEnv,
    prev_burning: np.ndarray | None = None,
) -> np.ndarray:
    """Build (4, rows, cols) grid observation channels.

    Channel 0: burning_map (binary)
    Channel 1: spread_probabilities (float [0, 1])
    Channel 2: units_per_cell (int, raw counts)
    Channel 3: recently_extinguished (binary) — cells burning last step but not now
    """
    rows, cols = jenv.rows, jenv.cols
    channels = np.zeros((4, rows, cols), dtype=np.float32)

    channels[0] = jenv.burning_map.astype(np.float32)
    channels[1] = jenv.spread_probabilities(jenv.burning_map).astype(np.float32)
    channels[2] = jenv.units_per_cell().astype(np.float32)

    if prev_burning is not None:
        channels[3] = (prev_burning & ~jenv.burning_map).astype(np.float32)

    return channels


def build_unit_features(jenv: JurisdictionEnv) -> np.ndarray:
    """Build (num_units, 5) per-unit feature matrix.

    Features: [row/rows, col/cols, fuel/max_fuel, must_return, drone_index]
    """
    n = jenv.num_units
    features = np.zeros((n, 5), dtype=np.float32)

    if n == 0:
        return features

    unit_r = jenv.cell_row[jenv.unit_positions]
    unit_c = jenv.cell_col[jenv.unit_positions]

    features[:, 0] = unit_r / max(jenv.rows - 1, 1)
    features[:, 1] = unit_c / max(jenv.cols - 1, 1)

    if jenv.unit_fuel is not None and jenv.max_fuel is not None:
        features[:, 2] = jenv.unit_fuel / jenv.max_fuel
    else:
        features[:, 2] = 1.0

    features[:, 3] = must_return_to_base(jenv).astype(np.float32)

    # Drone index normalized to [0, 1] — breaks input symmetry for co-located drones
    if n > 1:
        features[:, 4] = np.arange(n, dtype=np.float32) / (n - 1)
    else:
        features[:, 4] = 0.0

    return features


def build_global_features(
    jenv: JurisdictionEnv,
    timestep: int,
    max_steps: int,
    prev_burning_count: int,
    max_units: int | None = None,
) -> np.ndarray:
    """Build (4,) global feature vector.

    Features: [burning_fraction, time_progress, delta_burning_fraction, active_drone_ratio]
    """
    total_cells = jenv.rows * jenv.cols
    burning_count = jenv.burning_count

    features = np.zeros(4, dtype=np.float32)
    features[0] = burning_count / total_cells
    features[1] = timestep / max(max_steps, 1)
    features[2] = (burning_count - prev_burning_count) / total_cells

    # Active drone ratio: how many of max_units are active
    if max_units is not None and max_units > 0:
        features[3] = jenv.num_units / max_units
    else:
        features[3] = 1.0

    return features


def compute_k_nearest_fires(
    jenv: JurisdictionEnv,
    k: int = K_NEAREST,
) -> np.ndarray:
    """For each drone, find the K nearest burning cells.

    Returns:
        targets: (num_units, k, 2) array of (row, col) for each target slot.
            If fewer than k fires exist, remaining slots are filled with center cell.
            If zero fires, all slots point to center.
    """
    n = jenv.num_units
    center_r, center_c = jenv.center_cell_row, jenv.center_cell_col

    targets = np.full((n, k, 2), [center_r, center_c], dtype=int)

    burning_rc = np.argwhere(jenv.burning_map)  # (num_fires, 2)
    if burning_rc.shape[0] == 0 or n == 0:
        return targets

    fire_r = burning_rc[:, 0]
    fire_c = burning_rc[:, 1]

    unit_r = jenv.cell_row[jenv.unit_positions]
    unit_c = jenv.cell_col[jenv.unit_positions]

    for i in range(n):
        dists = np.abs(fire_r - unit_r[i]) + np.abs(fire_c - unit_c[i])
        num_fires = len(dists)
        num_targets = min(k, num_fires)
        nearest_idx = np.argpartition(dists, num_targets - 1)[:num_targets]
        nearest_idx = nearest_idx[np.argsort(dists[nearest_idx])]
        targets[i, :num_targets, 0] = fire_r[nearest_idx]
        targets[i, :num_targets, 1] = fire_c[nearest_idx]

    return targets


def build_active_mask(n: int, max_units: int) -> np.ndarray:
    """Build (max_units,) boolean mask: True for active drones, False for padding."""
    mask = np.zeros(max_units, dtype=bool)
    mask[:n] = True
    return mask


def build_observation(
    jenv: JurisdictionEnv,
    timestep: int,
    max_steps: int,
    prev_burning: np.ndarray | None = None,
    prev_burning_count: int = 0,
    max_units: int | None = None,
) -> dict[str, np.ndarray]:
    """Build the full observation dictionary.

    Args:
        max_units: If provided (training), pad units/k_nearest to max_units and
                   include active_mask. If None (inference), raw shapes, no padding.

    Returns dict with keys:
        grid: (4, rows, cols) float32
        units: (N, 5) or (max_units, 5) float32
        global_features: (4,) float32
        k_nearest: (N, K, 2) or (max_units, K, 2) int
        active_mask: (max_units,) bool — only present if max_units is not None
    """
    n = jenv.num_units
    grid = build_grid_channels(jenv, prev_burning)
    units = build_unit_features(jenv)
    global_features = build_global_features(
        jenv, timestep, max_steps, prev_burning_count, max_units=max_units
    )
    k_nearest = compute_k_nearest_fires(jenv)

    obs = {
        "grid": grid,
        "units": units,
        "global_features": global_features,
        "k_nearest": k_nearest,
    }

    if max_units is not None:
        # Pad units to (max_units, 5)
        padded_units = np.zeros((max_units, 5), dtype=np.float32)
        if n > 0:
            padded_units[:n] = units
        obs["units"] = padded_units

        # Pad k_nearest to (max_units, K, 2)
        center_r, center_c = jenv.center_cell_row, jenv.center_cell_col
        padded_kn = np.full((max_units, K_NEAREST, 2), [center_r, center_c], dtype=int)
        if n > 0:
            padded_kn[:n] = k_nearest
        obs["k_nearest"] = padded_kn

        obs["active_mask"] = build_active_mask(n, max_units)

    return obs
