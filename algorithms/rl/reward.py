"""Reward function for the RL suppression agent.

Computes a dense reward based on:
- Primary: negative burning count (penalize total fire)
- Bonus: extinguishment reward (encourage active suppression)
"""

import numpy as np

from environment.jurisdiction_env import JurisdictionEnv


def compute_reward(
    jenv: JurisdictionEnv,
    orig_burning: np.ndarray,
    next_burning: np.ndarray,
) -> float:
    """Compute the shaped reward for one environment step.

    Args:
        jenv: The jurisdiction environment (for grid dimensions).
        orig_burning: (rows, cols) bool array of burning cells before the step.
        next_burning: (rows, cols) bool array of burning cells after the step.

    Returns:
        Scalar reward, approximately in [-1, 0.5].
    """
    total_cells = jenv.rows * jenv.cols
    burning_after = float(np.sum(next_burning))
    extinguished = float(np.sum(orig_burning & ~next_burning))

    # Primary: penalize total fire
    reward = -burning_after / total_cells

    # Bonus: reward successful suppression
    reward += 0.5 * extinguished / total_cells

    return reward
