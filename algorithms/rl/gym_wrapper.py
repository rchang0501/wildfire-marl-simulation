"""Gymnasium wrapper around JurisdictionEnv for RL training.

Wraps the wildfire simulation as a standard Gymnasium environment with:
- Dict observation space (grid, units, global_features, k_nearest)
- MultiDiscrete action space (K+1 options per drone)
- Custom reward function
- Episode management with configurable max steps
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.jurisdiction_env import JurisdictionEnv
from algorithms.rl.observation import (
    K_NEAREST,
    build_observation,
)
from algorithms.rl.action_translation import translate_actions
from algorithms.rl.reward import compute_reward


class WildfireEnv(gym.Env):
    """Gymnasium wrapper for single-jurisdiction wildfire suppression."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        num_units: int = 8,
        base_spread_prob: float = 0.06,
        suppression_success_prob: float = 0.8,
        movement_per_step: int = 4,
        lightning_mu_log: float = -2.0,
        lightning_sigma_log: float = 2.0,
        max_fuel: int | None = None,
        fuel_refuel_rate: int = 1,
        max_steps: int = 200,
        seed: int | None = None,
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.num_units = num_units
        self.max_steps = max_steps

        # Store env params for reset
        self._env_kwargs = dict(
            rows=rows,
            cols=cols,
            base_spread_prob=base_spread_prob,
            suppression_success_prob=suppression_success_prob,
            movement_per_step=movement_per_step,
            lightning_mu_log=lightning_mu_log,
            lightning_sigma_log=lightning_sigma_log,
            num_units=num_units,
            max_fuel=max_fuel,
            fuel_refuel_rate=fuel_refuel_rate,
        )

        self.jenv: JurisdictionEnv | None = None
        self._timestep = 0
        self._prev_burning: np.ndarray | None = None
        self._prev_burning_count = 0

        # Action space: each drone picks from K+1 options
        num_actions = K_NEAREST + 1
        self.action_space = spaces.MultiDiscrete(
            [num_actions] * num_units
        )

        # Observation space
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=np.inf, shape=(4, rows, cols), dtype=np.float32
            ),
            "units": spaces.Box(
                low=0.0, high=1.0, shape=(num_units, 5), dtype=np.float32
            ),
            "global_features": spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            "k_nearest": spaces.Box(
                low=0, high=max(rows, cols),
                shape=(num_units, K_NEAREST, 2), dtype=np.int64
            ),
        })

        # RNGs initialized on reset
        self._rng_spread: np.random.Generator | None = None
        self._rng_lightning: np.random.Generator | None = None
        self._base_seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)

        # Derive independent RNGs
        if seed is not None:
            self._base_seed = seed
        actual_seed = self._base_seed if self._base_seed is not None else 0
        seq = np.random.SeedSequence(actual_seed)
        child_seqs = seq.spawn(2)
        self._rng_spread = np.random.default_rng(child_seqs[0])
        self._rng_lightning = np.random.default_rng(child_seqs[1])

        self.jenv = JurisdictionEnv(**self._env_kwargs)
        self._timestep = 0
        self._prev_burning = np.zeros((self.rows, self.cols), dtype=bool)
        self._prev_burning_count = 0

        obs = build_observation(
            self.jenv,
            self._timestep,
            self.max_steps,
            self._prev_burning,
            self._prev_burning_count,
        )
        return obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        assert self.jenv is not None, "Must call reset() before step()"

        # Build k_nearest targets for action translation
        obs_pre = build_observation(
            self.jenv,
            self._timestep,
            self.max_steps,
            self._prev_burning,
            self._prev_burning_count,
        )
        k_nearest = obs_pre["k_nearest"]

        # Translate RL actions to (dx, dy)
        env_actions = translate_actions(self.jenv, action, k_nearest)

        # Record pre-step state for reward
        orig_burning = self.jenv.burning_map.copy()

        # Step the environment
        next_burning, _, _, _, _ = self.jenv.step(
            env_actions,
            rng_spread=self._rng_spread,
            rng_lightning=self._rng_lightning,
        )

        # Compute reward
        reward = compute_reward(self.jenv, orig_burning, next_burning)

        # Update tracking state
        self._prev_burning = orig_burning
        self._prev_burning_count = int(np.sum(orig_burning))
        self._timestep += 1

        # Check termination
        terminated = False
        truncated = self._timestep >= self.max_steps

        # Build next observation
        obs = build_observation(
            self.jenv,
            self._timestep,
            self.max_steps,
            self._prev_burning,
            self._prev_burning_count,
        )

        info = {
            "burning_count": self.jenv.burning_count,
            "timestep": self._timestep,
        }

        return obs, reward, terminated, truncated, info
