"""Gymnasium wrapper around JurisdictionEnv for MARL training.

Wraps the wildfire simulation as a standard Gymnasium environment with:
- Variable drone count per episode (sampled from [min_units, max_units])
- Dict observation space padded to max_units, with active_mask
- MultiDiscrete action space (K+1 options per drone, max_units drones)
- Custom reward function
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.jurisdiction_env import JurisdictionEnv
from algorithms.marl.observation import (
    K_NEAREST,
    build_observation,
)
from algorithms.rl.action_translation import translate_actions
from algorithms.rl.reward import compute_reward


class WildfireEnv(gym.Env):
    """Gymnasium wrapper for single-jurisdiction wildfire suppression with variable N."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rows: int = 16,
        cols: int = 16,
        max_units: int = 12,
        min_units: int = 4,
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
        self.max_units = max_units
        self.min_units = min_units
        self.max_steps = max_steps

        # Store env params for reset (num_units set per episode)
        self._env_kwargs = dict(
            rows=rows,
            cols=cols,
            base_spread_prob=base_spread_prob,
            suppression_success_prob=suppression_success_prob,
            movement_per_step=movement_per_step,
            lightning_mu_log=lightning_mu_log,
            lightning_sigma_log=lightning_sigma_log,
            max_fuel=max_fuel,
            fuel_refuel_rate=fuel_refuel_rate,
        )

        self._current_n: int = max_units
        self.jenv: JurisdictionEnv | None = None
        self._timestep = 0
        self._prev_burning: np.ndarray | None = None
        self._prev_burning_count = 0

        # Action space: each drone picks from K+1 options, max_units drones
        num_actions = K_NEAREST + 1
        self.action_space = spaces.MultiDiscrete(
            [num_actions] * max_units
        )

        # Observation space (padded to max_units)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=np.inf, shape=(4, rows, cols), dtype=np.float32
            ),
            "units": spaces.Box(
                low=0.0, high=1.0, shape=(max_units, 5), dtype=np.float32
            ),
            "global_features": spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            ),
            "k_nearest": spaces.Box(
                low=0, high=max(rows, cols),
                shape=(max_units, K_NEAREST, 2), dtype=np.int64
            ),
            "active_mask": spaces.MultiBinary(max_units),
        })

        # RNGs
        self._rng_spread: np.random.Generator | None = None
        self._rng_lightning: np.random.Generator | None = None
        self._rng_episode: np.random.Generator | None = None
        self._base_seed = seed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)

        if seed is not None:
            self._base_seed = seed
        actual_seed = self._base_seed if self._base_seed is not None else 0
        seq = np.random.SeedSequence(actual_seed)
        child_seqs = seq.spawn(3)
        self._rng_spread = np.random.default_rng(child_seqs[0])
        self._rng_lightning = np.random.default_rng(child_seqs[1])
        self._rng_episode = np.random.default_rng(child_seqs[2])

        # Sample drone count for this episode
        self._current_n = int(
            self._rng_episode.integers(self.min_units, self.max_units + 1)
        )

        # Increment seed so next reset gets different randomness
        if self._base_seed is not None:
            self._base_seed += 1

        self.jenv = JurisdictionEnv(
            **self._env_kwargs, num_units=self._current_n
        )
        self._timestep = 0
        self._prev_burning = np.zeros((self.rows, self.cols), dtype=bool)
        self._prev_burning_count = 0

        obs = build_observation(
            self.jenv,
            self._timestep,
            self.max_steps,
            self._prev_burning,
            self._prev_burning_count,
            max_units=self.max_units,
        )
        return obs, {"num_units": self._current_n}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        assert self.jenv is not None, "Must call reset() before step()"

        n = self._current_n

        # Slice action to actual drone count
        active_action = action[:n]

        # Build k_nearest targets for action translation (raw, not padded)
        from algorithms.marl.observation import compute_k_nearest_fires
        k_nearest = compute_k_nearest_fires(self.jenv)

        # Translate RL actions to (dx, dy)
        env_actions = translate_actions(self.jenv, active_action, k_nearest)

        # Record pre-step state
        orig_burning = self.jenv.burning_map.copy()

        # Step the environment
        next_burning, _, _, _, _ = self.jenv.step(
            env_actions,
            rng_spread=self._rng_spread,
            rng_lightning=self._rng_lightning,
        )

        # Compute reward
        reward = compute_reward(self.jenv, orig_burning, next_burning)

        # Update tracking
        self._prev_burning = orig_burning
        self._prev_burning_count = int(np.sum(orig_burning))
        self._timestep += 1

        terminated = False
        truncated = self._timestep >= self.max_steps

        obs = build_observation(
            self.jenv,
            self._timestep,
            self.max_steps,
            self._prev_burning,
            self._prev_burning_count,
            max_units=self.max_units,
        )

        info = {
            "burning_count": self.jenv.burning_count,
            "timestep": self._timestep,
            "num_units": self._current_n,
        }

        return obs, reward, terminated, truncated, info
