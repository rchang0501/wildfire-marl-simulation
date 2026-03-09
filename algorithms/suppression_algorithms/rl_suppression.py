"""RL-based suppression algorithm.

Loads a trained PPO model and uses it to assign drones to fires.
Integrates with the existing SuppressionAlgorithm interface.

Usage in main.py / compare.py:
    --suppression-algorithm rl --suppression-param-dir trained_models/rl
"""

import os

import numpy as np
import torch

from algorithms.suppression_algorithms.algorithm_base import SuppressionAlgorithm
from algorithms.rl.observation import K_NEAREST, build_observation
from algorithms.rl.action_translation import translate_actions
from algorithms.rl.network import WildfireActorCritic
from environment.jurisdiction_env import JurisdictionEnv


class RLSuppressionAlgorithm(SuppressionAlgorithm):
    """RL suppression using a trained PPO actor-critic model."""

    name = "rl"

    def __init__(self, param_dir: str | None = None, params: dict | None = None):
        super().__init__(param_dir=param_dir, params=params)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model parameters from params.json
        num_units = self.params.get("num_units", 8)
        k = self.params.get("k", K_NEAREST)
        rows = self.params.get("rows", 16)
        cols = self.params.get("cols", 16)
        model_file = self.params.get("model_file", "best_model.pt")

        # Build model
        self._model = WildfireActorCritic(
            num_units=num_units, k=k, rows=rows, cols=cols
        ).to(self._device)

        # Load weights if param_dir provided
        if self.param_dir:
            model_path = os.path.join(self.param_dir, model_file)
            if os.path.isfile(model_path):
                state_dict = torch.load(model_path, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state_dict)
                self._model.eval()
            else:
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Train a model first with: python -m algorithms.rl.train"
                )

        # Episode tracking for observation construction
        self._timestep = 0
        self._max_steps = self.params.get("max_steps", 200)
        self._prev_burning: np.ndarray | None = None
        self._prev_burning_count = 0

    def actions(self, jenv: JurisdictionEnv, rng: np.random.Generator) -> np.ndarray:
        # Initialize tracking on first call or after reset
        if self._prev_burning is None:
            self._prev_burning = np.zeros((jenv.rows, jenv.cols), dtype=bool)
            self._prev_burning_count = 0
            self._timestep = 0

        # Build observation
        obs = build_observation(
            jenv,
            self._timestep,
            self._max_steps,
            self._prev_burning,
            self._prev_burning_count,
        )

        # Convert to tensors
        grid_t = torch.tensor(obs["grid"], dtype=torch.float32, device=self._device).unsqueeze(0)
        units_t = torch.tensor(obs["units"], dtype=torch.float32, device=self._device).unsqueeze(0)
        global_t = torch.tensor(obs["global_features"], dtype=torch.float32, device=self._device).unsqueeze(0)
        kn_t = torch.tensor(obs["k_nearest"], dtype=torch.float32, device=self._device).unsqueeze(0)

        # Get action from policy (deterministic at inference — use argmax)
        with torch.no_grad():
            logits, _ = self._model(grid_t, units_t, global_t, k_nearest=kn_t)
            agent_actions = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Translate to (dx, dy)
        env_actions = translate_actions(jenv, agent_actions, obs["k_nearest"])

        # Update tracking state
        self._prev_burning = jenv.burning_map.copy()
        self._prev_burning_count = jenv.burning_count
        self._timestep += 1

        return env_actions
