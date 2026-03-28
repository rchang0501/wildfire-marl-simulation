"""PyTorch policy/value network for the RL suppression agent.

Architecture:
- Grid Encoder (CNN): Processes (4, rows, cols) grid channels
- Unit Encoder (shared MLP): Processes per-drone features
- K-Nearest Encoder (MLP): Processes per-drone relative fire target coordinates
- Global Encoder (MLP): Processes global scalar features
- Combined representation feeds into per-drone policy heads and a shared value head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.rl.observation import K_NEAREST


class GridEncoder(nn.Module):
    """CNN encoder for the (4, rows, cols) grid observation."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, rows, cols)
        Returns:
            (batch, embed_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (batch, 64)
        return F.relu(self.fc(x))


class UnitEncoder(nn.Module):
    """Shared MLP encoder for per-drone features."""

    def __init__(self, in_features: int = 5, embed_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_units, 4)
        Returns:
            per_unit: (batch, num_units, embed_dim)
        """
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class GlobalEncoder(nn.Module):
    """MLP encoder for global scalar features."""

    def __init__(self, in_features: int = 3, embed_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3)
        Returns:
            (batch, embed_dim)
        """
        return F.relu(self.fc(x))


class KNearestEncoder(nn.Module):
    """MLP encoder for per-drone k-nearest fire target coordinates."""

    def __init__(self, k: int = K_NEAREST, embed_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(k * 2, 32)
        self.fc2 = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_units, k*2) flattened relative coordinates
        Returns:
            (batch, num_units, embed_dim)
        """
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class WildfireActorCritic(nn.Module):
    """Combined actor-critic network for wildfire suppression.

    Combines grid, unit, k-nearest, and global encoders into a shared
    representation, then produces per-drone action logits and a shared value head.
    """

    def __init__(
        self,
        num_units: int = 8,
        grid_channels: int = 4,
        unit_features: int = 5,
        global_features: int = 3,
        grid_embed_dim: int = 128,
        unit_embed_dim: int = 32,
        k_nearest_embed_dim: int = 32,
        global_embed_dim: int = 16,
        hidden_dim: int = 128,
        k: int = K_NEAREST,
        rows: int = 16,
        cols: int = 16,
    ):
        super().__init__()
        self.num_units = num_units
        self.k = k
        self.num_actions = k + 1  # K fires + idle
        self.rows = rows
        self.cols = cols

        self.grid_encoder = GridEncoder(grid_channels, grid_embed_dim)
        self.unit_encoder = UnitEncoder(unit_features, unit_embed_dim)
        self.k_nearest_encoder = KNearestEncoder(k, k_nearest_embed_dim)
        self.global_encoder = GlobalEncoder(global_features, global_embed_dim)

        combined_dim = grid_embed_dim + unit_embed_dim + global_embed_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-drone policy head: shared repr + drone embed + k_nearest embed
        policy_input_dim = hidden_dim + unit_embed_dim + k_nearest_embed_dim
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        # Value head: from shared representation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _encode_k_nearest(
        self,
        k_nearest: torch.Tensor,
        units: torch.Tensor,
    ) -> torch.Tensor:
        """Compute relative coordinates and encode k_nearest targets.

        Args:
            k_nearest: (B, N, K, 2) absolute (row, col) fire targets
            units: (B, N, 4) per-unit features (first 2 are normalized row, col)

        Returns:
            (B, N, k_nearest_embed_dim)
        """
        # Drone positions in absolute coords
        drone_row = units[:, :, 0:1] * (self.rows - 1)  # (B, N, 1)
        drone_col = units[:, :, 1:2] * (self.cols - 1)  # (B, N, 1)

        # Relative coords normalized by grid size
        rel_row = (k_nearest[:, :, :, 0] - drone_row) / self.rows  # (B, N, K)
        rel_col = (k_nearest[:, :, :, 1] - drone_col) / self.cols  # (B, N, K)

        # Flatten to (B, N, K*2)
        rel_coords = torch.cat([rel_row, rel_col], dim=-1)

        return self.k_nearest_encoder(rel_coords)

    def forward(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        k_nearest: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing action logits and state value.

        Args:
            grid: (batch, 4, rows, cols)
            units: (batch, num_units, 4)
            global_features: (batch, 3)
            k_nearest: (batch, num_units, K, 2) absolute fire target coords

        Returns:
            logits: (batch, num_units, num_actions) — raw action logits per drone
            value: (batch, 1) — state value estimate
        """
        grid_embed = self.grid_encoder(grid)                    # (B, 128)
        unit_embeds = self.unit_encoder(units)                  # (B, N, 32)
        global_embed = self.global_encoder(global_features)     # (B, 16)

        # Mean-pool unit embeddings for summary
        unit_summary = unit_embeds.mean(dim=1)                  # (B, 32)

        combined = torch.cat([grid_embed, unit_summary, global_embed], dim=1)
        shared = self.shared_mlp(combined)                      # (B, 128)

        # Value estimate
        value = self.value_head(shared)                         # (B, 1)

        # Encode k_nearest if provided, else zeros
        batch_size = grid.shape[0]
        if k_nearest is not None:
            kn_embeds = self._encode_k_nearest(k_nearest, units)  # (B, N, 32)
        else:
            kn_embeds = torch.zeros(
                batch_size, self.num_units, 32,
                device=grid.device, dtype=grid.dtype,
            )

        # Per-drone policy: concatenate shared + per-drone embed + k_nearest embed
        logits_list = []
        for i in range(self.num_units):
            drone_embed = unit_embeds[:, i, :]                  # (B, 32)
            kn_embed = kn_embeds[:, i, :]                       # (B, 32)
            policy_input = torch.cat([shared, drone_embed, kn_embed], dim=1)  # (B, 192)
            logits_i = self.policy_head(policy_input)           # (B, num_actions)
            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=1)                # (B, N, num_actions)

        return logits, value

    def _resolve_cell_index(
        self,
        k_nearest: torch.Tensor,
        drone_idx: int,
        action_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Resolve action indices to flat cell indices for a single drone.

        Args:
            k_nearest: (B, N, K, 2) absolute (row, col) fire targets
            drone_idx: which drone
            action_idx: (B,) action indices

        Returns:
            (B,) flat cell indices, or -1 for idle actions
        """
        B = action_idx.shape[0]
        idle_mask = action_idx == self.k  # idle action
        # Clamp to valid range for gather (idle will be overwritten)
        safe_idx = action_idx.clamp(0, self.k - 1)

        # k_nearest[:, drone_idx, :, :] -> (B, K, 2)
        drone_targets = k_nearest[:, drone_idx, :, :]
        # Gather the target row/col for selected action
        row = drone_targets[torch.arange(B, device=action_idx.device), safe_idx, 0].long()
        col = drone_targets[torch.arange(B, device=action_idx.device), safe_idx, 1].long()

        flat = row * self.cols + col
        flat[idle_mask] = -1
        return flat

    def get_action_and_value(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        k_nearest: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions with sequential spatial masking.

        Drones are processed in order 0..N-1. For each drone, fire targets
        that map to an already-claimed cell are masked (logit = -inf).
        Idle (index K) is never masked. This guarantees no two drones
        target the same cell.

        Args:
            grid, units, global_features: observation tensors (batch dims)
            k_nearest: (B, N, K, 2) fire target coords
            action: (B, N) int tensor. If None, sample from policy.

        Returns:
            action: (B, N) sampled or given actions
            log_prob: (B,) sum of per-drone log probs
            entropy: (B,) sum of per-drone entropies
            value: (B, 1) state value
        """
        logits, value = self.forward(grid, units, global_features, k_nearest)
        # logits: (B, N, num_actions)

        B = logits.shape[0]
        N = self.num_units
        sampling = action is None

        if sampling:
            action = torch.zeros(B, N, dtype=torch.long, device=logits.device)

        # Track claimed cells per batch element: (B, rows*cols)
        claimed = torch.zeros(B, self.rows * self.cols, dtype=torch.bool, device=logits.device)

        total_log_prob = torch.zeros(B, device=logits.device)
        total_entropy = torch.zeros(B, device=logits.device)

        for i in range(N):
            drone_logits = logits[:, i, :].clone()  # (B, num_actions)

            # Mask fire actions whose target cell is already claimed
            if k_nearest is not None:
                for a in range(self.k):
                    cell_flat = self._resolve_cell_index(k_nearest, i, torch.full((B,), a, dtype=torch.long, device=logits.device))
                    # cell_flat: (B,) — flat cell index for action a of drone i
                    # Check if claimed for each batch element
                    is_claimed = claimed[torch.arange(B, device=logits.device), cell_flat.clamp(0)] & (cell_flat >= 0)
                    drone_logits[is_claimed, a] = float("-inf")

            dist = torch.distributions.Categorical(logits=drone_logits)

            if sampling:
                action[:, i] = dist.sample()

            total_log_prob += dist.log_prob(action[:, i])
            total_entropy += dist.entropy()

            # Claim the chosen cell (if not idle)
            if k_nearest is not None:
                chosen_cell = self._resolve_cell_index(k_nearest, i, action[:, i])
                valid = chosen_cell >= 0
                if valid.any():
                    claimed[valid, chosen_cell[valid]] = True

        return action, total_log_prob, total_entropy, value

    def get_greedy_action_masked(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        k_nearest: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic (argmax) action selection with sequential spatial masking.

        Same masking logic as get_action_and_value but uses argmax instead of sample.

        Returns:
            action: (B, N) greedy actions
        """
        logits, _ = self.forward(grid, units, global_features, k_nearest)
        B = logits.shape[0]
        N = self.num_units

        action = torch.zeros(B, N, dtype=torch.long, device=logits.device)
        claimed = torch.zeros(B, self.rows * self.cols, dtype=torch.bool, device=logits.device)

        for i in range(N):
            drone_logits = logits[:, i, :].clone()

            if k_nearest is not None:
                for a in range(self.k):
                    cell_flat = self._resolve_cell_index(k_nearest, i, torch.full((B,), a, dtype=torch.long, device=logits.device))
                    is_claimed = claimed[torch.arange(B, device=logits.device), cell_flat.clamp(0)] & (cell_flat >= 0)
                    drone_logits[is_claimed, a] = float("-inf")

            action[:, i] = drone_logits.argmax(dim=-1)

            if k_nearest is not None:
                chosen_cell = self._resolve_cell_index(k_nearest, i, action[:, i])
                valid = chosen_cell >= 0
                if valid.any():
                    claimed[valid, chosen_cell[valid]] = True

        return action
