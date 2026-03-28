"""PyTorch attention-based actor-critic network for the MARL suppression agent.

Architecture:
- Per-drone representation: cat(grid_embed, unit_embed_i, kn_embed_i, global_embed) → (208,)
- DroneProjection: Linear(208→128, ReLU)
- DroneAttention: 4-head multi-head attention with pre-LN residual
- PolicyHead: per-drone Linear(128→64→11) (batched, no for-loop)
- ValueHead: masked mean-pool → Linear(128→64→1) (centralized)

Model is fully N-agnostic; drone count is derived from input tensor shapes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.marl.observation import K_NEAREST


# ---------------------------------------------------------------------------
# Encoder modules (copied from algorithms/rl/network.py to avoid coupling)
# ---------------------------------------------------------------------------

class GridEncoder(nn.Module):
    """CNN encoder for the (4, rows, cols) grid observation."""

    def __init__(self, in_channels: int = 4, embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 4, rows, cols) → (B, embed_dim)"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return F.relu(self.fc(x))


class UnitEncoder(nn.Module):
    """Shared MLP encoder for per-drone features."""

    def __init__(self, in_features: int = 5, embed_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, in_features) → (B, N, embed_dim)"""
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class GlobalEncoder(nn.Module):
    """MLP encoder for global scalar features."""

    def __init__(self, in_features: int = 4, embed_dim: int = 16):
        super().__init__()
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_features) → (B, embed_dim)"""
        return F.relu(self.fc(x))


class KNearestEncoder(nn.Module):
    """MLP encoder for per-drone k-nearest fire target coordinates."""

    def __init__(self, k: int = K_NEAREST, embed_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(k * 2, 32)
        self.fc2 = nn.Linear(32, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, k*2) → (B, N, embed_dim)"""
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


# ---------------------------------------------------------------------------
# Attention module
# ---------------------------------------------------------------------------

class DroneAttention(nn.Module):
    """Pre-LN multi-head self-attention for inter-drone communication.

    output = x + MHA(LN(x), LN(x), LN(x), key_padding_mask)
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, embed_dim)
            key_padding_mask: (B, N) bool — True for positions to IGNORE (padding).

        Returns:
            (B, N, embed_dim)
        """
        x_ln = self.ln(x)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln, key_padding_mask=key_padding_mask)
        return x + attn_out


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------

class WildfireActorCritic(nn.Module):
    """Attention-based actor-critic for multi-agent wildfire suppression.

    Fully N-agnostic: drone count is derived from input tensor shapes.
    No stored num_units / max_units.
    """

    def __init__(
        self,
        grid_channels: int = 4,
        unit_features: int = 5,
        global_features: int = 4,
        grid_embed_dim: int = 128,
        unit_embed_dim: int = 32,
        k_nearest_embed_dim: int = 32,
        global_embed_dim: int = 16,
        drone_repr_dim: int = 128,
        num_attn_heads: int = 4,
        k: int = K_NEAREST,
        rows: int = 16,
        cols: int = 16,
    ):
        super().__init__()
        self.k = k
        self.num_actions = k + 1  # K fires + idle
        self.rows = rows
        self.cols = cols

        # Encoders
        self.grid_encoder = GridEncoder(grid_channels, grid_embed_dim)
        self.unit_encoder = UnitEncoder(unit_features, unit_embed_dim)
        self.k_nearest_encoder = KNearestEncoder(k, k_nearest_embed_dim)
        self.global_encoder = GlobalEncoder(global_features, global_embed_dim)

        # Drone projection: concat of all embeds → drone_repr_dim
        cat_dim = grid_embed_dim + unit_embed_dim + k_nearest_embed_dim + global_embed_dim
        self.drone_projection = nn.Sequential(
            nn.Linear(cat_dim, drone_repr_dim),
            nn.ReLU(),
        )

        # Self-attention
        self.drone_attention = DroneAttention(drone_repr_dim, num_attn_heads)

        # Policy head (applied per drone, batched)
        self.policy_head = nn.Sequential(
            nn.Linear(drone_repr_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions),
        )

        # Value head (from masked mean-pooled representation)
        self.value_head = nn.Sequential(
            nn.Linear(drone_repr_dim, 64),
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
            units: (B, N, 5) per-unit features (first 2 are normalized row, col)

        Returns:
            (B, N, k_nearest_embed_dim)
        """
        drone_row = units[:, :, 0:1] * (self.rows - 1)  # (B, N, 1)
        drone_col = units[:, :, 1:2] * (self.cols - 1)  # (B, N, 1)

        rel_row = (k_nearest[:, :, :, 0] - drone_row) / self.rows  # (B, N, K)
        rel_col = (k_nearest[:, :, :, 1] - drone_col) / self.cols  # (B, N, K)

        rel_coords = torch.cat([rel_row, rel_col], dim=-1)  # (B, N, K*2)
        return self.k_nearest_encoder(rel_coords)

    def forward(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        k_nearest: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing action logits and state value.

        Args:
            grid: (B, 4, rows, cols)
            units: (B, N, 5)
            global_features: (B, 4)
            k_nearest: (B, N, K, 2) absolute fire target coords
            active_mask: (B, N) bool — True for real drones, False for padding

        Returns:
            logits: (B, N, num_actions)
            value: (B, 1)
        """
        B = grid.shape[0]
        N = units.shape[1]

        # Shared encodings
        grid_embed = self.grid_encoder(grid)                    # (B, grid_embed_dim)
        unit_embeds = self.unit_encoder(units)                  # (B, N, unit_embed_dim)
        global_embed = self.global_encoder(global_features)     # (B, global_embed_dim)

        # K-nearest encoding
        if k_nearest is not None:
            kn_embeds = self._encode_k_nearest(k_nearest, units)  # (B, N, kn_embed_dim)
        else:
            kn_embeds = torch.zeros(
                B, N, 32, device=grid.device, dtype=grid.dtype,
            )

        # Broadcast shared embeddings to per-drone: (B, N, dim)
        grid_broadcast = grid_embed.unsqueeze(1).expand(-1, N, -1)      # (B, N, 128)
        global_broadcast = global_embed.unsqueeze(1).expand(-1, N, -1)  # (B, N, 16)

        # Cat per-drone representation
        per_drone = torch.cat(
            [grid_broadcast, unit_embeds, kn_embeds, global_broadcast], dim=-1
        )  # (B, N, 208)

        # Project to drone repr
        drone_repr = self.drone_projection(per_drone)  # (B, N, 128)

        # Self-attention with padding mask
        # nn.MultiheadAttention key_padding_mask: True = IGNORE
        if active_mask is not None:
            key_padding_mask = ~active_mask  # True for padding positions
        else:
            key_padding_mask = None

        attended = self.drone_attention(drone_repr, key_padding_mask=key_padding_mask)
        # (B, N, 128)

        # Policy: batched over all drones at once
        logits = self.policy_head(attended)  # (B, N, num_actions)

        # Value: masked mean-pool → value head
        if active_mask is not None:
            # Mask out padding drones
            mask_float = active_mask.float().unsqueeze(-1)  # (B, N, 1)
            pooled = (attended * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            pooled = attended.mean(dim=1)  # (B, 128)

        value = self.value_head(pooled)  # (B, 1)

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
        idle_mask = action_idx == self.k
        safe_idx = action_idx.clamp(0, self.k - 1)

        drone_targets = k_nearest[:, drone_idx, :, :]  # (B, K, 2)
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
        active_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions with sequential spatial masking.

        Drones are processed in order. For each drone, fire targets that map to
        an already-claimed cell are masked (logit = -inf). Idle is never masked.
        Inactive drones (active_mask=False) are forced to idle and excluded from
        log_prob/entropy computation.

        Args:
            grid, units, global_features: observation tensors
            k_nearest: (B, N, K, 2) fire target coords
            action: (B, N) int tensor. If None, sample from policy.
            active_mask: (B, N) bool. If None, all drones active.

        Returns:
            action: (B, N)
            log_prob: (B,) sum of active per-drone log probs
            entropy: (B,) sum of active per-drone entropies
            value: (B, 1)
        """
        logits, value = self.forward(
            grid, units, global_features, k_nearest, active_mask
        )

        B = logits.shape[0]
        N = units.shape[1]
        sampling = action is None

        if sampling:
            action = torch.zeros(B, N, dtype=torch.long, device=logits.device)

        claimed = torch.zeros(
            B, self.rows * self.cols, dtype=torch.bool, device=logits.device
        )

        total_log_prob = torch.zeros(B, device=logits.device)
        total_entropy = torch.zeros(B, device=logits.device)

        for i in range(N):
            # Check if this drone is active
            if active_mask is not None:
                is_active = active_mask[:, i]  # (B,)
            else:
                is_active = torch.ones(B, dtype=torch.bool, device=logits.device)

            drone_logits = logits[:, i, :].clone()  # (B, num_actions)

            # Force inactive drones to idle
            if active_mask is not None:
                inactive = ~is_active
                if inactive.any():
                    drone_logits[inactive, :self.k] = float("-inf")

            # Mask fire actions whose target cell is already claimed
            if k_nearest is not None:
                for a in range(self.k):
                    cell_flat = self._resolve_cell_index(
                        k_nearest, i,
                        torch.full((B,), a, dtype=torch.long, device=logits.device),
                    )
                    is_claimed = (
                        claimed[torch.arange(B, device=logits.device), cell_flat.clamp(0)]
                        & (cell_flat >= 0)
                    )
                    drone_logits[is_claimed, a] = float("-inf")

            dist = torch.distributions.Categorical(logits=drone_logits)

            if sampling:
                action[:, i] = dist.sample()

            # Only accumulate log_prob/entropy for active drones
            lp = dist.log_prob(action[:, i])
            ent = dist.entropy()
            if active_mask is not None:
                lp = lp * is_active.float()
                ent = ent * is_active.float()
            total_log_prob += lp
            total_entropy += ent

            # Claim the chosen cell (only for active drones, not idle)
            if k_nearest is not None:
                chosen_cell = self._resolve_cell_index(k_nearest, i, action[:, i])
                valid = (chosen_cell >= 0) & is_active
                if valid.any():
                    claimed[valid, chosen_cell[valid]] = True

        return action, total_log_prob, total_entropy, value

    def get_greedy_action_masked(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        k_nearest: torch.Tensor | None = None,
        active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic (argmax) action selection with sequential spatial masking.

        Returns:
            action: (B, N) greedy actions
        """
        logits, _ = self.forward(
            grid, units, global_features, k_nearest, active_mask
        )

        B = logits.shape[0]
        N = units.shape[1]

        action = torch.zeros(B, N, dtype=torch.long, device=logits.device)
        claimed = torch.zeros(
            B, self.rows * self.cols, dtype=torch.bool, device=logits.device
        )

        for i in range(N):
            if active_mask is not None:
                is_active = active_mask[:, i]
            else:
                is_active = torch.ones(B, dtype=torch.bool, device=logits.device)

            drone_logits = logits[:, i, :].clone()

            # Force inactive drones to idle
            if active_mask is not None:
                inactive = ~is_active
                if inactive.any():
                    drone_logits[inactive, :self.k] = float("-inf")

            if k_nearest is not None:
                for a in range(self.k):
                    cell_flat = self._resolve_cell_index(
                        k_nearest, i,
                        torch.full((B,), a, dtype=torch.long, device=logits.device),
                    )
                    is_claimed = (
                        claimed[torch.arange(B, device=logits.device), cell_flat.clamp(0)]
                        & (cell_flat >= 0)
                    )
                    drone_logits[is_claimed, a] = float("-inf")

            action[:, i] = drone_logits.argmax(dim=-1)

            if k_nearest is not None:
                chosen_cell = self._resolve_cell_index(k_nearest, i, action[:, i])
                valid = (chosen_cell >= 0) & is_active
                if valid.any():
                    claimed[valid, chosen_cell[valid]] = True

        return action
