"""PyTorch policy/value network for the RL suppression agent.

Architecture:
- Grid Encoder (CNN): Processes (4, rows, cols) grid channels
- Unit Encoder (shared MLP): Processes per-drone features
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

    def __init__(self, in_features: int = 4, embed_dim: int = 32):
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


class WildfireActorCritic(nn.Module):
    """Combined actor-critic network for wildfire suppression.

    Combines grid, unit, and global encoders into a shared representation,
    then produces per-drone action logits and a state value estimate.
    """

    def __init__(
        self,
        num_units: int = 8,
        grid_channels: int = 4,
        unit_features: int = 4,
        global_features: int = 3,
        grid_embed_dim: int = 128,
        unit_embed_dim: int = 32,
        global_embed_dim: int = 16,
        hidden_dim: int = 128,
        k: int = K_NEAREST,
    ):
        super().__init__()
        self.num_units = num_units
        self.k = k
        self.num_actions = k + 1  # K fires + idle

        self.grid_encoder = GridEncoder(grid_channels, grid_embed_dim)
        self.unit_encoder = UnitEncoder(unit_features, unit_embed_dim)
        self.global_encoder = GlobalEncoder(global_features, global_embed_dim)

        combined_dim = grid_embed_dim + unit_embed_dim + global_embed_dim
        self.shared_mlp = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-drone policy head: takes shared representation + this drone's embedding
        policy_input_dim = hidden_dim + unit_embed_dim
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

    def forward(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing action logits and state value.

        Args:
            grid: (batch, 4, rows, cols)
            units: (batch, num_units, 4)
            global_features: (batch, 3)

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

        # Per-drone policy: concatenate shared with per-drone embedding
        batch_size = grid.shape[0]
        logits_list = []
        for i in range(self.num_units):
            drone_embed = unit_embeds[:, i, :]                  # (B, 32)
            policy_input = torch.cat([shared, drone_embed], dim=1)  # (B, 160)
            logits_i = self.policy_head(policy_input)           # (B, num_actions)
            logits_list.append(logits_i)

        logits = torch.stack(logits_list, dim=1)                # (B, N, num_actions)

        return logits, value

    def get_action_and_value(
        self,
        grid: torch.Tensor,
        units: torch.Tensor,
        global_features: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions or evaluate given actions.

        Args:
            grid, units, global_features: observation tensors (batch dims)
            action: (batch, num_units) int tensor. If None, sample from policy.

        Returns:
            action: (batch, num_units) sampled or given actions
            log_prob: (batch,) sum of per-drone log probs
            entropy: (batch,) sum of per-drone entropies
            value: (batch, 1) state value
        """
        logits, value = self.forward(grid, units, global_features)
        # logits: (B, N, num_actions)

        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()                              # (B, N)

        log_prob = dist.log_prob(action).sum(dim=1)             # (B,)
        entropy = dist.entropy().sum(dim=1)                     # (B,)

        return action, log_prob, entropy, value
