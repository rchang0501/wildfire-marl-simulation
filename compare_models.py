"""Compare RL v1 vs v2 models: do drones spread out or cluster?

Loads both models, runs them on identical fire scenarios, and reports
per-step drone position uniqueness and action diversity.
"""

import numpy as np
import torch

from algorithms.rl.gym_wrapper import WildfireEnv
from algorithms.rl.network import WildfireActorCritic
from algorithms.rl.observation import K_NEAREST, build_observation


class WildfireActorCriticV1(torch.nn.Module):
    """v1 architecture (no k_nearest_encoder, policy_input_dim=160)."""

    def __init__(self, num_units=8, k=K_NEAREST):
        super().__init__()
        from algorithms.rl.network import GridEncoder, UnitEncoder, GlobalEncoder
        self.num_units = num_units
        self.k = k
        self.num_actions = k + 1

        self.grid_encoder = GridEncoder(4, 128)
        self.unit_encoder = UnitEncoder(4, 32)
        self.global_encoder = GlobalEncoder(3, 16)

        self.shared_mlp = torch.nn.Sequential(
            torch.nn.Linear(128 + 32 + 16, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(128 + 32, 64),  # 160
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.num_actions),
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, grid, units, global_features):
        grid_embed = self.grid_encoder(grid)
        unit_embeds = self.unit_encoder(units)
        global_embed = self.global_encoder(global_features)
        unit_summary = unit_embeds.mean(dim=1)
        combined = torch.cat([grid_embed, unit_summary, global_embed], dim=1)
        shared = self.shared_mlp(combined)
        value = self.value_head(shared)
        logits_list = []
        for i in range(self.num_units):
            drone_embed = unit_embeds[:, i, :]
            policy_input = torch.cat([shared, drone_embed], dim=1)
            logits_list.append(self.policy_head(policy_input))
        logits = torch.stack(logits_list, dim=1)
        return logits, value


def load_model(param_dir: str, version: str):
    """Load a model, handling v1 vs v2 architecture."""
    import json, os
    with open(os.path.join(param_dir, "params.json")) as f:
        params = json.load(f)

    model_path = os.path.join(param_dir, params.get("model_file", "best_model.pt"))
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if version == "v1":
        model = WildfireActorCriticV1(
            num_units=params.get("num_units", 8),
            k=params.get("k", K_NEAREST),
        )
    else:
        model = WildfireActorCritic(
            num_units=params.get("num_units", 8),
            k=params.get("k", K_NEAREST),
            rows=params.get("rows", 16),
            cols=params.get("cols", 16),
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model


def run_episode(model, env, num_steps=200, use_k_nearest=True):
    """Run one episode, collecting per-step diagnostics."""
    obs, _ = env.reset()
    device = torch.device("cpu")

    results = {
        "unique_positions": [],   # how many distinct cells drones occupy
        "unique_actions": [],     # how many distinct action indices chosen
        "action_arrays": [],      # raw actions per step
        "drone_positions": [],    # (row, col) per drone per step
    }

    for t in range(num_steps):
        grid_t = torch.tensor(obs["grid"], dtype=torch.float32).unsqueeze(0)
        units_t = torch.tensor(obs["units"], dtype=torch.float32).unsqueeze(0)
        global_t = torch.tensor(obs["global_features"], dtype=torch.float32).unsqueeze(0)
        kn_t = torch.tensor(obs["k_nearest"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if use_k_nearest and isinstance(model, WildfireActorCritic):
                logits, _ = model(grid_t, units_t, global_t, k_nearest=kn_t)
            else:
                logits, _ = model(grid_t, units_t, global_t)
            actions = logits.argmax(dim=-1).squeeze(0).numpy()

        # Drone positions from unit features (normalized row, col)
        rows, cols = 16, 16
        drone_rows = (obs["units"][:, 0] * (rows - 1)).round().astype(int)
        drone_cols = (obs["units"][:, 1] * (cols - 1)).round().astype(int)
        positions = set(zip(drone_rows.tolist(), drone_cols.tolist()))

        results["unique_positions"].append(len(positions))
        results["unique_actions"].append(len(set(actions.tolist())))
        results["action_arrays"].append(actions.copy())
        results["drone_positions"].append(
            list(zip(drone_rows.tolist(), drone_cols.tolist()))
        )

        # Step env
        from algorithms.rl.action_translation import translate_actions
        env_actions = translate_actions(env.jenv, actions, obs["k_nearest"])
        next_obs, _, terminated, truncated, _ = env.step(actions)

        if terminated or truncated:
            break
        obs = next_obs

    return results


def print_report(name, results):
    n = len(results["unique_positions"])
    avg_pos = np.mean(results["unique_positions"])
    avg_act = np.mean(results["unique_actions"])
    min_pos = np.min(results["unique_positions"])
    max_pos = np.max(results["unique_positions"])

    # Count steps where ALL drones pick the same action
    all_same_action = sum(
        1 for a in results["action_arrays"] if len(set(a.tolist())) == 1
    )
    # Count steps where ALL drones are on the same cell
    all_same_cell = sum(
        1 for p in results["unique_positions"] if p == 1
    )

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Steps run:                    {n}")
    print(f"  Avg unique drone positions:   {avg_pos:.2f} / 8")
    print(f"  Min/Max unique positions:     {min_pos} / {max_pos}")
    print(f"  Avg unique actions:           {avg_act:.2f} / 8")
    print(f"  Steps all-same-action:        {all_same_action} / {n} ({100*all_same_action/n:.1f}%)")
    print(f"  Steps all-same-cell:          {all_same_cell} / {n} ({100*all_same_cell/n:.1f}%)")

    # Show first 10 steps action detail
    print(f"\n  First 10 steps — actions per drone:")
    for t in range(min(10, n)):
        acts = results["action_arrays"][t]
        pos = results["drone_positions"][t]
        print(f"    t={t:3d}  actions={acts}  positions={pos}")


def main():
    seed = 123  # Fixed seed so both models see identical fire

    print("Loading models...")
    model_v1 = load_model("trained_models/rl", version="v1")
    model_v2 = load_model("trained_models/rl_v2", version="v2")

    # Count parameters to confirm architectures differ
    p_v1 = sum(p.numel() for p in model_v1.parameters())
    p_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"v1 parameters: {p_v1:,}")
    print(f"v2 parameters: {p_v2:,}")
    print(f"Architecture different: {p_v1 != p_v2}")

    # Compare weight differences in shared layers
    sd_v1 = model_v1.state_dict()
    sd_v2 = model_v2.state_dict()
    shared_keys = [k for k in sd_v1 if k in sd_v2]
    print(f"\nShared layer weight comparison ({len(shared_keys)} layers):")
    for k in shared_keys[:6]:  # first few
        diff = (sd_v1[k] - sd_v2[k]).abs().mean().item()
        print(f"  {k}: mean abs diff = {diff:.6f}")

    v2_only = [k for k in sd_v2 if k not in sd_v1]
    if v2_only:
        print(f"\nv2-only layers (new k_nearest_encoder):")
        for k in v2_only:
            print(f"  {k}: shape={sd_v2[k].shape}")

    # Run both on same scenario
    print("\nRunning episodes on identical fire scenario (seed=123)...")

    env_v1 = WildfireEnv(seed=seed)
    results_v1 = run_episode(model_v1, env_v1, use_k_nearest=False)

    env_v2 = WildfireEnv(seed=seed)
    results_v2 = run_episode(model_v2, env_v2, use_k_nearest=True)

    print_report("RL v1 (no k_nearest fed to network)", results_v1)
    print_report("RL v2 (k_nearest fed to network)", results_v2)

    # Diagnosis
    print(f"\n{'='*60}")
    print("  DIAGNOSIS")
    print(f"{'='*60}")
    avg1 = np.mean(results_v1["unique_actions"])
    avg2 = np.mean(results_v2["unique_actions"])
    if avg2 - avg1 < 0.5:
        print("  WARNING: v2 action diversity is NOT meaningfully better than v1.")
        print("  The k_nearest encoder may not be providing enough signal,")
        print("  or 500K timesteps is insufficient for the model to learn")
        print("  to use the new input. Consider:")
        print("    - Training longer (1M+ steps)")
        print("    - Increasing overlap penalty")
        print("    - Checking if k_nearest relative coords have good variance")

        # Check k_nearest variance
        env_check = WildfireEnv(seed=seed)
        obs, _ = env_check.reset()
        # Step a few times to get fires going
        for _ in range(20):
            obs, _, _, _, _ = env_check.step(np.zeros(8, dtype=int))
        kn = obs["k_nearest"]  # (8, 10, 2)
        units = obs["units"]   # (8, 4)
        drone_rows = units[:, 0] * 15
        drone_cols = units[:, 1] * 15
        print(f"\n  Sample k_nearest diagnostic (after 20 steps):")
        for i in range(min(3, 8)):
            rel_r = kn[i, :, 0] - drone_rows[i]
            rel_c = kn[i, :, 1] - drone_cols[i]
            print(f"    Drone {i} at ({drone_rows[i]:.0f},{drone_cols[i]:.0f})")
            print(f"      k_nearest targets: {kn[i].tolist()}")
            print(f"      relative offsets:  row={rel_r.round(1).tolist()}, col={rel_c.round(1).tolist()}")
    else:
        print("  v2 shows improved action diversity over v1!")


if __name__ == "__main__":
    main()
