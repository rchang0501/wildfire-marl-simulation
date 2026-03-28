"""PPO training script for the MARL wildfire suppression agent.

Usage:
    python -m algorithms.marl.train --total-timesteps 500000 --output-dir trained_models/marl

Trains a multi-agent actor-critic policy with attention-based inter-drone
communication using Proximal Policy Optimization. Supports variable drone
counts per episode.
"""

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.marl.gym_wrapper import WildfireEnv
from algorithms.marl.network import WildfireActorCritic
from algorithms.marl.observation import K_NEAREST


def collect_rollout(
    env: WildfireEnv,
    model: WildfireActorCritic,
    num_steps: int,
    device: torch.device,
    obs: dict[str, np.ndarray],
) -> tuple[dict, dict[str, np.ndarray]]:
    """Collect a rollout of num_steps transitions.

    Returns:
        rollout: dict of arrays with keys including active_mask
        last_obs: observation after the last step (for bootstrapping)
    """
    storage = {
        "grid": [],
        "units": [],
        "global_features": [],
        "k_nearest": [],
        "active_mask": [],
        "actions": [],
        "log_probs": [],
        "rewards": [],
        "values": [],
        "dones": [],
    }

    for _ in range(num_steps):
        grid_t = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
        units_t = torch.tensor(obs["units"], dtype=torch.float32, device=device).unsqueeze(0)
        global_t = torch.tensor(obs["global_features"], dtype=torch.float32, device=device).unsqueeze(0)
        kn_t = torch.tensor(obs["k_nearest"], dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.tensor(obs["active_mask"], dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(
                grid_t, units_t, global_t, k_nearest=kn_t, active_mask=mask_t,
            )

        action_np = action.squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        storage["grid"].append(obs["grid"])
        storage["units"].append(obs["units"])
        storage["global_features"].append(obs["global_features"])
        storage["k_nearest"].append(obs["k_nearest"])
        storage["active_mask"].append(obs["active_mask"])
        storage["actions"].append(action_np)
        storage["log_probs"].append(log_prob.item())
        storage["rewards"].append(reward)
        storage["values"].append(value.item())
        storage["dones"].append(float(done))

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    rollout = {
        k: np.array(v) for k, v in storage.items()
    }

    return rollout, obs


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation."""
    num_steps = len(rewards)
    advantages = np.zeros(num_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(
    model: WildfireActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    advantages: np.ndarray,
    returns: np.ndarray,
    device: torch.device,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    minibatch_size: int = 64,
) -> dict[str, float]:
    """Perform PPO policy and value updates."""
    num_steps = len(advantages)
    indices = np.arange(num_steps)

    grid_t = torch.tensor(rollout["grid"], dtype=torch.float32, device=device)
    units_t = torch.tensor(rollout["units"], dtype=torch.float32, device=device)
    global_t = torch.tensor(rollout["global_features"], dtype=torch.float32, device=device)
    kn_t = torch.tensor(rollout["k_nearest"], dtype=torch.float32, device=device)
    mask_t = torch.tensor(rollout["active_mask"], dtype=torch.bool, device=device)
    actions_t = torch.tensor(rollout["actions"], dtype=torch.long, device=device)
    old_log_probs_t = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    num_updates = 0

    for _ in range(num_epochs):
        np.random.shuffle(indices)

        for start in range(0, num_steps, minibatch_size):
            end = min(start + minibatch_size, num_steps)
            mb_idx = indices[start:end]

            _, new_log_prob, entropy, new_value = model.get_action_and_value(
                grid_t[mb_idx],
                units_t[mb_idx],
                global_t[mb_idx],
                k_nearest=kn_t[mb_idx],
                action=actions_t[mb_idx],
                active_mask=mask_t[mb_idx],
            )

            ratio = torch.exp(new_log_prob - old_log_probs_t[mb_idx])
            surr1 = ratio * advantages_t[mb_idx]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_t[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(
                new_value.squeeze(-1), returns_t[mb_idx]
            )

            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            num_updates += 1

    return {
        "policy_loss": total_policy_loss / max(num_updates, 1),
        "value_loss": total_value_loss / max(num_updates, 1),
        "entropy": total_entropy / max(num_updates, 1),
    }


def find_latest_checkpoint(output_dir: str) -> tuple[str | None, int]:
    """Find the latest checkpoint file and extract its step count."""
    pattern = os.path.join(output_dir, "checkpoint_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None, 0

    best_path = None
    best_steps = 0
    for f in files:
        match = re.search(r"checkpoint_(\d+)\.pt$", f)
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best_path = f

    return best_path, best_steps


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create environment
    env = WildfireEnv(
        rows=args.rows,
        cols=args.cols,
        max_units=args.max_units,
        min_units=args.min_units,
        base_spread_prob=args.base_spread_prob,
        suppression_success_prob=args.suppression_success_prob,
        movement_per_step=args.movement_per_step,
        lightning_mu_log=args.lightning_mu_log,
        lightning_sigma_log=args.lightning_sigma_log,
        max_fuel=args.max_fuel,
        fuel_refuel_rate=args.fuel_refuel_rate,
        max_steps=args.episode_steps,
        seed=args.seed,
    )

    # Create model — no max_units param, fully N-agnostic
    model = WildfireActorCritic(
        global_features=4,
        k=K_NEAREST,
        rows=args.rows,
        cols=args.cols,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume from checkpoint
    total_steps = 0
    num_updates = 0
    best_mean_reward = -float("inf")

    if args.resume:
        ckpt_path, ckpt_steps = find_latest_checkpoint(args.output_dir)
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                total_steps = checkpoint.get("total_steps", ckpt_steps)
                num_updates = checkpoint.get("num_updates", total_steps // args.rollout_steps)
                best_mean_reward = checkpoint.get("best_mean_reward", -float("inf"))
                print(f"Resumed from {ckpt_path} (full checkpoint) at step {total_steps}")
            else:
                model.load_state_dict(checkpoint)
                total_steps = ckpt_steps
                num_updates = total_steps // args.rollout_steps
                print(f"Resumed from {ckpt_path} (model weights only) at step {total_steps}")
        else:
            print(f"--resume set but no checkpoints found in {args.output_dir}, starting fresh")

    # Save training config
    config = vars(args)
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    episode_rewards = []

    obs, _ = env.reset()
    start_time = time.time()

    while total_steps < args.total_timesteps:
        rollout, obs = collect_rollout(
            env, model, args.rollout_steps, device, obs
        )

        # Bootstrap value for last observation
        with torch.no_grad():
            grid_t = torch.tensor(obs["grid"], dtype=torch.float32, device=device).unsqueeze(0)
            units_t = torch.tensor(obs["units"], dtype=torch.float32, device=device).unsqueeze(0)
            global_t = torch.tensor(obs["global_features"], dtype=torch.float32, device=device).unsqueeze(0)
            kn_t = torch.tensor(obs["k_nearest"], dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(obs["active_mask"], dtype=torch.bool, device=device).unsqueeze(0)
            _, last_value = model(grid_t, units_t, global_t, k_nearest=kn_t, active_mask=mask_t)
            last_value = last_value.item()

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"],
            last_value,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        losses = ppo_update(
            model, optimizer, rollout, advantages, returns, device,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
        )

        total_steps += args.rollout_steps
        num_updates += 1

        ep_reward = float(np.sum(rollout["rewards"]))
        episode_rewards.append(ep_reward)

        if num_updates % args.log_interval == 0:
            elapsed = time.time() - start_time
            recent_rewards = episode_rewards[-args.log_interval:]
            mean_reward = np.mean(recent_rewards)

            print(
                f"Update {num_updates} | "
                f"Steps {total_steps}/{args.total_timesteps} | "
                f"Mean rollout reward: {mean_reward:.3f} | "
                f"Policy loss: {losses['policy_loss']:.4f} | "
                f"Value loss: {losses['value_loss']:.4f} | "
                f"Entropy: {losses['entropy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, "best_model.pt"),
                )

        if num_updates % args.save_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "total_steps": total_steps,
                    "num_updates": num_updates,
                    "best_mean_reward": best_mean_reward,
                },
                os.path.join(args.output_dir, f"checkpoint_{total_steps}.pt"),
            )

    # Save final model
    torch.save(
        model.state_dict(),
        os.path.join(args.output_dir, "final_model.pt"),
    )

    # Save params.json for inference integration
    params = {
        "k": K_NEAREST,
        "rows": args.rows,
        "cols": args.cols,
        "unit_features": 5,
        "global_features": 4,
        "architecture": "attention",
        "model_file": "best_model.pt",
    }
    with open(os.path.join(args.output_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    print(f"Training complete. Models saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train MARL suppression agent with PPO.")

    # Environment parameters
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--min-units", type=int, default=4)
    parser.add_argument("--base-spread-prob", type=float, default=0.06)
    parser.add_argument("--suppression-success-prob", type=float, default=0.8)
    parser.add_argument("--movement-per-step", type=int, default=4)
    parser.add_argument("--lightning-mu-log", type=float, default=-2.0)
    parser.add_argument("--lightning-sigma-log", type=float, default=2.0)
    parser.add_argument("--max-fuel", type=int, default=None)
    parser.add_argument("--fuel-refuel-rate", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=200)

    # Training hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # Logging and output
    parser.add_argument("--output-dir", type=str, default="trained_models/marl")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in output-dir")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
