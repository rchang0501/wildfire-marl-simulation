import argparse
import json
from pathlib import Path
import numpy as np

from environment import Fire_Environment
from algorithms.sharing_algorithms import SHARING_ALGORITHM_REGISTRY
from algorithms.suppression_algorithms import SUPPRESSION_ALGORITHM_REGISTRY


DEFAULTS = {
    "num_juris_rows": 2,
    "num_juris_cols": 2,
    "per_juris_rows": 16,
    "per_juris_cols": 16,
    "base_spread_prob": 0.06,
    "num_units_per_juris": 8,
    "suppression_success_prob": 0.8,
    "movement_per_step": 4,
    "juris_travel_time": 4,
    "lightning_mu_log": -2.0,
    "lightning_sigma_log": 2.0,
}


def count_units_per_juris(env: Fire_Environment) -> list[int]:
    counts = np.zeros(env.num_juris, dtype=int)
    cells = env.unit_positions[:, 1]
    locs = env.unit_positions[:, 0]
    mask = cells >= 0
    if np.any(mask):
        np.add.at(counts, locs[mask], 1)
    return counts.tolist()


def build_rngs(lightning_seed: int, spread_seed: int | None):
    if spread_seed is None:
        seed_seq = np.random.SeedSequence(lightning_seed)
        child_seqs = seed_seq.spawn(3)
        rng_lightning = np.random.default_rng(child_seqs[0])
        rng_spread = np.random.default_rng(child_seqs[1])
        rng_algo = np.random.default_rng(child_seqs[2])
    else:
        rng_lightning = np.random.default_rng(lightning_seed)
        rng_spread = np.random.default_rng(spread_seed)
        rng_algo = np.random.default_rng(spread_seed + 1_000_000)
    return rng_lightning, rng_spread, rng_algo


def run_instance(
    sharing_algorithm_name: str,
    suppression_algorithm_name: str,
    sharing_param_dir: str,
    suppression_param_dir: str,
    sharing_params: dict,
    suppression_params: dict,
    steps: int,
    lightning_seed: int,
    spread_seed: int | None,
    save_snapshots: bool,
    output_dir: str,
    verbose: bool,
    run_label: str,
    num_juris_rows: int,
    num_juris_cols: int,
    per_juris_rows: int,
    per_juris_cols: int,
    base_spread_prob: float,
    num_units_per_juris: int,
    suppression_success_prob: float,
    movement_per_step: int,
    juris_travel_time: int,
    lightning_mu_log: float,
    lightning_sigma_log: float,
):
    if sharing_algorithm_name not in SHARING_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown sharing algorithm: {sharing_algorithm_name}")
    if suppression_algorithm_name not in SUPPRESSION_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown suppression algorithm: {suppression_algorithm_name}")

    sharing_cls = SHARING_ALGORITHM_REGISTRY[sharing_algorithm_name]
    suppression_cls = SUPPRESSION_ALGORITHM_REGISTRY[suppression_algorithm_name]

    sharing_algo = sharing_cls(param_dir=sharing_param_dir, params=sharing_params)
    suppression_algo = suppression_cls(param_dir=suppression_param_dir, params=suppression_params)

    output_path = Path(output_dir)
    if save_snapshots:
        output_path.mkdir(parents=True, exist_ok=True)

    env = Fire_Environment(
        num_juris_rows=num_juris_rows,
        num_juris_cols=num_juris_cols,
        per_juris_rows=per_juris_rows,
        per_juris_cols=per_juris_cols,
        base_spread_prob=base_spread_prob,
        num_units_per_juris=num_units_per_juris,
        suppression_success_prob=suppression_success_prob,
        movement_per_step=movement_per_step,
        juris_travel_time=juris_travel_time,
        lightning_mu_log=lightning_mu_log,
        lightning_sigma_log=lightning_sigma_log,
    )

    rng_lightning, rng_spread, rng_algo = build_rngs(lightning_seed, spread_seed)

    if save_snapshots:
        burn_snap = np.zeros((steps + 1, env.num_juris, env.per_juris_rows, env.per_juris_cols), dtype=bool)
        unit_pos_snap = np.zeros((steps + 1, env.num_units_total, 2), dtype=int)
        burn_snap[0] = env.burning_map
        unit_pos_snap[0] = env.unit_positions

    if verbose:
        burning_count = int(np.sum(env.burning_map))
        units_per_juris = count_units_per_juris(env)
        print(
            f"[sharing={sharing_algorithm_name} suppression={suppression_algorithm_name}] "
            f"{run_label}: step 0 | "
            f"burning={burning_count} | units_per_juris={units_per_juris}"
        )

    for step_idx in range(steps):
        assigned_mask, assigned_actions = sharing_algo.get_assignments(env, rng_algo)
        actions = suppression_algo.get_actions(env, rng_algo, assigned_mask, assigned_actions)
        burning_map, unit_positions, rewards, counts_burning = env.step(
            actions,
            rng_spread=rng_spread,
            rng_lightning=rng_lightning,
        )

        if save_snapshots:
            burn_snap[step_idx + 1] = burning_map
            unit_pos_snap[step_idx + 1] = unit_positions

        if verbose:
            burning_count = int(np.sum(burning_map))
            units_per_juris = count_units_per_juris(env)
            print(
                f"[sharing={sharing_algorithm_name} suppression={suppression_algorithm_name}] "
                f"{run_label}: step {step_idx + 1} | "
                f"burning={burning_count} | units_per_juris={units_per_juris}"
            )

    final_damage = np.sum(env.burning_map, axis=(1, 2)).astype(int).tolist()
    print(
        f"[sharing={sharing_algorithm_name} suppression={suppression_algorithm_name}] "
        f"{run_label}: final_damage_per_juris={final_damage}"
    )

    if save_snapshots:
        out_file = output_path / (
            f"{run_label}__sharing_{sharing_algorithm_name}__"
            f"suppression_{suppression_algorithm_name}.npz"
        )
        np.savez_compressed(
            out_file,
            burning_map=burn_snap,
            unit_positions=unit_pos_snap,
            steps=steps,
            lightning_seed=lightning_seed,
            spread_seed=spread_seed,
        )
        meta_file = out_file.with_name(f"{out_file.stem}__meta.json")
        metadata = {
            "run_label": run_label,
            "sharing_algorithm": sharing_algorithm_name,
            "suppression_algorithm": suppression_algorithm_name,
            "num_juris_rows": env.num_juris_rows,
            "num_juris_cols": env.num_juris_cols,
            "per_juris_rows": env.per_juris_rows,
            "per_juris_cols": env.per_juris_cols,
            "base_spread_prob": env.base_spread_prob,
            "num_units_per_juris": env.num_units_per_juris,
            "suppression_success_prob": env.suppression_success_prob,
            "movement_per_step": env.movement_per_step,
            "juris_travel_time": env.juris_travel_time,
            "adj_matrix": env.adj_matrix,
            "lightning_mu_log": env.lightning_mu_log,
            "lightning_sigma_log": env.lightning_sigma_log,
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run algorithm comparisons on fire instances.")
    parser.add_argument(
        "--sharing-algorithm",
        default="none",
        help=f"Sharing algorithm. Available: {sorted(SHARING_ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--suppression-algorithm",
        default="greedy",
        help=f"Suppression algorithm. Available: {sorted(SUPPRESSION_ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--sharing-param-dir",
        default="",
        help="Directory containing params.json for the sharing algorithm (optional).",
    )
    parser.add_argument(
        "--suppression-param-dir",
        default="",
        help="Directory containing params.json for the suppression algorithm (optional).",
    )
    parser.add_argument(
        "--period-s",
        type=int,
        default=None,
        help="Periodic transfer interval (only used by sharing algorithm periodic_transfer).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of steps per run.")
    parser.add_argument("--lightning-seed", type=int, default=0, help="Starting lightning seed.")
    parser.add_argument(
        "--spread-seed",
        type=int,
        default=None,
        help="Starting spread/suppression seed (optional).",
    )
    parser.add_argument("--save-snapshots", action="store_true", help="Save state snapshots for each run.")
    parser.add_argument("--output-dir", default="results", help="Directory for saved snapshots.")
    parser.add_argument("--verbose", action="store_true", help="Verbose per-step logging.")
    parser.add_argument("--run-label", default="cli", help="Label used to name output snapshot files.")
    parser.add_argument("--num-juris-rows", type=int, default=DEFAULTS["num_juris_rows"])
    parser.add_argument("--num-juris-cols", type=int, default=DEFAULTS["num_juris_cols"])
    parser.add_argument("--per-juris-rows", type=int, default=DEFAULTS["per_juris_rows"])
    parser.add_argument("--per-juris-cols", type=int, default=DEFAULTS["per_juris_cols"])
    parser.add_argument("--base-spread-prob", type=float, default=DEFAULTS["base_spread_prob"])
    parser.add_argument("--num-units-per-juris", type=int, default=DEFAULTS["num_units_per_juris"])
    parser.add_argument("--suppression-success-prob", type=float, default=DEFAULTS["suppression_success_prob"])
    parser.add_argument("--movement-per-step", type=int, default=DEFAULTS["movement_per_step"])
    parser.add_argument("--juris-travel-time", type=int, default=DEFAULTS["juris_travel_time"])
    parser.add_argument("--lightning-mu-log", type=float, default=DEFAULTS["lightning_mu_log"])
    parser.add_argument("--lightning-sigma-log", type=float, default=DEFAULTS["lightning_sigma_log"])

    args = parser.parse_args()

    sharing_params: dict[str, int | float | str | list] = {}
    suppression_params: dict[str, int | float | str | list] = {}

    if args.period_s is not None:
        sharing_params["period_s"] = args.period_s
    sharing_params["total_steps"] = args.steps

    run_instance(
        sharing_algorithm_name=args.sharing_algorithm,
        suppression_algorithm_name=args.suppression_algorithm,
        sharing_param_dir=args.sharing_param_dir,
        suppression_param_dir=args.suppression_param_dir,
        sharing_params=sharing_params,
        suppression_params=suppression_params,
        steps=args.steps,
        lightning_seed=args.lightning_seed,
        spread_seed=args.spread_seed,
        save_snapshots=args.save_snapshots,
        output_dir=args.output_dir,
        verbose=args.verbose,
        run_label=args.run_label,
        num_juris_rows=args.num_juris_rows,
        num_juris_cols=args.num_juris_cols,
        per_juris_rows=args.per_juris_rows,
        per_juris_cols=args.per_juris_cols,
        base_spread_prob=args.base_spread_prob,
        num_units_per_juris=args.num_units_per_juris,
        suppression_success_prob=args.suppression_success_prob,
        movement_per_step=args.movement_per_step,
        juris_travel_time=args.juris_travel_time,
        lightning_mu_log=args.lightning_mu_log,
        lightning_sigma_log=args.lightning_sigma_log,
    )


if __name__ == "__main__":
    main()
