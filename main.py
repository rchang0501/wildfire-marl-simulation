import argparse
import json
from pathlib import Path
import numpy as np

from environment.jurisdiction_env import JurisdictionEnv
from environment.multi_jurisdiction_env import MultiJurisdictionEnv
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


# ======================================================================
# Single-jurisdiction mode
# ======================================================================

def run_single(
    suppression_algorithm_name: str,
    suppression_param_dir: str,
    suppression_params: dict,
    steps: int,
    lightning_seed: int,
    spread_seed: int | None,
    save_snapshots: bool,
    output_dir: str,
    verbose: bool,
    run_label: str,
    per_juris_rows: int,
    per_juris_cols: int,
    base_spread_prob: float,
    num_units_per_juris: int,
    suppression_success_prob: float,
    movement_per_step: int,
    lightning_mu_log: float,
    lightning_sigma_log: float,
):
    if suppression_algorithm_name not in SUPPRESSION_ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown suppression algorithm: {suppression_algorithm_name}")

    suppression_cls = SUPPRESSION_ALGORITHM_REGISTRY[suppression_algorithm_name]
    suppression_algo = suppression_cls(param_dir=suppression_param_dir, params=suppression_params)

    output_path = Path(output_dir)
    if save_snapshots:
        output_path.mkdir(parents=True, exist_ok=True)

    jenv = JurisdictionEnv(
        rows=per_juris_rows,
        cols=per_juris_cols,
        base_spread_prob=base_spread_prob,
        suppression_success_prob=suppression_success_prob,
        movement_per_step=movement_per_step,
        lightning_mu_log=lightning_mu_log,
        lightning_sigma_log=lightning_sigma_log,
        num_units=num_units_per_juris,
    )

    rng_lightning, rng_spread, rng_algo = build_rngs(lightning_seed, spread_seed)

    # Snapshot arrays: store as (J=1, R, C) for animator compatibility
    if save_snapshots:
        burn_snap = np.zeros((steps + 1, 1, per_juris_rows, per_juris_cols), dtype=bool)
        unit_pos_snap = np.zeros((steps + 1, num_units_per_juris, 2), dtype=int)
        burn_snap[0, 0] = jenv.burning_map
        unit_pos_snap[0, :, 0] = 0  # jurisdiction 0
        unit_pos_snap[0, :, 1] = jenv.unit_positions

    if verbose:
        print(
            f"[mode=single suppression={suppression_algorithm_name}] "
            f"{run_label}: step 0 | burning={jenv.burning_count} | "
            f"units={jenv.num_units}"
        )

    for step_idx in range(steps):
        actions = suppression_algo.get_actions(jenv, rng_algo)
        _, _, reward, count = jenv.step(actions, rng_spread=rng_spread, rng_lightning=rng_lightning)

        if save_snapshots:
            burn_snap[step_idx + 1, 0] = jenv.burning_map
            unit_pos_snap[step_idx + 1, :, 0] = 0
            unit_pos_snap[step_idx + 1, :, 1] = jenv.unit_positions

        if verbose:
            print(
                f"[mode=single suppression={suppression_algorithm_name}] "
                f"{run_label}: step {step_idx + 1} | burning={count} | "
                f"units={jenv.num_units}"
            )

    final_damage = jenv.burning_count
    print(
        f"[mode=single suppression={suppression_algorithm_name}] "
        f"{run_label}: final_damage={final_damage}"
    )

    if save_snapshots:
        out_file = output_path / (
            f"{run_label}__mode_single__suppression_{suppression_algorithm_name}.npz"
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
            "mode": "single",
            "suppression_algorithm": suppression_algorithm_name,
            "num_juris_rows": 1,
            "num_juris_cols": 1,
            "per_juris_rows": per_juris_rows,
            "per_juris_cols": per_juris_cols,
            "base_spread_prob": base_spread_prob,
            "num_units_per_juris": num_units_per_juris,
            "suppression_success_prob": suppression_success_prob,
            "movement_per_step": movement_per_step,
            "juris_travel_time": 0,
            "adj_matrix": [[0]],
            "lightning_mu_log": lightning_mu_log,
            "lightning_sigma_log": lightning_sigma_log,
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ======================================================================
# Multi-jurisdiction mode
# ======================================================================

def run_multi(
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

    multi_env = MultiJurisdictionEnv(
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
        burn_snap = np.zeros(
            (steps + 1, multi_env.num_juris, per_juris_rows, per_juris_cols), dtype=bool
        )
        unit_pos_snap = np.zeros((steps + 1, multi_env.num_units_total, 2), dtype=int)
        burning_0, positions_0 = multi_env.get_snapshot()
        burn_snap[0] = burning_0
        unit_pos_snap[0] = positions_0

    if verbose:
        print(
            f"[sharing={sharing_algorithm_name} suppression={suppression_algorithm_name}] "
            f"{run_label}: step 0 | "
            f"burning={multi_env.total_burning} | units_per_juris={multi_env.unit_count_per_juris}"
        )

    for step_idx in range(steps):
        # 1) Sharing: decide transfers
        transfers = sharing_algo.decide_transfers(multi_env, rng_algo)
        for unit_id, target_juris in transfers:
            multi_env.initiate_transfer(unit_id, target_juris)

        # 2) Advance transit (deliver arrivals before computing actions)
        multi_env.advance_transit()

        # 3) Sharing: get steering overrides (may select new active unit)
        steering = sharing_algo.get_steering_actions(multi_env, rng_algo)

        # 4) Suppression: get actions for each jurisdiction
        suppression_actions: dict[int, np.ndarray] = {}
        for j_idx, jenv in enumerate(multi_env.jurisdictions):
            if jenv.num_units > 0:
                suppression_actions[j_idx] = suppression_algo.get_actions(jenv, rng_algo)

        # 5) Apply steering overrides
        for uid, (dx, dy) in steering.items():
            cur_j = int(multi_env.unit_jurisdiction[uid])
            if cur_j < 0:
                continue
            local_idx = int(multi_env.unit_local_index[uid])
            if cur_j in suppression_actions:
                suppression_actions[cur_j][local_idx] = (dx, dy)

        # 6) Step the environment
        rewards, counts = multi_env.step(
            suppression_actions,
            rng_spread=rng_spread,
            rng_lightning=rng_lightning,
        )

        if save_snapshots:
            burning_t, positions_t = multi_env.get_snapshot()
            burn_snap[step_idx + 1] = burning_t
            unit_pos_snap[step_idx + 1] = positions_t

        if verbose:
            print(
                f"[sharing={sharing_algorithm_name} suppression={suppression_algorithm_name}] "
                f"{run_label}: step {step_idx + 1} | "
                f"burning={multi_env.total_burning} | "
                f"units_per_juris={multi_env.unit_count_per_juris}"
            )

    final_damage = multi_env.burning_counts
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
            "mode": "multi",
            "sharing_algorithm": sharing_algorithm_name,
            "suppression_algorithm": suppression_algorithm_name,
            "num_juris_rows": multi_env.num_juris_rows,
            "num_juris_cols": multi_env.num_juris_cols,
            "per_juris_rows": per_juris_rows,
            "per_juris_cols": per_juris_cols,
            "base_spread_prob": multi_env.base_spread_prob,
            "num_units_per_juris": multi_env.num_units_per_juris,
            "suppression_success_prob": multi_env.suppression_success_prob,
            "movement_per_step": multi_env.movement_per_step,
            "juris_travel_time": multi_env.juris_travel_time,
            "adj_matrix": multi_env.adj_matrix,
            "lightning_mu_log": multi_env.lightning_mu_log,
            "lightning_sigma_log": multi_env.lightning_sigma_log,
        }
        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Run fire suppression/sharing simulations.")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="multi",
        help="single: one jurisdiction, suppression only. multi: multiple jurisdictions with sharing.",
    )
    parser.add_argument(
        "--sharing-algorithm",
        default="none",
        help=f"Sharing algorithm (multi mode only). Available: {sorted(SHARING_ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--suppression-algorithm",
        default="greedy",
        help=f"Suppression algorithm. Available: {sorted(SUPPRESSION_ALGORITHM_REGISTRY.keys())}",
    )
    parser.add_argument("--sharing-param-dir", default="")
    parser.add_argument("--suppression-param-dir", default="")
    parser.add_argument("--period-s", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lightning-seed", type=int, default=0)
    parser.add_argument("--spread-seed", type=int, default=None)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--run-label", default="cli")
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

    # Aliases for single mode
    parser.add_argument("--rows", type=int, default=None, help="Alias for --per-juris-rows (single mode).")
    parser.add_argument("--cols", type=int, default=None, help="Alias for --per-juris-cols (single mode).")
    parser.add_argument("--num-units", type=int, default=None, help="Alias for --num-units-per-juris (single mode).")

    args = parser.parse_args()

    # Apply single-mode aliases
    per_juris_rows = args.rows if args.rows is not None else args.per_juris_rows
    per_juris_cols = args.cols if args.cols is not None else args.per_juris_cols
    num_units_per_juris = args.num_units if args.num_units is not None else args.num_units_per_juris

    sharing_params: dict = {}
    suppression_params: dict = {}

    if args.period_s is not None:
        sharing_params["period_s"] = args.period_s
    sharing_params["total_steps"] = args.steps

    if args.mode == "single":
        run_single(
            suppression_algorithm_name=args.suppression_algorithm,
            suppression_param_dir=args.suppression_param_dir,
            suppression_params=suppression_params,
            steps=args.steps,
            lightning_seed=args.lightning_seed,
            spread_seed=args.spread_seed,
            save_snapshots=args.save_snapshots,
            output_dir=args.output_dir,
            verbose=args.verbose,
            run_label=args.run_label,
            per_juris_rows=per_juris_rows,
            per_juris_cols=per_juris_cols,
            base_spread_prob=args.base_spread_prob,
            num_units_per_juris=num_units_per_juris,
            suppression_success_prob=args.suppression_success_prob,
            movement_per_step=args.movement_per_step,
            lightning_mu_log=args.lightning_mu_log,
            lightning_sigma_log=args.lightning_sigma_log,
        )
    else:
        run_multi(
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
            per_juris_rows=per_juris_rows,
            per_juris_cols=per_juris_cols,
            base_spread_prob=args.base_spread_prob,
            num_units_per_juris=num_units_per_juris,
            suppression_success_prob=args.suppression_success_prob,
            movement_per_step=args.movement_per_step,
            juris_travel_time=args.juris_travel_time,
            lightning_mu_log=args.lightning_mu_log,
            lightning_sigma_log=args.lightning_sigma_log,
        )


if __name__ == "__main__":
    main()
