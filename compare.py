"""Algorithm comparison tool.

Runs the cartesian product of suppression x sharing algorithms across
multiple random seeds, collects per-step metrics, computes summary
statistics, and generates comparison plots.

Usage:
    python compare.py
    python compare.py --suppression greedy lp_suppression --sharing none periodic_transfer --seeds 0 1 2 3 4
    python compare.py --steps 50 --num-seeds 3
    python compare.py --num-juris-rows 3 --num-juris-cols 3 --label "3x3_experiment"
"""

import argparse
import json
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from algorithms.sharing_algorithms import SHARING_ALGORITHM_REGISTRY
from algorithms.suppression_algorithms import SUPPRESSION_ALGORITHM_REGISTRY
from main import DEFAULTS, run_multi


# ------------------------------------------------------------------
# Derived metrics
# ------------------------------------------------------------------

def gini_coefficient(values: list[float]) -> float:
    """Compute the Gini coefficient for a list of non-negative values.

    Returns 0 for perfectly equal distributions, approaches 1 for
    maximally unequal distributions. Returns 0 for all-zero inputs.
    """
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    arr_sorted = np.sort(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr_sorted) - (n + 1) * np.sum(arr_sorted)) / (n * np.sum(arr_sorted)))


def compute_derived_timeseries(metrics: dict) -> dict:
    """Compute derived time-series from raw per-step metrics."""
    total_burning = metrics["total_burning"]
    burning_per_juris = metrics["burning_per_juris"]

    cumulative_burning = list(np.cumsum(total_burning).tolist())
    burning_gini = [gini_coefficient(bpj) for bpj in burning_per_juris]
    burning_cv = []
    for bpj in burning_per_juris:
        arr = np.array(bpj, dtype=float)
        mean = arr.mean()
        if mean == 0:
            burning_cv.append(0.0)
        else:
            burning_cv.append(float(arr.std() / mean))

    return {
        "cumulative_burning": cumulative_burning,
        "burning_gini": burning_gini,
        "burning_cv": burning_cv,
    }


def compute_summary_statistics(metrics: dict, derived: dict) -> dict:
    """Compute scalar summary statistics from one seed-run."""
    total_burning = metrics["total_burning"]
    burning_per_juris = metrics["burning_per_juris"]
    burning_gini = derived["burning_gini"]
    burning_cv = derived["burning_cv"]

    burning_per_juris_arr = np.array(burning_per_juris)  # (steps+1, num_juris)

    return {
        "final_total_burning": total_burning[-1],
        "peak_total_burning": int(max(total_burning)),
        "mean_total_burning": float(np.mean(total_burning)),
        "cumulative_fire_steps": int(sum(total_burning)),
        "final_burning_per_juris": burning_per_juris[-1],
        "peak_burning_per_juris": burning_per_juris_arr.max(axis=0).tolist(),
        "mean_burning_gini": float(np.mean(burning_gini)),
        "peak_burning_gini": float(max(burning_gini)),
        "mean_burning_cv": float(np.mean(burning_cv)),
    }


def aggregate_across_seeds(seeds_data: list[dict]) -> dict:
    """Compute mean and std of scalar summaries across seeds.

    Per-juris lists are aggregated element-wise.
    """
    if not seeds_data:
        return {}

    scalar_keys = [
        "final_total_burning", "peak_total_burning", "mean_total_burning",
        "cumulative_fire_steps", "mean_burning_gini", "peak_burning_gini",
        "mean_burning_cv",
    ]
    list_keys = ["final_burning_per_juris", "peak_burning_per_juris"]

    result = {}
    for key in scalar_keys:
        vals = [d[key] for d in seeds_data]
        result[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    for key in list_keys:
        arr = np.array([d[key] for d in seeds_data])  # (num_seeds, num_juris)
        result[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
        }

    return result


# ------------------------------------------------------------------
# Comparison runner
# ------------------------------------------------------------------

def run_comparison(
    suppression_names: list[str],
    sharing_names: list[str],
    seeds: list[int],
    steps: int,
    period_s: int | None,
    env_params: dict,
    verbose: bool,
    suppression_param_dir: str = "",
) -> dict:
    """Run all algorithm combos x seeds and return structured results."""
    combos = list(product(suppression_names, sharing_names))
    results = {
        "config": {
            "suppression_algorithms": suppression_names,
            "sharing_algorithms": sharing_names,
            "seeds": seeds,
            "steps": steps,
            "period_s": period_s,
            "env_params": env_params,
        },
        "combos": {},
    }

    total_runs = len(combos) * len(seeds)
    run_idx = 0

    for supp_name, share_name in combos:
        combo_key = f"{supp_name} + {share_name}"
        combo_data = {"seeds": {}}

        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {combo_key}, seed={seed}")

            sharing_params: dict = {}
            if period_s is not None:
                sharing_params["period_s"] = period_s
            sharing_params["total_steps"] = steps

            metrics = run_multi(
                sharing_algorithm_name=share_name,
                suppression_algorithm_name=supp_name,
                sharing_param_dir="",
                suppression_param_dir=suppression_param_dir,
                sharing_params=sharing_params,
                suppression_params={},
                steps=steps,
                lightning_seed=seed,
                spread_seed=None,
                save_snapshots=False,
                output_dir="",
                verbose=verbose,
                run_label=f"{combo_key}_seed{seed}",
                num_juris_rows=env_params["num_juris_rows"],
                num_juris_cols=env_params["num_juris_cols"],
                per_juris_rows=env_params["per_juris_rows"],
                per_juris_cols=env_params["per_juris_cols"],
                base_spread_prob=env_params["base_spread_prob"],
                num_units_per_juris=env_params["num_units_per_juris"],
                suppression_success_prob=env_params["suppression_success_prob"],
                movement_per_step=env_params["movement_per_step"],
                juris_travel_time=env_params["juris_travel_time"],
                lightning_mu_log=env_params["lightning_mu_log"],
                lightning_sigma_log=env_params["lightning_sigma_log"],
                max_fuel=env_params.get("max_fuel"),
                fuel_refuel_rate=env_params.get("fuel_refuel_rate", 1),
            )

            derived = compute_derived_timeseries(metrics)
            summary = compute_summary_statistics(metrics, derived)

            combo_data["seeds"][str(seed)] = {
                "metrics": metrics,
                "derived": derived,
                "summary": summary,
            }

        # Aggregate across seeds
        all_summaries = [combo_data["seeds"][str(s)]["summary"] for s in seeds]
        combo_data["aggregate"] = aggregate_across_seeds(all_summaries)

        results["combos"][combo_key] = combo_data

    return results


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def save_results(results: dict, output_dir: Path) -> None:
    """Write config, summary, and raw metrics to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # config.json
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(results["config"], indent=2), encoding="utf-8")

    # summary.json (aggregate stats only, compact)
    summary = {}
    for combo_key, combo_data in results["combos"].items():
        summary[combo_key] = combo_data["aggregate"]
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # raw_metrics.json (full per-step data for all seeds)
    raw = {}
    for combo_key, combo_data in results["combos"].items():
        raw[combo_key] = {}
        for seed_key, seed_data in combo_data["seeds"].items():
            raw[combo_key][seed_key] = {
                "metrics": seed_data["metrics"],
                "derived": seed_data["derived"],
                "summary": seed_data["summary"],
            }
    raw_path = output_dir / "raw_metrics.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    print(f"Results saved to {output_dir}/")


def print_summary_table(results: dict) -> None:
    """Print a formatted ASCII summary table to stdout."""
    combos = results["combos"]
    if not combos:
        return

    headers = [
        "Algorithm Combo",
        "Final Burn",
        "Peak Burn",
        "Mean Burn",
        "Cum. Fire",
        "Mean Gini",
        "Mean CV",
    ]
    rows = []
    for combo_key, combo_data in combos.items():
        agg = combo_data["aggregate"]
        rows.append([
            combo_key,
            f"{agg['final_total_burning']['mean']:.1f} +/- {agg['final_total_burning']['std']:.1f}",
            f"{agg['peak_total_burning']['mean']:.1f} +/- {agg['peak_total_burning']['std']:.1f}",
            f"{agg['mean_total_burning']['mean']:.1f} +/- {agg['mean_total_burning']['std']:.1f}",
            f"{agg['cumulative_fire_steps']['mean']:.0f} +/- {agg['cumulative_fire_steps']['std']:.0f}",
            f"{agg['mean_burning_gini']['mean']:.3f} +/- {agg['mean_burning_gini']['std']:.3f}",
            f"{agg['mean_burning_cv']['mean']:.3f} +/- {agg['mean_burning_cv']['std']:.3f}",
        ])

    # Compute column widths
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    # Print
    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * w for w in col_widths)

    print()
    print("=" * len(header_line))
    print("COMPARISON SUMMARY")
    print("=" * len(header_line))
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))))
    print()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare suppression x sharing algorithm combinations across seeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Algorithm selection
    all_supp = sorted(SUPPRESSION_ALGORITHM_REGISTRY.keys())
    all_share = sorted(SHARING_ALGORITHM_REGISTRY.keys())
    parser.add_argument(
        "--suppression", nargs="+", default=all_supp, metavar="ALG",
        help=f"Suppression algorithms to compare. Available: {all_supp}",
    )
    parser.add_argument(
        "--sharing", nargs="+", default=all_share, metavar="ALG",
        help=f"Sharing algorithms to compare. Available: {all_share}",
    )

    # Seeds
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Explicit seed list (e.g., 0 1 2 3 4).",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds (0..N-1). Ignored if --seeds is set.",
    )

    # Simulation
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--period-s", type=int, default=None)

    # Environment params
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
    parser.add_argument("--max-fuel", type=int, default=None, help="Max fuel per unit (None = unlimited).")
    parser.add_argument("--fuel-refuel-rate", type=int, default=1, help="Fuel gained per step at base.")

    # Algorithm param dirs
    parser.add_argument("--suppression-param-dir", default="",
                        help="Param directory for suppression algorithm (e.g. trained_models/rl_v3 for rl).")

    # Output
    parser.add_argument("--output-dir", default="comparisons")
    parser.add_argument("--label", default="comparison")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    # Validate algorithms
    for name in args.suppression:
        if name not in SUPPRESSION_ALGORITHM_REGISTRY:
            print(f"Error: unknown suppression algorithm '{name}'. Available: {all_supp}", file=sys.stderr)
            sys.exit(1)
    for name in args.sharing:
        if name not in SHARING_ALGORITHM_REGISTRY:
            print(f"Error: unknown sharing algorithm '{name}'. Available: {all_share}", file=sys.stderr)
            sys.exit(1)

    seeds = args.seeds if args.seeds is not None else list(range(args.num_seeds))

    env_params = {
        "num_juris_rows": args.num_juris_rows,
        "num_juris_cols": args.num_juris_cols,
        "per_juris_rows": args.per_juris_rows,
        "per_juris_cols": args.per_juris_cols,
        "base_spread_prob": args.base_spread_prob,
        "num_units_per_juris": args.num_units_per_juris,
        "suppression_success_prob": args.suppression_success_prob,
        "movement_per_step": args.movement_per_step,
        "juris_travel_time": args.juris_travel_time,
        "lightning_mu_log": args.lightning_mu_log,
        "lightning_sigma_log": args.lightning_sigma_log,
        "max_fuel": args.max_fuel,
        "fuel_refuel_rate": args.fuel_refuel_rate,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{timestamp}_{args.label}"

    print(f"Comparing: {args.suppression} x {args.sharing}")
    print(f"Seeds: {seeds}, Steps: {args.steps}")
    print(f"Output: {output_dir}")
    print()

    results = run_comparison(
        suppression_names=args.suppression,
        sharing_names=args.sharing,
        seeds=seeds,
        steps=args.steps,
        period_s=args.period_s,
        env_params=env_params,
        verbose=args.verbose,
        suppression_param_dir=args.suppression_param_dir,
    )

    save_results(results, output_dir)
    print_summary_table(results)

    if not args.no_plots:
        try:
            from compare_plots import generate_all_plots
            plots_dir = output_dir / "plots"
            generate_all_plots(results, plots_dir)
            print(f"Plots saved to {plots_dir}/")
        except ImportError:
            print("Warning: matplotlib not available, skipping plots.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: plot generation failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
