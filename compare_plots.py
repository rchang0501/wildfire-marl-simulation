"""Plotting functions for algorithm comparison results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _get_combo_styles(combo_keys: list[str]) -> dict[str, dict]:
    """Assign consistent colors to each combo."""
    cmap = plt.cm.tab10
    styles = {}
    for i, key in enumerate(combo_keys):
        styles[key] = {"color": cmap(i % 10), "label": key}
    return styles


def _plot_timeseries(
    results: dict,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    extract_fn=None,
):
    """Plot a time-series metric with mean +/- 1 std shaded band per combo.

    extract_fn: optional callable (seed_data) -> list[float] to extract the
    series from a seed's data. Defaults to seed_data["metrics"][metric_key].
    """
    combos = results["combos"]
    combo_keys = list(combos.keys())
    styles = _get_combo_styles(combo_keys)

    fig, ax = plt.subplots(figsize=(10, 5))

    for combo_key in combo_keys:
        combo_data = combos[combo_key]
        all_series = []
        for seed_data in combo_data["seeds"].values():
            if extract_fn is not None:
                series = extract_fn(seed_data)
            else:
                series = seed_data["metrics"][metric_key]
            all_series.append(series)

        arr = np.array(all_series, dtype=float)  # (num_seeds, num_steps+1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = np.arange(len(mean))

        style = styles[combo_key]
        ax.plot(steps, mean, color=style["color"], label=style["label"], linewidth=1.5)
        ax.fill_between(steps, mean - std, mean + std, color=style["color"], alpha=0.15)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_total_burning(results: dict, output_path: Path) -> None:
    _plot_timeseries(
        results,
        metric_key="total_burning",
        title="Total Burning Cells Over Time",
        ylabel="Total Burning Cells",
        output_path=output_path,
    )


def plot_burning_gini(results: dict, output_path: Path) -> None:
    _plot_timeseries(
        results,
        metric_key=None,
        title="Fire Inequality (Gini Coefficient) Over Time",
        ylabel="Gini Coefficient",
        output_path=output_path,
        extract_fn=lambda sd: sd["derived"]["burning_gini"],
    )


def plot_units_in_transit(results: dict, output_path: Path) -> None:
    _plot_timeseries(
        results,
        metric_key="units_in_transit",
        title="Units In Transit Over Time",
        ylabel="Units In Transit",
        output_path=output_path,
    )


def plot_burning_per_juris(results: dict, output_path: Path) -> None:
    """Grid of subplots (one per jurisdiction), one line per combo."""
    combos = results["combos"]
    combo_keys = list(combos.keys())
    styles = _get_combo_styles(combo_keys)

    # Determine number of jurisdictions from first combo's first seed
    first_combo = combos[combo_keys[0]]
    first_seed = next(iter(first_combo["seeds"].values()))
    num_juris = len(first_seed["metrics"]["burning_per_juris"][0])

    ncols = min(num_juris, 4)
    nrows = (num_juris + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for j_idx in range(num_juris):
        row, col = divmod(j_idx, ncols)
        ax = axes[row][col]

        for combo_key in combo_keys:
            combo_data = combos[combo_key]
            all_series = []
            for seed_data in combo_data["seeds"].values():
                bpj = seed_data["metrics"]["burning_per_juris"]
                series = [step_vals[j_idx] for step_vals in bpj]
                all_series.append(series)

            arr = np.array(all_series, dtype=float)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            steps = np.arange(len(mean))

            style = styles[combo_key]
            ax.plot(steps, mean, color=style["color"], label=style["label"], linewidth=1)
            ax.fill_between(steps, mean - std, mean + std, color=style["color"], alpha=0.1)

        ax.set_title(f"Jurisdiction {j_idx}", fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        ax.set_ylabel("Burning", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_juris, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(combo_keys), 4), fontsize=8)
    fig.suptitle("Burning Per Jurisdiction Over Time", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bars(results: dict, metric_key: str, title: str, ylabel: str, output_path: Path) -> None:
    """Grouped bar chart for a scalar summary metric with error bars."""
    combos = results["combos"]
    combo_keys = list(combos.keys())
    styles = _get_combo_styles(combo_keys)

    means = []
    stds = []
    for combo_key in combo_keys:
        agg = combos[combo_key]["aggregate"][metric_key]
        means.append(agg["mean"])
        stds.append(agg["std"])

    fig, ax = plt.subplots(figsize=(max(6, len(combo_keys) * 1.5), 5))
    x = np.arange(len(combo_keys))
    colors = [styles[k]["color"] for k in combo_keys]

    ax.bar(x, means, yerr=stds, color=colors, capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(combo_keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results: dict, plots_dir: Path) -> None:
    """Generate all comparison plots and save to plots_dir."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Time-series plots
    plot_total_burning(results, plots_dir / "timeseries_total_burning.png")
    plot_burning_gini(results, plots_dir / "timeseries_burning_gini.png")
    plot_units_in_transit(results, plots_dir / "timeseries_units_in_transit.png")
    plot_burning_per_juris(results, plots_dir / "timeseries_burning_per_juris.png")

    # Summary bar charts
    summary_plots = [
        ("final_total_burning", "Final Total Burning", "Burning Cells"),
        ("cumulative_fire_steps", "Cumulative Fire-Steps", "Cumulative Burning"),
        ("mean_burning_gini", "Mean Burning Gini", "Gini Coefficient"),
        ("peak_total_burning", "Peak Total Burning", "Burning Cells"),
    ]
    for metric_key, title, ylabel in summary_plots:
        plot_summary_bars(results, metric_key, title, ylabel, plots_dir / f"summary_{metric_key}.png")
