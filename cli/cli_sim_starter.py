"""
CLI tool for running pond_spawn simulations.

Usage:
    python -m cli.cli_sim_starter [options]
"""

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path

import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import numpy as np

from simulation import Environment

_GENOME_PATH = Path(__file__).resolve().parent.parent / "genomes" / "genome.json"


@contextmanager
def _suppress_stdout():
    """Silence prints inside worker processes (e.g. torch device banner)."""
    old = sys.stdout
    sys.stdout = StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _print_progress(step, total):
    """Print an in-place progress bar showing tick count only."""
    bar_width = 30
    filled = int(bar_width * step / total)
    progress_bar = "█" * filled + "─" * (bar_width - filled)
    print(f"\r  [{progress_bar}] {step}/{total}", end="", flush=True)


# ---------------------------------------------------------------------------
# Module-level worker — must be importable by name for multiprocessing pickle
# ---------------------------------------------------------------------------


def _run_worker(task):
    """
    Run a single simulation in a worker process.

    Args:
        task (tuple): (args_dict, seed, run_idx, queue_or_none)
            queue_or_none: multiprocessing.Queue for progress updates, or None

    Returns:
        dict: logged_stats, final_stats, avg_traits, initial_population, run_idx
    """
    args_dict, seed, run_idx, queue = task
    random.seed(seed)
    np.random.seed(seed % (2**31))

    total_steps = args_dict["steps"]
    # Throttle queue writes: ~100 updates per run regardless of step count
    report_every = max(1, total_steps // 100)

    with _suppress_stdout():
        env = Environment(
            grid_size=args_dict["grid_size"],
            num_agents=args_dict["population"],
        )
    logged_stats = {}
    for step_num in range(total_steps):
        env.step()
        stats = env.get_stats()
        logged_stats = env.log_stats(stats, logged_stats)
        if queue is not None and step_num % report_every == 0:
            queue.put((run_idx, stats["step"], total_steps, False))
        if stats["alive_agents"] == 0:
            break

    final_stats = env.get_stats()
    avg_traits = env.get_average_genome_traits()
    if queue is not None:
        queue.put((run_idx, final_stats["step"], total_steps, True))  # sentinel: done
    return {
        "run_idx": run_idx,
        "logged_stats": logged_stats,
        "final_stats": final_stats,
        "avg_traits": avg_traits,
        "initial_population": args_dict["population"],
    }


# ---------------------------------------------------------------------------
# Shared plot/table helpers
# ---------------------------------------------------------------------------


def _style_ax(ax, title, ylabel, xlabel="Step"):
    """Apply consistent style to a plot axes."""
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_table(ax, col_labels, rows, title=""):
    """
    Render a data table on *ax* (axes is turned off; table fills it).

    Args:
        ax: matplotlib Axes
        col_labels (list[str]): Column headers
        rows (list[list[str]]): Cell data rows
        title (str): Optional bold title rendered above the table
    """
    ax.axis("off")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=11, pad=6)

    if not rows:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            color="#888",
        )
        return

    n_cols = len(col_labels)

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(list(range(n_cols)))

    # Header row styling
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Data row styling — alternate bands
    for i in range(len(rows)):
        bg = "#f0f4f8" if i % 2 == 0 else "#ffffff"
        for j in range(n_cols):
            table[i + 1, j].set_facecolor(bg)
            table[i + 1, j].set_text_props(color="#1a1a1a")


def _load_genome_bounds():
    """Return {trait_name: {min, max}} from genome.json, or {} on failure."""
    try:
        with open(_GENOME_PATH, encoding="utf-8") as f:
            base = json.load(f)
        return base.get("traits", {})
    except Exception:  # pylint: disable=broad-except
        return {}


# ---------------------------------------------------------------------------
# Single-run report
# ---------------------------------------------------------------------------


# pylint: disable=too-many-locals,too-many-statements
def plot_simulation_stats(logged_stats, initial_population, avg_traits=None, death_stats=None):
    """
    Create a single-run simulation report.

    Args:
        logged_stats (dict): step -> stats dict
        initial_population (int): Initial agent population
        avg_traits (dict): Average genome traits of final population
        death_stats (dict): Final death breakdown by cause from get_stats()
    """
    if not logged_stats:
        print("No stats to plot.")
        return

    steps = sorted(logged_stats.keys())
    alive_agents = [logged_stats[s]["alive_agents"] for s in steps]
    total_food = [logged_stats[s]["total_food"] for s in steps]
    avg_energy = [logged_stats[s]["avg_energy"] for s in steps]
    median_lifespan = [logged_stats[s]["median_lifespan"] for s in steps]
    min_age = [logged_stats[s]["min_age"] for s in steps]
    max_age = [logged_stats[s]["max_age"] for s in steps]
    deaths_starvation = [logged_stats[s].get("deaths_starvation", 0) for s in steps]
    deaths_combat = [logged_stats[s].get("deaths_combat", 0) for s in steps]
    deaths_old_age = [logged_stats[s].get("deaths_old_age", 0) for s in steps]

    fig = plt.figure(figsize=(22, 28))
    fig.suptitle("Pond Spawn  ·  Simulation Report", fontsize=18, fontweight="bold", y=0.99)

    outer = mgridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[3.5, 1.1],
        top=0.97,
        bottom=0.01,
        hspace=0.06,
    )

    # Top: plots (left 2/3) + tables (right 1/3)
    top = mgridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[2, 1],
        wspace=0.35,
    )
    plot_gs = mgridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=top[0], hspace=0.48)
    ax_pop = fig.add_subplot(plot_gs[0])
    ax_food = fig.add_subplot(plot_gs[1])
    ax_energy = fig.add_subplot(plot_gs[2])
    ax_lifespan = fig.add_subplot(plot_gs[3])
    ax_deaths = fig.add_subplot(plot_gs[4])

    tbl_gs = mgridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=top[1], hspace=0.55)
    ax_summary_tbl = fig.add_subplot(tbl_gs[0])
    ax_death_tbl = fig.add_subplot(tbl_gs[1])
    ax_key = fig.add_subplot(tbl_gs[2])

    # Bottom: genome table full-width
    ax_genome = fig.add_subplot(outer[1])

    # --- Plots ---
    ax_pop.plot(steps, alive_agents, color="steelblue", linewidth=2)
    ax_pop.axhline(
        initial_population,
        color="steelblue",
        linestyle="--",
        linewidth=1,
        alpha=0.4,
        label=f"Start: {initial_population}",
    )
    ax_pop.legend(fontsize=9)
    _style_ax(ax_pop, "Population", "Agents")

    ax_food.plot(steps, total_food, color="forestgreen", linewidth=2)
    ax_food.fill_between(steps, total_food, alpha=0.12, color="forestgreen")
    _style_ax(ax_food, "Food Supply", "Units")

    ax_energy.plot(steps, avg_energy, color="darkorange", linewidth=2)
    _style_ax(ax_energy, "Average Agent Energy", "Energy")

    ax_lifespan.plot(steps, median_lifespan, color="mediumpurple", linewidth=2, label="Median")
    ax_lifespan.fill_between(
        steps, min_age, max_age, color="mediumpurple", alpha=0.15, label="Min–Max band"
    )
    ax_lifespan.legend(fontsize=9)
    _style_ax(ax_lifespan, "Lifespan", "Age")

    death_series = [
        (deaths_starvation, "crimson", "Starvation"),
        (deaths_combat, "royalblue", "Combat"),
        (deaths_old_age, "seagreen", "Old Age"),
    ]
    for series, color, label in death_series:
        ax_deaths.plot(steps, series, color=color, linewidth=2, label=label)
        if series and series[-1] > 0:
            ax_deaths.annotate(
                str(series[-1]),
                xy=(steps[-1], series[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color=color,
                va="center",
            )
    ax_deaths.legend(fontsize=9)
    _style_ax(ax_deaths, "Cumulative Deaths by Cause", "Total Deaths")

    # --- Summary stats table ---
    final_alive = alive_agents[-1] if alive_agents else 0
    final_food = total_food[-1] if total_food else 0
    final_energy = avg_energy[-1] if avg_energy else 0
    final_med_ls = median_lifespan[-1] if median_lifespan else 0
    final_min_age = min_age[-1] if min_age else 0
    final_max_age = max_age[-1] if max_age else 0
    final_step = steps[-1] if steps else 0
    surv_rate = 100.0 * final_alive / initial_population if initial_population else 0.0

    summary_rows = [
        ["Initial Population", str(initial_population)],
        ["Survivors", str(final_alive)],
        ["Survival Rate", f"{surv_rate:.1f}%"],
        ["Steps Run", str(final_step)],
        ["Food Remaining", f"{final_food:.0f}"],
        ["Avg Energy (final)", f"{final_energy:.1f}"],
        ["Median Lifespan", f"{final_med_ls:.0f}"],
        ["Min Age (final pop)", str(final_min_age)],
        ["Max Age (final pop)", str(final_max_age)],
    ]
    _draw_table(ax_summary_tbl, ["Metric", "Value"], summary_rows, title="Run Summary")

    # --- Cause-of-death table ---
    if death_stats:
        total_deaths = sum(d["count"] for d in death_stats.values())
        death_rows = []
        for cause, data in death_stats.items():
            pct = f"{100.0 * data['count'] / total_deaths:.1f}%" if total_deaths else "0%"
            death_rows.append([cause, str(data["count"]), pct, f"{data['avg_age']:.0f}"])
        death_rows.append(["Total", str(total_deaths), "100%", "—"])
        _draw_table(
            ax_death_tbl, ["Cause", "Deaths", "%", "Avg Age"], death_rows, title="Cause of Death"
        )
    else:
        ax_death_tbl.axis("off")

    # --- Plot key ---
    ax_key.axis("off")
    ax_key.set_title("Plot Key", fontweight="bold", fontsize=11)
    key_items = [
        ("steelblue", "─", "Population over time"),
        ("forestgreen", "─", "Food supply over time"),
        ("darkorange", "─", "Average agent energy"),
        ("mediumpurple", "─", "Median lifespan"),
        ("mediumpurple", "░", "Lifespan min–max band"),
        ("crimson", "─", "Starvation deaths (cumul.)"),
        ("royalblue", "─", "Combat deaths (cumul.)"),
        ("seagreen", "─", "Old age deaths (cumul.)"),
    ]
    y = 0.93
    for color, marker, desc in key_items:
        ls = "--" if marker == "░" else "-"
        ax_key.plot(
            [0.02, 0.10], [y, y], color=color, linewidth=2, linestyle=ls, transform=ax_key.transAxes
        )
        ax_key.text(
            0.13, y, desc, transform=ax_key.transAxes, fontsize=8.5, va="center", color="#222"
        )
        y -= 0.11

    # --- Genome table (full-width, bottom) ---
    genome_bounds = _load_genome_bounds()
    if avg_traits:
        genome_rows = []
        for trait, value in avg_traits.items():
            info = genome_bounds.get(trait, {})
            lo = f"{info['min']:.3f}" if "min" in info else "—"
            hi = f"{info['max']:.3f}" if "max" in info else "—"
            genome_rows.append([trait, f"{value:.4f}", lo, hi])
        _draw_table(
            ax_genome,
            ["Trait", "Survivors' Avg Value", "Genome Min", "Genome Max"],
            genome_rows,
            title="Survivors' Average Genome Traits",
        )
    else:
        ax_genome.axis("off")
        ax_genome.text(
            0.5,
            0.5,
            "No survivors — genome data unavailable",
            ha="center",
            va="center",
            transform=ax_genome.transAxes,
            fontsize=11,
            color="#888",
        )

    _save_figure(fig, "simulation_report")


# ---------------------------------------------------------------------------
# Meta-run report
# ---------------------------------------------------------------------------


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def plot_meta_stats(all_results, args):
    """
    Generate an overlay report across multiple runs.

    Args:
        all_results (list[dict]): Results from _run_worker, one per run
        args: Parsed CLI args (for labels/params)
    """
    if not all_results:
        print("No results to plot.")
        return

    n_runs = len(all_results)
    all_steps_sets = [set(r["logged_stats"].keys()) for r in all_results]
    max_step = max(max(s) for s in all_steps_sets if s)
    all_steps = list(range(1, max_step + 1))

    def run_series(key, run):
        ls = run["logged_stats"]
        vals, valid_s = [], []
        for s in all_steps:
            if s in ls:
                vals.append(ls[s][key])
                valid_s.append(s)
        return valid_s, vals

    def avg_series(key):
        result_steps, result_vals = [], []
        for s in all_steps:
            pts = [r["logged_stats"][s][key] for r in all_results if s in r["logged_stats"]]
            if pts:
                result_steps.append(s)
                result_vals.append(np.mean(pts))
        return result_steps, result_vals

    def avg_death_series(key):
        result_steps, result_vals = [], []
        for s in all_steps:
            pts = [r["logged_stats"][s].get(key, 0) for r in all_results if s in r["logged_stats"]]
            if pts:
                result_steps.append(s)
                result_vals.append(np.mean(pts))
        return result_steps, result_vals

    palette = plt.get_cmap("tab10").colors

    fig = plt.figure(figsize=(22, 34))
    fig.suptitle(
        f"Pond Spawn  ·  Meta-Run Report  ·  {n_runs} runs  ·  "
        f"{args.population} agents  ·  {args.grid_size}×{args.grid_size} grid",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )

    outer = mgridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[3.2, 0.85, 1.1],
        top=0.97,
        bottom=0.01,
        hspace=0.07,
    )

    top = mgridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer[0],
        width_ratios=[2, 1],
        wspace=0.35,
    )
    plot_gs = mgridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=top[0], hspace=0.48)
    ax_pop = fig.add_subplot(plot_gs[0])
    ax_food = fig.add_subplot(plot_gs[1])
    ax_energy = fig.add_subplot(plot_gs[2])
    ax_lifespan = fig.add_subplot(plot_gs[3])
    ax_deaths = fig.add_subplot(plot_gs[4])

    tbl_gs = mgridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=top[1], hspace=0.45)
    ax_config = fig.add_subplot(tbl_gs[0])
    ax_key = fig.add_subplot(tbl_gs[1])

    ax_per_run = fig.add_subplot(outer[1])
    ax_genome = fig.add_subplot(outer[2])

    # --- Overlay plots ---
    def _overlay(ax, key, avg_color, title, ylabel):
        for i, run in enumerate(all_results):
            sx, sy = run_series(key, run)
            if sx:
                ax.plot(sx, sy, color=palette[i % len(palette)], linewidth=1.0, alpha=0.35)
        avx, avy = avg_series(key)
        if avx:
            ax.plot(avx, avy, color=avg_color, linewidth=2.5, label="Average", zorder=5)
        _style_ax(ax, title, ylabel)
        ax.legend(fontsize=9)

    _overlay(ax_pop, "alive_agents", "steelblue", "Population", "Agents")
    _overlay(ax_food, "total_food", "forestgreen", "Food Supply", "Units")
    _overlay(ax_energy, "avg_energy", "darkorange", "Average Agent Energy", "Energy")
    _overlay(ax_lifespan, "median_lifespan", "mediumpurple", "Median Lifespan", "Age")

    death_cfg = [
        ("deaths_starvation", "crimson", "Starvation"),
        ("deaths_combat", "royalblue", "Combat"),
        ("deaths_old_age", "seagreen", "Old Age"),
    ]
    for key, avg_color, label in death_cfg:
        for run in all_results:
            sx, sy = [], []
            for s in all_steps:
                if s in run["logged_stats"]:
                    sx.append(s)
                    sy.append(run["logged_stats"][s].get(key, 0))
            if sx:
                ax_deaths.plot(sx, sy, color=avg_color, linewidth=0.8, alpha=0.22)
        avx, avy = avg_death_series(key)
        if avx:
            ax_deaths.plot(
                avx, avy, color=avg_color, linewidth=2.5, label=f"{label} (avg)", zorder=5
            )
    ax_deaths.legend(fontsize=9)
    _style_ax(ax_deaths, "Cumulative Deaths by Cause — Average", "Total Deaths")

    # --- Config table ---
    config_rows = [
        ["Grid size", f"{args.grid_size}×{args.grid_size}"],
        ["Initial pop", str(args.population)],
        ["Steps", str(args.steps)],
        ["Runs", str(n_runs)],
        ["Workers", str(args.workers)],
    ]
    _draw_table(ax_config, ["Parameter", "Value"], config_rows, title="Run Configuration")

    # --- Plot key ---
    ax_key.axis("off")
    ax_key.set_title("Plot Key", fontweight="bold", fontsize=11)
    ax_key.text(
        0.0,
        1.00,
        "Bold line = average across all runs",
        transform=ax_key.transAxes,
        fontsize=8,
        color="#555",
        style="italic",
    )
    ax_key.text(
        0.0,
        0.95,
        "Faint lines = individual runs",
        transform=ax_key.transAxes,
        fontsize=8,
        color="#555",
        style="italic",
    )
    key_items = [
        ("steelblue", "Population"),
        ("forestgreen", "Food Supply"),
        ("darkorange", "Avg Energy"),
        ("mediumpurple", "Median Lifespan"),
        ("crimson", "Starvation Deaths"),
        ("royalblue", "Combat Deaths"),
        ("seagreen", "Old Age Deaths"),
    ]
    y = 0.87
    for color, desc in key_items:
        ax_key.plot([0.02, 0.10], [y, y], color=color, linewidth=2, transform=ax_key.transAxes)
        ax_key.text(
            0.13, y, desc, transform=ax_key.transAxes, fontsize=8.5, va="center", color="#222"
        )
        y -= 0.11

    # --- Per-run summary table (full-width middle) ---
    run_rows = []
    for run in sorted(all_results, key=lambda r: r["run_idx"]):
        fs = run["final_stats"]
        ds = fs.get("death_stats", {})
        starv = ds.get("Starvation", {}).get("count", 0)
        combat = ds.get("Combat", {}).get("count", 0)
        age = ds.get("Old Age", {}).get("count", 0)
        run_rows.append(
            [
                str(run["run_idx"] + 1),
                str(run["initial_population"]),
                str(fs["alive_agents"]),
                f"{100.0 * fs['alive_agents'] / run['initial_population']:.1f}%",
                str(fs["step"]),
                f"{fs['avg_energy']:.1f}",
                f"{fs['median_lifespan']:.0f}",
                str(starv),
                str(combat),
                str(age),
                "Yes" if fs["alive_agents"] == 0 else "No",
            ]
        )

    # Average / totals row
    avg_surv = np.mean([r["final_stats"]["alive_agents"] for r in all_results])
    avg_steps = np.mean([r["final_stats"]["step"] for r in all_results])
    avg_enrg = np.mean([r["final_stats"]["avg_energy"] for r in all_results])
    avg_mls = np.mean([r["final_stats"]["median_lifespan"] for r in all_results])
    avg_starv = np.mean(
        [
            r["final_stats"].get("death_stats", {}).get("Starvation", {}).get("count", 0)
            for r in all_results
        ]
    )
    avg_combat = np.mean(
        [
            r["final_stats"].get("death_stats", {}).get("Combat", {}).get("count", 0)
            for r in all_results
        ]
    )
    avg_age = np.mean(
        [
            r["final_stats"].get("death_stats", {}).get("Old Age", {}).get("count", 0)
            for r in all_results
        ]
    )
    extinct_n = sum(1 for r in all_results if r["final_stats"]["alive_agents"] == 0)
    run_rows.append(
        [
            "AVG",
            str(args.population),
            f"{avg_surv:.1f}",
            f"{100.0 * avg_surv / args.population:.1f}%",
            f"{avg_steps:.1f}",
            f"{avg_enrg:.1f}",
            f"{avg_mls:.1f}",
            f"{avg_starv:.1f}",
            f"{avg_combat:.1f}",
            f"{avg_age:.1f}",
            f"{extinct_n}/{n_runs} runs",
        ]
    )

    _draw_table(
        ax_per_run,
        [
            "Run",
            "Start Pop",
            "Survivors",
            "Surv%",
            "Steps",
            "Avg Energy",
            "Med Lifespan",
            "Starv Deaths",
            "Combat Deaths",
            "Age Deaths",
            "Extinct",
        ],
        run_rows,
        title="Per-Run Summary",
    )

    # --- Genome table (full-width, bottom) ---
    genome_bounds = _load_genome_bounds()
    trait_names = next((list(r["avg_traits"].keys()) for r in all_results if r["avg_traits"]), [])
    if trait_names:
        genome_rows = []
        for trait in trait_names:
            run_vals = [
                r["avg_traits"][trait]
                for r in all_results
                if r["avg_traits"] and trait in r["avg_traits"]
            ]
            if not run_vals:
                continue
            info = genome_bounds.get(trait, {})
            lo = f"{info['min']:.3f}" if "min" in info else "—"
            hi = f"{info['max']:.3f}" if "max" in info else "—"
            genome_rows.append(
                [
                    trait,
                    f"{np.mean(run_vals):.4f}",
                    f"{min(run_vals):.4f}",
                    f"{max(run_vals):.4f}",
                    lo,
                    hi,
                ]
            )
        _draw_table(
            ax_genome,
            ["Trait", "Grand Avg", "Run-Min Avg", "Run-Max Avg", "Genome Min", "Genome Max"],
            genome_rows,
            title="Survivors' Genome Traits — Aggregated Across All Runs",
        )
    else:
        ax_genome.axis("off")
        ax_genome.text(
            0.5,
            0.5,
            "No survivors in any run — genome data unavailable",
            ha="center",
            va="center",
            transform=ax_genome.transAxes,
            fontsize=11,
            color="#888",
        )

    _save_figure(fig, f"meta_report_{n_runs}runs")


def _save_figure(fig, prefix):
    """Save *fig* to charts/ with a timestamp filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_dir = Path(__file__).resolve().parent.parent / "charts"
    chart_dir.mkdir(exist_ok=True)
    save_path = chart_dir / f"{prefix}_{timestamp}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Report saved → {save_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------


def _print_master_bar(done, total, result=None):
    """
    Print/update a single master progress bar showing how many runs have completed.
    When *result* is provided it also prints the just-completed run's summary.
    """
    bar_width = 28
    filled = int(bar_width * done / total)
    progress_bar = "█" * filled + "─" * (bar_width - filled)
    if result is not None:
        fs = result["final_stats"]
        tag = "EXTINCT" if fs["alive_agents"] == 0 else f"{fs['alive_agents']} survivors"
        print(f"  Run {result['run_idx'] + 1:>2}  done  ·  step {fs['step']:>5}  ·  {tag}")
    print(f"\r  [{progress_bar}]  {done}/{total} runs complete", end="", flush=True)


def run_meta_simulation(args):
    """Run --runs parallel simulations then emit a combined report."""
    n_runs = args.runs
    workers = min(args.workers, n_runs)
    col_width = 60

    print(
        f"\n  Meta-run: {n_runs} simulations  ·  {workers} parallel workers\n"
        f"  {args.population} agents  ·  {args.grid_size}×{args.grid_size} grid  ·  "
        f"{args.steps} steps"
    )
    print("  " + "─" * (col_width - 2))

    args_dict = vars(args)
    seeds = [random.randint(0, 2**31 - 1) for _ in range(n_runs)]

    start = time.time()

    if workers <= 1:
        # Sequential: one labeled bar per run, updating in place on a single line
        all_results = []
        for i in range(n_runs):
            random.seed(seeds[i])
            np.random.seed(seeds[i] % (2**31))
            with _suppress_stdout():
                env = Environment(
                    grid_size=args_dict["grid_size"],
                    num_agents=args_dict["population"],
                )
            logged_stats = {}
            for _ in range(args_dict["steps"]):
                env.step()
                stats = env.get_stats()
                logged_stats = env.log_stats(stats, logged_stats)
                step = stats["step"]
                total = args_dict["steps"]
                filled = int(28 * step / total)
                progress_bar = "█" * filled + "─" * (28 - filled)
                print(f"\r  Run {i + 1:>2}  [{progress_bar}]  {step}/{total}", end="", flush=True)
                if stats["alive_agents"] == 0:
                    break
            final_stats = env.get_stats()
            avg_traits = env.get_average_genome_traits()
            result = {
                "run_idx": i,
                "logged_stats": logged_stats,
                "final_stats": final_stats,
                "avg_traits": avg_traits,
                "initial_population": args_dict["population"],
            }
            tag = (
                "EXTINCT"
                if final_stats["alive_agents"] == 0
                else f"{final_stats['alive_agents']} survivors"
            )
            print(f"\r  Run {i + 1:>2}  complete  ·  step {final_stats['step']:>5}  ·  {tag:<30}")
            all_results.append(result)

    else:
        # Parallel: master bar advances as each run finishes (imap_unordered)
        # Workers pass None for the queue — no per-step IPC needed
        tasks = [(args_dict, seeds[i], i, None) for i in range(n_runs)]

        ctx = mp.get_context("spawn")
        all_results = []
        _print_master_bar(0, n_runs)
        with ctx.Pool(workers) as pool:
            for result in pool.imap_unordered(_run_worker, tasks):
                all_results.append(result)
                _print_master_bar(len(all_results), n_runs, result)
        print()  # newline after the master bar

    elapsed = time.time() - start
    extinct_count = sum(1 for r in all_results if r["final_stats"]["alive_agents"] == 0)

    print("\n  " + "═" * (col_width - 2))
    print(
        f"  {n_runs} runs complete in {elapsed:.1f}s  "
        f"({elapsed / n_runs:.1f}s avg)  ·  {extinct_count}/{n_runs} extinctions"
    )
    print("  " + "═" * (col_width - 2))

    plot_meta_stats(all_results, args)


# pylint: disable=too-many-branches,too-many-statements
def run_simulation(args):
    """
    Run the simulation with provided arguments.

    Args:
        args: Parsed command line arguments
    """
    if args.runs > 1:
        run_meta_simulation(args)
        return

    col_width = 54

    print(
        f"  Initializing simulation — {args.population} agents, "
        f"{args.grid_size}x{args.grid_size} grid"
    )
    env = Environment(
        grid_size=args.grid_size,
        num_agents=args.population,
    )
    logged_stats = {}

    if args.show_initial:
        initial_grid = env.capture_grid_state()
        env.print_grid_state(initial_grid, "Initial Grid")

    print(f"\n  Running {args.steps} steps")
    print("  " + "─" * (col_width - 2))

    for _ in range(args.steps):
        env.step()
        stats = env.get_stats()
        logged_stats = env.log_stats(stats, logged_stats)

        if args.no_visual and not args.stats_only:
            _print_progress(stats["step"], args.steps)
        elif not args.no_visual or args.stats_only:
            ds = stats["death_stats"]
            print(
                f"  step {stats['step']:>5}  |  alive {stats['alive_agents']:>4}  "
                f"|  food {stats['total_food']:>6.1f}  |  energy {stats['avg_energy']:>6.1f}  "
                f"|  deaths  starve {ds['Starvation']['count']}  "
                f"combat {ds['Combat']['count']}  "
                f"old age {ds['Old Age']['count']}"
            )

        if not args.no_visual and not args.stats_only:
            env.print_grid()

        if stats["alive_agents"] == 0:
            if args.no_visual and not args.stats_only:
                print()
            print("\n  " + "═" * (col_width - 2))
            print("  EXTINCTION — all agents have died")
            print(f"  Simulation ended at step {stats['step']}")
            print("  " + "═" * (col_width - 2))
            break

        if args.delay > 0 and not args.no_visual:
            time.sleep(args.delay)

    if args.no_visual and not args.stats_only:
        print()

    final_stats = env.get_stats()
    avg_traits = env.get_average_genome_traits()
    death_stats = final_stats.get("death_stats", {})

    print("\n  " + "═" * (col_width - 2))
    print("  POND SPAWN  ·  Simulation Complete")
    print("  " + "═" * (col_width - 2))

    print(f"\n  {'Survivors':<16} {final_stats['alive_agents']} agents")
    print(f"  {'Food left':<16} {final_stats['total_food']:.0f} units")
    print(f"  {'Steps run':<16} {final_stats['step']}")
    print(f"  {'Avg energy':<16} {final_stats['avg_energy']:.1f}")
    print(
        f"\n  Lifespan   median {final_stats['median_lifespan']:.0f}"
        f"  ·  min {final_stats['min_age']}"
        f"  ·  max {final_stats['max_age']}"
    )

    if any(d["count"] > 0 for d in death_stats.values()):
        print(f"\n  {'Cause of Death':<18} {'Deaths':>7}   {'Avg Age':>8}")
        print("  " + "─" * (col_width - 2))
        for cause, data in death_stats.items():
            print(f"  {cause:<18} {data['count']:>7}   {data['avg_age']:>8.1f}")

    if avg_traits:
        print("\n  Survivors' Avg Traits")
        print("  " + "─" * (col_width - 2))
        pairs = [f"{k}: {v:.2f}" for k, v in avg_traits.items()]
        line = ""
        for pair in pairs:
            if len(line) + len(pair) + 5 > col_width - 4:
                print(f"  {line}")
                line = pair
            else:
                line = f"{line}  ·  {pair}" if line else pair
        if line:
            print(f"  {line}")

    if args.show_initial:
        env.print_grid_state(initial_grid, "Initial Grid")
        env.print_grid_state(env.capture_grid_state(), "Final Grid")

    print("\n  " + "═" * (col_width - 2) + "\n")

    plot_simulation_stats(logged_stats, args.population, avg_traits, death_stats)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a pond_spawn artificial life simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grid-size", type=int, default=10, help="Size of the square grid (grid_size x grid_size)"
    )
    parser.add_argument("--population", type=int, default=50, help="Initial number of agents")
    parser.add_argument(
        "--food-resupply",
        type=int,
        default=10,
        help="Max steps between food regen events per tile (higher = slower, 0 = never)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps to run")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0001,
        help="Delay between steps in seconds (ignored in --no-visual mode)",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable grid rendering; shows progress bar instead",
    )
    parser.add_argument(
        "--show-initial", action="store_true", help="Print initial and final grid states"
    )
    parser.add_argument("--stats-only", action="store_true", help="Show stats output without grid")
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of simulations to run (>1 enables meta-run mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes for meta-runs (--runs > 1 only)",
    )
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
