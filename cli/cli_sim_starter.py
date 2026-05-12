"""
CLI tool for running pond_spawn simulations.

Usage:
    python -m cli.cli_sim_starter [options]
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from simulation import Environment


# pylint: disable=too-many-locals,too-many-statements
def plot_simulation_stats(logged_stats, initial_population, avg_traits=None, death_tally=None):
    """
    Create line graphs for simulation statistics over time.

    Args:
        logged_stats (dict): Dictionary mapping step -> stats dict
        initial_population (int): Initial agent population
        avg_traits (dict): Average genome traits of final population
        death_tally (dict): cause_of_death string -> count over the run
    """
    if not logged_stats:
        print("No stats to plot.")
        return

    # Extract data
    steps = sorted(logged_stats.keys())
    alive_agents = [logged_stats[s]["alive_agents"] for s in steps]
    total_food = [logged_stats[s]["total_food"] for s in steps]
    avg_energy = [logged_stats[s]["avg_energy"] for s in steps]
    median_lifespan = [logged_stats[s]["median_lifespan"] for s in steps]
    min_age = [logged_stats[s]["min_age"] for s in steps]
    max_age = [logged_stats[s]["max_age"] for s in steps]

    has_death_table = bool(death_tally)

    # 4 time-series plots + optional death table row
    nrows = 5 if has_death_table else 4
    height_ratios = [3, 3, 3, 3, 1.4] if has_death_table else [1, 1, 1, 1]
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(10, 17 if has_death_table else 14),
        gridspec_kw={"height_ratios": height_ratios},
    )
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
    fig.suptitle("Pond Spawn Simulation Statistics", fontsize=16, fontweight="bold")

    # Plot 1: Agent Population
    ax1.plot(steps, alive_agents, "b-", linewidth=2, label="Alive Agents")
    ax1.axhline(
        y=initial_population,
        color="r",
        linestyle="--",
        linewidth=1,
        label=f"Initial Pop ({initial_population})",
    )
    ax1.set_xlabel("Step", fontsize=12)
    ax1.set_ylabel("Agent Count", fontsize=12)
    ax1.set_title("Agent Population Over Time", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Food Supply
    ax2.plot(steps, total_food, "g-", linewidth=2, label="Total Food")
    ax2.set_xlabel("Step", fontsize=12)
    ax2.set_ylabel("Food Units", fontsize=12)
    ax2.set_title("Food Supply Over Time", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Average Energy
    ax3.plot(steps, avg_energy, "orange", linewidth=2, label="Avg Energy")
    ax3.set_xlabel("Step", fontsize=12)
    ax3.set_ylabel("Energy", fontsize=12)
    ax3.set_title("Average Agent Energy Over Time", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Lifespan Statistics (Median with Min/Max Range)
    ax4.plot(steps, median_lifespan, "purple", linewidth=2, label="Median Lifespan")
    ax4.fill_between(steps, min_age, max_age, color="purple", alpha=0.2, label="Min-Max Range")
    ax4.set_xlabel("Step", fontsize=12)
    ax4.set_ylabel("Age", fontsize=12)
    ax4.set_title("Agent Lifespan Over Time", fontsize=14, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Table 5: Death causes
    if has_death_table:
        ax5 = axes[4]
        ax5.axis("off")
        ax5.set_title("Deaths by Cause", fontsize=14, fontweight="bold", pad=8)
        total_deaths = sum(death_tally.values())
        rows = sorted(death_tally.items(), key=lambda x: -x[1])
        rows.append(("TOTAL", total_deaths))
        table_data = [
            [cause, str(count), f"{count / total_deaths * 100:.1f}%"] for cause, count in rows
        ]
        tbl = ax5.table(
            cellText=table_data,
            colLabels=["Cause of Death", "Count", "%"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.4)
        # Style header row
        for col in range(3):
            tbl[(0, col)].set_facecolor("#2c3e50")
            tbl[(0, col)].set_text_props(color="white", fontweight="bold")
        # Style TOTAL row
        total_row = len(rows)
        for col in range(3):
            tbl[(total_row, col)].set_facecolor("#dfe6e9")
            tbl[(total_row, col)].set_text_props(fontweight="bold")

    plt.tight_layout(rect=[0, 0.06 if avg_traits else 0, 1, 0.97])

    # Add genome trait averages as text annotation at the bottom if provided
    if avg_traits:
        trait_text = "Final Population Avg Traits:  "
        trait_text += "  |  ".join(f"{name}: {value:.3f}" for name, value in avg_traits.items())

        # Place text centered at the bottom of the figure
        fig.text(
            0.5,
            0.02,
            trait_text,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_stats_{timestamp}.png"
    chart_dir = Path(__file__).resolve().parent.parent / "charts"
    chart_dir.mkdir(exist_ok=True)
    save_path = chart_dir / filename

    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"\nGraph saved as '{filename}'")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a pond_spawn artificial life simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Size of the square grid (grid_size x grid_size)",
    )

    parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Initial number of agents",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of simulation steps to run",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.0001,
        help="Delay between steps in seconds",
    )

    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable grid visualization (faster)",
    )

    parser.add_argument(
        "--show-initial",
        action="store_true",
        help="Show initial grid state",
    )

    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show statistics, no grid",
    )

    return parser.parse_args()


# pylint: disable=too-many-branches,too-many-statements
def run_simulation(args):
    """
    Run the simulation with provided arguments.

    Args:
        args: Parsed command line arguments
    """
    # Create simulation environment
    print(f"Initializing simulation with {args.population} agents...")
    env = Environment(
        grid_size=args.grid_size,
        num_agents=args.population,
    )
    logged_stats = {}

    # Capture and show initial state if requested
    if args.show_initial:
        initial_grid = env.capture_grid_state()
        print("\nInitial Environment State:")
        stats = env.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        env.print_grid_state(initial_grid, "Initial Grid")

    # Run simulation
    print(f"\nRunning simulation for {args.steps} steps...")
    print("-" * 50)

    progress_interval = max(1, args.steps // 100)
    interrupted = False
    try:
        for _ in range(args.steps):
            # Run step
            env.step()

            # Get stats
            stats = env.get_stats()

            # Log stats for graphing
            logged_stats = env.log_stats(stats, logged_stats)

            # Print stats
            if not args.no_visual or args.stats_only:
                print(f"\nOriginal population: {args.population}")
                print(f"Step {stats['step']}, Food={stats['total_food']}")
                print("Current simulation stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            elif stats["step"] % progress_interval == 0:
                pct = 100 * stats["step"] / args.steps
                print(
                    f"  Step {stats['step']}/{args.steps} ({pct:.0f}%)"
                    f" | pop={stats['alive_agents']}"
                    f" | food={stats['total_food']}"
                    f" | avg_energy={stats['avg_energy']:.1f}",
                    flush=True,
                )

            # Print grid if visualization enabled
            if not args.no_visual and not args.stats_only:
                print("Current grid:")
                env.print_grid()

            # Check for extinction
            if stats["alive_agents"] == 0:
                print("\n" + "=" * 50)
                print("EXTINCTION EVENT - All agents have died")
                print(f"Simulation ended at step {stats['step']}")
                print("=" * 50)
                break

            # Delay between steps
            if args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "=" * 50)
        print("INTERRUPTED - saving results...")

    # Final summary
    print("\n" + "=" * 50)
    print("SIMULATION INTERRUPTED" if interrupted else "SIMULATION COMPLETE")
    final_stats = env.get_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    death_tally = env.get_death_tally()

    # Show average genome traits
    avg_traits = env.get_average_genome_traits()
    if avg_traits:
        print("\nAverage Genome Traits (Living Population):")
        for trait_name, avg_value in avg_traits.items():
            print(f"  {trait_name}: {avg_value:.4f}")

    if args.show_initial:
        print("\nComparison:")
        env.print_grid_state(initial_grid, "Initial Grid")
        final_grid = env.capture_grid_state()
        env.print_grid_state(final_grid, "Final Grid")

    print("=" * 50)

    # Generate graphs from logged stats
    plot_simulation_stats(logged_stats, args.population, avg_traits, death_tally)


def main():
    """Main entry point for CLI."""
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
