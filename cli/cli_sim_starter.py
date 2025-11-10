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


def plot_simulation_stats(logged_stats, initial_population):
    """
    Create line graphs for simulation statistics over time.

    Args:
        logged_stats (dict): Dictionary mapping step -> stats dict
        initial_population (int): Initial agent population
    """
    if not logged_stats:
        print("No stats to plot.")
        return

    # Extract data
    steps = sorted(logged_stats.keys())
    alive_agents = [logged_stats[s]["alive_agents"] for s in steps]
    total_food = [logged_stats[s]["total_food"] for s in steps]
    avg_energy = [logged_stats[s]["avg_energy"] for s in steps]
    avg_lifespan = [logged_stats[s]["avg_lifespan"] for s in steps]

    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
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

    # Plot 4: Average Lifespan
    ax4.plot(steps, avg_lifespan, "orange", linewidth=2, label="Avg Lifespan")
    ax4.set_xlabel("Step", fontsize=12)
    ax4.set_ylabel("Age", fontsize=12)
    ax4.set_title("Average Agent Lifespan Over Time", fontsize=14, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

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
        "--food-resupply",
        type=int,
        default=3,
        help="Food units to add per resupply cycle (scaled by biome fertility)",
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


def run_simulation(args):
    """
    Run the simulation with provided arguments.

    Args:
        args: Parsed command line arguments
    """
    # Create simulation environment
    print(f"Initializing simulation with {args.population} agents...")
    env = Environment(
        grid_size=args.grid_size, num_agents=args.population, food_units=args.food_resupply
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
    print(f"Food resupply: {args.food_resupply} units per resupply (scaled by biome fertility)")
    print("Food resupplies only when total food reaches 0")
    print("-" * 50)

    for i in range(args.steps):
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

    # Final summary
    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    final_stats = env.get_stats()
    print("\nFinal Statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")

    if args.show_initial:
        print("\nComparison:")
        env.print_grid_state(initial_grid, "Initial Grid")
        final_grid = env.capture_grid_state()
        env.print_grid_state(final_grid, "Final Grid")

    print("=" * 50)

    # Generate graphs from logged stats
    plot_simulation_stats(logged_stats, args.population)


def main():
    """Main entry point for CLI."""
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
