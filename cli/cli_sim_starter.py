"""
CLI tool for running pond_spawn simulations.

Usage:
    python -m cli.cli_sim_starter [options]
"""

import argparse
import time

from simulation import Environment


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
        default=500,
        help="Initial number of agents",
    )

    parser.add_argument(
        "--food-init",
        type=int,
        default=10,
        help="Initial food units to distribute",
    )

    parser.add_argument(
        "--food-resupply",
        type=int,
        default=3,
        help="Food units to add each step",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of simulation steps to run",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.001,
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
        grid_size=args.grid_size, num_agents=args.population, food_units=args.food_init
    )

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
    print(f"Food resupply: {args.food_resupply} units per step")
    print("-" * 50)

    for i in range(args.steps):
        # Update food resupply in environment
        env.FOOD_RESUPPLY = args.food_resupply

        # Run step
        env.step()

        # Get stats
        stats = env.get_stats()

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


def main():
    """Main entry point for CLI."""
    args = parse_args()
    run_simulation(args)


if __name__ == "__main__":
    main()
