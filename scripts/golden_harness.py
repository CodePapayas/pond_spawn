"""
Golden harness for Phase 0 parity testing.

Runs a seeded simulation and dumps a JSON fixture used to validate the Rust port.
Parity targets: initial genome/trait generation, biome structure, economy math
(metabolism drain, food value, reproduction cost, mutation bounds). Brain output
parity is NOT a target — PyTorch and the hand-rolled Rust MLP will diverge after
the first decision and that is expected.

Output:
  step_0_state  — full agent dump (traits, weight ranges) + grid before any steps
  population_curve — alive/food/energy per step for visual shape comparison only

Usage:
    python scripts/golden_harness.py --seed 42 --steps 200 --output data/golden.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simulation import Environment  # pylint: disable=wrong-import-position


def _dump_agent(agent):
    weights = agent.genome.brain_weights
    return {
        "id": agent.get_id(),
        "energy": agent.energy,
        "age": agent.age,
        "pos": list(agent.position),
        "heading": agent.get_heading(),
        "traits": {name: info.get("value") for name, info in agent.genome.traits.items()},
        "brain_weight_count": len(weights),
        "brain_weight_min": min(weights),
        "brain_weight_max": max(weights),
    }


def _dump_grid(env):
    tiles = []
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            biome = env.grid[x][y]
            tiles.append(
                {
                    "pos": [x, y],
                    "food_units": biome.get_food_units(),
                    "fertility": biome.features["fertility"]["value"],
                    "movement_speed": biome.features["movement_speed"]["value"],
                    "visibility": biome.features["visibility"]["value"],
                }
            )
    return tiles


def run_harness(seed, grid_size, population, steps, output_path):
    """Run a seeded simulation and write a golden JSON trace to output_path."""
    env = Environment(grid_size=grid_size, num_agents=population, seed=seed)

    # Step 0: dump initial state before any brain decisions touch it
    step_0_agents = [_dump_agent(a) for a in env.agents]
    step_0_grid = _dump_grid(env)
    step_0_stats = env.get_stats()

    population_curve = {}

    for step_num in range(1, steps + 1):
        env.step()
        stats = env.get_stats()
        population_curve[str(step_num)] = {
            "alive": stats["alive_agents"],
            "food": stats["total_food"],
            "avg_energy": round(stats["avg_energy"], 4),
        }

        if stats["alive_agents"] == 0:
            print(f"  Extinction at step {step_num}")
            break

        if step_num % 50 == 0:
            print(
                f"  step {step_num:>5}  |  alive {stats['alive_agents']:>4}"
                f"  |  food {stats['total_food']:>6.1f}"
                f"  |  energy {stats['avg_energy']:>6.1f}"
            )

    trace = {
        "meta": {
            "seed": seed,
            "grid_size": grid_size,
            "population": population,
            "steps": steps,
            "note": (
                "Brain output parity is NOT a target. "
                "Validate: trait bounds, weight ranges, biome structure, economy math."
            ),
        },
        "step_0_state": {
            "stats": step_0_stats,
            "agents": step_0_agents,
            "grid": step_0_grid,
        },
        "population_curve": population_curve,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    final_alive = env.get_stats()["alive_agents"]
    print(f"\n  Golden trace written → {out}")
    print(f"  Step 0 agents: {len(step_0_agents)}  |  final alive: {final_alive}")


def main():
    """Parse CLI args and run the golden harness."""
    parser = argparse.ArgumentParser(description="pond_spawn golden harness")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--grid-size", type=int, default=12, help="Grid size")
    parser.add_argument("--population", type=int, default=100, help="Initial agents")
    parser.add_argument("--steps", type=int, default=200, help="Steps to run")
    parser.add_argument("--output", type=str, default="data/golden.json", help="Output JSON path")
    args = parser.parse_args()

    print("\n  pond_spawn golden harness")
    print(
        f"  seed={args.seed}  grid={args.grid_size}x{args.grid_size}"
        f"  pop={args.population}  steps={args.steps}\n"
    )

    run_harness(
        seed=args.seed,
        grid_size=args.grid_size,
        population=args.population,
        steps=args.steps,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
