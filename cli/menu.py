"""
Interactive terminal menu for pond_spawn simulations.

Run with:  python -m cli.menu
"""

import types
from dataclasses import dataclass, field

from cli.cli_sim_starter import run_simulation

BANNER = r"""
  ____                _   ____
 |  _ \ ___  _ __   __| | / ___| _ __   __ ___      ___ __
 | |_) / _ \| '_ \ / _` | \___ \| '_ \ / _` \ \ /\ / / '_ \
 |  __/ (_) | | | | (_| |  ___) | |_) | (_| |\ V  V /| | | |
 |_|   \___/|_| |_|\__,_| |____/| .__/ \__,_| \_/\_/ |_| |_|
                                 |_|
"""

VISUAL_LABELS = {
    "progress": "Progress bar only (fast)",
    "stats": "Stats each step (slow)",
    "grid": "Full grid each step (very slow)",
}


@dataclass
class Settings:  # pylint: disable=too-many-instance-attributes
    """User-configurable simulation parameters, held between menu loops."""

    grid_size: int = 10
    population: int = 50
    steps: int = 1000
    runs: int = 1
    delay: float = 0.0001
    visual: str = "progress"  # "progress" | "stats" | "grid"
    show_initial: bool = False
    run_summaries: list = field(default_factory=list)


def _to_args(s: Settings):
    """Convert Settings to the namespace that run_simulation expects."""
    return types.SimpleNamespace(
        grid_size=s.grid_size,
        population=s.population,
        steps=s.steps,
        delay=s.delay,
        no_visual=s.visual == "progress",
        stats_only=s.visual == "stats",
        show_initial=s.show_initial,
    )


def _prompt_int(prompt: str, lo: int, hi: int, current: int) -> int:
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}] (current: {current}): ").strip()
        if raw == "":
            return current
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  Must be between {lo} and {hi}.")
        except ValueError:
            print("  Invalid number.")


def _prompt_float(prompt: str, lo: float, hi: float, current: float) -> float:
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}] (current: {current}): ").strip()
        if raw == "":
            return current
        try:
            val = float(raw)
            if lo <= val <= hi:
                return val
            print(f"  Must be between {lo} and {hi}.")
        except ValueError:
            print("  Invalid number.")


def _display_settings(s: Settings):
    print("\n  Current settings:")
    print(f"    Grid size   : {s.grid_size} x {s.grid_size}")
    print(f"    Population  : {s.population}")
    print(f"    Steps       : {s.steps}")
    print(f"    Runs        : {s.runs}")
    print(f"    Step delay  : {s.delay}s")
    print(f"    Visual mode : {VISUAL_LABELS[s.visual]}")
    print(f"    Show initial: {'yes' if s.show_initial else 'no'}")


def _display_menu():
    print("\n  ── Options ─────────────────────────────")
    print("  [1] Set grid size")
    print("  [2] Set population")
    print("  [3] Set steps per run")
    print("  [4] Set number of runs")
    print("  [5] Set step delay")
    print("  [6] Change visual mode")
    print("  [7] Toggle show-initial grid")
    print("  [8] RUN")
    print("  [0] Quit")
    print("  ────────────────────────────────────────")


def _choose_visual(current: str) -> str:
    options = list(VISUAL_LABELS.keys())
    print("\n  Visual modes:")
    for i, key in enumerate(options, 1):
        marker = " <-- current" if key == current else ""
        print(f"    [{i}] {VISUAL_LABELS[key]}{marker}")
    while True:
        raw = input("  Choose [1-3] (Enter = keep current): ").strip()
        if raw == "":
            return current
        if raw in ("1", "2", "3"):
            return options[int(raw) - 1]
        print("  Enter 1, 2, or 3.")


def _run_all(s: Settings):
    args = _to_args(s)
    summaries = []

    for run_num in range(1, s.runs + 1):
        header = f"RUN {run_num}/{s.runs}"
        sep = "=" * (len(header) + 4)
        print(f"\n{sep}")
        print(f"  {header}")
        print(f"{sep}")
        try:
            run_simulation(args)
            summaries.append({"run": run_num, "status": "completed"})
        except Exception as exc:  # pylint: disable=broad-except
            print(f"\n  Run {run_num} failed: {exc}")
            summaries.append({"run": run_num, "status": f"error: {exc}"})

    if s.runs > 1:
        print("\n" + "=" * 40)
        print("  MULTI-RUN SUMMARY")
        print("=" * 40)
        for summary in summaries:
            print(f"  Run {summary['run']}: {summary['status']}")
        print("=" * 40)


def main():
    """Launch the interactive menu loop."""
    print(BANNER)
    print("  Welcome to Pond Spawn — interactive launcher")

    s = Settings()

    while True:
        _display_settings(s)
        _display_menu()

        choice = input("\n  > ").strip()

        if choice == "0":
            print("\n  Goodbye.\n")
            break

        if choice == "1":
            s.grid_size = _prompt_int("Grid size", 4, 50, s.grid_size)

        elif choice == "2":
            max_pop = 2 * s.grid_size * s.grid_size
            s.population = _prompt_int("Population", 1, max_pop, s.population)

        elif choice == "3":
            s.steps = _prompt_int("Steps per run", 1, 100_000, s.steps)

        elif choice == "4":
            s.runs = _prompt_int("Number of runs", 1, 50, s.runs)

        elif choice == "5":
            s.delay = _prompt_float("Step delay (seconds)", 0.0, 5.0, s.delay)

        elif choice == "6":
            s.visual = _choose_visual(s.visual)

        elif choice == "7":
            s.show_initial = not s.show_initial
            state = "enabled" if s.show_initial else "disabled"
            print(f"  Show-initial {state}.")

        elif choice == "8":
            _run_all(s)
            input("\n  Press Enter to return to menu...")

        else:
            print("  Unknown option.")


if __name__ == "__main__":
    main()
