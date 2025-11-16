import random as r

from controllers.agent import Agent
from controllers.genome import Genome
from controllers.landscape import Biome

# Global variables
POPULATION = 500
FOOD_RESUPPLY = 3
MAX_FOOD_PER_TILE = 3
TICKS = 1000


class Environment:
    """
    Represents the simulation environment containing a grid of biomes and agents.

    The environment manages:
    - A 2D grid of biome tiles
    - A population of agents
    - Food distribution across the grid
    - Simulation step logic
    """

    def __init__(self, grid_size=10, num_agents=POPULATION, food_units=FOOD_RESUPPLY):
        """
        Initialize the simulation environment.

        Args:
            grid_size (int): Size of the square grid (grid_size x grid_size)
            num_agents (int): Number of agents to spawn initially
            food_units (int): Food units to add per resupply cycle
        """
        self.grid_size = grid_size
        self.grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.food_resupply_amount = food_units  # Store for resupply
        self.agents = []
        self.step_count = 0

        self._initialize_biomes()
        # Biomes already have 0-3 food from generation, don't add more initially
        self._spawn_agents(num_agents)

    def _initialize_biomes(self):
        """Generate random biomes for each tile in the grid."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                biome = Biome().generate()
                self.grid[x][y] = biome

    def _add_food(self, total_food):
        """
        Add food units across the grid based on biome fertility (doesn't replace existing food).

        Args:
            total_food (int): Base food units to distribute (modified by fertility)
        """
        for x, y, biome in self.iter_biomes():
            # Get fertility modifier (0.0 to 1.0)
            fertility = biome.get_fertility()

            # Add food based on fertility - more fertile biomes get more food
            # Use total_food as the max per tile, scaled by fertility
            food_to_add = int(total_food * fertility) % 100

            current_food = biome.get_food_units()
            biome.features["food_units"] = (current_food or 0) + food_to_add

    def _spawn_agents(self, num_agents):
        """
        Create and place agents randomly in the environment.

        Args:
            num_agents (int): Number of agents to spawn
        """
        for _ in range(num_agents):
            # Generate random genome for agent
            genome = Genome().generate()

            # Random starting position
            position = (
                r.randint(0, self.grid_size - 1),
                r.randint(0, self.grid_size - 1),
            )

            # Create agent
            agent = Agent(genome, position)
            self.agents.append(agent)

    def get_biome(self, x, y):
        """
        Get the biome at the specified coordinates.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            Biome: The biome at (x, y) or None if out of bounds
        """
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x][y]
        return None

    def iter_biomes(self):
        """
        Iterate over all biomes in the grid with their coordinates.

        Yields:
            tuple: (x, y, biome) for each position in the grid

        Example:
            for x, y, biome in env.iter_biomes():
                print(f"Biome at ({x}, {y}) has {biome.get_food_units()} food")
        """
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                yield x, y, self.grid[x][y]

    def get_all_biomes(self):
        """
        Get a list of all biomes with their coordinates.

        Returns:
            list: List of tuples (x, y, biome) for all positions
        """
        return list(self.iter_biomes())

    def get_agents_at(self, x, y):
        """
        Get all agents at the specified position.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            list: List of agents at this position
        """
        return [agent for agent in self.agents if agent.position == (x, y)]

    def count_agents_in_range(self, position, vision):
        """
        Count agents in the 180-degree field in front of the agent at `position`.

        Args:
            position (tuple): (x, y) position of the focal agent.

        Returns:
            int: Number of agents in forward field of view.
        """
        x, y = position

        # Find the agent at this position
        target_agent = None
        for agent in self.agents:
            if agent.position == position:
                target_agent = agent
                break

        if target_agent is None:
            return 0

        heading = target_agent.get_heading()  # 0 = N, 1 = E, 2 = S, 3 = W

        # Forward tiles for each heading (no wrap; petri dish world)
        if heading == 0:  # north
            positions = [
                (x, y + 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y + 1),
                (x + 1, y + 1),
            ]
        elif heading == 1:  # east
            positions = [
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x + 1, y - 1),
                (x + 1, y + 1),
            ]
        elif heading == 2:  # south
            positions = [
                (x, y - 1),
                (x - 1, y),
                (x + 1, y),
                (x - 1, y - 1),
                (x + 1, y - 1),
            ]
        elif heading == 3:  # west
            positions = [
                (x - 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y - 1),
                (x - 1, y + 1),
            ]
        else:
            positions = []

        count = 0
        for px, py in positions:
            if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
                agents_here = self.get_agents_at(px, py)
                count += len(agents_here)

        return count * vision

    def redist_food(self):
        """Add food to the grid based on resupply amount."""
        self._add_food(self.food_resupply_amount)

    def step(self):
        """
        Execute one simulation step.

        Process:
        1. Update all agents (age, metabolism, decision making, movement, turning)
        2. Process eating in speed-priority order (faster agents eat first)
        3. Add offspring from reproduction
        4. Remove dead agents
        """
        current_food = sum(biome.get_food_units() for _, _, biome in self.iter_biomes())

        self.step_count += 1

        # Replenish food
        if current_food < len(self.agents) / 50:
            self.redist_food()

        # Update all agents and collect offspring
        new_agents = []
        agents_wanting_to_eat = []  # Track agents that chose to eat

        for agent in self.agents:
            if not agent.is_alive():
                return
            # Agent updates but doesn't eat yet
            action, offspring = agent.update_without_eating(self)

            # Track if agent wants to eat
            if action == 2:  # ACTION_EAT
                agents_wanting_to_eat.append(agent)

            # Collect offspring
            if offspring:
                new_agents.append(offspring)

        # Let agents eat in order of speed, grouped by tile
        # This prevents checking food after it's gone
        for agent in agents_wanting_to_eat:
            if not agent.is_alive():
                continue

            # Check if there's food available at agent's position
            x, y = agent.position
            biome = self.get_biome(x, y)

            # Only attempt to eat if food is available
            if biome and biome.get_food_units() > 0:
                agent.eat(self)

        # Add new offspring to population
        self.agents.extend(new_agents)

        # Remove dead agents
        self.agents = [agent for agent in self.agents if agent.is_alive()]

    def get_stats(self):
        """
        Get current simulation statistics.

        Returns:
            dict: Statistics about the current state
        """
        total_food = sum(biome.get_food_units() for _, _, biome in self.iter_biomes())

        return {
            "step": self.step_count,
            "alive_agents": len([a for a in self.agents if a.is_alive()]),
            "total_food": total_food,
            "avg_energy": sum(a.energy for a in self.agents) / len(self.agents)
            if self.agents
            else 0,
            "avg_lifespan": sum(a.age for a in self.agents) / len(self.agents)
            if self.agents
            else 0,
        }

    def log_stats(self, stats, state_dict):
        """
        Log statistics to a state dictionary for graphing.

        Args:
            stats (dict): Statistics dictionary with keys: step, alive_agents, total_food, avg_energy
            state_dict (dict): Dictionary to accumulate stats over time

        Returns:
            dict: Updated state dictionary
        """
        step = stats["step"]
        state_dict[step] = {
            "alive_agents": stats["alive_agents"],
            "total_food": stats["total_food"],
            "avg_energy": stats["avg_energy"],
            "avg_lifespan": stats["avg_lifespan"],
        }
        return state_dict

    def capture_grid_state(self):
        """
        Capture current grid state as data structure.

        Returns:
            list: 2D list of tuples (agent_count, food_count) for each cell
        """
        state = []
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                agents_here = len(self.get_agents_at(x, y))
                food = self.grid[x][y].get_food_units()
                row.append((agents_here, food))
            state.append(row)
        return state

    def print_grid_state(self, grid_state, label=""):
        """
        Print a previously captured grid state.

        Args:
            grid_state (list): Output from capture_grid_state()
            label (str): Optional label to print before grid
        """
        # ANSI color codes
        red = "\033[91m"
        yellow = "\033[93m"
        green = "\033[92m"
        reset = "\033[0m"

        if label:
            print(f"\n=== {label} ===")

        for row in grid_state:
            cells = []
            for agents_here, food in row:
                # Color code agents: 0=red, 1=yellow, 2+=green
                if agents_here == 0:
                    agent_color = red
                elif agents_here == 1:
                    agent_color = yellow
                else:
                    agent_color = green

                # Color code food: 0=red, 1=yellow, 2+=green
                if food == 0:
                    food_color = red
                elif food == 1:
                    food_color = yellow
                else:
                    food_color = green

                cell = f"{agent_color}A:{agents_here}{reset} {food_color}F:{food}{reset}"
                cells.append(cell)
            print(" | ".join(cells))
        print()

    def print_grid(self):
        """
        Print a visual representation of the grid with color coding.

        Color scheme:
        - Agents: 0 = red, 1 = yellow, 2+ = green
        - Food: 0 = red, 1 = yellow, 2+ = green (inverse priority)
        """
        # ANSI color codes
        red = "\033[91m"
        yellow = "\033[93m"
        green = "\033[92m"
        reset = "\033[0m"

        print(f"\n=== Step {self.step_count} ===")
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                agents_here = len(self.get_agents_at(x, y))
                food = self.grid[x][y].get_food_units()

                # Color code agents: 0=red, 1=yellow, 2+=green
                if agents_here == 0:
                    agent_color = red
                elif agents_here == 1:
                    agent_color = yellow
                else:
                    agent_color = green

                # Color code food: 0=red, 1=yellow, 2+=green
                if food == 0:
                    food_color = red
                elif food == 1:
                    food_color = yellow
                else:
                    food_color = green

                cell = f"{agent_color}A:{agents_here}{reset} {food_color}F:{food}{reset}"
                row.append(cell)
            print(" | ".join(row))
        print()


if __name__ == "__main__":
    # For backwards compatibility, run with default settings
    from cli.cli_sim_starter import main

    main()
