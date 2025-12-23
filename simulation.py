import random as r

import torch as t

from controllers.agent import Agent
from controllers.genome import Genome
from controllers.landscape import Biome

# Global variables
POPULATION = 300
FOOD_RESUPPLY = 3
MAX_FOOD_PER_TILE = 3
TICKS = 1000
MAX_AGENTS_PER_TILE = 1


class Environment:
    """
    Represents the simulation environment containing a grid of biomes and agents.

    The environment manages:
    - A 2D grid of biome tiles
    - A population of agents
    - Food distribution across the grid
    - Simulation step logic
    """

    def __init__(self, grid_size=12, num_agents=POPULATION, food_units=FOOD_RESUPPLY):
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
        self.position_map = {}
        self.agents_by_id = {}
        self.agents = []
        self.step_count = 0
        self.lifespans = []  # Track all lifespans for median/min/max
        self.logged_lifespans = set()  # Prevent duplicate lifespan logging

        # GPU setup for batched brain inference
        self.device = t.device("cuda" if t.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self._initialize_biomes()
        # Biomes already have 0-3 food from generation, don't add more initially

        # Cap population at 3 * grid_size * grid_size (max capacity)
        max_capacity = MAX_AGENTS_PER_TILE * grid_size * grid_size
        if num_agents > max_capacity:
            print(
                f"Warning: Requested population {num_agents} exceeds max capacity {max_capacity}. Capping at {max_capacity}."
            )
            num_agents = max_capacity

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

    def _record_lifespan(self, agent):
        """Store a single lifespan entry per agent to avoid double counting."""
        agent_id = agent.get_id()
        if agent_id in self.logged_lifespans:
            return
        self.lifespans.append(agent.age)
        self.logged_lifespans.add(agent_id)

    def _spawn_agents(self, num_agents):
        """
        Create and place agents randomly in the environment.

        Args:
            num_agents (int): Number of agents to spawn
        """
        # Create a list of all possible positions, repeated MAX_AGENTS_PER_TILE times
        all_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                all_positions.append((x, y))

        # Shuffle to randomize placement
        r.shuffle(all_positions)

        # Take only as many positions as we need
        spawn_positions = all_positions[:num_agents]

        for position in spawn_positions:
            # Generate random genome for agent
            genome = Genome().generate()

            # Create agent
            agent = Agent(genome, position)
            agent_id = agent.get_id()

            self.agents.append(agent)
            self.agents_by_id[agent_id] = agent

            if position not in self.position_map:
                self.position_map[position] = set()
            self.position_map[position].add(agent_id)

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
        Get all agents at the specified position using position map.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            list: List of agents at this position
        """
        position = (x, y)
        if position not in self.position_map:
            return []

        agent_ids = self.position_map[position]
        return [
            self.agents_by_id[agent_id] for agent_id in agent_ids if agent_id in self.agents_by_id
        ]

    def count_agents_in_range(self, position, vision):
        """
        Count agents in the 180-degree field in front of the agent at `position`.

        Vision determines range:
        - vision > 1.0: Can see 2 tiles ahead (wide field)
        - vision 0.5-1.0: Can see 1 tile ahead (normal field)
        - vision < 0.5: Blind - returns random guess (0-3)

        Args:
            position (tuple): (x, y) position of the focal agent.
            vision (float): Agent's vision trait value.

        Returns:
            int: Number of agents in forward field of view (or random if blind).
        """
        x, y = position

        # Blind agents just guess
        if vision < 0.5:
            return r.randint(0, 3)

        # Find the agent at this position using position map
        if position not in self.position_map:
            return 0

        agent_ids_at_pos = self.position_map[position]
        if not agent_ids_at_pos:
            return 0

        # Get first agent at this position
        target_agent_id = next(iter(agent_ids_at_pos))
        target_agent = self.agents_by_id.get(target_agent_id)

        if target_agent is None:
            return 0

        heading = target_agent.get_heading()  # 0 = N, 1 = E, 2 = S, 3 = W

        # Determine vision range: 2 tiles if vision > 1.0, else 1 tile
        can_see_far = vision > 1.0

        # Build positions based on heading and vision range
        # Near positions (1 tile): 180-degree arc in front
        # Far positions (2 tiles): extended 180-degree arc
        positions = []

        if heading == 0:  # North (y decreases)
            # Near: front + sides
            positions.extend(
                [
                    (x, y - 1),  # directly ahead
                    (x - 1, y),  # left
                    (x + 1, y),  # right
                    (x - 1, y - 1),  # front-left
                    (x + 1, y - 1),  # front-right
                ]
            )
            if can_see_far:
                positions.extend(
                    [
                        (x, y - 2),  # far ahead
                        (x - 1, y - 2),  # far front-left
                        (x + 1, y - 2),  # far front-right
                        (x - 2, y - 1),  # wide left
                        (x + 2, y - 1),  # wide right
                    ]
                )
        elif heading == 1:  # East (x increases)
            positions.extend(
                [
                    (x + 1, y),  # directly ahead
                    (x, y - 1),  # left (up)
                    (x, y + 1),  # right (down)
                    (x + 1, y - 1),  # front-left
                    (x + 1, y + 1),  # front-right
                ]
            )
            if can_see_far:
                positions.extend(
                    [
                        (x + 2, y),  # far ahead
                        (x + 2, y - 1),  # far front-left
                        (x + 2, y + 1),  # far front-right
                        (x + 1, y - 2),  # wide left
                        (x + 1, y + 2),  # wide right
                    ]
                )
        elif heading == 2:  # South (y increases)
            positions.extend(
                [
                    (x, y + 1),  # directly ahead
                    (x + 1, y),  # left
                    (x - 1, y),  # right
                    (x + 1, y + 1),  # front-left
                    (x - 1, y + 1),  # front-right
                ]
            )
            if can_see_far:
                positions.extend(
                    [
                        (x, y + 2),  # far ahead
                        (x + 1, y + 2),  # far front-left
                        (x - 1, y + 2),  # far front-right
                        (x + 2, y + 1),  # wide left
                        (x - 2, y + 1),  # wide right
                    ]
                )
        elif heading == 3:  # West (x decreases)
            positions.extend(
                [
                    (x - 1, y),  # directly ahead
                    (x, y + 1),  # left (down)
                    (x, y - 1),  # right (up)
                    (x - 1, y + 1),  # front-left
                    (x - 1, y - 1),  # front-right
                ]
            )
            if can_see_far:
                positions.extend(
                    [
                        (x - 2, y),  # far ahead
                        (x - 2, y + 1),  # far front-left
                        (x - 2, y - 1),  # far front-right
                        (x - 1, y + 2),  # wide left
                        (x - 1, y - 2),  # wide right
                    ]
                )

        # Wrap positions for toroidal map
        wrapped_positions = [(px % self.grid_size, py % self.grid_size) for px, py in positions]

        # Count agents using position_map (O(1) per tile, not per agent)
        count = 0
        for pos_key in wrapped_positions:
            if pos_key in self.position_map:
                count += len(self.position_map[pos_key])

        return count

    def redist_food(self):
        """Add food to the grid based on resupply amount."""
        self._add_food(self.food_resupply_amount)

    def update_agent_position(self, agent_id, old_position, new_position):
        """
        Update the position map when an agent moves.

        Args:
            agent_id: ID of the agent that moved
            old_position (tuple): Previous (x, y) position
            new_position (tuple): New (x, y) position
        """
        # Remove from old position
        if old_position in self.position_map:
            self.position_map[old_position].discard(agent_id)
            if not self.position_map[old_position]:
                del self.position_map[old_position]

        # Add to new position
        if new_position not in self.position_map:
            self.position_map[new_position] = set()
        self.position_map[new_position].add(agent_id)

    def _batch_decide(self, agents, batch_perceptions):
        """
        Run batched decision-making for all agents on GPU.

        Args:
            agents (list): List of Agent objects
            batch_perceptions (torch.Tensor): Batched perception tensor [N, 5]

        Returns:
            list: List of action indices for each agent
        """
        # Check for critical survival rules per agent
        actions = []
        batch_indices_for_brain = []

        for i, agent in enumerate(agents):
            perception = batch_perceptions[i]
            energy, food, agents_nearby, visibility, movement = perception

            # Critical: Low energy and no food available -> must move to find food
            if energy < 0.25 and food == 0:
                actions.append(0)  # ACTION_MOVE
                continue

            # Too much competition for available food -> move to find better location
            if food > 0 and agents_nearby > (food * 2 + 1):
                actions.append(0)  # ACTION_MOVE
                continue

            # This agent needs brain decision
            actions.append(None)
            batch_indices_for_brain.append(i)

        # If we have agents that need brain decisions, batch them
        if batch_indices_for_brain:
            # Run forward pass for all agents needing brain decisions
            batch_outputs = []
            for idx in batch_indices_for_brain:
                agent = agents[idx]
                perception = batch_perceptions[idx : idx + 1].to(self.device)
                agent.brain.eval()
                with t.no_grad():
                    output = agent.brain(perception)
                batch_outputs.append(output)

            # Stack and get argmax for all
            if batch_outputs:
                stacked_outputs = t.cat(batch_outputs, dim=0)
                brain_actions = t.argmax(stacked_outputs, dim=1).cpu().tolist()

                # Fill in the brain decisions
                for i, idx in enumerate(batch_indices_for_brain):
                    actions[idx] = brain_actions[i]

        return actions

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

        # Filter alive agents and do basic updates (age, metabolism, death checks)
        alive_agents = []
        for agent in self.agents:
            if not agent.is_alive():
                continue

            # Natural aging and random lifespan limit
            agent.age += 1
            if agent.reached_natural_death():
                agent.kill_agent()
                self._record_lifespan(agent)
                continue

            # Metabolism drains energy; death here also counts toward lifespan stats
            metabolism = agent.get_trait("metabolism") or 1.0
            agent.consume_energy(0.1 * metabolism)
            if not agent.is_alive():
                self._record_lifespan(agent)
                continue

            alive_agents.append(agent)

        if not alive_agents:
            return

        # Separate agents into those who act this turn vs those skipping
        acting_agents = []
        skipping_agents = []
        for agent in alive_agents:
            if agent.should_skip():
                skipping_agents.append(agent)
            else:
                acting_agents.append(agent)

        # BATCH PERCEPTION: Gather perceptions only for acting agents
        perceptions = []
        for agent in acting_agents:
            perception = agent.perceive(self)
            perceptions.append(perception)

        # Stack into batch tensor (only if we have acting agents)
        if acting_agents:
            batch_perceptions = t.cat(perceptions, dim=0)

            # BATCH DECISION: Run all brains together on GPU
            batch_actions = self._batch_decide(acting_agents, batch_perceptions)
        else:
            batch_actions = []

        # Update all agents and collect offspring
        new_agents = []

        for i, agent in enumerate(acting_agents):
            old_position = agent.position
            action = batch_actions[i]

            # Execute the action
            offspring = agent.execute_action(action, self)

            # Update position map if agent moved
            if agent.position != old_position:
                self.update_agent_position(agent.get_id(), old_position, agent.position)

            # Collect offspring
            if offspring:
                new_agents.append(offspring)

        # Add new offspring to population
        for offspring in new_agents:
            self.agents.append(offspring)
            offspring_id = offspring.get_id()
            self.agents_by_id[offspring_id] = offspring
            offspring_pos = offspring.position
            if offspring_pos not in self.position_map:
                self.position_map[offspring_pos] = set()
            self.position_map[offspring_pos].add(offspring_id)

        # Remove dead agents
        dead_agents = [agent for agent in self.agents if not agent.is_alive()]
        for agent in dead_agents:
            self._record_lifespan(agent)
            agent_id = agent.get_id()
            position = agent.position

            # Remove from position map
            if position in self.position_map:
                self.position_map[position].discard(agent_id)
                if not self.position_map[position]:
                    del self.position_map[position]

            # Remove from agents_by_id
            if agent_id in self.agents_by_id:
                del self.agents_by_id[agent_id]

        # Remove from agents list
        self.agents = [agent for agent in self.agents if agent.is_alive()]

    def get_stats(self):
        """
        Get current simulation statistics.

        Returns:
            dict: Statistics about the current state
        """
        total_food = sum(biome.get_food_units() for _, _, biome in self.iter_biomes())

        # Count alive agents from agents_by_id for O(1) per agent
        alive_count = sum(1 for agent in self.agents_by_id.values() if agent.is_alive())
        total_energy = sum(agent.energy for agent in self.agents_by_id.values() if agent.is_alive())

        # Calculate lifespan statistics
        median_lifespan = 0
        min_age = 0
        max_age = 0
        if self.lifespans:
            sorted_lifespans = sorted(self.lifespans)
            n = len(sorted_lifespans)
            median_lifespan = (
                sorted_lifespans[n // 2]
                if n % 2 == 1
                else (sorted_lifespans[n // 2 - 1] + sorted_lifespans[n // 2]) / 2
            )
            min_age = sorted_lifespans[0]
            max_age = sorted_lifespans[-1]

        return {
            "step": self.step_count,
            "alive_agents": alive_count,
            "total_food": total_food,
            "avg_energy": total_energy / alive_count if alive_count > 0 else 0,
            "median_lifespan": median_lifespan,
            "min_age": min_age,
            "max_age": max_age,
        }

    def get_average_genome_traits(self):
        """
        Calculate average values for all genome traits across living population.

        Returns:
            dict: Dictionary mapping trait names to their average values
        """
        if not self.agents_by_id:
            return {}

        alive_agents = [agent for agent in self.agents_by_id.values() if agent.is_alive()]
        if not alive_agents:
            return {}

        # Get all trait names from first agent
        trait_names = list(alive_agents[0].genome.traits.keys())

        # Calculate averages
        avg_traits = {}
        for trait_name in trait_names:
            values = [agent.get_trait(trait_name) or 0.0 for agent in alive_agents]
            avg_traits[trait_name] = sum(values) / len(values) if values else 0.0

        return avg_traits

    def log_stats(self, stats, state_dict):
        """
        Log statistics to a state dictionary for graphing.

        Args:
            stats (dict): Statistics dictionary with keys: step, alive_agents, total_food, avg_energy, median_lifespan, min_age, max_age
            state_dict (dict): Dictionary to accumulate stats over time

        Returns:
            dict: Updated state dictionary
        """
        step = stats["step"]
        state_dict[step] = {
            "alive_agents": stats["alive_agents"],
            "total_food": stats["total_food"],
            "avg_energy": stats["avg_energy"],
            "median_lifespan": stats["median_lifespan"],
            "min_age": stats["min_age"],
            "max_age": stats["max_age"],
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
                # Use position_map for O(1) lookup
                position = (x, y)
                agents_here = len(self.position_map.get(position, set()))
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
                # Use position_map for O(1) lookup
                position = (x, y)
                agents_here = len(self.position_map.get(position, set()))
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

    def is_tile_full(self, x, y):
        """
        Check if a tile has reached maximum agent capacity.

        Args:
            x (int): X coordinate
            y (int): Y coordinate

        Returns:
            bool: True if tile is full, False otherwise
        """
        position = (x, y)
        agents_here = len(self.position_map.get(position, set()))
        return agents_here >= MAX_AGENTS_PER_TILE


if __name__ == "__main__":
    # For backwards compatibility, run with default settings
    from cli.cli_sim_starter import main

    main()
