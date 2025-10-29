import random as r
from pathlib import Path
import time
import copy as c

from controllers.landscape import Biome
from controllers.genome import Genome
from controllers.agent import Agent


# Global variables
POPULATION = 50
FOOD_RESUPPLY = 10
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
            food_units (int): Total amount of food to distribute across the grid
        """
        self.grid_size = grid_size
        self.grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.agents = []
        self.step_count = 0
        
        self._initialize_biomes()
        self._distribute_food(food_units)
        self._spawn_agents(num_agents)
    
    def _initialize_biomes(self):
        """Generate random biomes for each tile in the grid."""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                biome = Biome().generate()
                self.grid[x][y] = biome
    
    def _distribute_food(self, total_food):
        """
        Distribute food units randomly across the grid.
        
        Args:
            total_food (int): Total food units to distribute
        """
        for _ in range(total_food):
            x = r.randint(0, self.grid_size - 1)
            y = r.randint(0, self.grid_size - 1)
            biome = self.grid[x][y]
            
            # Get current food units
            current_food = biome.get_food_units()
            if isinstance(current_food, int):
                biome.features["food_units"] = current_food + 1
            else:
                biome.features["food_units"] = 1
    
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
                r.randint(0, self.grid_size - 1)
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
    
    def count_agents_in_range(self, position, radius):
        """
        Count the number of agents within a given radius of a position.
        
        Uses Manhattan distance (grid-based distance).
        
        Args:
            position (tuple): Center position (x, y)
            radius (float): Search radius
            
        Returns:
            int: Number of agents within radius (excluding self at exact position)
        """
        x, y = position
        count = 0
        
        for agent in self.agents:
            ax, ay = agent.position
            # Calculate Manhattan distance
            distance = abs(ax - x) + abs(ay - y)
            
            # Count agents within radius (but not at exact same position)
            if distance > 0 and distance <= radius:
                count += 1
        
        return count
    
    def redist_food(self, amount):
        if amount < 0 or type(amount) is not int:
            print("INVALID FOOD QUANTITY: SIMULATION TERMINATED")

        self._distribute_food(amount)
    
    def step(self):
        """
        Execute one simulation step.
        
        Process:
        1. Update all agents (age, metabolism, decision making, movement, turning)
        2. Process eating in speed-priority order (faster agents eat first)
        3. Add offspring from reproduction
        4. Remove dead agents
        """
        self.step_count += 1

        # Replenish food
        self.redist_food(FOOD_RESUPPLY)
        
        # Update all agents and collect offspring
        new_agents = []
        agents_wanting_to_eat = []  # Track agents that chose to eat
        

        for agent in self.agents:
            # Agent updates but doesn't eat yet
            action, offspring = agent.update_without_eating(self)
            
            # Track if agent wants to eat
            if action == 2:  # ACTION_EAT
                agents_wanting_to_eat.append(agent)
            
            # Collect offspring
            if offspring:
                new_agents.append(offspring)
        
        # Process eating in speed-priority order
        # Sort agents by speed (highest first)
        agents_wanting_to_eat.sort(key=lambda a: a.get_trait("speed") or 1.0, reverse=True)

        agents_in_line = 0

        for agent in agents_wanting_to_eat:
            agents_in_line += 1
        
        print(f"# of agents in line to eat: ", {agents_in_line})
        
        # Let agents eat in order of speed
        for agent in agents_wanting_to_eat:
            if agent.is_alive():  # Only if still alive after metabolism
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
        total_food = sum(
            self.grid[x][y].get_food_units() 
            for x in range(self.grid_size) 
            for y in range(self.grid_size)
        )
        
        return {
            "step": self.step_count,
            "population": len(self.agents),
            "alive_agents": len([a for a in self.agents if a.is_alive()]),
            "total_food": total_food,
            "avg_energy": sum(a.energy for a in self.agents) / len(self.agents) if self.agents else 0,
        }
    
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
        RED = '\033[91m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        
        if label:
            print(f"\n=== {label} ===")
        
        for row in grid_state:
            cells = []
            for agents_here, food in row:
                # Color code agents: 0=red, 1=yellow, 2+=green
                if agents_here == 0:
                    agent_color = RED
                elif agents_here == 1:
                    agent_color = YELLOW
                else:
                    agent_color = GREEN
                
                # Color code food: 0=red, 1=yellow, 2+=green
                if food == 0:
                    food_color = RED
                elif food == 1:
                    food_color = YELLOW
                else:
                    food_color = GREEN
                
                cell = f"{agent_color}A:{agents_here}{RESET} {food_color}F:{food}{RESET}"
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
        RED = '\033[91m'
        YELLOW = '\033[93m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        
        print(f"\n=== Step {self.step_count} ===")
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                agents_here = len(self.get_agents_at(x, y))
                food = self.grid[x][y].get_food_units()
                
                # Color code agents: 0=red, 1=yellow, 2+=green
                if agents_here == 0:
                    agent_color = RED
                elif agents_here == 1:
                    agent_color = YELLOW
                else:
                    agent_color = GREEN
                
                # Color code food: 0=red, 1=yellow, 2+=green
                if food == 0:
                    food_color = RED
                elif food == 1:
                    food_color = YELLOW
                else:
                    food_color = GREEN
                
                cell = f"{agent_color}A:{agents_here}{RESET} {food_color}F:{food}{RESET}"
                row.append(cell)
            print(" | ".join(row))
        print()


if __name__ == "__main__":
    # Create simulation environment
    env = Environment(grid_size=10, num_agents=POPULATION, food_units=10)
    
    # Capture initial state
    initial_grid = env.capture_grid_state()
    
    # Print initial state
    print("Initial Environment State:")
    stats = env.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    env.print_grid_state(initial_grid, "Initial Grid")
    
    # Run a few simulation steps
    print("\nRunning simulation...")
    for i in range(TICKS):
        env.step()
        stats = env.get_stats()
        print(f"Step {stats['step']}: Population={stats['population']}, Food={stats['total_food']}")
        print("Current simulation stats: ")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("Current grid: ")
        env.print_grid()
        time.sleep(0.001)
    
    # Print comparison
    print("\n" + "="*50)
    env.print_grid_state(initial_grid, "Initial Grid")
    final_grid = env.capture_grid_state()
    env.print_grid_state(final_grid, "Final Grid")
