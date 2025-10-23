import random as r
from pathlib import Path

from controllers.landscape import Biome
from controllers.genome import Genome
from controllers.agent import Agent


class Environment:
    """
    Represents the simulation environment containing a grid of biomes and agents.
    
    The environment manages:
    - A 2D grid of biome tiles
    - A population of agents
    - Food distribution across the grid
    - Simulation step logic
    """
    
    def __init__(self, grid_size=10, num_agents=100, food_units=50):
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
    
    def step(self):
        """
        Execute one simulation step.
        
        Updates all agents and removes dead ones.
        """
        self.step_count += 1
        
        # Update all agents and collect offspring
        new_agents = []
        for agent in self.agents:
            offspring = agent.update(self)
            if offspring:
                new_agents.append(offspring)
        
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
    
    def print_grid(self):
        """Print a visual representation of the grid."""
        print(f"\n=== Step {self.step_count} ===")
        for y in range(self.grid_size):
            row = []
            for x in range(self.grid_size):
                agents_here = len(self.get_agents_at(x, y))
                food = self.grid[x][y].get_food_units()
                cell = f"A:{agents_here} F:{food}"
                row.append(cell)
            print(" | ".join(row))
        print()


if __name__ == "__main__":
    # Create simulation environment
    env = Environment(grid_size=10, num_agents=100, food_units=50)
    
    # Print initial state
    print("Initial Environment State:")
    stats = env.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Print grid
    env.print_grid()
    
    # Run a few simulation steps
    print("\nRunning simulation...")
    for i in range(5):
        env.step()
        stats = env.get_stats()
        print(f"Step {stats['step']}: Population={stats['population']}, Food={stats['total_food']}")
    
    # Print final grid
    env.print_grid()
