import torch as t
from pathlib import Path

from controllers.brain import Brain
from controllers.genome import Genome
from controllers.landscape import Biome


class Agent:
    """
    Represents an individual agent in the simulation.
    
    Each agent has:
    - A unique genome defining its traits and neural network weights
    - A brain (neural network) for decision making
    - A position in the environment grid
    - Energy level for survival
    - Internal state for tracking history and observations
    """
    
    def __init__(self, genome, position):
        """
        Initialize an agent with a genome and starting position.
        
        Args:
            genome (Genome): The genetic blueprint for this agent
            position (tuple): Starting (x, y) coordinates
        """
        self.genome = genome
        self.position = position
        self.energy = 100.0
        self.age = 0
        
        # Create and configure brain
        brain_config_path = Path(__file__).resolve().parent.parent / "brains" / "brain.json"
        self.brain = Brain(str(brain_config_path))
        self.brain.load_from_genome(genome.to_dict())
    
    def perceive(self, environment):
        """
        Gather sensory information from the environment.
        
        This method should create an input tensor for the brain based on:
        - Local food availability
        - Nearby agents
        - Biome features (visibility, movement speed)
        - Internal state (energy, age)
        
        Args:
            environment: The simulation environment
            
        Returns:
            torch.Tensor: Input tensor for the brain (shape: [1, input_size])
        """

        # Collect observational data
        x, y = self.position
        biome = environment.get_biome(x, y)
        food_available = biome.get_food_units()
        movement_speed = biome.get_movement_speed()
        visibility = biome.get_visibility()

        # Count nearby agents based on visual ability
        visual_range = self.get_trait("vision") or 1.0
        nearby_agents = environment.count_agents_in_range(self.position, visual_range)

        #Normalize inputs for processing
        normalized_energy = self.energy / 100.0
        normalized_food = food_available / 3.0
        normalized_agent_count = min(nearby_agents / 10.0, 1.0)
        normalized_visibility = visibility
        normalized_movement = movement_speed
        
        # Placeholder perception - replace with actual logic
        perception = [
            normalized_energy,
            normalized_food,
            normalized_agent_count,
            normalized_visibility,
            normalized_movement
        ]
        
        return t.tensor([perception], dtype=t.float32)
    
    def decide(self, perception):
        """
        Use the brain to make a decision based on perception.
        
        The brain outputs a decision vector that should be interpreted as:
        - Movement direction (up, down, left, right)
        - Whether to eat
        - Whether to reproduce
        - etc.
        
        Args:
            perception (torch.Tensor): Sensory input tensor
            
        Returns:
            torch.Tensor: Decision output from the brain
        """
        # TODO: Implement decision making
        # Pass perception through brain
        self.brain.eval()  # Set to evaluation mode
        with t.no_grad():
            decision = self.brain(perception)
        
        return decision
    
    def move(self, environment):
        """
        Move the agent based on brain decision and environment constraints.
        
        This method should:
        - Get decision from brain about movement direction
        - Apply biome movement speed modifier
        - Check boundary constraints
        - Update position
        - Consume energy based on movement
        
        Args:
            environment: The simulation environment
        """
        # TODO: Implement movement logic
        # 1. Get perception
        # 2. Make decision with brain
        # 3. Interpret decision as movement direction
        # 4. Check if move is valid (within bounds)
        # 5. Apply movement speed modifier from biome
        # 6. Update position
        # 7. Consume energy
        
        pass
    
    def eat(self, environment):
        """
        Attempt to eat food at current position.
        
        This method should:
        - Check if food is available at current position
        - Consume food and add energy
        - Update biome food count
        
        Args:
            environment: The simulation environment
        """
        # TODO: Implement eating logic
        # 1. Get biome at current position
        # 2. Check if food is available
        # 3. Consume food
        # 4. Add energy to agent
        # 5. Remove food from biome
        
        pass
    
    def reproduce(self, environment):
        """
        Attempt to reproduce if energy threshold is met.
        
        This method should:
        - Check if energy is above reproduction threshold from genome
        - Create mutated offspring genome
        - Spawn new agent at nearby position
        - Reduce parent energy
        
        Args:
            environment: The simulation environment
            
        Returns:
            Agent or None: New offspring agent if reproduction successful
        """
        # TODO: Implement reproduction logic
        # 1. Get clone_energy_threshold from genome
        # 2. Check if current energy exceeds threshold
        # 3. Create mutated genome from parent genome
        # 4. Find valid nearby position for offspring
        # 5. Create new agent with mutated genome
        # 6. Split energy between parent and offspring
        # 7. Return new agent
        
        return None
    
    def update(self, environment):
        """
        Main update method called each simulation step.
        
        This orchestrates all agent behaviors:
        - Perception
        - Decision making
        - Movement
        - Eating
        - Reproduction
        - Energy consumption
        - Aging
        
        Args:
            environment: The simulation environment
            
        Returns:
            Agent or None: New offspring if reproduction occurred
        """
        # TODO: Implement main update loop
        # 1. Increment age
        # 2. Consume base metabolic energy
        # 3. Get perception from environment
        # 4. Make decision with brain
        # 5. Execute actions based on decision (move, eat)
        # 6. Check for reproduction conditions
        # 7. Return offspring if any
        
        self.age += 1
        
        # Base metabolic cost
        metabolism = self.genome.traits.get("metabolism", {}).get("value", 1.0)
        self.consume_energy(0.1 * metabolism)
        
        return None
    
    def consume_energy(self, amount):
        """
        Reduce agent's energy by the specified amount.
        
        Args:
            amount (float): Energy to consume
        """
        self.energy -= amount
    
    def add_energy(self, amount):
        """
        Increase agent's energy by the specified amount.
        
        Args:
            amount (float): Energy to add
        """
        self.energy += amount
    
    def is_alive(self):
        """
        Check if agent is still alive.
        
        An agent is alive if:
        - Energy is positive
        - Has not exceeded maximum age (optional)
        
        Returns:
            bool: True if agent is alive
        """
        return self.energy > 0
    
    def get_trait(self, trait_name):
        """
        Get a specific trait value from the genome.
        
        Args:
            trait_name (str): Name of the trait
            
        Returns:
            float or None: Trait value if it exists
        """
        trait_info = self.genome.traits.get(trait_name, {})
        return trait_info.get("value")


if __name__ == "__main__":
    pass
