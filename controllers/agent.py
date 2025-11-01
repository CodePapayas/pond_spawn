import torch as t
import random as r
from pathlib import Path
import numpy as np

from controllers.brain import Brain


# Action constants - brain output with highest value determines action
ACTION_MOVE = 0  # Move in the direction of the agent's heading
ACTION_TURN = 1  # Turn 90 degrees clockwise
ACTION_EAT = 2  # Attempt to eat food at current position
ACTION_REPRODUCE = 3  # Attempt to reproduce (costs 25% of current energy)
# Action 4
# Action 5
# Action 6
# Action 7

# Directional headings - where the agent can point
headings = [0, 1, 2, 3]


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

    #

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
        self.heading = r.choice(headings)  # 0=North, 1=East, 2=South, 3=West

        # Create and configure brain
        brain_config_path = (
            Path(__file__).resolve().parent.parent / "brains" / "brain.json"
        )
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
        terrain_speed = biome.get_movement_speed()
        visibility = biome.get_visibility()

        # Count nearby agents based on visual ability
        visual_range = self.get_trait("vision") or 1.0
        nearby_agents = environment.count_agents_in_range(self.position, visual_range)

        # Determine how good the agents vision is
        agent_vision = (
            nearby_agents / (visibility * visual_range)
            if (visibility * visual_range) > 0
            else 0
        )

        # Determine how well the agent moves on the terrain
        speed_stat = self.get_trait("speed") or 1.0
        movement_speed = terrain_speed * speed_stat

        # Normalize inputs for processing (all values clipped to [0.0, 1.0])
        normalized_energy = np.clip(self.energy / 100.0, 0.0, 1.0)
        normalized_food = np.clip(food_available / 3.0, 0.0, 1.0)  # Food ranges 0-3
        normalized_agent_count = np.clip(nearby_agents / 10.0, 0.0, 1.0)
        normalized_visibility = np.clip(agent_vision, 0.0, 1.0)
        normalized_movement = np.clip(movement_speed, 0.0, 1.0)

        # Return what was perceived
        perception = [
            normalized_energy,
            normalized_food,
            normalized_agent_count,
            normalized_visibility,
            normalized_movement,
        ]

        return t.tensor([perception], dtype=t.float32)

    def decide(self, perception):
        """
        Use the brain to make a decision based on perception.

        Uses hard-coded survival rules for critical situations, otherwise
        uses winner-takes-all neural network decision.

        Critical survival rules:
        - Low energy + no food -> MOVE (search for food)
        - Too many competing agents -> MOVE (avoid competition)

        Args:
            perception (torch.Tensor): Sensory input tensor

        Returns:
            int: Action index (0=MOVE, 1=TURN, 2=EAT, 3=REPRODUCE)
        """
        energy, food, agents, visibility, movement = perception[0]

        # Critical: Low energy and no food available -> must move to find food
        if energy < 0.25 and food == 0:
            return ACTION_MOVE

        # Too much competition for available food -> move to find better location
        if food > 0 and agents > (food * 2 + 1):
            return ACTION_MOVE

        # Otherwise, let the brain decide using winner-takes-all
        self.brain.eval()  # Set to evaluation mode
        with t.no_grad():
            output = self.brain(perception)

        # Winner-takes-all: select action with highest output value
        action = t.argmax(output).item()

        return action

    def move(self, environment):
        """
        Move the agent in the direction of its current heading.

        Movement is affected by:
        - Current heading (0=North, 1=East, 2=South, 3=West)
        - Biome movement speed modifier
        - Grid boundaries
        - Energy cost based on speed trait and metabolism

        Args:
            environment: The simulation environment
        """
        x, y = self.position
        terrain_speed = environment.get_biome(x, y).get_movement_speed()

        # Determine new position based on heading
        # 0=North (y-1), 1=East (x+1), 2=South (y+1), 3=West (x-1)
        if self.heading == 0:  # North
            new_x, new_y = x, y - 1
        elif self.heading == 1:  # East
            new_x, new_y = x + 1, y
        elif self.heading == 2:  # South
            new_x, new_y = x, y + 1
        else:  # West (heading == 3)
            new_x, new_y = x - 1, y

        # Check if new position is within bounds
        if 0 <= new_x < environment.grid_size and 0 <= new_y < environment.grid_size:
            self.position = (new_x, new_y)

            # Energy cost for movement (affected by speed and metabolism)
            speed = self.get_trait("speed") or 1.0
            metabolism = self.get_trait("metabolism") or 1.0
            movement_cost = terrain_speed * speed * metabolism
            self.consume_energy(movement_cost)

    def turn(self):
        """
        Turn the agent 90 degrees clockwise.

        Heading transitions: North -> East -> South -> West -> North
        Energy cost affected by metabolism trait.
        """
        self.heading = (self.heading + 1) % 4
        # Energy cost for turning (affected by metabolism)
        metabolism = self.get_trait("metabolism") or 1.0
        self.consume_energy(0.1 * metabolism)

    def eat(self, environment):
        """
        Attempt to eat food at current position.

        Agent will eat until either:
        - Food is depleted
        - Agent reaches energy capacity

        If multiple agents on same tile, faster agents eat first.

        Args:
            environment: The simulation environment

        Returns:
            bool: True if food was consumed
        """
        x, y = self.position
        biome = environment.get_biome(x, y)
        food_available = biome.get_food_units()

        if food_available <= 0:
            return False

        # Calculate energy capacity
        energy_capacity_trait = self.get_trait("energy_capacity") or 1.0
        max_energy = 100.0 * energy_capacity_trait

        # Calculate how much energy we can consume
        energy_needed = max_energy - self.energy

        if energy_needed > 0:
            # Consume 1 food unit (provides energy)
            food_energy_value = 25.0  # Each food unit provides 25 energy
            energy_gained = min(food_energy_value, energy_needed)

            self.add_energy(energy_gained)
            biome.features["food_units"] = food_available - 1
            return True

        return False

    def reproduce(self, environment):
        """
        Attempt to reproduce if energy threshold is met.

        Reproduction rules:
        - Requires minimum energy threshold (50 energy)
        - Costs 40% of current energy
        - Creates a mutated offspring at nearby position
        - Offspring gets the energy cost as starting energy

        Args:
            environment: The simulation environment

        Returns:
            Agent or None: New offspring agent if reproduction successful
        """
        x, y = self.position
        biome_locale = environment.get_biome(x, y)

        # Need minimum energy to reproduce
        if self.energy < 50:
            return None

        # Can't reproduce on empty food tiles (not sustainable)
        if biome_locale.get_food_units() == 0:
            return None

        # Calculate reproduction cost (40% of current energy)
        reproduction_cost = self.energy * 0.40

        # Deduct reproduction cost from parent
        self.consume_energy(reproduction_cost)

        # Create mutated genome from parent
        offspring_genome = self.genome.mutate()

        # Try to find a nearby valid position for offspring
        possible_positions = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        # Filter valid positions (within bounds)
        valid_positions = [
            pos
            for pos in possible_positions
            if 0 <= pos[0] < environment.grid_size
            and 0 <= pos[1] < environment.grid_size
        ]

        if not valid_positions:
            # No valid position, reproduction fails but energy already spent
            return None

        # Choose random valid position
        offspring_position = r.choice(valid_positions)

        # Create offspring with starting energy from parent
        offspring = Agent(offspring_genome, offspring_position)
        offspring.energy = reproduction_cost  # Offspring gets the energy parent spent

        return offspring

    def update(self, environment):
        """
        Main update method called each simulation step.

        Process:
        1. Age the agent
        2. Consume base metabolic energy (affected by metabolism trait)
        3. Get perception from environment
        4. Use brain to decide action (winner-takes-all)
        5. Execute the chosen action (all actions cost energy, scaled by metabolism)
        6. Return offspring if reproduction occurred

        Args:
            environment: The simulation environment

        Returns:
            Agent or None: New offspring if reproduction occurred
        """
        # Increment age
        self.age += 1

        # Base metabolic cost (just for staying alive)
        metabolism = self.get_trait("metabolism") or 1.0
        self.consume_energy(0.1 * metabolism)

        if not self.is_alive():
            return None

        # Get perception and make decision
        perception = self.perceive(environment)
        action = self.decide(perception)

        # Execute action based on winner-takes-all decision
        offspring = None

        if action == ACTION_MOVE:
            self.move(environment)
        elif action == ACTION_TURN:
            self.turn()
        elif action == ACTION_EAT:
            self.eat(environment)
        elif action == ACTION_REPRODUCE:
            offspring = self.reproduce(environment)

        return offspring

    def update_without_eating(self, environment):
        """
        Update method that defers eating to allow speed-based priority.

        This version returns the action so the environment can process
        eating in speed-priority order. If eating fails (no food), agent
        will move instead.

        Args:
            environment: The simulation environment

        Returns:
            tuple: (action_index, offspring or None)
        """
        # Increment age
        self.age += 1

        # Base metabolic cost (just for staying alive)
        metabolism = self.get_trait("metabolism") or 1.0
        self.consume_energy(0.15 * metabolism)

        if not self.is_alive():
            return (None, None)

        # Get perception and make decision
        perception = self.perceive(environment)
        action = self.decide(perception)

        # Execute action based on winner-takes-all decision
        # NOTE: Eating is deferred to allow speed-based priority
        offspring = None

        if action == ACTION_MOVE:
            self.move(environment)
            self.consume_energy(0.08 * metabolism)
        elif action == ACTION_TURN:
            self.turn()
            self.consume_energy(0.04 * metabolism)
        elif action == ACTION_REPRODUCE:
            offspring = self.reproduce(environment)
            self.consume_energy(0.1 * metabolism)
        elif action == ACTION_EAT:
            # Check if eating is possible before returning the action
            x, y = self.position
            biome = environment.get_biome(x, y)
            if biome.get_food_units() <= 0:
                # No food available - move instead
                action = ACTION_MOVE
                self.move(environment)
                self.consume_energy(0.08 * metabolism)

        return (action, offspring)

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
