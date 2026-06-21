"""Agent behavior and lifecycle logic.

Defines the :class:`~controllers.agent.Agent` type and related constants used by
the simulation runtime. Agents perceive the environment, decide on actions via
their neural network, and execute actions that affect their energy, position,
and reproduction.
"""

import random as r
from pathlib import Path

import numpy as np
import torch as t

from controllers.brain import Brain

# Action constants - brain output is softmax-sampled to determine action
ACTION_MOVE = 0  # Move in the direction of the agent's heading
ACTION_TURN = 1  # Turn 90 degrees clockwise
ACTION_EAT = 2  # Attempt to eat food at current position
ACTION_REPRODUCE = 3  # Attempt to reproduce (costs 25% of current energy)
ACTION_SLEEP = 4  # Rest; Burns energy, but less than anything else
ACTION_NOTHING = 5  # Agent can choose to do nothing.
ACTION_TURN_COUNTER = 6  # Agent can turn 90 degrees counter-clockwise
ACTION_ATTACK = 7  # Agent can attempt to absorb another agent

MATURITY_AGE = 100  # ticks before an agent can reproduce
CHILDHOOD_TICKS = 50  # ticks of boosted defense and parent damage routing
BIRTH_FAIL_CHANCE = 0.02  # probability a reproduction attempt produces no offspring
FAIL_COUNTS_CHANCE = 0.20  # on a failed birth, chance it still burns one offspring slot

# Directional headings - where the agent can point
headings = [0, 1, 2, 3]


def create_death_range(size=200, early_death_chance=0.15, late_death_start=500):
    """
    Create death probability range with mostly zeros.
    - early_death_chance: probability of early death values
    - late_death_start: when death becomes certain
    """
    death_range = []

    for i in range(size):
        if i < 5 and r.random() < early_death_chance:
            # Small chance of very early death
            death_range.append(r.randint(50, 150))
        elif 15 < i < 20 and r.random() < 0.05:
            # Tiny chance in middle age
            death_range.append(r.randint(200, 400))
        else:
            death_range.append(500 + (i + late_death_start) // 4)

    return death_range


class Agent:  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """
    Represents an individual agent in the simulation.

    Each agent has:
    - A unique genome defining its traits and neural network weights
    - A brain (neural network) for decision making
    - A position in the environment grid
    - Energy level for survival
    - Internal state for tracking history and observations
    """

    death_range = create_death_range()

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
        self.alive = True
        self.id = genome.id
        self.parent = self.id
        self.parent_id = None  # set to parent's genome id at birth
        self.parent_defense_bonus = 0.0  # extra defense inherited from parent during childhood
        self.death_age = self._assign_death_age()
        self.cause_of_death = None
        self.skip_turn = False  # Flag to skip next turn after doing nothing

        self.max_offspring = r.randint(1, 10)
        self.offspring_count = 0
        reproductive_window = max((self.death_age or 0) - MATURITY_AGE, 1)
        self.reproduction_cooldown = reproductive_window // self.max_offspring
        self.last_reproduced_age = None

        # Create and configure brain (will be moved to GPU by environment for batching)
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

        biome = environment.get_biome(*self.position)
        food_available = biome.get_food_units()
        visibility = biome.get_visibility()

        visual_range = self.get_trait("vision") or 1.0
        nearby_agents = environment.count_agents_in_range(self.position, visual_range)

        denom = visibility * visual_range
        normalized_visibility = np.clip(nearby_agents / denom if denom > 0 else 0.0, 0.0, 1.0)

        normalized_movement = np.clip(
            biome.get_movement_speed() * (self.get_trait("speed") or 1.0),
            0.0,
            1.0,
        )

        perception = [
            np.clip(self.energy / 100.0, 0.0, 1.0),
            np.clip(food_available / 5.0, 0.0, 1.0),
            np.clip(nearby_agents / 10.0, 0.0, 1.0),
            normalized_visibility,
            normalized_movement,
        ]

        return t.tensor([perception], dtype=t.float32)

    def decide(self, perception):
        """
        Use the brain to make a decision based on perception.

        Uses hard-coded survival rules for critical situations, otherwise
        uses softmax-sampled neural network decision.

        Critical survival rules:
        - Low energy + no food -> MOVE (search for food)
        - Too many competing agents -> MOVE (avoid competition)

        Args:
            perception (torch.Tensor): Sensory input tensor

        Returns:
            int: Action index (0=MOVE, 1=TURN, 2=EAT, 3=REPRODUCE)
        """
        # energy, food, agents, visibility, movement = perception[0]

        # Otherwise, let the brain decide using softmax sampling
        self.brain.eval()  # Set to evaluation mode
        with t.no_grad():
            output = self.brain(perception)

        # Softmax sampling: sample action proportional to output probabilities
        probs = t.softmax(output, dim=-1)
        action = t.multinomial(probs, num_samples=1).item()

        return action

    def move(self, environment):
        """
        Move the agent in the direction of its current heading.

        Movement is affected by:
        - Current heading (0=North, 1=East, 2=South, 3=West)
        - Biome movement speed modifier
        - Wrapping edges (toroidal map)
        - Energy cost based on speed trait and metabolism

        Args:
            environment: The simulation environment
        """
        x, y = self.position
        terrain_speed = environment.get_biome(x, y).get_movement_speed()
        heading = self.get_heading()

        # Determine new position based on heading
        # 0=North (y-1), 1=East (x+1), 2=South (y+1), 3=West (x-1)
        if heading == 0:  # North
            new_x, new_y = x, y - 1
        elif heading == 1:  # East
            new_x, new_y = x + 1, y
        elif heading == 2:  # South
            new_x, new_y = x, y + 1
        else:  # West (heading == 3)
            new_x, new_y = x - 1, y

        # Wrap around edges (toroidal map)
        new_x = new_x % environment.grid_size
        new_y = new_y % environment.grid_size
        self.position = (new_x, new_y)

        # Energy cost for movement (affected by speed and metabolism)
        speed = self.get_trait("speed") or 1.0
        metabolism = self.get_trait("metabolism") or 1.0
        movement_cost = terrain_speed * speed * metabolism * 0.15
        self.consume_energy(movement_cost)

    def turn(self):
        """
        Turn the agent 90 degrees clockwise.

        Heading transitions: North -> East -> South -> West -> North
        Energy cost affected by metabolism trait.
        """
        self.heading = (self.get_heading() + 1) % 4
        # Energy cost for turning (affected by metabolism)
        metabolism = self.get_trait("metabolism") or 1.0
        self.consume_energy(0.1 * metabolism)

    def turn_left(self):
        """
        Turn the agent 90 degrees counter-clockwise.

        Heading transitions: North -> West -> South -> East -> North
        Energy cost affected by metabolism trait.
        """
        self.heading = (self.get_heading() - 1) % 4
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
            food_energy_value = 33.3  # Each food unit provides 33.3 energy
            energy_gained = min(food_energy_value, energy_needed)

            self.add_energy(energy_gained)
            biome.features["food_units"] = food_available - 1
            return True

        return False

    def get_effective_defense(self):
        """Defense linearly blended from (own + parent) at age 0 down to own at CHILDHOOD_TICKS."""
        own = self.get_trait("defense")
        if self.age >= CHILDHOOD_TICKS or self.parent_defense_bonus == 0.0:
            return own
        ratio = self.age / CHILDHOOD_TICKS
        return own + self.parent_defense_bonus * (1.0 - ratio)

    def eat_agent(self, target_agent):
        """Win combat: gain 12.5% of target's energy, kill target."""
        energy_capacity_trait = self.get_trait("energy_capacity") or 1.0
        max_energy = 100.0 * energy_capacity_trait
        energy_gained = target_agent.energy * 0.125
        self.energy = min(self.energy + energy_gained, max_energy)
        target_agent.add_cause_of_death("Killed in combat")
        target_agent.kill_agent()

    def attack(self, target_agent):
        """
        Attempt to attack a target agent.

        Requires aggression >= 0.80 to initiate.
        Outcome based on attack vs defense ratio:
          - attack > defense * 0.66: attacker wins
          - attack > defense * 0.33: 50/50
          - attack <= defense * 0.33: defender wins
        Costs 0.2 * metabolism energy to initiate.

        Returns:
            bool: True if attack was attempted, False if aggression too low
        """
        agent_attack = self.get_trait("attack")
        agent_aggression = self.get_trait("aggression")
        target_defense = target_agent.get_effective_defense()

        if agent_aggression is None or agent_aggression < 0.80:
            return False

        metabolism = self.get_trait("metabolism") or 1.0
        self.consume_energy(0.2 * metabolism)

        if agent_attack > target_defense * 0.66:
            self.eat_agent(target_agent)
            return True

        if agent_attack > target_defense * 0.33:
            if r.random() >= 0.5:
                self.eat_agent(target_agent)
            else:
                target_agent.eat_agent(self)
            return True

        target_agent.eat_agent(self)
        return True

    def reproduce(self, environment):
        """
        Attempt to reproduce if energy threshold is met.

        Reproduction rules:
        - Requires minimum energy threshold (40 energy)
        - Costs 50% of current energy (× reproduction_cost trait)
        - Creates a mutated offspring at nearby position
        - Offspring gets the energy cost as starting energy

        Args:
            environment: The simulation environment

        Returns:
            Agent or None: New offspring agent if reproduction successful
        """

        if self.age < MATURITY_AGE:
            return None
        if self.energy < 40:
            return None
        if self.offspring_count >= self.max_offspring:
            return None
        if self.last_reproduced_age is not None:
            if self.age - self.last_reproduced_age < self.reproduction_cooldown:
                return None

        x, y = self.position
        procreation_modifier = self.get_trait("reproduction_cost")

        # Calculate and pay energy cost before outcome is decided
        reproduction_cost = self.energy * (0.50 * procreation_modifier)
        self.consume_energy(reproduction_cost)

        # Small chance birth fails entirely
        if r.random() < BIRTH_FAIL_CHANCE:
            # Rare: failed birth still burns a slot (miscarriage-equivalent)
            if r.random() < FAIL_COUNTS_CHANCE:
                self.offspring_count += 1
                self.last_reproduced_age = self.age
            return None

        # Successful birth
        self.offspring_count += 1
        self.last_reproduced_age = self.age

        offspring_genome = self.genome.mutate()

        grid_size = environment.grid_size
        possible_positions = [
            ((x + 1) % grid_size, y),
            ((x - 1) % grid_size, y),
            (x, (y + 1) % grid_size),
            (x, (y - 1) % grid_size),
        ]
        empty_positions = [p for p in possible_positions if p not in environment.position_map]
        offspring_position = r.choice(empty_positions if empty_positions else possible_positions)

        offspring = Agent(offspring_genome, offspring_position)
        offspring.energy = reproduction_cost
        offspring.parent = self.id
        offspring.parent_id = self.id
        offspring.parent_defense_bonus = self.get_trait("defense")

        return offspring

    def sleep(self, metabolism: float):
        """
        Make da lil guys sleep
        """
        energy_gain = 0.15 * metabolism
        self.add_energy(energy_gain)

    def get_energy_lvl(self):
        """Return the agent's current energy level."""
        return self.energy

    def consume_energy(self, amount: float):
        """
        Reduce agent's energy by the specified amount.

        Args:
            amount (float): Energy to consume
        """
        self.energy -= amount

    def add_energy(self, amount: float):
        """
        Increase agent's energy by the specified amount.

        Args:
            amount (float): Energy to add
        """
        energy_capacity_trait = self.get_trait("energy_capacity") or 1.0
        max_energy = 100.0 * energy_capacity_trait
        self.energy = min(self.energy + amount, max_energy)

    def is_alive(self):
        """
        Check if agent is still alive.

        An agent is alive if:
        - Energy is positive
        - Has not exceeded maximum age (optional)

        Returns:
            bool: True if agent is alive
        """
        if self.energy <= 0:
            if self.cause_of_death not in (
                "Reached assigned death age",
                "Eaten alive",
                "Killed in combat",
            ):
                self.add_cause_of_death("Died of starvation")
            self.alive = False
            return False

        return True

    def kill_agent(self):
        """
        Kills da agent
        """
        self.energy = 0
        self.alive = False

    def get_trait(self, trait_name: str):
        """
        Get a specific trait value from the genome.

        Args:
            trait_name (str): Name of the trait

        Returns:
            float or None: Trait value if it exists
        """
        trait_info = self.genome.traits.get(trait_name, {})
        return trait_info.get("value")

    def get_heading(self):
        """
        Get da heading
        """
        return self.heading

    def get_id(self):
        """
        Return the agents id; Generated by the genome
        """
        return self.id

    def execute_action(self, action, environment):
        """
        Execute a pre-determined action (from batched decision-making).

        Args:
            action (int): The action to execute
            environment: The simulation environment

        Returns:
            Agent or None: offspring if reproduction occurred
        """
        metabolism = self.get_trait("metabolism") or 1.0
        offspring = None

        if action == ACTION_MOVE:
            self.move(environment)
        elif action == ACTION_TURN:
            self.turn()
            self.consume_energy(0.04 * metabolism)
        elif action == ACTION_REPRODUCE:
            offspring = self.reproduce(environment)
            self.consume_energy(0.1 * metabolism)
        elif action == ACTION_SLEEP:
            self.sleep(metabolism)
        elif action == ACTION_NOTHING:
            self.loaf_around()
        elif action == ACTION_EAT:
            self.eat(environment)
        elif action == ACTION_TURN_COUNTER:
            self.turn_left()
            self.consume_energy(0.04 * metabolism)
        elif action == ACTION_ATTACK:
            victim = environment.get_agents_at(*self.position)[0]
            if victim and victim is not self:
                pre_attack_energy = victim.energy
                self.attack_agent(victim)
                if victim.age < CHILDHOOD_TICKS and victim.parent_id:
                    parent = environment.agents_by_id.get(victim.parent_id)
                    if parent and parent.is_alive():
                        parent_ratio = 0.45 * (1.0 - victim.age / CHILDHOOD_TICKS)
                        parent.consume_energy(pre_attack_energy * parent_ratio)

        return offspring

    def loaf_around(self):
        """
        Allow da bois to take a vacation day.
        Sets skip_turn flag so agent skips the next tick.
        """
        self.consume_energy(0.005 * self.get_trait("metabolism"))
        self.skip_turn = True  # Skip next turn after doing nothing

    def should_skip(self):
        """
        Check if agent should skip this turn.
        Resets the flag after checking so agent acts on the following turn.

        Returns:
            bool: True if agent should skip this turn
        """
        if self.skip_turn:
            self.skip_turn = False
            return True
        return False

    def reached_natural_death(self):
        """Check if the agent reached its assigned death age."""
        return self.death_age is not None and self.age >= self.death_age

    def _assign_death_age(self):
        """Assign a single death age using the configured distribution."""
        candidates = [value for value in Agent.death_range if value > 0]
        if not candidates:
            return None
        return r.choice(candidates)

    def add_cause_of_death(self, cod: str):
        """Record a human-readable cause of death string for the agent."""
        if not isinstance(cod, str):
            raise TypeError("Cause of death must be type:str")

        self.cause_of_death = cod

    def attack_agent(self, victim):
        """
        Attempt to absorb energy from a victim agent.

        If the attacker's attack exceeds the victim's defense, the attacker steals a
        fraction of the victim's energy. Otherwise the victim's defense overpowers the
        attacker and the attacker loses energy instead. Either agent is killed if their
        energy reaches zero.
        """
        aggression = self.get_trait("aggression")
        defense, attack = self.get_trait("defense"), self.get_trait("attack")
        v_defense = victim.get_effective_defense()

        metabolism = self.get_trait("metabolism") or 1.0

        if aggression <= 0.55:
            self.consume_energy(0.1)
            return

        # Chosen attacks are costly — agent burns energy just by trying.
        self.consume_energy(0.5 * metabolism)

        if attack > v_defense:
            self.add_energy(victim.get_energy_lvl() * 0.125)
            victim.add_cause_of_death("Eaten alive")
            victim.kill_agent()

        else:
            self.consume_energy(self.get_energy_lvl() * defense)
            if self.get_energy_lvl() == 0:
                self.add_cause_of_death("Eaten alive")
                self.kill_agent()


if __name__ == "__main__":
    dummy_range = create_death_range()
    print(f"{dummy_range}")
