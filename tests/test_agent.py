import copy
import random as r
import sys
from pathlib import Path

import pytest
import torch as t

# Ensure the repository root is on the Python path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controllers.agent import (  # noqa: E402
    ACTION_REPRODUCE,
    Agent,
)
from controllers.brain import Brain  # noqa: E402


class DummyGenome:
    """Lightweight genome object tailored for Agent unit tests."""

    _weight_count = None
    _counter = 0

    def __init__(self):
        if DummyGenome._weight_count is None:
            brain_config = REPO_ROOT / "brains" / "brain.json"
            DummyGenome._weight_count = Brain(str(brain_config)).count_weights()

        DummyGenome._counter += 1
        self.id = f"dummy_{DummyGenome._counter}"
        self.traits = {
            "vision": {"value": 2.0},
            "speed": {"value": 1.5},
            "energy_capacity": {"value": 1.2},
            "metabolism": {"value": 0.8},
            "clone_energy_threshold": {"value": 1.0},
            "mutation_rate": {"value": 0.1},
            "reproduction_cost": {"value": 0.7},
        }
        self.brain_weights = [0.0] * DummyGenome._weight_count

    def to_dict(self):
        return {
            "id": self.id,
            "traits": self.traits,
            "brain_weights": self.brain_weights,
        }

    def mutate(self):
        offspring = DummyGenome()
        offspring.traits = copy.deepcopy(self.traits)
        offspring.brain_weights = list(self.brain_weights)
        return offspring


class MockBiome:
    def __init__(self, *, food_units=0.0, movement_speed=1.0, visibility=1.0):
        self.features = {
            "food_units": food_units,
            "movement_speed": movement_speed,
            "visibility": visibility,
        }

    def get_food_units(self):
        return self.features["food_units"]

    def get_movement_speed(self):
        return self.features["movement_speed"]

    def get_visibility(self):
        return self.features["visibility"]


class MockEnvironment:
    def __init__(self, biome, *, grid_size=10, agents_in_range=0):
        self.biome = biome
        self.grid_size = grid_size
        self.agents_in_range = agents_in_range
        self.full_tiles = set()  # Track which tiles are "full"

    def get_biome(self, x, y):
        return self.biome

    def count_agents_in_range(self, position, visual_range):
        return self.agents_in_range

    def is_tile_full(self, x, y):
        return (x, y) in self.full_tiles


@pytest.fixture
def genome():
    return DummyGenome()


@pytest.fixture
def environment_factory():
    def _factory(
        *, food_units=2, movement_speed=0.5, visibility=0.8, agents_in_range=3, grid_size=10
    ):
        biome = MockBiome(
            food_units=food_units,
            movement_speed=movement_speed,
            visibility=visibility,
        )
        return MockEnvironment(
            biome,
            grid_size=grid_size,
            agents_in_range=agents_in_range,
        )

    return _factory


def test_perceive_returns_normalized_tensor(genome, environment_factory):
    env = environment_factory()
    agent = Agent(genome, position=(1, 1))
    agent.energy = 60.0

    perception = agent.perceive(env)

    assert perception.shape == (1, 5)
    # normalized_energy = 60 / 100 = 0.6
    # normalized_food = 2 / 5 = 0.4 (default food_units=2)
    # normalized_agent_count = 3 / 10 = 0.3 (default agents_in_range=3)
    # agent_vision = nearby_agents / (visibility * visual_range) = 3 / (0.8 * 2.0) = 1.875, clipped to 1.0
    # movement_speed = terrain_speed * speed = 0.5 * 1.5 = 0.75
    expected = t.tensor([[0.6, 0.4, 0.3, 1.0, 0.75]], dtype=t.float32)
    t.testing.assert_close(perception, expected, atol=1e-5, rtol=1e-5)


def test_decide_uses_brain_output_when_energy_sufficient(genome, environment_factory):
    env = environment_factory()
    agent = Agent(genome, position=(0, 0))
    agent.brain = type(
        "MockBrain",
        (),
        {
            "eval": staticmethod(lambda: None),
            "__call__": staticmethod(lambda perception: t.tensor([[0.1, 0.2, 0.05, 0.8]])),
        },
    )()

    perception = agent.perceive(env)
    perception[0][0] = 0.8  # ensure energy is high enough to skip survival rules
    action = agent.decide(perception)

    assert action == ACTION_REPRODUCE


def test_move_updates_position_and_consumes_energy(genome, environment_factory):
    env = environment_factory(movement_speed=1.0, agents_in_range=0)
    agent = Agent(genome, position=(2, 2))
    agent.heading = 1  # East
    starting_energy = agent.energy

    agent.move(env)

    assert agent.position == (3, 2)
    # Movement cost = terrain_speed * speed * metabolism * 0.15
    # = 1.0 * 1.5 * 0.8 * 0.15 = 0.18
    expected_energy = starting_energy - (1.0 * 1.5 * 0.8 * 0.15)
    assert agent.energy == pytest.approx(expected_energy)


def test_eat_consumes_food_and_increases_energy(genome, environment_factory):
    env = environment_factory(food_units=2)
    agent = Agent(genome, position=(0, 0))
    agent.energy = 80.0

    success = agent.eat(env)

    assert success is True
    # Food provides 33.3 energy, max energy = 100 * 1.2 = 120
    # energy_needed = 120 - 80 = 40, so gains full 33.3
    assert agent.energy == pytest.approx(113.3)
    assert env.get_biome(0, 0).get_food_units() == 1


def test_reproduce_creates_offspring_and_reduces_energy(genome, environment_factory):
    env = environment_factory(food_units=3, agents_in_range=0)
    agent = Agent(genome, position=(1, 1))
    agent.energy = 80.0
    agent.age = 100  # Must be >= 100 to reproduce

    r.seed(0)
    offspring = agent.reproduce(env)

    assert isinstance(offspring, Agent)
    # reproduction_cost = energy * (0.50 * reproduction_cost_trait)
    # = 80.0 * (0.50 * 0.7) = 80.0 * 0.35 = 28.0
    assert agent.energy == pytest.approx(80.0 - 28.0)
    assert offspring.position in {(2, 1), (0, 1), (1, 2), (1, 0)}
    assert offspring.energy == pytest.approx(28.0)


def test_agent_dies_when_killed(genome):
    agent = Agent(genome, position=(1, 1))
    agent.energy = 100

    agent.kill_agent()

    assert not agent.is_alive()


def test_agent_natural_death_age_exactly(genome):
    agent = Agent(genome, position=(1, 1))
    agent.death_age = 500
    agent.age = 500

    assert agent.reached_natural_death()


def test_agent_natural_death_age_above(genome):
    agent = Agent(genome, position=(1, 1))
    agent.death_age = 500
    agent.age = 501

    assert agent.reached_natural_death()


def test_agent_natural_death_age_below(genome):
    agent = Agent(genome, position=(1, 1))
    agent.death_age = 500
    agent.age = 499

    assert not agent.reached_natural_death()


def test_agent_assigned_death(genome):
    agent = Agent(genome, position=(1, 1))
    death_age = agent._assign_death_age()
    agent.age = death_age
    agent.death_age = death_age

    assert agent.reached_natural_death()

def test_agent_skip_turn(genome):
    agent = Agent(genome, position=(1, 1))
    agent.loaf_around()

    assert agent.should_skip()

def test_agent_skip_turn_flag_reset(genome):
    agent = Agent(genome, position=(1, 1))

    assert not agent.should_skip()
