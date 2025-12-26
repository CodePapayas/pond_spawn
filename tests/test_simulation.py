import sys
from pathlib import Path

import pytest
import torch as t

# Ensure the repository root is on the Python path for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from controllers.agent import Agent  # noqa: E402
from controllers.brain import Brain  # noqa: E402
from simulation import (  # noqa: E402
    Environment,
)

# -----------------------------------------------------------------------------
# Dummy / Mock Classes
# -----------------------------------------------------------------------------


class DummyGenome:
    """Lightweight genome object for Environment unit tests."""

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
        import copy

        offspring = DummyGenome()
        offspring.traits = copy.deepcopy(self.traits)
        offspring.brain_weights = list(self.brain_weights)
        return offspring


class MockBiome:
    """Mock biome for controlled testing."""

    def __init__(self, *, food_units=0, movement_speed=1.0, visibility=1.0, fertility=0.5):
        self.features = {
            "food_units": food_units,
            "movement_speed": movement_speed,
            "visibility": visibility,
            "fertility": fertility,
        }

    def get_food_units(self):
        return self.features["food_units"]

    def get_movement_speed(self):
        return self.features["movement_speed"]

    def get_visibility(self):
        return self.features["visibility"]

    def get_fertility(self):
        return self.features.get("fertility", 0.5)

    def generate(self):
        return self


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def small_environment():
    """Create a small 4x4 environment with minimal agents for quick tests."""
    return Environment(grid_size=4, num_agents=4, food_units=1)


@pytest.fixture
def empty_environment():
    """Create an environment with no agents."""
    return Environment(grid_size=4, num_agents=0, food_units=1)


@pytest.fixture
def environment_factory():
    """Factory fixture to create environments with custom parameters."""

    def _create(grid_size=6, num_agents=10, food_units=2):
        return Environment(grid_size=grid_size, num_agents=num_agents, food_units=food_units)

    return _create


@pytest.fixture
def dummy_genome():
    """Create a dummy genome for testing."""
    return DummyGenome()


@pytest.fixture
def mock_biome_factory():
    """Factory fixture to create mock biomes with custom parameters."""

    def _create(food_units=0, movement_speed=1.0, visibility=1.0, fertility=0.5):
        return MockBiome(
            food_units=food_units,
            movement_speed=movement_speed,
            visibility=visibility,
            fertility=fertility,
        )

    return _create


@pytest.fixture
def agent_factory(dummy_genome):
    """Factory fixture to create agents with controlled genomes."""

    def _create(position=(0, 0), genome=None):
        g = genome if genome is not None else DummyGenome()
        return Agent(g, position)

    return _create


@pytest.fixture
def populated_environment():
    """Create an environment and run a few steps to establish state."""
    env = Environment(grid_size=6, num_agents=20, food_units=2)
    for _ in range(5):
        env.step()
    return env


@pytest.fixture
def device():
    """Get the torch device being used."""
    return t.device("cuda" if t.cuda.is_available() else "cpu")
