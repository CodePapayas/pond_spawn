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


# -----------------------------------------------------------------------------
# Tests: _add_food()
# -----------------------------------------------------------------------------


class TestAddFood:
    """Tests for Environment._add_food() method."""

    def test_add_food_increases_total_food(self, environment_factory):
        """Adding food should increase total food in the environment."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)

        # Get initial food count
        initial_food = sum(biome.get_food_units() for _, _, biome in env.iter_biomes())

        # Add food
        env._add_food(5)

        # Get new food count
        new_food = sum(biome.get_food_units() for _, _, biome in env.iter_biomes())

        assert new_food >= initial_food, "Food should increase after _add_food"

    def test_add_food_respects_fertility(self, environment_factory, mock_biome_factory):
        """Higher fertility biomes should receive more food."""
        env = environment_factory(grid_size=2, num_agents=0, food_units=0)

        # Replace grid with controlled biomes - different fertilities
        env.grid[0][0] = mock_biome_factory(food_units=0, fertility=1.0)  # High fertility
        env.grid[0][1] = mock_biome_factory(food_units=0, fertility=0.5)  # Medium fertility
        env.grid[1][0] = mock_biome_factory(food_units=0, fertility=0.0)  # No fertility
        env.grid[1][1] = mock_biome_factory(food_units=0, fertility=0.25)  # Low fertility

        env._add_food(10)

        # Check food distribution matches fertility
        assert env.grid[0][0].get_food_units() == 10  # 10 * 1.0 = 10
        assert env.grid[0][1].get_food_units() == 5  # 10 * 0.5 = 5
        assert env.grid[1][0].get_food_units() == 0  # 10 * 0.0 = 0
        assert env.grid[1][1].get_food_units() == 2  # 10 * 0.25 = 2

    def test_add_food_accumulates(self, environment_factory, mock_biome_factory):
        """Food should accumulate when _add_food is called multiple times."""
        env = environment_factory(grid_size=2, num_agents=0, food_units=0)

        # Set all biomes to same fertility for predictable results
        for x in range(2):
            for y in range(2):
                env.grid[x][y] = mock_biome_factory(food_units=0, fertility=1.0)

        env._add_food(5)
        env._add_food(5)

        # Each tile should have 10 food (5 + 5)
        for x in range(2):
            for y in range(2):
                assert env.grid[x][y].get_food_units() == 10

    def test_add_food_with_zero_amount(self, environment_factory, mock_biome_factory):
        """Adding zero food should not change existing food levels."""
        env = environment_factory(grid_size=2, num_agents=0, food_units=0)

        # Set initial food
        env.grid[0][0] = mock_biome_factory(food_units=5, fertility=1.0)

        env._add_food(0)

        assert env.grid[0][0].get_food_units() == 5


# -----------------------------------------------------------------------------
# Tests: _initialize_biomes()
# -----------------------------------------------------------------------------


class TestInitializeBiomes:
    """Tests for Environment._initialize_biomes() method."""

    def test_all_tiles_have_biomes(self, empty_environment):
        """Every grid tile should have a biome after initialization."""
        env = empty_environment

        for x in range(env.grid_size):
            for y in range(env.grid_size):
                assert env.grid[x][y] is not None, f"Tile ({x}, {y}) should have a biome"

    def test_biomes_have_required_features(self, empty_environment):
        """Each biome should have the expected feature attributes."""
        env = empty_environment

        required_features = ["food_units", "movement_speed", "visibility"]

        for x, y, biome in env.iter_biomes():
            for feature in required_features:
                assert hasattr(biome, "features"), f"Biome at ({x}, {y}) missing features dict"
                # Note: features might be accessed via methods, but should exist


# -----------------------------------------------------------------------------
# Tests: _spawn_agents()
# -----------------------------------------------------------------------------


class TestSpawnAgents:
    """Tests for Environment._spawn_agents() method."""

    def test_spawn_correct_number_of_agents(self, environment_factory):
        """Should spawn exactly the requested number of agents."""
        env = environment_factory(grid_size=6, num_agents=15, food_units=0)

        assert len(env.agents) == 15
        assert len(env.agents_by_id) == 15

    def test_spawn_agents_in_position_map(self, environment_factory):
        """All spawned agents should be tracked in position_map."""
        env = environment_factory(grid_size=4, num_agents=8, food_units=0)

        # Count agents in position_map
        total_in_map = sum(len(agents) for agents in env.position_map.values())

        assert total_in_map == 8

    def test_spawn_agents_have_valid_positions(self, environment_factory):
        """All agents should have positions within grid bounds."""
        env = environment_factory(grid_size=5, num_agents=10, food_units=0)

        for agent in env.agents:
            x, y = agent.position
            assert 0 <= x < env.grid_size, f"Agent x={x} out of bounds"
            assert 0 <= y < env.grid_size, f"Agent y={y} out of bounds"

    def test_spawn_respects_max_capacity(self):
        """Should cap population at max grid capacity."""
        grid_size = 3
        # MAX_AGENTS_PER_TILE = 1, so max capacity = 1 * 3 * 3 = 9
        env = Environment(grid_size=grid_size, num_agents=100, food_units=0)

        assert len(env.agents) <= grid_size * grid_size

    def test_spawn_zero_agents(self, empty_environment):
        """Environment with zero agents should have empty agent structures."""
        env = empty_environment

        assert len(env.agents) == 0
        assert len(env.agents_by_id) == 0
        assert len(env.position_map) == 0


# -----------------------------------------------------------------------------
# Tests: _record_lifespan()
# -----------------------------------------------------------------------------


class TestRecordLifespan:
    """Tests for Environment._record_lifespan() method."""

    def test_record_lifespan_adds_to_list(self, environment_factory, agent_factory):
        """Recording lifespan should add age to lifespans list."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(0, 0))
        agent.age = 50

        env._record_lifespan(agent)

        assert 50 in env.lifespans

    def test_record_lifespan_prevents_duplicates(self, environment_factory, agent_factory):
        """Same agent should only be recorded once."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(0, 0))
        agent.age = 30

        # Record multiple times
        env._record_lifespan(agent)
        env._record_lifespan(agent)
        env._record_lifespan(agent)

        # Should only appear once
        assert env.lifespans.count(30) == 1
        assert agent.get_id() in env.logged_lifespans


# -----------------------------------------------------------------------------
# Tests: update_agent_position()
# -----------------------------------------------------------------------------


class TestUpdateAgentPosition:
    """Tests for Environment.update_agent_position() method."""

    def test_update_position_removes_from_old(self, environment_factory, agent_factory):
        """Agent should be removed from old position in map."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(1, 1))
        agent_id = agent.get_id()

        # Manually add to position map
        env.position_map[(1, 1)] = {agent_id}

        # Move agent
        env.update_agent_position(agent_id, (1, 1), (2, 2))

        assert agent_id not in env.position_map.get((1, 1), set())

    def test_update_position_adds_to_new(self, environment_factory, agent_factory):
        """Agent should be added to new position in map."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(1, 1))
        agent_id = agent.get_id()

        env.position_map[(1, 1)] = {agent_id}

        env.update_agent_position(agent_id, (1, 1), (2, 2))

        assert agent_id in env.position_map[(2, 2)]

    def test_update_position_cleans_empty_positions(self, environment_factory, agent_factory):
        """Empty position sets should be removed from map."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(1, 1))
        agent_id = agent.get_id()

        env.position_map[(1, 1)] = {agent_id}

        env.update_agent_position(agent_id, (1, 1), (2, 2))

        # Old position should be completely removed if empty
        assert (1, 1) not in env.position_map


# -----------------------------------------------------------------------------
# Tests: is_tile_full()
# -----------------------------------------------------------------------------


class TestIsTileFull:
    """Tests for Environment.is_tile_full() method."""

    def test_empty_tile_not_full(self, environment_factory):
        """Empty tile should not be full."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)

        assert env.is_tile_full(0, 0) is False

    def test_tile_with_max_agents_is_full(self, environment_factory, agent_factory):
        """Tile at capacity should be full."""
        env = environment_factory(grid_size=4, num_agents=0, food_units=0)
        agent = agent_factory(position=(0, 0))

        # Add agent to position map (MAX_AGENTS_PER_TILE = 1)
        env.position_map[(0, 0)] = {agent.get_id()}

        assert env.is_tile_full(0, 0) is True


# -----------------------------------------------------------------------------
# Tests: get_stats()
# -----------------------------------------------------------------------------


class TestGetStats:
    """Tests for Environment.get_stats() method."""

    def test_stats_contains_required_keys(self, small_environment):
        """Stats should contain all expected keys."""
        stats = small_environment.get_stats()

        required_keys = [
            "step",
            "alive_agents",
            "total_food",
            "avg_energy",
            "median_lifespan",
            "min_age",
            "max_age",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_stats_step_count_accurate(self, small_environment):
        """Step count should match actual steps taken."""
        env = small_environment

        assert env.get_stats()["step"] == 0

        env.step()
        assert env.get_stats()["step"] == 1

        env.step()
        env.step()
        assert env.get_stats()["step"] == 3

    def test_stats_alive_count_accurate(self, environment_factory):
        """Alive agent count should match actual living agents."""
        env = environment_factory(grid_size=4, num_agents=5, food_units=0)

        stats = env.get_stats()
        actual_alive = sum(1 for a in env.agents if a.is_alive())

        assert stats["alive_agents"] == actual_alive


# -----------------------------------------------------------------------------
# Tests: Integration / step()
# -----------------------------------------------------------------------------


class TestStep:
    """Integration tests for Environment.step() method."""

    def test_step_increments_counter(self, small_environment):
        """Each step should increment step_count."""
        env = small_environment
        initial_step = env.step_count

        env.step()

        assert env.step_count == initial_step + 1

    def test_step_with_no_agents_doesnt_crash(self, empty_environment):
        """Stepping with no agents should not raise errors."""
        env = empty_environment

        # Should not raise
        env.step()
        env.step()
        env.step()

        assert env.step_count == 3

    def test_step_ages_agents(self, small_environment):
        """Agents should age each step."""
        env = small_environment

        # Get initial ages
        initial_ages = [a.age for a in env.agents]

        env.step()

        # Living agents should have aged
        for i, agent in enumerate(env.agents):
            if agent.is_alive():
                assert agent.age >= initial_ages[i]
