"""
Tests for landscape.py (Biome class and related functions).
"""

import pytest

from controllers.landscape import Biome, generate_biome_id


@pytest.fixture
def base_biome_dict():
    """Factory for creating a basic biome dictionary."""
    return {
        "id": "b_test123",
        "features": {
            "movement_speed": {"value": 1.0, "min": 0.5, "max": 1.5},
            "visibility": {"value": 1.0, "min": 0.5, "max": 2.0},
            "fertility": {"value": 1.0, "min": 0.0, "max": 2.0},
            "food_units": 3,
        },
    }


@pytest.fixture
def biome_factory():
    """Factory for creating Biome instances with custom parameters."""

    def _create_biome(
        biome_id="b_test123",
        movement_speed=1.0,
        visibility=1.0,
        fertility=1.0,
        food_units=3,
    ):
        biome_dict = {
            "id": biome_id,
            "features": {
                "movement_speed": {"value": movement_speed, "min": 0.5, "max": 1.5},
                "visibility": {"value": visibility, "min": 0.5, "max": 2.0},
                "fertility": {"value": fertility, "min": 0.0, "max": 2.0},
                "food_units": food_units,
            },
        }
        return Biome(biome_dict)

    return _create_biome


@pytest.fixture
def generated_biome():
    """Create a randomly generated biome."""
    return Biome().generate()


##################
# Biome ID Tests #
##################


def test_biome_id_generates():
    """Test that generate_biome_id returns a string."""
    id = generate_biome_id()
    assert isinstance(id, str)


def test_biome_id_prefix():
    """Test that generated biome IDs start with 'b_' prefix."""
    id = generate_biome_id()
    assert id.startswith("b_")


def test_biome_id_length():
    """Test that generated biome IDs are exactly 10 characters long."""
    id = generate_biome_id()
    assert len(id) == 10


def test_unique_biome_ids():
    """Test that consecutively generated biome IDs are unique."""
    id_1 = generate_biome_id()
    id_2 = generate_biome_id()

    assert id_1 != id_2


###############
# Biome Tests #
###############


def test_biome_generates(biome_factory):
    """Test that a Biome instance can be created with a valid ID."""
    biome = biome_factory()
    assert isinstance(biome.id, str)


def test_biome_to_dict(biome_factory):
    """Test that a Biome instance can be serialized to a dictionary."""
    biome = biome_factory()
    assert isinstance(biome.to_dict(), dict)


def test_biome_from_dict(biome_factory):
    """Test that a Biome instance can be created from a dictionary."""
    biome = biome_factory()
    b_dict = biome.to_dict()
    new_biome = biome.from_dict(b_dict)

    assert isinstance(new_biome.id, str)


def test_biome_get_movement_speed(biome_factory):
    """Test that get_movement_speed returns the correct value."""
    biome = biome_factory()
    assert biome.get_movement_speed() == 1.0


def test_biome_get_fertility(biome_factory):
    """Test that get_fertility returns the correct value."""
    biome = biome_factory()
    assert biome.get_fertility() == 1.0


def test_biome_get_visibility(biome_factory):
    """Test that get_visibility returns the correct value."""
    biome = biome_factory()
    assert biome.get_visibility() == 1.0


def test_biome_get_food_units(biome_factory):
    """Test that get_food_units returns the correct value."""
    biome = biome_factory()
    assert biome.get_food_units() == 3
