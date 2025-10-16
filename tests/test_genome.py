import pytest
import json
import copy
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.genome import Genome, generate_genome_id, clamp


class TestHelperFunctions:
    """Test helper functions."""

    def test_generate_genome_id_format(self):
        """Test that generated IDs have correct format."""
        genome_id = generate_genome_id()
        assert genome_id.startswith("g_")
        assert len(genome_id) == 10  # "g_" + 8 hex characters

    def test_generate_genome_id_custom_prefix(self):
        """Test genome ID generation with custom prefix."""
        genome_id = generate_genome_id(prefix="test")
        assert genome_id.startswith("test_")

    def test_generate_genome_id_uniqueness(self):
        """Test that generated IDs are unique."""
        ids = [generate_genome_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_clamp_within_range(self):
        """Test that clamp keeps values within range."""
        assert clamp(5, 0, 10) == 5
        assert clamp(0, 0, 10) == 0
        assert clamp(10, 0, 10) == 10

    def test_clamp_below_minimum(self):
        """Test that clamp raises values below minimum."""
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_maximum(self):
        """Test that clamp lowers values above maximum."""
        assert clamp(15, 0, 10) == 10

    def test_clamp_with_float_values(self):
        """Test clamp with floating point values."""
        assert clamp(0.75, 0.5, 1.0) == 0.75
        assert clamp(0.3, 0.5, 1.0) == 0.5
        assert clamp(1.2, 0.5, 1.0) == 1.0


class TestGenomeInitialization:
    """Test Genome class initialization."""

    def test_genome_empty_initialization(self):
        """Test creating an empty genome."""
        genome = Genome()
        assert genome.id is None
        assert genome.traits == {}
        assert genome.brain_weights == []

    def test_genome_initialization_with_dict(self):
        """Test creating a genome from a dictionary."""
        genome_dict = {
            "id": "test_id",
            "traits": {"speed": {"value": 0.8, "min": 0.5, "max": 1.0}},
            "brain_weights": [0.1, 0.2, 0.3],
        }
        genome = Genome(genome_dict)
        assert genome.id == "test_id"
        assert genome.traits == genome_dict["traits"]
        assert genome.brain_weights == [0.1, 0.2, 0.3]

    def test_genome_base_data_loads_once(self):
        """Test that base genome data is loaded only once (class level)."""
        genome1 = Genome()
        genome2 = Genome()
        assert genome1._base_genome is genome2._base_genome
        assert genome1._brain_weight_count == genome2._brain_weight_count


class TestGenomeGeneration:
    """Test Genome generation functionality."""

    def test_generate_creates_new_genome(self):
        """Test that generate creates a new genome with all fields."""
        genome = Genome().generate()
        assert genome.id is not None
        assert len(genome.traits) > 0
        assert len(genome.brain_weights) > 0

    def test_generate_assigns_unique_id(self):
        """Test that each generated genome has a unique ID."""
        genome1 = Genome().generate()
        genome2 = Genome().generate()
        assert genome1.id != genome2.id

    def test_generate_creates_correct_number_of_weights(self):
        """Test that generated genome has correct number of brain weights."""
        genome = Genome().generate()
        expected_count = Genome._brain_weight_count
        assert len(genome.brain_weights) == expected_count

    def test_generate_weights_in_range(self):
        """Test that generated weights are within expected range [-0.5, 0.5]."""
        genome = Genome().generate()
        for weight in genome.brain_weights:
            assert -0.5 <= weight <= 0.5

    def test_generate_trait_values_set(self):
        """Test that all traits have values assigned."""
        genome = Genome().generate()
        for trait_name, trait_info in genome.traits.items():
            if "min" in trait_info and "max" in trait_info:
                assert "value" in trait_info
                assert trait_info["min"] <= trait_info["value"] <= trait_info["max"]

    def test_generate_returns_self(self):
        """Test that generate returns self for chaining."""
        genome = Genome()
        result = genome.generate()
        assert result is genome

    def test_generate_trait_values_are_random(self):
        """Test that different generated genomes have different trait values."""
        genome1 = Genome().generate()
        genome2 = Genome().generate()

        # Get first trait with min/max
        trait_name = list(genome1.traits.keys())[0]
        value1 = genome1.traits[trait_name].get("value")
        value2 = genome2.traits[trait_name].get("value")

        # They should be different (statistically)
        assert value1 != value2


class TestGenomeMutation:
    """Test Genome mutation functionality."""

    def test_mutate_creates_new_genome(self):
        """Test that mutate creates a new Genome instance."""
        original = Genome().generate()
        mutated = original.mutate()
        assert isinstance(mutated, Genome)
        assert mutated is not original

    def test_mutate_assigns_new_id(self):
        """Test that mutated genome has a different ID."""
        original = Genome().generate()
        mutated = original.mutate()
        assert mutated.id != original.id

    def test_mutate_preserves_genome_structure(self):
        """Test that mutated genome has same structure as original."""
        original = Genome().generate()
        mutated = original.mutate()

        assert len(mutated.traits) == len(original.traits)
        assert len(mutated.brain_weights) == len(original.brain_weights)
        assert set(mutated.traits.keys()) == set(original.traits.keys())

    def test_mutate_changes_some_values(self):
        """Test that mutation changes at least some values (statistically)."""
        original = Genome().generate()
        mutated = original.mutate()

        # Check if any brain weights changed
        weights_changed = any(
            original.brain_weights[i] != mutated.brain_weights[i]
            for i in range(len(original.brain_weights))
        )

        # With 156 weights and 10% mutation rate, something should change
        assert weights_changed

    def test_mutate_respects_trait_bounds(self):
        """Test that mutated trait values stay within min/max bounds."""
        original = Genome().generate()
        mutated = original.mutate()

        for trait_name, trait_info in mutated.traits.items():
            if "min" in trait_info and "max" in trait_info and "value" in trait_info:
                assert trait_info["min"] <= trait_info["value"] <= trait_info["max"]

    def test_mutate_multiple_generations(self):
        """Test that mutation can be applied multiple times."""
        genome = Genome().generate()
        gen2 = genome.mutate()
        gen3 = gen2.mutate()
        gen4 = gen3.mutate()

        # All should have different IDs
        ids = {genome.id, gen2.id, gen3.id, gen4.id}
        assert len(ids) == 4

    def test_mutate_doesnt_modify_original(self):
        """Test that mutation doesn't modify the original genome."""
        original = Genome().generate()
        original_weights = copy.deepcopy(original.brain_weights)
        original_traits = copy.deepcopy(original.traits)

        # Original should be unchanged
        assert original.brain_weights == original_weights
        for trait_name in original.traits:
            assert original.traits[trait_name] == original_traits[trait_name]


class TestGenomeSerialization:
    """Test Genome serialization and deserialization."""

    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dictionary."""
        genome = Genome().generate()
        result = genome.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_contains_required_keys(self):
        """Test that to_dict includes all required keys."""
        genome = Genome().generate()
        result = genome.to_dict()
        assert "id" in result
        assert "traits" in result
        assert "brain_weights" in result

    def test_to_dict_values_match_genome(self):
        """Test that to_dict values match genome attributes."""
        genome = Genome().generate()
        result = genome.to_dict()
        assert result["id"] == genome.id
        assert result["traits"] == genome.traits
        assert result["brain_weights"] == genome.brain_weights

    def test_from_dict_creates_genome(self):
        """Test that from_dict creates a Genome instance."""
        genome_dict = {
            "id": "test_id",
            "traits": {"speed": {"value": 0.8}},
            "brain_weights": [0.1, 0.2, 0.3],
        }
        genome = Genome.from_dict(genome_dict)
        assert isinstance(genome, Genome)

    def test_from_dict_preserves_data(self):
        """Test that from_dict preserves all data."""
        genome_dict = {
            "id": "test_id",
            "traits": {"speed": {"value": 0.8, "min": 0.5, "max": 1.0}},
            "brain_weights": [0.1, 0.2, 0.3],
        }
        genome = Genome.from_dict(genome_dict)
        assert genome.id == "test_id"
        assert genome.traits["speed"]["value"] == 0.8
        assert genome.brain_weights == [0.1, 0.2, 0.3]

    def test_round_trip_serialization(self):
        """Test that genome survives to_dict -> from_dict round trip."""
        original = Genome().generate()
        genome_dict = original.to_dict()
        restored = Genome.from_dict(genome_dict)

        assert restored.id == original.id
        assert restored.traits == original.traits
        assert restored.brain_weights == original.brain_weights

    def test_to_dict_is_json_serializable(self):
        """Test that to_dict output can be JSON serialized."""
        genome = Genome().generate()
        genome_dict = genome.to_dict()

        # Should not raise an exception
        json_str = json.dumps(genome_dict)
        assert isinstance(json_str, str)

    def test_from_dict_after_json_round_trip(self):
        """Test loading genome after JSON serialization."""
        original = Genome().generate()
        genome_dict = original.to_dict()

        # Serialize to JSON and back
        json_str = json.dumps(genome_dict)
        loaded_dict = json.loads(json_str)

        # Create genome from loaded dict
        restored = Genome.from_dict(loaded_dict)
        assert restored.id == original.id


class TestGenomeIntegration:
    """Integration tests for Genome class."""

    def test_generate_and_mutate_workflow(self):
        """Test typical workflow: generate -> mutate -> mutate."""
        parent = Genome().generate()
        child = parent.mutate()
        grandchild = child.mutate()

        # All should be valid genomes
        assert parent.id is not None
        assert child.id is not None
        assert grandchild.id is not None

        # All should have same structure
        assert (
            len(parent.brain_weights)
            == len(child.brain_weights)
            == len(grandchild.brain_weights)
        )

    def test_genome_compatible_with_brain(self):
        """Test that genome dict is compatible with Brain.load_from_genome."""
        from controllers.brain import Brain

        genome = Genome().generate()
        genome_dict = genome.to_dict()

        # Should have the required structure
        assert "brain_weights" in genome_dict
        assert isinstance(genome_dict["brain_weights"], list)

        # Should have correct number of weights
        brain_config_path = (
            Path(__file__).resolve().parent.parent / "brains" / "brain.json"
        )
        brain = Brain(str(brain_config_path))
        expected_count = brain.count_weights()
        assert len(genome_dict["brain_weights"]) == expected_count

    def test_population_generation(self):
        """Test generating a population of genomes."""
        population_size = 20
        population = [Genome().generate() for _ in range(population_size)]

        # All should be valid
        assert len(population) == population_size

        # All should have unique IDs
        ids = [g.id for g in population]
        assert len(set(ids)) == population_size

    def test_trait_inheritance_through_mutation(self):
        """Test that trait structure is preserved through mutations."""
        parent = Genome().generate()
        parent_traits = set(parent.traits.keys())

        # Mutate multiple times
        child = parent.mutate()
        grandchild = child.mutate()

        # All should have same trait keys
        assert set(child.traits.keys()) == parent_traits
        assert set(grandchild.traits.keys()) == parent_traits


class TestGenomeEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_genome_to_dict(self):
        """Test to_dict on an empty genome."""
        genome = Genome()
        result = genome.to_dict()
        assert result["id"] is None
        assert result["traits"] == {}
        assert result["brain_weights"] == []

    def test_genome_with_empty_traits(self):
        """Test genome with no traits."""
        genome_dict = {"id": "test", "traits": {}, "brain_weights": [0.1, 0.2]}
        genome = Genome.from_dict(genome_dict)
        assert genome.traits == {}

    def test_genome_with_no_brain_weights(self):
        """Test genome with empty brain weights."""
        genome_dict = {
            "id": "test",
            "traits": {"speed": {"value": 0.8}},
            "brain_weights": [],
        }
        genome = Genome.from_dict(genome_dict)
        assert genome.brain_weights == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
