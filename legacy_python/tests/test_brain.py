import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch as t

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.brain import Brain


@pytest.fixture
def sample_config():
    """Fixture providing a sample brain configuration."""
    return {
        "layers": [
            {"type": "linear", "input_size": 5, "output_size": 8},
            {"type": "activation", "function": "relu"},
            {"type": "linear", "input_size": 8, "output_size": 8},
            {"type": "activation", "function": "relu"},
            {"type": "linear", "input_size": 8, "output_size": 8},
            {"type": "activation", "function": "relu"},
            {"type": "linear", "input_size": 8, "output_size": 6},
        ]
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Fixture that creates a temporary config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_config, f)
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def brain(temp_config_file):
    """Fixture providing a Brain instance."""
    return Brain(temp_config_file)


class TestBrainInitialization:
    """Test Brain class initialization."""

    def test_brain_initializes_from_config_file(self, temp_config_file):
        """Test that Brain can be initialized from a config file."""
        brain = Brain(temp_config_file)
        assert brain is not None
        assert isinstance(brain.layers, t.nn.ModuleList)

    def test_brain_has_correct_number_of_layers(self, brain):
        """Test that Brain builds the correct number of layers."""
        assert len(brain.layers) == 7

    def test_brain_layers_are_correct_types(self, brain):
        """Test that layers are of correct types."""
        assert isinstance(brain.layers[0], t.nn.Linear)
        assert isinstance(brain.layers[1], t.nn.ReLU)
        assert isinstance(brain.layers[2], t.nn.Linear)
        assert isinstance(brain.layers[3], t.nn.ReLU)
        assert isinstance(brain.layers[4], t.nn.Linear)
        assert isinstance(brain.layers[5], t.nn.ReLU)
        assert isinstance(brain.layers[6], t.nn.Linear)

    def test_brain_linear_layers_have_correct_dimensions(self, brain):
        """Test that linear layers have correct input/output dimensions."""
        assert brain.layers[0].in_features == 5
        assert brain.layers[0].out_features == 8
        assert brain.layers[2].in_features == 8
        assert brain.layers[2].out_features == 8
        assert brain.layers[4].in_features == 8
        assert brain.layers[4].out_features == 8
        assert brain.layers[6].in_features == 8
        assert brain.layers[6].out_features == 6

    def test_brain_with_nonexistent_file_raises_error(self):
        """Test that initializing with a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Brain("/nonexistent/path/config.json")


class TestBrainActivationFunctions:
    """Test different activation function types."""

    @pytest.mark.parametrize(
        "activation,expected_type",
        [
            ("tanh", t.nn.Tanh),
            ("relu", t.nn.ReLU),
            ("sigmoid", t.nn.Sigmoid),
        ],
    )
    def test_activation_functions(self, activation, expected_type):
        """Test that different activation functions are correctly instantiated."""
        config = {
            "layers": [
                {"type": "linear", "input_size": 3, "output_size": 3},
                {"type": "activation", "function": activation},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            brain = Brain(temp_path)
            assert isinstance(brain.layers[1], expected_type)
        finally:
            Path(temp_path).unlink()


class TestBrainForwardPass:
    """Test Brain forward pass functionality."""

    def test_forward_pass_with_correct_input_shape(self, brain):
        """Test forward pass with correct input dimensions."""
        input_tensor = t.randn(1, 5, device=brain.device)  # batch_size=1, features=5
        output = brain(input_tensor)
        assert output.shape == (1, 6)  # batch_size=1, output_features=6

    def test_forward_pass_with_batch(self, brain):
        """Test forward pass with multiple samples in batch."""
        batch_size = 10
        input_tensor = t.randn(batch_size, 5, device=brain.device)
        output = brain(input_tensor)
        assert output.shape == (batch_size, 6)

    def test_forward_pass_output_is_tensor(self, brain):
        """Test that forward pass returns a tensor."""
        input_tensor = t.randn(1, 5, device=brain.device)
        output = brain(input_tensor)
        assert isinstance(output, t.Tensor)

    def test_forward_pass_with_wrong_input_shape_raises_error(self, brain):
        """Test that wrong input shape raises an error."""
        input_tensor = t.randn(1, 3, device=brain.device)  # Wrong: should be 5 features
        with pytest.raises(RuntimeError):
            brain(input_tensor)


class TestBrainWeightCounting:
    """Test Brain weight counting functionality."""

    def test_count_weights_returns_int(self, brain):
        """Test that count_weights returns an integer."""
        count = brain.count_weights()
        assert isinstance(count, int)

    def test_count_weights_is_positive(self, brain):
        """Test that weight count is positive."""
        count = brain.count_weights()
        assert count > 0

    def test_count_weights_correct_value(self, brain):
        """Test that weight count matches expected value."""
        # Layer 0: 5*8 + 8 = 48
        # Layer 2: 8*8 + 8 = 72
        # Layer 4: 8*8 + 8 = 72
        # Layer 6: 8*6 + 6 = 54
        # Total: 246
        expected_count = 246
        assert brain.count_weights() == expected_count

    def test_count_weights_with_simple_network(self):
        """Test weight counting with a simple network."""
        config = {
            "layers": [
                {"type": "linear", "input_size": 3, "output_size": 2},
            ]
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            temp_path = f.name

        try:
            brain = Brain(temp_path)
            # 3*2 weights + 2 biases = 8
            assert brain.count_weights() == 8
        finally:
            Path(temp_path).unlink()


class TestBrainLoadFromGenome:
    """Test Brain genome loading functionality."""

    def test_load_from_genome_with_valid_genome(self, brain):
        """Test loading weights from a valid genome."""
        weight_count = brain.count_weights()
        genome = {
            "id": "test_genome",
            "brain_weights": [0.1] * weight_count,
            "traits": {},
        }

        brain.load_from_genome(genome)

        # Check that first layer weights are set to 0.1
        assert t.allclose(brain.layers[0].weight.data, t.tensor(0.1))

    def test_load_from_genome_changes_weights(self, brain):
        """Test that loading from genome actually changes the weights."""
        # Get original weights
        original_weights = brain.layers[0].weight.data.clone()

        # Load new weights
        weight_count = brain.count_weights()
        genome = {
            "brain_weights": [0.5] * weight_count,
        }
        brain.load_from_genome(genome)

        # Check weights have changed
        assert not t.allclose(brain.layers[0].weight.data, original_weights)

    def test_load_from_genome_with_different_values(self, brain):
        """Test loading weights with varying values."""
        weight_count = brain.count_weights()
        genome = {
            "brain_weights": [i * 0.01 for i in range(weight_count)],
        }

        brain.load_from_genome(genome)

        # Verify some weights are loaded correctly
        assert brain.layers[0].weight.data[0, 0].item() == pytest.approx(0.0)

    def test_load_from_genome_without_brain_weights_key_raises_error(self, brain):
        """Test that missing brain_weights key raises KeyError."""
        genome = {"id": "test", "traits": {}}
        with pytest.raises(KeyError):
            brain.load_from_genome(genome)

    def test_load_from_genome_with_wrong_length_raises_error(self, brain):
        """Test that wrong number of weights raises RuntimeError or IndexError."""
        genome = {
            "brain_weights": [0.1, 0.2, 0.3],  # Too few weights
        }
        with pytest.raises((IndexError, RuntimeError)):
            brain.load_from_genome(genome)

    def test_load_from_genome_preserves_network_structure(self, brain):
        """Test that loading weights doesn't change network structure."""
        original_layer_count = len(brain.layers)
        weight_count = brain.count_weights()

        genome = {
            "brain_weights": [0.1] * weight_count,
        }
        brain.load_from_genome(genome)

        assert len(brain.layers) == original_layer_count
        assert isinstance(brain.layers[0], t.nn.Linear)


class TestBrainIntegration:
    """Integration tests for Brain class."""

    def test_full_pipeline_generate_load_forward(self, brain):
        """Test complete pipeline: generate genome weights, load, and forward pass."""
        import random

        weight_count = brain.count_weights()
        genome = {
            "brain_weights": [random.uniform(-0.5, 0.5) for _ in range(weight_count)],
        }

        brain.load_from_genome(genome)

        input_tensor = t.randn(5, 5, device=brain.device)  # batch of 5
        output = brain(input_tensor)

        assert output.shape == (5, 6)
        assert not t.isnan(output).any()

    def test_multiple_forward_passes_are_consistent(self, brain):
        """Test that multiple forward passes with same input produce same output."""
        brain.eval()  # Set to evaluation mode
        input_tensor = t.randn(1, 5, device=brain.device)

        output1 = brain(input_tensor)
        output2 = brain(input_tensor)

        assert t.allclose(output1, output2)

    def test_brain_can_be_used_for_training(self, brain):
        """Test that brain can be used in a training loop."""
        optimizer = t.optim.SGD(brain.parameters(), lr=0.01)
        input_tensor = t.randn(10, 5, device=brain.device)
        target = t.randn(10, 6, device=brain.device)

        # Training step
        optimizer.zero_grad()
        output = brain(input_tensor)
        loss = t.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

        # Should complete without errors
        assert loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
