import json

import torch as t
import torch.nn as nn


class Brain(nn.Module):
    """
    A neural network module that constructs itself from a JSON configuration file.

    This class builds a PyTorch neural network by parsing a JSON file that specifies
    the architecture layer by layer. It supports linear (fully connected) layers and
    various activation functions. The network can load pre-trained weights from a
    genome structure and count the total number of parameters.
    """

    def __init__(self, config_path, device=None):
        """
        Initialize the Brain neural network from a JSON configuration file.

        This constructor reads a JSON file containing layer specifications and
        constructs the neural network architecture accordingly. The JSON file
        should contain a 'layers' key with a list of layer configurations.

        Args:
            config_path (str): Path to the JSON configuration file that defines
                the network architecture. The file should contain layer definitions
                including types (linear, activation) and their parameters.

        Example JSON structure:
            {
                "layers": [
                    {"type": "linear", "input_size": 5, "output_size": 8},
                    {"type": "activation", "function": "tanh"},
                    ...
                ]
            }
        """
        super().__init__()

        if device is None:
            device = t.device("cuda" if t.cuda.is_available() else "cpu")
        self.device = device

        with open(config_path) as f:
            config = json.load(f)
        self.layers = nn.ModuleList()
        self._build_network(config["layers"])

        # Move the entire network to the target device
        self.to(self.device)

    def _build_network(self, layer_configs):
        """
        Construct the neural network layers from configuration dictionaries.

        This private method iterates through a list of layer configurations and
        instantiates the corresponding PyTorch layers. It supports two main types
        of layers:

        1. Linear layers: Fully connected layers with specified input and output sizes
        2. Activation layers: Non-linear activation functions (tanh, relu, sigmoid)

        Each layer is appended to self.layers (a ModuleList) in the order specified,
        which defines the forward pass order through the network.

        Args:
            layer_configs (list): A list of dictionaries, where each dictionary
                describes a layer with keys:
                - 'type': Either 'linear' or 'activation'
                - For linear layers: 'input_size' and 'output_size' (int)
                - For activation layers: 'function' (str: 'tanh', 'relu', or 'sigmoid')

        Raises:
            KeyError: If required keys are missing from layer configurations
            ValueError: If an unsupported layer type or activation function is specified
        """
        for layer_config in layer_configs:
            layer_type = layer_config["type"]

            if layer_type == "linear":
                layer = nn.Linear(layer_config["input_size"], layer_config["output_size"])
                self.layers.append(layer)
            elif layer_type == "activation":
                activation_func = layer_config["function"]
                if activation_func == "tanh":
                    layer = nn.Tanh()
                elif activation_func == "relu":
                    layer = nn.ReLU()
                elif activation_func == "sigmoid":
                    layer = nn.Sigmoid()
                self.layers.append(layer)

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        This method defines how input data flows through the network. It sequentially
        passes the input tensor through each layer in self.layers (linear layers and
        activation functions) in the order they were defined during network construction.

        The forward pass implements the computation:
            output = layer_n(...layer_2(layer_1(x)))

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_features).
                The number of input features must match the input_size of the first
                linear layer in the network configuration.

        Returns:
            torch.Tensor: Output tensor after passing through all layers. The shape
                will be (batch_size, output_features), where output_features matches
                the output_size of the final linear layer in the network.

        Example:
            If the network has input_size=5 and output_size=4:
            >>> brain = Brain('config.json')
            >>> input_tensor = torch.randn(1, 5)  # batch_size=1, features=5
            >>> output = brain(input_tensor)
            >>> output.shape
            torch.Size([1, 4])
        """
        x = x.to(self.device)

        for layer in self.layers:
            x = layer(x)
        return x

    def count_weights(self):
        """
        Count the total number of trainable parameters in the neural network.

        This method iterates through all layers in the network and sums up the
        number of trainable parameters (weights and biases) in linear layers.
        Activation layers have no trainable parameters and are skipped.

        For each linear layer:
        - Counts all weight matrix elements (input_size × output_size)
        - Counts all bias vector elements (output_size), if bias is enabled

        This count is useful for:
        - Understanding model complexity
        - Allocating memory for genome weight storage
        - Debugging network architecture

        Returns:
            int: The total number of trainable parameters across all linear layers
                in the network. This includes both weights and biases.

        Example:
            For a network with layers:
            - Linear(5 -> 8): 5×8 weights + 8 biases = 48 parameters
            - Linear(8 -> 4): 8×4 weights + 4 biases = 36 parameters
            Total: 84 parameters
        """
        total = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                total += layer.weight.numel()
                if layer.bias is not None:
                    total += layer.bias.numel()
        return total

    def load_from_genome(self, genome):
        """
        Load neural network weights from a genome dictionary.

        This method takes a genome structure (typically from genome.py) and loads
        the flattened weight values stored in genome["brain_weights"] into the
        actual PyTorch layer parameters. The weights are stored as a flat list
        in the genome and must be reshaped to match each layer's dimensions.

        The method processes layers sequentially, extracting weights and biases
        in the same order they appear in the network. This ensures consistency
        between genome generation, mutation, and loading.

        Weight loading process for each linear layer:
        1. Calculate the number of weight parameters (input_size × output_size)
        2. Extract that many values from the flat brain_weights list
        3. Convert to a PyTorch tensor and reshape to match layer dimensions
        4. Assign to layer.weight.data (in-place modification)
        5. Repeat for biases if present
        6. Move index forward for the next layer

        Args:
            genome (dict): A genome dictionary containing at minimum:
                - "brain_weights" (list of float): A flat list of weight values
                  with length equal to count_weights(). Values should be ordered
                  as: [layer1_weights, layer1_biases, layer2_weights, layer2_biases, ...]

        Raises:
            KeyError: If genome doesn't contain "brain_weights" key
            IndexError: If brain_weights list is too short for the network architecture
            ValueError: If weight values cannot be reshaped to layer dimensions

        Example:
            >>> brain = Brain('config.json')
            >>> genome = genome_generator()  # Contains "brain_weights" key
            >>> brain.load_from_genome(genome)
            # Brain now uses weights from the genome

        Note:
            - This method modifies the network parameters in-place
            - Activation layers are skipped as they have no parameters
            - The genome must have exactly count_weights() values in brain_weights
        """
        brain_weights = genome["brain_weights"]
        idx = 0

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Load weights
                weight_size = layer.weight.numel()
                weight_values = brain_weights[idx : idx + weight_size]
                layer.weight.data = t.tensor(
                    weight_values, dtype=t.float32, device=self.device
                ).view_as(layer.weight)
                idx += weight_size

                # Load biases
                if layer.bias is not None:
                    bias_size = layer.bias.numel()
                    bias_values = brain_weights[idx : idx + bias_size]
                    layer.bias.data = t.tensor(
                        bias_values, dtype=t.float32, device=self.device
                    ).view_as(layer.bias)
                    idx += bias_size


if __name__ == "__main__":
    pass
