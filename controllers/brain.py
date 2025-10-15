import torch as t
import torch.nn as nn
import json


class Brain(nn.Module):
    def __init__(self, config_path):
        super(Brain, self).__init__()
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.layers = nn.ModuleList()
        self._build_network(config['layers'])
    
    def _build_network(self, layer_configs):
        for layer_config in layer_configs:
            layer_type = layer_config['type']
            
            if layer_type == 'linear':
                layer = nn.Linear(layer_config['input_size'], layer_config['output_size'])
                self.layers.append(layer)
            elif layer_type == 'activation':
                activation_func = layer_config['function']
                if activation_func == 'tanh':
                    layer = nn.Tanh()
                elif activation_func == 'relu':
                    layer = nn.ReLU()
                elif activation_func == 'sigmoid':
                    layer = nn.Sigmoid()
                self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def count_weights(self):
        total = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                total += layer.weight.numel()
                if layer.bias is not None:
                    total += layer.bias.numel()
        return total

if __name__ == "__main__":
    pass