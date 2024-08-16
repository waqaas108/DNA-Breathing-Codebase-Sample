import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class DNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DNN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out
