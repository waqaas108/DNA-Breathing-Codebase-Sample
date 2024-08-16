import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self, kernel, pool, filters, layers, input_channels=4):
        super(CNN, self).__init__()
        self.conv_kernel = kernel
        self.pool_kernel = pool
        self.input_channels = input_channels
        self.filters = filters
        self.layers = layers
        self.n_channels = 36  # Initialize with the input length
        self.current_filters = self.filters
        self.padding = (self.conv_kernel - 1) // 2  # Calculate padding for each convolutional layer

        for _ in range(self.layers):
            self.n_channels = (self.n_channels - self.conv_kernel + 2 * self.padding + 1) // self.pool_kernel

        modules = []
        in_channels = input_channels
        for _ in range(self.layers):
            modules.extend([
                nn.Conv1d(in_channels, filters, kernel_size=self.conv_kernel, padding=self.padding),
                nn.MaxPool1d(kernel_size=self.pool_kernel, stride=self.pool_kernel),
                nn.ReLU()
            ])
            in_channels = filters
            filters *= 2

        self.conv_net = nn.Sequential(*modules)
        self.regressor = nn.Sequential(
            nn.Linear(in_channels * self.n_channels, 1)
        )

    def forward(self, x):
        #print("Output before conv:", x.size())
        out = self.conv_net(x)
        #print("Output after conv:", out.size())
        out = out.view(out.size(0), -1)
        #print("Output after reshape:", out.size())
        out = self.regressor(out)
        #print("Final output:", out.size())
        return out