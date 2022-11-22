import torch
from torch import nn
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))
        return x

class NormNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NormNN, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = torch.nn.functional.relu(self.layer_1(x))
        # bound norm to 1
        norm = torch.norm(x1, dim=-1, keepdim=True).repeat(1, x1.size(dim=-1)) + 1e-16
        x2 = torch.div(x1, norm)
        x3 = self.layer_2(x2)
        x4 = torch.nn.functional.sigmoid(x3)
        return x4