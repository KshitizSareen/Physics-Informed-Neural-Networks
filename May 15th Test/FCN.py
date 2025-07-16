import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Fully connected neural network (FCN) used as a PINN
class FCN(nn.Module):
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        # First layer (input to hidden)
        layers = [nn.Linear(N_INPUT, N_HIDDEN), nn.Tanh()]
        # Hidden layers (N_LAYERS - 1 blocks of Linear + Tanh)
        layers += [layer for _ in range(N_LAYERS - 1) for layer in (nn.Linear(N_HIDDEN, N_HIDDEN), nn.Tanh())]
        # Final layer (hidden to output)
        layers += [nn.Linear(N_HIDDEN, N_OUTPUT)]
        # Combine into one sequential model
        self.net = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.net(x)