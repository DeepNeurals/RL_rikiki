import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, deck_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(64, deck_size+1)   # Hidden layer to output layer (4 actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)              # No activation on output
        return x
