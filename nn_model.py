import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = 7  # State representation size
output_size = 11  # Number of possible bids (0 to 10, assuming a max deck size of 10)
model = QNetwork(input_size, output_size)
target_model = QNetwork(input_size, output_size)
target_model.load_state_dict(model.state_dict())
target_model.eval()
