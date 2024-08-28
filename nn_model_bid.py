import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Input layer to first hidden layer  #with lower input state
        self.fc2 = nn.Linear(128, 128)  # Hidden layer to hidden layer
        #print(f" total_rounds: {total_rounds}")
        self.fc3 = nn.Linear(128, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)              # No activation on output
        #print(f"the shape of the output of bidding is {x.shape}")
        return x

