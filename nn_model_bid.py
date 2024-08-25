import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, total_rounds):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(16, 16)  # Hidden layer to hidden layer
        #print(f" total_rounds: {total_rounds}")
        self.fc3 = nn.Linear(16, total_rounds+1)   # Hidden layer to output layer (4 actions), lets make this output a 1x9 tensor so it can learn
                                                #for 8 cards in hand the best strategy

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)              # No activation on output
        #print(f"the shape of the output of bidding is {x.shape}")
        return x

