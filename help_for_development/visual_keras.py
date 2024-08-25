import torch
from torchviz import make_dot
import torch.nn as nn

# Define your QNetwork as above
class QNetwork(nn.Module):
    def __init__(self, deck_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(8, 64)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # Hidden layer to hidden layer
        self.fc3 = nn.Linear(64, deck_size + 1)  # Hidden layer to output layer (4 actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)  # No activation on output
        return x

# Instantiate the model
deck_size = 3
model = QNetwork(deck_size)

# Create dummy input to pass through the model
dummy_input = torch.randn(1, 8)  # Adjust the size as per your input size

# Use torchviz to create a visual graph
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))

# Save and render the graph
dot.format = 'png'
dot.render('qnetwork_visualization')
