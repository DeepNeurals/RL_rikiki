import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
    A neural network model for approximating Q-values in reinforcement learning.

    This network takes in a state representation and outputs Q-values for each action.
    It is a feedforward neural network with two hidden layers.

    :param state_size: The size of the input state vector.
    :param action_size: The number of possible actions.
    """

    def __init__(self, state_size, action_size):
        """
        Initialize the QNetwork.

        :param state_size: The size of the input state vector.
        :param action_size: The number of possible actions the network can output Q-values for.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  # Input layer to first hidden layer  #with lower input state
        self.fc2 = nn.Linear(128, 128)  # Hidden layer to hidden layer
        #print(f" total_rounds: {total_rounds}")
        self.fc3 = nn.Linear(128, action_size)
    def forward(self, x):
        """
        Perform a forward pass through the network.

        :param x: Input tensor representing the state.
        :return: Output tensor with Q-values for each action.
        """
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = self.fc3(x)              # No activation on output
        return x

