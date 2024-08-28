import torch
import torch.nn as nn
import torch.nn.functional as F

class CardSelectionNN(nn.Module):
    """
    A neural network model for selecting cards from a set of available cards.

    This network processes input features related to cards and tricks, and outputs logits for each possible card choice.
    
    :param total_cards: The number of available card choices (output size).
    """
    def __init__(self, total_cards):
        """
        Initialize the CardSelectionNN model.

        :param total_cards: The number of available card choices (output size).
        """
        super(CardSelectionNN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(19, 64)   # First layer takes in 19 features (cards + tricks info)
        self.fc2 = nn.Linear(64, 32)   # Second layer reduces dimensionality
        self.fc3 = nn.Linear(32, total_cards)    # Output layer corresponds to 8 available card slots

    def forward(self, x):
        """
        Perform a forward pass through the network.

        :param x: Input tensor with shape (batch_size, 19) representing card and trick information.
        :return: Output tensor with logits for each card choice, with shape (batch_size, total_cards).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Apply global pooling across all rows (max pooling here as an example)
        x, _ = torch.max(x, dim=0, keepdim=True)  # Global max pooling to get 1x32
        logits = self.fc3(x)  
        return logits  # Logits now has shape 1x8 representing the 8 card choices


    def masked_softmax(self, logits, mask):
        """
        Apply masked softmax to the logits to get probabilities while ignoring masked positions.

        :param logits: Tensor of logits with shape (batch_size, total_cards).
        :param mask: Binary mask tensor of the same shape as logits, where 1 indicates valid positions and 0 indicates masked positions.
        :return: Tensor of probabilities with the same shape as logits, after applying masked softmax.
        """
        mask = mask.view_as(logits)   
        #for robustness do not change the logits directly, but clone
        masked_logits = logits.clone()

        # Set logits to a large negative value where mask is 0
        masked_logits[mask == 0] = -1e9

        # Apply softmax to get probabilities
        softmax_output = F.softmax(masked_logits, dim=1)  # Apply softmax across the row
        return softmax_output