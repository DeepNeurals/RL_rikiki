import torch
import torch.nn as nn
import torch.nn.functional as F

class CardSelectionNN(nn.Module):
    def __init__(self):
        super(CardSelectionNN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(19, 64)   # First layer takes in 19 features (cards + tricks info)
        self.fc2 = nn.Linear(64, 32)   # Second layer reduces dimensionality
        self.fc3 = nn.Linear(32, 17)    # Output layer corresponds to 8 available card slots

    def forward(self, x):
        # Process input through dense layers
        x = F.relu(self.fc1(x))
        print(f"Size after first layer: {x.shape}")
        x = F.relu(self.fc2(x))
        print(f"Size after second layer: {x.shape}")
        # Output logits for the 8 cards in the hand (first 8 rows)
        logits = self.fc3(x[:8, :])  # Consider the 8 card rows
        print(f"Size of logits layer: {logits.shape}")
        print(f"logits: {logits}")
        return logits

    def masked_softmax(self, logits, mask):
        # Apply mask by setting the logits of unavailable (zero-vector) cards to -inf
        masked_logits = logits + mask
        # Apply softmax to get probabilities
        return F.softmax(masked_logits, dim=0)  # Apply softmax over the card slots
