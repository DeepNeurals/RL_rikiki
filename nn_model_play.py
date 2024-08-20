import torch
import torch.nn as nn
import torch.nn.functional as F

class CardSelectionNN(nn.Module):
    def __init__(self):
        super(CardSelectionNN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(19, 64)   # First layer takes in 19 features (cards + tricks info)
        self.fc2 = nn.Linear(64, 32)   # Second layer reduces dimensionality
        self.fc3 = nn.Linear(32, 8)    # Output layer corresponds to 8 available card slots

    def forward(self, x):
        # Process input through dense layers
        x = F.relu(self.fc1(x))
        #print(f"Size after first layer: {x.shape}")
        x = F.relu(self.fc2(x))
        #print(f"Size after second layer: {x.shape}")

        # Apply global pooling across all rows (max pooling here as an example)
        x, _ = torch.max(x, dim=0, keepdim=True)  # Global max pooling to get 1x32
        # Output logits for the 8 card actions
        logits = self.fc3(x)  # Now logits is 1x8
        #print(f"Size of logits layer: {logits.shape}")
        #print(f"logits: {logits}")
        return logits  # Logits now has shape 1x8 representing the 8 card choices


    def masked_softmax(self, logits, mask):
        # Reshape the mask to match the shape of logits
        mask = mask.view_as(logits)   
        print(f"view mask as logits: {mask}")

        #for robustness do not chnage the logits directly, but clone
        masked_logits = logits.clone()

        # Set logits to a large negative value where mask is 0
        masked_logits[mask == 0] = -1e9
        print(f"masked logits with zero: {masked_logits}")

        # Apply softmax to get probabilities
        softmax_output = F.softmax(masked_logits, dim=1)  # Apply softmax across the row
        
        return softmax_output