import torch
from nn_model_bid import QNetwork  # Import your model class
import random

total_rounds=8 
# Initialize the model
model = QNetwork(total_rounds)

# Load the saved model weights
model.load_state_dict(torch.load('bid_model_outputs/model20240825_170830_300.pth'))


# Set the model to evaluation mode
model.eval()

def generate_random_state():
    # Define the ranges or limits for each state component
    num_aces = random.randint(0, 4)         # Example range for num_aces
    num_kings = random.randint(0, 4)        # Example range for num_kings
    num_queens = random.randint(0, 5)       # Example range for num_queens
    num_atouts = random.randint(0, 5)       # Example range for num_atouts
    deck_size = random.randint(0, 8)       # Example range for deck_size
    p1_bid = 0        # Example range for p1_bid
    p2_bid = 0        # Example range for p2_bid
    p3_bid = 0        # Example range for p3_bid

    # Create the state list
    state = [num_aces, num_kings, num_queens, num_atouts, deck_size, p1_bid, p2_bid, p3_bid]
    
    # Convert the list to a tensor and add batch dimension
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    
    return state_tensor

# Generate a list of random state tensors
num_samples = 10  # Number of random state tensors you want
random_state_tensors = [generate_random_state() for _ in range(num_samples)]

# Print the generated tensors
for i, tensor in enumerate(random_state_tensors):
    print(f"State Tensor {i+1}:")
    print(tensor)
    # Run inference
    with torch.no_grad():  # Disable gradient calculation
        predictions = model(tensor)
        bid = torch.argmax(predictions).item()

        # Print or process predictions
        print("Model predictions:", predictions)
        print("Model BID pred:", bid)

# # Prepare your test state
# num_aces = 3
# num_kings = 0
# num_queens = 2
# num_atouts = 0
# deck_size = 6
# p1_bid = 5
# p2_bid = 5
# p3_bid = 5
# state = [num_aces, num_kings, num_queens, num_atouts, deck_size, p1_bid, p2_bid, p3_bid]  # Replace with your actual state data
# state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed


