import torch
from nn_model_bid import QNetwork  # Import your model class

# Initialize the model
model = QNetwork()

filename =  

# Load the saved model weights
model.load_state_dict(torch.load(filename))

# Set the model to evaluation mode
model.eval()

# Prepare your test state
state = [3, 0, 0, 0, 3, 5, 5, 5]  # Replace with your actual state data
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension if needed

# Run inference
with torch.no_grad():  # Disable gradient calculation
    predictions = model(state_tensor)
    bid = torch.argmax(predictions).item()

# Print or process predictions
print("Model predictions:", predictions)
print("Model BID pred:", bid)
