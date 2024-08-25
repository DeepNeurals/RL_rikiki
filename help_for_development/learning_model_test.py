import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate synthetic data for demonstration purposes
def generate_data(num_samples):
    # Each player has 2 cards, represented by integers from 1 to 13
    card_hands = np.random.randint(1, 14, (num_samples, 2))  # AI agent's cards
    other_bids = np.random.randint(0, 3, (num_samples, 3))   # Other players' bids
    bids = np.random.randint(0, 3, num_samples)  # AI agent's bids
    actual_wins = np.random.randint(0, 3, num_samples)  # Simulated winning rounds
    return card_hands, other_bids, bids, actual_wins

num_samples = 1000
card_hands, other_bids, bids, actual_wins = generate_data(num_samples)

# Inputs: AI's cards and other players' bids
inputs = np.concatenate([card_hands, other_bids], axis=1).astype('float32')

# Outputs: The AI agent's bids
outputs = bids.astype('int32')

# Model Definition
model = models.Sequential([
    layers.InputLayer(input_shape=(5,)),  # 2 cards + 3 bids
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 possible bids (0, 1, 2)
])

# Custom Loss Function
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    y_pred = tf.cast(y_pred, tf.float32)  # Ensure y_pred is float32

    # Convert softmax output to integer predictions
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, tf.float32)

    # Calculate absolute difference between true and predicted bids
    difference = tf.abs(y_true - y_pred_labels)
    return tf.reduce_mean(difference)

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)

# Train the model
model.fit(inputs, outputs, epochs=50, batch_size=32, verbose=1)

# Evaluation and Prediction
test_card_hands, test_other_bids, test_bids, test_actual_wins = generate_data(100)
test_inputs = np.concatenate([test_card_hands, test_other_bids], axis=1).astype('float32')

# Predict and evaluate
predicted_bids = model.predict(test_inputs)
predicted_bids = np.argmax(predicted_bids, axis=-1)

# Print some results
for i in range(5):
    print(f"Hand: {test_card_hands[i]}, Other Bids: {test_other_bids[i]}, Predicted Bid: {predicted_bids[i]}, Actual Wins: {test_actual_wins[i]}")
