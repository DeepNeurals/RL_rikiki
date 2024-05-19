import tensorflow as tf

# Define the input shape
input_shape = (7,)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer with 1 neuron (predicts the number of bids)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error as the loss function

# Display the model summary
model.summary()
