import numpy as np
import random
from collections import deque
import tensorflow as tf
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most of the TensorFlow logs

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Hyperparameters
STATE_SIZE = 4  # Example: CartPole has a state space of size 4
ACTION_SIZE = 2  # Example: CartPole has 2 possible actions (left, right)
DISCOUNT_FACTOR = 0.95  # Gamma (discount factor for future rewards)
LEARNING_RATE = 0.001  # Learning rate for the optimizer
BATCH_SIZE = 64  # Batch size for experience replay
MEMORY_SIZE = 2000  # Size of the replay memory
EPISODES = 1000  # Number of episodes to train
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.995  # Epsilon decay per episode
EPSILON_MIN = 0.01  # Minimum exploration rate
TARGET_UPDATE_FREQUENCY = 10  # Frequency of updating the target network

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON  # Exploration-exploitation balance
        self.model = self.build_model()  # Main network
        self.target_model = self.build_model()  # Target network
        self.update_target_model()  # Copy initial weights to target model

    def build_model(self):
        """Builds the neural network model."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))  # Output layer for Q-values
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        """Copy the weights from the main network to the target network."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore: choose a random action
        q_values = self.model.predict(state)  # Exploit: choose action with max Q-value
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        """Train the network with experience replay."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + DISCOUNT_FACTOR * np.amax(t[0])
            self.model.fit(state, target, epochs=1, verbose=0)

        # Decay the exploration rate
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def load(self, name):
        """Load the model weights."""
        self.model.load_weights(name)

    def save(self, name):
        """Save the model weights."""
        self.model.save_weights(name)


# Main training loop
if __name__ == "__main__":
    env = gym.make('CartPole-v1')  # Replace with any environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):  # Maximum steps per episode
            action = agent.act(state)  # Select an action
            next_state, reward, done, _ = env.step(action)  # Take the action
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)  # Store experience
            state = next_state

            if done:
                print(f"Episode: {e}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break

        # Train the agent using experience replay
        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

        # Update the target network every few episodes
        if e % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_model()
