import numpy as np

# Define the MDP
num_states = 3
num_actions = 2

# Define the immediate rewards matrix R(s, a)
# R[state, action]
R = np.array([
    [0, 0],
    [1, 2],
    [4, 0]
])

# Define the transition probability matrix P(s' | s, a)
# P[state, action, next_state]
P = np.array([
    [[0.5, 0.5, 0], [0, 1, 0], [0.8, 0.2, 0]],
    [[0, 1, 0], [0.1, 0.8, 0.1], [0, 0, 1]],
    [[1, 0, 0], [0, 1, 0], [0, 0.5, 0.5]]
])

# Define the discount factor
gamma = 0.9

# Initialize value function
V = np.zeros(num_states)

# Perform value iteration to solve the Bellman equation
num_iterations = 100
for _ in range(num_iterations):
    new_V = np.zeros(num_states)
    for s in range(num_states):
        for a in range(num_actions):
            new_V[s] = max(new_V[s], R[s, a] + gamma * np.sum(P[s, a] * V))
    V = new_V

print("Optimal Value Function:")
print(V)
