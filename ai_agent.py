import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from nn_model import QNetwork
from collections import deque

class AIAgent:
    def __init__(self, learning_rate, deck_size, gamma=0.99, epsilon=0.30):
        self.agent_state = None
        self.n_games = 0
        self.deck_size = deck_size
        self.model = QNetwork(self.deck_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon  # Exploration rate
        self.losses = []  # List to store loss values

        #for checking condition bid
        self.sum_bids = 0
        self.position_bidders = 0
    
    #receive the game-state from the main.py Game
    def update_agent_state(self, game_state):
        self.agent_state = game_state
        self.deck_size = game_state[4].item()
        #print(' test check for deck size:', self.deck_size)

    ##FUNCION 1: that interferes with the Rikiki_game
    def make_bid(self):
        x = self.agent_state #last state information we have
        print(f"Given this last state: {x} the agents make the following bid")

        # if self.position_bidders != 4: #you are not last, you can bid whatever you want
        #     prediction = self.model(x)  #forward pass
        #     bid = torch.argmax(prediction).item()
        if self.position_bidders != 4:  # You are not last, you can bid whatever you want
            # ε-greedy action selection
            if random.random() < self.epsilon:  # With probability ε, explore
                bid = random.randint(0, self.deck_size)  # Assuming there are 5 possible actions (0 to 4)
            else:  # With probability 1 - ε, exploit
                prediction = self.model(x)  # Forward pass
                bid = torch.argmax(prediction).item()
        else:
            while True:
                prediction = self.model(x)  #forward pass
                bid = torch.argmax(prediction).item()
                if bid + self.sum_bids != self.deck_size:
                    break
                else:
                    continue
        return bid
    
    #function to save the model
    @staticmethod
    def save_model(model, filename):
        torch.save(model.state_dict(), filename)
    

    def update(self, state, action, reward, next_state, done):
        """Update the Q-values based on experience"""
        # Compute the target
        with torch.no_grad():
            target = reward
            if not done:
                next_q_values = self.model(next_state)
                target += self.gamma * torch.max(next_q_values)
        
        # Compute the current Q-value
        q_values = self.model(state)
        print('q_values:', q_values)
        q_value = q_values[action]

        # Compute loss and update model
        #loss = self.criterion(q_value, target)
        # Compute the loss
        loss = self.criterion(q_value, torch.tensor([target], dtype=torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Print the loss value
        print(f"\033[91mLoss: {loss.item()}\033[0m")
        # Store the loss value
        self.losses.append(loss.item())