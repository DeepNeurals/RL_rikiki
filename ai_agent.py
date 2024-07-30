import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from nn_model import QNetwork
from collections import deque

#hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class AIAgent:
    def __init__(self, player_index, num_players, learning_rate=0.001, gamma=0.99):
        self.player_index = player_index
        self.num_players = num_players
        self.agent_state = None
        self.n_games = 0
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.MSELoss()

        #for checking condition bid
        self.sum_bids = 0
        self.position_bidders = 0
        self.deck_size = 2 #the deck size starts at 2

    
    #receive the game-state from the main.py Game
    def update_agent_state(self, game_state):
        self.agent_state = game_state
        self.deck_size = game_state[4].item()
        #print(' test check for deck size:', self.deck_size)

    ##FUNCION 1: that interferes with the Rikiki_game
    def make_bid(self):
        x = self.agent_state #last state information we have
        print(f"Given this last state: {x} the agents make the following bid")

        if self.position_bidders != 4: #you are not last, you can bid whatever you want
            prediction = self.model(x)  #forward pass
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
        loss = self.criterion(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
