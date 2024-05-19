import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from nn_model import QNetwork
from collections import deque

#hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
EPSILON = 0.1
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.1
INPUT_SIZE = 8
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1

class AIAgent:
    def __init__(self, player_index, num_players):
        self.player_index = player_index
        self.input_size = INPUT_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.output_size = OUTPUT_SIZE
        self.lr = LR
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.memory = MAX_MEMORY
        self.batch_size = BATCH_SIZE
        self.q_net = QNetwork(self.input_size, self.output_size)
        self.target_net = QNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.num_players = num_players
        self.agent_state = None
        self.update_target_network = None
    
    #receive the game-state from the main.py Game
    def update_agent_state(self, game_state):
        self.agent_state = game_state

    ##FUNCION 1: that interferes with the Rikiki_game
    def make_bid(self):  # get_action
        # Select a bid based on epsilon-greedy policy
        # if np.random.rand() < self.epsilon:
        #     print('AI-player made a random choice')
        #     return np.random.choice(self.output_size)  # Explore
        # else:
        with torch.no_grad():
            q_values = self.q_net(self.agent_state)
            print(' Q-values argmax: ', torch.argmax(q_values).item())
            return torch.argmax(q_values).item()  # Exploit

    ##FUNCION 2: that interferes with the Rikiki_game
    def play_card(self, hand, leading_suit, atout):
    # Ensure there are valid cards to play
        if leading_suit:
            valid_cards = [card for card in hand if card.suit == leading_suit]
            if not valid_cards:
                valid_cards = hand  # If no valid cards for leading suit, play any card
        else:
            valid_cards = hand
        return random.choice(valid_cards)
    

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_net(next_state)).item()
            target_f = self.q_net(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            output = self.q_net(state)
            loss = self.criterion(output, target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_q_net(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            self.replay(self.batch_size)
        if done:
            self.update_target_network()

    def calculate_reward(self):
        # Reward for winning the game
        if self.scores[self.ai_player_index] == max(self.scores.values()):
            game_reward = 100  # Significant reward for winning the game
        else:
            game_reward = 0
        # Reward for winning rounds
        round_reward = 5  # Small reward for winning rounds
        # Penalty for inaccurate bids
        bid_penalty = abs(self.bids[self.ai_player_index] - self.pli_scores[self.ai_player_index])
        # Calculate total reward
        reward = game_reward + round_reward - bid_penalty
        return reward
    


