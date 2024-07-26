import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from nn_model import Linear_QNet, QTrainer
from collections import deque

#hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

#states and actions
NUM_STATES = 8
NUM_ACTIONS = 1

class AIAgent:
    def __init__(self, player_index, num_players):
        self.player_index = player_index
        self.num_players = num_players
        self.agent_state = None
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(9, 256, 1)
        self.q_trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        #update this later
        self.deck_size = 2
    
    #receive the game-state from the main.py Game
    def update_agent_state(self, game_state, deck_size):
        self.agent_state = game_state
        self.deck_size = deck_size

    ##FUNCION 1: that interferes with the Rikiki_game
    def make_bid(self):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_bid = 0
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, self.deck_size)
            final_bid = move
        else:
            state0 = self.agent_state
            print(type(state0))
            prediction = self.model(state0)  #forward pass
            move = torch.argmax(prediction).item()
            final_bid =  move
        return final_bid
    
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
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.q_trainer.train_step(state, action, reward, next_state, done)
    
    #remember function used for DQL 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

