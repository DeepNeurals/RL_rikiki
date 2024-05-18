import torch
import torch.nn as nn
import torch.optim as optim
#from rikiki_game_AI import RikikiGame
import numpy as np

class AIAgent:
    def __init__(self, player_index, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.player_index = player_index
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define neural network model
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_size)
        )
        return model

    def get_state(self, RikikiGame):
    # Define state components
        cards_in_hand = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for cards in hand
        cards_played = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for cards played in current trick
        current_bid = RikikiGame.current_bid  # Current bid amount
        atout_card = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for atout card
        remaining_cards = RikikiGame.remaining_cards  # Number of remaining cards in the deck
        bids_made = np.array(RikikiGame.bids_made)  # Bids made by other players
    
        # Convert current state into a tensor
        state = np.concatenate([
            cards_in_hand,
            cards_played,
            np.array([current_bid]),
            atout_card,
            np.array([remaining_cards]),
            bids_made
        ])
    
        return state.astype(int)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = self.get_state_tensor(state)
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def train(self, state, action, reward, next_state, done):
        state_tensor = self.get_state_tensor(state)
        next_state_tensor = self.get_state_tensor(next_state)

        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        q_value = q_values[0, action]
        next_q_value = torch.max(next_q_values).item()

        target = reward + (1 - done) * self.gamma * next_q_value

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
