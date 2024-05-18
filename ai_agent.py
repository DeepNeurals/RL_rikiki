# import torch
# import torch.nn as nn
# import torch.optim as optim
# #from rikiki_game_AI import RikikiGame
# import numpy as np


# class AIAgent:
#     def __init__(self, player_index, state_size, action_size, learning_rate=0.001, gamma=0.99):
#         self.player_index = player_index
#         #self.state_size = state_size
#         #self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.gamma = gamma
#         #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Define neural network model
#         self.model = self._build_model().to(self.device)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

#     def _build_model(self):
#         model = nn.Sequential(
#             nn.Linear(self.state_size, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, self.action_size)
#         )
#         return model

#     def get_state(self, RikikiGame):
#     # Define state components
#         cards_in_hand = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for cards in hand
#         cards_played = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for cards played in current trick
#         current_bid = RikikiGame.current_bid  # Current bid amount
#         atout_card = np.zeros(RikikiGame.current_deck_size)  # One-hot encoding for atout card
#         remaining_cards = RikikiGame.remaining_cards  # Number of remaining cards in the deck
#         bids_made = np.array(RikikiGame.bids_made)  # Bids made by other players
    
#         # Convert current state into a tensor
#         state = np.concatenate([
#             cards_in_hand,
#             cards_played,
#             np.array([current_bid]),
#             atout_card,
#             np.array([remaining_cards]),
#             bids_made
#         ])
    
#         return state.astype(int)

#     def get_action(self, state):
#         with torch.no_grad():
#             state_tensor = self.get_state_tensor(state)
#             q_values = self.model(state_tensor)
#             action = torch.argmax(q_values).item()
#         return action

#     def train(self, state, action, reward, next_state, done):
#         state_tensor = self.get_state_tensor(state)
#         next_state_tensor = self.get_state_tensor(next_state)

#         q_values = self.model(state_tensor)
#         next_q_values = self.model(next_state_tensor)

#         q_value = q_values[0, action]
#         next_q_value = torch.max(next_q_values).item()

#         target = reward + (1 - done) * self.gamma * next_q_value

#         loss = nn.MSELoss()(q_value, target)

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

import random
class AIAgent:
    def __init__(self, player_index):
        self.player_index = player_index

    def evaluate_hand(self, hand):
        # Assign scores to cards based on their value
        card_scores = {'2': 1, '3': 1, '4': 2, '5': 2, '6': 3, '7': 3, '8': 4, '9': 4, '10': 5, 'J': 6, 'Q': 7, 'K': 8, 'A': 10}
        total_score = sum(card_scores.get(card.value, 0) for card in hand)
        return total_score

    def make_bid(self, hand, bids, current_deck_size, atout, num_players, other_bids=None):
        if sum(bid for bid in bids if bid is not None) is None:
            possible_bids = [bid for bid in range(current_deck_size + 1)]
        else:
            possible_bids = [bid for bid in range(current_deck_size + 1) if sum(bid for bid in bids if bid is not None) + bid != current_deck_size]

        # Evaluate the quality of the hand
        hand_score = self.evaluate_hand(hand)

        # Adjust the bid based on the hand score and other players' bids
        if other_bids is not None:
            total_other_bids = sum(bid for bid in other_bids if bid is not None)
            remaining_deck_size = current_deck_size - sum(1 for bid in bids if bid is not None)
            average_bid = total_other_bids / len(other_bids)
            bid = max(0, hand_score // 10 - average_bid)  # Adjust bid based on average of other bids
            bid = min(remaining_deck_size, bid)  # Ensure bid does not exceed remaining deck size
        else:
            if hand_score >= 20:
                bid = min(current_deck_size, hand_score // 10)  # Higher bid for stronger hands
            else:
                bid = 0  # Conservative bid for weaker hands
        return bid
    
    # def play_card(self, hand, leading_suit, atout):
    #     # Implement a basic strategy for playing a card
    #     # For now, the AI will play a random card that follows the rules
    #     if leading_suit:
    #         valid_cards = [card for card in hand if card.suit == leading_suit]
    #         if valid_cards:
    #             return random.choice(valid_cards)
        
    def play_card(self, hand, leading_suit, atout):
        # Implement a basic strategy for playing a card
        # For now, the AI will play a random card that follows the rules
        if leading_suit:
            valid_cards = [card for card in hand if card.suit == leading_suit]
            if valid_cards:
                return random.choice(valid_cards)
        
        return random.choice(hand)