import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from nn_model_bid import QNetwork
from nn_model_play import CardSelectionNN
from collections import deque
from collections import namedtuple
import pydealer

value_mapping = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
suit_mapping = ["Diamonds", "Spades", "Hearts", "Clubs"]


class CustomCard(pydealer.Card):
    def __init__(self, number, suit, custom_value=None):
        super().__init__(number, suit)
        self.custom_value = custom_value if custom_value is not None else self.default_value()

    def default_value(self):
        # Define default values for cards if no custom value is provided
        value_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
        }
        return value_map.get(self.value, 0)


class AIAgent:
    def __init__(self, learning_rate, deck_size, gamma=0.99, epsilon=0.30):
        self.agent_state = None
        self.playing_state = None
        self.n_games = 0
        self.deck_size = deck_size
        self.model = QNetwork(self.deck_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon  # Exploration rate
        self.losses = []  # List to store loss values
        self.card_model = CardSelectionNN()

        #for checking condition bid
        self.sum_bids = 0
        self.position_bidders = 0
    
    #receive the game-state from the main.py Game
    def update_agent_state(self, game_state):
        self.agent_state = game_state
        self.deck_size = game_state[4].item()
        #print(' test check for deck size:', self.deck_size)

    #called every time the AI needs to play a card
    def update_playing_state(self, playing_state):
        self.playing_state = playing_state
    
    #create Mask function
    def create_mask(self, input_tensor, LD_condition):
        # Select the first 17 columns corresponding to the card encoding (assuming card info is in the first 15 features)
        card_features = input_tensor[:8, :17]
        # Create a mask: rows with all 0s in the first 17 features should be masked
        if LD_condition == -1:
            zero_card_mask = (card_features.sum(dim=1) == 0).float() * -1e9
            print(f"card features: {card_features}")
            print(f"zero card mask{zero_card_mask}")
            # Expand the mask to match the shape of the logits (rows for cards)
            mask = zero_card_mask.unsqueeze(1)  # Convert to column vector

        # elif LD_condition >= 0 and LD_condition<4: #One of the suits is leading
        #     zero_card_mask = (card_features.sum(dim=1) == 0).float() * -1e9
        #     print(f"card features: {card_features}")
        #     print(f"zero card mask{zero_card_mask}")
        #     print("TESTTTTT")

        #     # Expand the mask to match the shape of the logits (rows for cards)
        #     mask = zero_card_mask.unsqueeze(1)  # Convert to column vector
        # else:
        #     raise ValueError("Invalid LD_condition")

        elif 0 <= LD_condition < 4:  # One of the suits is leading
            # Extract the suit features (last 4 features)
            num_suits = 4
            suit_features = card_features[:, -num_suits:]
            
            # Create mask for the cards that match the LD_condition suit
            suit_mask = suit_features[:, LD_condition] > 0
            print(f"card features: {card_features}")
            print(f"suit features: {suit_features}")
            print(f"suit mask: {suit_mask.float()}")
            
            # Expand the mask to match the shape of the logits (rows for cards)
            mask = suit_mask.float().unsqueeze(1)  # Convert to column vector
            print(f"The mask with LD-condition is: {mask}")
        else:
            raise ValueError("Invalid LD_condition")
        return mask
    
    #select row function
    def select_row(self, output_tensor, input_tensor):
        """
        Select the row with the highest probability from the output tensor
        and map it to the corresponding card in the input tensor.

        Args:
        - output_tensor (Tensor): Tensor containing output logits/probabilities (N x C).
        - input_tensor (Tensor): Tensor containing card information (N x F).

        Returns:
        - Card: The selected card with the highest probability.
        """
        # Compute the sum of each row in the output tensor
        row_sums = output_tensor.sum(dim=1)
        
        # Find the index of the row with the highest sum
        max_row_index = row_sums.argmax().item()
        
        # Retrieve the corresponding row from the input tensor
        selected_row = input_tensor[max_row_index]
        
        # For demonstration purposes, assuming the row is a card
        print(f"For testing this is the selected row: {selected_row}")
        return selected_row


    def tensor_to_card(self, tensor_row):
        """
        Convert a tensor row representing a card back into a Card object.

        Args:
        - tensor_row (Tensor): Tensor representing a card.

        Returns:
        - Card: The corresponding Card object.
        """
        # Assuming the first part of the tensor encodes the card value
        value_index = tensor_row[:len(value_mapping)].argmax().item()
        value = value_mapping[value_index]

        # Assuming the next part encodes the card suit
        suit_index = tensor_row[len(value_mapping):len(value_mapping) + len(suit_mapping)].argmax().item()
        suit = suit_mapping[suit_index]
        return CustomCard(value, suit)
    

    
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
    
    ##FUNCION 2: that interferes with the Rikiki_game
    def ai_choose_card(self, input_tensor, LD_condition):
        blue_text = "\033[34m"  # Blue text
        reset_text = "\033[0m"  # Reset to default color
        # Print in blue
        print(f"{blue_text}LD_condition: {LD_condition}{reset_text}")
        # Create the mask for nul vectors and if LD is active, masks non leading cards
        mask = self.create_mask(input_tensor, LD_condition)
        # Call the forward function to get logits
        logits = self.card_model.forward(input_tensor)
        # Apply the mask and softmax to get the probabilities
        output_tensor = self.card_model.masked_softmax(logits, mask)
        tensor_row = self.select_row(output_tensor, input_tensor)
        # Convert tensor row to Card
        selected_card = self.tensor_to_card(tensor_row)
        print("Selected Card:", selected_card)
        return selected_card

    
    #UPDATE MODEL --> 
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

    #function to save a model to a specific filename
    @staticmethod
    def save_model(model, filename):
        torch.save(model.state_dict(), filename)
    