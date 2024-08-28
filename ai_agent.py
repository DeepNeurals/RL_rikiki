import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import pydealer
from nn_model_bid import QNetwork
from nn_model_play import CardSelectionNN
from collections import deque


value_mapping = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
suit_mapping = ["Diamonds", "Spades", "Hearts", "Clubs"]


class CustomCard(pydealer.Card):
    """
    A custom card class that extends pydealer's Card to include a custom value.

    :param number: The card's number (e.g., '2', '3', ... 'Ace').
    :param suit: The card's suit (e.g., 'Diamonds', 'Spades', 'Hearts', 'Clubs').
    :param custom_value: An optional custom value for the card. If not provided, the default value is used.
    """
    def __init__(self, number, suit, custom_value=None):
        super().__init__(number, suit)
        self.custom_value = custom_value if custom_value is not None else self.default_value()

    def default_value(self):
        """
        Define default values for cards if no custom value is provided.

        :return: Default integer value of the card based on its value.
        """
        value_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
        }
        return value_map.get(self.value, 0)

class AIAgent:
    """
    An AI agent for playing and bidding in a card game using reinforcement learning models.

    :param deck_size: The total number of cards in the deck.
    :param state_size: The size of the state representation.
    :param total_rounds: The number of rounds in the bidding phase.
    :param bid_model_weights: Path to the weights file for the bidding model.
    :param card_model_weights: Path to the weights file for the card selection model.
    :param lr_bid: Learning rate for the bidding model.
    :param lr_play: Learning rate for the card selection/playing  model.
    :param epsilon: Exploration rate for the bidding model.
    :param gamma: Discount factor for future rewards.
    """
    def __init__(self, deck_size, state_size, total_rounds, bid_model_weights, card_model_weights, lr_bid, lr_play, epsilon=0.05, gamma=0.99):
        self.agent_state = None
        self.playing_state = None
        self.n_games = 0
        self.deck_size = deck_size
        self.action_size = total_rounds + 1
        self.bid_model = QNetwork(state_size, self.action_size) #total rounds = action_size-1
        self.epsilon = epsilon
        self.gamma = gamma
        self.bid_optimizer = optim.Adam(self.bid_model.parameters(), lr=lr_bid )
        self.criterion = nn.MSELoss()
        self.losses_bid = []  # List to store loss values

        #For PlayingModel
        self.card_model = CardSelectionNN(total_rounds)
        self.card_optimizer = optim.Adam(self.card_model.parameters(), lr=lr_play)
        self.losses_card = []  # List to store loss values

        if bid_model_weights:
            self.load_weights(self.bid_model, bid_model_weights)
        if card_model_weights:
            self.load_weights(self.card_model, card_model_weights)

        #for checking condition bid
        self.sum_bids = 0
        self.position_bidders = 0

    def load_weights(self, model, weight_path):
        """
        Load weights for a model from a specified file path.

        :param model: The model to load weights into.
        :param weight_path: Path to the weights file.
        """
        model.load_state_dict(torch.load(weight_path))
        model.eval()  # Set the model to evaluation mode
    
    #create Mask function
    def create_mask(self, input_tensor, LD_condition, len_hand):
        """
        Create a mask to filter out invalid or irrelevant cards based on the condition.

        :param input_tensor: Tensor containing card information.
        :param LD_condition: Condition determining which cards to mask (e.g., leading suit).
        :param len_hand: The length of the hand (number of cards).
        :return: A mask tensor that highlights valid cards.
        """
        # Select the first 17 columns corresponding to the card encoding (assuming card info is in the first 15 features)
        card_features = input_tensor[:8, :17]
        # Create a mask: rows with all 0s in the first 17 features should be masked
        if LD_condition == -1:
            zero_card_mask = (card_features.sum(dim=1) != 0).float() 
            # Expand the mask to match the shape of the logits (rows for cards)
            mask = zero_card_mask.unsqueeze(1)  # Convert to column vector

        elif 0 <= LD_condition < 4:  # One of the suits is leading
            # Extract the suit features (last 4 features)
            num_suits = 4
            suit_features = card_features[:, -num_suits:]
            
            # Create mask for the cards that match the LD_condition suit
            suit_mask = suit_features[:, LD_condition] > 0
            
            # Expand the mask to match the shape of the logits (rows for cards)
            mask = suit_mask.float().unsqueeze(1)  # Convert to column vector
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
        max_value_index = output_tensor.argmax().item() 

        # Retrieve the corresponding row from the input tensor
        selected_row = input_tensor[max_value_index]

        return selected_row, max_value_index

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
    def make_bid(self, x):
        """
        Decide on a bid based on the current state and exploration strategy.

        :param x: Input tensor representing the state for bidding.
        :return: The selected bid action.
        """
        if self.position_bidders != 4:  # You are not last, you can bid whatever you want

            # In A low probability select randomly an action from the action_size 
            if random.random() < self.epsilon:  
                bid = random.randint(0, self.action_size-1)  # Assuming there are 5 possible actions (0 to 4)

            else:  # With probability 1 - ε, exploit
                prediction = self.bid_model(x)  # Forward pass
                #print(f"Output of bidding model: {prediction}")
                bid = torch.argmax(prediction).item()
        else:
            while True:
                prediction = self.bid_model(x)  #forward pass
                bid = torch.argmax(prediction).item()
                if bid + self.sum_bids != self.deck_size:
                    break
                else:
                    continue
        return bid
    
    ##FUNCION 2: that interferes with the Rikiki_game
    def ai_choose_card(self, input_tensor, LD_condition, len_hand):
        """
        Choose a card based on the current state, leading suit condition, and hand length.

        :param input_tensor: Tensor representing the current hand and game state.
        :param LD_condition: Condition for the leading suit.
        :param len_hand: Length of the hand.
        :return: A tuple containing the selected CustomCard and its index.
        """
        # Create the mask for nul vectors and if LD is active, masks non leading cards
        mask = self.create_mask(input_tensor, LD_condition, len_hand)
        # Call the forward function to get logits
        logits = self.card_model.forward(input_tensor)
        # Apply the mask and softmax to get the probabilities
        output_tensor = self.card_model.masked_softmax(logits, mask) ##this is the final output of the model
        #DECODING BACK to card 
        tensor_row, index_card = self.select_row(output_tensor, input_tensor)
        # Convert tensor row to Card
        selected_card = self.tensor_to_card(tensor_row)
        return selected_card, index_card

    
    #UPDATE BID MODEL
    def update_bid_model(self, state, action, reward, next_state, done):
        """
        Update the bidding model based on experience using the Bellman equation.

        :param state: Current state tensor.
        :param action: Action taken.
        :param reward: Reward received after taking the action.
        :param next_state: Next state tensor.
        :param done: Boolean indicating if the episode is done.
        """
        # Compute the target
        with torch.no_grad():
            target = reward  #Basically the target is the value of that state plus expected value in future state
            if not done:
                next_q_values = self.bid_model(next_state) #forward pass on next state
                target += self.gamma * torch.max(next_q_values)  #discount factor x the max argument action
                print(f"target: {target}")
        
        # Compute the current Q-value
        q_values = self.bid_model(state)  #forward pass returns a 1x3 tensor
        q_value = q_values[action] 

        # Compute the loss-> the loss is used to 
        loss = self.criterion(q_value, torch.tensor([target], dtype=torch.float32))
        self.bid_optimizer.zero_grad()
        loss.backward()
        #update the model parameters: 
        self.bid_optimizer.step()


    #UPDATE PLAY MODEL
    def update_play_model(self, state, action, reward, next_state, done):
        """
        Update the card selection model based on experience using the Bellman equation.

        :param state: Current state tensor.
        :param action: Action taken.
        :param reward: Reward received after taking the action.
        :param next_state: Next state tensor.
        :param done: Boolean indicating if the episode is done.
        """
        # Compute the target
        with torch.no_grad():
            target = reward  #Basically the target is the value of that state
            if not done:
                #print(f"next state: {next_state}")  
                next_q_values = self.card_model(next_state) #forward pass on next state

                #print(f"next q-values: {next_q_values}")
                target += self.gamma * torch.max(next_q_values)  #discount factor x the max argument action
                #print(f"target: {target}")
        
        # Compute the current Q-value
        q_values = self.card_model(state)  #forward pass returns a 1x8 tensor
        q_value = q_values[0, action] 
        loss = self.criterion(q_value, torch.tensor([target], dtype=torch.float32))
        self.card_optimizer.zero_grad()
        loss.backward()
        self.card_optimizer.step()

    #function to save a model to a specific filename
    @staticmethod
    def save_model(model, filename):
        torch.save(model.state_dict(), filename)

    
    