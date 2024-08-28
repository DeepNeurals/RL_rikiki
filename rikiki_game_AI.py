import pydealer
import torch
from collections import defaultdict
from tabulate import tabulate

value_to_idx = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
    '9': 7, '10': 8, 'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
}

suit_to_idx = {
    'Diamonds': 0, 'Spades': 1, 'Hearts': 2, 'Clubs': 3
}

###GAME RELATED STUFF####
#create CustomCard class
class CustomCard(pydealer.Card):
    """
    A custom card class that extends the pydealer.Card class to include a custom value attribute.
    """
    def __init__(self, number, suit, custom_value=None):
        """
        Initializes a CustomCard instance.

        Args:
            number (str): The value of the card (e.g., '2', 'Jack').
            suit (str): The suit of the card (e.g., 'Hearts').
            custom_value (int, optional): A custom numerical value for the card. Defaults to None.
        """
        super().__init__(number, suit)
        self.custom_value = custom_value if custom_value is not None else self.default_value()

    def default_value(self):
        """
        Provides a default numerical value for the card based on its face value.

        Returns:
            int: The numerical value of the card.
        """
        value_map = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 10,
            'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14
        }
        return value_map.get(self.value, 0)
    
#create a Custom Deck
def create_custom_deck():
    """
    Creates a custom deck of cards using the CustomCard class.

    Returns:
        list: A list of CustomCard objects representing the deck.
    """
    deck = pydealer.Deck()
    custom_deck = [CustomCard(card.value, card.suit) for card in deck]
    return custom_deck

custom_deck = create_custom_deck()

class RikikiGame:
    def __init__(self, num_players, ai_player_index, conservative_player_index, HUMAN_player_index, ALICE_player_index, starting_deck_size, total_rounds):
        """
        Initializes a RikikiGame instance.

        Args:
            num_players (int): The number of players in the game.
            ai_player_index (int): The index of the AI player.
            conservative_player_index (int): The index of the conservative player.
            HUMAN_player_index (int): The index of the human player.
            ALICE_player_index (int): The index of the Alice  player.
            starting_deck_size (int): The number of cards dealt to each player at the first round.
            total_rounds (int): The total number of rounds to be played in the game.
        """
        self.num_players = num_players
        self.ai_player_index = ai_player_index
        self.conservative_player_index = conservative_player_index
        self.HUMAN_player_index = HUMAN_player_index
        self.ALICE_player_index = ALICE_player_index
        self.starting_deck_size = starting_deck_size
        self.current_deck_size = starting_deck_size
        self.deck = pydealer.Deck()
        self.players = []
        self.bids = []
        self.atout = None
        self.trick = pydealer.Stack()
        self.scores = defaultdict(int)
        self.pli_scores = {0: 0, 1: 0, 2: 0, 3: 0}
        
        self.rewards = defaultdict(int)
        self.game_rewards = defaultdict(int)
        self.starting_player = 0
        self.consec_wins_bonus = -1 #initialise at start -1
        self.max_consec = 0
        self.total_rounds = total_rounds
         
    def deal_cards(self, round_number):
        """
        Deals cards to each player for the current round and resets relevant game state.

        Args:
            round_number (int): The number of the current round.
        """
        self.deck.shuffle()
        self.players = []
        for _ in range(self.num_players):
            dealt_cards = self.deck.deal(self.current_deck_size)
            player_hand = [CustomCard(card.value, card.suit) for card in dealt_cards]
            self.players.append(player_hand)
        self.bids = [None] * self.num_players
        self.pli_scores = defaultdict(int)
        self.starting_player = round_number % self.num_players
    
    def select_atout(self):
        """
        Selects the atout for the round.
        """
        self.atout = self.deck.deal(1)[0]

    #determine the winner of the pli in this trick based on: leading suit & atout 
    def determine_winner(self, trick_cards, leading_suit):
        """
        Determines the winner of the current trick based on the leading suit and atout.

        Args:
            trick_cards (list): A list of tuples, where each tuple contains a card and the player number.
            leading_suit (str): The suit of the card that started the trick.

        Returns:
            tuple: A tuple containing the winning card and the player number who won the trick.
        """
        winning_card = None
        winning_player = None

        for card, player_num in trick_cards:
            if winning_card is None:
                winning_card = card
                winning_player = player_num
            else:
                #if the card suit is the atout suit 
                if card.suit == self.atout.suit:
                    #and the winning card is not atout or the new card is bigger than winning_card, new card wins
                    if winning_card.suit != self.atout.suit or card > winning_card:
                        winning_card = card
                        winning_player = player_num
                #else: the card suit is the leading suit
                elif card.suit == leading_suit:
                    #and the winning card is not atout and card bigger than winning card, update the winning card
                    if winning_card.suit != self.atout.suit and card > winning_card:
                        winning_card = card
                        winning_player = player_num
        return winning_card, winning_player

    #calculate the scores for the round: based on predicted pli_scores and actual
    def calculate_scores(self):
        """
        Calculates the scores for the round based on predicted and actual pli_scores.
        """
        for player_num in range(self.num_players):
            predicted = self.bids[player_num]
            actual = self.pli_scores[player_num]
            if predicted == actual: #if correct you win 5 points per correct predicted pli
                self.consec_wins_bonus += 1 
                self.rewards[player_num] = max(5, 5 + (3* predicted)) + self.consec_wins_bonus*40 #5 for correct pli, and +3 per correctb pli. 
                self.game_rewards[player_num] = max(5, 5 + (3* predicted))
                self.scores[player_num] += self.game_rewards[player_num] 
            else: #if not correct you loose -2 per  difference between true pli and predicted
                self.consec_wins_bonus = -1
                self.rewards[player_num] = -abs(predicted-actual)*2  # penalising the errors
                self.game_rewards[player_num] = -abs(predicted-actual)*2
                self.scores[player_num] += self.game_rewards[player_num]  
        if  self.consec_wins_bonus > self.max_consec:
            self.max_consec = self.consec_wins_bonus
        
    #reset deck and trick for next round
    def reset_for_next_round(self):
        """
        Resets the deck and trick for the next round of play.
        """
        self.deck = pydealer.Deck()
        self.trick = pydealer.Stack()

    def get_player_role(self, player_num):
        """
        Retrieves the role of a player based on their player number.

        Args:
            player_num (int): The player number.

        Returns:
            str: The role of the player.
        """
        if player_num == self.conservative_player_index:
            return "JOE Conservative Player"
        elif player_num == self.ai_player_index:
            return "AI Player"
        elif player_num == self.HUMAN_player_index:
            return "HUMAN Player"
        elif player_num == self.ALICE_player_index:
            return "Alice random Player"
        else:
            return "Random Player"
    
    def get_player_index(self, role):
        """
        Retrieves the player number based on their role.

        Args:
            role (str): The role of the player.

        Returns:
            int: The player number corresponding to the role.
        """
        if role == "JOE Conservative Player":
            return self.conservative_player_index
        elif role == "AI Player":
            return self.ai_player_index
        elif role == "HUMAN Player":
            return self.HUMAN_player_index
        elif role == "Alice random Player":
            return self.ALICE_player_index
        else:
            return "Role not recognised"

    #printing function
    def print_scores(self):
        """
        Prints the scores of all players in the game.
        """
        for player_num in range(self.num_players):
            role = self.get_player_role(player_num)
            print(f"{role} (Player {player_num + 1}) score: {self.scores[player_num]}")
            
    def print_overview_table(self):
        """
        Prints an overview table of the round, including bids, actual tricks, rewards, and scores.
        """
        table_data = []
        for player_num in range(self.num_players):
            row = [
                f"{self.get_player_role(player_num)} (Player {player_num + 1})",
                self.bids[player_num],
                self.pli_scores[player_num],
                self.game_rewards[player_num],
                self.scores[player_num]
            ]
            table_data.append(row)
        headers = ["Player", "Bid", "Actual Tricks","Reward", "Score"]
        print(f"\nRound Overview - hand size: {self.current_deck_size}")
        print(tabulate(table_data, headers, tablefmt="grid"))

    
    def update_game_state(self, player_index):
        """
        Updates and encodes the game state for a specific player into a tensor representation.

        Args:
            player_index (int): The index of the player whose game state is to be updated.

        Returns:
            torch.Tensor: A tensor representation of the game state for the specified player.
        """
        # general game state
        game_state = {
            "atout": self.atout,
            "current_deck_size": self.current_deck_size,
        }
        # Count the number of cards with each rank in the AI player's hand
        for rank, value in [('ace', 'Ace'), ('king', 'King'), ('queen', 'Queen')]:
            game_state[f"num_{rank}s_in_hand"] = sum(card.value == value for card in self.players[player_index])
        # # Count atout cards in AI player's hand
        game_state["num_atout_cards_in_hand"] = sum(card.suit == self.atout.suit for card in self.players[player_index])
        
        for player_idx in range(self.num_players):
            if player_idx != player_index:
                player_bid = self.bids[player_idx] if player_idx < len(self.bids) else 0
                game_state[f"player_{player_idx + 1}_bid"] = player_bid
    
        # Extract relevant game state information
        num_aces = game_state.get("num_aces_in_hand", 0)
        # num_kings = game_state.get("num_kings_in_hand", 0)
        # num_queens = game_state.get("num_queens_in_hand", 0)
        num_atout_cards = game_state.get("num_atout_cards_in_hand", 0)
        current_deck_size = game_state.get("current_deck_size", 0)
        # player_1_bid = game_state.get("player_1_bid")
        # player_1_bid = -1 if player_1_bid is None else player_1_bid
        # player_2_bid = game_state.get("player_2_bid")
        # player_2_bid = -1 if player_2_bid is None else player_2_bid
        # player_3_bid = game_state.get("player_3_bid")
        # player_3_bid = -1 if player_3_bid is None else player_3_bid

        # Encode game state information into a tensor  
        # state_representation = torch.tensor([
        #     num_aces, num_kings, num_queens, num_atout_cards, current_deck_size,
        #     player_1_bid, player_2_bid, player_3_bid
        # ], dtype=torch.float)   
        
        #reduced version for learning quicker
        state_representation = torch.tensor([
            num_aces, num_atout_cards, current_deck_size
        ], dtype=torch.float)  #state is a 1x3 tensor
        return state_representation

    #Input tensor encoding
    def one_hot_encode_card(self, card):
        """
        Encodes a card into a one-hot tensor representation.

        Args:
            card (CustomCard): The card to be encoded.

        Returns:
            torch.Tensor: A tensor representing the card in one-hot encoding format.
        """
        # One-hot encoding for card value (13 possible values)
        value_one_hot = torch.zeros(13)
        if card is not None:
            value_index = value_to_idx[card.value]
            value_one_hot[value_index] = 1

        # One-hot encoding for card suit (4 possible suits)
        suit_one_hot = torch.zeros(4)
        if card is not None:
            suit_index = suit_to_idx[card.suit]
            suit_one_hot[suit_index] = 1

        # Concatenate the two one-hot encodings (value + suit)
        card_one_hot = torch.cat((value_one_hot, suit_one_hot))

        return card_one_hot

    def process_hand(self, cards, trick_cards, tricks_won, tricks_predicted):
        """
        Processes and encodes the player's hand and trick cards into a tensor representation.

        Args:
            cards (list): A list of cards in the player's hand.
            trick_cards (list): A list of tuples containing trick cards and the corresponding player.
            tricks_won (int): The number of tricks won by the player.
            tricks_predicted (int): The number of tricks predicted by the player.

        Returns:
            torch.Tensor: A tensor representing the processed hand and trick cards.
        """
        encoded_hand = []
        
        # Add played cards as null vectors if a card is played
        for i in range(self.total_rounds):
            if i < len(cards):
                encoded_hand.append(self.one_hot_encode_card(cards[i]))
            else:
                # Represent played cards as null vectors (zeros)
                encoded_hand.append(torch.zeros(17))

        # Trick Cards mapping
        player_indices = {'JOE Conservative Player': 0, 'HUMAN bidder': 1, 'Alice random bidder': 2}
        
        # Initialize with None or zeros for all players
        trick_card_placeholders = [None, None, None]

        # Place trick cards in the correct positions
        for card, player in trick_cards:
            idx = player_indices.get(player)
            if idx is not None:
                trick_card_placeholders[idx] = card

        # Encode the trick cards (if no card, use null vector)
        encoded_trick_cards = [self.one_hot_encode_card(card) if card is not None else torch.zeros(17) for card in trick_card_placeholders]

        # Combine the hand and trick cards
        combined_cards = encoded_hand + encoded_trick_cards

        # Add tricks won and predicted tricks as additional information
        tricks_info = torch.tensor([tricks_won, tricks_predicted], dtype=torch.float32)

        # Add the tricks info to each card vector, expanding the 17-dim vectors to 19-dim
        combined_input = [torch.cat((card, tricks_info)) for card in combined_cards]

        # Stack all rows into a single tensor
        return torch.stack(combined_input)
