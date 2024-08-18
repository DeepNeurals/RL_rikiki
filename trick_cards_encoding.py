import torch

# Define the possible values and suits
value_to_idx = {
    '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
    '9': 7, '10': 8, 'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
}

suit_to_idx = {
    'Diamonds': 0, 'Spades': 1, 'Hearts': 2, 'Clubs': 3
}

# Define the Card class
class Card:
    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

def one_hot_encode_card(card):
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

def process_hand(cards, trick_cards, tricks_won, tricks_predicted):
    # Process cards in hand (4 cards)
    encoded_hand = [one_hot_encode_card(card) for card in cards]

    # Trick Cards mapping
    player_indices = {'JOE Conservative Player': 0, 'BOB liberal bidder': 1, 'Alice random bidder': 2}
    
    # Initialize with None or zeros for all players
    trick_card_placeholders = [None, None, None]

    # Place trick cards in the correct positions
    for card, player in trick_cards:
        idx = player_indices.get(player)
        if idx is not None:
            trick_card_placeholders[idx] = card

    # Encode the trick cards (if no card, use null vector)
    encoded_trick_cards = [one_hot_encode_card(card) if card is not None else torch.zeros(17) for card in trick_card_placeholders]

    # Combine the hand and trick cards
    combined_cards = encoded_hand + encoded_trick_cards

    # Add tricks won and predicted tricks as additional information
    tricks_info = torch.tensor([tricks_won, tricks_predicted], dtype=torch.float32)

    # Add the tricks info to each card vector, expanding the 17-dim vectors to 19-dim
    combined_input = [torch.cat((card, tricks_info)) for card in combined_cards]

    # Stack all rows into a single tensor
    return torch.stack(combined_input)

# Example usage
hand = [
    Card(value='5', suit='Diamonds'), 
    Card(value='4', suit='Spades'), 
    Card(value='2', suit='Diamonds'), 
    # Card(value='10', suit='Spades')
]

# Trick cards (played cards by players in the trick)
# trick_cards = [
#     (Card(value='2', suit='Spades'), 'JOE Conservative Player'),
#     (Card(value='4', suit='Hearts'), 'BOB liberal bidder'),
#     (Card(value='3', suit='Hearts'), 'Alice random bidder')
# ]

trick_cards = [
        (Card(value='3', suit='Hearts'), 'Alice random bidder')
]

# Tricks information
tricks_won = 2  # Tricks won so far
tricks_predicted = 3  # Tricks predicted (goal)

# Process the hand and game state into the input tensor
hand_tensor = process_hand(hand, trick_cards, tricks_won, tricks_predicted)

print("Input Tensor:")
print(hand_tensor)
