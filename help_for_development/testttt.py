def get_highest_atout_card_value(trick_cards, atout_suit):
    # Filter the trick cards for those that match the atout suit
    atout_cards = [tc[0] for tc in trick_cards if tc[0].suit == atout_suit]
    
    # If there are atout cards in the trick, find the highest value; otherwise, return None
    if atout_cards:
        highest_atout_card_value = max(card_value(card) for card in atout_cards)
        return highest_atout_card_value
    else:
        return None

# Usage example:
trick_cards = [(Card(value='King', suit='Clubs'), 'JOE Conservative Player')]
suit = 'Clubs'

highest_atout_card_value = get_highest_atout_card_value(trick_cards, suit)

