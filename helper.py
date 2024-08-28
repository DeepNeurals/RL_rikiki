    
from tabulate import tabulate
from colorama import Fore, Style, init
# import numpy as np
import csv

##Helper function 
def store_score(score, actions, true_actions, CSV_FILE_PATH):
    # Open the CSV file in append mode
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the score to the CSV file
        writer.writerow([score]+actions+ true_actions)

def store_human_score(scorehuman, CSV_HUMAN_PATH):
    print(f'Scorehuman: {scorehuman}')
   
    # Convert the Tensor to a list or another iterable format
    if isinstance(scorehuman, (list, tuple)):
        score_to_write = scorehuman
    else:
        # Convert a scalar Tensor to a single-element list
        score_to_write = [scorehuman.item()] if hasattr(scorehuman, 'item') else [scorehuman]
    with open(CSV_HUMAN_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the score to the CSV file
        writer.writerow(score_to_write)
    

def display_game_info(atout, bids_played_by_others, hand_of_human_player):
    # Formatting the hand as a string
    hand_str = ", ".join([f"{card.value} of {card.suit}" for card in hand_of_human_player])

    # Creating the table data
    table_data = [
        ["Atout", atout],
        ["Bids Played by Others", str(bids_played_by_others)],
        ["Hand of Human Player", hand_str]
    ]
    # Print the table
    print(tabulate(table_data, headers=["", "Game information"], tablefmt="pretty"))

def display_game_status(atout, trick_cards, hand_of_human_player, tricks_predicted, tricks_won):
    # Formatting the hand and trick cards as strings
    hand_str = ", ".join([f"{card.value} of {card.suit}" for card in hand_of_human_player])
    trick_cards_str = ", ".join([f"{card.value} of {card.suit} by {player}" for card, player in trick_cards])

    # Creating the table data
    table_data = [
        ["Atout", atout],
        ["Trick Cards Played", trick_cards_str if trick_cards else "None"],
        ["HUMAN Player Hand", hand_str],
        ["You Predicted", tricks_predicted],
        ["Currently Won", tricks_won]
    ]
    # Print the table
    print(tabulate(table_data, headers=["", "Game status"], tablefmt="pretty"))

def print_table(atout_card, leading_suit, trick_cards, winning_player_role):
    """
    Print a formatted table showing the atout card, leading suit, trick cards, and highlight the winner.

    :param atout_card: The atout card.
    :param leading_suit: The leading suit card.
    :param trick_cards: List of tuples containing Card and player role.
    :param winning_player_role: Index of the player who won the trick.
    """
    # Print the atout card and leading suit
    print(f"Atout card: {atout_card}")
    print(f"Leading suit card: {leading_suit}")

    # Prepare data for tabulate
    table_data = []
    for card, player_role in trick_cards:
        if player_role == winning_player_role:
            # Highlight the winner in green
            player_role = f"{Fore.GREEN}{player_role}{Style.RESET_ALL}"
            card = f"{Fore.GREEN}{card}{Style.RESET_ALL}"
        table_data.append([player_role, card])

    # Print the table
    print("\nTrick Cards Played:")
    print(tabulate(table_data, headers=['Player Role', 'Card Played'], tablefmt='grid'))

    # Print who won the trick
    print(f"\n{Fore.GREEN}{winning_player_role} wins the trick{Style.RESET_ALL}")
