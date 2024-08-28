#This is a minimum version 
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from ascii_art_utils import create_ascii_art, colorize_text
from helper import store_score, store_human_score, display_game_info, display_game_status, print_table
from rikiki_game_AI import RikikiGame
from ai_agent import AIAgent
from collections import defaultdict
from datetime import datetime

suit_to_idx = {
    'Diamonds': 0,
    'Spades': 1,
    'Hearts': 2,
    'Clubs': 3
}

class Training:
    def __init__(self, game, ai_agent, ai_player_index, ALICE_player_index, HUMAN_player_index, manual_input):
        """
        Initializes the Training class with game, AI agent, player indices, and input mode.

        Parameters:
        - game: The game instance to be used for training.
        - ai_agent: The AI agent that will be trained.
        - ai_player_index: The index of the AI player.
        - ALICE_player_index: The index of the ALICE player.
        - HUMAN_player_index: The index of the human player.
        - manual_input: Boolean indicating whether to use manual input for the human player.
        """
        self.game = game
        self.ai_agent = ai_agent
        self.ai_player_index = ai_player_index
        self.ALICE_player_index = ALICE_player_index
        self.HUMAN_player_index = HUMAN_player_index
        self.states = None
        self.manual_input = manual_input

        #for learning initialise these
        self.state_old = torch.zeros(1,3)
        self.AI_action = 0
        self.AI_bid = 0 
        self.AI_reward  = 0
        self.done = 0
        self.counter_of_past_bidders = 0 #initialise with zero
        self.total_round_reward = 0
        self.list_of_actions = []
        self.list_actual = []

        #for Card NN update
        self.counter_played = 0
        self.AI_play_reward = 0
        self.old_input_tensor = torch.zeros(11,19)
    
    

    def bidding_phase(self):
        """
        Conducts the bidding phase where each player makes a bid in a round-robin fashion.
        """
        starting_player = (self.game.starting_player + 1) % self.game.num_players #wrap the 4 players from 0-3. 
        for i in range(self.game.num_players):
            current_player = (starting_player + i) % self.game.num_players
            self.get_bid(current_player)
        self.counter_of_past_bidders = 0 #reinistialise the counter of bidders for next round

    def get_bid(self, player_num):
        """
        Determines the bid for a given player based on their role and current game state.

        Parameters:
        - player_num: The index of the player making the bid.
        """
        total_bid_sum = sum(bid if bid is not None else 0 for bid in self.game.bids)
        self.counter_of_past_bidders += 1  # Increment counter
        num_players = self.game.num_players

        #Joe and Alice play 1 when possible:
        if player_num == self.game.conservative_player_index or player_num == self.game.ALICE_player_index:
            if self.counter_of_past_bidders != num_players:
                bid = 1
                #print('I bid whatever I want, I am not last!', self.counter_of_past_bidders)
            else:
                required_bid = self.game.current_deck_size - total_bid_sum
                if required_bid == 1:
                    bid = 0
                    #print(f'I bid {bid} to meet the condition, my bet is:', bid)
                else:
                    bid = 1
                    #print('Bidding zero works')

        # # Joe (Conservative Player)
        # if player_num == self.game.conservative_player_index:
        #     if self.counter_of_past_bidders != num_players:
        #         bid = 0
        #         #print('I bid whatever I want, I am not last!', self.counter_of_past_bidders)
        #     else:
        #         required_bid = self.game.current_deck_size - total_bid_sum
        #         if required_bid == 0:
        #             bid = 1
        #             #print(f'I bid {bid} to meet the condition, my bet is:', bid)
        #         else:
        #             bid = 0
        #             #print('Bidding zero works')
            
        # HUMAN PLAYER
        elif player_num == self.HUMAN_player_index:
            display_game_info(self.game.atout, self.game.bids, self.game.players[player_num])
             # Ask for the human player's input
            if self.manual_input == True:
                if self.counter_of_past_bidders == num_players:
                    while True:
                        try:
                            bid = int(input("Please enter your bid: "))
                            if 0 <= bid <= self.game.current_deck_size and total_bid_sum+bid != self.game.current_deck_size:
                                break
                            else:
                                print(f"Invalid bid. Please enter a number between 0 and {self.game.current_deck_size}. And not equal to the sum of number of cards.")
                        except ValueError:
                            print("Invalid input. Please enter a valid number.")
                else:
                    # Ask the human player for their bid
                    while True:
                        try:
                            bid = int(input("Please enter your bid: "))
                            if 0 <= bid <= self.game.current_deck_size:
                                break
                            else:
                                print(f"Invalid bid. Please enter a number between 0 and {self.game.current_deck_size}.")
                        except ValueError:
                            print("Invalid input. Please enter a valid number.")
            else:
                state_human = self.game.update_game_state(self.HUMAN_player_index)
                print(f'State of human player: {state_human}')
                n_atout_human = state_human[1]
                if self.counter_of_past_bidders == num_players:
                    if n_atout_human + total_bid_sum != self.game.current_deck_size:
                        bid = n_atout_human
                    else:
                        bid = n_atout_human + 1
                #print(f'Bob bids {bid} as the last player')
                else:
                    bid = n_atout_human
               
        # # Alice (Random Bidder)
        # elif player_num == self.game.ALICE_player_index:
        #     # self.alice_states = self.game.update_game_state(self.ALICE_player_index) #define the state information
        #     # self.alice_ai_agent.update_agent_state(self.alice_states) #pass the state informations

        #     if self.counter_of_past_bidders != num_players:
        #         bid = random.randint(0, self.game.current_deck_size)
        #     else:
        #         while True:
        #             bid = random.randint(0, self.game.current_deck_size)
        #             if bid + total_bid_sum != self.game.current_deck_size:
        #                 break

        # AI Player: here the AI agent makes the bid
        elif player_num == self.game.ai_player_index:   
            # bid = 1  ##set like this to reduce the processing effort for the model
            # self.AI_bid = bid
            self.ai_agent.sum_bids = total_bid_sum
            self.ai_agent.position_bidders = self.counter_of_past_bidders

            #Before bidding retrieve the states of the AI-player
            self.states = self.game.update_game_state(self.ai_player_index)
            #self.ai_agent.memory.append((self.state_old, self.AI_action, self.AI_reward, self.states, self.done))

            bid = self.ai_agent.make_bid(self.states) #in this function the forward pass happens
            #print('The AI agent made its choice', bid)
            self.AI_bid = bid

            #UPDATE bidding model with previous round information and new state 
            self.ai_agent.update_bid_model(self.state_old, self.AI_action, self.AI_reward, self.states, self.done)
        
        # Handle unexpected player numbers
        else:
            #print('Unexpected player index:', player_num)
            raise Exception('Error: more than 4 players')

        #print(f"{role} (Player {player_num + 1}) bids: {bid}")
        self.game.bids[player_num] = bid

    #define winning card function
    def win_card(self, trick_cards, leading_suit):
        """
        Determines the winning card from a trick based on the leading suit and atout suit.

        Parameters:
        - trick_cards: List of tuples containing cards and their respective player roles.
        - leading_suit: The suit that is leading in the current trick.

        Returns:
        - Tuple containing the winning card and its associated player role.
        """
        # Separate cards into atout and leading suit groups
        atout_cards = [(card, player_role) for card, player_role in trick_cards if card.suit == self.game.atout.suit]
        leading_cards = [(card, player_role) for card, player_role in trick_cards if card.suit == leading_suit]

        if atout_cards:
            # Determine the highest atout card and the associated player role
            winning_card, winning_player_role = max(atout_cards, key=lambda x: x[0].custom_value)
        else:
            # Determine the highest card of the leading suit and the associated player role
            winning_card, winning_player_role = max(leading_cards, key=lambda x: x[0].custom_value)

        return winning_card, winning_player_role

    def det_play_reward(self, winnin_ply_role, input_tensor):
        """
        Determines the reward based on whether the AI player won the trick and its prediction.

        Parameters:
        - winnin_ply_role: The role of the player who won the trick.
        - input_tensor: The tensor containing the trick details and predictions.

        Returns:
        - Reward value based on the outcome of the trick.
        """
        trick_won = input_tensor[0, -2]        # Value from row 0, column -2
        trick_predicted = input_tensor[0, -1]  # Value from row 0, column -1

        if winnin_ply_role == "AI Player": # you won the trick
            if trick_won<trick_predicted:
                reward = 10 #big reward for winning when you predicted higher
            else:
                reward = -10 #big reward for winning when you predicted higher

        else: #you lose the trick
            if trick_won==trick_predicted:
                reward = 10
            elif trick_won<trick_predicted:
                reward = -10
            else:
                reward = 10
        return reward

    def det_leading_suit_cards(self, leading_cards):
        """
        Determines the lowest and highest cards in the leading suit from the player's hand.

        Parameters:
        - leading_cards: List of cards in the leading suit.

        Returns:
        - Tuple containing the lowest and highest leading suit cards.
        """
        print(f'Leading card in hand player: { leading_cards }')
        lowest_leading_suit_card = min(leading_cards, key=lambda x: x.custom_value)
        highest_leading_suit_card = max(leading_cards, key=lambda x: x.custom_value)
        return lowest_leading_suit_card, highest_leading_suit_card
    
    def det_posses_low_atout(self, hand):
        """
        Checks if the hand contains any atout cards with a value lower than 9.

        Parameters:
        - hand: List of cards in the player's hand.

        Returns:
        - Boolean indicating whether the player has low atout cards.
        """
        atout_in_hand = [card for card in hand if card.suit == self.game.atout.suit]
        if len(atout_in_hand) > 0:
            print(f'atout_in_hand {atout_in_hand}')
            for card in atout_in_hand:
                if card.custom_value < 9:
                    low_atout = True
                else:
                    low_atout = False
        else:#no atout cards
            low_atout = False

        return low_atout
    
    def det_atout_cards(self, atout_cards, highest_atout_trick):
        """
        Determines the lowest, medium, and highest atout cards from a hand.

        Parameters:
        - atout_cards: List of atout cards in the player's hand.
        - highest_atout_trick: The highest atout card played in the current trick.

        Returns:
        - Tuple containing the lowest, medium, and highest atout cards.
        """
        lowest_atout_card = min(atout_cards, key=lambda x: x.custom_value)
        highest_atout_card = max(atout_cards, key=lambda x: x.custom_value)
        print(f'Hand {atout_cards}, highest_atout_trick: {highest_atout_trick}')

        # Filter the atout cards to find potential medium atout cards
        print(f'Card in atout: {atout_cards}')
        potential_medium_atouts = [
            card for card in atout_cards 
            if card.custom_value != lowest_atout_card.custom_value and
            card.custom_value != highest_atout_card.custom_value and
            card.custom_value < highest_atout_trick
        ]

        # If there are any valid medium atout cards, choose the highest among them
        if potential_medium_atouts:
            medium_atout_card = max(potential_medium_atouts, key=lambda x: x.custom_value)
        else:
            medium_atout_card = None  # No medium atout card that meets the criteria

        return lowest_atout_card, medium_atout_card, highest_atout_card, 
    
    def det_non_atout_cards(self, hand):
        """
        Determines the lowest and highest non-atout cards from a hand.

        Parameters:
        - hand: List of cards in the player's hand.

        Returns:
        - Tuple containing the lowest and highest non-atout cards. If there are no non-atout cards, returns None for both.
        """
        non_atout_cards = [card for card in hand if card.suit != self.game.atout.suit]
        if len(non_atout_cards)>0:
            lowest_non_atout_card = min(non_atout_cards, key=lambda x: x.custom_value)
            highest_non_atout_card = max(non_atout_cards, key=lambda x: x.custom_value)
        else:
            print(f'Only atout cards in hand!!!')
            lowest_non_atout_card = None
            highest_non_atout_card = None

        return lowest_non_atout_card, highest_non_atout_card


    def play_all_tricks(self): #play all tricks in the current round
        """
        Plays all tricks in the current round of the game.
        Iterates through the number of tricks based on the deck size and plays each trick.
        """
        for _ in range(self.game.current_deck_size):
            self.play_trick()

    def play_trick(self):
        """
        Plays a single trick in the current round of the game.
        Each player plays a card according to their role and the current state of the game.
        Determines the winning card and updates the game state accordingly.
        """
        trick_cards = []
        leading_suit = None
        for player_num in range(self.game.num_players):
            current_player = (self.game.starting_player + player_num) % self.game.num_players
            role = self.game.get_player_role(current_player)

            #For JOE and Alice bid automatically
            if current_player != self.ai_player_index and current_player != self.HUMAN_player_index:
                #Playing strategy for all the players excluding the AI player and human player
                if leading_suit: 
                    leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                    if leading_cards: # Play the highest card of the leading suit that you have
                        card = max(leading_cards, key=lambda x: x.custom_value)
                    else: # No leading suit cards, check for atout cards
                        atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout.suit]
                        if atout_cards: # Play the highest atout card
                            card = max(atout_cards, key=lambda x: x.custom_value)
                        else: # Play the highest non-atout card
                            card = max(self.game.players[current_player], key=lambda x: x.custom_value)
                else: # No leading suit established, play the highest card available in your player' s deck
                    card = max(self.game.players[current_player], key=lambda x: x.custom_value)
            
            
            # For human player bid manually or automatically 
            elif current_player == self.HUMAN_player_index:
                if len(self.game.players[current_player]) == 1:
                    idx_card = 0
                    card = self.game.players[current_player][idx_card]
                else:
                    tricks_won_human = self.game.pli_scores[self.HUMAN_player_index]
                    tricks_predicted_human = self.game.bids[self.HUMAN_player_index]
                    if self.manual_input == True:
                        display_game_status(self.game.atout, trick_cards, self.game.players[current_player], tricks_predicted_human, tricks_won_human)
                        if leading_suit: #leading suit to respect
                            leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                            if leading_cards:
                                print(f"You have to play a card of this leading suit: {leading_suit}")
                                while True:
                                    try:
                                        idx_card = int(input("\033[1mPlease enter the index of the card you want to play:\033[0m "))
                                        if idx_card < len(self.game.players[current_player]):
                                            card = self.game.players[current_player][idx_card]
                                            if card.suit == leading_suit:
                                                confirm = input(f"You selected: {card}. Do you want to play this card? (Y/N): ").strip().upper()
                                                if confirm == 'Y':
                                                    break
                                                else:
                                                    print(f'\033[ You can play a different card\033[0m')
                                            else:
                                                print(f'\033[ Card does not match the leading suit\033[0m')
                                        else:
                                            print(f'\033[Invalid index, try again\033[0m')
                                    except ValueError:
                                        print(f'\033[91mInvalid input. Please enter a valid number.\033[0m')
                            else:
                                print(f"You have no leading suit, you can play whatever you want!")
                                # No leading suit to respect
                                while True:
                                    try:
                                        idx_card = int(input("\033[1mPlease enter the index of the card you want to play:\033[0m "))
                                        if idx_card < len(self.game.players[current_player]):
                                            card = self.game.players[current_player][idx_card]
                                            confirm = input(f"You selected: {card}. Do you want to play this card? (Y/N): ").strip().upper()
                                            if confirm == 'Y':
                                                break
                                            else:
                                                print(f'\033[You need to play a different card\033[0m')
                                        else:
                                            print(f'\033[Invalid index, try again\033[0m')
                                    except ValueError:
                                        print(f'\033[91mInvalid input. Please enter a valid number.\033[0m')
                        else:
                            print(f"You are the first to play, play whatever you want!")
                            # No leading suit to respect
                            while True:
                                try:
                                    idx_card = int(input("\033[1mPlease enter the index of the card you want to play:\033[0m "))
                                    if idx_card < len(self.game.players[current_player]):
                                        card = self.game.players[current_player][idx_card]
                                        confirm = input(f"You selected: {card}. Do you want to play this card? (Y/N): ").strip().upper()
                                        if confirm == 'Y':
                                            break
                                        else:
                                            print(f'\033[You need to play a different card\033[0m')
                                    else:
                                        print(f'\033[Invalid index, try again\033[0m')
                                except ValueError:
                                    print(f'\033[91mInvalid input. Please enter a valid number.\033[0m')

                    else: #play according to my optimal strategy
                        tricks_predicted_human = self.game.bids[self.HUMAN_player_index]
                        if leading_suit: #there is a leading suit 
                            leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                            if leading_cards: #you have leading cards
                                if len(leading_cards) >= 2: #you have multiple leading cards
                                    highest_leading_card_value = max(tc[0].value for tc in trick_cards if tc[0].suit == leading_suit)

                                    # Check if the player has a card of the leading suit that is higher than the highest leading suit card in trick_cards
                                    higher_card_exists = any(card.suit == leading_suit and card.value > highest_leading_card_value 
                                                            for card in self.game.players[current_player])
                                    
                                    low_leading_suit_card, high_leading_suit_card = self.det_leading_suit_cards(leading_cards)

                                    #adapt playing strategy
                                    if higher_card_exists is True and tricks_won_human<tricks_predicted_human:
                                        #print('Play the higher leading suit card that you have')
                                        card = high_leading_suit_card
                                    elif higher_card_exists is True and tricks_won_human>=tricks_predicted_human:
                                        #print('Play the lower leading suit card to be sure not to win')
                                        card = low_leading_suit_card
                                    elif higher_card_exists is False and tricks_won_human>=tricks_predicted_human:
                                        #print('consume the highest leading suit card for safety')
                                        card = high_leading_suit_card
                                    elif higher_card_exists is False and tricks_won_human<tricks_predicted_human:
                                        #print('play lowest leading suit card in order to conserve the highest for next tricks')
                                        card = low_leading_suit_card
                                    else:
                                        #print('Play any leading card')
                                        card = high_leading_suit_card

                                else: #play the only leading card you have
                                    print("I played an exceptional card!!!")
                                    card = max(leading_cards, key=lambda x: x.custom_value)
                            
                            else: #You have no leading card, play smart
                                #print(f'cards in trick_cards: {trick_cards}')
                                atout_cards_trick = [tc[0] for tc in trick_cards if tc[0].suit == self.game.atout.suit]
                                #print(f'atout_cards_trick: {atout_cards_trick}')
                                if len(atout_cards_trick) == 0:
                                    highest_trick_atout_card = 0
                                else: 
                                    highest_trick_atout_card = max(tc.custom_value for tc in atout_cards_trick if tc.suit == self.game.atout.suit)

                                #print(f'Highest trick atout card: {highest_trick_atout_card}')

                                #print(f'Hand of human player: {self.game.players[current_player]}')
                                higher_atout_card_exists = any(card.suit == self.game.atout.suit and card.custom_value > highest_trick_atout_card 
                                                            for card in self.game.players[current_player])
                                #print(f'Value of highest atout card exists: {higher_atout_card_exists}')

                                atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout.suit]
                                #print(f'Atout cards before passing: {atout_cards}')

                                if len(atout_cards) > 0: #more than one atout card
                                    lowest_atout_card, medium_atout_card, highest_atout_card = self.det_atout_cards(atout_cards, highest_trick_atout_card)
                                else:
                                    lowest_atout_card, medium_atout_card, highest_atout_card = None, None, None
                                
                                lowest_non_atout_card, highest_non_atout_card = self.det_non_atout_cards(self.game.players[current_player])

                                if higher_atout_card_exists is True and tricks_won_human<tricks_predicted_human:
                                    #print(f'play higher atout_card (the highest you have)')
                                    card = highest_atout_card
                                elif higher_atout_card_exists is True and tricks_won_human>=tricks_predicted_human:
                                    #print(f'play an atout_card that is lower than the one played or a very high non-atout card')
                                    if medium_atout_card is not None:
                                        card = medium_atout_card
                                    else:
                                        if highest_non_atout_card is not None:
                                            card = highest_non_atout_card
                                        else:
                                            #print('Very unlikerly scenario, play lowest card in hand')
                                            card = min(self.game.players[current_player], key=lambda x: x.custom_value)

                                elif higher_atout_card_exists is False and tricks_won_human>=tricks_predicted_human and highest_atout_card is not None:
                                    #print(f'play your highest atout card')
                                    card = highest_atout_card
                                elif higher_atout_card_exists is False and tricks_won_human<tricks_predicted_human:
                                    #print('play lowest card that is not an atout card') 
                                    if lowest_non_atout_card is not None:
                                        card = lowest_non_atout_card
                                    else:
                                        #print('Play lowest atout card to preserve the others')
                                        card = min(self.game.players[current_player], key=lambda x: x.custom_value)
                                else:
                                    #print("CANNOT IMAGINE THE SCENARIO")
                                    card = min(self.game.players[current_player], key=lambda x: x.custom_value)

                        else: #You are first to play smart!
                            posses_low_atout = self.det_posses_low_atout(self.game.players[current_player])
                            highest_trick_atout_card = 0
                            atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout.suit]
                            if len(atout_cards)>0:
                                lowest_atout_card, medium_atout_card, highest_atout_card = self.det_atout_cards(atout_cards, highest_trick_atout_card)
                            else:
                                lowest_atout_card, medium_atout_card, highest_atout_card = None, None, None

                            lowest_non_atout_card, highest_non_atout_card = self.det_non_atout_cards(self.game.players[current_player])

                            if tricks_won_human<tricks_predicted_human and posses_low_atout is True:
                                #print(f'play low atout to force other players to spend their atouts')
                                if lowest_atout_card is not None:
                                    card = lowest_atout_card
                                else:
                                    card = min(self.game.players[current_player], key=lambda x: x.custom_value)

                            elif tricks_won_human<tricks_predicted_human and posses_low_atout is False:
                                #print(f'Play highest card you have that is not atout to make it a leading suit')
                                if lowest_non_atout_card is not None:
                                    card = lowest_non_atout_card
                                else:
                                    card = min(self.game.players[current_player], key=lambda x: x.custom_value)

                            elif tricks_won_human>=tricks_predicted_human:
                                #print('Play lowest atout or non atout card that you have!')
                                if posses_low_atout is True:
                                    if lowest_non_atout_card is not None:
                                        card = lowest_atout_card
                                    else:
                                        card = min(self.game.players[current_player], key=lambda x: x.custom_value)
                                else: 
                                    if lowest_non_atout_card is not None:
                                        card = lowest_non_atout_card
                                    else:
                                        card = min(self.game.players[current_player], key=lambda x: x.custom_value)
                            else:
                                #print('Play any card you have')
                                card = max(self.game.players[current_player], key=lambda x: x.custom_value)
                            

            #bid using inference for AI
            elif current_player == self.ai_player_index:
                hand = self.game.players[current_player]
                len_hand = len(hand)
                tricks_won  = self.game.pli_scores[self.ai_player_index]
                tricks_predicted  = self.game.bids[self.ai_player_index]
                input_tensor = self.game.process_hand(hand, trick_cards, tricks_won, tricks_predicted)

                #before choosing a card we update the model
                if self.counter_played > 0: #do not update the first time 
                    self.ai_agent.update_play_model(self.old_input_tensor, self.old_index_card_chosen, self.AI_play_reward, input_tensor, self.done) 
                
                if leading_suit:
                    leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                    if leading_cards:
                        LD_condition = suit_to_idx[leading_suit] #create a Leading card condition to meet a Rikiki Rules
                        card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition, len_hand)  #this function will provide the card to play
                    else:
                        LD_condition = -1 #means there no condition
                        card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition, len_hand) #the model understands only card index for update
                else:
                    LD_condition = -1 #means there no condition
                    card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition, len_hand)

                ###add the end store the old input tensor
                self.counter_played += 1
                self.old_input_tensor = input_tensor
                self.old_index_card_chosen = idx_card

            else:
                raise Exception("Error with current player of players") 

            ###COLLECT card played by the players 
            trick_cards.append((card, role))
            if not leading_suit: #first player establishes leading suit
                leading_suit = card.suit
            print(f"Card player {role} wants to remove out of hand: {card}")
            self.game.players[current_player].remove(card)
        
        #DETERMINING a Trick winner
        winning_card, winning_player_role = self.win_card(trick_cards, leading_suit)
        self.AI_play_reward = self.det_play_reward(winning_player_role, self.old_input_tensor) #reward for play model

        #HERE we need to transform role->index 
        winning_player_index = self.game.get_player_index(winning_player_role)
        print_table(self.game.atout, leading_suit, trick_cards, winning_player_role)
        print(f'winning player index: {winning_player_index}')
        print(f'Starting player index before: {self.game.starting_player}')
        self.game.starting_player = winning_player_index
        print(f'Starting player index after: {self.game.starting_player}')
        self.game.pli_scores[winning_player_index] += 1

        

    def play_round(self, round_number):
        """
        Plays a round of the game based on the round number.

        Parameters:
        - round_number: Integer indicating the round number.
        """
        self.game.current_deck_size = self.game.starting_deck_size + round_number ##added round number so from 0-8
        self.game.deal_cards(round_number)
        self.game.select_atout()
        self.bidding_phase() 
        self.play_all_tricks()  
        self.game.calculate_scores()   #here the round reward is calculated
        #self.game.print_scores() #print the results
        self.game.print_overview_table() 

        ###HERE WE STORE important variables for update after a full round 
        self.state_old = self.states
        self.AI_action = self.AI_bid  
        self.AI_reward = self.game.rewards[3] #here we have the reward for the bid model
        
        #you need to store here the true bids
        self.AI_actual = self.game.pli_scores[3]

        #reset the cards for next round card distribution
        self.game.reset_for_next_round() #--> end of round 
        self.game.pli_scores = defaultdict(int) #for next round

        #shift the starting player with one at each round 
        self.game.starting_player += 1

    # Train the model 
    def trainer(self):
        """
        Trains the AI model by playing a specified number of games, recording the scores, and generating performance plots.

        This method initializes game settings, plays a series of games, and records scores and actions for each game. 
        It then saves these results to CSV files and generates plots to visualize the AI's performance.
        """
        scores_after_each_game = []
        scores_human_after_each_game = []
        n_games_played = 0 #initialisation of game counter

        while n_games_played<NUMBER_GAMES: 
            self.game.scores = defaultdict(int) ###--> this resets the scores to zero after each game 
            current_game_actions = []
            true_game_actions = []
            #Play all the rounds of a full game
            for round_number in range(TOTAL_ROUNDS): 
                self.play_round(round_number)
                current_game_actions.append(self.AI_action)
                true_game_actions.append(self.AI_actual)
            
            #at the end of every game: --> store into csv file for efficiency
            print(f"AI player index: {self.game.scores[ai_player_index]}")
            print(f"AI player index: {self.game.scores[HUMAN_player_index]}")
            store_score(self.game.scores[ai_player_index], current_game_actions, true_game_actions, CSV_FILE_PATH)
            store_human_score(self.game.scores[HUMAN_player_index], CSV_HUMAN)
            current_game_actions.clear()
            true_game_actions.clear()
            self.game.consec_wins_bonus = -1 
            n_games_played += 1

        ##WHILE LOOP FINSIHED
        print(f"----GAME {NUMBER_GAMES} PLAYED---, started with game zero")
        print(f"MAx consecutive round wins: {self.game.max_consec}")

        # Load scores from the CSV file
        scores_df = pd.read_csv(CSV_FILE_PATH, header=None)
        scores_after_each_game = scores_df[0].tolist()

        # Load scores from the CSV file
        scores_human = pd.read_csv(CSV_HUMAN, header=None)
        scores_human_after_each_game = scores_human[0].tolist()

        # Separate predicted bids (columns 1 to 8) and true bids (columns 9 to 16)
        predicted_bids = scores_df.iloc[:, 1:9]
        true_bids = scores_df.iloc[:, 9:]

        #SAVE THE MODELS- (Comment two lines below if you do not want to save!)
        #self.ai_agent.save_model(self.ai_agent.bid_model, model_filename)
        self.ai_agent.save_model(self.ai_agent.card_model, play_model_filename)

        #Game won vs game lost: (AI VS HUMAN)
        won = 0; lost = 0
        for ai_score, human_score in zip(scores_after_each_game, scores_human_after_each_game):
            if ai_score > human_score:
                won += 1  # AI wins if its score is higher than the human's score
            else:
                lost += 1  # AI loses if its score is less than or equal to the human's score
        labels = ['Won', 'Lost']
        sizes = [won, lost]
        
        # Calculate the average predicted/true bids per round
        average_predicted_bids = predicted_bids.mean(axis=0)
        average_true_bids = true_bids.mean(axis=0)
        # Calculate the overall average of predicted/true bids
        overall_average_predicted = predicted_bids.values.mean()
        overall_average_true = true_bids.values.mean()

        #CREATE FIGURE  with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # First subplot: Rewards over Time
        axs[0].plot(scores_after_each_game,color='green', label='AI Game Reward')
        axs[0].plot(scores_human_after_each_game,color='blue', label='Human player Game Reward')
        axs[0].set_xlabel('Game')
        axs[0].set_ylabel('Total Reward')
        axs[0].set_title('Rewards over Time')
        axs[0].legend()
        axs[0].grid(True)
        axs[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5252'])
        axs[1].set_title('Games Won vs. Lost')
        axs[2].plot(range(2, 10), average_predicted_bids, marker='o', label='Average Predicted Bids')
        axs[2].plot(range(2, 10), average_true_bids, marker='x', label='Average True Bids')
        axs[2].axhline(y=overall_average_predicted, color='darkblue', linestyle='--', label='Overall Average Predicted Bids')
        axs[2].axhline(y=overall_average_true, color='darkorange', linestyle='--', label='Overall Average True Bids')
        axs[2].set_xlabel('Round')
        axs[2].set_ylabel('Average Bid')
        axs[2].legend()
        axs[2].set_title('Average Predicted Bids vs Average True Bids per Round')
        plt.savefig(filename) #Save the plot to a file
        plt.close()
        print(f"Plot saved as {filename}")


if __name__ == "__main__":
    # Game Title in ASCII art
    text = "RIKIKI"
    ascii_art = create_ascii_art(text, font='standard')
    fg_color = 'cyan'  # Foreground color
    bold = True  # Make text bold
    print(colorize_text(ascii_art, fg_color, bold)) # Print the colored and styled ASCII art

    #Global parameters
    NUMBER_GAMES = 300; TOTAL_ROUNDS=8  
    #hyperparameters
    lr_bid = 0.001
    lr_play = 0.005

    #Game parameters
    num_players = 4  
    #Starting Player's order
    conservative_player_index = 0  # The JOE first player is the conservative player
    HUMAN_player_index = 1 #Bob/HUMAN player is second to play
    ALICE_player_index = 2 # ALice is third to play
    ai_player_index = 3  # The last player is the AI agent
    manual_input = False #play manually or not with human player
    starting_deck_size = 2
    state_size = 3 #bidmodel state size, reduced to 3 for faster computations

    #Image/CSV related settings
    image_output_dir = 'image_outputs' # Create the 'outputs' folders
    os.makedirs(image_output_dir, exist_ok=True)
    csv_output_dir = 'csv_outputs'
    os.makedirs(csv_output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{image_output_dir}/plot_{timestamp}.png'
    CSV_FILE_PATH = f'{csv_output_dir}/scores{timestamp}.csv'
    action_FILE_PATH = f'{csv_output_dir}/scores{timestamp}.csv'
    CSV_HUMAN = f'{csv_output_dir}/human_scores{timestamp}.csv'

    #Model related settings
    play_model_output_dir = 'play_model_outputs'
    os.makedirs(play_model_output_dir, exist_ok=True)
    play_model_filename = f'{play_model_output_dir}/model{timestamp}_{NUMBER_GAMES}.pth'
    bid_model_weights = 'bid_model_outputs/model20240827_085917_300.pth'
    card_model_weights = 'play_model_outputs/model20240827_111253_500.pth'

    ##Instatiate game and agent
    game = RikikiGame(num_players, ai_player_index, conservative_player_index, HUMAN_player_index, ALICE_player_index, starting_deck_size, TOTAL_ROUNDS) 
    ai_agent = AIAgent(starting_deck_size, state_size, TOTAL_ROUNDS, bid_model_weights, card_model_weights, lr_bid, lr_play)
    trainer = Training(game, ai_agent, ai_player_index, ALICE_player_index, HUMAN_player_index, manual_input)
    trainer.trainer()






