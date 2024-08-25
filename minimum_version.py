#This is a minimum version 

from rikiki_game_AI import RikikiGame
from ai_agent import AIAgent
import torch
import random
import matplotlib.pyplot as plt
from collections import defaultdict
# import numpy as np
import csv
import pandas as pd
import os
from datetime import datetime

suit_to_idx = {
    'Diamonds': 0,
    'Spades': 1,
    'Hearts': 2,
    'Clubs': 3
}

class Training:
    def __init__(self, game, ai_agent, alice_ai_agent, ai_player_index, ALICE_player_index):
        self.game = game
        self.ai_agent = ai_agent
        self.ai_player_index = ai_player_index
        self.alice_ai_agent = alice_ai_agent
        self.ALICE_player_index = ALICE_player_index
        self.states = None

        #for learning initialise these
        self.state_old = torch.zeros(1,8)
        self.state_alice_old = torch.zeros(1,8)
        self.AI_action = 0
        self.AI_bid = 0 
        self.AI_reward  = 0
        self.done = 0
        self.counter_of_past_bidders = 0 #initialise with zero
        self.total_round_reward = 0
        self.list_of_actions = []
        self.list_actual = []

        #ALICE rewards
        self.ALICE_reward = 0
        self.ALICE_action = 0
        self.alice_total_round_reward = 0
        self.alice_states = None

        # #for Card NN update
        self.counter_played = 0

    def bidding_phase(self):
        starting_player = (self.game.starting_player + 1) % self.game.num_players #wrap the 4 players from 0-3. 
        for i in range(self.game.num_players):
            current_player = (starting_player + i) % self.game.num_players
            self.get_bid(current_player)
        #reinistialise the counter of bidders for next round
        self.counter_of_past_bidders = 0

    def get_bid(self, player_num):
        total_bid_sum = sum(bid if bid is not None else 0 for bid in self.game.bids)
        self.counter_of_past_bidders += 1  # Increment counter
        #role = game.get_player_role(player_num)
        #print(f"{role} (Player {player_num + 1}), your hand:")
        #for card in self.game.players[player_num]:
            #print(card)
        num_players = self.game.num_players

        # Joe (Conservative Player)
        if player_num == self.game.conservative_player_index or player_num == self.game.BOB_player_index or player_num == self.game.ALICE_player_index:
            if self.counter_of_past_bidders != num_players:
                bid = 0
                #print('I bid whatever I want, I am not last!', self.counter_of_past_bidders)
            else:
                required_bid = self.game.current_deck_size - total_bid_sum
                if required_bid == 0:
                    bid = 1
                    #print(f'I bid {bid} to meet the condition, my bet is:', bid)
                else:
                    bid = 0
                    #print('Bidding zero works')
        
        # # Bob (Liberal Player)
        # elif player_num == self.game.BOB_player_index:
        #     if self.counter_of_past_bidders == num_players:
        #         bid = self.game.current_deck_size - total_bid_sum
        #         #print(f'Bob bids {bid} as the last player')
        #     else:
        #         bid = self.game.current_deck_size
            
            
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

        #-----------------------------------------------------------
            # #make ALice play like a AI-player  --> THIS IS WIP
            # self.alice_ai_agent.sum_bids = total_bid_sum
            # self.alice_ai_agent.position_bidders = self.counter_of_past_bidders
        
            # #update alice model based on the reward received in previous round 
            # self.ai_agent.update(self.ai_agent.bid_model, self.state_alice_old, self.ALICE_action, self.ALICE_reward, self.alice_states, self.done) 

            #TEST if states are updated correctly
            # print(f"HAND OF ALICE")
            # print(f"STATE OF ALICE {self.alice_states}")

            # bid = self.alice_ai_agent.make_bid() #in this function the forward pass happens --> we miss a input state
            # self.ALICE_action = bid
            
        # AI Player: here the AI agent makes the bid
        elif player_num == self.game.ai_player_index:
            self.ai_agent.sum_bids = total_bid_sum
            self.ai_agent.position_bidders = self.counter_of_past_bidders

            #Before bidding retrieve the states of the AI-player
            self.states = self.game.update_game_state(self.ai_player_index)
            self.ai_agent.update_agent_state(self.states) 

            bid = self.ai_agent.make_bid() #in this function the forward pass happens
            #print('The AI agent made its choice', bid)
            self.AI_bid = bid
        
            #UPDATE bidding model with previous round information and new state 
            self.ai_agent.update_bid_model(self.state_old, self.AI_action, self.AI_reward, self.states, self.done) #from previous round: state_old, AI_action, AI_reward #current: self.states
            
        # Handle unexpected player numbers
        else:
            #print('Unexpected player index:', player_num)
            raise Exception('Error: more than 4 players')

        #print(f"{role} (Player {player_num + 1}) bids: {bid}")
        self.game.bids[player_num] = bid

    #define winning card function
    def win_card(self, trick_cards, leading_suit):
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

    #assign points based on leadership position
    def assign_points(self, leadership_points, player_index):
        list_points  =  list(leadership_points.values())
        max_point = max(list_points)
        index_player = list_points.index(max_point)
        # print('Type of self.game.scores:', type(leadership_points))
        #print('Content of self.game.scores:', leadership_points)
        # print('leadership points values:', list(leadership_points.values()))
        #print('max points values:', max_point)
        # print('index of player', index_player)
        if index_player ==player_index and max_point>0:
            return 300
        elif index_player==player_index:
            return 50
        else:
            return 0

    def play_all_tricks(self): #play all tricks in the current round
        for _ in range(self.game.current_deck_size):
            self.play_trick()
            #print('-------------Trick Played-----------')

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        #for player in 4 players
        #print("START TRICK")
        for player_num in range(self.game.num_players):

            #PlAYER SELLECTION
            #print("PLAYER SELECTION")
            #print('Player_num:', player_num)
            current_player = (self.game.starting_player + player_num) % self.game.num_players
            #print('Current_player:', current_player)
            role = self.game.get_player_role(current_player)
            #print('Role', role)

            #Playing strategy for all the players INCLUDING the AI player
            if leading_suit: 
                leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                if leading_cards: # Play the highest card of the leading suit that you have
                    card = max(leading_cards, key=lambda x: x.custom_value)
                else: # No leading suit cards, check for atout cards
                    atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout.suit]
                    if atout_cards: # Play the highest atout card
                        card = max(atout_cards, key=lambda x: x.custom_value)
                    else: # Play the highest non-atout card
                        #print('TEST', self.game.players[current_player])
                        card = max(self.game.players[current_player], key=lambda x: x.custom_value)
            else: # No leading suit established, play the highest card available in your player' s deck
                card = max(self.game.players[current_player], key=lambda x: x.custom_value)

            ###Commented out until bidding is sorted out!!!
            # if current_player != self.ai_player_index:
            #     #Playing strategy for all the players excluding the AI player
            #     if leading_suit: 
            #         leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
            #         if leading_cards: # Play the highest card of the leading suit that you have
            #             card = max(leading_cards, key=lambda x: x.custom_value)
            #         else: # No leading suit cards, check for atout cards
            #             atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout.suit]
            #             if atout_cards: # Play the highest atout card
            #                 card = max(atout_cards, key=lambda x: x.custom_value)
            #             else: # Play the highest non-atout card
            #                 #print('TEST', self.game.players[current_player])
            #                 card = max(self.game.players[current_player], key=lambda x: x.custom_value)
            #     else: # No leading suit established, play the highest card available in your player' s deck
            #         card = max(self.game.players[current_player], key=lambda x: x.custom_value)

            # elif current_player == self.ai_player_index:
            #     print("HAND:", self.game.players[current_player])
            #     hand = self.game.players[current_player]
            #     #Information prior to make choice
            #     print(f"Info before AI makes choice trick_cards:{trick_cards}")
            #     #pli wons and predicted
            #     # print(f"\033[32mPli actual won {self.game.pli_scores[self.ai_player_index]}\033[0m")
            #     # print(f"\033[32mPli predicted: {self.game.bids[self.ai_player_index]}\033[0m")

            #     tricks_won  = self.game.pli_scores[self.ai_player_index]
            #     tricks_predicted  = self.game.bids[self.ai_player_index]
            #     #create the input tensor 
            #     input_tensor = self.game.process_hand(hand, trick_cards, tricks_won, tricks_predicted)
            #     print(f"\033[32minput_tensor is: {input_tensor}\033[0m")

            #     #before choosing a card we update the model
            #     if self.counter_played > 0: #do not update the first time 
            #         self.ai_agent.update(self.ai_agent.card_model, self.old_input_tensor, self.old_index_card_chosen, self.AI_reward, input_tensor, self.done) #state = input tensor, 
            #         #test of seeing if AI-reward can be shared
                
            #     if leading_suit:
            #         leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
            #         if leading_cards:
            #             LD_condition = suit_to_idx[leading_suit] #create a Leading card condition to meet a Rikiki Rules
            #             card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition)  ##this function will provide the card to play
            #         else:
            #             LD_condition = -1 #means there no condition
            #             card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition)
            #     else:
            #         LD_condition = -1 #means there no condition
            #         card, idx_card = self.ai_agent.ai_choose_card(input_tensor, LD_condition)

            #     ###add the end store the old input tensor
            #     self.counter_played += 1
            #     self.old_input_tensor = input_tensor
            #     self.old_index_card_chosen = idx_card

            # else:
            #     raise Exception("Error with current player of players") 

            ###COLLECT card played by the players 
            #append card to the trick
            trick_cards.append((card, role))
            #first player establishes leading suit
            if not leading_suit:
                leading_suit = card.suit
            #remove played card from player' s hand
            # print(f"Players hand {self.game.players[current_player]}")
            # print(f"Card player wants to remove out of hand: {card}")
            self.game.players[current_player].remove(card)
            #print what card was played by who
            #print(f"{role} (Player {current_player + 1}) plays: {card}") ##commented out for efficiency
        
        #DETERMINING a round winner::
        # print(f"These are the trick_cards:{trick_cards}")
        #determine who won the plit in this trick (a trick being 1-play of card in a round)
        #print('trick cards:', trick_cards)
        winning_card, winning_player_role = self.win_card(trick_cards, leading_suit)
        #print('winning card is:', winning_card, 'atout:', self.game.atout, 'leading suit:', leading_suit)
        #print('trick cards:', trick_cards)
        # winning_player = trick_cards.index(winning_card)
        winning_player_index = next(i for i, card_tuple in enumerate(trick_cards) if card_tuple == (winning_card, winning_player_role))
        #winner_role = self.game.get_player_role(winning_player)
        #print(f" {winning_player_role}  wins the trick with {winning_card}")   #COMMENTED OUT FOR EFFICEINCY

        # Update the starting player to the winning player for the next trick
        self.game.starting_player = (self.game.starting_player + winning_player_index) % self.game.num_players
        #print('Starting player next trick:', self.game.starting_player)
        #update the pli score of the winning player
        self.game.pli_scores[winning_player_index] += 1

    
    def play_round(self, round_number):
        self.game.current_deck_size = self.game.starting_deck_size + round_number ##added round number so from 0-8
        self.game.deal_cards(round_number)
        self.game.select_atout()

        #get the bids from the 4 players
        self.bidding_phase() 
        #play one full round
        self.play_all_tricks()  
        #calculate the scores after one full round
        self.game.calculate_scores()   #here the reward of the round is calculated

        #print the results
        #self.game.print_scores()
        #self.game.print_overview_table() ## for efficiency commented this out

        ###HERE WE STORE important variables for update after a full round 
        self.state_old = self.states
        self.AI_action = self.AI_bid  #is good
        self.AI_reward = self.game.rewards[3] #here we are gonna refer to a very small reward in function of how good it predicts
        
        #you need to store here the true bids
        self.AI_actual = self.game.pli_scores[3]

        #reset the cards for next round card distribution
        self.game.reset_for_next_round() #--> end of round 


    ##Helper function 
    def store_score(self, score, actions, true_actions):
        # Open the CSV file in append mode
        with open(CSV_FILE_PATH, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Write the score to the CSV file
            writer.writerow([score]+actions+ true_actions)

    # Train the model 
    def trainer(self):
        scores_after_each_game = []
        #reward_after_each_round = []
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
                #reward_after_each_round.append(self.AI_reward) #this creates a list of all rewards won during the round 
            
            #at the end of every game: --> store into csv file for efficiency
            self.store_score(self.game.scores[ai_player_index], current_game_actions, true_game_actions)
            current_game_actions.clear()
            true_game_actions.clear()

            #You can win a big reward at the end of game 
            # big_reward = self.assign_points(self.game.scores, 3)  ###removed BIG reward for analysis purposes
            # big_rewards_list.append(big_reward)
            # self.AI_reward += big_reward

            #check if alice wins a big reward this game
            # big_reward_alice = self.assign_points(self.game.scores, 2)
            # self.ALICE_reward += big_reward_alice
            n_games_played += 1

        ##WHILE LOOP FINSIHED
        print(f"----GAME {NUMBER_GAMES} PLAYED---, started with game zero")

        # Load scores from the CSV file
        scores_df = pd.read_csv(CSV_FILE_PATH, header=None)
        scores_after_each_game = scores_df[0].tolist()


        # Separate predicted bids (columns 1 to 8) and true bids (columns 9 to 16)
        predicted_bids = scores_df.iloc[:, 1:9]
        true_bids = scores_df.iloc[:, 9:]


       #SAVE THE MODEL
        self.ai_agent.save_model(self.ai_agent.bid_model, model_filename)

        # Creating a single figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # First subplot: Rewards over Time
        axs[0].plot(scores_after_each_game,color='green', label='AI Game Reward')
        # axs[0].plot(alice_acc_rewards, color='pink', label='ALICE Game Reward')
        axs[0].set_xlabel('Game')
        axs[0].set_ylabel('Total Reward')
        axs[0].set_title('Rewards over Time')
        axs[0].legend()
        axs[0].grid(True)

        #Game won vs game lost:
        won = sum(1 for x in scores_after_each_game if x > 0)
        lost = sum(1 for x in scores_after_each_game if x < 0)
        labels = ['Won', 'Lost']
        sizes = [won, lost]
        axs[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5252'])
        axs[1].set_title('Games Won vs. Lost')

        # Calculate the average predicted bids per round
        average_predicted_bids = predicted_bids.mean(axis=0)
        # Calculate the average true bids per round
        average_true_bids = true_bids.mean(axis=0)
        # Calculate the overall average of predicted bids
        overall_average_predicted = predicted_bids.values.mean()
        # Calculate the overall average of true bids
        overall_average_true = true_bids.values.mean()

        axs[2].plot(range(2, 10), average_predicted_bids, marker='o', label='Average Predicted Bids')
        axs[2].plot(range(2, 10), average_true_bids, marker='x', label='Average True Bids')
        axs[2].axhline(y=overall_average_predicted, color='darkblue', linestyle='--', label='Overall Average Predicted Bids')
        axs[2].axhline(y=overall_average_true, color='darkorange', linestyle='--', label='Overall Average True Bids')
        axs[2].set_xlabel('Round')
        axs[2].set_ylabel('Average Bid')
        axs[2].legend()
        axs[2].set_title('Average Predicted Bids vs Average True Bids per Round')

        # # Second subplot: Occurrences of Each Action
        # axs[1].bar(range(len(action_counts)), action_counts, color='blue', alpha=0.6, label='Predicted')
        # axs[1].bar(range(len(actual_counts)), actual_counts, color='red', alpha=0.6, label='True')
        # axs[1].set_xlabel('Action')
        # axs[1].set_ylabel('Occurrences')
        # axs[1].set_title('Occurrences of Each Action')
        # axs[1].set_xticks(range(len(action_counts)))
        # axs[1].set_xticklabels([f'Action {i}' for i in range(len(action_counts))])
        # axs[1].grid(axis='y')
        # # Add a legend
        # axs[1].legend()

        # # # Third subplot: Loss Curve
        # # axs[2].plot(self.ai_agent.losses_card, label='Training Loss')
        # # axs[2].set_xlabel('Training Steps')
        # # axs[2].set_ylabel('Loss')
        # # axs[2].set_title('Loss Curve')
        # # axs[2].legend()
        # # axs[2].grid(True)

        # Adjust layout to prevent overlap
        # plt.tight_layout()
        # plt.show()
        # Save the plot to a file
        plt.savefig(filename)

        # Close the figure to avoid display in interactive environments
        plt.close()

        print(f"Plot saved as {filename}")


if __name__ == "__main__":
    #global parameters
    NUMBER_GAMES = 300; TOTAL_ROUNDS=8 
    # Create the 'outputs' folders
    image_output_dir = 'image_outputs'
    os.makedirs(image_output_dir, exist_ok=True)
    csv_output_dir = 'csv_outputs'
    os.makedirs(csv_output_dir, exist_ok=True)
    bid_model_output_dir = 'bid_model_outputs'
    os.makedirs(bid_model_output_dir, exist_ok=True)

    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{image_output_dir}/plot_{timestamp}.png'
    CSV_FILE_PATH = f'{csv_output_dir}/scores{timestamp}.csv'
    action_FILE_PATH = f'{csv_output_dir}/scores{timestamp}.csv'
    model_filename = f'{bid_model_output_dir}/model{timestamp}_{NUMBER_GAMES}.pth'
    
    #Model hyperparameters 
    LR = 0.1 #0.010 
    #game parameters 
    num_players = 4  # Adjust as needed
    #Starting Player's order
    conservative_player_index = 0  # The JOE first player is the conservative player
    BOB_player_index = 1 #Bob is second to play
    ALICE_player_index = 2 # ALice is third to play
    ai_player_index = 3  # The last player is the AI agent

    #we start with 2 cards in the hand
    starting_deck_size = 2

    # Create game instance: initialises all attributes
    game = RikikiGame(num_players, ai_player_index, conservative_player_index, BOB_player_index, ALICE_player_index, starting_deck_size) #only play rounds of 3 cards
    # #create an ai_agent instance of class AIAgent
    ai_agent = AIAgent(LR, starting_deck_size, TOTAL_ROUNDS)

    #ALICE ai_agent instance 
    alice_ai_agent = AIAgent(LR, starting_deck_size, TOTAL_ROUNDS)

    #start one game
    trainer = Training(game, ai_agent, alice_ai_agent, ai_player_index, ALICE_player_index)
    trainer.trainer()






