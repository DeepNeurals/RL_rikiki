#This is a minimum version 

from rikiki_game_AI import RikikiGame
from ai_agent import AIAgent
import random
import torch
import matplotlib.pyplot as plt
import numpy as np



class Training:
    def __init__(self, game, ai_agent):
        self.game = game
        self.ai_agent = ai_agent
        self.states = None
        #for learning initialise these
        self.state_old = torch.zeros(1,8)
        self.AI_action = 0
        self.AI_bid = 0 
        self.AI_reward  = 0
        self.done = 0
        self.counter_of_past_bidders = 0 #initialise with zero

    def bidding_phase(self):
        starting_player = (self.game.starting_player + 1) % self.game.num_players #wrap the 4 players from 0-3. 
        self.states = self.game.update_game_state()
        self.ai_agent.update_agent_state(self.states)
        for i in range(self.game.num_players):
            current_player = (starting_player + i) % self.game.num_players
            self.get_bid(current_player)
        #reinistialise the counter of bidders for next round
        self.counter_of_past_bidders = 0


    def get_bid(self, player_num):
        total_bid_sum = sum(bid if bid is not None else 0 for bid in self.game.bids)
        self.counter_of_past_bidders += 1  # Increment counter

        role = game.get_player_role(player_num)
        #print(f"{role} (Player {player_num + 1}), your hand:")
        #for card in self.game.players[player_num]:
            #print(card)
        
        num_players = self.game.num_players
        #last_player_index = num_players - 1

        # Joe (Conservative Player)
        if player_num == self.game.conservative_player_index:
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
        
        # Bob (Liberal Player)
        elif player_num == self.game.BOB_player_index:
            if self.counter_of_past_bidders == num_players:
                bid = self.game.current_deck_size - total_bid_sum
                print(f'Bob bids {bid} as the last player')
            else:
                bid = self.game.current_deck_size

        # Alice (Random Bidder)
        elif player_num == self.game.ALICE_player_index:
            if self.counter_of_past_bidders != num_players:
                bid = random.randint(0, self.game.current_deck_size)
            else:
                while True:
                    bid = random.randint(0, self.game.current_deck_size)
                    if bid + total_bid_sum != self.game.current_deck_size:
                        break
            self.states = self.game.update_game_state()
            self.ai_agent.update_agent_state(self.states)

        # AI Player: here the AI agent makes the bid
        elif player_num == self.game.ai_player_index:
            self.ai_agent.sum_bids = total_bid_sum
            self.ai_agent.position_bidders = self.counter_of_past_bidders
            #we update the model with previous round information
            self.ai_agent.update(self.state_old, self.AI_action, self.AI_reward, self.states, self.done) #from previous round: state_old, AI_action, AI_reward #current: self.states
            bid = self.ai_agent.make_bid() #in this function the forward pass happens
            print('The AI agent made its choice', bid)
            self.AI_bid = bid
        
        # Handle unexpected player numbers
        else:
            print('Unexpected player index:', player_num)
            raise Exception('Error: more than 4 players')

        #print(f"{role} (Player {player_num + 1}) bids: {bid}")
        self.game.bids[player_num] = bid

    
    # def win_card(self, trick_cards, leading_suit):
    #     #what is the highest atout
    #     atout_cards = [card for card in trick_cards if card.suit == self.game.atout.suit]
    #     leading_cards = [card for card in trick_cards if card.suit == leading_suit]
    #     if atout_cards:
    #         #print('some atout card were played')
    #         winning_card = max(atout_cards, key=lambda x: x.custom_value)
    #     else:
    #         #print('highest leading suit wins')
    #         winning_card = max(leading_cards, key=lambda x: x.custom_value)
    #     return winning_card
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



    def play_all_tricks(self): #play all tricks in the current round
        for _ in range(self.game.current_deck_size):
            self.play_trick()
            print('-------------Trick Played-----------')

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        #for player in 4 players
        for player_num in range(self.game.num_players):
            #print('player_num:', player_num)
            current_player = (self.game.starting_player + player_num) % self.game.num_players
            #print('current_player:', current_player)
            role = self.game.get_player_role(current_player)

            #Playing strategy for all the players incuding the AI player
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

            #append card to the trick
            trick_cards.append((card, role))
            #first player establishes leading suit
            if not leading_suit:
                leading_suit = card.suit
            #remove played card from player' s hand
            self.game.players[current_player].remove(card)
            #print what card was played by who
            print(f"{role} (Player {current_player + 1}) plays: {card}")

        #determine who won the plit in this trick (a trick being 1-play of card in a round)
        #print('trick cards:', trick_cards)
        winning_card, winning_player_role = self.win_card(trick_cards, leading_suit)
        #print('winning card is:', winning_card, 'atout:', self.game.atout, 'leading suit:', leading_suit)
        #print('trick cards:', trick_cards)
        # winning_player = trick_cards.index(winning_card)
        winning_player_index = next(i for i, card_tuple in enumerate(trick_cards) if card_tuple == (winning_card, winning_player_role))
        #winner_role = self.game.get_player_role(winning_player)
        print(f" {winning_player_role}  wins the trick with {winning_card}")   #==> need to add here that the next to play the trick is the winner

        # Update the starting player to the winning player for the next trick
        self.game.starting_player = (self.game.starting_player + winning_player_index) % self.game.num_players

        #update the pli score of the winning player
        self.game.pli_scores[winning_player_index] += 1


    # Train One-full episode
    def trainer(self):
        #initiliasation of counters
        n_games = 0 
        accumulated_rewards = []
        list_of_actions = []
        total_reward = 0

        while n_games<NUMBER_GAMES: 

            #we start here with no information about the state

            #during the bidding phase we get info about the state
            #we derive an action

            #we derive the next_state at the next bidding phase

            for round_number in range(TOTAL_ROUNDS-2): #the minus 2 is there such that we play 5 games of only 1 round
                #print('round_number:', round_number)
                #define the actual game round params
                #self.game.current_deck_size = self.game.starting_deck_size + round_number
                #only fortotal_reward minimum version -- IMPORTANT CHANGE ---
                self.game.current_deck_size = self.game.starting_deck_size
                self.game.deal_cards(round_number)
                self.game.select_atout()
                #Update the atout information and player' s card to the state// in case AI bid first
                #self.states = self.game.update_game_state() #not necessary since done when starting bidding phase

                #get the bids from the 4 players
                self.bidding_phase()  #in between one of the players the AI makes it bid, model is updated here
                #play one full round
                self.play_all_tricks()  
                #calculate the scores after one full round
                self.game.calculate_scores()   #after this moment we know the reward

                #print the results
                #self.game.print_scores()
                self.game.print_overview_table()

                #store for next round optimisation the state of this round
                self.state_old = self.states
                #print('print the rewards for the 4 players:', self.game.rewards[3])
                self.AI_action = self.AI_bid
                self.AI_reward = self.game.rewards[3]

                #reset the game cards for next round
                self.game.reset_for_next_round()

                #We need to feed (state, action, reward, next_state) to the update function
                self.ai_agent.update_agent_state(self.states)

                total_reward += self.AI_reward

                print('--------ROUND PLAYED------------')

            # Save the model every 10 episodes (or at your desired interval)
            #if NUMBER_GAMES % 10 == 0:
                #self.ai_agent.save_model(self.ai_agent.model, f'model_checkpoint_episode_{NUMBER_GAMES}.pth')
            
            accumulated_rewards.append(total_reward)
            list_of_actions.append(self.AI_action)
            n_games += 1
            print('n_games played:', n_games)
        print(f"This many episodes were played {NUMBER_GAMES}")
        #print(list_of_actions)


       # Counting the occurrences of each action
        action_counts = np.bincount(list_of_actions)


        # Creating a single figure with three subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # First subplot: Rewards over Time
        axs[0].plot(accumulated_rewards, label='Episode Reward')
        axs[0].set_xlabel('Episode')
        axs[0].set_ylabel('Total Reward')
        axs[0].set_title('Rewards over Time')
        axs[0].legend()
        axs[0].grid(True)

        # Second subplot: Occurrences of Each Action
        axs[1].bar(range(len(action_counts)), action_counts, color='blue')
        axs[1].set_xlabel('Action')
        axs[1].set_ylabel('Occurrences')
        axs[1].set_title('Occurrences of Each Action')
        axs[1].set_xticks(range(len(action_counts)))
        axs[1].set_xticklabels([f'Action {i}' for i in range(len(action_counts))])
        axs[1].grid(axis='y')

        # Third subplot: Loss Curve
        axs[2].plot(self.ai_agent.losses, label='Training Loss')
        axs[2].set_xlabel('Training Steps')
        axs[2].set_ylabel('Loss')
        axs[2].set_title('Loss Curve')
        axs[2].legend()
        axs[2].grid(True)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    #hyperparameters 
    LR = 0.010 #0.010 
    #global parameters
    NUMBER_GAMES = 5000; TOTAL_ROUNDS=3 #actually here the number of games are equal to number of rounds since a game has 1 round. Total rounds= number of ticks per round
    #game parameters 
    num_players = 4  # Adjust as needed
    #Starting Player's order
    conservative_player_index = 0  # The first player is the conservative player
    BOB_player_index = 1 #Bob is second to play
    ALICE_player_index = 2 # ALice is third to play
    ai_player_index = 3  # The last player is the AI agent
    # Create game instance: initialises all attributes
    game = RikikiGame(num_players, ai_player_index, conservative_player_index, BOB_player_index, ALICE_player_index, starting_deck_size=3) #only play rounds of 3 cards
    # #create an ai_agent instance of class AIAgent
    ai_agent = AIAgent(ai_player_index, num_players, LR)
    #start one game
    trainer = Training(game, ai_agent)
    trainer.trainer()






