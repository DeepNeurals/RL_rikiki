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
        self.AI_bid = 0 
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

            # #update states for AI-player  ###THIS happens now always just after ALice did bid
            # self.states = self.game.update_game_state()
            # self.ai_agent.update_agent_state(self.states) 

    def get_bid(self, player_num):
        print("TESTING", self.game.bids)
        total_bid_sum = sum(bid if bid is not None else 0 for bid in self.game.bids)
        self.counter_of_past_bidders += 1 #a new person is bidding 

        #print the role and hand of the current player
        role = game.get_player_role(player_num)
        print(f"{role} (Player {player_num + 1}), your hand:")
        for card in self.game.players[player_num]:
            print(card)
        
        #JOE Conservative player
        if player_num == self.game.conservative_player_index: #bid 0 for the conservative player
            if self.counter_of_past_bidders != 4: #if you are not the last person bid zero
                bid = 0
            else: # if you are the last one to bid (counter =4)
                if total_bid_sum == self.game.current_deck_size: #otherwhise bidding zero does not met condition
                    bid = 1
                else: #you can bid zero
                    bid = 0
        
        #FOR ALICE random bidder (always bids just before AI player)
        if player_num == self.game.ALICE_player_index: #bid 0 for the conservative player
            if self.counter_of_past_bidders != 4: #if ALICE is not the last person bid something random
                bid = random.randint(0, self.game.current_deck_size)
            else:
                while True:
                    bid = random.randint(0, self.game.current_deck_size)
                    if bid + total_bid_sum != self.game.current_deck_size:
                        break #you made a valide bid
                    else:
                        continue #re-evaluate with another bid
            #update states for AI-player
            self.states = self.game.update_game_state()
            self.ai_agent.update_agent_state(self.states) 

        #the AI_player
        elif player_num == self.game.ai_player_index:
            # send current bid sum + position in line of bid
            self.ai_agent.sum_bids = total_bid_sum
            self.ai_agent.position_bidders = self.counter_of_past_bidders

            bid = self.ai_agent.make_bid() #goes check AI code to make bid decision (this code also checks to meet the condition) 
            print('the AI agent made its choice', bid)
            self.AI_bid = bid 
        
        #For BOB
        else: 
            while True:
                bid = random.randint(0, self.game.current_deck_size)
                if player_num == self.game.num_players - 1:  # Last player
                    if sum(bid for bid in self.game.bids if bid is not None) + bid != self.game.current_deck_size:
                        break
                    else:
                        continue
                else:
                    break
        print(f"{role} (Player {player_num + 1}) bids: {bid}")
        self.game.bids[player_num] = bid


    def play_all_tricks(self): #play all tricks in the current round
        for _ in range(self.game.current_deck_size):
            self.play_trick()
            print('-------------Trick Played-----------')

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        for player_num in range(self.game.num_players):
            current_player = (self.game.starting_player + player_num) % self.game.num_players
            role = self.game.get_player_role(current_player)
            # if current_player == self.game.ai_player_index:
            #     #here the ai needs to play a wise card
            #     card = self.ai_agent.play_card(self.game.players[current_player], leading_suit, self.game.atout)
            #     self.game.players[current_player].get(str(card))
            # else:  ###UNCOMMENT THESE LINES AND SHIFT them for AI-agent PLAY strategy development

            #Playing strategy for all the players incuding the AI player
            if leading_suit: 
                leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                if leading_cards: # Play the highest card of the leading suit that you have
                    card = max(leading_cards, key=lambda x: x.custom_value)
                else: # No leading suit cards, check for atout cards
                    atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout]
                    if atout_cards: # Play the highest atout card
                        card = max(atout_cards, key=lambda x: x.custom_value)
                    else: # Play the highest non-atout card
                        #print('TEST', self.game.players[current_player])
                        card = max(self.game.players[current_player], key=lambda x: x.custom_value)
            else: # No leading suit established, play the highest card available in your player' s deck
                card = max(self.game.players[current_player], key=lambda x: x.custom_value)

            #append card to the trick
            trick_cards.append(card)
            #first player establishes leading suit
            if not leading_suit:
                leading_suit = card.suit
            #remove played card from player' s hand
            self.game.players[current_player].remove(card)
            #print what card was played by who
            print(f"{role} (Player {current_player + 1}) plays: {card}")

        #determine who won the plit in this trick (a trick being 1-play of card in a round)
        winning_card = max(trick_cards, key=lambda x: x.custom_value)
        winning_player = trick_cards.index(winning_card)
        print(f"Player {winning_player} wins the trick with {winning_card}")   #==> need to add here that the next to play the trick is the winner

        #update the pli score of the winning player
        self.game.pli_scores[winning_player] += 1


    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    # Train One-full 11-round game loop
    def trainer(self):
        n_games = 0 
        plot_correct_bids = []
        plot_pred_bids = []
        total_score = 0
        record = 0
        self.old_state = torch.zeros((1, 8)) #initialise the state vector with only zero's
        while n_games<NUMBER_GAMES:
            #for every round in a game
            for round_number in range(TOTAL_ROUNDS-1):
                #define the actual game round params
                self.game.current_deck_size = self.game.starting_deck_size + round_number
                self.game.deal_cards(round_number)
                self.game.select_atout()
                #Update the atout information and player' s card to the state
                self.states = self.game.update_game_state()
                #state_old = self.old_state

                #get the bids from the 4 players
                self.bidding_phase()
                #play one full round
                self.play_all_tricks()
                #calculate the scores after one full round
                self.game.calculate_scores()

                #print the results
                self.game.print_scores()
                self.game.print_overview_table()
                state_new = self.states
                print(self.states)

                #reward for AI based on score_AI 
                score_AI = self.game.scores[self.game.ai_player_index] 
                # Determine if the AI has the highest score
                max_score = max(self.game.scores)  # Find the highest score among all players
                # Initialize reward
                reward = 0
                # Check if the AI has the highest score
                if score_AI == max_score:
                    reward = 1  # Example reward value; adjust as needed
                else:
                    reward = 0  # No reward if the AI does not have the highest score
                print('reward for this round:', reward)

                #reset the game cards for next round
                self.game.reset_for_next_round()

                # train short memory
                self.ai_agent.train_short_memory(self.state_old, self.AI_bid, reward, state_new, self.game.current_deck_size)
                self.ai_agent.update_agent_state(self.states)
                #next state becomes old state
                self.old_state = state_new
                self.ai_agent.remember(self.state_old, self.AI_bid, reward, state_new, self.game.current_deck_size)
                plot_pred_bids.append(self.AI_bid)
                #plot_correct_bids.append(true_bid)
                print('--------ROUND PLAYED------------')

            n_games += 1
            print('n_games played:', n_games)
            print('AI-score:', score_AI)

        # # Create a figure and axis object
        # fig, ax = plt.subplots()
        # # Plot the predicted bids
        # ax.plot(plot_pred_bids, label='Predicted Bids', color='blue')
        # # Plot the correct bids
        # ax.plot(plot_correct_bids, label='Correct Bids', color='red')
        # # Calculate the errors
        # errors = np.abs(np.array(plot_pred_bids) - np.array(plot_correct_bids))
        # # Plot the errors
        # ax.plot(errors, label='Errors', color='green')
        # # Add labels and title
        # ax.set_xlabel('Sample Index')
        # ax.set_ylabel('Bid Value')
        # ax.set_title('Predicted vs Correct Bids')
        # # Add a legend
        # ax.legend()
        # # Show the plot
        # plt.show()


if __name__ == "__main__":
    #hyperparameters 
    MAX_MEMORY = 100_000; BATCH_SIZE = 1000; LR = 0.001
    #global parameters
    NUMBER_GAMES = 1; TOTAL_ROUNDS=3
    #game parameters 
    num_players = 4  # Adjust as needed
    conservative_player_index = 0  # The first player is the conservative player
    BOB_player_index = 1 #Bob is second to play
    ALICE_player_index = 2 # ALice is third to play
    ai_player_index = 3  # The last player is the AI agent
    conservative_player_index = 0  # The first player is the conservative player
    # Create game instance: initialises all attributes
    game = RikikiGame(num_players, ai_player_index, conservative_player_index, BOB_player_index, ALICE_player_index, starting_deck_size=2)
    # #create an ai_agent instance of class AIAgent
    ai_agent = AIAgent(ai_player_index, num_players)
    #start one game
    trainer = Training(game, ai_agent)
    trainer.trainer()






