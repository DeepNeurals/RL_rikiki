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

    def bidding_phase(self, round_number):
        starting_player = (self.game.starting_player + 1) % self.game.num_players #wrap the 4 players from 0-3. 
        self.states = self.game.update_game_state()
        self.ai_agent.update_agent_state(self.states)
        for i in range(self.game.num_players):
            current_player = (starting_player + i) % self.game.num_players
            self.get_bid(current_player)
            #update states for AI-player
            self.states = self.game.update_game_state()
            self.ai_agent.update_agent_state(self.states) #removed the deck size since already included in states

    def get_bid(self, player_num):
        #print the role and hand of the current player
        role = game.get_player_role(player_num)
        print(f"{role} (Player {player_num + 1}), your hand:")
        for card in self.game.players[player_num]:
            print(card)
        

        
        #conservative player
        if player_num == self.game.conservative_player_index: #bid 0 for the conservative player
            bid = 0
        
        #the AI_player
        elif player_num == self.game.ai_player_index:
            bid = self.ai_agent.make_bid() #the AI player needs to make a bid 
            print('the AI agent made its choice', bid)
            self.AI_bid = bid 
        #the random bidders
        else: #for the two other players bid a random number, but respect the rule that the sum of all bid can never be the exact same number of cards in the hand
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
            # else:  ###UNCOMMENT THESE LINES AND SHIFT them for AI-agent play strategy development

            #Playing strategy for all the players incuding the AI player
            if leading_suit: #if there is a leading suit, you need to play a card of that suit if you have one 
                leading_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                if leading_cards:
                    # Play the highest card of the leading suit that you have
                    card = max(leading_cards, key=lambda x: x.value) 
                else:
                # No leading suit cards, check for atout cards
                    atout_cards = [card for card in self.game.players[current_player] if card.suit == self.game.atout]
                    if atout_cards:
                    # Play the highest atout card
                        card = max(atout_cards, key=lambda x: x.value)
                    else:
                    # No leading suit or atout cards, play the lowest non-atout card
                        # for card in self.game.players[current_player]:
                        #     print('Attributes and methods of card:', dir(card))
                        #     print('Attributes of card:', vars(card))

                        # non_atout_cards = [card for card in self.game.players[current_player] if card.suit != self.game.atout]
                        # print('non_atout_cards', non_atout_cards)
                        # Filter out non-atout cards
                        atout_suit = self.game.atout
                        non_atout_cards = [card for card in self.game.players[current_player] if card.suit != atout_suit]
                        print('Non-atout cards:', [card.name for card in non_atout_cards])


                        if non_atout_cards:
                            card = min(non_atout_cards, key=lambda x: x.value)
                        
                        else: #prevent edge-cases
                        # Fallback if no non-atout cards are available (in case all cards are atout or no atout is defined)
                            card = min(self.game.players[current_player], key=lambda x: x.value)
            else:
                # No leading suit established, play the highest card available in your player' s deck
                card = max(self.game.players[current_player], key=lambda x: x.value)
            
            #remove the card played from the players hand
            self.game.players[current_player].get(str(card))
            #save the cards played by each player
            trick_cards.append((card, current_player))
            #initialise the leading suit after the first player played his card
            if leading_suit is None:
                leading_suit = card.suit
            print(f"{role} (Player {current_player + 1}) plays: {card}")

        #determine who won the pli in this trick (a trick being 1-play of card in a round)
        winning_card, winning_player = self.game.determine_winner(trick_cards, leading_suit) 
        winning_role = self.game.get_player_role(winning_player)
        print(f"{winning_role} (Player {winning_player + 1}) wins the trick with {winning_card}")
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
        self.old_state = torch.ones((1, 8))
        while n_games<NUMBER_GAMES:
            #for every round in a game
            for round_number in range(TOTAL_ROUNDS-1):
                #define the actual game round params
                self.game.current_deck_size = self.game.starting_deck_size + round_number
                self.game.deal_cards(round_number)
                self.game.select_atout()
                #get the old state
                self.states = self.game.update_game_state()
                state_old = self.old_state
                #get the final_bid, in the bidding phase the state is updated regularly so the nn can be learned
                self.bidding_phase(round_number)
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

                #reset the game for next round
                self.game.reset_for_next_round()

                # train short memory
                self.ai_agent.train_short_memory(state_old, self.AI_bid, reward, state_new, self.game.current_deck_size)
                self.ai_agent.update_agent_state(self.states, self.game.current_deck_size)
                #next state becomes old state
                self.old_state = state_new
                self.ai_agent.remember(state_old, self.AI_bid, reward, state_new, self.game.current_deck_size)
                plot_pred_bids.append(self.AI_bid)
                plot_correct_bids.append(true_bid)

            n_games += 1
            print('n_games played:', n_games)
            print('AI-score:', score_AI)

        # Create a figure and axis object
        fig, ax = plt.subplots()
        # Plot the predicted bids
        ax.plot(plot_pred_bids, label='Predicted Bids', color='blue')
        # Plot the correct bids
        ax.plot(plot_correct_bids, label='Correct Bids', color='red')
        # Calculate the errors
        errors = np.abs(np.array(plot_pred_bids) - np.array(plot_correct_bids))
        # Plot the errors
        ax.plot(errors, label='Errors', color='green')
        # Add labels and title
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Bid Value')
        ax.set_title('Predicted vs Correct Bids')
        # Add a legend
        ax.legend()
        # Show the plot
        plt.show()




if __name__ == "__main__":
    #hyperparameters 
    MAX_MEMORY = 100_000; BATCH_SIZE = 1000; LR = 0.001
    #global parameters
    NUMBER_GAMES = 10; TOTAL_ROUNDS=12
    #game parameters 
    num_players = 4  # Adjust as needed
    ai_player_index = 3  # The last player is the AI agent
    conservative_player_index = 0  # The first player is the conservative player
    # Create game instance: initialises all attributes
    game = RikikiGame(num_players, ai_player_index, conservative_player_index)
    # #create an ai_agent instance of class AIAgent
    ai_agent = AIAgent(ai_player_index, num_players)
    #start one game
    trainer = Training(game, ai_agent)
    trainer.trainer()






