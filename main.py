from rikiki_game_AI import RikikiGame
from ai_agent import AIAgent
import random


class Training:
    def __init__(self, game, ai_agent):
        self.game = game
        self.ai_agent = ai_agent
        self.states = None

    def bidding_phase(self, round_number):
        starting_player = (self.game.starting_player + 1) % self.game.num_players
        for i in range(self.game.num_players):
            current_player = (starting_player + i) % self.game.num_players
            self.get_bid(current_player)
            #update states for AI-player
            self.states = self.game.update_game_state()
            self.ai_agent.update_agent_state(self.states)

    def get_bid(self, player_num):
        role = game.get_player_role(player_num)
        print(f"{role} (Player {player_num + 1}), your hand:")
        for card in self.game.players[player_num]:
            print(card)
        if player_num == self.game.conservative_player_index: #bid 0 for the conservative player
            bid = 0
        elif player_num == self.game.ai_player_index:
            bid = self.ai_agent.make_bid() #the AI player needs to make a bid
            print('the AI agent made its choice', bid)
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
    
    

    def play_all_tricks(self):
        for _ in range(self.game.current_deck_size):
            self.play_trick()

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        for player_num in range(self.game.num_players):
            current_player = (self.game.starting_player + player_num) % self.game.num_players
            role = self.game.get_player_role(current_player)
            if current_player == self.game.ai_player_index:
                #here the ai needs to play a wise card
                card = self.ai_agent.play_card(self.game.players[current_player], leading_suit, self.game.atout)
                self.game.players[current_player].get(str(card))
            else:
                if leading_suit:
                    valid_cards = [card for card in self.game.players[current_player] if card.suit == leading_suit]
                    if valid_cards:
                        card = random.choice(valid_cards)
                    else:
                        card = random.choice(self.game.players[current_player])
                else:
                    card = random.choice(self.game.players[current_player])
                self.game.players[current_player].get(str(card))

            trick_cards.append((card, current_player))
            if leading_suit is None:
                leading_suit = card.suit

            print(f"{role} (Player {current_player + 1}) plays: {card}")

        winning_card, winning_player = self.game.determine_winner(trick_cards, leading_suit)
        winning_role = self.game.get_player_role(winning_player)
        print(f"{winning_role} (Player {winning_player + 1}) wins the trick with {winning_card}")
        self.game.pli_scores[winning_player] += 1


    # One-full game training loop
    def one_game(self):
        for round_number in range(11):
            self.game.current_deck_size = self.game.starting_deck_size + round_number
            self.game.deal_cards(round_number)
            self.game.select_atout()
            self.states = self.game.update_game_state()
            print(self.states)
            self.bidding_phase(round_number)
            self.play_all_tricks()
            self.game.calculate_scores()
            self.game.print_scores()
            self.game.print_overview_table()
            self.ai_agent.update_agent_state(self.states)
            self.game.reset_for_next_round()




if __name__ == "__main__":
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
    trainer.one_game()






