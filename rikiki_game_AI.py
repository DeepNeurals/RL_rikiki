import pydealer
from collections import defaultdict
from tabulate import tabulate
import random
from ai_agent import AIAgent

class RikikiGame:
    def __init__(self, num_players, ai_player_index, conservative_player_index=0, starting_deck_size=2):
        self.num_players = num_players
        self.ai_player_index = ai_player_index
        self.conservative_player_index = conservative_player_index
        self.starting_deck_size = starting_deck_size
        self.current_deck_size = starting_deck_size
        self.deck = pydealer.Deck()
        self.players = []
        self.bids = []
        self.atout = None
        self.trick = pydealer.Stack()
        self.scores = defaultdict(int)
        self.pli_scores = defaultdict(int)
        # self.ai_agent = AIAgent(ai_player_index)
        self.ai_agent = AIAgent(ai_player_index, state_size=state_size, action_size=action_size)
        self.starting_player = 0

    def start_game(self):
        for round_number in range(11):
            self.current_deck_size = self.starting_deck_size + round_number
            self.deal_cards(round_number)
            self.select_atout()
            self.bidding_phase(round_number)
            self.play_all_tricks()
            self.calculate_scores()
            self.print_scores()
            self.print_overview_table()
            self.reset_for_next_round()

    def deal_cards(self, round_number):
        self.deck.shuffle()
        self.players = [self.deck.deal(self.current_deck_size) for _ in range(self.num_players)]
        self.bids = [None] * self.num_players
        self.pli_scores = defaultdict(int)
        self.starting_player = round_number % self.num_players

    def select_atout(self):
        self.atout = self.deck.deal(1)[0]
        print(f"Atout card: {self.atout}")

    def bidding_phase(self, round_number):
        starting_player = (self.starting_player + 1) % self.num_players
        for i in range(self.num_players):
            current_player = (starting_player + i) % self.num_players
            self.get_bid(current_player)

    def get_bid(self, player_num):
        role = self.get_player_role(player_num)
        print(f"{role} (Player {player_num + 1}), your hand:")
        for card in self.players[player_num]:
            print(card)
        if player_num == self.conservative_player_index:
            bid = 0
        elif player_num == self.ai_player_index:
            bid = self.ai_agent.make_bid(self.players[player_num], self.bids, self.current_deck_size, self.atout, self.num_players)
        else:
            while True:
                bid = random.randint(0, self.current_deck_size)
                if player_num == self.num_players - 1:  # Last player
                    if sum(bid for bid in self.bids if bid is not None) + bid != self.current_deck_size:
                        break
                    else:
                        continue
                else:
                    break
        print(f"{role} (Player {player_num + 1}) bids: {bid}")
        self.bids[player_num] = bid

    def play_all_tricks(self):
        for _ in range(self.current_deck_size):
            self.play_trick()

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        for player_num in range(self.num_players):
            current_player = (self.starting_player + player_num) % self.num_players
            role = self.get_player_role(current_player)
            if current_player == self.ai_player_index:
                card = self.ai_agent.play_card(self.players[current_player], leading_suit, self.atout)
                self.players[current_player].get(str(card))
            else:
                if leading_suit:
                    valid_cards = [card for card in self.players[current_player] if card.suit == leading_suit]
                    if valid_cards:
                        card = random.choice(valid_cards)
                    else:
                        card = random.choice(self.players[current_player])
                else:
                    card = random.choice(self.players[current_player])
                self.players[current_player].get(str(card))

            trick_cards.append((card, current_player))
            if leading_suit is None:
                leading_suit = card.suit

            print(f"{role} (Player {current_player + 1}) plays: {card}")

        winning_card, winning_player = self.determine_winner(trick_cards, leading_suit)
        winning_role = self.get_player_role(winning_player)
        print(f"{winning_role} (Player {winning_player + 1}) wins the trick with {winning_card}")
        self.pli_scores[winning_player] += 1

    def determine_winner(self, trick_cards, leading_suit):
        winning_card = None
        winning_player = None

        for card, player_num in trick_cards:
            if winning_card is None:
                winning_card = card
                winning_player = player_num
            else:
                if card.suit == self.atout.suit:
                    if winning_card.suit != self.atout.suit or card > winning_card:
                        winning_card = card
                        winning_player = player_num
                elif card.suit == leading_suit:
                    if winning_card.suit != self.atout.suit and card > winning_card:
                        winning_card = card
                        winning_player = player_num

        return winning_card, winning_player

    def calculate_scores(self):
        for player_num in range(self.num_players):
            predicted = self.bids[player_num]
            actual = self.pli_scores[player_num]
            if predicted == actual:
                self.scores[player_num] += 5 + actual
            else:
                self.scores[player_num] -= abs(predicted - actual)

    def print_scores(self):
        for player_num in range(self.num_players):
            role = self.get_player_role(player_num)
            print(f"{role} (Player {player_num + 1}) score: {self.scores[player_num]}")

    def print_overview_table(self):
        table_data = []
        for player_num in range(self.num_players):
            row = [
                f"{self.get_player_role(player_num)} (Player {player_num + 1})",
                self.bids[player_num],
                self.pli_scores[player_num],
                self.scores[player_num]
            ]
            table_data.append(row)
        headers = ["Player", "Bid", "Actual Tricks", "Score"]
        print("\nRound Overview:")
        print(tabulate(table_data, headers, tablefmt="grid"))

    def reset_for_next_round(self):
        self.deck = pydealer.Deck()
        self.trick = pydealer.Stack()

    def get_player_role(self, player_num):
        if player_num == self.conservative_player_index:
            return "Conservative Player"
        elif player_num == self.ai_player_index:
            return "AI Player"
        else:
            return "Random Player"

if __name__ == "__main__":
    num_players = 4  # Adjust as needed
    ai_player_index = 3  # The last player is the AI agent
    conservative_player_index = 0  # The first player is the conservative player
    game = RikikiGame(num_players, ai_player_index, conservative_player_index)
    game.start_game()

