import pydealer
from collections import defaultdict
from tabulate import tabulate

class RikikiGame:
    def __init__(self, num_players, starting_deck_size=2):
        self.num_players = num_players
        self.starting_deck_size = starting_deck_size
        self.current_deck_size = starting_deck_size
        self.deck = pydealer.Deck()
        self.players = []
        self.bids = []
        self.atout = None
        self.trick = pydealer.Stack()
        self.scores = defaultdict(int)
        self.pli_scores = defaultdict(int)

    def start_game(self):
        for round_number in range(11):
            self.current_deck_size = self.starting_deck_size + round_number
            self.deal_cards()
            self.select_atout()
            self.bidding_phase()
            self.play_all_tricks()
            self.calculate_scores()
            self.print_scores()
            self.print_overview_table()
            self.reset_for_next_round()

    def deal_cards(self):
        self.deck.shuffle()
        self.players = [self.deck.deal(self.current_deck_size) for _ in range(self.num_players)]
        self.bids = [None] * self.num_players
        self.pli_scores = defaultdict(int)

    def select_atout(self):
        self.atout = self.deck.deal(1)[0]
        print(f"Atout card: {self.atout}")

    def bidding_phase(self):
        for i in range(self.num_players):
            self.get_bid(i)

    def get_bid(self, player_num):
        print(f"Player {player_num + 1}, your hand:")
        for card in self.players[player_num]:
            print(card)
        while True:
            bid = input(f"Player {player_num + 1}, enter your bid (0-{self.current_deck_size}): ")
            if bid.isdigit() and int(bid) in range(self.current_deck_size + 1):
                bid = int(bid)
                if player_num == self.num_players - 1:  # Last player
                    if sum(self.bids[:player_num]) + bid != self.current_deck_size:
                        self.bids[player_num] = bid
                        break
                    else:
                        print(f"Invalid bid. The sum of bids cannot equal {self.current_deck_size}.")
                else:
                    self.bids[player_num] = bid
                    break
            else:
                print(f"Invalid input. Please enter a number between 0 and {self.current_deck_size}.")

    def play_all_tricks(self):
        for _ in range(self.current_deck_size):
            self.play_trick()

    def play_trick(self):
        trick_cards = []
        leading_suit = None

        for player_num in range(self.num_players):
            card = self.players[player_num].deal(1)[0]  # Deal the top card
            trick_cards.append((card, player_num))

            if leading_suit is None:
                leading_suit = card.suit

            print(f"Player {player_num + 1} plays: {card}")

        winning_card, winning_player = self.determine_winner(trick_cards, leading_suit)
        print(f"Player {winning_player + 1} wins the trick with {winning_card}")
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
            print(f"Player {player_num + 1} score: {self.scores[player_num]}")

    def print_overview_table(self):
        table_data = []
        for player_num in range(self.num_players):
            row = [
                f"Player {player_num + 1}",
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

if __name__ == "__main__":
    num_players = 4  # Adjust as needed
    game = RikikiGame(num_players)
    game.start_game()
