from main import PokerState
from main import Game
from main import Card
for i in range(0, 100):
        game = Game()
        print("-------")
        state = PokerState([game.deck.draw_card(),game.deck.draw_card()],[100, 100], [game.deck.draw_card(),game.deck.draw_card(),game.deck.draw_card(),game.deck.draw_card(),game.deck.draw_card()], 0, [game.deck.draw_card(), game.deck.draw_card()])
        game.deck.populate_deck()
        print(state.hero_cards)
        print(state.community_cards)
        print(state.evaluate_hand(True))
        print("-----\n")
# community_cards = [Card("K", "D"), Card("9", "D"), Card("Q", "D"), Card("10", "S"), Card("7", "S")]
# state = PokerState([Card("4", "H"), Card("J", "H")], [100, 100], community_cards, 0, [Card("4", "H"), Card("J", "H")])
# print(state.evaluate_hand(True))
