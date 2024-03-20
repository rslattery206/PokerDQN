import random
import torch
import DQN
import treys
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import numpy as np
from encode import ActionEncoderDecoder, CardEncoderDecoder, StateDecoder


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __lt__(self, other):
        if self.rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10"] and other.rank not in ["2", "3", "4", "5", "6",
                                                                                              "7", "8", "9", "10"]:
            return False
        elif self.rank in ["2", "3", "4", "5", "6", "7", "8", "9", "10"] and other.rank in ["2", "3", "4", "5", "6",
                                                                                            "7", "8", "9", "10"]:
            return int(self.rank) < int(other.rank)
        elif self.rank not in ["2", "3", "4", "5", "6", "7", "8", "9", "10"] and other.rank not in ["2", "3", "4", "5",
                                                                                                    "6", "7", "8", "9",
                                                                                                    "10"]:
            return True
        else:
            return True

    def __repr__(self):
        return f"{self.rank} of {self.suit}"


class Deck:
    def __init__(self):
        self.cards = []
        self.populate_deck()

    def populate_deck(self):
        self.cards = []
        suits = ["S", "C", "D", "H"]
        values = [str(i) for i in range(2, 10)] + ["T", "J", "Q", "K", "A"]
        for suit in suits:
            for value in values:
                self.cards.append(Card(value, suit))

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        return self.cards.pop()

    def verify(self):
        for c in self.cards:
            print(c)


class PokerState:
    def __init__(self, hero_cards, stacks, community_cards, pot_size, villain_cards, raised=False, hero_on_button=True,
                 actions=None,
                 prev_actions=None):
        self.stacks = stacks
        self.hero_cards = hero_cards
        self.community_cards = community_cards
        self.pot_size = pot_size
        self.raised = raised
        self.hero_on_button = hero_on_button
        self.prev_actions = prev_actions if prev_actions else []
        self.actions = actions if actions else []
        self.villain_cards = villain_cards

        self.hero_turn = hero_on_button  # bool, True means action is on the hero
        self.hero_put_in_pot = 0
        self.villain_put_in_pot = 0

    def deal_community_cards(self, cards):
        # for a preflop all in
        for i, c in enumerate(cards):
            self.community_cards[i] = c

    def deal_flop(self, cards):
        # list of 3 cards
        for i, c in enumerate(cards):
            self.community_cards[i] = c

    def deal_turn(self, card):
        self.community_cards[3] = card

    def deal_river(self, card):
        self.community_cards[4] = card

    def create_copy(self, hero_turn):
        new_state = PokerState(
            hero_cards=self.hero_cards,
            villain_cards=self.villain_cards,
            stacks=self.stacks.copy(),
            community_cards=self.community_cards.copy(),
            pot_size=self.pot_size,
            raised=self.raised,
            hero_on_button=self.hero_on_button,
            actions=self.actions.copy(),
            prev_actions=self.prev_actions.copy()

        )
        new_state.hero_turn = hero_turn
        return new_state


    def evaluate_hand(self, hero=True):
        hole_cards = self.hero_cards if hero else self.villain_cards
        trey_hole_cards = []
        trey_community_cards = []
        for card in hole_cards:
            trey_hole_cards.append(treys.Card.new(card.rank + card.suit.lower()))
        for card in self.community_cards:
            if card is not None:
                trey_community_cards.append(treys.Card.new(card.rank + card.suit.lower()))
        evaluator = treys.Evaluator()
        if len(trey_community_cards) == 0:
            return 3731
        return evaluator.evaluate(trey_hole_cards, trey_community_cards)

    def showdown(self):
        hero_strength = self.evaluate_hand(True)
        villain_strength = self.evaluate_hand(False)
        print("hero strength " + str(hero_strength))
        print("villain strength " + str(villain_strength))
        # treys evaluator returns lower number for better hands
        # < 10 is a straight flush
        # 7462 is 3 high
        if hero_strength < villain_strength:
            self.awardPot(True, "showdown")
            return 1
        elif hero_strength > villain_strength:
            self.awardPot(False, "showdown")
            return -1
        else:
            print("same hand")
            half_pot = self.pot_size / 2
            self.stacks = [self.stacks[0] + half_pot, self.stacks[1] + half_pot]
            self.pot_size = 0
            return 0

    def __repr__(self):
        return f"PokerState(hero_cards={self.hero_cards}, stacks={self.stacks}, " \
               f"community_cards={self.community_cards}, pot_size={self.pot_size}, " \
               f"raised={self.raised}, hero_on_button={self.hero_on_button}, " \
               f"prev_actions={self.prev_actions}, actions={self.actions}, " \
               f"hero_turn={self.hero_turn}, " \
               f"villain_cards={self.villain_cards})"

    def add_action(self, hero, action_type, pip):
        # hero bool, the decision made, put in pot. pip is 0 for fold, check
        self.actions.append((1 if hero else 0, action_type, pip))

    def check(self, hero):
        # change turn
        self.hero_turn = not self.hero_turn
        # THIS IS WHERE THE RPBOELM IS DONT KNOW WHETERH TO REWARD HERO OR VILLAIN
        self.add_action(hero, "check", 0)

    def awardPot(self, hero, manner):  # manner in which pot was awarded
        if hero:
            self.stacks[0] += self.pot_size
        else:
            self.stacks[1] += self.pot_size
        if manner == "fold":
            self.add_action(not hero, "fold", 0)
        else:  # handle shodwon
            pass
        self.pot_size = 0

    def switch(self):
        self.hero_on_button = not self.hero_on_button

    def put_in_pot(self, hero, amount, action_type):
        # action type is the manner in which the money is being payed
        # change turn
        self.hero_turn = not hero
        if hero:
            self.stacks[0] -= amount
            self.hero_put_in_pot += amount
            self.pot_size += amount
        else:
            self.stacks[1] -= amount
            self.villain_put_in_pot += amount
            self.pot_size += amount
        if self.hero_put_in_pot != self.villain_put_in_pot:
            self.raised = True
        else:
            self.raised = False
        if action_type is not None:  # only none if its paying the blinds
            self.add_action(hero, action_type, amount)

    def encode(self):
        """
        Encode the state into a tensor representation.

        [:19] hero card 1
        [19:38] hero card 2
        [38: 57] community card 1
        [57:76] community card 2
        [76:95] community card 3
        [95:114] community card 4
        [114:133] community card 5
        [135] pot size / 100
        [136] raised (is there discrepancy in amount of money put in pot)
        [137] hero_on_button (position)
        [138:238] actions (10 indices per action)
        hero hand strength divided by 100
        [238:257] villain card 1
        [257:276] villain card 2
        """
        card_encoder = CardEncoderDecoder()
        action_encoder = ActionEncoderDecoder()
        # Encode hero cards
        encoded_hero_cards = card_encoder.encode_cards(self.hero_cards)
        encoded_community_cards = card_encoder.encode_cards(self.community_cards)
        encoded_actions = action_encoder.encode_actions(self.actions)
        # eval hand strength
        encoded_hand_strength = (self.evaluate_hand(True) - 3731) / 1000
        encoded_stacks_pot = [stack / 100 for stack in self.stacks] + [self.pot_size / 100]
        encoded_raised = 1 if self.raised else 0
        encoded_hero_on_button = 1 if self.hero_on_button else 0
        encoded_villain_cards = card_encoder.encode_cards(self.villain_cards)
        encoded_state = np.concatenate((np.concatenate(encoded_hero_cards),
                                        np.concatenate(encoded_community_cards),
                                        encoded_stacks_pot,
                                        [encoded_raised, encoded_hero_on_button],
                                        np.concatenate(encoded_actions), [encoded_hand_strength]))
        # REMOVED np.concatenate(encoded_villain_cards
        encoded_state_tensor = torch.tensor(encoded_state, dtype=torch.float)
        # print("\n------")
        # print(len(encoded_hero_cards[0]))
        # print(len(encoded_hero_cards[1]))
        # print(len(encoded_community_cards[0]))
        # print(len(encoded_community_cards[1]))
        # print(len(encoded_community_cards[2]))
        # print(len(encoded_community_cards[3]))
        # print(len(encoded_community_cards[4]))
        # print("Hero card encoded length:" + str(len(encoded_hero_cards)))
        # print("encoded community cards" + str(len(encoded_community_cards)))
        # print("stacks and pot" + str(len(encoded_stacks_pot)))
        # print("raised, hero on button" + str(len([encoded_raised, encoded_hero_on_button])))
        # print("actions")
        # print(self.actions)
        # for i in encoded_actions:
        #     print(len(i))
        #     print(i)
        # print("shape:")
        # print(encoded_state_tensor.shape)

        #
        # print("------\n")

        return encoded_state_tensor


class Game:
    def __init__(self):
        self.deck = Deck()
        self.deck.shuffle()


def parse_action(action_integer):
    reverse_action_mapping = {
        0: 'fold',
        1: 'check',
        2: 'call',
        3: 'bet_3bb',
        4: 'bet_half_pot',
        5: 'bet_pot',
        6: 'raise_3x',
        7: 'all_in'
    }
    return reverse_action_mapping[action_integer]


def hard_ai_flop(state):
    if state.raised:
        return "call"
    else:
        return "check"

    diceroll = random.randint(0, 10)
    if state.stacks[1] < 10:
        if state.raised:
            if diceroll < 5:
                return "all_in"
            else:
                return "fold"
    if state.raised:
        return "call"
    else:
        return "check" if diceroll < 5 else "bet_half_pot"


def hard_ai_preflop(state):
    if state.raised:
        return "call"
    else:
        return "check"
    diceroll = random.randint(0, 10)  # 11 numbers
    if len(state.actions) > 0 and state.actions[-1][1] == "all_in":
        if diceroll > 5:
            return "call"
        else:
            return "fold"
    if state.stacks[1] < 10:
        if state.raised:
            if diceroll < 5:
                return "all_in"
            else:
                return "fold"

    if state.raised:
        if diceroll > 8:
            return "all_in"
        elif diceroll < 4:
            return "call"
        else:
            return "fold"
    else:
        if diceroll < 5:
            return "check"
        elif diceroll < 7:
            return "bet_3bb"
        else:
            if state.pot_size > state.stacks[1]:
                return "bet_pot"
            else:
                return "bet_half_pot"


def preflop(hero_on_button, stacks, pokerDQN):
    """
    Initialize PokerState here
    Deal cards, posts blinds, allow preflop action
    If hero is on BB, hero_on_button is false

    stacks is a list containing stack sizes e.g [100, 100]
    :return:
    """
    game.deck.populate_deck()
    game.deck.shuffle()
    hero_card_1 = game.deck.draw_card()
    hero_card2 = game.deck.draw_card()
    villain_card1 = game.deck.draw_card()
    villain_card2 = game.deck.draw_card()
    # initial state before any action, even the blinds
    state = PokerState([hero_card_1, hero_card2], stacks,
                       [None, None, None, None, None],
                       0, [villain_card1, villain_card2], False, hero_on_button)

    # Blinds
    if hero_on_button:
        state.put_in_pot(True, .5, None)
        state.put_in_pot(False, 1, None)
    else:
        state.put_in_pot(False, .5, None)
        state.put_in_pot(True, 1, None)

    # raised is now true

    # pre_state: the state directly before the action about to be taken
    pre_state = state.create_copy(state.hero_turn)
    reward = 0
    gg = False  # the game has ended preflop, somebody folded
    all_in_before_river = False  # players have gone all in, no more action until river showdown
    round_over = False
    print(" Preflop begins: ")
    while not round_over:
        """ Get the Action """
        if state.hero_turn:
            # the hero's memory will be updated with [prestate, action, reward, resulting state] later
            # prestate is set to the current state, which is the state directly before the hero's action
            hero_making_decision = True
            pre_state = state.create_copy(state.hero_turn)
            action = pokerDQN.select_action(pre_state.encode())
            decision = parse_action(action)  # parse action
        else:
            hero_making_decision = False
            # it's the villains turn
            decision = hard_ai_preflop(state)

        """ Handle the Action """
        if state.hero_turn:
            print("the hero does action: " + decision)
        else:
            print("the villain does action: " + decision)
        if decision == "call":
            state.put_in_pot(state.hero_turn, abs(state.villain_put_in_pot - state.hero_put_in_pot), "call")
            if len(state.actions) != 1:  # this call is not a limp
                # calling a raise preflop will always advance to the flop unless that raise is the big blind
                round_over = True
        elif decision == "check":
            # checking preflop always advances to the flop
            state.check(state.hero_turn)
            round_over = True
        elif decision == "bet_half_pot":
            # betting any amount cannot immidietly end the round
            bet_size = .5 * state.pot_size
            state.put_in_pot(state.hero_turn, .5 * state.pot_size, "bet_half_pot")
            if state.stacks[0] < 0 or state.stacks[1] == 0:
                reward = -420
                gg = True
                break


        elif decision == "bet_pot":
            bet_size = state.pot_size
            state.put_in_pot(state.hero_turn, state.pot_size, "bet_pot")
            if state.stacks[0] < 0 or state.stacks[1] == 0:
                reward = -420
                gg = True
                break


        elif decision == "all_in":
            state.put_in_pot(state.hero_turn, state.stacks[0] if state.hero_turn else state.stacks[1], "all_in")
            if state.stacks[0] == 0 and state.stacks[1] == 0:  # hero went all in, the correct decision is "call"
                print("villain went all in and the hero went all in directly after")
                reward = -420
                gg = True
                break
        elif decision == "bet_3bb":
            state.put_in_pot(state.hero_turn, 3, "bet_3bb")
            if state.stacks[0] < 0 or state.stacks[1] == 0:
                reward = -420
                gg = True
                break

        elif decision == "raise_3x":
            print("stacks:" )
            print(state.stacks)
            if abs(state.hero_put_in_pot - state.villain_put_in_pot) * 3 > state.stacks[0]:
                print("not enough to bet this")
                # not enough to bet this
                reward = -420
                gg = True
                break
            if not state.raised:
                print("raised when not raised")
                reward = -5
                gg = True
                break
                # punish for choosing "raise" when not raised
            print("raised 3x: " + str(abs(state.hero_put_in_pot - state.villain_put_in_pot) * 3))
            state.put_in_pot(state.hero_turn, abs(state.hero_put_in_pot - state.villain_put_in_pot) * 3, "raise_3x")
        else:  # decision == "fold":
            # folding always ends the hand
            adj_reward = state.pot_size - abs(state.hero_put_in_pot - state.villain_put_in_pot)
            reward = -adj_reward if state.hero_turn else adj_reward
            state.awardPot(not state.hero_turn, "fold")
            state.raised = False
            round_over = True
            gg = True
        if state.stacks[0] == 0 and state.stacks[1] == 0:  # "players are all in"
            if not state.raised:
                all_in_before_river = True
            else:
                print("somebody went all in but its not raised")
        # a player's decision has been completed
        # if it was the villain who completed and action and the round not over,

    # no more action

    if all_in_before_river:
        state.deal_community_cards([game.deck.draw_card(),
                                    game.deck.draw_card(),
                                    game.deck.draw_card(),
                                    game.deck.draw_card(),
                                    game.deck.draw_card()])
        print("showdown results")
        showdown_pot_size = state.pot_size
        result = state.showdown()
        print(result)
        # pot awarding occurs during the showdown call
        gg = True

        if result == 1:
            reward = showdown_pot_size / 2
        elif result == -1:
            reward = -showdown_pot_size / 2
        else:
            reward = 0

    if state.raised:
        # note that state.raised set to false if somebody folded (game is ending)
        print("the round ended but the put was raised, somebody broke the rules")
        print("state.raised: " + str(state.raised))
        print("hero pip vs villain pip: " + str(state.hero_put_in_pot - state.villain_put_in_pot))
        # The round ended but the pot was raised, means somebody broke the rules
        reward = -420
    # re-obtain the action
    action = pokerDQN.select_action(pre_state.encode())
    print(state.hero_turn)
    if pre_state.hero_turn:
        print("pushing")
        pokerDQN.memory.push([pre_state.encode(), action, reward/100, state.encode()])
        pokerDQN.optimize_model()
        print("----")
        print(pre_state)
        print("action: " + str(action))
        print("reward: " + str(reward/100))
        print(state)
        print("-----")

    if gg:  # the game is over
        return
    return post_flop(state, pokerDQN, 0)


def post_flop(state, pokerDQN, round_counter):
    print("post flop with " + str(round_counter))
    """
    0 is flop, 1 is turn, 2 is river
    Initialize PokerState here
    Deal community cards, allow action
    If hero is on BB, hero_on_button is false

    stacks is a list containing stack sizes e.g [100, 100]
    :return:
    """
    if round_counter == 0:
        state.deal_flop([game.deck.draw_card(), game.deck.draw_card(), game.deck.draw_card()])
        name = "flop"
    elif round_counter == 1:
        state.deal_turn(game.deck.draw_card())
        name = "turn"
    else:
        state.deal_river(game.deck.draw_card())
        name = 'river'
    print('\nAdvanced to ' + name + "\n")
    print(state.community_cards)
    if not state.hero_on_button:
        state.hero_turn = True
    else:
        state.hero_turn = False
    # reset the amounts put in the pot
    state.hero_put_in_pot = 0
    state.villain_put_in_pot = 0
    pre_state = state.create_copy(state.hero_turn)
    reward = 0
    action = -1  # should never be pushed to memory as -1
    round_over = False
    all_in_before_river = False
    gg = False
    print(" Round begins: ")
    num_flops_actions = 0
    while not round_over:
        num_flops_actions += 1
        """ Get the Action """
        if state.hero_turn:
            # the hero's memory will be updated with [prestate, action, reward, resulting state] later
            # prestate is set to the current state, which is the state directly before the hero's action
            hero_making_decision = True
            pre_state = state.create_copy(state.hero_turn)
            action = pokerDQN.select_action(pre_state.encode())
            decision = parse_action(action)  # parse action
        else:
            hero_making_decision = False
            # it's the villains turn
            decision = hard_ai_flop(state)

        """ Handle FLop Action"""
        if state.hero_turn:
            print("the hero does action: " + decision)
        else:
            print("the villain does action: " + decision)

        if decision == "call":
            state.put_in_pot(state.hero_turn, abs(state.villain_put_in_pot - state.hero_put_in_pot), "call")
            if num_flops_actions == 0 or not state.raised:  # the correct option is "check"
                reward = -420
                gg = True
                break
            round_over = True  # calling a bet on a flop in heads up always ends the round
        # FLOP
        elif decision == "check":
            state.check(state.hero_turn)
            if state.raised:
                print("checked while raised on the flop")
                reward = -420
                gg = True
                break
            elif num_flops_actions == 2:  # the second check on the flop goes to the turn
                break
        elif decision == "bet_half_pot":
            # betting any amount cannot immidietly end the round
            bet_size = .5 * state.pot_size
            state.put_in_pot(state.hero_turn, .5 * state.pot_size, "bet_half_pot")
            if state.stacks[0] < 0:
                reward = -420
                gg = True
                break
        elif decision == "bet_pot":
            bet_size = state.pot_size
            state.put_in_pot(state.hero_turn, state.pot_size, "bet_pot")
            if state.stacks[0] < 0 or state.stacks[1] == 0:
                reward = -420
                gg = True
                break
        elif decision == "all_in":
            state.put_in_pot(state.hero_turn, state.stacks[0] if state.hero_turn else state.stacks[1], "all_in")
            if state.stacks[0] == 0 and state.stacks[1] == 0:  # villain went all in, and the hero JUST did, the correct decision is "call"
                reward = -420
                gg = True
                break

                # TODO THIS IS THE FLOP
        elif decision == "bet_3bb":
            state.put_in_pot(state.hero_turn, 3, "bet_3bb")
            if state.stacks[0] < 0 or state.stacks[1] == 0:
                reward = -420
                gg = True
                break
        elif decision == "raise_3x":
            bet_size = abs(state.hero_put_in_pot - state.villain_put_in_pot) * 3
            state.put_in_pot(state.hero_turn, bet_size, "raise_3x")
            if not state.raised:
                print("raised when not raised")
                reward = -420
                gg = True
                break
                # punish for choosing "raise" when not raised
            if bet_size > state.stacks[0]:
                # not enough to bet this
                reward = -420
                gg = True
                break
            print("raised 3x: " + str(abs(state.hero_put_in_pot - state.villain_put_in_pot) * 3))
        else:  # decision == "fold":
            # folding always ends the hand
            adj_reward = state.pot_size - abs(state.hero_put_in_pot - state.villain_put_in_pot)
            reward = -adj_reward if state.hero_turn else adj_reward
            state.awardPot(not state.hero_turn, "fold")
            state.raised = False
            round_over = True
            gg = True
        if state.stacks[0] == 0 and state.stacks[1] == 0:  # "players are all in"
            if not state.raised and round_counter != 2:
                all_in_before_river = True
            else:
                print("somebody went all in but its not raised")


        # FLOP
    if all_in_before_river:  # not the river
        if round_counter == 0:
            state.deal_turn(game.deck.draw_card())
            state.deal_river(game.deck.draw_card())
        elif round_counter == 1:
            state.deal_river(game.deck.draw_card())
        print("pre-river showdown results")
        print("community cards:")
        print(state.community_cards)
        showdown_pot_size = state.pot_size
        result = state.showdown()
        print(result)
        # pot awarding occurs during the showdown call
        gg = True

        if result == 1:
            reward = showdown_pot_size / 2
        elif result == -1:
            reward = -showdown_pot_size / 2
        else:
            reward = 0
    print(num_flops_actions)
    print(state.raised)
    print("round counter" + str(round_counter))
    print("num_flop_actions" + str(num_flops_actions))
    print("raised " + str(state.raised))
    if state.raised:
        print("the round ended (post flop) but the put was raised, somebody broke the rules")
        print("state.raised: " + str(state.raised))
        print("hero pip vs villain pip: " + str(state.hero_put_in_pot - state.villain_put_in_pot))
        reward = -420
    elif round_counter == 2 and num_flops_actions > 1 and state.pot_size > 0:
        print("river showdown:")
        showdown_pot_size = state.pot_size
        print(showdown_pot_size)
        result = state.showdown()
        print(result)
        if result == 1:
            reward = showdown_pot_size / 2
        elif result == -1:
            reward = -showdown_pot_size / 2
        else:
            reward = 0
    action = pokerDQN.select_action(pre_state.encode())
    if pre_state.hero_turn:  # FLOP
        pokerDQN.memory.push([pre_state.encode(), action, reward/100, state.encode()])
        pokerDQN.optimize_model()
        print("----")
        print(pre_state)
        print(action)
        print(reward)
        print(state)
        print("-----")
    else:
        # the action is depicting what the villain did
        pass


    if gg or round_counter == 2:
        print("hand ended")
        print(name)
        return
    post_flop(state, pokerDQN, round_counter + 1)



if __name__ == "__main__":
    game = Game()
    dc1 = game.deck.draw_card()
    dc2 = game.deck.draw_card()
    villain1 = game.deck.draw_card()
    villain2 = game.deck.draw_card()
    dc5 = game.deck.draw_card()
    pokerAgent = DQN.DQNAgent(239, 8)
    for i in range(0, 1000):
        preflop(i % 2 == 0, [100, 100], pokerAgent)
        print("\n")
    plt.plot(pokerAgent.loss_history, label="loss")
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    def plot_q_values_heatmap(q_values_history, num_states, num_actions):
        # Concatenate the Q-values from the history
        q_values = np.concatenate(q_values_history)

        # Reshape the Q-values into a grid
        q_values_grid = q_values.reshape(-1, num_actions)  # Assuming q_values is already flattened

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(q_values_grid, cmap="viridis", cbar=True)
        plt.xlabel('Actions')
        plt.ylabel('States')
        plt.title('Q-Values Heatmap')
        plt.show()


    # Example usage:
    num_states = 239
    num_actions = 8
    plot_q_values_heatmap(pokerAgent.q_values_history, num_states, num_actions)
    gradients_fc1 = pokerAgent.feature_gradients
    gradients_fc1_flat = gradients_fc1.reshape(-1)

    # Plot histogram of gradients
    plt.figure(figsize=(8, 6))
    plt.hist(gradients_fc1_flat, bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Input Gradients for First Layer (fc1)')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

