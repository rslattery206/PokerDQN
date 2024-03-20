import numpy as np
class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank} of {self.suit}"

class ActionEncoderDecoder:
    def __init__(self):
        self.actor_mapping = {'Hero': 1, 'Villain': 0}
        self.action_mapping = {'fold': 0,
                               'check': 1,
                               'call': 2,
                               'bet_3bb': 3,
                               'bet_half_pot': 4,
                               'bet_pot': 5,
                               'raise_3x': 6,
                               'all_in': 7}
        self.num_actors = len(self.actor_mapping)
        self.num_actions = len(self.action_mapping)

    def encode_actions(self, action_list):
        encoded_actions = []
        for action in action_list[:10]:  # Consider only the first 10 actions
            actor = action[0]
            action_type = action[1]
            actor_encoded = [actor]  # Use the actor as it is provided
            action_encoded = [0] * self.num_actions

            if action_type in self.action_mapping:
                action_encoded[self.action_mapping[action_type]] = 1

            amount_encoded = [action[2]]  # Amount as a single value

            # Pad the action encoding to ensure a fixed length
            padding_length = self.num_actions + 2  # Actor + Action + Amount
            padded_action = actor_encoded + action_encoded + amount_encoded

            # Fill with "empty" actions until it reaches the desired length
            while len(padded_action) < padding_length:
                padded_action += [0] * (self.num_actions + 1)

            # Trim to desired length if it exceeds the padding length
            padded_action = padded_action[:padding_length]

            encoded_actions.append(padded_action)

        # Fill with "empty" actions until it reaches 10 actions
        while len(encoded_actions) < 10:
            empty_action = [0] * (self.num_actions + 2)
            encoded_actions.append(empty_action)

        return encoded_actions

    def decode_actions(self, encoded_list):
        decoded_actions = []
        for encoded_action in encoded_list:
            actor = 'Hero' if encoded_action[0] == 1 else 'Villain'  # Decode actor based on the provided value
            action_index = encoded_action[1:self.num_actors + self.num_actions].index(1)
            action_type = next((k for k, v in self.action_mapping.items() if v == action_index), None)
            amount = encoded_action[-1]

            decoded_actions.append((actor, action_type, amount))

        return decoded_actions


class CardEncoderDecoder:
    def __init__(self):
        self.rank_mapping = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10,
                             'K': 11, 'A': 12, 'None': 13}
        self.suit_mapping = {'H': 0, 'S': 1, 'C': 2, 'D': 3, 'None': 4}
        self.num_ranks = len(self.rank_mapping)
        self.num_suits = len(self.suit_mapping)
        self.num_cards = self.num_ranks * self.num_suits + 1  # Including None

    def encode_cards(self, card_list):
        encoded_cards = []
        for card in card_list:
            rank_encoded = [0] * self.num_ranks
            suit_encoded = [0] * self.num_suits

            if card is not None:
                if card.rank in self.rank_mapping:
                    rank_encoded[self.rank_mapping[card.rank]] = 1
                if card.suit in self.suit_mapping:
                    suit_encoded[self.suit_mapping[card.suit]] = 1
            else:
                # None card
                rank_encoded[-1] = 1
                suit_encoded[-1] = 1

            encoded_cards.append(rank_encoded + suit_encoded)
        return encoded_cards

    def decode_one_hot(self, encoded_card):
        rank_index = np.where(encoded_card[:14] == 1)[0][0]  # Extract index of rank
        suit_index = np.where(encoded_card[14:] == 1)[0][0]  # Extract index of suit
        rank = next((k for k, v in self.rank_mapping.items() if v == rank_index), None)
        suit = next((k for k, v in self.suit_mapping.items() if v == suit_index), None)
        return rank, suit


class StateDecoder:
    def __init__(self):
        self.card_decoder = CardEncoderDecoder()
        self.action_decoder = ActionEncoderDecoder()

    def decode(self, encoded_state):
        # Decode hero cards
        hero_card1_encoded = encoded_state[:19]  # First 19 indices for hero card 1
        hero_card1_rank, hero_card1_suit = self.card_decoder.decode_one_hot(hero_card1_encoded)

        hero_card2_encoded = encoded_state[19:38]  # Next 19 indices for hero card 2
        hero_card2_rank, hero_card2_suit = self.card_decoder.decode_one_hot(hero_card2_encoded)
        community_card1_encoded = encoded_state[38:57]
        community_card1_rank, community_card_suit = self.card_decoder.decode_one_hot(community_card1_encoded)
        cc2enconded = encoded_state[57:76]
        cc2rank, cc2suit = self.card_decoder.decode_one_hot(cc2enconded)
        cc3enconded = encoded_state[76:95]
        cc3rank, cc3suit = self.card_decoder.decode_one_hot(cc3enconded)
        cc4enconded = encoded_state[95:114]
        cc4rank, cc4suit = self.card_decoder.decode_one_hot(cc4enconded)
        cc5enconded = encoded_state[114:133]
        cc5rank, cc5suit = self.card_decoder.decode_one_hot(cc5enconded)

        # Decode stacks and pot size
        stacks = [stack * 100 for stack in encoded_state[133:135]]  # Indices 66 to 67 for stacks
        pot_size = encoded_state[135] * 100  # Index 68 for pot size

        # Decode other state attributes
        raised = bool(encoded_state[136])
        hero_on_button = bool(encoded_state[137])

        # Decode actions
        all_encoded_actions = encoded_state[138:]
        encoded_actions = []


        i = 0
        while i <= len(all_encoded_actions) - 10:
            encoded_single_action = all_encoded_actions[i:i+10]

            encoded_actions.append(encoded_single_action)
            i = i+10

        encoded_actions = [tensor.tolist() for tensor in encoded_actions]
        actions = self.action_decoder.decode_actions(encoded_actions)

        return (hero_card1_rank, hero_card1_suit), (hero_card2_rank, hero_card2_suit), \
               (community_card1_rank, community_card_suit), (cc2rank, cc2suit), (cc3rank, cc3suit), (cc4rank, cc4suit), \
               (cc5rank, cc5suit), \
               stacks, pot_size, raised, hero_on_button, actions


