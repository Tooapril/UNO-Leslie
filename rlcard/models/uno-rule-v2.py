''' UNO rule models
'''

import numpy as np

import rlcard
from rlcard.models.model import Model


class UNORuleAgentV2(object):
    ''' UNO Rule agent version 2
    '''

    def __init__(self):
        self.use_raw = True

    def step(self, state):
        ''' Predict the action given raw state. A naive rule. Choose the color
            that appears least in the hand from legal actions. Try to keep wild
            cards as long as it can.

        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''

        legal_actions = self.filter_draw(state['raw_legal_actions'])
        state = state['raw_obs']
        hand = state['hand']

        # Always choose the card with the most colors
        color_nums = self.count_colors(self.filter_wild(hand))
        color = max(color_nums, key=color_nums.get)  # type: ignore
        action = np.random.choice(self.filter_color(color, legal_actions))
        
        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    @staticmethod
    def filter_wild(hand):
        ''' Filter the wild cards. If all are wild cards, we do not filter

        Args:
            hand (list): A list of UNO card string

        Returns:
            filtered_hand (list): A filtered list of UNO string
        '''
        filtered_hand = []
        for card in hand:
            if not card[2:6] == 'wild':
                filtered_hand.append(card)

        if len(filtered_hand) == 0:
            filtered_hand = hand

        return filtered_hand

    @staticmethod
    def filter_draw(actions):
        ''' Filter the draw card. If we only have a draw card, we do not filter

        Args:
            action (list): A list of UNO card string

        Returns:
            filtered_draw (list): A filtered list of UNO string
        '''
        filtered_action = []
        for card in actions:
            if card != 'draw':
                filtered_action.append(card)

        if len(filtered_action) == 0:
            filtered_action = actions

        return filtered_action

    @staticmethod
    def filter_color(color, actions):
        ''' Choose a color action in hand

        Args:
            color (list): A String of UNO card color

        Returns:
            action (string): The actions should be return
        '''
        cards = []
        for card in actions:
            if card[0] == color:
                cards.append(card)
        
        if len(cards) == 0:
            cards = actions

        return cards

    @staticmethod
    def count_colors(hand):
        ''' Count the number of cards in each color in hand

        Args:
            hand (list): A list of UNO card string

        Returns:
            color_nums (dict): The number cards of each color
        '''
        color_nums = {}
        for card in hand:
            color = card[0]
            if color not in color_nums:
                color_nums[color] = 0
            color_nums[color] += 1

        return color_nums

class UNORuleModelV2(Model):
    ''' UNO Rule Model version 1
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        env = rlcard.make('uno')

        rule_agent = UNORuleAgentV2()
        self.rule_agents = [rule_agent for _ in range(env.num_players)]

    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.rule_agents

    @property
    def use_raw(self):
        ''' Indicate whether use raw state and action

        Returns:
            use_raw (boolean): True if using raw state and action
        '''
        return True



