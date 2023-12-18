''' UNO rule models
'''

import numpy as np

import rlcard
from rlcard.models.model import Model
from collections import defaultdict


class UNORuleAgentV2(object):
    ''' UNO Rule agent version 2
    '''

    def __init__(self):
        self.use_raw = False
        self.use_net = False

    def step(self, state):
        ''' Predict the action given raw state. A naive rule. Choose the color
            that appears least in the hand from legal actions. Try to keep wild
            cards as long as it can.

        Args:
            state (dict): Raw state from the game

        Returns:
            action (str): Predicted action
        '''

        legal_actions = self.filter_draw(np.array(list(state['legal_actions'])))

        # Always choose the card with the most colors
        color_nums = self.count_colors(legal_actions)
        color = np.random.choice(color_nums[max(color_nums)])
        action = np.random.choice(self.filter_color(color, legal_actions))
        
        return action

    def eval_step(self, state):
        ''' Step for evaluation. The same to step
        '''
        return self.step(state), []

    @staticmethod
    def filter_draw(actions):
        ''' Filter the draw card. If we only have a draw card, we do not filter

        Args:
            action (list): A list of UNO card string

        Returns:
            filtered_draw (list): A filtered list of UNO string
        '''
        filtered_action = []
        for key in actions:
            if key != 60:
                filtered_action.append(key)

        if len(filtered_action) == 0:
            filtered_action = actions

        return filtered_action

    @staticmethod
    def filter_color(index, actions):
        ''' Choose a color action in hand

        Args:
            color (list): A String of UNO card color

        Returns:
            action (string): The actions should be return
        '''
        cards = []
        for key in actions:
            if key // 15 == index:
                cards.append(key)
        
        if len(cards) == 0:
            cards = actions

        return cards

    @staticmethod
    def count_colors(actions):
        ''' Count the number of cards in each color in hand

        Args:
            hand (list): A list of UNO card string

        Returns:
            color_nums (dict): The number cards of each color
        '''
        color_nums = {}
        nums_color = defaultdict(list)
        for key in actions:
            index = key // 15
            if index not in color_nums:
                color_nums[index] = 0
            color_nums[index] += 1
        
        for k, v in color_nums.items():
            nums_color[v].append(k)
        
        return {k: v for k, v in nums_color.items()}

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
        return False



