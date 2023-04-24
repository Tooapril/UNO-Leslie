import json
import os
from collections import OrderedDict

import numpy as np

import rlcard
from rlcard.games.uno.card import UnoCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]  # type: ignore

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'games/uno/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

# a map of color to its index
COLOR_MAP = {'r': 0, 'g': 1, 'b': 2, 'y': 3}

# a map of trait to its index
TRAIT_MAP = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
             '8': 8, '9': 9, 'skip': 10, 'reverse': 11, 'draw_2': 12,
             'wild': 13, 'wild_draw_4': 14}

WILD = ['r-wild', 'g-wild', 'b-wild', 'y-wild']

WILD_DRAW_4 = ['r-wild_draw_4', 'g-wild_draw_4', 'b-wild_draw_4', 'y-wild_draw_4']


def init_deck():
    ''' Generate uno deck of 108 cards
    '''
    deck = []
    card_info = Card.info
    for color in card_info['color']:

        # init number cards —— 初始化数字牌
        for num in card_info['trait'][:10]:
            deck.append(Card('number', color, num))
            if num != '0':
                deck.append(Card('number', color, num))

        # init action cards —— 初始化功能牌
        for action in card_info['trait'][10:13]:
            deck.append(Card('action', color, action))
            deck.append(Card('action', color, action))

        # init wild cards —— 初始化万能牌
        for wild in card_info['trait'][-2:]:
            deck.append(Card('wild', color, wild))
    return deck


def cards2list(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of UnoCards objects

    Returns:
        (string): string representation of cards
    '''
    cards_list = []
    for card in cards:
        cards_list.append(card.get_str())
    return cards_list

def hand2dict(hand):
    ''' Get the corresponding dict representation of hand

    Args:
        hand (list): list of string of hand's card

    Returns:
        (dict): dict of hand
    '''
    hand_dict = {}
    for card in hand:
        if card not in hand_dict:
            hand_dict[card] = 1
        else:
            hand_dict[card] += 1
    return hand_dict

def encode_hand(hand):
    ''' Encode hand and represerve it into plane

    Args:
        plane (array): 3*4*15 numpy array
        hand (list): list of string of hand's card

    Returns:
        (array): 3*4*15 numpy array
    '''
    plane = np.zeros((3, 4, 15), dtype=int)
    plane[0] = np.ones((4, 15), dtype=int)
    hand = hand2dict(hand) # 统计各种牌拥有张数
    for card, count in hand.items():
        card_info = card.split('-')
        color = COLOR_MAP[card_info[0]] # 获取当前牌的颜色
        trait = TRAIT_MAP[card_info[1]] # 获取当前牌的数字或种类
        if trait >= 13: # 万能牌
            if plane[1][0][trait] == 0:
                for index in range(4):
                    plane[0][index][trait] = 0
                    plane[1][index][trait] = 1
        else: #❗️tips 除万能牌外，同一个颜色的牌型最多有且仅有 2 张
            plane[0][color][trait] = 0
            plane[count][color][trait] = 1 
    
    plane2 = np.zeros((4, 12), dtype=int)
    for i in range(4):
        plane2[i] = plane[2][i][1:13]
    # result = np.concatenate((plane[:2][:][:].flatten(), plane2.flatten()))
    return np.concatenate((plane[:2][:][:].flatten(), plane2.flatten()))


def encode_target(target):
    ''' Encode target and represerve it into plane

    Args:
        plane (array): 1*4*15 numpy array
        target(str): string of target card

    Returns:
        (array): 1*4*15 numpy array
    '''
    plane = np.zeros((4, 15), dtype=int)
    target_info = target.split('-')
    color = COLOR_MAP[target_info[0]]
    trait = TRAIT_MAP[target_info[1]]
    plane[color][trait] = 1
    return plane.flatten()

def encode_action(action):
    if action == '':
        return np.zeros(63, dtype=int)
    
    plane = np.zeros((4, 15), dtype=int)
    other_actions = np.zeros(3, dtype=int) # 记录 draw query pass 动作
    
    if action == 'draw':
        other_actions[0] = 1
    elif action == 'query':
        other_actions[1] = 1
    elif action == 'pass':
        other_actions[2] = 1
    else:
        target_info = action.split('-')
        color = COLOR_MAP[target_info[0]]
        trait = TRAIT_MAP[target_info[1]]
        plane[color][trait] = 1
    
    return np.concatenate((plane.flatten(), other_actions))

def encode_action_sequence(action_list, size=63):
    plane = np.zeros((len(action_list), size), dtype=int)
    for row, card in enumerate(action_list):
        plane[row, :] = encode_action(card)
    plane = plane.reshape(3, 252)
    return plane

def get_one_hot_array(num_left_cards, max_num_cards=10):
    one_hot = np.zeros(max_num_cards, dtype=int)
    if num_left_cards > max_num_cards:
        one_hot[max_num_cards - 1] = 1
    else:
        one_hot[num_left_cards - 1] = 1
    return one_hot 
