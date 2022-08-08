from rlcard.games.uno.card import UnoCard
from rlcard.games.uno.judger import UnoJudger
from rlcard.games.uno.utils import cards2list, WILD, WILD_DRAW_4

import numpy as np

class UnoRound:

    def __init__(self, dealer, num_players, np_random):
        ''' Initialize the round class

        Args:
            dealer (object): the object of UnoDealer
            num_players (int): the number of players in game
        '''
        self.np_random = np_random
        self.dealer = dealer
        self.target = None
        self.current_player = np.random.randint(0, num_players)
        self.num_players = num_players
        self.direction = 1
        self.played_cards = []
        self.is_over = False
        self.winner = None
        self.payoffs = [0 for _ in range(self.num_players)]

    def flip_top_card(self):
        ''' Flip the top card of the card pile

        Returns:
            (object of UnoCard): the top card in game

        '''
        top = self.dealer.flip_top_card()
        if top.trait == 'wild': # 如果首张是换色牌，则随机选一个颜色
            top.color = self.np_random.choice(UnoCard.info['color'])
        self.target = top
        self.played_cards.append(top)
        return top

    def perform_top_card(self, players, top_card):
        ''' Perform the top card

        Args:
            players (list): list of UnoPlayer objects
            top_card (object): object of UnoCard
        '''
        if top_card.trait == 'skip': # 首牌为 ‘跳过’
            self.current_player = 1
        elif top_card.trait == 'reverse': # 首牌为 ‘反转’
            self.direction = -1
            self.current_player = (0 + self.direction) % self.num_players
        elif top_card.trait == 'draw_2': # 首牌为 ‘+2’
            player = players[self.current_player]
            self.dealer.deal_cards(player, 2)

    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of UnoPlayer
            action (str): string of legal action
        '''
        if action == 'draw': # 当前 action 为 ‘抽牌’
            self._perform_draw_action(players)
            return None
        player = players[self.current_player]
        card_info = action.split('-')
        color = card_info[0]
        trait = card_info[1]
        # remove correspongding card —— 移除对应牌值
        remove_index = None
        if trait == 'wild' or trait == 'wild_draw_4': # 记录移除的万能牌
            for index, card in enumerate(player.hand):
                if trait == card.trait:
                    remove_index = index
                    break
        else:
            for index, card in enumerate(player.hand): # 记录移除的数字牌或功能牌
                if color == card.color and trait == card.trait:
                    remove_index = index
                    break
        card = player.hand.pop(remove_index) # 移除当前 action 对应手牌
        if not player.hand: # 当前玩家手牌为空，游戏结束
            self.is_over = True
            self.winner = [self.current_player]
        self.played_cards.append(card)

        # perform the number action —— 执行当前 action（数字牌）
        if card.type == 'number':
            self.current_player = (self.current_player + self.direction) % self.num_players
            self.target = card

        # perform the wild action —— 执行当前 action（万能牌）
        elif card.type == 'wild':
            card = UnoCard('wild', color, trait)
            self._preform_non_number_action(players, card)
        # perform other actions —— 执行当前 action（功能牌）
        else:
            self._preform_non_number_action(players, card)

    def get_legal_actions(self, players, player_id):
        wild_flag = 0
        wild_draw_4_flag = 0
        legal_actions = []
        wild_4_actions = []
        hand = players[player_id].hand
        target = self.target
        if target.type == 'wild':
            for card in hand: # 记录当前玩家可出牌型
                if card.type == 'wild':
                    if card.trait == 'wild_draw_4': # 当前手牌内有 ‘+4’
                        if wild_draw_4_flag == 0:
                            wild_draw_4_flag = 1
                            wild_4_actions.extend(WILD_DRAW_4)
                    else: # 当前手牌有 ‘换色’
                        if wild_flag == 0:
                            wild_flag = 1
                            legal_actions.extend(WILD)
                elif card.color == target.color: # 当前手牌有 可出的数字牌
                    legal_actions.append(card.str)

        # target is aciton card or number card
        else:
            for card in hand:
                if card.type == 'wild':
                    if card.trait == 'wild_draw_4': # 当前手牌内有 ‘+4’
                        if wild_draw_4_flag == 0:
                            wild_draw_4_flag = 1
                            wild_4_actions.extend(WILD_DRAW_4)
                    else: # 当前手牌有 ‘换色’
                        if wild_flag == 0:
                            wild_flag = 1
                            legal_actions.extend(WILD)
                elif card.color == target.color or card.trait == target.trait: # 当前手牌有 可出的数字牌或功能牌
                    legal_actions.append(card.str)
        if not legal_actions: # 只有 ‘+4’ 牌的情况下，legal_actions 为 ‘+4’
            legal_actions = wild_4_actions
        if not legal_actions: # 没有任何可出牌型的情况下，legal_actions 为 ‘加牌’
            legal_actions = ['draw']
        return legal_actions

    def get_state(self, players, player_id):
        ''' Get player's state

        Args:
            players (list): The list of UnoPlayer
            player_id (int): The id of the player
        '''
        state = {}
        player = players[player_id]
        state['hand'] = cards2list(player.hand)
        state['target'] = self.target.str
        state['played_cards'] = cards2list(self.played_cards)
        state['legal_actions'] = self.get_legal_actions(players, player_id) # 获取当前玩家可出牌型
        state['num_cards'] = []
        for player in players: # 统计每个玩家当前手牌数
            state['num_cards'].append(len(player.hand))
        return state

    def get_payoffs(self, players):
        '''Get player's payoffs'''
        # 计分策略：取二、三、四名游戏结束时的手牌分总和正数与第一名的手牌分相加
        winner_payoffs = 0
        for index, player in enumerate(players):
            player_score = self.count_hand_score(player.hand)
            self.payoffs[index] = player_score
            winner_payoffs += -(player_score)
            
        if self.winner is None:
            self.winner = UnoJudger.judge_winner(self.payoffs)
        
        for i in self.winner:
            self.payoffs[i] += winner_payoffs
        return self.payoffs

    def count_hand_score(self, cards):
        '''Count player hand card score'''
        count = 0
        for card in cards:
            if card.type == 'number':
                count += int(card.trait)
            elif card.type == 'action':
                count += 20
            elif card.type == 'wild':
                count += 50
        return -count

    def replace_deck(self):
        ''' Add cards have been played to deck
        '''
        self.dealer.deck.extend(self.played_cards)
        self.dealer.shuffle()
        self.played_cards = []

    def is_draw_available(self, card):
        '''Judge the card whether is available'''
        # draw a card with the same color or the same trait of target —— 抽牌（数字牌或功能牌）
        if card.color == self.target.color or card.trait == self.target.trait:
            return True
        # draw a wild card —— 抽牌（万能牌）
        elif card.type == 'wild':
            return True
        # draw a card with the diffrent color of target —— 抽牌（其他牌）
        return False

    def _perform_draw_action(self, players):
        # replace deck if there is no card in draw pile
        if not self.dealer.deck: # 当牌盒内的牌不够时
            # 游戏循环：从已出牌型中重新洗牌抽牌
            # self.replace_deck()
            # 游戏结束：统计所有玩家当前牌值
            self.is_over = True
            return None

        card = self.dealer.deck.pop()
        players[self.current_player].hand.append(card)
            
        if not self.is_draw_available(card): # 如果抽牌不合法，则置玩家为下一玩家
            self.current_player = (self.current_player + self.direction) % self.num_players

    def _preform_non_number_action(self, players, card):
        current = self.current_player
        direction = self.direction
        num_players = self.num_players

        # perform reverse card —— 反转操作，更新方向
        if card.trait == 'reverse':
            self.direction = -1 * direction

        # perfrom skip card —— 跳过操作，禁止下家出牌
        elif card.trait == 'skip':
            current = (current + direction) % num_players

        # perform draw_2 card —— ‘+2’操作，给下家加牌并跳过
        elif card.trait == 'draw_2':
            if len(self.dealer.deck) < 2: # 当牌盒内的牌不够时
                # 游戏循环：从已出牌型中重新洗牌抽牌
                # self.replace_deck()
                # 游戏结束：统计所有玩家当前牌值
                self.is_over = True
                return None
            self.dealer.deal_cards(players[(current + direction) % num_players], 2)
            current = (current + direction) % num_players

        # perfrom wild_draw_4 card —— ‘+4’操作，
        elif card.trait == 'wild_draw_4': # 当牌盒内的牌不够时
            if len(self.dealer.deck) < 4:
                # 游戏循环：从已出牌型中重新洗牌抽牌
                # self.replace_deck()
                # 游戏结束：统计所有玩家当前牌值
                self.is_over = True
                return None
            self.dealer.deal_cards(players[(current + direction) % num_players], 4)
            current = (current + direction) % num_players
        self.current_player = (current + self.direction) % num_players
        self.target = card
