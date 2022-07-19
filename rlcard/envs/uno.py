import numpy as np
from collections import OrderedDict

from rlcard.envs import Env
from rlcard.games.uno import Game
from rlcard.games.uno.utils import encode_hand, encode_target
from rlcard.games.uno.utils import ACTION_SPACE, ACTION_LIST
from rlcard.games.uno.utils import cards2list

DEFAULT_GAME_CONFIG = {
        'game_num_players': 4,
        }

class UnoEnv(Env):

    def __init__(self, config):
        self.name = 'uno'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.state_shape = [[4, 4, 15] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def _extract_state(self, state):
        obs = np.zeros((4, 4, 15), dtype=int)
        encode_hand(obs[:3], state['hand']) # obs[0] - obs[2] 记录玩家当前手牌
        encode_target(obs[3], state['target']) # obs[3] 记录当前牌面牌值
        legal_action_id = self._get_legal_actions() # 记录当前玩家对应当前牌面所有 legal_actions 的 id
        extracted_state = {'obs': obs, 'legal_actions': legal_action_id} # 记录编码后的 obs 和 legal_action_id 值
        extracted_state['raw_obs'] = state # 记录原始 state 值
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']] # 记录原始 legal_actions 值
        extracted_state['action_record'] = self.action_recorder # 记录 action_recorder 值
        return extracted_state

    def get_payoffs(self):

        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        legal_ids = self._get_legal_actions()
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        # if (len(self.game.dealer.deck) + len(self.game.round.played_cards)) > 17:
        #    return ACTION_LIST[60]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions} # 获取当前 legal_actions 的所有 id
        return OrderedDict(legal_ids)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['num_players'] = self.num_players
        state['hand_cards'] = [cards2list(player.hand)
                               for player in self.game.players]
        state['played_cards'] = cards2list(self.game.round.played_cards)
        state['target'] = self.game.round.target.str
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.round.get_legal_actions(
            self.game.players, state['current_player'])
        return state
