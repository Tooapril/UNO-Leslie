import numpy as np
import torch
from torch import nn
from collections import defaultdict


class DMCNet(nn.Module):
    def __init__(self,
                 state_shape,
                 action_shape,
                 mlp_layers=[512,512,512,512,512]):
        super().__init__()
        self.lstm = nn.LSTM(252, 128, batch_first=True)
        input_dim = np.prod(state_shape) + 128 + np.prod(action_shape) # 将不同 Agent 状态空间与动作空间大小相加
        layer_dims = [input_dim] + mlp_layers
        fc = []
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], 1)) # 网络层最后输出为 1
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, x, z, actions):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:, -1, :] # 仅保留最后一层 LSTM 的结果
        obs = torch.cat([x, lstm_out], dim=-1)
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        x = torch.cat((obs, actions), dim=1)
        values = self.fc_layers(x).flatten()
        return values

class DMCAgent:
    def __init__(self,
                 state_shape,
                 action_shape,
                 mlp_layers=[512,512,512,512,512],
                 exp_epsilon=0.01,
                 device=0):
        self.use_raw = False
        self.use_net = True
        self.device = torch.device('cuda:'+str(device))
        self.net = DMCNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def step(self, state):
        action_keys, values = self.predict(state)

        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon: # 以 𝛆 的概率探索
            action = np.random.choice(action_keys)
        else: # 以 1 - 𝛆 的概率利用
            action_idx = np.argmax(values)
            action = action_keys[action_idx]

        return action

    def eval_step(self, state):
        action_keys, values = self.predict(state)

        action_idx = np.argmax(values)
        action = action_keys[action_idx]

        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(values[i]) for i in range(len(action_keys))}

        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def predict(self, state):
        # Prepare obs and actions
        # obs = state['obs'].astype(np.float32)
        x_batch = state['x_batch'].astype(np.float32)
        z_batch = state['z_batch'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_keys = np.array(list(legal_actions.keys()))
        action_values = list(legal_actions.values())
        # One-hot encoding if there is no action features —— 给 action_values 按照 action 下标进行 one-hot 编码
        for i in range(len(action_values)):
            if action_values[i] is None:
                action_values[i] = np.zeros(self.action_shape[0])
                action_values[i][action_keys[i]-1] = 1
        action_values = np.array(action_values, dtype=np.float32) # 统一 action_values 的数据格式

        x_batch = np.repeat(x_batch[np.newaxis, :], len(action_keys), axis=0) # 统一 x_batch 的数据格式
        z_batch = np.repeat(z_batch[np.newaxis, :, :], len(action_keys), axis=0)# 统一 z_batch 的数据格式
        
        # Predict Q values
        values = self.net.forward(torch.from_numpy(x_batch).to(self.device),
                                  torch.from_numpy(z_batch).to(self.device),
                                  torch.from_numpy(action_values).to(self.device))

        return action_keys, values.cpu().detach().numpy()

    def forward(self, x, z, actions):
        return self.net.forward(x, z, actions)

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = device

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

class DMCModel:
    def __init__(self,
                 state_shape,
                 action_shape,
                 mlp_layers=[512,512,512,512,512],
                 exp_epsilon=0.01,
                 device=0):
        self.agents = []
        for player_id in range(len(state_shape)):
            agent = DMCAgent(state_shape[player_id],
                             action_shape[player_id],
                             mlp_layers,
                             exp_epsilon,
                             device)
            self.agents.append(agent)

    def share_memory(self):
        for agent in self.agents:
            agent.share_memory()

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def parameters(self, index):
        return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
    
class RuleModel:
    def __init__(self, num_player):
        self.agents = []
        for i in range(num_player):
            agent = UNORuleAgentV2()
            self.agents.append(agent)

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
    
class MultiModel:
    def __init__(self,
                 state_shape,
                 action_shape,
                 mlp_layers=[512,512,512,512,512],
                 exp_epsilon=0.01,
                 device=0):
        agents = [[None] for _ in range(len(state_shape))]
        agents[0] = DMCAgent(state_shape[0],
                             action_shape[0],
                             mlp_layers,
                             exp_epsilon,
                             device)
        agents[1] = UNORuleAgentV2()
        agents[2] = DMCAgent(state_shape[2],
                             action_shape[2],
                             mlp_layers,
                             exp_epsilon,
                             device)
        agents[3] = UNORuleAgentV2()
        
        self.agents = agents
        
    def share_memory(self):
        self.agents[0].share_memory()
        self.agents[2].share_memory()

    def eval(self):
        self.agents[0].eval()
        self.agents[2].eval()

    def parameters(self, index):
        if index in [0,2]:
            return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents

