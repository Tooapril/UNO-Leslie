''' A toy example of playing against rule-based bot on UNO
'''

import os

import torch

import rlcard
from rlcard import models
from rlcard.agents.human_agents.uno_human_agent import (HumanAgent,
                                                        _print_action)
from rlcard.utils import get_device

# use cuda 1
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# get device
device = get_device()

# Make environment
env = rlcard.make('uno')

# Load models
human_agent = HumanAgent(env.num_actions)
dmc_agent1 = torch.load('experiments/uno/dmc/v3.1.0/1_2003497600.pth', map_location=device)
dmc_agent1.set_device(device)
dmc_agent2 = torch.load('experiments/uno/dmc/v3.1.0/2_2003497600.pth', map_location=device)
dmc_agent2.set_device(device)
dmc_agent3 = torch.load('experiments/uno/dmc/v3.1.0/3_2003497600.pth', map_location=device)
dmc_agent3.set_device(device)
# uno_rule_agent = models.load('uno-rule-v1').agents[0]
env.set_agents([human_agent, dmc_agent1, dmc_agent2, dmc_agent3])

print(">> UNO DMC Model")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action
    final_state = trajectories[0][-1]
    action_record = final_state['action_record']
    state = final_state['raw_obs']
    _action_list = []
    for i in range(1, len(action_record)+1):
        if action_record[-i][0] == 0: # 记录非自己操作的 action
            break
        _action_list.insert(0, action_record[-i])
    for pair in _action_list:
        print('>> Player', pair[0], 'chooses ', end='')
        _print_action(pair[1])
        print('')

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('Your team win!')
    else:
        print('Your team lose!')
    print('')
    input("Press any key to continue...")
