''' An example of evluating the trained models in RLCard
'''
import argparse
import os

import rlcard
from rlcard.agents import CFRAgent, DQNAgent, NFSPAgent, RandomAgent
from rlcard.utils import get_device, set_seed, tournament


def load_model(model_path, env, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    # elif os.path.isdir(model_path):  # CFR model ❓
    #     from rlcard.agents import CFRAgent
    #     agent = CFRAgent(env, model_path)
    #     agent.load()
    elif model_path == 'random':  # Random model
        agent = RandomAgent(num_actions=env.num_actions)
    # elif model_path == 'dqn': # 需要训练
    #     agent = DQNAgent(num_actions=env.num_actions, 
    #                      state_shape=env.state_shape[position], 
    #                      mlp_layers=[512,512,512,512,512],
    #                      device=device)
    # elif model_path == 'nfsp': # 需要训练 
    #     agent = NFSPAgent(num_actions=env.num_actions, 
    #                       state_shape=env.state_shape[position], 
    #                       hidden_layers_sizes=[512,512,512,512,512],
    #                       q_mlp_layers=[512,512,512,512,512],
    #                       device=device)
    # elif model_path == 'cfr': # 需要训练
    #     agent = CFRAgent(env=env) # ❓
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--models', nargs='*', default=['experiments/uno/dmc/v3.1.0/0_2003497600.pth', 'uno-rule-v1', 'experiments/uno/dmc/v3.1.0/2_2003497600.pth', 'uno-rule-v1'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_games', type=int, default=100000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

