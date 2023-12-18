''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import rlcard
from rlcard.utils import get_device, set_seed, tournament

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)  # type: ignore
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position] # 'uno-rule-v1'
    
    return agent

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed, 'num_cards': args.num_cards})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)
    if args.num_cards > 0:
        print('0,2', env.count1 / args.num_games)
        print('1,3', env.count2 / args.num_games)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--models', nargs='*', default=['experiments/uno/dmc/v4.0.0/0_2000518400.pth', 'experiments/uno/dmc/v4.0.0/1_2000518400.pth', 'experiments/uno/dmc/v4.0.0/2_2000518400.pth', 'experiments/uno/dmc/v4.0.0/3_2000518400.pth'])
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_cards', type=int, default=0)
    parser.add_argument('--num_games', type=int, default=10000)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

