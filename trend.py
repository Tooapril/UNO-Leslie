''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import torch
import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, set_seed, tournament, Logger, plot_curve

def load_model(model_path, env, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
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
    
    # Identify model file
    x = [f for f in os.listdir(args.log_dir)
                if os.path.isfile(os.path.join(args.log_dir, f)) and f.startswith(str(args.position) + "_")] # 获取日志文件下所有 “0_” 开头的
    x.sort(key=lambda x:int(x.split('.')[0])) # 将所有 0 号位的日志文件排序
    
    with Logger(args.savedir) as logger:
        for k, v in enumerate(x): # type: ignore
            # Load models
            agents = [[None] for _ in range(env.num_players)]
            agents[args.position] = load_model(args.log_dir + v, env, device=device)  # type: ignore
            agents[1 - args.position] = load_model("random", env, device=device)  # type: ignore
            env.set_agents(agents)
            
            # Evaluate the performance. Play with random agents.
            if k % args.evaluate_every == 0:
                logger.log_performance(v[v.rfind('_')+1:v.rfind('.')], tournament(env, args.num_games)[args.position]) # 获取玩家 0 的胜率存入日志

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm, args.position)  # type: ignore

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dmc')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--num_games', type=int, default=100000)
    parser.add_argument('--evaluate_every', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='experiments/uno/dmc/v3.1.0/')
    parser.add_argument('--savedir', type=str, default='experiments/uno/dmc/test/')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

