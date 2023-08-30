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
    
    # Ensure the player position
    player = args.position
    teammate = (args.position + 2) % env.num_players
    opponent_left = (args.position - 1) % env.num_players
    opponent_right = (args.position + 1) % env.num_players
    
    # Identify model file
    y = [f for f in os.listdir(args.log_dir)
                if os.path.isfile(os.path.join(args.log_dir, f)) and f.startswith(str(player) + "_")] # 获取日志文件下所有玩家位开头的
    y = [x.split('_')[1] for x in y] # 仅取断点数保存
    y.sort(key=lambda x:int(x.split('.')[0])) # 按数字顺序排序
    
    with Logger(args.savedir) as logger:
        for k, v in enumerate(y): # type: ignore
            # Load models
            agents = [[None] for _ in range(env.num_players)]
            agents[player] = load_model(args.log_dir + str(player) + '_' + v, env, device=device)  # type: ignore
            agents[teammate] = load_model(args.log_dir + str(teammate) + '_' + v, env, device=device)  # type: ignore
            agents[opponent_left] = load_model("random", env, device=device)  # type: ignore
            agents[opponent_right] = load_model("random", env, device=device)  # type: ignore
            env.set_agents(agents)
            
            # Evaluate the performance. Play with random agents.
            if k % args.evaluate_every == 0:
                logger.log_performance(v[v.rfind('_')+1:v.rfind('.')], tournament(env, args.num_games)[player]) # 获取玩家 0 的胜率存入日志

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm, player)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dmc')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--num_games', type=int, default=10000)
    parser.add_argument('--evaluate_every', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='experiments/uno/dmc/')
    parser.add_argument('--savedir', type=str, default='experiments/uno/dmc/test/')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)
