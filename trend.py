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
    left_opponent = (args.position - 1) % env.num_players
    right_opponent = (args.position + 1) % env.num_players
    
    # Identify model file
    # 加载位置 0, 2 的模型文件名
    if os.path.exists(args.log_dir1):
        x1 = [f for f in os.listdir(args.log_dir1)
                    if os.path.isfile(os.path.join(args.log_dir1, f)) and f.startswith(str(player) + "_")] # 获取日志文件下所有 “0_” 开头的
        x1.sort(key=lambda z:int(z.split('.')[0])) # 将所有 position 位的日志文件排序
        
        x2 = [f for f in os.listdir(args.log_dir1)
                    if os.path.isfile(os.path.join(args.log_dir1, f)) and f.startswith(str(teammate) + "_")] # 获取日志文件下所有 “2_” 开头的
        x2.sort(key=lambda z:int(z.split('.')[0])) # 将所有 position 位的日志文件排序
    
    # 加载位置 1, 3 的模型文件名
    if os.path.exists(args.log_dir2):
        y1 = [f for f in os.listdir(args.log_dir2)
                    if os.path.isfile(os.path.join(args.log_dir2, f)) and f.startswith(str(left_opponent) + "_")] # 获取日志文件下所有 “1_” 开头的
        y1.sort(key=lambda z:int(z.split('.')[0])) # 将所有 position 位的日志文件排序
        
        y2 = [f for f in os.listdir(args.log_dir2)
                    if os.path.isfile(os.path.join(args.log_dir2, f)) and f.startswith(str(right_opponent) + "_")] # 获取日志文件下所有 “3_” 开头的
        y2.sort(key=lambda z:int(z.split('.')[0])) # 将所有 position 位的日志文件排序
    
    with Logger(args.savedir) as logger:
        for index in range(len(x1)): # 以 log_dir1 下的断点文件数为标准
            agents = [[None] for _ in range(env.num_players)]
            if os.path.exists(args.log_dir1) and os.path.exists(args.log_dir2):
                agents[player] = load_model(args.log_dir1 + x1[index], env, device=device)
                agents[teammate] = load_model(args.log_dir1 + x2[index], env, device=device)
                agents[left_opponent] = load_model(args.log_dir2 + y1[index], env, device=device)
                agents[right_opponent] = load_model(args.log_dir2 + y2[index], env, device=device)
            else:
                agents[player] = load_model(args.log_dir1 + x1[index], env, device=device)
                agents[teammate] = load_model(args.log_dir1 + x2[index], env, device=device)
                agents[left_opponent] = load_model('uno-rule-v2', env, position=1, device=device)
                agents[right_opponent] = load_model('uno-rule-v2', env, position=3, device=device)
            env.set_agents(agents)
            
            logger.log_performance(x1[index][x1[index].rfind('_')+1 : x1[index].rfind('.')], tournament(env, args.num_games)[args.position]) # 获取玩家 0 的胜率存入日志

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm, args.position)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='DMC VS Rule')
    parser.add_argument('--cuda', type=str, default='1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--num_games', type=int, default=10000)
    parser.add_argument('--log_dir1', type=str, default='')
    parser.add_argument('--log_dir2', type=str, default='')
    parser.add_argument('--savedir', type=str, default='experiments/uno/dmc/test/')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

