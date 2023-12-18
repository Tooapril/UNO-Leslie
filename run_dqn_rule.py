''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard import models
from rlcard.agents import RandomAgent, DQNAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})
    
    # Ensure the player position
    team_one = [args.position, (args.position + 2) % env.num_players]
    team_two = [(args.position - 1) % env.num_players, (args.position + 1) % env.num_players]

    # Initialize the agent and use random agents as opponents —— 训练时与随机产生数据
    agents = [[None] for _ in range(env.num_players)]
    for player_id in range(env.num_players):
        if player_id in team_one:
            agents[player_id] = DQNAgent(
                                    num_actions=env.num_actions,
                                    state_shape=env.state_shape[player_id],
                                    mlp_layers=[512,512,512,512,512],
                                    train_every=args.train_every,
                                    device=device,
                                )
        elif player_id in team_two:
            agents[player_id] = models.load('uno-rule-v2').agents[player_id]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir, env.num_players) as logger:
        for episode in range(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any) —— 将 position 位置上的 trajectories 一个一个传入 Memory Buffer
            for player_id in range(env.num_players):
                if player_id in team_one:
                    for ts in trajectories[player_id]:
                        agents[player_id].feed(ts)
                
            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                # 评估时与基于规则的比较       
                logger.log_performance(env.timestep, tournament(env, args.num_eval_games))

        # Get the paths
        csv_path_list, fig_path_list = logger.csv_path_list, logger.fig_path_list

    # Plot the learning curve
    for i in range(logger.num_players):
        plot_curve(csv_path_list[i], fig_path_list[i], args.algorithm, i)

    # Save model
    for i in range(env.num_players):
        save_path = os.path.join(args.log_dir, f'model_{i}.pth')
        torch.save(agents[i], save_path)
        print('Model %d saved in', i, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--env', type=str, default='uno')
    parser.add_argument('--algorithm', type=str, default='dqn')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', type=int, default=0)
    parser.add_argument('--train_every', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--num_eval_games', type=int, default=10000)
    parser.add_argument('--evaluate_every', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='experiments/uno/dqn/')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

