''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import argparse
import os

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import (Logger, get_device, plot_curve, reorganize, set_seed,
                          tournament)


def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    agents = []
    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        for player_id in range(env.num_players):
            agent = DQNAgent(num_actions=env.num_actions, 
                             state_shape=env.state_shape[player_id], 
                             mlp_layers=[512,512,512,512,512], 
                             device=device)
            agents.append(agent)
    elif args.algorithm == 'nfsp':
        from rlcard.agents import NFSPAgent
        for player_id in range(env.num_players):
            agent = NFSPAgent(num_actions=env.num_actions, 
                              state_shape=env.state_shape[player_id], 
                              hidden_layers_sizes=[512,512,512,512,512], 
                              q_mlp_layers=[512,512,512,512,512], 
                              device=device)
            agents.append(agent)
    
    env.set_agents(agents) # 将对应 agent 初始化到环境中

    # Start training
    for player_id in range(env.num_players): # ❓
        with Logger(args.log_dir) as logger:
            for episode in range(args.num_episodes):

                if args.algorithm == 'nfsp':
                    agents[player_id].sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                # and the other players play randomly (if any)
                for ts in trajectories[player_id]:
                    agents[player_id].feed(ts)

                # Evaluate the performance. Play with random agents.
                if episode % args.evaluate_every == 0:
                    logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[player_id])

            # Get the paths
            csv_path, fig_path = logger.csv_path, logger.fig_path

        # Plot the learning curve —— 命名重合问题❓
        plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    for position in range(env.num_players):
        save_path = os.path.expandvars(os.path.expanduser(
            '%s/%s' % (args.log_dir, str(position) + '_' + str(env.timestep) + '.pth')))
        torch.save(agents[position], save_path)
        print('Model saved in', save_path)
    # save_path = os.path.join(args.log_dir, 'model.pth')
    # torch.save(agent, save_path)
    # print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN/NFSP example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='nfsp', choices=['dqn', 'nfsp'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/blackjack_dqn_result/')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

