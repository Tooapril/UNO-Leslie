''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def train(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    teammate = (args.position + 2) % env.num_players
    opponent_left = (args.position - 1) % env.num_players
    opponent_right = (args.position + 1) % env.num_players
    
    agents = [[None] for _ in range(env.num_players)]
    agents[args.position] = DQNAgent(   # type: ignore
                                num_actions=env.num_actions,
                                state_shape=env.state_shape[args.position],
                                mlp_layers=[512,512,512,512,512],
                                device=device,
                            )
    agents[teammate] =  DQNAgent(  # type: ignore
                            num_actions=env.num_actions,
                            state_shape=env.state_shape[args.position],
                            mlp_layers=[512,512,512,512,512],
                            device=device,
                        )
    agents[opponent_left] = RandomAgent(num_actions=env.num_actions)  # type: ignore
    agents[opponent_right] = RandomAgent(num_actions=env.num_actions)  # type: ignore
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            for ts in trajectories[args.position]:
                agent.feed(ts)  # type: ignore
                
            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[args.position])

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm, args.position)  # type: ignore

    # Save model
    save_path = os.path.join(args.log_dir, 'model.pth')
    torch.save(agent, save_path)  # type: ignore
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN example in RLCard")
    parser.add_argument('--env', type=str, default='uno',
            choices=['blackjack', 'leduc-holdem', 'limit-holdem', 'doudizhu', 'mahjong', 'no-limit-holdem', 'uno', 'gin-rummy'])
    parser.add_argument('--algorithm', type=str, default='dqn')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--position', type=int, default=1)
    parser.add_argument('--num_episodes', type=int, default=100000)
    parser.add_argument('--num_eval_games', type=int, default=10000)
    parser.add_argument('--evaluate_every', type=int, default=2000)
    parser.add_argument('--log_dir', type=str, default='experiments/uno/dqn/')
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

