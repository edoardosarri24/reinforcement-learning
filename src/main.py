import argparse
import wandb
import gymnasium
from myModel import PolicyNet
from reinforce import reinforce
from core import run_episode
from sweep import do_sweep

def parse_args():
    'The argument parser for the main training script.'
    parser = argparse.ArgumentParser(description='A script implementing REINFORCE on the Cartpole environment.')
    parser.add_argument('--env',
                        type=str,
                        choices=['cart_pole', 'lunar_lander'],
                        help='Environment to solve.')
    parser.add_argument('--run_name',
                        type=str,
                        default='test',
                        help='Wandb run name to log to.')
    parser.add_argument('--episodes',
                        type=int,
                        default=500,
                        help='Number of training episodes')
    parser.add_argument('--baseline',
                        type=str,
                        default='none',
                        choices=['none', 'std', 'stateValue'],
                        help='Baseline to use (none, std, stateValue)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.99,
                        help='Discount factor for future rewards')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='Learning rate')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        choices=['cpu', 'mps', 'cuda'],
                        help='device for computiing.')
    parser.add_argument('--visualize',
                        action='store_true',
                        help='Visualize final agent')
    parser.add_argument('--sweep',
                        action='store_true',
                        help='Make only sweep')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.env == 'cart_pole':
        env_name = 'CartPole-v1'
        project_name = 'Cart Pole'
    elif args.env == 'lunar_lander':
        env_name = 'LunarLander-v3'
        project_name = 'Lunar Lander'

    if args.sweep:
        do_sweep(args, env_name, project_name, 0.9, 1e-5)
        do_sweep(args, env_name, project_name, 0.95, 1e-4)
    
    else:
        wandb.init(
            project=project_name,
            config={
                'learning_rate': args.lr,
                'baseline': args.baseline,
                'gamma': args.gamma,
                'episodes': args.episodes
            },
            name = args.run_name
        )

        env = gymnasium.make(env_name)
        layers = [env.observation_space.shape[0],
                  32, 64, 128, 256, 256, 128, 64, 32,
                  env.action_space.n]
        policy = PolicyNet(layers)
        policy = reinforce(policy, env, args.gamma, args.lr, args.baseline, args.episodes, args.device, layers)
        if args.visualize:
            env_render = gymnasium.make(env_name, render_mode='human')
            for _ in range(10):
                run_episode(env_render, policy, args.device)
            env_render.close()

        env.close()
        wandb.finish()