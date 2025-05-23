import wandb
import gymnasium
from reinforce import reinforce
from myModel import PolicyNet

def do_sweep(args, env_name, project_name, gamma_min, lr_min):
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'episode_returns',
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {
                'distribution': 'log_uniform_values',
                'min': lr_min,
                'max': 1e-2
            },
            'gamma': {
                'distribution': 'uniform',
                'min': gamma_min,
                'max': 0.999
            }
        }
    }
    
    def mySweep():
        with wandb.init() as run:
            config = wandb.config
            run.name = f'sweep(lr: {round(config.lr, 6)}, gamma: {round(config.gamma, 3)})'
            env = gymnasium.make(env_name)
            layers = [env.observation_space.shape[0], 128, env.action_space.n]
            policy = PolicyNet(layers)
            reinforce(
                policy,
                env,
                config.gamma,
                config.lr,
                'stateValue',
                3000,
                args.device,
                layers
            )
            env.close()

    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    wandb.agent(sweep_id, function=mySweep, count=10)