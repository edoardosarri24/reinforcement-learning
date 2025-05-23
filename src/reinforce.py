import time
import torch
import wandb
from tqdm import tqdm
from core import run_episode, compute_returns
from utility import *
from myModel import StateValueNet
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim import Adam

def reinforce(policy, env, gamma, lr, baseline, episodes, device, layers):
    '''
        An implementation of the REINFORCE policy gradient algorithm.
        Checkpoints best model at each iteration to the wandb run directory.
        Args:
            policy: The policy network to be trained.
            env: The environment in which the agent operates.
            run: An object that handles logging and running episodes.
            gamma: The discount factor for future rewards.
            lr: Learning rate for the optimizer.
            baseline: The type of baseline to use.
            num_episodes: The number of episodes to train the policy.
        Returns:
            running_rewards: A list of running rewards over episodes.
    '''
    start_time = time.time()
    policy.to(device)
    stateValue_Net = StateValueNet(layers).to(device)
    opt_policy = Adam(policy.parameters(), lr=lr)
    opt_valueState = Adam(stateValue_Net.parameters(), lr=lr)
    scheduler_policy = StepLR(opt_policy, step_size=1500, gamma=0.9)
    scheduler_value = StepLR(opt_valueState, step_size=1500, gamma=0.9)
    best_returns = 0.0
    best_episode = 0

    policy.train()
    for episode in tqdm(range(episodes)):
        log_probs, rewards, observations = run_episode(env, policy, device)
        returns = compute_returns(rewards, gamma, device)

        # checkpoint
        if sum(rewards) > best_returns:
            save_checkpoint(episode, policy, wandb.run.dir)
            best_returns = sum(rewards)
            best_episode = episode

        #baseline
        if baseline == 'none':
            base_returns = returns
        elif baseline == 'std':
            base_returns = (returns - returns.mean()) / returns.std()
        elif baseline == 'stateValue':
            state_values = stateValue_Net(observations).squeeze()
            base_returns = returns - state_values.detach()
        
        # optimization
            opt_valueState.zero_grad()
            value_loss = torch.nn.functional.smooth_l1_loss(returns, state_values) # più stabile rispetto alla MSE: unisce MSE e MAE
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(stateValue_Net.parameters(), max_norm=2.0)
            opt_valueState.step()
            scheduler_value.step()
        opt_policy.zero_grad()
        policy_loss = -(log_probs * base_returns).mean()
        policy_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=2.0)
        opt_policy.step()
        scheduler_policy.step()

        #log
        wandb.log({
            'policy_loss': policy_loss.item(),
            'episode_returns': sum(rewards),
            'train_time': time.time()-start_time,
            'lr_policy': scheduler_policy.get_last_lr()[0],
            'lr_value': scheduler_value.get_last_lr()[0],
            'norm': norm})

    # restore
    policy = restore_checkpoint(wandb.run.dir, best_episode, policy, device)
    return policy.eval()