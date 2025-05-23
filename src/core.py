import torch
from torch.distributions import Categorical

def select_action(obs, policy):
    '''
        Given an observation, sample from the policy's action distribution.
        Parameters:
            obs (torch.Tensor): The observation input.
            policy (nn.Module): The policy network.
        Returns:
            tuple: A tuple containing:
                - action (int): The selected action
                - log_prob (torch.Tensor): The log probability (1D tensor).
    '''
    distr = Categorical(policy(obs))
    action = distr.sample()
    log_prob = distr.log_prob(action)
    return action.item(), log_prob.unsqueeze(0)

def compute_returns(rewards, gamma, device):
    '''
        Compute the discounted returns for a sequence of rewards.
        Parameters:
        rewards (list): List of rewards.
        gamma (float): Discount factor.
        device (str): Device to use for tensor operations.
        Returns:
        torch.Tensor: Tensor of discounted returns.
    '''
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32, device=device)

def run_episode(env, policy, device, maxlen=500):
    '''
        Run an episode in the given environment using the specified policy up to a maximum number of steps.
        Parameters:
        env: The environment to run the episode in.
        policy: The policy used to select actions.
        maxlen (int): The maximum number of steps to run in the episode. Default is 500.
        Returns:
        tuple: A tuple containing:
            - observations (list): A list of observations throughout the episode.
            - actions (list): A list of actions taken during the episode.
            - log_probs (torch.Tensor): A tensor containing the log probabilities of the actions taken.
            - rewards (list): A list of rewards received at each step.
    '''
    log_probs = []
    rewards = []
    observations = []
    obs, _ = env.reset()
    term = False
    trunc = False

    while not term and not trunc:
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        action, log_prob = select_action(obs, policy)
        log_probs.append(log_prob)
        observations.append(obs)
        obs, reward, term, trunc, _  = env.step(action)
        rewards.append(reward)
    log_probs = torch.cat(log_probs)
    observations = torch.stack(observations).to(device)
    return log_probs, rewards, observations