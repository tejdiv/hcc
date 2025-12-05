"""
GRPO + PPO training logic.

GRPO: Group Relative Policy Optimization
- No learned value function
- Advantages computed via per-step return normalization across batch

PPO: Proximal Policy Optimization
- Clipped surrogate objective
- Multiple epochs per batch
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns for a single trajectory.

    G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...

    Args:
        rewards: list of rewards [r_0, r_1, ..., r_T]
        gamma: discount factor

    Returns:
        returns: list of returns [G_0, G_1, ..., G_T]
    """
    T = len(rewards)
    returns = [0.0] * T
    G = 0.0

    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns


def compute_advantages_grpo(trajectories: List[List[Dict]], gamma: float = 0.99):
    """
    GRPO per-step advantage computation.

    1. Compute G_t for every timestep in every trajectory
    2. Normalize across ALL timesteps from ALL trajectories

    Args:
        trajectories: list of trajectories, each trajectory is a list of step dicts
        gamma: discount factor

    Returns:
        trajectories: same structure with 'return' and 'advantage' added to each step
    """
    all_returns = []

    # Step 1: Compute returns for every timestep
    for traj in trajectories:
        rewards = [step['reward'] for step in traj]
        returns = compute_returns(rewards, gamma)

        for t, G_t in enumerate(returns):
            traj[t]['return'] = G_t
            all_returns.append(G_t)

    # Step 2: Normalize across ALL timesteps
    all_returns = np.array(all_returns)
    mean_G = np.mean(all_returns)
    std_G = np.std(all_returns) + 1e-8

    for traj in trajectories:
        for step in traj:
            step['advantage'] = (step['return'] - mean_G) / std_G

    return trajectories


def flatten_trajectories(trajectories: List[List[Dict]]) -> List[Dict]:
    """Flatten list of trajectories into single list of steps."""
    flat = []
    for traj in trajectories:
        flat.extend(traj)
    return flat


def create_batch_tensors(steps: List[Dict], device: torch.device):
    """
    Convert list of step dicts to batched tensors.

    Args:
        steps: list of step dicts with keys:
            s, X, c, a, log_prob, reward, s_next, advantage
        device: torch device

    Returns:
        dict of batched tensors
    """
    batch = {
        's': torch.stack([torch.as_tensor(step['s'], dtype=torch.float32) for step in steps]).to(device),
        'X': torch.stack([torch.as_tensor(step['X'], dtype=torch.float32) for step in steps]).to(device),
        'c': torch.stack([torch.as_tensor(step['c'], dtype=torch.float32) for step in steps]).to(device),
        'a': torch.stack([torch.as_tensor(step['a'], dtype=torch.float32) for step in steps]).to(device),
        'old_log_prob': torch.tensor([step['log_prob'] for step in steps], dtype=torch.float32).to(device),
        'advantage': torch.tensor([step['advantage'] for step in steps], dtype=torch.float32).to(device),
        's_next': torch.stack([torch.as_tensor(step['s_next'], dtype=torch.float32) for step in steps]).to(device),
    }
    return batch


def ppo_loss(batch: Dict[str, torch.Tensor], policy, clip_eps: float = 0.2):
    """
    Compute clipped PPO loss.

    L = -E[min(ratio * A, clip(ratio) * A)]

    Args:
        batch: dict with s, X, c, a, old_log_prob, advantage
        policy: Policy network
        clip_eps: clipping epsilon

    Returns:
        loss: scalar tensor
        info: dict with diagnostics
    """
    s = batch['s']
    X = batch['X']
    c = batch['c']
    a = batch['a']
    old_log_prob = batch['old_log_prob']
    advantage = batch['advantage']

    # Current log prob
    new_log_prob = policy.log_prob(s, X, c, a)

    # Ratio
    ratio = torch.exp(new_log_prob - old_log_prob)

    # Clipped objective
    L1 = ratio * advantage
    L2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage

    # Take min (pessimistic bound)
    loss = -torch.mean(torch.min(L1, L2))

    # Diagnostics
    with torch.no_grad():
        approx_kl = (old_log_prob - new_log_prob).mean().item()
        clip_fraction = ((ratio - 1).abs() > clip_eps).float().mean().item()

    info = {
        'ppo_loss': loss.item(),
        'approx_kl': approx_kl,
        'clip_fraction': clip_fraction,
        'ratio_mean': ratio.mean().item(),
        'ratio_std': ratio.std().item(),
    }

    return loss, info


def state_prediction_loss(batch: Dict[str, torch.Tensor]):
    """
    Compute state prediction loss: ||X_t - s_{t+1}||²

    Args:
        batch: dict with X (predicted) and s_next (actual)

    Returns:
        loss: scalar tensor
    """
    X = batch['X']
    s_next = batch['s_next']

    loss = torch.mean((X - s_next) ** 2)
    return loss


def entropy_bonus(batch: Dict[str, torch.Tensor], policy):
    """
    Compute entropy bonus for exploration.

    Args:
        batch: dict with s, X, c
        policy: Policy network

    Returns:
        entropy: scalar tensor (higher = more exploration)
    """
    s = batch['s']
    X = batch['X']
    c = batch['c']

    dist = policy.get_distribution(s, X, c)
    entropy = dist.entropy().mean()

    return entropy


class GRPOPPOTrainer:
    """
    Trainer class combining GRPO advantage computation with PPO updates.
    """

    def __init__(
        self,
        policy,
        theta,
        phi,
        optimizer,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
        lambda_theta: float = 1.0,
        entropy_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        device: torch.device = None,
    ):
        """
        Args:
            policy: Policy network (psi)
            theta: StateProposer network (can be None if frozen)
            phi: ContextEncoder network (can be None in Phase 1)
            optimizer: torch optimizer
            gamma: discount factor
            clip_eps: PPO clipping epsilon
            ppo_epochs: number of PPO epochs per update
            minibatch_size: size of minibatches within PPO epochs
            lambda_theta: weight for state prediction loss
            entropy_coef: entropy bonus coefficient
            max_grad_norm: gradient clipping norm
            device: torch device
        """
        self.policy = policy
        self.theta = theta
        self.phi = phi
        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.lambda_theta = lambda_theta
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device('cpu')

    def update(self, trajectories: List[List[Dict]], phase: int = 1):
        """
        Perform one update using collected trajectories.

        Args:
            trajectories: list of trajectories
            phase: 1 or 2 (determines which losses to use)

        Returns:
            info: dict with training metrics
        """
        # Compute GRPO advantages
        trajectories = compute_advantages_grpo(trajectories, self.gamma)

        # Flatten to steps
        steps = flatten_trajectories(trajectories)

        # Convert to tensors
        batch = create_batch_tensors(steps, self.device)

        # Track metrics
        all_info = {
            'ppo_loss': [],
            'theta_loss': [],
            'entropy': [],
            'approx_kl': [],
            'clip_fraction': [],
        }

        # PPO epochs
        n_samples = len(steps)
        indices = np.arange(n_samples)

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.minibatch_size):
                end = min(start + self.minibatch_size, n_samples)
                mb_indices = indices[start:end]

                # Create minibatch
                mb = {k: v[mb_indices] for k, v in batch.items()}

                # PPO loss
                loss_ppo, info = ppo_loss(mb, self.policy, self.clip_eps)
                total_loss = loss_ppo

                # State prediction loss (Phase 1 only)
                if phase == 1 and self.theta is not None:
                    loss_theta = state_prediction_loss(mb)
                    total_loss = total_loss + self.lambda_theta * loss_theta
                    all_info['theta_loss'].append(loss_theta.item())

                # Entropy bonus
                if self.entropy_coef > 0:
                    ent = entropy_bonus(mb, self.policy)
                    total_loss = total_loss - self.entropy_coef * ent
                    all_info['entropy'].append(ent.item())

                # Gradient update
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                if self.max_grad_norm > 0:
                    params = list(self.policy.parameters())
                    if self.theta is not None and phase == 1:
                        params += list(self.theta.parameters())
                    if self.phi is not None and phase == 2:
                        params += list(self.phi.parameters())
                    torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

                self.optimizer.step()

                all_info['ppo_loss'].append(info['ppo_loss'])
                all_info['approx_kl'].append(info['approx_kl'])
                all_info['clip_fraction'].append(info['clip_fraction'])

        # Aggregate metrics
        result = {
            'ppo_loss': np.mean(all_info['ppo_loss']),
            'approx_kl': np.mean(all_info['approx_kl']),
            'clip_fraction': np.mean(all_info['clip_fraction']),
        }

        if all_info['theta_loss']:
            result['theta_loss'] = np.mean(all_info['theta_loss'])
        if all_info['entropy']:
            result['entropy'] = np.mean(all_info['entropy'])

        # Add trajectory stats
        all_rewards = [step['reward'] for step in steps]
        all_returns = [step['return'] for step in steps]
        result['mean_reward'] = np.mean(all_rewards)
        result['mean_return'] = np.mean(all_returns)
        result['std_return'] = np.std(all_returns)

        return result
