"""
Phase 1 and Phase 2 trainers.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from torch.optim import Adam

from .networks import StateProposer, ContextEncoder, Policy
from .grpo_ppo import GRPOPPOTrainer
from .rollout import (
    collect_batch_phase1,
    collect_batch_phase2,
    collect_batch_vectorized_phase1,
    collect_batch_vectorized_phase2,
    collect_batch_parallel_phase1,
    collect_batch_parallel_phase2,
    compute_trajectory_stats,
    HAS_PARALLEL_EXECUTOR,
)


class Phase1Trainer:
    """
    Phase 1: Train on single initial environment.

    Trains: theta (state proposer) + psi (policy)
    Context: c_t = 0 (not used)
    Loss: L_ppo + lambda * ||X_t - s_{t+1}||²
    """

    def __init__(
        self,
        env,
        theta: StateProposer,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
        env_fn: callable = None,
    ):
        self.env = env
        self.env_fn = env_fn  # For vectorized rollout
        self.theta = theta
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        # Move networks to device
        self.theta.to(self.device)
        self.psi.to(self.device)

        # Optimizer for both theta and psi
        self.optimizer = Adam(
            list(self.theta.parameters()) + list(self.psi.parameters()),
            lr=config.get('lr', 3e-4),
        )

        # GRPO-PPO trainer
        self.trainer = GRPOPPOTrainer(
            policy=self.psi,
            theta=self.theta,
            phi=None,
            optimizer=self.optimizer,
            gamma=config.get('gamma', 0.99),
            clip_eps=config.get('clip_eps', 0.2),
            ppo_epochs=config.get('ppo_epochs', 10),
            minibatch_size=config.get('minibatch_size', 64),
            lambda_theta=config.get('lambda_theta', 1.0),
            entropy_coef=config.get('entropy_coef', 0.0),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            device=self.device,
        )

        # Logging
        self.log_history = []

    def train(self, n_iterations: int = None, save_dir: str = None):
        """
        Run Phase 1 training.

        Args:
            n_iterations: number of training iterations
            save_dir: directory to save checkpoints

        Returns:
            theta, psi: trained networks
        """
        n_iterations = n_iterations or self.config.get('phase1_iterations', 500)
        batch_size = self.config.get('batch_size', 20)
        max_path_length = self.config.get('max_path_length', 1000)
        save_every = self.config.get('save_every', 50)
        context_dim = self.config.get('context_dim', 16)
        vectorized = self.config.get('vectorized', False)
        parallel = self.config.get('parallel', False)
        n_parallel = self.config.get('n_parallel', 5)

        print(f"Phase 1: Training for {n_iterations} iterations")
        print(f"  Batch size: {batch_size} trajectories")
        print(f"  Max path length: {max_path_length}")
        print(f"  Vectorized: {vectorized}, Parallel: {parallel}")

        for iteration in range(n_iterations):
            # Collect trajectories
            self.theta.eval()
            self.psi.eval()

            if parallel and HAS_PARALLEL_EXECUTOR:
                trajectories = collect_batch_parallel_phase1(
                    env=self.env,
                    theta=self.theta,
                    psi=self.psi,
                    context_dim=context_dim,
                    n_trajectories=batch_size,
                    max_steps=max_path_length,
                    n_parallel=n_parallel,
                    device=self.device,
                )
            elif vectorized and self.env_fn is not None:
                trajectories = collect_batch_vectorized_phase1(
                    env_fn=self.env_fn,
                    theta=self.theta,
                    psi=self.psi,
                    context_dim=context_dim,
                    n_trajectories=batch_size,
                    max_steps=max_path_length,
                    device=self.device,
                )
            else:
                trajectories = collect_batch_phase1(
                    env=self.env,
                    theta=self.theta,
                    psi=self.psi,
                    context_dim=context_dim,
                    n_trajectories=batch_size,
                    max_steps=max_path_length,
                    device=self.device,
                )

            # Compute trajectory stats
            traj_stats = compute_trajectory_stats(trajectories)

            # Update networks
            self.theta.train()
            self.psi.train()

            update_info = self.trainer.update(trajectories, phase=1)

            # Combine stats
            log_entry = {
                'iteration': iteration,
                **traj_stats,
                **update_info,
            }
            self.log_history.append(log_entry)

            # Print progress
            if iteration % 10 == 0:
                print(f"  Iter {iteration:4d} | "
                      f"Reward: {traj_stats['mean_episode_reward']:7.2f} | "
                      f"PPO Loss: {update_info['ppo_loss']:.4f} | "
                      f"Theta Loss: {update_info.get('theta_loss', 0):.4f}")

            # Save checkpoint
            if save_dir and (iteration + 1) % save_every == 0:
                self.save_checkpoint(save_dir, iteration + 1)

        # Final save
        if save_dir:
            self.save_checkpoint(save_dir, 'final')
            self.save_logs(save_dir)

        return self.theta, self.psi

    def save_checkpoint(self, save_dir: str, tag):
        """Save model checkpoint."""
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        path = os.path.join(save_dir, 'checkpoints', f'phase1_{tag}.pt')
        torch.save({
            'theta': self.theta.state_dict(),
            'psi': self.psi.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def save_logs(self, save_dir: str):
        """Save training logs."""
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        path = os.path.join(save_dir, 'logs', 'phase1_train.json')
        with open(path, 'w') as f:
            json.dump(self.log_history, f, indent=2)


class Phase2Trainer:
    """
    Phase 2: Joint training across training environments.

    Trains: phi (context encoder) + psi (policy)
    Frozen: theta (state proposer)
    Context: c_t = f_phi(c_{t-1}, X_{t-1}, s_{t-1})
    Loss: L_ppo (GRPO advantages)
    """

    def __init__(
        self,
        train_envs: List,
        theta: StateProposer,
        phi: ContextEncoder,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
        env_fns: List = None,
    ):
        self.train_envs = train_envs
        self.env_fns = env_fns  # For vectorized rollout
        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        # Move networks to device
        self.theta.to(self.device)
        self.phi.to(self.device)
        self.psi.to(self.device)

        # Freeze theta
        self.theta.eval()
        for p in self.theta.parameters():
            p.requires_grad = False

        # Optimizer for phi and psi only
        self.optimizer = Adam(
            list(self.phi.parameters()) + list(self.psi.parameters()),
            lr=config.get('lr', 3e-4),
        )

        # GRPO-PPO trainer
        self.trainer = GRPOPPOTrainer(
            policy=self.psi,
            theta=None,  # theta is frozen
            phi=self.phi,
            optimizer=self.optimizer,
            gamma=config.get('gamma', 0.99),
            clip_eps=config.get('clip_eps', 0.2),
            ppo_epochs=config.get('ppo_epochs', 10),
            minibatch_size=config.get('minibatch_size', 64),
            lambda_theta=0.0,  # no theta loss in phase 2
            entropy_coef=config.get('entropy_coef', 0.0),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            device=self.device,
        )

        # Logging
        self.log_history = []

    def train(self, n_iterations: int = None, save_dir: str = None):
        """
        Run Phase 2 training.

        Args:
            n_iterations: number of training iterations
            save_dir: directory to save checkpoints

        Returns:
            phi, psi: trained networks
        """
        n_iterations = n_iterations or self.config.get('phase2_iterations', 500)
        n_traj_per_env = self.config.get('n_traj_per_env', 5)
        max_path_length = self.config.get('max_path_length', 1000)
        save_every = self.config.get('save_every', 50)
        vectorized = self.config.get('vectorized', False)
        parallel = self.config.get('parallel', False)
        n_parallel = self.config.get('n_parallel', 5)

        print(f"Phase 2: Training for {n_iterations} iterations")
        print(f"  Training envs: {len(self.train_envs)}")
        print(f"  Trajectories per env: {n_traj_per_env}")
        print(f"  Vectorized: {vectorized}, Parallel: {parallel}")

        for iteration in range(n_iterations):
            # Collect trajectories from all training envs
            self.phi.eval()
            self.psi.eval()

            if parallel and HAS_PARALLEL_EXECUTOR:
                # Collect from each env type using parallel executor
                trajectories = []
                for env in self.train_envs:
                    env_trajs = collect_batch_parallel_phase2(
                        env=env,
                        theta=self.theta,
                        phi=self.phi,
                        psi=self.psi,
                        n_trajectories=n_traj_per_env,
                        max_steps=max_path_length,
                        n_parallel=min(n_parallel, n_traj_per_env),
                        device=self.device,
                    )
                    trajectories.extend(env_trajs)
            elif vectorized and self.env_fns is not None:
                trajectories = collect_batch_vectorized_phase2(
                    env_fns=self.env_fns,
                    theta=self.theta,
                    phi=self.phi,
                    psi=self.psi,
                    n_trajectories_per_env=n_traj_per_env,
                    max_steps=max_path_length,
                    device=self.device,
                )
            else:
                trajectories = collect_batch_phase2(
                    envs=self.train_envs,
                    theta=self.theta,
                    phi=self.phi,
                    psi=self.psi,
                    n_trajectories_per_env=n_traj_per_env,
                    max_steps=max_path_length,
                    device=self.device,
                )

            # Compute trajectory stats
            traj_stats = compute_trajectory_stats(trajectories)

            # Update networks
            self.phi.train()
            self.psi.train()

            update_info = self.trainer.update(trajectories, phase=2)

            # Combine stats
            log_entry = {
                'iteration': iteration,
                **traj_stats,
                **update_info,
            }
            self.log_history.append(log_entry)

            # Print progress
            if iteration % 10 == 0:
                print(f"  Iter {iteration:4d} | "
                      f"Reward: {traj_stats['mean_episode_reward']:7.2f} | "
                      f"PPO Loss: {update_info['ppo_loss']:.4f} | "
                      f"KL: {update_info['approx_kl']:.4f}")

            # Save checkpoint
            if save_dir and (iteration + 1) % save_every == 0:
                self.save_checkpoint(save_dir, iteration + 1)

        # Final save
        if save_dir:
            self.save_checkpoint(save_dir, 'final')
            self.save_logs(save_dir)

        return self.phi, self.psi

    def save_checkpoint(self, save_dir: str, tag):
        """Save model checkpoint."""
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        path = os.path.join(save_dir, 'checkpoints', f'phase2_{tag}.pt')
        torch.save({
            'theta': self.theta.state_dict(),
            'phi': self.phi.state_dict(),
            'psi': self.psi.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def save_logs(self, save_dir: str):
        """Save training logs."""
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        path = os.path.join(save_dir, 'logs', 'phase2_train.json')
        with open(path, 'w') as f:
            json.dump(self.log_history, f, indent=2)


def load_phase1_checkpoint(
    path: str,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 16,
    device: torch.device = None,
):
    """
    Load Phase 1 checkpoint.

    Returns:
        theta, psi: loaded networks
    """
    theta = StateProposer(obs_dim, hidden_dim)
    psi = Policy(obs_dim, act_dim, context_dim, hidden_dim)

    checkpoint = torch.load(path, map_location=device)
    theta.load_state_dict(checkpoint['theta'])
    psi.load_state_dict(checkpoint['psi'])

    if device:
        theta.to(device)
        psi.to(device)

    return theta, psi


def load_phase2_checkpoint(
    path: str,
    obs_dim: int,
    act_dim: int,
    hidden_dim: int = 64,
    context_dim: int = 16,
    device: torch.device = None,
):
    """
    Load Phase 2 checkpoint.

    Returns:
        theta, phi, psi: loaded networks
    """
    theta = StateProposer(obs_dim, hidden_dim)
    phi = ContextEncoder(obs_dim, context_dim)
    psi = Policy(obs_dim, act_dim, context_dim, hidden_dim)

    checkpoint = torch.load(path, map_location=device)
    theta.load_state_dict(checkpoint['theta'])
    phi.load_state_dict(checkpoint['phi'])
    psi.load_state_dict(checkpoint['psi'])

    if device:
        theta.to(device)
        phi.to(device)
        psi.to(device)

    return theta, phi, psi
