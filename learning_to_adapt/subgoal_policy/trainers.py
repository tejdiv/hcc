"""
Phase 1 and Phase 2 trainers with GrBAL-style MAML adaptation.

Phase 1: Train simple policy π_ψ(a|s) + world model p̂(s_{t+1}|s_t,a_t) on single env
Phase 2: GrBAL-style MAML training:
    - Collect trajectories into dataset D
    - Sample segments: τ(t-M, t-1) for adaptation, τ(t, t+K) for evaluation
    - Inner loop: adapt φ, ψ on adaptation segment
    - Outer loop: compute loss on evaluation segment with adapted params

Supports:
    - n_parallel: CPU parallelization for env sampling
    - GPU batching for forward passes
"""

import os
import json
import csv
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from torch.optim import Adam
from collections import OrderedDict
import copy
from multiprocessing import Process, Pipe
import pickle

from .networks import (
    WorldModel,
    ContextEncoder,
    Policy,
    clone_params,
    get_params_dict,
    set_params_dict,
)


# ============== Parallel Environment Executor ==============

class ParallelEnvExecutor:
    """
    Executes multiple environments in parallel using multiprocessing.
    Adapted from GrBAL's vectorized_env_executor.py.
    """

    def __init__(self, env, n_parallel, num_rollouts, max_path_length):
        assert num_rollouts % n_parallel == 0
        self.envs_per_proc = int(num_rollouts / n_parallel)
        self._num_envs = n_parallel * self.envs_per_proc
        self.n_parallel = n_parallel
        self.max_path_length = max_path_length

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(target=_env_worker,
                    args=(work_remote, remote, pickle.dumps(env),
                          self.envs_per_proc, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """Execute actions on all envs in parallel."""
        assert len(actions) == self.num_envs
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_proc = chunks(actions, self.envs_per_proc)

        for remote, action_list in zip(self.remotes, actions_per_proc):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))
        return obs, rewards, dones, env_infos

    def reset(self):
        """Reset all environments."""
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def close(self):
        """Close all workers."""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return self._num_envs


class IterativeEnvExecutor:
    """
    Executes multiple environments iteratively (no parallelization).
    """

    def __init__(self, env, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.envs = [copy.deepcopy(env) for _ in range(num_rollouts)]
        self.ts = np.zeros(num_rollouts, dtype='int')
        self.max_path_length = max_path_length

    def step(self, actions):
        assert len(actions) == self.num_envs
        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)

        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones.tolist(), env_infos

    def reset(self):
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    def close(self):
        pass

    @property
    def num_envs(self):
        return self._num_envs


def _env_worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """Worker process for parallel env execution."""
    parent_remote.close()
    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)
    ts = np.zeros(n_envs, dtype='int')

    while True:
        cmd, data = remote.recv()

        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        elif cmd == 'close':
            remote.close()
            break


# ============== Rollout Collection ==============

def collect_rollout_phase1(env, psi, max_steps, device):
    """
    Collect a single rollout with simple policy (no context).
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []

    s = env.reset()
    for t in range(max_steps):
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            a, log_prob = psi.sample(s_tensor, c=None)

        a_np = a.squeeze(0).cpu().numpy()
        s_next, r, done, info = env.step(a_np)

        states.append(s)
        actions.append(a_np)
        rewards.append(r)
        next_states.append(s_next)
        dones.append(done)

        s = s_next
        if done:
            break

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones),
    }


def collect_rollouts_parallel_phase1(vec_env, psi, max_steps, device):
    """
    Collect rollouts in parallel using vectorized env executor.
    Uses GPU batching for policy forward passes.
    """
    num_envs = vec_env.num_envs
    running_paths = [{'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
                     for _ in range(num_envs)]
    completed_paths = []

    obses = vec_env.reset()
    n_samples = 0
    total_samples = num_envs * max_steps

    while n_samples < total_samples and len(completed_paths) < num_envs:
        # GPU batch: get actions for all envs at once
        obses_tensor = torch.as_tensor(np.array(obses), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _ = psi.sample(obses_tensor, c=None)
        actions_np = actions.cpu().numpy()

        # Step all envs in parallel
        next_obses, rewards, dones, _ = vec_env.step(list(actions_np))

        for idx in range(num_envs):
            if len(running_paths[idx]['states']) < max_steps:
                running_paths[idx]['states'].append(obses[idx])
                running_paths[idx]['actions'].append(actions_np[idx])
                running_paths[idx]['rewards'].append(rewards[idx])
                running_paths[idx]['next_states'].append(next_obses[idx])
                running_paths[idx]['dones'].append(dones[idx])

                if dones[idx]:
                    completed_paths.append({
                        'states': np.array(running_paths[idx]['states']),
                        'actions': np.array(running_paths[idx]['actions']),
                        'rewards': np.array(running_paths[idx]['rewards']),
                        'next_states': np.array(running_paths[idx]['next_states']),
                        'dones': np.array(running_paths[idx]['dones']),
                    })
                    n_samples += len(running_paths[idx]['states'])
                    running_paths[idx] = {'states': [], 'actions': [], 'rewards': [],
                                          'next_states': [], 'dones': []}

        obses = next_obses

    # Add any incomplete paths
    for idx in range(num_envs):
        if len(running_paths[idx]['states']) > 0:
            completed_paths.append({
                'states': np.array(running_paths[idx]['states']),
                'actions': np.array(running_paths[idx]['actions']),
                'rewards': np.array(running_paths[idx]['rewards']),
                'next_states': np.array(running_paths[idx]['next_states']),
                'dones': np.array(running_paths[idx]['dones']),
            })

    return completed_paths


def collect_rollouts_parallel_phase2(vec_env, world_model, phi, psi, max_steps, device):
    """
    Collect rollouts in parallel with context-conditioned policy.
    Uses GPU batching for all forward passes.
    """
    num_envs = vec_env.num_envs
    running_paths = [{'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': [],
                      'states_history': [], 'actions_history': []}
                     for _ in range(num_envs)]
    completed_paths = []

    # Initialize contexts for all envs (batch on GPU)
    contexts = phi.reset_context(batch_size=num_envs, device=device)

    obses = vec_env.reset()
    for idx in range(num_envs):
        running_paths[idx]['states_history'].append(
            obses[idx].copy() if isinstance(obses[idx], np.ndarray) else np.array(obses[idx]))

    t = 0
    while len(completed_paths) < num_envs and t < max_steps:
        # GPU batch: update contexts for all envs at t>=2
        if t >= 2:
            # Gather states and actions from history
            s_t_minus_2_list = []
            a_t_minus_2_list = []
            s_prev_list = []
            valid_indices = []

            for idx in range(num_envs):
                if len(running_paths[idx]['states_history']) >= 3:
                    s_t_minus_2_list.append(running_paths[idx]['states_history'][-3])
                    a_t_minus_2_list.append(running_paths[idx]['actions_history'][-2])
                    s_prev_list.append(running_paths[idx]['states_history'][-2])
                    valid_indices.append(idx)

            if valid_indices:
                s_t_minus_2_batch = torch.as_tensor(np.array(s_t_minus_2_list), dtype=torch.float32, device=device)
                a_t_minus_2_batch = torch.as_tensor(np.array(a_t_minus_2_list), dtype=torch.float32, device=device)
                s_prev_batch = torch.as_tensor(np.array(s_prev_list), dtype=torch.float32, device=device)

                with torch.no_grad():
                    p_hat_batch = world_model(s_t_minus_2_batch, a_t_minus_2_batch)
                    c_subset = contexts[valid_indices]
                    c_new = phi(c_subset, p_hat_batch, s_prev_batch)
                    contexts[valid_indices] = c_new

        # GPU batch: get actions for all envs
        obses_tensor = torch.as_tensor(np.array(obses), dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, _ = psi.sample(obses_tensor, contexts)
        actions_np = actions.cpu().numpy()

        # Step all envs in parallel
        next_obses, rewards, dones, _ = vec_env.step(list(actions_np))

        for idx in range(num_envs):
            running_paths[idx]['states'].append(obses[idx])
            running_paths[idx]['actions'].append(actions_np[idx])
            running_paths[idx]['rewards'].append(rewards[idx])
            running_paths[idx]['next_states'].append(next_obses[idx])
            running_paths[idx]['dones'].append(dones[idx])
            running_paths[idx]['actions_history'].append(actions_np[idx])
            running_paths[idx]['states_history'].append(
                next_obses[idx].copy() if isinstance(next_obses[idx], np.ndarray) else np.array(next_obses[idx]))

            if dones[idx]:
                completed_paths.append({
                    'states': np.array(running_paths[idx]['states']),
                    'actions': np.array(running_paths[idx]['actions']),
                    'rewards': np.array(running_paths[idx]['rewards']),
                    'next_states': np.array(running_paths[idx]['next_states']),
                    'dones': np.array(running_paths[idx]['dones']),
                })
                # Reset this env's running path and context
                running_paths[idx] = {'states': [], 'actions': [], 'rewards': [],
                                      'next_states': [], 'dones': [],
                                      'states_history': [next_obses[idx]], 'actions_history': []}
                contexts[idx] = phi.reset_context(batch_size=1, device=device).squeeze(0)

        obses = next_obses
        t += 1

    # Add incomplete paths
    for idx in range(num_envs):
        if len(running_paths[idx]['states']) > 0:
            completed_paths.append({
                'states': np.array(running_paths[idx]['states']),
                'actions': np.array(running_paths[idx]['actions']),
                'rewards': np.array(running_paths[idx]['rewards']),
                'next_states': np.array(running_paths[idx]['next_states']),
                'dones': np.array(running_paths[idx]['dones']),
            })

    return completed_paths


def collect_rollout_phase2(env, world_model, phi, psi, max_steps, device):
    """
    Collect a single rollout with context-conditioned policy.

    Context update at time t:
        c_t = f_φ(c_{t-1}, p̂_{t-1}, s_{t-1})
    where:
        p̂_{t-1} = p̂(s_{t-1} | s_{t-2}, a_{t-2}) = prediction OF s_{t-1}
        s_{t-1} = actual state at t-1
    """
    states, actions, rewards, next_states, dones = [], [], [], [], []

    s = env.reset()
    c = phi.reset_context(batch_size=1, device=device)

    # History for context updates
    states_history = [s.copy() if isinstance(s, np.ndarray) else np.array(s)]
    actions_history = []

    for t in range(max_steps):
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        # Update context at t>=2 using prediction error from previous transition
        if t >= 2:
            s_t_minus_2 = torch.as_tensor(states_history[t-2], dtype=torch.float32, device=device).unsqueeze(0)
            a_t_minus_2 = torch.as_tensor(actions_history[t-2], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                p_hat_prev = world_model(s_t_minus_2, a_t_minus_2)
            s_prev_tensor = torch.as_tensor(states_history[t-1], dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                c = phi(c, p_hat_prev, s_prev_tensor)

        # Get action
        with torch.no_grad():
            a, log_prob = psi.sample(s_tensor, c)

        a_np = a.squeeze(0).cpu().numpy()
        s_next, r, done, info = env.step(a_np)

        states.append(s)
        actions.append(a_np)
        rewards.append(r)
        next_states.append(s_next)
        dones.append(done)

        # Update history
        actions_history.append(a_np)
        states_history.append(s_next.copy() if isinstance(s_next, np.ndarray) else np.array(s_next))
        s = s_next

        if done:
            break

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_states': np.array(next_states),
        'dones': np.array(dones),
    }


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns)


# ============== Phase 1 Trainer ==============

class Phase1Trainer:
    """
    Phase 1: Train on single initial environment.

    Trains:
        - π_ψ(a|s): Simple policy without context
        - p̂(s_{t+1}|s_t,a_t): World model

    Supports:
        - n_parallel: CPU parallelization for env sampling
        - GPU batching for policy forward passes
    """

    def __init__(
        self,
        env,
        world_model: WorldModel,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
    ):
        self.env = env
        self.world_model = world_model
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        self.world_model.to(self.device)
        self.psi.to(self.device)

        # Parallelization
        self.n_parallel = config.get('n_parallel', 1)
        self.num_rollouts = config.get('batch_size', 10)
        max_path_length = config.get('max_path_length', 1000)

        # Setup vectorized env executor
        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, self.n_parallel, self.num_rollouts, max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, self.num_rollouts, max_path_length)

        self.policy_optimizer = Adam(
            self.psi.parameters(),
            lr=config.get('policy_lr', 3e-4),
        )
        self.world_model_optimizer = Adam(
            self.world_model.parameters(),
            lr=config.get('world_model_lr', 1e-3),
        )

        self.replay_buffer = {
            'states': [],
            'actions': [],
            'next_states': [],
        }
        self.max_buffer_size = config.get('max_buffer_size', 100000)

        self.gamma = config.get('gamma', 0.99)
        self.clip_eps = config.get('clip_eps', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.minibatch_size = config.get('minibatch_size', 64)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)

        # Track timesteps
        self.n_timesteps = 0

        self.log_history = []

    def add_to_buffer(self, trajectory):
        """Add trajectory data to replay buffer."""
        self.replay_buffer['states'].extend(trajectory['states'])
        self.replay_buffer['actions'].extend(trajectory['actions'])
        self.replay_buffer['next_states'].extend(trajectory['next_states'])

        if len(self.replay_buffer['states']) > self.max_buffer_size:
            excess = len(self.replay_buffer['states']) - self.max_buffer_size
            self.replay_buffer['states'] = self.replay_buffer['states'][excess:]
            self.replay_buffer['actions'] = self.replay_buffer['actions'][excess:]
            self.replay_buffer['next_states'] = self.replay_buffer['next_states'][excess:]

    def train_world_model(self, n_epochs=10, batch_size=256):
        """Train world model on replay buffer."""
        if len(self.replay_buffer['states']) < batch_size:
            return {'world_model_loss': 0.0}

        states = torch.tensor(np.array(self.replay_buffer['states']),
                            dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.replay_buffer['actions']),
                             dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(self.replay_buffer['next_states']),
                                  dtype=torch.float32, device=self.device)

        self.world_model.compute_normalization(states, actions, next_states)

        losses = []
        n_samples = len(states)

        for epoch in range(n_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                s_next_batch = next_states[batch_idx]

                self.world_model_optimizer.zero_grad()
                loss = self.world_model.loss(s_batch, a_batch, s_next_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.max_grad_norm)
                self.world_model_optimizer.step()
                losses.append(loss.item())

        return {'world_model_loss': np.mean(losses)}

    def update_policy(self, trajectories):
        """Update policy using PPO."""
        all_states = []
        all_actions = []
        all_returns = []
        all_old_log_probs = []

        for traj in trajectories:
            returns = compute_returns(traj['rewards'], self.gamma)
            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_returns.extend(returns)

            states_t = torch.tensor(traj['states'], dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(traj['actions'], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                old_log_probs = self.psi.log_prob(states_t, None, actions_t)
            all_old_log_probs.extend(old_log_probs.cpu().numpy())

        states = torch.tensor(np.array(all_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(all_actions), dtype=torch.float32, device=self.device)
        returns = torch.tensor(np.array(all_returns), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array(all_old_log_probs), dtype=torch.float32, device=self.device)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        n_samples = len(states)
        ppo_losses = []
        entropy_losses = []

        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(n_samples, device=self.device)
            for i in range(0, n_samples, self.minibatch_size):
                batch_idx = indices[i:i+self.minibatch_size]

                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                ret_batch = returns[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]

                new_log_probs = self.psi.log_prob(s_batch, None, a_batch)
                dist = self.psi.get_distribution(s_batch, None)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - old_lp_batch)
                surr1 = ratio * ret_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * ret_batch
                ppo_loss = -torch.min(surr1, surr2).mean()

                loss = ppo_loss - self.entropy_coef * entropy

                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.psi.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                ppo_losses.append(ppo_loss.item())
                entropy_losses.append(entropy.item())

        return {
            'ppo_loss': np.mean(ppo_losses),
            'entropy': np.mean(entropy_losses),
        }

    def train(self, n_iterations: int = None, save_dir: str = None):
        """Run Phase 1 training with parallel env sampling."""
        n_iterations = n_iterations or self.config.get('phase1_iterations', 500)
        batch_size = self.config.get('batch_size', 20)
        max_path_length = self.config.get('max_path_length', 1000)
        save_every = self.config.get('save_every', 50)
        world_model_train_freq = self.config.get('world_model_train_freq', 5)
        world_model_epochs = self.config.get('world_model_epochs', 10)

        print(f"Phase 1: Training for {n_iterations} iterations")
        print(f"  Batch size: {batch_size} trajectories")
        print(f"  Max path length: {max_path_length}")
        print(f"  n_parallel: {self.n_parallel}")

        self.csv_file = None
        self.csv_writer = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, 'progress.csv')
            self.csv_file = open(csv_path, 'w', newline='')

        for iteration in range(n_iterations):
            self.psi.eval()

            # Collect trajectories in parallel with GPU batching
            trajectories = collect_rollouts_parallel_phase1(
                vec_env=self.vec_env,
                psi=self.psi,
                max_steps=max_path_length,
                device=self.device,
            )

            for traj in trajectories:
                self.add_to_buffer(traj)
                self.n_timesteps += len(traj['states'])

            episode_rewards = [traj['rewards'].sum() for traj in trajectories]
            episode_lengths = [len(traj['rewards']) for traj in trajectories]

            self.psi.train()
            policy_info = self.update_policy(trajectories)

            world_model_info = {'world_model_loss': 0.0}
            if iteration % world_model_train_freq == 0:
                self.world_model.train()
                world_model_info = self.train_world_model(
                    n_epochs=world_model_epochs,
                    batch_size=self.config.get('world_model_batch_size', 256),
                )

            log_entry = {
                'iteration': iteration,
                'phase': 1,
                'n_timesteps': self.n_timesteps,
                'mean_episode_reward': np.mean(episode_rewards),
                'std_episode_reward': np.std(episode_rewards),
                'mean_episode_length': np.mean(episode_lengths),
                'buffer_size': len(self.replay_buffer['states']),
                **policy_info,
                **world_model_info,
            }
            self.log_history.append(log_entry)

            if self.csv_file:
                if self.csv_writer is None:
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=log_entry.keys())
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(log_entry)
                self.csv_file.flush()

            if iteration % 10 == 0:
                print(f"  Iter {iteration:4d} | "
                      f"Reward: {np.mean(episode_rewards):7.2f} | "
                      f"PPO Loss: {policy_info['ppo_loss']:.4f} | "
                      f"WM Loss: {world_model_info['world_model_loss']:.4f}")

            if save_dir and (iteration + 1) % save_every == 0:
                self.save_checkpoint(save_dir, iteration + 1)

        if save_dir:
            self.save_checkpoint(save_dir, 'final')
            self.save_logs(save_dir)

        return self.world_model, self.psi

    def save_checkpoint(self, save_dir: str, tag):
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        path = os.path.join(save_dir, 'checkpoints', f'phase1_{tag}.pt')
        torch.save({
            'world_model': self.world_model.state_dict(),
            'psi': self.psi.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
        }, path)

    def save_logs(self, save_dir: str):
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        path = os.path.join(save_dir, 'logs', 'phase1_train.json')
        with open(path, 'w') as f:
            json.dump(self.log_history, f, indent=2)
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()


# ============== Phase 2 Trainer (GrBAL-style) ==============

class Phase2Trainer:
    """
    Phase 2: GrBAL-style MAML meta-training.

    Like GrBAL Algorithm 1:
        - Collect trajectories into dataset D
        - Sample contiguous segments: τ(t-M, t-1) for adaptation, τ(t, t+K) for evaluation
        - Inner loop: adapt φ', ψ' on adaptation segment using RL loss
        - Outer loop: compute RL loss on evaluation segment with adapted params
        - Meta-update φ, ψ

    Frozen: p̂ (world model)
    """

    def __init__(
        self,
        train_envs: List,
        world_model: WorldModel,
        phi: ContextEncoder,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
        test_envs: List = None,
        test_env_ids: List = None,
    ):
        self.train_envs = train_envs
        self.test_envs = test_envs or []
        self.test_env_ids = test_env_ids or []
        self.world_model = world_model
        self.phi = phi
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        self.world_model.to(self.device)
        self.phi.to(self.device)
        self.psi.to(self.device)

        # Parallelization
        self.n_parallel = config.get('n_parallel', 1)
        self.num_rollouts_per_env = config.get('num_rollouts', 5)
        max_path_length = config.get('max_path_length', 1000)

        # Setup vectorized env executors for each training environment
        self.vec_envs = []
        for env in train_envs:
            if self.n_parallel > 1:
                vec_env = ParallelEnvExecutor(env, self.n_parallel, self.num_rollouts_per_env, max_path_length)
            else:
                vec_env = IterativeEnvExecutor(env, self.num_rollouts_per_env, max_path_length)
            self.vec_envs.append(vec_env)

        # Freeze world model
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad = False

        # Meta-optimizer
        self.meta_optimizer = Adam(
            list(self.phi.parameters()) + list(self.psi.parameters()),
            lr=config.get('meta_lr', 1e-3),
        )

        # GrBAL-style hyperparameters
        self.inner_lr = config.get('inner_lr', 0.01)
        self.adapt_steps = config.get('adapt_steps', 1)  # Number of gradient steps
        self.M = config.get('M', 20)  # Adaptation segment length
        self.K = config.get('K', 20)  # Evaluation segment length
        self.gamma = config.get('gamma', 0.99)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.task_sampling_freq = config.get('task_sampling_freq', 1)  # n_S in GrBAL

        # Dataset D for each environment
        self.datasets = {i: [] for i in range(len(train_envs))}

        # Sample efficiency tracking
        self.n_timesteps = 0  # Total environment steps
        self.eval_every = config.get('eval_every', 10)  # Evaluate on test envs every N iterations

        self.log_history = []

    def collect_and_store_trajectories(self, env_idx):
        """Collect trajectories in parallel and add to dataset D for environment."""
        max_path_length = self.config.get('max_path_length', 1000)

        self.phi.eval()
        self.psi.eval()

        # Use parallel collection with GPU batching
        trajectories = collect_rollouts_parallel_phase2(
            vec_env=self.vec_envs[env_idx],
            world_model=self.world_model,
            phi=self.phi,
            psi=self.psi,
            max_steps=max_path_length,
            device=self.device,
        )

        for traj in trajectories:
            self.datasets[env_idx].append(traj)
            # Track total timesteps for sample efficiency
            self.n_timesteps += len(traj['states'])

        # Limit dataset size
        max_trajs = self.config.get('max_trajs_per_env', 50)
        if len(self.datasets[env_idx]) > max_trajs:
            self.datasets[env_idx] = self.datasets[env_idx][-max_trajs:]

        return trajectories

    def sample_segments(self, env_idx):
        """
        Sample contiguous segments from dataset D:
            τ(t-M, t-1): M steps for adaptation
            τ(t, t+K): K steps for evaluation
        """
        if not self.datasets[env_idx]:
            return None, None

        # Concatenate all trajectories for this env
        all_states = []
        all_actions = []
        all_rewards = []

        for traj in self.datasets[env_idx]:
            all_states.extend(traj['states'])
            all_actions.extend(traj['actions'])
            all_rewards.extend(traj['rewards'])

        total_len = len(all_states)
        required_len = self.M + self.K

        if total_len < required_len:
            return None, None

        # Sample starting point t such that we can get τ(t-M, t-1) and τ(t, t+K)
        # t must be >= M and t + K <= total_len
        max_t = total_len - self.K
        min_t = self.M

        if max_t < min_t:
            return None, None

        t = np.random.randint(min_t, max_t + 1)

        # Adaptation segment: τ(t-M, t-1) = indices [t-M, t-1] inclusive = [t-M, t)
        adapt_segment = {
            'states': np.array(all_states[t - self.M:t]),
            'actions': np.array(all_actions[t - self.M:t]),
            'rewards': np.array(all_rewards[t - self.M:t]),
        }

        # Evaluation segment: τ(t, t+K) = indices [t, t+K-1] inclusive = [t, t+K)
        eval_segment = {
            'states': np.array(all_states[t:t + self.K]),
            'actions': np.array(all_actions[t:t + self.K]),
            'rewards': np.array(all_rewards[t:t + self.K]),
        }

        return adapt_segment, eval_segment

    def compute_segment_loss(self, segment):
        """
        Compute policy gradient loss on a segment.
        Uses REINFORCE with returns as advantages.

        Context updates use: c_t = f_φ(c_{t-1}, p̂_{t-1}, s_{t-1})
        where p̂_{t-1} = p̂(s_{t-1} | s_{t-2}, a_{t-2})
        """
        states = torch.tensor(segment['states'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(segment['actions'], dtype=torch.float32, device=self.device)
        rewards = segment['rewards']

        # Compute returns
        returns = compute_returns(rewards, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass through segment with context updates
        T = len(states)
        c = self.phi.reset_context(batch_size=1, device=self.device)

        log_probs = []
        for t in range(T):
            s_t = states[t:t+1]
            a_t = actions[t:t+1]

            # Update context at t>=2
            if t >= 2:
                with torch.no_grad():
                    p_hat_prev = self.world_model(states[t-2:t-1], actions[t-2:t-1])
                s_prev = states[t-1:t]
                c = self.phi(c, p_hat_prev, s_prev)

            log_prob = self.psi.log_prob(s_t, c, a_t)
            log_probs.append(log_prob)

        log_probs = torch.cat(log_probs)
        loss = -(log_probs * returns).mean()

        return loss

    def adapt_on_segment(self, adapt_segment):
        """
        Perform inner loop adaptation on adaptation segment.
        Returns adapted parameter dicts.
        """
        for step in range(self.adapt_steps):
            loss = self.compute_segment_loss(adapt_segment)

            # Compute gradients
            phi_grads = torch.autograd.grad(
                loss,
                list(self.phi.parameters()),
                create_graph=True,
                allow_unused=True,
            )
            psi_grads = torch.autograd.grad(
                loss,
                list(self.psi.parameters()),
                create_graph=True,
                allow_unused=True,
            )

            # Adapt parameters
            adapted_phi_params = OrderedDict()
            for (name, param), grad in zip(self.phi.named_parameters(), phi_grads):
                if grad is not None:
                    adapted_phi_params[name] = param - self.inner_lr * grad
                else:
                    adapted_phi_params[name] = param

            adapted_psi_params = OrderedDict()
            for (name, param), grad in zip(self.psi.named_parameters(), psi_grads):
                if grad is not None:
                    adapted_psi_params[name] = param - self.inner_lr * grad
                else:
                    adapted_psi_params[name] = param

            # Set adapted params for next step (if multiple adapt steps)
            if step < self.adapt_steps - 1:
                set_params_dict(self.phi, adapted_phi_params)
                set_params_dict(self.psi, adapted_psi_params)

        return adapted_phi_params, adapted_psi_params

    def evaluate_on_test_envs(self, n_episodes: int = 3):
        """
        Evaluate current policy on test environments.
        Returns average return across all test envs (for sample efficiency plots).
        """
        if not self.test_envs:
            return None, None

        self.phi.eval()
        self.psi.eval()

        all_returns = []
        max_path_length = self.config.get('max_path_length', 1000)

        for env in self.test_envs:
            for _ in range(n_episodes):
                traj = collect_rollout_phase2(
                    env=env,
                    world_model=self.world_model,
                    phi=self.phi,
                    psi=self.psi,
                    max_steps=max_path_length,
                    device=self.device,
                )
                episode_return = np.sum(traj['rewards'])
                all_returns.append(episode_return)

        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)

        return mean_return, std_return

    def train(self, n_iterations: int = None, save_dir: str = None):
        """Run Phase 2 GrBAL-style training."""
        n_iterations = n_iterations or self.config.get('phase2_iterations', 500)
        N = self.config.get('meta_batch_size') or len(self.train_envs)  # Tasks per meta-batch
        save_every = self.config.get('save_every', 50)

        print(f"Phase 2: GrBAL-style MAML training for {n_iterations} iterations")
        print(f"  Training envs: {len(self.train_envs)}")
        print(f"  n_parallel: {self.n_parallel}, rollouts per env: {self.num_rollouts_per_env}")
        print(f"  M (adapt steps): {self.M}, K (eval steps): {self.K}")
        print(f"  Inner LR: {self.inner_lr}, Adapt gradient steps: {self.adapt_steps}")
        print(f"  Meta batch size: {N}")

        # Initialize CSV
        self.csv_file = None
        self.csv_writer = None
        self.phase1_iterations = self.config.get('phase1_iterations', 50)
        if save_dir:
            csv_path = os.path.join(save_dir, 'progress.csv')
            self.csv_file = open(csv_path, 'a', newline='')

        for iteration in range(n_iterations):
            # Collect new trajectories periodically (like GrBAL's n_S)
            if iteration % self.task_sampling_freq == 0:
                for env_idx in range(len(self.train_envs)):
                    self.collect_and_store_trajectories(env_idx)

            meta_loss = 0.0
            valid_tasks = 0
            all_rewards = []

            self.meta_optimizer.zero_grad()

            # Sample N tasks
            task_indices = np.random.choice(
                len(self.train_envs),
                size=min(N, len(self.train_envs)),
                replace=False
            )

            for env_idx in task_indices:
                # Store original parameters
                original_phi_params = get_params_dict(self.phi)
                original_psi_params = get_params_dict(self.psi)

                # Sample segments
                adapt_segment, eval_segment = self.sample_segments(env_idx)

                if adapt_segment is None:
                    # Not enough data yet
                    set_params_dict(self.phi, original_phi_params)
                    set_params_dict(self.psi, original_psi_params)
                    continue

                # Inner loop: adapt on adaptation segment
                self.phi.train()
                self.psi.train()
                adapted_phi_params, adapted_psi_params = self.adapt_on_segment(adapt_segment)

                # Set adapted parameters
                set_params_dict(self.phi, adapted_phi_params)
                set_params_dict(self.psi, adapted_psi_params)

                # Outer loop: compute loss on evaluation segment
                task_loss = self.compute_segment_loss(eval_segment)
                meta_loss += task_loss
                valid_tasks += 1

                # Track rewards
                all_rewards.append(eval_segment['rewards'].sum())

                # Restore original parameters
                set_params_dict(self.phi, original_phi_params)
                set_params_dict(self.psi, original_psi_params)

            if valid_tasks > 0:
                meta_loss = meta_loss / valid_tasks
                meta_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.phi.parameters()) + list(self.psi.parameters()),
                    self.max_grad_norm
                )
                self.meta_optimizer.step()

            # Evaluate on test envs periodically (for sample efficiency plots)
            test_mean_return, test_std_return = None, None
            if self.test_envs and iteration % self.eval_every == 0:
                test_mean_return, test_std_return = self.evaluate_on_test_envs(n_episodes=3)

            # Log (like GrBAL Figure 4: AverageReturn vs n_timesteps)
            log_entry = {
                'iteration': iteration,
                'phase': 2,
                'n_timesteps': self.n_timesteps,
                'mean_segment_reward': np.mean(all_rewards) if all_rewards else 0.0,
                'meta_loss': meta_loss.item() if valid_tasks > 0 else 0.0,
                'valid_tasks': valid_tasks,
                'test_AverageReturn': test_mean_return,
                'test_StdReturn': test_std_return,
            }
            self.log_history.append(log_entry)

            if self.csv_file:
                csv_entry = log_entry.copy()
                csv_entry['iteration'] = iteration + self.phase1_iterations
                if self.csv_writer is None:
                    fieldnames = ['iteration', 'phase', 'n_timesteps', 'mean_segment_reward',
                                  'meta_loss', 'valid_tasks', 'test_AverageReturn', 'test_StdReturn']
                    self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
                    self.csv_writer.writeheader()
                self.csv_writer.writerow(csv_entry)
                self.csv_file.flush()

            if iteration % 10 == 0:
                test_str = f" | Test: {test_mean_return:.1f}±{test_std_return:.1f}" if test_mean_return is not None else ""
                print(f"  Iter {iteration:4d} | "
                      f"Steps: {self.n_timesteps:,} | "
                      f"Reward: {np.mean(all_rewards) if all_rewards else 0:.2f} | "
                      f"Loss: {meta_loss.item() if valid_tasks > 0 else 0:.4f}{test_str}")

            if save_dir and (iteration + 1) % save_every == 0:
                self.save_checkpoint(save_dir, iteration + 1)

        if save_dir:
            self.save_checkpoint(save_dir, 'final')
            self.save_logs(save_dir)

        return self.phi, self.psi

    def save_checkpoint(self, save_dir: str, tag):
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        path = os.path.join(save_dir, 'checkpoints', f'phase2_{tag}.pt')
        torch.save({
            'world_model': self.world_model.state_dict(),
            'phi': self.phi.state_dict(),
            'psi': self.psi.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
        }, path)

    def save_logs(self, save_dir: str):
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        path = os.path.join(save_dir, 'logs', 'phase2_train.json')
        with open(path, 'w') as f:
            json.dump(self.log_history, f, indent=2)
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()


# ============== Checkpoint Loading ==============

def load_phase1_checkpoint(
    path: str,
    obs_dim: int,
    act_dim: int,
    context_dim: int = 64,
    policy_hidden_sizes: tuple = (256, 256),
    world_model_hidden_sizes: tuple = (512, 512, 512),
    device: torch.device = None,
):
    """Load Phase 1 checkpoint."""
    world_model = WorldModel(obs_dim, act_dim, world_model_hidden_sizes)
    psi = Policy(obs_dim, act_dim, context_dim, policy_hidden_sizes)

    checkpoint = torch.load(path, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    psi.load_state_dict(checkpoint['psi'])

    if device:
        world_model.to(device)
        psi.to(device)

    return world_model, psi


def load_phase2_checkpoint(
    path: str,
    obs_dim: int,
    act_dim: int,
    context_dim: int = 64,
    policy_hidden_sizes: tuple = (256, 256),
    world_model_hidden_sizes: tuple = (512, 512, 512),
    device: torch.device = None,
):
    """Load Phase 2 checkpoint."""
    world_model = WorldModel(obs_dim, act_dim, world_model_hidden_sizes)
    phi = ContextEncoder(obs_dim, context_dim)
    psi = Policy(obs_dim, act_dim, context_dim, policy_hidden_sizes)

    checkpoint = torch.load(path, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    phi.load_state_dict(checkpoint['phi'])
    psi.load_state_dict(checkpoint['psi'])

    if device:
        world_model.to(device)
        phi.to(device)
        psi.to(device)

    return world_model, phi, psi
