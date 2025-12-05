"""
Trajectory collection utilities.

Supports both sequential and vectorized (batched) rollout collection.
Vectorized rollouts batch network inference for GPU speedup.

Can use ParallelEnvExecutor from the original repo for CPU parallelism.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable

# Import parallel env executor from original repo
try:
    from learning_to_adapt.samplers.vectorized_env_executor import (
        ParallelEnvExecutor,
        IterativeEnvExecutor,
    )
    HAS_PARALLEL_EXECUTOR = True
except ImportError:
    HAS_PARALLEL_EXECUTOR = False


def collect_trajectory_phase1(
    env,
    theta,
    psi,
    context_dim: int,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[Dict]:
    """
    Collect one trajectory for Phase 1 (no context adaptation).

    In Phase 1, c_t = 0 always (context not used yet).

    Args:
        env: environment instance
        theta: StateProposer network
        psi: Policy network
        context_dim: dimension of context vector (for creating zeros)
        max_steps: max trajectory length
        device: torch device

    Returns:
        trajectory: list of step dicts
    """
    if device is None:
        device = next(theta.parameters()).device

    trajectory = []
    s = env.reset()

    # Fixed zero context for Phase 1
    c = torch.zeros(1, context_dim, device=device)

    for t in range(max_steps):
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            X = theta(s_tensor)
            a, log_prob = psi.sample(s_tensor, X, c)

        # Convert to numpy
        a_np = a.squeeze(0).cpu().numpy()
        X_np = X.squeeze(0).cpu().numpy()
        c_np = c.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.item()

        # Step environment
        s_next, reward, done, info = env.step(a_np)

        trajectory.append({
            's': s.copy() if isinstance(s, np.ndarray) else np.array(s),
            'X': X_np,
            'c': c_np,
            'a': a_np,
            'log_prob': log_prob_np,
            'reward': reward,
            's_next': s_next.copy() if isinstance(s_next, np.ndarray) else np.array(s_next),
        })

        s = s_next

        if done:
            break

    return trajectory


def collect_trajectory_phase2(
    env,
    theta,
    phi,
    psi,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[Dict]:
    """
    Collect one trajectory for Phase 2 (with context adaptation).

    Context c_t is updated via f_phi at each step (except t=0).

    Args:
        env: environment instance
        theta: StateProposer network (frozen)
        phi: ContextEncoder network
        psi: Policy network
        max_steps: max trajectory length
        device: torch device

    Returns:
        trajectory: list of step dicts
    """
    if device is None:
        device = next(psi.parameters()).device

    trajectory = []
    s = env.reset()

    # Initialize context
    c = phi.reset_context(batch_size=1, device=device)

    X_prev = None
    s_prev = None

    for t in range(max_steps):
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Propose next state
            X = theta(s_tensor)

            # Update context (skip at t=0)
            if t > 0:
                X_prev_tensor = torch.as_tensor(X_prev, dtype=torch.float32, device=device).unsqueeze(0)
                s_prev_tensor = torch.as_tensor(s_prev, dtype=torch.float32, device=device).unsqueeze(0)
                c = phi(c, X_prev_tensor, s_prev_tensor)

            # Sample action
            a, log_prob = psi.sample(s_tensor, X, c)

        # Convert to numpy
        a_np = a.squeeze(0).cpu().numpy()
        X_np = X.squeeze(0).cpu().numpy()
        c_np = c.squeeze(0).cpu().numpy()
        log_prob_np = log_prob.item()

        # Step environment
        s_next, reward, done, info = env.step(a_np)

        trajectory.append({
            's': s.copy() if isinstance(s, np.ndarray) else np.array(s),
            'X': X_np,
            'c': c_np,
            'a': a_np,
            'log_prob': log_prob_np,
            'reward': reward,
            's_next': s_next.copy() if isinstance(s_next, np.ndarray) else np.array(s_next),
        })

        # Store for next iteration
        X_prev = X_np
        s_prev = s.copy() if isinstance(s, np.ndarray) else np.array(s)
        s = s_next

        if done:
            break

    return trajectory


def collect_batch_phase1(
    env,
    theta,
    psi,
    context_dim: int,
    n_trajectories: int,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect batch of trajectories for Phase 1.

    Args:
        env: environment instance (will be reset for each trajectory)
        theta: StateProposer network
        psi: Policy network
        context_dim: context dimension
        n_trajectories: number of trajectories to collect
        max_steps: max steps per trajectory
        device: torch device

    Returns:
        trajectories: list of trajectories
    """
    trajectories = []

    for _ in range(n_trajectories):
        traj = collect_trajectory_phase1(
            env=env,
            theta=theta,
            psi=psi,
            context_dim=context_dim,
            max_steps=max_steps,
            device=device,
        )
        trajectories.append(traj)

    return trajectories


def collect_batch_phase2(
    envs: List,
    theta,
    phi,
    psi,
    n_trajectories_per_env: int = 1,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect batch of trajectories for Phase 2.

    Collects from multiple environments.

    Args:
        envs: list of environment instances
        theta: StateProposer network (frozen)
        phi: ContextEncoder network
        psi: Policy network
        n_trajectories_per_env: trajectories per environment
        max_steps: max steps per trajectory
        device: torch device

    Returns:
        trajectories: list of trajectories from all envs
    """
    trajectories = []

    for env in envs:
        for _ in range(n_trajectories_per_env):
            traj = collect_trajectory_phase2(
                env=env,
                theta=theta,
                phi=phi,
                psi=psi,
                max_steps=max_steps,
                device=device,
            )
            trajectories.append(traj)

    return trajectories


def compute_trajectory_stats(trajectories: List[List[Dict]]) -> Dict:
    """
    Compute statistics over collected trajectories.

    Returns:
        stats dict with mean_reward, mean_length, etc.
    """
    all_rewards = []
    all_lengths = []

    for traj in trajectories:
        traj_reward = sum(step['reward'] for step in traj)
        all_rewards.append(traj_reward)
        all_lengths.append(len(traj))

    return {
        'mean_episode_reward': np.mean(all_rewards),
        'std_episode_reward': np.std(all_rewards),
        'min_episode_reward': np.min(all_rewards),
        'max_episode_reward': np.max(all_rewards),
        'mean_episode_length': np.mean(all_lengths),
    }


# =============================================================================
# VECTORIZED ROLLOUT (Batched inference for GPU speedup)
# =============================================================================

def collect_batch_vectorized_phase1(
    env_fn: Callable,
    theta,
    psi,
    context_dim: int,
    n_trajectories: int,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect batch of trajectories with vectorized (batched) network inference.

    Runs N environments in lockstep, batching all network forward passes.
    Gives ~3-5x speedup on GPU compared to sequential collection.

    Args:
        env_fn: callable that returns a new environment instance
        theta: StateProposer network
        psi: Policy network
        context_dim: context dimension
        n_trajectories: number of trajectories to collect
        max_steps: max steps per trajectory
        device: torch device

    Returns:
        trajectories: list of trajectories
    """
    if device is None:
        device = next(theta.parameters()).device

    N = n_trajectories

    # Create N environments
    envs = [env_fn() for _ in range(N)]

    # Initialize trajectories storage
    trajectories = [[] for _ in range(N)]

    # Reset all environments and stack observations
    obs_list = [env.reset() for env in envs]
    obs_np = np.stack(obs_list)  # (N, obs_dim)

    # Fixed zero context for Phase 1
    c = torch.zeros(N, context_dim, device=device)

    # Track which envs are still running
    active = np.ones(N, dtype=bool)

    for t in range(max_steps):
        if not np.any(active):
            break

        # Convert to tensor
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Batched forward passes
            X = theta(obs_tensor)  # (N, obs_dim)
            a, log_prob = psi.sample(obs_tensor, X, c)  # (N, act_dim), (N,)

        # Convert to numpy
        X_np = X.cpu().numpy()
        a_np = a.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        c_np = c.cpu().numpy()

        # Step each environment
        next_obs_list = []
        for i in range(N):
            if not active[i]:
                next_obs_list.append(obs_np[i])  # placeholder
                continue

            next_obs, reward, done, info = envs[i].step(a_np[i])

            trajectories[i].append({
                's': obs_np[i].copy(),
                'X': X_np[i].copy(),
                'c': c_np[i].copy(),
                'a': a_np[i].copy(),
                'log_prob': float(log_prob_np[i]),
                'reward': reward,
                's_next': next_obs.copy() if isinstance(next_obs, np.ndarray) else np.array(next_obs),
            })

            if done:
                active[i] = False

            next_obs_list.append(next_obs if isinstance(next_obs, np.ndarray) else np.array(next_obs))

        obs_np = np.stack(next_obs_list)

    return trajectories


def collect_batch_vectorized_phase2(
    env_fns: List[Callable],
    theta,
    phi,
    psi,
    n_trajectories_per_env: int = 1,
    max_steps: int = 1000,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect batch of trajectories with vectorized (batched) network inference.

    Runs environments from all env_fns in lockstep, batching network calls.

    Args:
        env_fns: list of callables, each returns a new environment instance
        theta: StateProposer network (frozen)
        phi: ContextEncoder network
        psi: Policy network
        n_trajectories_per_env: trajectories per environment type
        max_steps: max steps per trajectory
        device: torch device

    Returns:
        trajectories: list of trajectories from all envs
    """
    if device is None:
        device = next(psi.parameters()).device

    # Create all environments
    envs = []
    for env_fn in env_fns:
        for _ in range(n_trajectories_per_env):
            envs.append(env_fn())

    N = len(envs)

    # Initialize trajectories storage
    trajectories = [[] for _ in range(N)]

    # Reset all environments
    obs_list = [env.reset() for env in envs]
    obs_np = np.stack(obs_list)  # (N, obs_dim)

    # Initialize contexts
    c = phi.reset_context(batch_size=N, device=device)  # (N, context_dim)

    # Track previous X and s for context update
    X_prev_np = None
    s_prev_np = None

    # Track which envs are still running
    active = np.ones(N, dtype=bool)

    for t in range(max_steps):
        if not np.any(active):
            break

        # Convert to tensor
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            # Propose next state (batched)
            X = theta(obs_tensor)  # (N, obs_dim)

            # Update context (skip at t=0)
            if t > 0:
                X_prev_tensor = torch.as_tensor(X_prev_np, dtype=torch.float32, device=device)
                s_prev_tensor = torch.as_tensor(s_prev_np, dtype=torch.float32, device=device)
                c = phi(c, X_prev_tensor, s_prev_tensor)

            # Sample actions (batched)
            a, log_prob = psi.sample(obs_tensor, X, c)  # (N, act_dim), (N,)

        # Convert to numpy
        X_np = X.cpu().numpy()
        a_np = a.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        c_np = c.cpu().numpy()

        # Step each environment
        next_obs_list = []
        for i in range(N):
            if not active[i]:
                next_obs_list.append(obs_np[i])  # placeholder
                continue

            next_obs, reward, done, info = envs[i].step(a_np[i])

            trajectories[i].append({
                's': obs_np[i].copy(),
                'X': X_np[i].copy(),
                'c': c_np[i].copy(),
                'a': a_np[i].copy(),
                'log_prob': float(log_prob_np[i]),
                'reward': reward,
                's_next': next_obs.copy() if isinstance(next_obs, np.ndarray) else np.array(next_obs),
            })

            if done:
                active[i] = False

            next_obs_list.append(next_obs if isinstance(next_obs, np.ndarray) else np.array(next_obs))

        # Store for context update
        X_prev_np = X_np.copy()
        s_prev_np = obs_np.copy()
        obs_np = np.stack(next_obs_list)

    return trajectories


# =============================================================================
# PARALLEL ROLLOUT (Using original repo's ParallelEnvExecutor)
# =============================================================================

def collect_batch_parallel_phase1(
    env,
    theta,
    psi,
    context_dim: int,
    n_trajectories: int,
    max_steps: int = 1000,
    n_parallel: int = 5,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect trajectories using ParallelEnvExecutor for CPU parallelism + batched GPU.

    Args:
        env: base environment (will be copied for each worker)
        theta: StateProposer network
        psi: Policy network
        context_dim: context dimension
        n_trajectories: number of trajectories to collect
        max_steps: max steps per trajectory
        n_parallel: number of parallel CPU workers
        device: torch device

    Returns:
        trajectories: list of completed trajectories
    """
    if not HAS_PARALLEL_EXECUTOR:
        raise RuntimeError("ParallelEnvExecutor not available. Use vectorized rollout instead.")

    if device is None:
        device = next(theta.parameters()).device

    # Create parallel executor
    # n_parallel workers, each running (n_trajectories // n_parallel) envs
    assert n_trajectories % n_parallel == 0, \
        f"n_trajectories ({n_trajectories}) must be divisible by n_parallel ({n_parallel})"

    vec_env = ParallelEnvExecutor(env, n_parallel, n_trajectories, max_steps)
    N = vec_env.num_envs

    # Storage for running trajectories
    running_trajs = [[] for _ in range(N)]
    completed_trajs = []

    # Fixed zero context for Phase 1
    c = torch.zeros(N, context_dim, device=device)

    # Reset all envs
    obs_list = vec_env.reset()
    obs_np = np.stack(obs_list)

    for t in range(max_steps):
        # Batched GPU forward pass
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            X = theta(obs_tensor)
            a, log_prob = psi.sample(obs_tensor, X, c)

        X_np = X.cpu().numpy()
        a_np = a.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        c_np = c.cpu().numpy()

        # Parallel CPU env stepping (this is where the speedup happens)
        next_obs_list, rewards, dones, infos = vec_env.step(list(a_np))
        next_obs_np = np.stack(next_obs_list)

        # Store transitions
        for i in range(N):
            running_trajs[i].append({
                's': obs_np[i].copy(),
                'X': X_np[i].copy(),
                'c': c_np[i].copy(),
                'a': a_np[i].copy(),
                'log_prob': float(log_prob_np[i]),
                'reward': float(rewards[i]) if isinstance(rewards[i], np.ndarray) else rewards[i],
                's_next': next_obs_np[i].copy(),
            })

            # If done, save trajectory and it auto-resets
            if dones[i]:
                completed_trajs.append(running_trajs[i])
                running_trajs[i] = []

        obs_np = next_obs_np

        # Stop if we have enough trajectories
        if len(completed_trajs) >= n_trajectories:
            break

    # Add any incomplete trajectories
    for traj in running_trajs:
        if len(traj) > 0:
            completed_trajs.append(traj)

    return completed_trajs[:n_trajectories]


def collect_batch_parallel_phase2(
    env,
    theta,
    phi,
    psi,
    n_trajectories: int,
    max_steps: int = 1000,
    n_parallel: int = 5,
    device: torch.device = None,
) -> List[List[Dict]]:
    """
    Collect trajectories for Phase 2 using ParallelEnvExecutor.

    Note: All parallel envs will be copies of the same base env.
    For multiple env types, call this function multiple times.

    Args:
        env: base environment
        theta: StateProposer network (frozen)
        phi: ContextEncoder network
        psi: Policy network
        n_trajectories: number of trajectories
        max_steps: max steps per trajectory
        n_parallel: number of parallel workers
        device: torch device

    Returns:
        trajectories: list of completed trajectories
    """
    if not HAS_PARALLEL_EXECUTOR:
        raise RuntimeError("ParallelEnvExecutor not available. Use vectorized rollout instead.")

    if device is None:
        device = next(psi.parameters()).device

    assert n_trajectories % n_parallel == 0, \
        f"n_trajectories ({n_trajectories}) must be divisible by n_parallel ({n_parallel})"

    vec_env = ParallelEnvExecutor(env, n_parallel, n_trajectories, max_steps)
    N = vec_env.num_envs

    # Storage
    running_trajs = [[] for _ in range(N)]
    completed_trajs = []

    # Context per env
    c = phi.reset_context(batch_size=N, device=device)
    X_prev_np = None
    s_prev_np = None

    # Reset
    obs_list = vec_env.reset()
    obs_np = np.stack(obs_list)

    for t in range(max_steps):
        obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            X = theta(obs_tensor)

            if t > 0:
                X_prev_tensor = torch.as_tensor(X_prev_np, dtype=torch.float32, device=device)
                s_prev_tensor = torch.as_tensor(s_prev_np, dtype=torch.float32, device=device)
                c = phi(c, X_prev_tensor, s_prev_tensor)

            a, log_prob = psi.sample(obs_tensor, X, c)

        X_np = X.cpu().numpy()
        a_np = a.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        c_np = c.cpu().numpy()

        # Parallel stepping
        next_obs_list, rewards, dones, infos = vec_env.step(list(a_np))
        next_obs_np = np.stack(next_obs_list)

        for i in range(N):
            running_trajs[i].append({
                's': obs_np[i].copy(),
                'X': X_np[i].copy(),
                'c': c_np[i].copy(),
                'a': a_np[i].copy(),
                'log_prob': float(log_prob_np[i]),
                'reward': float(rewards[i]) if isinstance(rewards[i], np.ndarray) else rewards[i],
                's_next': next_obs_np[i].copy(),
            })

            if dones[i]:
                completed_trajs.append(running_trajs[i])
                running_trajs[i] = []
                # Reset context for this env
                c[i] = 0

        X_prev_np = X_np.copy()
        s_prev_np = obs_np.copy()
        obs_np = next_obs_np

        if len(completed_trajs) >= n_trajectories:
            break

    for traj in running_trajs:
        if len(traj) > 0:
            completed_trajs.append(traj)

    return completed_trajs[:n_trajectories]
