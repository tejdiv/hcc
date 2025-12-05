#!/usr/bin/env python
"""
Main entry point for Subgoal-Conditioned Policy training.

Usage:
    # Run full pipeline (Phase 1, 2, 3)
    python run_scripts/run_subgoal_policy.py

    # Custom experiment name
    python run_scripts/run_subgoal_policy.py --exp_name my_experiment

    # Custom train/test split
    python run_scripts/run_subgoal_policy.py --initial_env 1 --train_envs 2 3 4 --test_envs 5

    # Skip phases (e.g., only evaluate)
    python run_scripts/run_subgoal_policy.py --skip_phase1 --skip_phase2 --checkpoint path/to/phase2.pt

    # Adjust iterations
    python run_scripts/run_subgoal_policy.py --phase1_iterations 1000 --phase2_iterations 1000
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_to_adapt.envs.half_cheetah_env import HalfCheetahEnv
from learning_to_adapt.envs.normalized_env import normalize
from learning_to_adapt.subgoal_policy import (
    StateProposer,
    ContextEncoder,
    Policy,
    Phase1Trainer,
    Phase2Trainer,
    Phase3Evaluator,
    load_phase1_checkpoint,
    load_phase2_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Subgoal-Conditioned Policy Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === Experiment ===
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--save_dir', type=str, default='./data/subgoal_policy',
                        help='Base directory for saving')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # === Environment ===
    parser.add_argument('--task', type=str, default='cripple',
                        help='Task type (cripple or None)')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max steps per episode')
    parser.add_argument('--normalize_env', action='store_true', default=True,
                        help='Normalize environment observations/actions')

    # === Train/Test Split ===
    parser.add_argument('--initial_env', type=int, default=1,
                        help='Crippled leg for Phase 1 (initial env)')
    parser.add_argument('--train_envs', type=int, nargs='+', default=[2, 3, 4],
                        help='Crippled legs for Phase 2 (training envs)')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[5],
                        help='Crippled legs for Phase 3 (test envs)')

    # === Architecture ===
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--context_dim', type=int, default=16,
                        help='Context vector dimension')

    # === Training ===
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clipping epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help='PPO epochs per update')
    parser.add_argument('--minibatch_size', type=int, default=64,
                        help='Minibatch size for PPO')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Gradient clipping norm')
    parser.add_argument('--entropy_coef', type=float, default=0.0,
                        help='Entropy bonus coefficient')

    # === Phase 1 ===
    parser.add_argument('--phase1_iterations', type=int, default=50,
                        help='Phase 1 training iterations')
    parser.add_argument('--phase1_batch_size', type=int, default=5,
                        help='Trajectories per Phase 1 iteration')
    parser.add_argument('--lambda_theta', type=float, default=1.0,
                        help='State prediction loss weight')

    # === Phase 2 ===
    parser.add_argument('--phase2_iterations', type=int, default=50,
                        help='Phase 2 training iterations')
    parser.add_argument('--n_traj_per_env', type=int, default=2,
                        help='Trajectories per environment in Phase 2')

    # === Phase 3 ===
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Episodes per test environment')
    parser.add_argument('--deterministic_eval', action='store_true', default=True,
                        help='Use deterministic (mean) actions for evaluation')

    # === Checkpointing ===
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to load')

    # === Phase Control ===
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Phase 1 (requires --checkpoint)')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip Phase 2 (requires --checkpoint)')
    parser.add_argument('--skip_phase3', action='store_true',
                        help='Skip Phase 3 evaluation')

    # === Device ===
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda, cuda:0, etc.)')

    # === Parallelism ===
    parser.add_argument('--vectorized', action='store_true',
                        help='Use vectorized (batched) rollout for GPU speedup')
    parser.add_argument('--parallel', action='store_true',
                        help='Use ParallelEnvExecutor for CPU parallelism (implies --vectorized)')
    parser.add_argument('--n_parallel', type=int, default=5,
                        help='Number of parallel CPU workers (only with --parallel)')

    return parser.parse_args()


def make_env(crippled_leg: int, task: str = 'cripple', normalize_env: bool = True):
    """Create a HalfCheetah environment with specified crippled leg."""
    env = HalfCheetahEnv(task=task, reset_every_episode=False)
    env.reset()
    env.reset_task(value=crippled_leg)

    if normalize_env:
        env = normalize(env)

    return env


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def main():
    args = parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Experiment directory
    if args.exp_name is None:
        args.exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Save config
    config = vars(args)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Get observation and action dimensions from a sample env
    sample_env = make_env(args.initial_env, args.task, args.normalize_env)
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Update config with dims
    config['obs_dim'] = obs_dim
    config['act_dim'] = act_dim
    config['save_dir'] = save_dir
    config['batch_size'] = args.phase1_batch_size

    # Initialize networks
    theta = StateProposer(obs_dim, args.hidden_dim)
    phi = ContextEncoder(obs_dim, args.context_dim)
    psi = Policy(obs_dim, act_dim, args.context_dim, args.hidden_dim)

    print(f"\nNetwork parameters:")
    print(f"  theta (StateProposer): {sum(p.numel() for p in theta.parameters()):,}")
    print(f"  phi (ContextEncoder):  {sum(p.numel() for p in phi.parameters()):,}")
    print(f"  psi (Policy):          {sum(p.numel() for p in psi.parameters()):,}")
    print(f"  Total:                 {sum(p.numel() for p in theta.parameters()) + sum(p.numel() for p in phi.parameters()) + sum(p.numel() for p in psi.parameters()):,}")

    # ==================== PHASE 1 ====================
    if not args.skip_phase1:
        print("\n" + "=" * 60)
        print("PHASE 1: Training on initial environment")
        print("=" * 60)
        print(f"  Initial env: crippled_leg={args.initial_env}")

        initial_env = make_env(args.initial_env, args.task, args.normalize_env)

        # Create env_fn for vectorized rollout
        initial_env_fn = lambda: make_env(args.initial_env, args.task, args.normalize_env)

        phase1_trainer = Phase1Trainer(
            env=initial_env,
            theta=theta,
            psi=psi,
            config=config,
            device=device,
            env_fn=initial_env_fn if args.vectorized else None,
        )

        theta, psi = phase1_trainer.train(
            n_iterations=args.phase1_iterations,
            save_dir=save_dir,
        )

        print("Phase 1 complete!")

    elif args.checkpoint:
        # Load from checkpoint
        print(f"\nLoading Phase 1 from checkpoint: {args.checkpoint}")
        theta, psi = load_phase1_checkpoint(
            path=args.checkpoint,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            device=device,
        )

    # ==================== PHASE 2 ====================
    if not args.skip_phase2:
        print("\n" + "=" * 60)
        print("PHASE 2: Meta-training across environments")
        print("=" * 60)
        print(f"  Training envs: crippled_leg={args.train_envs}")

        train_envs = [make_env(leg, args.task, args.normalize_env)
                      for leg in args.train_envs]

        # Create env_fns for vectorized rollout
        train_env_fns = [
            (lambda leg=leg: make_env(leg, args.task, args.normalize_env))
            for leg in args.train_envs
        ]

        phase2_trainer = Phase2Trainer(
            train_envs=train_envs,
            theta=theta,
            phi=phi,
            psi=psi,
            config=config,
            device=device,
            env_fns=train_env_fns if args.vectorized else None,
        )

        phi, psi = phase2_trainer.train(
            n_iterations=args.phase2_iterations,
            save_dir=save_dir,
        )

        print("Phase 2 complete!")

    elif args.checkpoint and args.skip_phase1:
        # Load Phase 2 checkpoint
        print(f"\nLoading Phase 2 from checkpoint: {args.checkpoint}")
        theta, phi, psi = load_phase2_checkpoint(
            path=args.checkpoint,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=args.hidden_dim,
            context_dim=args.context_dim,
            device=device,
        )

    # ==================== PHASE 3 ====================
    if not args.skip_phase3:
        print("\n" + "=" * 60)
        print("PHASE 3: Evaluating on test environments")
        print("=" * 60)
        print(f"  Test envs: crippled_leg={args.test_envs}")

        test_envs = [make_env(leg, args.task, args.normalize_env)
                     for leg in args.test_envs]

        evaluator = Phase3Evaluator(
            test_envs=test_envs,
            test_env_ids=args.test_envs,
            theta=theta,
            phi=phi,
            psi=psi,
            config=config,
            device=device,
        )

        results = evaluator.evaluate(
            n_episodes=args.eval_episodes,
            save_dir=save_dir,
        )

        print("\nPhase 3 complete!")
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Mean test reward: {results['aggregate']['mean_reward']:.2f}")
        print(f"  Std test reward:  {results['aggregate']['std_reward']:.2f}")

    print(f"\nExperiment complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
