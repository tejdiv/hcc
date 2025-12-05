#!/usr/bin/env python
"""
Main entry point for GrBAL-style Subgoal-Conditioned Policy training.

Architecture:
- Phase 1: Train π_ψ(a|s) simple policy + p̂(s_{t+1}|s_t,a_t) world model on single env
- Phase 2: GrBAL-style MAML training with segment sampling:
    - Sample τ(t-M, t-1) for adaptation, τ(t, t+K) for evaluation
    - Inner loop: adapt φ, ψ on adaptation segment
    - Outer loop: compute loss on evaluation segment
- Phase 3: GrBAL-style online test-time adaptation at every timestep

Usage:
    # Run full pipeline
    python run_scripts/run_subgoal_policy.py

    # Custom experiment name
    python run_scripts/run_subgoal_policy.py --exp_name my_experiment

    # Custom train/test split
    python run_scripts/run_subgoal_policy.py --initial_env 1 --train_envs 2 3 4 --test_envs 5

    # Skip phases
    python run_scripts/run_subgoal_policy.py --skip_phase1 --skip_phase2 --checkpoint path/to/phase2.pt
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
    WorldModel,
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
        description='MAML-style Subgoal-Conditioned Policy Training',
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

    # === Architecture (~275K params, half of GrBAL) ===
    parser.add_argument('--context_dim', type=int, default=32,
                        help='Context vector dimension')
    parser.add_argument('--policy_hidden_sizes', type=int, nargs='+', default=[128, 128],
                        help='Policy network hidden layer sizes')
    parser.add_argument('--world_model_hidden_sizes', type=int, nargs='+', default=[256, 256],
                        help='World model hidden layer sizes')

    # === Phase 1 Training ===
    parser.add_argument('--phase1_iterations', type=int, default=100,
                        help='Phase 1 training iterations')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Trajectories per Phase 1 iteration')
    parser.add_argument('--policy_lr', type=float, default=3e-4,
                        help='Policy learning rate')
    parser.add_argument('--world_model_lr', type=float, default=1e-3,
                        help='World model learning rate')
    parser.add_argument('--world_model_train_freq', type=int, default=5,
                        help='Train world model every N iterations')
    parser.add_argument('--world_model_epochs', type=int, default=10,
                        help='World model training epochs per update')
    parser.add_argument('--world_model_batch_size', type=int, default=256,
                        help='World model minibatch size')

    # === Phase 2 GrBAL-style MAML Training ===
    parser.add_argument('--phase2_iterations', type=int, default=100,
                        help='Phase 2 training iterations')
    parser.add_argument('--meta_lr', type=float, default=1e-3,
                        help='Meta-learning rate (outer loop)')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                        help='Inner loop learning rate')
    parser.add_argument('--adapt_steps', type=int, default=1,
                        help='Number of gradient steps in inner loop')
    parser.add_argument('--M', type=int, default=20,
                        help='Adaptation segment length τ(t-M, t-1)')
    parser.add_argument('--K', type=int, default=20,
                        help='Evaluation segment length τ(t, t+K)')
    parser.add_argument('--task_sampling_freq', type=int, default=1,
                        help='Collect new trajectories every N iterations (n_S in GrBAL)')
    parser.add_argument('--max_trajs_per_env', type=int, default=50,
                        help='Max trajectories to keep per environment in dataset')
    parser.add_argument('--meta_batch_size', type=int, default=None,
                        help='Tasks per meta-batch (default: all train envs)')

    # === Phase 3 GrBAL-style Online Evaluation ===
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate on test envs every N iterations during Phase 2 (for sample efficiency)')
    parser.add_argument('--eval_episodes', type=int, default=10,
                        help='Episodes per test environment')
    parser.add_argument('--test_M', type=int, default=None,
                        help='Adaptation window at test time (default: same as M)')
    parser.add_argument('--test_adapt_steps', type=int, default=None,
                        help='Gradient steps at test time (default: same as adapt_steps)')
    parser.add_argument('--test_inner_lr', type=float, default=None,
                        help='Inner LR at test time (default: same as training)')
    parser.add_argument('--no_adapt', action='store_true',
                        help='Skip online adaptation at test time')
    parser.add_argument('--deterministic_eval', action='store_true', default=True,
                        help='Use deterministic (mean) actions for evaluation')

    # === PPO Hyperparameters ===
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
    parser.add_argument('--entropy_coef', type=float, default=0.01,
                        help='Entropy bonus coefficient')

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

    # Convert hidden sizes to tuples
    policy_hidden_sizes = tuple(args.policy_hidden_sizes)
    world_model_hidden_sizes = tuple(args.world_model_hidden_sizes)

    # Save config
    config = vars(args)
    config['policy_hidden_sizes'] = policy_hidden_sizes
    config['world_model_hidden_sizes'] = world_model_hidden_sizes
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Get observation and action dimensions
    sample_env = make_env(args.initial_env, args.task, args.normalize_env)
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.shape[0]
    print(f"Observation dim: {obs_dim}, Action dim: {act_dim}")

    # Update config with dims
    config['obs_dim'] = obs_dim
    config['act_dim'] = act_dim
    config['save_dir'] = save_dir

    # Set test-time defaults from training values if not specified
    if config['test_M'] is None:
        config['test_M'] = config['M']
    if config['test_adapt_steps'] is None:
        config['test_adapt_steps'] = config['adapt_steps']
    if config['test_inner_lr'] is None:
        config['test_inner_lr'] = config['inner_lr']

    # Initialize networks
    world_model = WorldModel(obs_dim, act_dim, world_model_hidden_sizes)
    phi = ContextEncoder(obs_dim, args.context_dim)
    psi = Policy(obs_dim, act_dim, args.context_dim, policy_hidden_sizes)

    # Print parameter counts
    wm_params = sum(p.numel() for p in world_model.parameters())
    phi_params = sum(p.numel() for p in phi.parameters())
    psi_params = sum(p.numel() for p in psi.parameters())

    print(f"\nNetwork parameters:")
    print(f"  WorldModel (p̂):        {wm_params:,}")
    print(f"  ContextEncoder (φ):    {phi_params:,}")
    print(f"  Policy (ψ):            {psi_params:,}")
    print(f"  Total:                 {wm_params + phi_params + psi_params:,}")

    # ==================== PHASE 1 ====================
    if not args.skip_phase1:
        print("\n" + "=" * 60)
        print("PHASE 1: Training policy + world model on initial environment")
        print("=" * 60)
        print(f"  Initial env: crippled_leg={args.initial_env}")
        print(f"  Policy: π_ψ(a|s) - simple, no context")
        print(f"  World model: p̂(s_{t+1}|s_t, a_t)")

        initial_env = make_env(args.initial_env, args.task, args.normalize_env)

        phase1_trainer = Phase1Trainer(
            env=initial_env,
            world_model=world_model,
            psi=psi,
            config=config,
            device=device,
        )

        world_model, psi = phase1_trainer.train(
            n_iterations=args.phase1_iterations,
            save_dir=save_dir,
        )

        print("Phase 1 complete!")

    elif args.checkpoint:
        print(f"\nLoading Phase 1 from checkpoint: {args.checkpoint}")
        world_model, psi = load_phase1_checkpoint(
            path=args.checkpoint,
            obs_dim=obs_dim,
            act_dim=act_dim,
            context_dim=args.context_dim,
            policy_hidden_sizes=policy_hidden_sizes,
            world_model_hidden_sizes=world_model_hidden_sizes,
            device=device,
        )

    # ==================== PHASE 2 ====================
    if not args.skip_phase2:
        print("\n" + "=" * 60)
        print("PHASE 2: GrBAL-style MAML meta-training across environments")
        print("=" * 60)
        print(f"  Training envs: crippled_leg={args.train_envs}")
        print(f"  Test envs (for sample efficiency): crippled_leg={args.test_envs}")
        print(f"  World model: FROZEN")
        print(f"  MAML training: φ (context encoder) + ψ (policy)")
        print(f"  M (adapt segment): {args.M}, K (eval segment): {args.K}")
        print(f"  Inner LR: {args.inner_lr}, Adapt steps: {args.adapt_steps}")

        train_envs = [make_env(leg, args.task, args.normalize_env)
                      for leg in args.train_envs]

        # Create test envs for periodic evaluation during training (sample efficiency)
        test_envs_for_eval = [make_env(leg, args.task, args.normalize_env)
                              for leg in args.test_envs]

        phase2_trainer = Phase2Trainer(
            train_envs=train_envs,
            world_model=world_model,
            phi=phi,
            psi=psi,
            config=config,
            device=device,
            test_envs=test_envs_for_eval,
            test_env_ids=args.test_envs,
        )

        phi, psi = phase2_trainer.train(
            n_iterations=args.phase2_iterations,
            save_dir=save_dir,
        )

        print("Phase 2 complete!")

    elif args.checkpoint and args.skip_phase1:
        print(f"\nLoading Phase 2 from checkpoint: {args.checkpoint}")
        world_model, phi, psi = load_phase2_checkpoint(
            path=args.checkpoint,
            obs_dim=obs_dim,
            act_dim=act_dim,
            context_dim=args.context_dim,
            policy_hidden_sizes=policy_hidden_sizes,
            world_model_hidden_sizes=world_model_hidden_sizes,
            device=device,
        )

    # ==================== PHASE 3 ====================
    if not args.skip_phase3:
        print("\n" + "=" * 60)
        print("PHASE 3: GrBAL-style online evaluation on test environments")
        print("=" * 60)
        print(f"  Test envs: crippled_leg={args.test_envs}")
        print(f"  Online adaptation: {not args.no_adapt}")
        if not args.no_adapt:
            test_M = args.test_M or args.M
            test_adapt_steps = args.test_adapt_steps or args.adapt_steps
            print(f"  M (adaptation window): {test_M}")
            print(f"  Adapt steps: {test_adapt_steps}")
            print(f"  Inner LR: {args.test_inner_lr or args.inner_lr}")

        test_envs = [make_env(leg, args.task, args.normalize_env)
                     for leg in args.test_envs]

        evaluator = Phase3Evaluator(
            test_envs=test_envs,
            test_env_ids=args.test_envs,
            world_model=world_model,
            phi=phi,
            psi=psi,
            config=config,
            device=device,
        )

        results = evaluator.evaluate(
            n_episodes=args.eval_episodes,
            save_dir=save_dir,
            adapt=not args.no_adapt,
        )

        print("\nPhase 3 complete!")
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Mean test reward (post-adapt): {results['aggregate']['mean_reward']:.2f}")
        if results['aggregate'].get('pre_adapt_mean_reward'):
            print(f"  Mean test reward (pre-adapt):  {results['aggregate']['pre_adapt_mean_reward']:.2f}")

    print(f"\nExperiment complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
