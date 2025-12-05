"""
Phase 3: Test evaluation.

All weights frozen. Only c_t updates via RNN forward pass.
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional

from .networks import StateProposer, ContextEncoder, Policy


@torch.no_grad()
def evaluate_single_episode(
    env,
    theta: StateProposer,
    phi: ContextEncoder,
    psi: Policy,
    max_steps: int = 1000,
    device: torch.device = None,
    deterministic: bool = True,
) -> Dict:
    """
    Evaluate a single episode.

    Args:
        env: environment instance
        theta: StateProposer (frozen)
        phi: ContextEncoder (frozen, but c_t updates via forward pass)
        psi: Policy (frozen)
        max_steps: max episode length
        device: torch device
        deterministic: if True, use mean action; else sample

    Returns:
        dict with episode stats
    """
    if device is None:
        device = next(psi.parameters()).device

    s = env.reset()
    c = phi.reset_context(batch_size=1, device=device)

    X_prev = None
    s_prev = None

    total_reward = 0.0
    rewards = []
    states = []
    predictions = []

    for t in range(max_steps):
        s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)

        # Propose next state
        X = theta(s_tensor)

        # Update context (skip at t=0)
        if t > 0:
            X_prev_tensor = torch.as_tensor(X_prev, dtype=torch.float32, device=device).unsqueeze(0)
            s_prev_tensor = torch.as_tensor(s_prev, dtype=torch.float32, device=device).unsqueeze(0)
            c = phi(c, X_prev_tensor, s_prev_tensor)

        # Get action
        if deterministic:
            a = psi.mean_action(s_tensor, X, c)
        else:
            a, _ = psi.sample(s_tensor, X, c)

        a_np = a.squeeze(0).cpu().numpy()
        X_np = X.squeeze(0).cpu().numpy()

        # Step environment
        s_next, reward, done, info = env.step(a_np)

        total_reward += reward
        rewards.append(reward)
        states.append(s.copy() if isinstance(s, np.ndarray) else np.array(s))
        predictions.append(X_np)

        # Store for context update
        X_prev = X_np
        s_prev = s.copy() if isinstance(s, np.ndarray) else np.array(s)
        s = s_next

        if done:
            break

    # Compute prediction error
    states_arr = np.array(states[:-1])  # s_0 to s_{T-1}
    predictions_arr = np.array(predictions[:-1])  # X_0 to X_{T-1}
    next_states_arr = np.array(states[1:])  # s_1 to s_T

    if len(states_arr) > 0:
        pred_error = np.mean((predictions_arr - next_states_arr) ** 2)
    else:
        pred_error = 0.0

    return {
        'total_reward': total_reward,
        'episode_length': len(rewards),
        'mean_step_reward': np.mean(rewards),
        'prediction_error': pred_error,
    }


@torch.no_grad()
def evaluate_on_env(
    env,
    theta: StateProposer,
    phi: ContextEncoder,
    psi: Policy,
    n_episodes: int = 10,
    max_steps: int = 1000,
    device: torch.device = None,
    deterministic: bool = True,
) -> Dict:
    """
    Evaluate on a single environment over multiple episodes.

    Args:
        env: environment instance
        theta, phi, psi: networks (all frozen)
        n_episodes: number of evaluation episodes
        max_steps: max steps per episode
        device: torch device
        deterministic: use mean action if True

    Returns:
        dict with aggregated stats
    """
    episode_results = []

    for ep in range(n_episodes):
        result = evaluate_single_episode(
            env=env,
            theta=theta,
            phi=phi,
            psi=psi,
            max_steps=max_steps,
            device=device,
            deterministic=deterministic,
        )
        episode_results.append(result)

    # Aggregate
    rewards = [r['total_reward'] for r in episode_results]
    lengths = [r['episode_length'] for r in episode_results]
    pred_errors = [r['prediction_error'] for r in episode_results]

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_length': np.mean(lengths),
        'mean_prediction_error': np.mean(pred_errors),
        'n_episodes': n_episodes,
        'episode_rewards': rewards,
    }


class Phase3Evaluator:
    """
    Phase 3: Evaluate on test environments.

    All weights frozen. Only c_t updates via f_phi forward pass.
    """

    def __init__(
        self,
        test_envs: List,
        test_env_ids: List,
        theta: StateProposer,
        phi: ContextEncoder,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
    ):
        """
        Args:
            test_envs: list of test environment instances
            test_env_ids: list of env identifiers (e.g., crippled_leg values)
            theta, phi, psi: trained networks
            config: configuration dict
            device: torch device
        """
        self.test_envs = test_envs
        self.test_env_ids = test_env_ids
        self.theta = theta
        self.phi = phi
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        # Move to device and set to eval mode
        self.theta.to(self.device).eval()
        self.phi.to(self.device).eval()
        self.psi.to(self.device).eval()

        # Freeze all parameters
        for net in [self.theta, self.phi, self.psi]:
            for p in net.parameters():
                p.requires_grad = False

    def evaluate(
        self,
        n_episodes: int = 10,
        save_dir: str = None,
    ) -> Dict:
        """
        Run evaluation on all test environments.

        Args:
            n_episodes: episodes per environment
            save_dir: directory to save results

        Returns:
            dict with results per environment and aggregate stats
        """
        max_steps = self.config.get('max_path_length', 1000)
        deterministic = self.config.get('deterministic_eval', True)

        print(f"Phase 3: Evaluating on {len(self.test_envs)} test environments")
        print(f"  Episodes per env: {n_episodes}")
        print(f"  Deterministic: {deterministic}")

        results = {}
        all_rewards = []

        for env, env_id in zip(self.test_envs, self.test_env_ids):
            print(f"  Evaluating env {env_id}...")

            env_results = evaluate_on_env(
                env=env,
                theta=self.theta,
                phi=self.phi,
                psi=self.psi,
                n_episodes=n_episodes,
                max_steps=max_steps,
                device=self.device,
                deterministic=deterministic,
            )

            results[f'env_{env_id}'] = env_results
            all_rewards.extend(env_results['episode_rewards'])

            print(f"    Mean reward: {env_results['mean_reward']:.2f} "
                  f"(+/- {env_results['std_reward']:.2f})")

        # Aggregate across all envs
        results['aggregate'] = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'n_total_episodes': len(all_rewards),
        }

        print(f"\nAggregate results:")
        print(f"  Mean reward: {results['aggregate']['mean_reward']:.2f} "
              f"(+/- {results['aggregate']['std_reward']:.2f})")

        # Save results
        if save_dir:
            self.save_results(save_dir, results)

        return results

    def save_results(self, save_dir: str, results: Dict):
        """Save evaluation results."""
        os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
        path = os.path.join(save_dir, 'results', 'test_results.json')

        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)

        print(f"Results saved to {path}")


def run_evaluation(
    test_envs: List,
    test_env_ids: List,
    checkpoint_path: str,
    config: Dict,
    device: torch.device = None,
) -> Dict:
    """
    Convenience function to run evaluation from a checkpoint.

    Args:
        test_envs: list of test environments
        test_env_ids: environment identifiers
        checkpoint_path: path to Phase 2 checkpoint
        config: configuration dict
        device: torch device

    Returns:
        evaluation results
    """
    from .trainers import load_phase2_checkpoint

    # Load checkpoint
    theta, phi, psi = load_phase2_checkpoint(
        path=checkpoint_path,
        obs_dim=config['obs_dim'],
        act_dim=config['act_dim'],
        hidden_dim=config.get('hidden_dim', 64),
        context_dim=config.get('context_dim', 16),
        device=device,
    )

    # Create evaluator
    evaluator = Phase3Evaluator(
        test_envs=test_envs,
        test_env_ids=test_env_ids,
        theta=theta,
        phi=phi,
        psi=psi,
        config=config,
        device=device,
    )

    # Run evaluation
    return evaluator.evaluate(
        n_episodes=config.get('eval_episodes', 10),
        save_dir=config.get('save_dir'),
    )
