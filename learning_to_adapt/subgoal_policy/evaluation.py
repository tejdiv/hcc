"""
Phase 3: GrBAL-style online test evaluation.

Like GrBAL Algorithm 2:
    For each timestep t:
        1. Adapt φ', ψ' on last M steps: D(t-M, t-1)
        2. Select action with adapted parameters
        3. Execute action, add to buffer D
"""

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from collections import OrderedDict

from .networks import (
    WorldModel,
    ContextEncoder,
    Policy,
    get_params_dict,
    set_params_dict,
)
from .trainers import compute_returns


class Phase3Evaluator:
    """
    Phase 3: GrBAL-style online evaluation on test environments.

    Like GrBAL Algorithm 2, at each timestep:
        1. θ'_t ← adapt on D(t-M, t-1) using last M steps
        2. a ← policy(θ'_t, s_t)
        3. Execute a, add to D

    This provides continuous online adaptation throughout the episode.
    """

    def __init__(
        self,
        test_envs: List,
        test_env_ids: List,
        world_model: WorldModel,
        phi: ContextEncoder,
        psi: Policy,
        config: Dict,
        device: torch.device = None,
    ):
        self.test_envs = test_envs
        self.test_env_ids = test_env_ids
        self.world_model = world_model
        self.phi = phi
        self.psi = psi
        self.config = config
        self.device = device or torch.device('cpu')

        self.world_model.to(self.device)
        self.phi.to(self.device)
        self.psi.to(self.device)

        # Freeze world model
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad = False

        # GrBAL-style hyperparameters
        self.M = config.get('test_M', config.get('M', 20))  # Adaptation window
        self.inner_lr = config.get('test_inner_lr', config.get('inner_lr', 0.01))
        self.adapt_steps = config.get('test_adapt_steps', config.get('adapt_steps', 1))
        self.gamma = config.get('gamma', 0.99)

    def compute_adaptation_loss(self, buffer, start_idx, end_idx):
        """
        Compute policy gradient loss on buffer segment [start_idx, end_idx).

        Context updates: c_t = f_φ(c_{t-1}, p̂_{t-1}, s_{t-1})
        where p̂_{t-1} = p̂(s_{t-1} | s_{t-2}, a_{t-2})
        """
        segment_len = end_idx - start_idx
        if segment_len < 1:
            return None

        states = torch.tensor(
            np.array(buffer['states'][start_idx:end_idx]),
            dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(buffer['actions'][start_idx:end_idx]),
            dtype=torch.float32, device=self.device
        )
        rewards = np.array(buffer['rewards'][start_idx:end_idx])

        # Compute returns
        returns = compute_returns(rewards, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Forward pass with context
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

    def adapt_online(self, buffer, current_t):
        """
        Perform online adaptation using last M steps.

        Like GrBAL: θ'_t ← u_ψ(D(t-M, t-1), θ)

        Args:
            buffer: experience buffer with states, actions, rewards
            current_t: current timestep

        Returns:
            adapted_phi_params, adapted_psi_params
        """
        # Get adaptation window: D(t-M, t-1)
        start_idx = max(0, current_t - self.M)
        end_idx = current_t  # Up to but not including current

        if end_idx - start_idx < 2:
            # Not enough data for adaptation
            return get_params_dict(self.phi), get_params_dict(self.psi)

        # Store original params
        original_phi_params = get_params_dict(self.phi)
        original_psi_params = get_params_dict(self.psi)

        adapted_phi_params = original_phi_params
        adapted_psi_params = original_psi_params

        for step in range(self.adapt_steps):
            # Set current params
            set_params_dict(self.phi, adapted_phi_params)
            set_params_dict(self.psi, adapted_psi_params)

            # Compute loss
            loss = self.compute_adaptation_loss(buffer, start_idx, end_idx)

            if loss is None:
                break

            # Compute gradients
            phi_grads = torch.autograd.grad(
                loss,
                list(self.phi.parameters()),
                create_graph=False,
                allow_unused=True,
            )
            psi_grads = torch.autograd.grad(
                loss,
                list(self.psi.parameters()),
                create_graph=False,
                allow_unused=True,
            )

            # Adapt parameters
            adapted_phi_params = OrderedDict()
            for (name, param), grad in zip(self.phi.named_parameters(), phi_grads):
                if grad is not None:
                    adapted_phi_params[name] = param.data - self.inner_lr * grad
                else:
                    adapted_phi_params[name] = param.data.clone()

            adapted_psi_params = OrderedDict()
            for (name, param), grad in zip(self.psi.named_parameters(), psi_grads):
                if grad is not None:
                    adapted_psi_params[name] = param.data - self.inner_lr * grad
                else:
                    adapted_psi_params[name] = param.data.clone()

        # Restore original params (we'll set adapted params when needed)
        set_params_dict(self.phi, original_phi_params)
        set_params_dict(self.psi, original_psi_params)

        return adapted_phi_params, adapted_psi_params

    def run_episode_online(
        self,
        env,
        max_steps: int = 1000,
        deterministic: bool = True,
        adapt: bool = True,
    ) -> Dict:
        """
        Run a single episode with GrBAL-style online adaptation.

        At each timestep:
            1. Adapt φ', ψ' on D(t-M, t-1)
            2. Update context c_t with adapted φ'
            3. Select action with adapted ψ'
            4. Execute action, add to buffer
        """
        # Experience buffer D
        buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
        }

        # Store original params
        original_phi_params = get_params_dict(self.phi)
        original_psi_params = get_params_dict(self.psi)

        s = env.reset()
        total_reward = 0.0
        rewards = []
        world_model_errors = []

        # Context (will be recomputed with adapted params)
        c = self.phi.reset_context(batch_size=1, device=self.device)

        for t in range(max_steps):
            # === Online Adaptation ===
            if adapt and t >= self.M:
                # Adapt using last M steps
                adapted_phi_params, adapted_psi_params = self.adapt_online(buffer, t)
                set_params_dict(self.phi, adapted_phi_params)
                set_params_dict(self.psi, adapted_psi_params)

            # === Update Context ===
            # c_t = f_φ(c_{t-1}, p̂_{t-1}, s_{t-1})
            # where p̂_{t-1} = p̂(s_{t-1} | s_{t-2}, a_{t-2})
            if t >= 2:
                s_t_minus_2 = torch.as_tensor(
                    buffer['states'][t-2], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                a_t_minus_2 = torch.as_tensor(
                    buffer['actions'][t-2], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                with torch.no_grad():
                    p_hat_prev = self.world_model(s_t_minus_2, a_t_minus_2)

                s_prev = torch.as_tensor(
                    buffer['states'][t-1], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                with torch.no_grad():
                    c = self.phi(c, p_hat_prev, s_prev)

                # Track world model error
                s_t_minus_1_actual = torch.as_tensor(
                    buffer['states'][t-1], dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                wm_error = torch.mean((p_hat_prev - s_t_minus_1_actual) ** 2).item()
                world_model_errors.append(wm_error)

            # === Select Action ===
            s_tensor = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    a = self.psi.mean_action(s_tensor, c)
                else:
                    a, _ = self.psi.sample(s_tensor, c)

            a_np = a.squeeze(0).cpu().numpy()

            # === Execute Action ===
            s_next, reward, done, info = env.step(a_np)

            # === Add to Buffer ===
            buffer['states'].append(s.copy() if isinstance(s, np.ndarray) else np.array(s))
            buffer['actions'].append(a_np)
            buffer['rewards'].append(reward)

            total_reward += reward
            rewards.append(reward)

            # Restore original params for next adaptation
            set_params_dict(self.phi, original_phi_params)
            set_params_dict(self.psi, original_psi_params)

            s = s_next
            if done:
                break

        return {
            'total_reward': total_reward,
            'episode_length': len(rewards),
            'mean_step_reward': np.mean(rewards) if rewards else 0.0,
            'mean_world_model_error': np.mean(world_model_errors) if world_model_errors else 0.0,
        }

    def evaluate_env(
        self,
        env,
        n_episodes: int = 10,
        adapt: bool = True,
        deterministic: bool = True,
    ) -> Dict:
        """Evaluate on a single environment with online adaptation."""
        max_steps = self.config.get('max_path_length', 1000)

        episode_results = []
        for _ in range(n_episodes):
            result = self.run_episode_online(
                env=env,
                max_steps=max_steps,
                deterministic=deterministic,
                adapt=adapt,
            )
            episode_results.append(result)

        rewards = [r['total_reward'] for r in episode_results]
        lengths = [r['episode_length'] for r in episode_results]
        wm_errors = [r['mean_world_model_error'] for r in episode_results]

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
            'mean_length': np.mean(lengths),
            'mean_world_model_error': np.mean(wm_errors),
            'n_episodes': n_episodes,
            'episode_rewards': rewards,
            'adapted': adapt,
        }

    def evaluate(
        self,
        n_episodes: int = 10,
        save_dir: str = None,
        adapt: bool = True,
    ) -> Dict:
        """Run GrBAL-style online evaluation on all test environments."""
        deterministic = self.config.get('deterministic_eval', True)

        print(f"Phase 3: GrBAL-style online evaluation on {len(self.test_envs)} test environments")
        print(f"  Episodes per env: {n_episodes}")
        print(f"  Online adaptation: {adapt}")
        print(f"  M (adaptation window): {self.M}")
        print(f"  Inner LR: {self.inner_lr}")
        print(f"  Adapt steps: {self.adapt_steps}")
        print(f"  Deterministic: {deterministic}")

        results = {}
        all_rewards = []

        for env, env_id in zip(self.test_envs, self.test_env_ids):
            print(f"  Evaluating env {env_id}...")

            # Evaluate with adaptation
            env_results = self.evaluate_env(
                env=env,
                n_episodes=n_episodes,
                adapt=adapt,
                deterministic=deterministic,
            )

            # Also evaluate without adaptation for comparison
            if adapt:
                no_adapt_results = self.evaluate_env(
                    env=env,
                    n_episodes=min(3, n_episodes),
                    adapt=False,
                    deterministic=deterministic,
                )
                env_results['no_adapt_mean_reward'] = no_adapt_results['mean_reward']

            results[f'env_{env_id}'] = env_results
            all_rewards.extend(env_results['episode_rewards'])

            no_adapt_str = f" (no-adapt: {env_results.get('no_adapt_mean_reward', 'N/A'):.2f})" if adapt else ""
            print(f"    Mean reward: {env_results['mean_reward']:.2f} "
                  f"(+/- {env_results['std_reward']:.2f}){no_adapt_str}")

        # Aggregate
        results['aggregate'] = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'n_total_episodes': len(all_rewards),
        }

        print(f"\nAggregate results:")
        print(f"  Mean reward: {results['aggregate']['mean_reward']:.2f} "
              f"(+/- {results['aggregate']['std_reward']:.2f})")

        if save_dir:
            self.save_results(save_dir, results)

        return results

    def save_results(self, save_dir: str, results: Dict):
        """Save evaluation results."""
        os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
        path = os.path.join(save_dir, 'results', 'test_results.json')

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
    adapt: bool = True,
) -> Dict:
    """Convenience function to run evaluation from a checkpoint."""
    from .trainers import load_phase2_checkpoint

    world_model, phi, psi = load_phase2_checkpoint(
        path=checkpoint_path,
        obs_dim=config['obs_dim'],
        act_dim=config['act_dim'],
        context_dim=config.get('context_dim', 64),
        policy_hidden_sizes=config.get('policy_hidden_sizes', (256, 256)),
        world_model_hidden_sizes=config.get('world_model_hidden_sizes', (512, 512, 512)),
        device=device,
    )

    evaluator = Phase3Evaluator(
        test_envs=test_envs,
        test_env_ids=test_env_ids,
        world_model=world_model,
        phi=phi,
        psi=psi,
        config=config,
        device=device,
    )

    return evaluator.evaluate(
        n_episodes=config.get('eval_episodes', 10),
        save_dir=config.get('save_dir'),
        adapt=adapt,
    )
