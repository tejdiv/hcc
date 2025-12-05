"""
Core networks for subgoal-conditioned policy.

f_theta: State Proposer - predicts next state goal X_t from s_t
f_phi:   Context Encoder - GRU that tracks world model error
pi_psi:  Policy - outputs action distribution given (s_t, X_t, c_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class StateProposer(nn.Module):
    """
    f_theta: s_t -> X_t

    Proposes a "goal" next state. Trained to be close to actual s_{t+1}.
    """

    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )

    def forward(self, s):
        """
        Args:
            s: state tensor (batch, obs_dim) or (obs_dim,)
        Returns:
            X: proposed next state (same shape as s)
        """
        return self.net(s)


class ContextEncoder(nn.Module):
    """
    f_phi: GRU-based context encoder.

    Tracks implicit world model error via:
        c_t = GRU(c_{t-1}, [X_{t-1}, s_{t-1}])

    The concatenation [X_{t-1}, s_{t-1}] lets the GRU see
    "what I predicted" vs "what actually happened".
    """

    def __init__(self, obs_dim, context_dim=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_dim = context_dim

        # Input: [X_{t-1}, s_{t-1}] = obs_dim * 2
        self.gru = nn.GRUCell(input_size=obs_dim * 2, hidden_size=context_dim)

    def reset_context(self, batch_size=1, device=None):
        """Initialize context to zeros."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.context_dim, device=device)

    def forward(self, c_prev, X_prev, s_prev):
        """
        Update context given previous prediction and actual state.

        Args:
            c_prev: previous context (batch, context_dim)
            X_prev: previous predicted state (batch, obs_dim)
            s_prev: previous actual state (batch, obs_dim)

        Returns:
            c_new: updated context (batch, context_dim)
        """
        # Concatenate prediction and reality
        gru_input = torch.cat([X_prev, s_prev], dim=-1)
        c_new = self.gru(gru_input, c_prev)
        return c_new


class Policy(nn.Module):
    """
    pi_psi: Gaussian policy.

    Outputs action distribution given (s_t, X_t, c_t).
    Uses tanh squashing for bounded actions.
    """

    def __init__(self, obs_dim, act_dim, context_dim=16, hidden_dim=64,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Input: [s_t, X_t, c_t]
        input_dim = obs_dim * 2 + context_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, s, X, c):
        """
        Compute mean and log_std of action distribution.

        Args:
            s: current state (batch, obs_dim)
            X: proposed next state (batch, obs_dim)
            c: context (batch, context_dim)

        Returns:
            mean: action mean (batch, act_dim)
            log_std: action log std (batch, act_dim)
        """
        x = torch.cat([s, X, c], dim=-1)
        features = self.net(x)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_distribution(self, s, X, c):
        """Get the action distribution."""
        mean, log_std = self.forward(s, X, c)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, s, X, c):
        """
        Sample action with reparameterization trick.

        Returns:
            action: sampled action (batch, act_dim)
            log_prob: log probability of action (batch,)
        """
        mean, log_std = self.forward(s, X, c)
        std = torch.exp(log_std)

        # Reparameterization trick
        noise = torch.randn_like(mean)
        action_raw = mean + std * noise

        # Tanh squashing
        action = torch.tanh(action_raw)

        # Log prob with tanh correction
        log_prob = self._log_prob_from_raw(action_raw, mean, std)

        return action, log_prob

    def log_prob(self, s, X, c, action):
        """
        Compute log probability of a given action.

        Args:
            action: action in [-1, 1] (already squashed)

        Returns:
            log_prob: log probability (batch,)
        """
        mean, log_std = self.forward(s, X, c)
        std = torch.exp(log_std)

        # Inverse tanh to get raw action
        # Clamp to avoid numerical issues at boundaries
        action_clamped = torch.clamp(action, -0.999, 0.999)
        action_raw = torch.atanh(action_clamped)

        return self._log_prob_from_raw(action_raw, mean, std)

    def _log_prob_from_raw(self, action_raw, mean, std):
        """Compute log prob from raw (pre-tanh) action."""
        # Gaussian log prob
        var = std ** 2
        log_prob = -0.5 * (((action_raw - mean) ** 2) / var +
                          2 * torch.log(std) +
                          np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        # Tanh correction: log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - torch.tanh(action_raw) ** 2 + 1e-6).sum(dim=-1)

        return log_prob

    def mean_action(self, s, X, c):
        """Get deterministic (mean) action for evaluation."""
        mean, _ = self.forward(s, X, c)
        return torch.tanh(mean)


class SubgoalPolicyNetwork(nn.Module):
    """
    Combined network containing all three components.
    Convenience wrapper for saving/loading.
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=64, context_dim=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        self.theta = StateProposer(obs_dim, hidden_dim)
        self.phi = ContextEncoder(obs_dim, context_dim)
        self.psi = Policy(obs_dim, act_dim, context_dim, hidden_dim)

    def forward(self, s, c_prev=None, X_prev=None, s_prev=None):
        """
        Full forward pass.

        Args:
            s: current state
            c_prev: previous context (None for t=0)
            X_prev: previous prediction (None for t=0)
            s_prev: previous state (None for t=0)

        Returns:
            action, log_prob, X, c
        """
        # Propose next state
        X = self.theta(s)

        # Update context (if not first step)
        if c_prev is None:
            c = self.phi.reset_context(batch_size=s.shape[0] if s.dim() > 1 else 1,
                                       device=s.device)
        else:
            c = self.phi(c_prev, X_prev, s_prev)

        # Sample action
        action, log_prob = self.psi.sample(s, X, c)

        return action, log_prob, X, c

    def get_params_phase1(self):
        """Get parameters for Phase 1 (theta + psi)."""
        return list(self.theta.parameters()) + list(self.psi.parameters())

    def get_params_phase2(self):
        """Get parameters for Phase 2 (phi + psi)."""
        return list(self.phi.parameters()) + list(self.psi.parameters())

    def freeze_theta(self):
        """Freeze theta for Phase 2."""
        self.theta.eval()
        for p in self.theta.parameters():
            p.requires_grad = False

    def unfreeze_theta(self):
        """Unfreeze theta (if needed)."""
        self.theta.train()
        for p in self.theta.parameters():
            p.requires_grad = True
