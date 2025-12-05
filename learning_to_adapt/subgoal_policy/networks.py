"""
Core networks for subgoal-conditioned policy with MAML-style adaptation.

WorldModel (p̂): Predicts s_{t+1} from (s_t, a_t) - trained in Phase 1, frozen after
ContextEncoder (f_φ): GRU that tracks world model error: c_t = f(c_{t-1}, p̂_{t-1}, s_{t-1})
Policy (π_ψ): Outputs action distribution
  - Phase 1: π_ψ(a|s) - simple, no context
  - Phase 2/3: π_ψ(a|s, c_t) - context-conditioned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import OrderedDict


class WorldModel(nn.Module):
    """
    p̂: (s_t, a_t) -> s_{t+1}

    Predicts next state. Trained in Phase 1 on rollouts, frozen after.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Build MLP layers
        layers = []
        input_dim = obs_dim + act_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, obs_dim))  # Output: delta (s_{t+1} - s_t)

        self.net = nn.Sequential(*layers)

        # Normalization stats (computed during training)
        self.register_buffer('obs_mean', torch.zeros(obs_dim))
        self.register_buffer('obs_std', torch.ones(obs_dim))
        self.register_buffer('act_mean', torch.zeros(act_dim))
        self.register_buffer('act_std', torch.ones(act_dim))
        self.register_buffer('delta_mean', torch.zeros(obs_dim))
        self.register_buffer('delta_std', torch.ones(obs_dim))

    def forward(self, s, a, normalized=False):
        """
        Predict next state.

        Args:
            s: current state (batch, obs_dim)
            a: action (batch, act_dim)
            normalized: if True, inputs are already normalized

        Returns:
            s_next: predicted next state (batch, obs_dim)
        """
        if not normalized:
            s_norm = (s - self.obs_mean) / (self.obs_std + 1e-8)
            a_norm = (a - self.act_mean) / (self.act_std + 1e-8)
        else:
            s_norm = s
            a_norm = a

        x = torch.cat([s_norm, a_norm], dim=-1)
        delta_norm = self.net(x)

        # Denormalize delta and add to state
        delta = delta_norm * (self.delta_std + 1e-8) + self.delta_mean
        s_next = s + delta

        return s_next

    def compute_normalization(self, obs, act, obs_next):
        """
        Compute normalization statistics from data.

        Args:
            obs: states (N, obs_dim)
            act: actions (N, act_dim)
            obs_next: next states (N, obs_dim)
        """
        delta = obs_next - obs

        self.obs_mean = torch.mean(obs, dim=0)
        self.obs_std = torch.std(obs, dim=0)
        self.act_mean = torch.mean(act, dim=0)
        self.act_std = torch.std(act, dim=0)
        self.delta_mean = torch.mean(delta, dim=0)
        self.delta_std = torch.std(delta, dim=0)

    def loss(self, s, a, s_next):
        """Compute prediction loss (MSE on delta)."""
        s_pred = self.forward(s, a)
        return F.mse_loss(s_pred, s_next)


class ContextEncoder(nn.Module):
    """
    f_φ: GRU-based context encoder.

    Tracks world model prediction error:
        c_t = GRU(c_{t-1}, [p̂_{t-1}, s_{t-1}])

    where:
        p̂_{t-1} = p̂(s_{t-1} | s_{t-2}, a_{t-2}) = prediction OF s_{t-1}, made at t-2
        s_{t-1} = actual state at t-1

    The GRU sees the prediction error: how wrong was the world model
    when predicting s_{t-1}? This implicitly encodes information about
    which environment/dynamics we're in.

    Note: Context updates start at t=2 (need 2 steps of history).
    """

    def __init__(self, obs_dim, context_dim=32, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_dim = context_dim

        # Input: [p̂_{t-1}, s_{t-1}] = obs_dim * 2
        self.input_proj = nn.Linear(obs_dim * 2, hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=context_dim)

    def reset_context(self, batch_size=1, device=None):
        """Initialize context to zeros."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.context_dim, device=device)

    def forward(self, c_prev, p_hat_prev, s_prev):
        """
        Update context given world model prediction and actual state.

        Args:
            c_prev: previous context (batch, context_dim)
            p_hat_prev: world model's prediction of s_t made at t-1 (batch, obs_dim)
            s_prev: actual previous state s_{t-1} (batch, obs_dim)

        Returns:
            c_new: updated context (batch, context_dim)
        """
        # Concatenate prediction and reality
        gru_input = torch.cat([p_hat_prev, s_prev], dim=-1)
        gru_input = F.relu(self.input_proj(gru_input))
        c_new = self.gru(gru_input, c_prev)
        return c_new

    def get_adapted_params(self, params_dict=None):
        """Get parameters as ordered dict for MAML adaptation."""
        if params_dict is None:
            params_dict = OrderedDict()
            for name, param in self.named_parameters():
                params_dict[name] = param
        return params_dict


class Policy(nn.Module):
    """
    π_ψ: Gaussian policy.

    Two modes:
    - Phase 1: π_ψ(a|s) - simple policy, no context
    - Phase 2/3: π_ψ(a|s, c_t) - context-conditioned

    Uses tanh squashing for bounded actions.
    Architecture scaled up for more capacity.
    """

    def __init__(self, obs_dim, act_dim, context_dim=32, hidden_sizes=(128, 128),
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = context_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Phase 1: simple policy (no context)
        self.simple_net = self._build_mlp(obs_dim, hidden_sizes)
        self.simple_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.simple_log_std = nn.Linear(hidden_sizes[-1], act_dim)

        # Phase 2/3: context-conditioned policy
        input_dim = obs_dim + context_dim
        self.context_net = self._build_mlp(input_dim, hidden_sizes)
        self.context_mean = nn.Linear(hidden_sizes[-1], act_dim)
        self.context_log_std = nn.Linear(hidden_sizes[-1], act_dim)

    def _build_mlp(self, input_dim, hidden_sizes):
        layers = []
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        return nn.Sequential(*layers)

    def forward(self, s, c=None):
        """
        Compute mean and log_std of action distribution.

        Args:
            s: current state (batch, obs_dim)
            c: context (batch, context_dim) - None for Phase 1

        Returns:
            mean: action mean (batch, act_dim)
            log_std: action log std (batch, act_dim)
        """
        if c is None:
            # Phase 1: simple policy
            features = self.simple_net(s)
            mean = self.simple_mean(features)
            log_std = self.simple_log_std(features)
        else:
            # Phase 2/3: context-conditioned
            x = torch.cat([s, c], dim=-1)
            features = self.context_net(x)
            mean = self.context_mean(features)
            log_std = self.context_log_std(features)

        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_distribution(self, s, c=None):
        """Get the action distribution."""
        mean, log_std = self.forward(s, c)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def sample(self, s, c=None):
        """
        Sample action with reparameterization trick.

        Returns:
            action: sampled action (batch, act_dim)
            log_prob: log probability of action (batch,)
        """
        mean, log_std = self.forward(s, c)
        std = torch.exp(log_std)

        # Reparameterization trick
        noise = torch.randn_like(mean)
        action_raw = mean + std * noise

        # Tanh squashing
        action = torch.tanh(action_raw)

        # Log prob with tanh correction
        log_prob = self._log_prob_from_raw(action_raw, mean, std)

        return action, log_prob

    def log_prob(self, s, c, action):
        """
        Compute log probability of a given action.

        Args:
            s: state
            c: context (can be None for Phase 1)
            action: action in [-1, 1] (already squashed)

        Returns:
            log_prob: log probability (batch,)
        """
        mean, log_std = self.forward(s, c)
        std = torch.exp(log_std)

        # Inverse tanh to get raw action
        action_clamped = torch.clamp(action, -0.999, 0.999)
        action_raw = torch.atanh(action_clamped)

        return self._log_prob_from_raw(action_raw, mean, std)

    def _log_prob_from_raw(self, action_raw, mean, std):
        """Compute log prob from raw (pre-tanh) action."""
        var = std ** 2
        log_prob = -0.5 * (((action_raw - mean) ** 2) / var +
                          2 * torch.log(std) +
                          np.log(2 * np.pi))
        log_prob = log_prob.sum(dim=-1)

        # Tanh correction: log(1 - tanh(x)^2)
        log_prob -= torch.log(1 - torch.tanh(action_raw) ** 2 + 1e-6).sum(dim=-1)

        return log_prob

    def mean_action(self, s, c=None):
        """Get deterministic (mean) action for evaluation."""
        mean, _ = self.forward(s, c)
        return torch.tanh(mean)

    def get_adapted_params(self, params_dict=None):
        """Get parameters as ordered dict for MAML adaptation."""
        if params_dict is None:
            params_dict = OrderedDict()
            for name, param in self.named_parameters():
                params_dict[name] = param
        return params_dict


# ============== MAML Utilities ==============

def clone_params(params):
    """Clone parameters for MAML inner loop."""
    return OrderedDict({k: v.clone() for k, v in params.items()})


def get_params_dict(module):
    """Get module parameters as ordered dict."""
    return OrderedDict({name: param for name, param in module.named_parameters()})


def set_params_dict(module, params_dict):
    """Set module parameters from ordered dict."""
    for name, param in module.named_parameters():
        if name in params_dict:
            param.data.copy_(params_dict[name].data)


def functional_forward_policy(params, s, c, obs_dim, act_dim, context_dim, hidden_sizes, log_std_min, log_std_max):
    """
    Functional forward pass for policy (for MAML gradient computation).
    Uses context-conditioned path only (Phase 2/3).
    """
    x = torch.cat([s, c], dim=-1)

    # context_net forward
    idx = 0
    for i, hidden_dim in enumerate(hidden_sizes):
        weight = params[f'context_net.{idx}.weight']
        bias = params[f'context_net.{idx}.bias']
        x = F.linear(x, weight, bias)
        x = F.relu(x)
        idx += 2  # Skip ReLU (no params)

    # Mean and log_std heads
    mean = F.linear(x, params['context_mean.weight'], params['context_mean.bias'])
    log_std = F.linear(x, params['context_log_std.weight'], params['context_log_std.bias'])
    log_std = torch.clamp(log_std, log_std_min, log_std_max)

    return mean, log_std


def functional_forward_context_encoder(params, c_prev, p_hat_prev, s_prev):
    """
    Functional forward pass for context encoder (for MAML gradient computation).
    """
    # Input projection
    gru_input = torch.cat([p_hat_prev, s_prev], dim=-1)
    gru_input = F.relu(F.linear(gru_input, params['input_proj.weight'], params['input_proj.bias']))

    # GRU cell forward (manual implementation for functional)
    # GRUCell: z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
    #          r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
    #          n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
    #          h' = (1 - z) * n + z * h

    hidden_size = c_prev.shape[-1]

    # GRU weights are stored as [W_ir, W_iz, W_in] concatenated
    weight_ih = params['gru.weight_ih']  # (3*hidden, input)
    weight_hh = params['gru.weight_hh']  # (3*hidden, hidden)
    bias_ih = params['gru.bias_ih']      # (3*hidden,)
    bias_hh = params['gru.bias_hh']      # (3*hidden,)

    gi = F.linear(gru_input, weight_ih, bias_ih)
    gh = F.linear(c_prev, weight_hh, bias_hh)

    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)

    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)

    c_new = (1 - z) * n + z * c_prev

    return c_new


# ============== Legacy compatibility ==============

class StateProposer(nn.Module):
    """
    Legacy f_theta for backward compatibility.
    Not used in new MAML architecture.
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
        return self.net(s)


class SubgoalPolicyNetwork(nn.Module):
    """
    Combined network for new MAML architecture.
    """

    def __init__(self, obs_dim, act_dim, context_dim=32,
                 policy_hidden_sizes=(128, 128),
                 world_model_hidden_sizes=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_dim = context_dim

        self.world_model = WorldModel(obs_dim, act_dim, world_model_hidden_sizes)
        self.phi = ContextEncoder(obs_dim, context_dim)
        self.psi = Policy(obs_dim, act_dim, context_dim, policy_hidden_sizes)

    def freeze_world_model(self):
        """Freeze world model after Phase 1."""
        self.world_model.eval()
        for p in self.world_model.parameters():
            p.requires_grad = False

    def get_maml_params(self):
        """Get parameters for MAML (phi + psi only, not world model)."""
        params = OrderedDict()
        for name, param in self.phi.named_parameters():
            params[f'phi.{name}'] = param
        for name, param in self.psi.named_parameters():
            params[f'psi.{name}'] = param
        return params
