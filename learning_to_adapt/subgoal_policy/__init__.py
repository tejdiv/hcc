"""
Subgoal-Conditioned Policy with MAML-style Adaptation.

A meta-learning approach that:
1. Learns a world model p̂(s_{t+1}|s_t, a_t) in Phase 1
2. Learns a simple policy π_ψ(a|s) in Phase 1
3. Learns a context encoder f_φ that tracks world model error: c_t = f(c_{t-1}, p̂_{t-1}, s_{t-1})
4. MAML-style meta-training of φ and ψ in Phase 2

Training phases:
- Phase 1: Train π_ψ(a|s) and p̂(s_{t+1}|s_t,a_t) on initial environment
- Phase 2: Freeze p̂, MAML-train φ and ψ across training environments
- Phase 3: MAML-style test-time adaptation on test environments
"""

from .networks import (
    WorldModel,
    ContextEncoder,
    Policy,
    SubgoalPolicyNetwork,
    StateProposer,  # Legacy compatibility
    get_params_dict,
    set_params_dict,
    clone_params,
)

from .trainers import (
    Phase1Trainer,
    Phase2Trainer,
    load_phase1_checkpoint,
    load_phase2_checkpoint,
    collect_rollout_phase1,
    collect_rollout_phase2,
    compute_returns,
)

from .evaluation import (
    Phase3Evaluator,
    run_evaluation,
)

__all__ = [
    # Networks
    'WorldModel',
    'ContextEncoder',
    'Policy',
    'SubgoalPolicyNetwork',
    'StateProposer',  # Legacy
    'get_params_dict',
    'set_params_dict',
    'clone_params',
    # Trainers
    'Phase1Trainer',
    'Phase2Trainer',
    'load_phase1_checkpoint',
    'load_phase2_checkpoint',
    'collect_rollout_phase1',
    'collect_rollout_phase2',
    'compute_returns',
    # Evaluation
    'Phase3Evaluator',
    'run_evaluation',
]
