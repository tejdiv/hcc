"""
Subgoal-Conditioned Policy with Context Adaptation.

A meta-learning approach that:
1. Learns a state proposer f_theta that predicts next state goals
2. Learns a policy pi_psi conditioned on (state, goal, context)
3. Learns a context encoder f_phi that tracks world model error

Training phases:
- Phase 1: Train theta, psi on initial environment (c=0)
- Phase 2: Freeze theta, train phi, psi across training environments
- Phase 3: Freeze all, evaluate with context adaptation on test environments
"""

from .networks import (
    StateProposer,
    ContextEncoder,
    Policy,
    SubgoalPolicyNetwork,
)

from .grpo_ppo import (
    compute_returns,
    compute_advantages_grpo,
    ppo_loss,
    state_prediction_loss,
    GRPOPPOTrainer,
)

from .rollout import (
    collect_trajectory_phase1,
    collect_trajectory_phase2,
    collect_batch_phase1,
    collect_batch_phase2,
    collect_batch_vectorized_phase1,
    collect_batch_vectorized_phase2,
    collect_batch_parallel_phase1,
    collect_batch_parallel_phase2,
    compute_trajectory_stats,
    HAS_PARALLEL_EXECUTOR,
)

from .trainers import (
    Phase1Trainer,
    Phase2Trainer,
    load_phase1_checkpoint,
    load_phase2_checkpoint,
)

from .evaluation import (
    evaluate_single_episode,
    evaluate_on_env,
    Phase3Evaluator,
    run_evaluation,
)

__all__ = [
    # Networks
    'StateProposer',
    'ContextEncoder',
    'Policy',
    'SubgoalPolicyNetwork',
    # Training
    'compute_returns',
    'compute_advantages_grpo',
    'ppo_loss',
    'state_prediction_loss',
    'GRPOPPOTrainer',
    # Rollout
    'collect_trajectory_phase1',
    'collect_trajectory_phase2',
    'collect_batch_phase1',
    'collect_batch_phase2',
    'collect_batch_vectorized_phase1',
    'collect_batch_vectorized_phase2',
    'collect_batch_parallel_phase1',
    'collect_batch_parallel_phase2',
    'compute_trajectory_stats',
    'HAS_PARALLEL_EXECUTOR',
    # Trainers
    'Phase1Trainer',
    'Phase2Trainer',
    'load_phase1_checkpoint',
    'load_phase2_checkpoint',
    # Evaluation
    'evaluate_single_episode',
    'evaluate_on_env',
    'Phase3Evaluator',
    'run_evaluation',
]
