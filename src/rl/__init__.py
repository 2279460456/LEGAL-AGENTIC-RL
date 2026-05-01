"""
RL Module
Provides reinforcement learning components for legal agent training.
"""

from .environment import EvidenceEnvironment, AgentState, StepResult, ROLE_PROMPTS
from .reward import RewardCalculator, RewardComponents, DEFAULT_WEIGHTS
from .grpo import GRPOTrainer, GRPOConfig, Trajectory

__all__ = [
    'EvidenceEnvironment',
    'AgentState',
    'StepResult',
    'ROLE_PROMPTS',
    'RewardCalculator',
    'RewardComponents',
    'DEFAULT_WEIGHTS',
    'GRPOTrainer',
    'GRPOConfig',
    'Trajectory'
]