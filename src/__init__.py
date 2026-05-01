"""
LEGAL-AGENTIC-RL Package
Multi-Agent Reinforcement Learning for Legal Judgment Prediction
"""

__version__ = "0.1.0"

from .data_processing import (
    CoTDistiller,
    EvidenceSplitter,
    DatasetBuilder
)

from .rl import (
    EvidenceEnvironment,
    RewardCalculator,
    GRPOTrainer
)

from .sft import SFTTrainer

from .evaluation import MetricsCalculator, BaselineComparator

__all__ = [
    'CoTDistiller',
    'EvidenceSplitter',
    'DatasetBuilder',
    'EvidenceEnvironment',
    'RewardCalculator',
    'GRPOTrainer',
    'SFTTrainer',
    'MetricsCalculator',
    'BaselineComparator'
]