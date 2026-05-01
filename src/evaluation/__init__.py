"""
Evaluation Module
提供SFT和RL阶段的评估功能
"""

from .metrics import MetricsCalculator, EvaluationResult, BaselineComparator
from .sft_metrics import (
    compute_role_accuracy,
    compute_legal_term_coverage,
    compute_cot_completeness,
    compute_bleu,
    compute_rouge_l,
)

__all__ = [
    'MetricsCalculator',
    'EvaluationResult',
    'BaselineComparator',
    'compute_role_accuracy',
    'compute_legal_term_coverage',
    'compute_cot_completeness',
    'compute_bleu',
    'compute_rouge_l',
]