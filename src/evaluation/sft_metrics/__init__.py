"""
SFT Metrics Package
提供SFT阶段评估指标
"""

from .role_metrics import (
    compute_role_accuracy,
    compute_legal_term_coverage,
    compute_cot_completeness,
    compute_format_compliance,
    compute_all_role_metrics,
    ROLE_KEYWORDS,
    LEGAL_TERMS,
)

from .judge_metrics import (
    compute_crime_accuracy,
    compute_sentence_mae,
    compute_law_recall,
    compute_judge_accuracy,
    extract_crime,
    extract_sentence,
    extract_laws,
)

from .text_metrics import (
    compute_bleu,
    compute_rouge_n,
    compute_rouge_l,
    compute_all_text_metrics,
    compute_exact_match,
)

__all__ = [
    # Role metrics
    'compute_role_accuracy',
    'compute_legal_term_coverage',
    'compute_cot_completeness',
    'compute_format_compliance',
    'compute_all_role_metrics',
    'ROLE_KEYWORDS',
    'LEGAL_TERMS',

    # Judge metrics
    'compute_crime_accuracy',
    'compute_sentence_mae',
    'compute_law_recall',
    'compute_judge_accuracy',
    'extract_crime',
    'extract_sentence',
    'extract_laws',

    # Text metrics
    'compute_bleu',
    'compute_rouge_n',
    'compute_rouge_l',
    'compute_all_text_metrics',
    'compute_exact_match',
]