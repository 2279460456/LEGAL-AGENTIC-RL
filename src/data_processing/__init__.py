"""
Data Processing Module
Provides tools for processing legal judgment documents.
"""

from .cot_distill import CoTDistiller, CoTOutput, COT_DISTILL_PROMPT
from .evidence_split import EvidenceSplitter, EvidenceLevel, HiddenEvidence, CaseEvidenceStructure
from .dataset_builder import DatasetBuilder, SFTDataPoint, ROLE_INSTRUCTIONS

__all__ = [
    'CoTDistiller',
    'CoTOutput',
    'COT_DISTILL_PROMPT',
    'EvidenceSplitter',
    'EvidenceLevel',
    'HiddenEvidence',
    'CaseEvidenceStructure',
    'DatasetBuilder',
    'SFTDataPoint',
    'ROLE_INSTRUCTIONS'
]