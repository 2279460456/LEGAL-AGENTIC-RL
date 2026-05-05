"""
Data Processing Module
Provides tools for processing legal judgment documents.

Main modules:
- llm_evidence_splitter: Evidence splitting with LLM assistance
- build_rl_data: Build RL training datasets
- generate_sft_data: Generate SFT training data
- sft_templates: SFT data templates for different roles
"""

from .llm_evidence_splitter import (
    LLMEvidenceSplitterBase,
    DefaultLLMSplitter,
    load_llm_config,
    LLMConfig,
    get_crime_specific_evidence_layers,
    get_evidence_triggers,
    get_required_elements,
    build_dynamic_evidence_split_prompt,
    EVIDENCE_ROLES_GENERIC,
    CRIME_SPECIFIC_EVIDENCE_LAYERS,
    REQUIRED_ELEMENTS,
    TRIGGER_KEYWORDS_GENERIC
)

from .sft_templates import (
    SFTDataPoint,
    INSTRUCTION_TEMPLATES,
    datapoint_to_json
)

__all__ = [
    # Evidence splitting
    'LLMEvidenceSplitterBase',
    'DefaultLLMSplitter',
    'load_llm_config',
    'LLMConfig',
    'get_crime_specific_evidence_layers',
    'get_evidence_triggers',
    'get_required_elements',
    'build_dynamic_evidence_split_prompt',
    'EVIDENCE_ROLES_GENERIC',
    'CRIME_SPECIFIC_EVIDENCE_LAYERS',
    'REQUIRED_ELEMENTS',
    'TRIGGER_KEYWORDS_GENERIC',

    # SFT templates
    'SFTDataPoint',
    'INSTRUCTION_TEMPLATES',
    'datapoint_to_json'
]