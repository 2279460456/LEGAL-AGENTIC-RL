"""
Evidence Splitting Module
Splits judgment information into visible and hidden evidence layers.

This module implements the "Information Hiding-Trigger" mechanism:
- Level 0 (Public): Initial case overview (visible from start)
- Level 1 (Subjective): Motive, premeditation evidence
- Level 2 (Objective): Action details, injury degree
- Level 3 (Sentencing): Post-crime behavior (confession, compensation)
"""

import json
import re
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class EvidenceLevel(Enum):
    """Evidence visibility levels"""
    PUBLIC = 0      # Always visible
    SUBJECTIVE = 1  # Motive, premeditation
    OBJECTIVE = 2   # Action details, injury
    SENTENCING = 3  # Post-crime behavior


# Trigger keywords for each evidence level
TRIGGER_KEYWORDS = {
    EvidenceLevel.SUBJECTIVE: [
        "作案动机", "事前准备", "预谋", "购买工具", "跟踪",
        "策划", "事先", "蓄意", "故意"
    ],
    EvidenceLevel.OBJECTIVE: [
        "作案手段", "伤害部位", "打击力度", "刺了几刀",
        "伤口位置", "作案工具", "具体行为", "行为方式"
    ],
    EvidenceLevel.SENTENCING: [
        "案后表现", "认罪态度", "自首", "赔偿", "取得谅解",
        "如实供述", "拨打120", "报警", "悔罪"
    ]
}


@dataclass
class HiddenEvidence:
    """Hidden evidence piece with trigger conditions"""
    level: EvidenceLevel
    content: str  # Evidence content
    triggers: List[str]  # Keywords that trigger this evidence
    category: str  # e.g., "motive", "injury_detail", "confession"
    triggered: bool = False  # Whether already triggered


@dataclass
class CaseEvidenceStructure:
    """Complete evidence structure for a case"""
    case_id: str
    public_info: str  # Initial visible information
    hidden_evidence: Dict[EvidenceLevel, List[HiddenEvidence]]
    ground_truth: Dict  # Final judgment (crime, sentence, laws)

    def get_untriggered_evidence(self) -> List[HiddenEvidence]:
        """Get all untriggered hidden evidence"""
        result = []
        for level, evidences in self.hidden_evidence.items():
            for ev in evidences:
                if not ev.triggered:
                    result.append(ev)
        return result


class EvidenceSplitter:
    """
    Splits judgment documents into evidence layers.

    Usage:
        splitter = EvidenceSplitter()
        structure = splitter.split(judgment_data)
    """

    def __init__(self, trigger_method: str = "keyword"):
        """
        Initialize the Evidence Splitter.

        Args:
            trigger_method: "keyword" or "semantic" matching
        """
        self.trigger_method = trigger_method
        self.trigger_keywords = TRIGGER_KEYWORDS

    def split(self, judgment_data: Dict) -> CaseEvidenceStructure:
        """
        Split judgment data into evidence layers.

        Args:
            judgment_data: Dictionary containing judgment information

        Returns:
            CaseEvidenceStructure: Complete evidence structure
        """
        case_id = judgment_data.get("case_id", "")

        # Extract public information (basic case overview)
        public_info = self._extract_public_info(judgment_data)

        # Extract hidden evidence at each level
        hidden_evidence = self._extract_hidden_evidence(judgment_data)

        # Extract ground truth judgment
        ground_truth = self._extract_ground_truth(judgment_data)

        return CaseEvidenceStructure(
            case_id=case_id,
            public_info=public_info,
            hidden_evidence=hidden_evidence,
            ground_truth=ground_truth
        )

    def _extract_public_info(self, data: Dict) -> str:
        """Extract initial public information"""
        # Basic case overview without detailed evidence
        # Should be vague and incomplete
        parts = []

        if "defendant" in data:
            parts.append(f"被告人{data['defendant']}")
        if "victim" in data:
            parts.append(f"被害人{data['victim']}")
        if "date" in data:
            parts.append(f"案发时间{data['date']}")
        if "basic_fact" in data:
            parts.append(data["basic_fact"])

        return "。".join(parts) + "。"

    def _extract_hidden_evidence(self, data: Dict) -> Dict[EvidenceLevel, List[HiddenEvidence]]:
        """Extract hidden evidence for each level"""
        hidden = {}

        # Level 1: Subjective evidence
        hidden[EvidenceLevel.SUBJECTIVE] = self._extract_subjective(data)

        # Level 2: Objective evidence
        hidden[EvidenceLevel.OBJECTIVE] = self._extract_objective(data)

        # Level 3: Sentencing evidence
        hidden[EvidenceLevel.SENTENCING] = self._extract_sentencing(data)

        return hidden

    def _extract_subjective(self, data: Dict) -> List[HiddenEvidence]:
        """Extract subjective (motive/premeditation) evidence"""
        evidences = []

        # Check for premeditation indicators
        if "premeditation" in data:
            evidences.append(HiddenEvidence(
                level=EvidenceLevel.SUBJECTIVE,
                content=data["premeditation"],
                triggers=TRIGGER_KEYWORDS[EvidenceLevel.SUBJECTIVE],
                category="premeditation"
            ))

        if "motive" in data:
            evidences.append(HiddenEvidence(
                level=EvidenceLevel.SUBJECTIVE,
                content=data["motive"],
                triggers=["作案动机", "动机", "为什么"],
                category="motive"
            ))

        return evidences

    def _extract_objective(self, data: Dict) -> List[HiddenEvidence]:
        """Extract objective (action/injury) evidence"""
        evidences = []

        if "action_detail" in data:
            evidences.append(HiddenEvidence(
                level=EvidenceLevel.OBJECTIVE,
                content=data["action_detail"],
                triggers=TRIGGER_KEYWORDS[EvidenceLevel.OBJECTIVE],
                category="action_detail"
            ))

        if "injury_detail" in data:
            evidences.append(HiddenEvidence(
                level=EvidenceLevel.OBJECTIVE,
                content=data["injury_detail"],
                triggers=["伤害部位", "伤口", "伤情"],
                category="injury_detail"
            ))

        return evidences

    def _extract_sentencing(self, data: Dict) -> List[HiddenEvidence]:
        """Extract sentencing (post-crime behavior) evidence"""
        evidences = []

        if "post_crime_behavior" in data:
            evidences.append(HiddenEvidence(
                level=EvidenceLevel.SENTENCING,
                content=data["post_crime_behavior"],
                triggers=TRIGGER_KEYWORDS[EvidenceLevel.SENTENCING],
                category="post_crime"
            ))

        return evidences

    def _extract_ground_truth(self, data: Dict) -> Dict:
        """Extract ground truth judgment"""
        return {
            "crime": data.get("crime", ""),
            "sentence_months": data.get("sentence_months", 0),
            "laws": data.get("laws", []),
            "sentence_text": data.get("sentence_text", "")
        }

    def check_trigger(
        self,
        query: str,
        evidence: HiddenEvidence
    ) -> bool:
        """
        Check if a query triggers hidden evidence.

        Args:
            query: User/agent query
            evidence: Hidden evidence to check

        Returns:
            bool: True if triggered
        """
        if self.trigger_method == "keyword":
            return self._keyword_trigger(query, evidence)
        else:
            # TODO: Implement semantic matching
            return self._keyword_trigger(query, evidence)

    def _keyword_trigger(self, query: str, evidence: HiddenEvidence) -> bool:
        """Keyword-based trigger matching"""
        for trigger in evidence.triggers:
            if trigger in query:
                return True
        return False

    def batch_split(
        self,
        judgments: List[Dict],
        output_path: Optional[str] = None
    ) -> List[CaseEvidenceStructure]:
        """
        Batch split multiple judgments.

        Args:
            judgments: List of judgment data
            output_path: Path to save results

        Returns:
            List of CaseEvidenceStructure
        """
        results = [self.split(j) for j in judgments]

        if output_path:
            self._save_results(results, output_path)

        return results

    def _save_results(self, results: List[CaseEvidenceStructure], path: str):
        """Save results to JSON"""
        data = []
        for r in results:
            item = {
                "case_id": r.case_id,
                "public_info": r.public_info,
                "hidden_evidence": {
                    level.name: [
                        {
                            "content": ev.content,
                            "triggers": ev.triggers,
                            "category": ev.category
                        }
                        for ev in evidences
                    ]
                    for level, evidences in r.hidden_evidence.items()
                },
                "ground_truth": r.ground_truth
            }
            data.append(item)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """Example usage"""
    sample_judgment = {
        "case_id": "case_001",
        "defendant": "张三",
        "victim": "李四",
        "date": "2026年3月",
        "basic_fact": "张三与李四发生冲突，李四受重伤",
        "premeditation": "张三案发前一周购买了折叠刀并记录了被害人行踪",
        "action_detail": "张三持刀连续刺了三刀，刀伤位于左胸部，深达肺部",
        "post_crime_behavior": "张三现场拨打120并在原地等待，如实供述",
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"],
        "sentence_text": "判处有期徒刑三年"
    }

    splitter = EvidenceSplitter()
    structure = splitter.split(sample_judgment)

    print(f"Public Info: {structure.public_info}")
    print(f"Hidden Evidence: {structure.hidden_evidence}")
    print(f"Ground Truth: {structure.ground_truth}")


if __name__ == "__main__":
    main()