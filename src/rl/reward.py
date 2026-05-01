"""
Reward Function Module
Implements the multi-dimensional reward function for legal RL training.

Reward components:
- R_accuracy: Judgment accuracy (crime, sentence, laws)
- R_information: Information gathering efficiency
- R_compliance: Procedure compliance

Total: R = λ₁·R_accuracy + λ₂·R_information + λ₃·R_compliance
"""

import json
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from collections import Counter


# Default reward weights
DEFAULT_WEIGHTS = {
    "accuracy": 0.5,     # λ₁
    "information": 0.3,  # λ₂
    "compliance": 0.2    # λ₃
}


# Accuracy sub-weights
ACCURACY_SUBWEIGHTS = {
    "crime": 0.4,        # Crime classification weight
    "sentence": 0.3,     # Sentence prediction weight
    "law": 0.3           # Law citation weight
}


# Information reward parameters
INFO_PARAMS = {
    "new_evidence_bonus": 0.1,
    "repeat_penalty": -0.05,
    "irrelevant_penalty": -0.02
}


# Compliance reward parameters
COMPLIANCE_PARAMS = {
    "term_bonus": 0.02,
    "step_bonus": 0.05,
    "round_penalty": -0.1
}


# Legal terminology list
LEGAL_TERMS = [
    '被告人', '被害人', '本院认为', '依照', '判决如下',
    '犯罪', '故意', '过失', '自首', '累犯',
    '有期徒刑', '拘役', '罚金', '死刑'
]


# Required reasoning steps
REQUIRED_STEPS = [
    '事实认定', '法律适用', '量刑考量', '判决结论'
]


# Legal keywords for query relevance
LEGAL_QUERY_KEYWORDS = [
    '动机', '手段', '伤害', '自首', '赔偿', '事实', '证据',
    '预谋', '案发', '作案', '认罪', '供述', '情节'
]


@dataclass
class RewardComponents:
    """Detailed reward breakdown"""
    accuracy: float
    information: float
    compliance: float
    total: float
    details: Dict


class RewardCalculator:
    """
    Calculates multi-dimensional rewards for legal RL.

    Usage:
        calculator = RewardCalculator(weights=DEFAULT_WEIGHTS)
        reward = calculator.calculate_total_reward(prediction, ground_truth, history)
    """

    def __init__(
        self,
        weights: Optional[Dict] = None,
        accuracy_weights: Optional[Dict] = None,
        info_params: Optional[Dict] = None,
        compliance_params: Optional[Dict] = None
    ):
        """
        Initialize Reward Calculator.

        Args:
            weights: Main reward weights
            accuracy_weights: Sub-weights for accuracy
            info_params: Information reward parameters
            compliance_params: Compliance reward parameters
        """
        self.weights = weights or DEFAULT_WEIGHTS
        self.accuracy_weights = accuracy_weights or ACCURACY_SUBWEIGHTS
        self.info_params = info_params or INFO_PARAMS
        self.compliance_params = compliance_params or COMPLIANCE_PARAMS

    def calculate_total_reward(
        self,
        prediction: Dict,
        ground_truth: Dict,
        conversation_history: List,
        revealed_evidence: Set,
        total_evidence_levels: int,
        current_round: int,
        max_rounds: int
    ) -> RewardComponents:
        """
        Calculate total reward with all components.

        Args:
            prediction: Model's prediction (crime, sentence, laws)
            ground_truth: Ground truth judgment
            conversation_history: List of (role, message) tuples
            revealed_evidence: Set of revealed evidence levels
            total_evidence_levels: Total number of hidden evidence levels
            current_round: Current conversation round
            max_rounds: Maximum allowed rounds

        Returns:
            RewardComponents with detailed breakdown
        """
        # Calculate each component
        accuracy = self.calculate_accuracy_reward(prediction, ground_truth)
        information = self.calculate_information_reward(
            revealed_evidence,
            total_evidence_levels,
            conversation_history,
            current_round
        )
        compliance = self.calculate_compliance_reward(
            prediction,
            conversation_history,
            current_round,
            max_rounds
        )

        # Total reward
        total = (
            self.weights["accuracy"] * accuracy +
            self.weights["information"] * information +
            self.weights["compliance"] * compliance
        )

        return RewardComponents(
            accuracy=accuracy,
            information=information,
            compliance=compliance,
            total=total,
            details={
                "accuracy_details": self._get_accuracy_details(prediction, ground_truth),
                "information_details": {
                    "revealed": len(revealed_evidence),
                    "total": total_evidence_levels,
                    "rounds": current_round
                },
                "compliance_details": {
                    "terms_used": self._count_legal_terms(str(prediction)),
                    "rounds": current_round
                }
            }
        )

    def calculate_accuracy_reward(
        self,
        prediction: Dict,
        ground_truth: Dict
    ) -> float:
        """
        Calculate accuracy reward.

        R_accuracy = w_crime * crime_f1 + w_sentence * sentence_score + w_law * law_recall

        Args:
            prediction: Model prediction
            ground_truth: Ground truth

        Returns:
            Accuracy reward (0-1)
        """
        # Crime classification score
        pred_crime = prediction.get("crime", "")
        true_crime = ground_truth.get("crime", "")
        crime_score = self._compute_crime_similarity(pred_crime, true_crime)

        # Sentence prediction score
        pred_months = prediction.get("sentence_months", 0)
        true_months = ground_truth.get("sentence_months", 0)
        sentence_score = self._compute_sentence_score(pred_months, true_months)

        # Law citation score
        pred_laws = set(prediction.get("laws", []))
        true_laws = set(ground_truth.get("laws", []))
        law_score = self._compute_law_recall(pred_laws, true_laws)

        # Weighted combination
        accuracy = (
            self.accuracy_weights["crime"] * crime_score +
            self.accuracy_weights["sentence"] * sentence_score +
            self.accuracy_weights["law"] * law_score
        )

        return accuracy

    def _compute_crime_similarity(self, pred: str, true: str) -> float:
        """Compute crime classification similarity"""
        # Exact match
        if pred == true:
            return 1.0

        # F1 score based on word overlap
        pred_words = set(pred.split())
        true_words = set(true.split())

        if not pred_words or not true_words:
            return 0.0

        intersection = pred_words & true_words
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(true_words)

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _compute_sentence_score(self, pred: int, true: int) -> float:
        """Compute sentence prediction score"""
        if true == 0:
            return 1.0 if pred == 0 else 0.0

        # Relative error
        relative_error = abs(pred - true) / true
        score = 1.0 - relative_error

        return max(0.0, min(1.0, score))

    def _compute_law_recall(self, pred: Set, true: Set) -> float:
        """Compute law citation recall"""
        if not true:
            return 1.0

        return len(pred & true) / len(true)

    def calculate_information_reward(
        self,
        revealed_evidence: Set,
        total_levels: int,
        conversation_history: List,
        current_round: int
    ) -> float:
        """
        Calculate information gathering efficiency reward.

        Args:
            revealed_evidence: Set of revealed evidence levels
            total_levels: Total number of hidden evidence levels
            conversation_history: Conversation history
            current_round: Current round number

        Returns:
            Information reward
        """
        reward = 0.0

        # Evidence discovery ratio
        if total_levels > 0:
            discovery_ratio = len(revealed_evidence) / total_levels
            reward += discovery_ratio * 0.5  # Base discovery reward

        # Efficiency bonus: more discoveries in fewer rounds
        if current_round > 0 and len(revealed_evidence) > 0:
            efficiency = len(revealed_evidence) / current_round
            reward += efficiency * 0.3

        return min(1.0, reward)

    def calculate_step_information_reward(
        self,
        query: str,
        triggered_new: bool,
        conversation_history: List
    ) -> float:
        """
        Calculate step-level information reward.

        Args:
            query: Current query
            triggered_new: Whether new evidence was triggered
            conversation_history: Previous conversation

        Returns:
            Step reward
        """
        reward = 0.0

        # New evidence bonus
        if triggered_new:
            reward += self.info_params["new_evidence_bonus"]

        # Repeat query penalty
        previous_queries = [msg for role, msg in conversation_history if role == "judge"]
        if query in previous_queries[:-1]:
            reward += self.info_params["repeat_penalty"]

        # Irrelevant query penalty
        if not self._is_legal_query(query):
            reward += self.info_params["irrelevant_penalty"]

        return reward

    def _is_legal_query(self, query: str) -> bool:
        """Check if query contains legal keywords"""
        return any(kw in query for kw in LEGAL_QUERY_KEYWORDS)

    def calculate_compliance_reward(
        self,
        prediction: Dict,
        conversation_history: List,
        current_round: int,
        max_rounds: int
    ) -> float:
        """
        Calculate procedure compliance reward.

        Args:
            prediction: Final judgment
            conversation_history: Conversation history
            current_round: Current round
            max_rounds: Maximum allowed rounds

        Returns:
            Compliance reward
        """
        reward = 0.0

        # Legal terminology usage
        judgment_text = str(prediction)
        term_count = self._count_legal_terms(judgment_text)
        reward += term_count * self.compliance_params["term_bonus"]

        # Reasoning steps completeness
        step_count = self._count_reasoning_steps(judgment_text)
        reward += step_count * self.compliance_params["step_bonus"]

        # Round penalty for exceeding limit
        if current_round > max_rounds:
            excess = current_round - max_rounds
            reward += self.compliance_params["round_penalty"] * excess

        return max(0.0, reward)

    def _count_legal_terms(self, text: str) -> int:
        """Count legal terms in text"""
        count = 0
        for term in LEGAL_TERMS:
            if term in text:
                count += 1
        return count

    def _count_reasoning_steps(self, text: str) -> int:
        """Count reasoning steps in text"""
        count = 0
        for step in REQUIRED_STEPS:
            if step in text:
                count += 1
        return count

    def _get_accuracy_details(self, prediction: Dict, ground_truth: Dict) -> Dict:
        """Get detailed accuracy breakdown"""
        return {
            "pred_crime": prediction.get("crime"),
            "true_crime": ground_truth.get("crime"),
            "pred_sentence": prediction.get("sentence_months"),
            "true_sentence": ground_truth.get("sentence_months"),
            "pred_laws": prediction.get("laws"),
            "true_laws": ground_truth.get("laws")
        }


def main():
    """Example usage"""
    calculator = RewardCalculator()

    prediction = {
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    ground_truth = {
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    reward = calculator.calculate_total_reward(
        prediction=prediction,
        ground_truth=ground_truth,
        conversation_history=[],
        revealed_evidence={"subjective", "objective", "sentencing"},
        total_evidence_levels=3,
        current_round=5,
        max_rounds=10
    )

    print(f"Total Reward: {reward.total}")
    print(f"Accuracy: {reward.accuracy}")
    print(f"Information: {reward.information}")
    print(f"Compliance: {reward.compliance}")
    print(f"Details: {reward.details}")


if __name__ == "__main__":
    main()