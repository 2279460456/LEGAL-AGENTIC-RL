"""
Evaluation Module
Provides evaluation metrics and comparison tools.
"""

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class EvaluationResult:
    """Evaluation result for a single case"""
    case_id: str
    crime_match: bool
    crime_f1: float
    sentence_mae: float  # months
    law_recall: float
    avg_rounds: int
    evidence_recall: float
    invalid_query_rate: float
    term_norm: float
    reasoning_complete: float


class MetricsCalculator:
    """
    Calculates evaluation metrics.

    Metrics:
    - Accuracy: crime_f1, sentence_mae, law_recall
    - Efficiency: avg_rounds, evidence_recall, invalid_query_rate
    - Compliance: term_norm, reasoning_complete
    """

    def __init__(self):
        pass

    def compute_all_metrics(
        self,
        prediction: Dict,
        ground_truth: Dict,
        conversation_stats: Dict
    ) -> EvaluationResult:
        """
        Compute all metrics for a case.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth judgment
            conversation_stats: Conversation statistics

        Returns:
            EvaluationResult
        """
        # Accuracy metrics
        crime_match = prediction.get("crime") == ground_truth.get("crime")
        crime_f1 = self._compute_crime_f1(
            prediction.get("crime", ""),
            ground_truth.get("crime", "")
        )

        sentence_mae = abs(
            prediction.get("sentence_months", 0) -
            ground_truth.get("sentence_months", 0)
        )

        pred_laws = set(prediction.get("laws", []))
        true_laws = set(ground_truth.get("laws", []))
        law_recall = len(pred_laws & true_laws) / max(len(true_laws), 1)

        # Efficiency metrics
        avg_rounds = conversation_stats.get("total_rounds", 0)
        evidence_recall = conversation_stats.get("evidence_recall", 0.0)
        invalid_query_rate = conversation_stats.get("invalid_query_rate", 0.0)

        # Compliance metrics
        term_norm = self._compute_term_norm(prediction)
        reasoning_complete = self._compute_reasoning_complete(prediction)

        return EvaluationResult(
            case_id=prediction.get("case_id", ""),
            crime_match=crime_match,
            crime_f1=crime_f1,
            sentence_mae=sentence_mae,
            law_recall=law_recall,
            avg_rounds=avg_rounds,
            evidence_recall=evidence_recall,
            invalid_query_rate=invalid_query_rate,
            term_norm=term_norm,
            reasoning_complete=reasoning_complete
        )

    def _compute_crime_f1(self, pred: str, true: str) -> float:
        """Compute F1 for crime classification"""
        if pred == true:
            return 1.0

        pred_words = set(pred.split())
        true_words = set(true.split())

        if not pred_words or not true_words:
            return 0.0

        intersection = pred_words & true_words
        precision = len(intersection) / len(pred_words)
        recall = len(intersection) / len(true_words)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _compute_term_norm(self, prediction: Dict) -> float:
        """Compute legal term normalization score"""
        text = str(prediction)
        legal_terms = ['被告人', '被害人', '依照', '判决', '犯罪']

        count = sum(1 for term in legal_terms if term in text)
        return count / len(legal_terms)

    def _compute_reasoning_complete(self, prediction: Dict) -> float:
        """Compute reasoning completeness score"""
        text = str(prediction)
        steps = ['事实认定', '法律适用', '量刑考量', '判决结论']

        count = sum(1 for step in steps if step in text)
        return count / len(steps)

    def aggregate_results(
        self,
        results: List[EvaluationResult]
    ) -> Dict:
        """
        Aggregate results across all cases.

        Args:
            results: List of EvaluationResult

        Returns:
            Aggregated metrics dictionary
        """
        if not results:
            return {}

        n = len(results)

        return {
            "accuracy": {
                "crime_accuracy": sum(r.crime_match for r in results) / n,
                "crime_f1_mean": np.mean([r.crime_f1 for r in results]),
                "sentence_mae_mean": np.mean([r.sentence_mae for r in results]),
                "law_recall_mean": np.mean([r.law_recall for r in results])
            },
            "efficiency": {
                "avg_rounds_mean": np.mean([r.avg_rounds for r in results]),
                "evidence_recall_mean": np.mean([r.evidence_recall for r in results]),
                "invalid_query_rate_mean": np.mean([r.invalid_query_rate for r in results])
            },
            "compliance": {
                "term_norm_mean": np.mean([r.term_norm for r in results]),
                "reasoning_complete_mean": np.mean([r.reasoning_complete for r in results])
            }
        }


class BaselineComparator:
    """
    Compares our method with baseline methods.

    Baselines:
    - Traditional LJP models
    - SFT-only model
    - Single-agent version
    """

    def __init__(self):
        self.results = defaultdict(list)

    def add_result(self, method_name: str, results: List[EvaluationResult]):
        """Add results for a method"""
        self.results[method_name] = results

    def compare(self) -> Dict:
        """
        Compare all methods.

        Returns:
            Comparison table
        """
        metrics_calc = MetricsCalculator()

        comparison = {}
        for method, results in self.results.items():
            comparison[method] = metrics_calc.aggregate_results(results)

        return comparison

    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table"""
        comparison = self.compare()

        if not comparison:
            return "No results to compare"

        # Build table header
        methods = list(comparison.keys())
        header = "| Metric | " + " | ".join(methods) + " |"

        # Build table rows
        rows = []

        # Accuracy metrics
        rows.append("| Crime Accuracy | " +
                   " | ".join([f"{comparison[m]['accuracy']['crime_accuracy']:.2%}"
                              for m in methods]) + " |")
        rows.append("| Crime F1 | " +
                   " | ".join([f"{comparison[m]['accuracy']['crime_f1_mean']:.3f}"
                              for m in methods]) + " |")
        rows.append("| Sentence MAE | " +
                   " | ".join([f"{comparison[m]['accuracy']['sentence_mae_mean']:.1f}"
                              for m in methods]) + " |")

        # Efficiency metrics
        rows.append("| Avg Rounds | " +
                   " | ".join([f"{comparison[m]['efficiency']['avg_rounds_mean']:.1f}"
                              for m in methods]) + " |")
        rows.append("| Evidence Recall | " +
                   " | ".join([f"{comparison[m]['efficiency']['evidence_recall_mean']:.2%}"
                              for m in methods]) + " |")

        return header + "\n" + "|---|" * (len(methods) + 1) + "\n" + "\n".join(rows)


def main():
    """Example usage"""
    calc = MetricsCalculator()

    # Example prediction and ground truth
    prediction = {
        "case_id": "test_001",
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    ground_truth = {
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    conversation_stats = {
        "total_rounds": 5,
        "evidence_recall": 0.8,
        "invalid_query_rate": 0.1
    }

    result = calc.compute_all_metrics(prediction, ground_truth, conversation_stats)

    print(f"Crime F1: {result.crime_f1}")
    print(f"Sentence MAE: {result.sentence_mae}")
    print(f"Evidence Recall: {result.evidence_recall}")


if __name__ == "__main__":
    main()