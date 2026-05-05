"""
RL Environment Module
Implements the legal multi-agent environment with information hiding mechanism.

This module provides:
- EvidenceEnvironment: The core environment class
- Role switching mechanism for single-model multi-role setup
- Hidden evidence triggering system
"""

import json
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class AgentAction(Enum):
    """Types of actions an agent can take"""
    QUERY = "query"      # Ask for more evidence
    JUDGE = "judge"      # Make final judgment
    DEFEND = "defend"    # Defender argument
    ACCUSE = "accuse"    # Prosecutor argument


@dataclass
class AgentState:
    """State of the conversation"""
    public_info: str
    revealed_evidence: List[str]
    conversation_history: List[Tuple[str, str]]  # (role, message)
    current_round: int
    max_rounds: int
    is_terminal: bool = False


@dataclass
class StepResult:
    """Result of environment step"""
    state: AgentState
    reward: float
    new_evidence: Optional[str]  # Evidence content if triggered
    is_terminal: bool
    info: Dict = field(default_factory=dict)


# Role prompt templates for single-model multi-role setup
ROLE_PROMPTS = {
    'judge': """你是一位资深法官，正在审理案件。你的任务是：
1. 分析案情和控辩双方陈述
2. 通过提问补充必要的证据细节
3. 最终给出判决结论（罪名、刑期、法条依据）

请严格按照法律推理程序进行。
你可以选择：
- 提问：询问某个方面的证据细节（如作案动机、伤害程度、案后表现等）
- 判决：根据已有信息作出最终判决

当前状态：
- 已知信息：{public_info}
- 已获取证据：{revealed_evidence}
- 当前轮次：第{current_round}轮（最多{max_rounds}轮）

请决定下一步行动。""",

    'prosecutor': """你是一位检察官，负责指控犯罪。
请根据案情事实，提出指控理由，强调定罪要素。

当前案情：{context}
请提出指控意见。""",

    'defender': """你是一位辩护律师，负责为被告人辩护。
请寻找量刑减轻情节，提出辩护意见。

当前案情：{context}
请提出辩护意见。"""
}


class EvidenceEnvironment:
    """
    Legal multi-agent environment with hidden evidence.

    Implements the "Information Hiding-Trigger" mechanism:
    - Public information is visible from start
    - Hidden evidence is released when triggered by specific queries
    - Agent (judge) decides whether to query or judge

    Usage:
        env = EvidenceEnvironment(case_data)
        state = env.reset()

        while not env.is_terminal():
            action = model.decide(state)
            result = env.step(action)
    """

    def __init__(
        self,
        trigger_method: str = "keyword",
        max_rounds: int = 10
    ):
        """
        Initialize the environment.

        Args:
            trigger_method: "keyword" or "semantic" matching
            max_rounds: Maximum conversation rounds
        """
        self.trigger_method = trigger_method
        self.max_rounds = max_rounds

        # Case data storage
        self.case_data: Dict = {}
        self.public_info: str = ""
        self.hidden_evidence: Dict = {}  # level -> evidence dict
        self.ground_truth: Dict = {}

        # Tracking
        self.revealed_levels: Set = set()
        self.conversation_history: List = []
        self.current_round: int = 0
        self.final_prediction: Optional[Dict] = None

    def reset(self, case_data: Dict) -> AgentState:
        """
        Reset environment with new case.

        Args:
            case_data: Dictionary containing case information

        Returns:
            Initial AgentState
        """
        self.case_data = case_data
        self.public_info = case_data.get("public_info", "")
        self.hidden_evidence = case_data.get("hidden_evidence", {})
        self.ground_truth = case_data.get("ground_truth", {})

        # Reset tracking
        self.revealed_levels = set()
        self.conversation_history = []
        self.current_round = 0
        self.final_prediction = None

        return self._get_state()

    def _get_state(self) -> AgentState:
        """Get current state"""
        revealed = [
            self.hidden_evidence.get(level, {}).get("content", "")
            for level in self.revealed_levels
        ]

        return AgentState(
            public_info=self.public_info,
            revealed_evidence=revealed,
            conversation_history=self.conversation_history.copy(),
            current_round=self.current_round,
            max_rounds=self.max_rounds,
            is_terminal=self.is_terminal()
        )

    def is_terminal(self) -> bool:
        """Check if episode is terminal"""
        return (
            self.current_round >= self.max_rounds or
            self.final_prediction is not None
        )

    def step(self, action: Dict) -> StepResult:
        """
        Execute one step in environment.

        Args:
            action: Dictionary with 'type' and 'content'
                - type: "query" or "judge"
                - content: query text or judgment dict

        Returns:
            StepResult with new state, reward, etc.
        """
        action_type = action.get("type")
        content = action.get("content")

        if action_type == "query":
            return self._handle_query(content)
        elif action_type == "judge":
            return self._handle_judge(content)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def _handle_query(self, query: str) -> StepResult:
        """Handle a query action"""
        self.current_round += 1
        self.conversation_history.append(("judge", query))

        # Check if query triggers hidden evidence
        new_evidence = self._check_trigger(query)

        # Calculate step reward
        reward = self._calculate_step_reward(
            triggered=bool(new_evidence),
            query=query
        )

        state = self._get_state()

        return StepResult(
            state=state,
            reward=reward,
            new_evidence=new_evidence,
            is_terminal=self.is_terminal(),
            info={"triggered": bool(new_evidence)}
        )

    def _handle_judge(self, judgment: Dict) -> StepResult:
        """Handle a judgment action"""
        self.current_round += 1
        self.final_prediction = judgment
        self.conversation_history.append(("judge", f"判决：{judgment}"))

        # Episode ends
        return StepResult(
            state=self._get_state(),
            reward=0,  # Final reward calculated separately
            new_evidence=None,
            is_terminal=True,
            info={"judgment": judgment}
        )

    def _check_trigger(self, query: str) -> Optional[str]:
        """
        Check if query triggers hidden evidence.

        Args:
            query: Query text from agent

        Returns:
            Evidence content if triggered, None otherwise
        """
        for level, evidence in self.hidden_evidence.items():
            if level not in self.revealed_levels:
                triggers = evidence.get("triggers", [])

                if self._match_triggers(query, triggers):
                    self.revealed_levels.add(level)
                    return evidence.get("content")

        return None

    def _match_triggers(self, query: str, triggers: List[str]) -> bool:
        """Match query against trigger keywords"""
        if self.trigger_method == "keyword":
            for trigger in triggers:
                if trigger in query:
                    return True
            return False
        else:
            # TODO: Implement semantic matching
            return self._match_triggers(query, triggers)

    def _calculate_step_reward(
        self,
        triggered: bool,
        query: str
    ) -> float:
        """
        Calculate step reward for information gathering.

        Args:
            triggered: Whether new evidence was triggered
            query: Query text

        Returns:
            Step reward value
        """
        reward = 0.0

        if triggered:
            reward += 0.1  # Bonus for new evidence

        # Check for repeated query
        for role, msg in self.conversation_history[:-1]:
            if msg == query:
                reward -= 0.05  # Penalty for repetition
                break

        # Check for irrelevant query (no legal keywords)
        legal_keywords = [
            "动机", "手段", "伤害", "自首", "赔偿", "事实", "证据",
            "预谋", "工具", "伤口", "部位", "认罪", "报案", "谅解",
            "刑期", "罪名", "法条", "故意", "重伤", "轻伤"
        ]
        if not any(kw in query for kw in legal_keywords):
            reward -= 0.02  # Penalty for irrelevant query

        return reward

    def get_final_reward(self) -> float:
        """
        Calculate final reward based on judgment accuracy.

        Returns:
            Final reward value
        """
        if not self.final_prediction:
            return 0.0

        prediction = self.final_prediction
        truth = self.ground_truth

        # Accuracy reward
        accuracy_reward = self._calculate_accuracy_reward(prediction, truth)

        # Compliance reward
        compliance_reward = self._calculate_compliance_reward()

        # Information efficiency reward
        info_reward = self._calculate_information_reward()

        # Total reward with weights
        total = (
            0.5 * accuracy_reward +
            0.3 * info_reward +
            0.2 * compliance_reward
        )

        return total

    def _calculate_accuracy_reward(
        self,
        prediction: Dict,
        truth: Dict
    ) -> float:
        """Calculate accuracy reward"""
        # Crime F1 (placeholder)
        crime_score = 1.0 if prediction.get("crime") == truth.get("crime") else 0.5

        # Sentence error
        pred_months = prediction.get("sentence_months", 0)
        true_months = truth.get("sentence_months", 0)
        sentence_score = 1 - abs(pred_months - true_months) / max(true_months, 1)

        # Law recall
        pred_laws = set(prediction.get("laws", []))
        true_laws = set(truth.get("laws", []))
        law_score = len(pred_laws & true_laws) / max(len(true_laws), 1)

        return 0.4 * crime_score + 0.3 * sentence_score + 0.3 * law_score

    def _calculate_compliance_reward(self) -> float:
        """Calculate compliance reward"""
        score = 0.0

        # Check legal terminology usage
        if self.final_prediction:
            judgment_text = str(self.final_prediction)
            legal_terms = ['被告人', '被害人', '依照', '判决', '犯罪']
            for term in legal_terms:
                if term in judgment_text:
                    score += 0.02

        # Round penalty
        if self.current_round > self.max_rounds:
            score -= 0.1 * (self.current_round - self.max_rounds)

        return max(0, score)

    def _calculate_information_reward(self) -> float:
        """Calculate information gathering efficiency reward"""
        # Proportion of evidence discovered
        total_levels = len(self.hidden_evidence)
        if total_levels == 0:
            return 0.0

        discovered = len(self.revealed_levels) / total_levels

        # Efficiency: discovered / rounds
        efficiency = discovered / max(self.current_round, 1)

        return efficiency

    def get_prediction(self) -> Optional[Dict]:
        """Get final prediction"""
        return self.final_prediction


def switch_role(model, role: str, context: str) -> str:
    """
    Switch model role using prompt templates.

    Args:
        model: The language model
        role: Role to switch to ('judge', 'prosecutor', 'defender')
        context: Context information

    Returns:
        Model output for the role
    """
    prompt_template = ROLE_PROMPTS.get(role, "")
    prompt = prompt_template.format(context=context)

    # TODO: Implement actual model call
    # output = model.generate(prompt)
    return f"[{role}] Response placeholder"


def main():
    """Example usage"""
    case_data = {
        "public_info": "2026年3月，被告人张三与李四发生冲突，李四受重伤。",
        "hidden_evidence": {
            "subjective": {
                "content": "张三案发前购买了折叠刀并记录被害人行踪",
                "triggers": ["动机", "预谋", "事前"]
            },
            "objective": {
                "content": "刀伤位于左胸部，深达肺部，连续刺了三刀",
                "triggers": ["手段", "伤害部位", "伤口"]
            },
            "sentencing": {
                "content": "张三现场拨打120并如实供述",
                "triggers": ["自首", "案后", "认罪"]
            }
        },
        "ground_truth": {
            "crime": "故意伤害罪",
            "sentence_months": 36,
            "laws": ["刑法第234条"]
        }
    }

    env = EvidenceEnvironment(max_rounds=10)
    state = env.reset(case_data)

    print(f"Initial state: {state.public_info}")
    print(f"Max rounds: {state.max_rounds}")

    # Simulate a query
    result = env.step({
        "type": "query",
        "content": "我想了解被告人的作案动机"
    })

    print(f"New evidence: {result.new_evidence}")
    print(f"Reward: {result.reward}")


if __name__ == "__main__":
    main()