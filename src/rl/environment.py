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
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter


def compute_rouge_similarity(text1: str, text2: str) -> float:
    """
    基于F1分数的ROUGE相似度计算

    用途：
    - 罪名相似度评估（如"故意伤害" vs "故意伤害致死"）

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        0-1范围的相似度分数
    """
    if not text1 or not text2:
        return 0.0

    try:
        import jieba
        words1 = jieba.lcut(text1)
        words2 = jieba.lcut(text2)
    except ImportError:
        # 如果jieba不可用，用简单的字符分割（适合中文罪名）
        # 将文本分割成2-4字的词组
        words1 = []
        words2 = []
        # 按常见中文词长度分割
        for i in range(len(text1)):
            for length in [4, 3, 2, 1]:
                if i + length <= len(text1):
                    words1.append(text1[i:i+length])
        for i in range(len(text2)):
            for length in [4, 3, 2, 1]:
                if i + length <= len(text2):
                    words2.append(text2[i:i+length])

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    count1 = Counter(words1)
    count2 = Counter(words2)

    common = count1 & count2
    num_common = sum(common.values())

    precision = num_common / len(words1)
    recall = num_common / len(words2)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


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

        # 动作追踪（用于人工审核和训练分析）
        self._all_actions: List[Dict] = []          # 所有动作历史
        self._valid_actions: List[Dict] = []        # 有效格式动作
        self._effective_actions: List[Dict] = []    # 触发证据的动作
        self._invalid_actions: List[Dict] = []      # 无效动作记录

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

        # 重置动作追踪
        self._all_actions = []
        self._valid_actions = []
        self._effective_actions = []
        self._invalid_actions = []

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

    def _track_action(self, action: Dict, result: StepResult):
        """
        记录每个动作的执行结果（用于人工审核和训练分析）

        Args:
            action: 执行的动作
            result: 动作执行结果
        """
        record = {
            "round": self.current_round,
            "type": action.get("type"),
            "content_preview": str(action.get("content", ""))[:50],
            "reward": result.reward,
            "triggered": result.new_evidence is not None,
            "is_valid": not result.info.get("invalid", False),
            "reason": result.info.get("reason", "")
        }

        self._all_actions.append(record)

        if record["is_valid"]:
            self._valid_actions.append(record)
            if record["triggered"]:
                self._effective_actions.append(record)
        else:
            self._invalid_actions.append(record)

    def get_action_statistics(self) -> Dict:
        """
        生成动作统计报告

        用于：
        - 训练统计分析
        - 人工审核/debug
        - 精细reward分析

        Returns:
            统计报告字典
        """
        total = len(self._all_actions)
        valid_count = len(self._valid_actions)
        effective_count = len(self._effective_actions)
        invalid_count = len(self._invalid_actions)

        return {
            "case_id": self.case_data.get("case_id", ""),
            "total_actions": total,
            "valid_count": valid_count,
            "effective_count": effective_count,
            "invalid_count": invalid_count,
            "valid_rate": valid_count / max(total, 1),
            "effective_rate": effective_count / max(total, 1),
            "invalid_details": self._invalid_actions,  # 用于人工审核
            "evidence_discovered": list(self.revealed_levels),
            "final_reward": self.get_final_reward() if self.final_prediction else 0
        }

    def step(self, action: Dict) -> StepResult:
        """
        Execute one step in environment.

        Args:
            action: Dictionary with 'type' and 'content'
                - type: "query", "judge", or "invalid"
                - content: query text, judgment dict, or invalid text
                - repetition_penalty: (optional) penalty for invalid actions

        Returns:
            StepResult with new state, reward, etc.
        """
        action_type = action.get("type")
        content = action.get("content")
        repetition_penalty = action.get("repetition_penalty", 0.0)

        if action_type == "query":
            result = self._handle_query(content)
        elif action_type == "judge":
            result = self._handle_judge(content)
        elif action_type == "invalid":
            result = self._handle_invalid(content, repetition_penalty)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        # 追踪动作执行结果
        self._track_action(action, result)

        return result

    def _handle_invalid(self, content: str, repetition_penalty: float) -> StepResult:
        """
        Handle an invalid action (repetitive/collapsed output).

        给予惩罚性reward，但不终止episode，让模型有机会修正。

        Args:
            content: Invalid action text
            repetition_penalty: Penalty value (default -0.2)

        Returns:
            StepResult with negative reward
        """
        self.current_round += 1
        self.conversation_history.append(("judge", f"[无效输出] {content[:50]}"))

        # 直接应用惩罚
        reward = repetition_penalty  # 通常是 -0.2

        state = self._get_state()

        # 如果连续无效输出超过3次，强制终止
        invalid_count = sum(1 for role, msg in self.conversation_history if "无效输出" in msg)
        if invalid_count >= 3:
            # 强制给出默认判决（失败）
            self.final_prediction = {"crime": "", "sentence_months": 0, "laws": []}
            return StepResult(
                state=state,
                reward=reward - 0.3,  # 额外惩罚：连续失败
                new_evidence=None,
                is_terminal=True,
                info={"reason": "连续无效输出终止"}
            )

        return StepResult(
            state=state,
            reward=reward,
            new_evidence=None,
            is_terminal=False,
            info={"invalid": True, "repetition_penalty": repetition_penalty}
        )

    def _handle_query(self, query: str) -> StepResult:
        """Handle a query action"""
        self.current_round += 1
        self.conversation_history.append(("judge", query))

        # Check if query triggers hidden evidence
        new_evidence = self._check_trigger(query)

        # 【新增】将控辩回复也加入对话历史
        if new_evidence:
            # 判断是控方还是辩方回复（基于证据层级）
            # 已在 _format_evidence_as_dialogue 中格式化，需要提取角色
            if "公诉人" in new_evidence:
                role = "prosecutor"
            elif "辩护人" in new_evidence:
                role = "defender"
            else:
                role = "unknown"
            self.conversation_history.append((role, new_evidence))

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
        """
        Handle a judgment action.

        新增规则：第1轮禁止判决，强制先调查
        """
        # 【关键规则】第1轮禁止判决，强制先调查
        if self.current_round == 0:
            # 第1轮尝试判决 → 拒绝，转为query惩罚
            self.current_round += 1
            self.conversation_history.append(("judge", f"[拒绝判决-第1轮] {judgment}"))

            return StepResult(
                state=self._get_state(),
                reward=-0.3,  # 早判惩罚
                new_evidence=None,
                is_terminal=False,
                info={"rejected": True, "reason": "第1轮禁止判决，请先提问调查"}
            )

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

        改进：证据释放以对话回复格式呈现，模拟控辩双方回答法官提问。

        证据层级与控辩对应关系：
        - subjective/objective (主观层/客观层) → 控方回复（定罪要素）
        - sentencing (量刑层) → 辽方回复（从轻情节）

        Args:
            query: Query text from agent

        Returns:
            Dialogue-formatted evidence content if triggered, None otherwise
        """
        for level, evidence in self.hidden_evidence.items():
            if level not in self.revealed_levels:
                triggers = evidence.get("triggers", [])

                if self._match_triggers(query, triggers):
                    self.revealed_levels.add(level)
                    raw_content = evidence.get("content", "")
                    # 格式化为对话回复
                    return self._format_evidence_as_dialogue(level, raw_content)

        return None

    def _format_evidence_as_dialogue(self, level: str, content: str) -> str:
        """
        将证据内容格式化为对话回复格式，模拟控辩双方回答法官提问。

        Args:
            level: 证据层级 (subjective/objective/sentencing)
            content: 原始证据内容

        Returns:
            对话格式的回复文本
        """
        # 如果没有证据内容，返回空
        if not content or content == "无相关信息":
            # 根据层级选择回复角色
            if level == "sentencing":
                return "辩护人：经核实，暂无相关减轻情节信息。"
            else:
                return "公诉人：经调查，暂无相关证据信息。"

        # 根据证据层级选择回复角色
        if level == "subjective":
            # 主观层 → 控方回复（定罪要素：动机、预谋等）
            return f"公诉人补充说明：经调查，{content}"

        elif level == "objective":
            # 客观层 → 控方回复（定罪要素：作案手段、后果等）
            return f"公诉人补充说明：关于作案情况，{content}"

        elif level == "sentencing":
            # 量刑层 → 辽方回复（从轻情节：自首、赔偿等）
            return f"辩护人补充说明：{content}"

        else:
            # 默认格式
            return f"补充信息：{content}"

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
        # 扩展覆盖所有罪名类型的法律关键词
        legal_keywords = [
            # 通用关键词
            "动机", "手段", "事实", "证据", "刑期", "罪名", "法条",

            # 主观层关键词
            "预谋", "故意", "明知", "主观", "目的", "事前准备",
            "获利", "非法占有", "报复", "泄愤",

            # 人身伤害类关键词
            "伤害", "伤口", "部位", "伤情", "重伤", "轻伤",
            "打击", "刺", "捅", "砍", "殴打", "持刀", "持械",

            # 财产犯罪类关键词
            "涉案金额", "数额", "骗取", "盗窃", "抢劫", "抢夺",
            "合同", "转账", "发票", "虚构", "入户", "退赃",

            # 毒品犯罪类关键词
            "毒品", "克数", "纯度", "贩卖", "运输", "持有",
            "冰毒", "海洛因", "甲基苯丙胺",

            # 职务犯罪类关键词
            "职务便利", "贪污", "受贿", "挪用", "公款", "职权",
            "侵吞", "索贿", "账目",

            # 交通犯罪类关键词
            "酒精含量", "血液", "驾驶", "事故责任", "醉酒",
            "交通肇事",

            # 性犯罪类关键词
            "强制", "猥亵", "违背意志", "强奸", "幼女",

            # 网络犯罪类关键词
            "数据", "信息系统", "侵入", "控制", "黑客",

            # 妨害社会管理类关键词
            "招嫖", "卖淫", "赌博", "淫秽", "妨害",

            # 量刑层关键词
            "自首", "案发", "作案", "认罪", "供述", "情节",
            "报案", "谅解", "投案", "赔偿", "退赃", "累犯"
        ]
        if not any(kw in query for kw in legal_keywords):
            reward -= 0.02  # Penalty for irrelevant query

        return reward

    def get_final_reward(self) -> float:
        """
        Calculate final reward based on judgment accuracy.

        新版reward计算（v5 - 混合相似度 + 详细过程奖励）：
        - 30% 准确性reward（罪名ROUGE + 刑期误差 + 法条召回）
        - 40% 信息收集reward（调查深度）
        - 30% 过程reward/惩罚

        关键改进：
        1. ROUGE相似度用于罪名评估
        2. 调查深度奖励（每层+0.1）
        3. 早判惩罚（第1轮判决且无调查 → -0.5）
        4. 效率奖励（3轮内且有调查 → +0.1）

        Returns:
            Final reward value
        """
        if not self.final_prediction:
            return -0.5  # 没有判决，严重失败

        prediction = self.final_prediction
        truth = self.ground_truth

        # ========== 1. 准确性奖励 (30%) ==========
        accuracy_reward = self._calculate_accuracy_reward(prediction, truth)

        # ========== 2. 信息收集奖励 (40% - 核心) ==========
        total_levels = len(self.hidden_evidence)
        discovered_levels = len(self.revealed_levels)
        if total_levels > 0:
            discovered_ratio = discovered_levels / total_levels
            info_reward = discovered_ratio  # 0-1
            # 每层额外奖励
            info_reward += 0.1 * discovered_levels
        else:
            discovered_ratio = 0.0
            info_reward = 0

        # ========== 3. 过程奖励/惩罚 ==========
        process_reward = 0.0

        # 3.1 早判惩罚（第1轮判决且无调查）
        if self.current_round == 1 and discovered_levels == 0:
            process_reward -= 0.5

        # 3.2 过早判决惩罚（未收集足够证据）
        elif total_levels > 0 and discovered_ratio < 0.5:
            penalty = -0.2 * (1 - discovered_ratio)
            process_reward += penalty

        # 3.3 判决格式有效性检查
        if prediction:
            crime = prediction.get("crime", "")
            if not crime or len(crime) < 2:
                process_reward -= 0.2  # 罪名提取失败

            if prediction.get("sentence_months", 0) <= 0:
                process_reward -= 0.1  # 刑期提取失败

        # 3.4 效率奖励（3轮内且有调查）
        if self.current_round <= 3 and discovered_levels >= 1:
            process_reward += 0.1

        # 3.5 最大轮次惩罚
        if self.current_round >= self.max_rounds:
            process_reward -= 0.2

        # ========== 4. 最终计算 ==========
        total = 0.3 * accuracy_reward + 0.4 * info_reward + process_reward

        # 范围 [-1.0, 1.5]
        total = max(-1.0, min(1.5, total))

        return total

    def _calculate_accuracy_reward(
        self,
        prediction: Dict,
        truth: Dict
    ) -> float:
        """
        混合相似度计算准确性reward

        组成：
        - 罪名：ROUGE F1 (权重40%)
        - 刑期：相对误差 (权重30%)
        - 法条：集合召回率 (权重30%)
        """
        # 1. 罪名相似度（ROUGE）
        pred_crime = prediction.get("crime", "")
        true_crime = truth.get("crime", "")
        crime_sim = compute_rouge_similarity(pred_crime, true_crime)
        crime_reward = 0.4 * crime_sim

        # 2. 刑期准确度（相对误差）
        pred_months = prediction.get("sentence_months", 0)
        true_months = truth.get("sentence_months", 0)
        if true_months > 0:
            sentence_error = abs(pred_months - true_months) / true_months
            sentence_reward = 0.3 * (1 - min(sentence_error, 1))
        else:
            sentence_reward = 0

        # 3. 法条召回率（集合匹配）
        pred_laws = set(prediction.get("laws", []))
        true_laws = set(truth.get("laws", []))
        if len(true_laws) > 0:
            law_recall = len(pred_laws & true_laws) / len(true_laws)
            law_reward = 0.3 * law_recall
        else:
            law_reward = 0

        return crime_reward + sentence_reward + law_reward

    def _calculate_compliance_reward(self) -> float:
        """Calculate compliance reward (deprecated, now in get_final_reward)"""
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
        """Calculate information gathering efficiency reward (deprecated)"""
        # Proportion of evidence discovered
        total_levels = len(self.hidden_evidence)
        if total_levels == 0:
            return 0.0

        discovered = len(self.revealed_levels) / total_levels

        # Efficiency: discovered / rounds
        efficiency = discovered / max(self.current_round, 1)

        return efficiency

    def _has_repetition(self, text: str) -> bool:
        """
        检测输出是否有重复循环

        Args:
            text: 输出文本

        Returns:
            True if repetition detected
        """
        if len(text) < 20:
            return False

        # 1. 检查句子重复（以句号分隔）
        sentences = re.split(r'[。，]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 3:
            # 检查是否有连续2个以上相同句子
            for i in range(len(sentences) - 1):
                if sentences[i] == sentences[i + 1] and len(sentences[i]) > 10:
                    return True

        # 2. 检查短语重复（如"判决：罪名：..."重复出现）
        # 匹配模式：相同短语重复出现
        pattern = r'(.{15,}?)[。，\s]*\1'
        if re.search(pattern, text):
            return True

        # 3. 检查单词/短语重复比例
        words = text.split()
        if len(words) >= 10:
            # 统计重复词（长度>3的词）
            word_counts = {}
            for word in words:
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1

            # 如果某个词出现超过30%的比例，认为有重复
            for word, count in word_counts.items():
                if count / len(words) > 0.3:
                    return True

        return False

    def _has_valid_format(self, text: str) -> bool:
        """
        检测输出格式是否正确

        Args:
            text: 输出文本

        Returns:
            True if format is valid
        """
        # 必须包含"判决"关键词
        if "判决" not in text:
            return False

        # 必须包含罪名相关词
        if "罪名" not in text and "犯" not in text:
            return False

        # 不能有明显的重复
        if self._has_repetition(text):
            return False

        return True

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