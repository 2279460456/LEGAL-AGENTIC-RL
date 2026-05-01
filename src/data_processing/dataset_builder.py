"""
Dataset Builder Module
Constructs training datasets for SFT and RL phases.

SFT Dataset Format:
    {"instruction": str, "input": str, "output": str}

    - Prosecutor: Case summary → Prosecution argument
    - Defender: Case summary → Defense argument
    - Judge: Case + arguments → CoT reasoning + judgment

RL Dataset Format:
    {
        "case_id": str,
        "public_info": str,
        "hidden_evidence": {...},
        "ground_truth": {...}
    }
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .cot_distill import CoTDistiller, CoTOutput
from .evidence_split import EvidenceSplitter, CaseEvidenceStructure


# Role-specific instruction templates
ROLE_INSTRUCTIONS = {
    "prosecutor": """你是一位检察官，负责指控犯罪。
请根据案情事实，提出指控理由。
你的任务：
1. 分析案件事实，确定涉嫌罪名
2. 列出定罪的关键要素
3. 引用相关法律条文
4. 提出量刑建议的理由

请以专业、严谨的语言提出指控意见。""",

    "defender": """你是一位辩护律师，负责为被告人辩护。
请根据案情事实，提出辩护意见。
你的任务：
1. 分析案件事实，寻找有利情节
2. 提出量刑减轻的理由
3. 指出证据可能存在的问题
4. 引用相关法律规定

请以专业、有理有据的语言提出辩护意见。""",

    "judge": """你是一位资深法官，正在审理案件。
请根据案情和控辩双方意见，作出判决。
你的任务：
1. 事实认定：综合控辩意见，确认案件事实
2. 法律适用：引用相关法条，说明适用理由
3. 量刑考量：分析加重和减轻情节
4. 判决结论：确定罪名、刑期、法律依据

请严格按照法律程序，作出公正判决。"""
}


@dataclass
class SFTDataPoint:
    """Single SFT training data point"""
    instruction: str
    input: str
    output: str
    role: str


class DatasetBuilder:
    """
    Builds training datasets for SFT and RL phases.

    Usage:
        builder = DatasetBuilder()
        sft_data = builder.build_sft_dataset(judgments, roles=["prosecutor", "defender", "judge"])
        rl_data = builder.build_rl_dataset(judgments)
    """

    def __init__(
        self,
        cot_distiller: Optional[CoTDistiller] = None,
        evidence_splitter: Optional[EvidenceSplitter] = None
    ):
        """
        Initialize Dataset Builder.

        Args:
            cot_distiller: CoT distillation module
            evidence_splitter: Evidence splitting module
        """
        self.cot_distiller = cot_distiller or CoTDistiller()
        self.evidence_splitter = evidence_splitter or EvidenceSplitter()
        self.role_instructions = ROLE_INSTRUCTIONS

    def build_sft_dataset(
        self,
        judgments: List[Dict],
        roles: List[str] = ["prosecutor", "defender", "judge"],
        output_dir: Optional[str] = None
    ) -> Dict[str, List[SFTDataPoint]]:
        """
        Build SFT training datasets for each role.

        Args:
            judgments: List of judgment documents
            roles: Roles to create datasets for
            output_dir: Directory to save datasets

        Returns:
            Dict mapping role to list of data points
        """
        datasets = {}

        for role in roles:
            role_data = []
            for judgment in judgments:
                data_point = self._create_sft_datapoint(judgment, role)
                if data_point:
                    role_data.append(data_point)

            datasets[role] = role_data

            if output_dir:
                self._save_sft_dataset(role_data, role, output_dir)

        return datasets

    def _create_sft_datapoint(
        self,
        judgment: Dict,
        role: str
    ) -> Optional[SFTDataPoint]:
        """Create a single SFT data point for a role"""

        instruction = self.role_instructions[role]

        if role == "prosecutor":
            # Prosecutor: case summary → prosecution argument
            input_text = self._format_case_summary(judgment)
            output_text = self._generate_prosecution_output(judgment)

        elif role == "defender":
            # Defender: case summary → defense argument
            input_text = self._format_case_summary(judgment)
            output_text = self._generate_defense_output(judgment)

        elif role == "judge":
            # Judge: case + arguments → judgment
            input_text = self._format_judge_input(judgment)
            output_text = self._generate_judge_output(judgment)

        else:
            return None

        return SFTDataPoint(
            instruction=instruction,
            input=input_text,
            output=output_text,
            role=role
        )

    def _format_case_summary(self, judgment: Dict) -> str:
        """Format case summary for input"""
        parts = []

        if "defendant" in judgment:
            parts.append(f"被告人：{judgment['defendant']}")
        if "victim" in judgment:
            parts.append(f"被害人：{judgment['victim']}")
        if "date" in judgment:
            parts.append(f"案发时间：{judgment['date']}")
        if "fact" in judgment:
            parts.append(f"案情事实：{judgment['fact']}")

        return "\n".join(parts)

    def _format_judge_input(self, judgment: Dict) -> str:
        """Format input for judge role"""
        summary = self._format_case_summary(judgment)

        # Add mock prosecution and defense arguments
        prosecution = judgment.get("prosecution_argument", "控方意见：被告人涉嫌犯罪...")
        defense = judgment.get("defense_argument", "辩方意见：被告人有减轻情节...")

        return f"{summary}\n\n控方指控：\n{prosecution}\n\n辩方辩护：\n{defense}"

    def _generate_prosecution_output(self, judgment: Dict) -> str:
        """Generate prosecution output"""
        # In actual implementation, use CoT distillation
        crime = judgment.get("crime", "")
        laws = judgment.get("laws", [])

        output = f"""
【指控意见】

一、案件事实分析
{judgment.get('fact', '案情事实...')}

二、罪名认定
被告人涉嫌{crime}，符合以下构成要件：
1. 主观方面：具有故意
2. 害观方面：实施了危害行为
3. 危害结果：造成损害

三、法律依据
引用法条：{', '.join(laws)}

四、量刑建议
建议依法惩处。
"""
        return output

    def _generate_defense_output(self, judgment: Dict) -> str:
        """Generate defense output"""
        output = f"""
【辩护意见】

一、事实认定意见
对案件事实的部分认定持有异议。

二、减轻情节
1. 被告人认罪态度较好
2. 有悔罪表现
3. 积极配合调查

三、法律适用意见
请求法庭充分考虑减轻情节。

四、量刑建议
建议从轻处罚。
"""
        return output

    def _generate_judge_output(self, judgment: Dict) -> str:
        """Generate judge output with CoT reasoning"""
        crime = judgment.get("crime", "")
        sentence = judgment.get("sentence_text", "")
        laws = judgment.get("laws", [])

        output = f"""
【判决书】

一、事实认定
经审理查明，{judgment.get('fact', '案件事实...')}

二、法律适用
本院认为，被告人的行为已构成{crime}。
依照{', '.join(laws)}之规定，应当追究刑事责任。

三、量刑考量
综合考虑：
1. 案件的具体情节
2. 被告人的认罪态度
3. 社会危害程度

四、判决结论
被告人犯{crime}，{sentence}。
"""
        return output

    def build_rl_dataset(
        self,
        judgments: List[Dict],
        output_path: Optional[str] = None
    ) -> List[CaseEvidenceStructure]:
        """
        Build RL environment dataset.

        Args:
            judgments: List of judgment documents
            output_path: Path to save dataset

        Returns:
            List of CaseEvidenceStructure
        """
        rl_cases = []

        for judgment in judgments:
            # Split evidence into visible/hidden layers
            structure = self.evidence_splitter.split(judgment)
            rl_cases.append(structure)

        if output_path:
            self._save_rl_dataset(rl_cases, output_path)

        return rl_cases

    def _save_sft_dataset(
        self,
        data: List[SFTDataPoint],
        role: str,
        output_dir: str
    ):
        """Save SFT dataset to JSON"""
        path = Path(output_dir) / f"{role}.json"

        json_data = [
            {
                "instruction": dp.instruction,
                "input": dp.input,
                "output": dp.output
            }
            for dp in data
        ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(data)} samples to {path}")

    def _save_rl_dataset(
        self,
        cases: List[CaseEvidenceStructure],
        path: str
    ):
        """Save RL dataset to JSON"""
        # Use evidence splitter's save method
        self.evidence_splitter.batch_split([], path)  # Reuse save logic


def main():
    """Example usage"""
    sample_judgments = [
        {
            "case_id": "case_001",
            "defendant": "张三",
            "victim": "李四",
            "date": "2026年3月",
            "fact": "张三与李四发生冲突，持刀将李四刺伤",
            "crime": "故意伤害罪",
            "laws": ["刑法第234条"],
            "sentence_text": "判处有期徒刑三年",
            "sentence_months": 36
        }
    ]

    builder = DatasetBuilder()

    # Build SFT datasets
    sft_datasets = builder.build_sft_dataset(
        sample_judgments,
        roles=["prosecutor", "defender", "judge"],
        output_dir="data/processed"
    )

    # Build RL dataset
    rl_cases = builder.build_rl_dataset(
        sample_judgments,
        output_path="data/rl_env/train_cases.json"
    )


if __name__ == "__main__":
    main()