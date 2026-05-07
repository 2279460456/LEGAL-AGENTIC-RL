"""
GRPO训练数据集构建脚本
从judge-data构建RL环境数据，使用LLM辅助证据拆分

运行方式：
    # 不使用LLM（规则匹配）
    python src/data_processing/build_rl_data.py --max_samples 600

    # 使用LLM拆分（需先配置configs/llm_config.yaml）
    python src/data_processing/build_rl_data.py --config configs/llm_config.yaml --use_llm

功能：
1. 加载judge-data/all.json
2. 调用LLM/规则匹配拆分Fact为三层隐藏证据
3. 提取ground_truth（罪名、刑期、法条）
4. 验证数据格式
5. 划分训练/测试集

用户自定义LLM：
    请修改 src/data_processing/llm_evidence_splitter.py
    继承 LLMEvidenceSplitterBase 类并实现 call_llm 方法
"""

import os
import sys
import json
import re
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from tqdm import tqdm
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入模块化的LLM拆分器
from llm_evidence_splitter import (
    LLMEvidenceSplitterBase,
    DefaultLLMSplitter,
    load_llm_config,
    LLMConfig,
    EVIDENCE_ROLES_GENERIC,
    CRIME_SPECIFIC_EVIDENCE_LAYERS,
    CRIME_CATEGORY_TRIGGERS,
    CRIME_TO_CATEGORY_MAP,
    REQUIRED_ELEMENTS,
    get_required_elements,
    get_crime_specific_evidence_layers,
    get_evidence_triggers,
    get_crime_category,
    EVIDENCE_SPLIT_PROMPT,
    build_dynamic_evidence_split_prompt
)


# 保留通用触发关键词定义（作为默认fallback）
TRIGGER_KEYWORDS_DEFAULT = {
    "subjective": [
        "作案动机", "动机", "为什么", "原因", "目的",
        "事前准备", "预谋", "策划", "蓄意", "谋划",
        "作案工具来源", "购买行为", "购买工具", "工具来源",
        "是否有预谋", "故意", "明知", "主观"
    ],
    "objective": [
        "作案手段", "手段", "方式", "怎么做的", "怎么实施", "具体行为",
        "伤害部位", "伤口位置", "伤口", "打击部位", "伤情",
        "打击力度", "刺了几刀", "连续", "多次", "几下",
        "伤害程度", "重伤", "轻伤", "伤情鉴定",
        "涉案金额", "金额", "数额", "骗取"
    ],
    "sentencing": [
        "案后表现", "自首", "投案", "主动投案", "报案情况", "拨打110", "拨打120",
        "赔偿", "赔偿情况", "医疗费", "损失赔偿", "和解",
        "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
        "谅解", "取得谅解", "被害人态度", "退赃"
    ]
}


# 保持向后兼容的TRIGGER_KEYWORDS
TRIGGER_KEYWORDS = TRIGGER_KEYWORDS_DEFAULT


def extract_sentence_months(sentence_list: List[str]) -> int:
    """
    从Sentence列表提取刑期月数

    Args:
        sentence_list: 刑期文本列表，如["有期徒刑一年", "拘役四个月"]

    Returns:
        刑期月数（-1=死刑，-2=无期，0=无法解析）
    """
    if not sentence_list:
        return 0

    sentence_text = " ".join(sentence_list)

    # 死刑
    if "死刑" in sentence_text:
        return -1

    # 无期徒刑
    if "无期" in sentence_text:
        return -2

    # 中文数字映射
    chinese_nums = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
        "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
        "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20
    }

    def chinese_to_int(text):
        for cn, num in chinese_nums.items():
            if cn in text:
                return num
        return 0

    # 有期徒刑（阿拉伯数字）
    match = re.search(r"有期徒刑(\d+)年", sentence_text)
    if match:
        return int(match.group(1)) * 12

    # 有期徒刑（中文数字）
    match = re.search(r"有期徒刑([一二三四五六七八九十]+)年", sentence_text)
    if match:
        return chinese_to_int(match.group(1)) * 12

    # 有期徒刑月数
    match = re.search(r"有期徒刑(\d+)个?月", sentence_text)
    if match:
        return int(match.group(1))

    # 拘役
    match = re.search(r"拘役(\d+)个?月", sentence_text)
    if match:
        return int(match.group(1))

    # 缓刑
    match = re.search(r"缓刑(\d+)年", sentence_text)
    if match:
        return int(match.group(1)) * 12

    return 0


def standardize_law_articles(article_numbers: List[int]) -> List[str]:
    """
    将法条数字转换为标准格式

    Args:
        article_numbers: 法条号列表，如[67, 133, 72]

    Returns:
        标准格式法条列表，如["刑法第67条", "刑法第133条"]
    """
    laws = []
    for num in article_numbers:
        if num > 0:
            laws.append(f"刑法第{num}条")
    return laws


def load_judge_data(data_path: str) -> List[Dict]:
    """
    加载judge-data数据

    Args:
        data_path: 数据路径

    Returns:
        案件列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} cases from {data_path}")
    return data


def build_single_rl_case(
    record: Dict,
    splitter: LLMEvidenceSplitterBase
) -> Optional[Dict]:
    """
    构建单个RL训练样本

    Args:
        record: judge-data单条记录
        splitter: LLM拆分器实例

    Returns:
        RL案例数据或None（无效样本）
    """
    case_id = record.get("CaseId", "")

    # 提取Fact
    fact = record.get("Fact", "")
    if not fact or len(fact) < 50:
        return None

    # 提取罪名
    crime_types = record.get("Crime Type", [])
    if not crime_types:
        return None

    crime = crime_types[0] if isinstance(crime_types, list) else crime_types

    # 提取刑期
    sentence_list = record.get("Sentence", [])
    sentence_months = extract_sentence_months(sentence_list)

    # 提取法条
    law_articles = record.get("Law Articles", [])
    laws = standardize_law_articles(law_articles)

    # 使用splitter拆分证据层（内部自动处理LLM失败回退）
    evidence_split = splitter.split(fact, crime)

    # 处理证据内容（兼容两种格式）
    def get_evidence_content(evidence_data):
        if evidence_data is None:
            return "无相关信息"
        if isinstance(evidence_data, dict):
            return evidence_data.get("content") or "无相关信息"
        return evidence_data or "无相关信息"

    def get_evidence_role(evidence_data, level, crime_type):
        """获取证据的法律作用描述"""
        if isinstance(evidence_data, dict) and evidence_data.get("role"):
            # LLM已返回正确的role，直接使用
            return evidence_data["role"]
        # 否则使用罪名特异性配置
        layers_config = get_crime_specific_evidence_layers(crime_type)
        return layers_config[level]["legal_role"]

    # 获取罪名特异性触发词
    crime_triggers = get_evidence_triggers(crime)

    # 获取必经要素清单
    required_elements = get_required_elements(crime)

    # 构建RL数据格式
    rl_case = {
        "case_id": case_id,
        "public_info": evidence_split["public_info"],
        "hidden_evidence": {
            "subjective": {
                "content": get_evidence_content(evidence_split.get("subjective_evidence")),
                "triggers": crime_triggers["subjective"],  # 使用罪名特异性触发词
                "role": get_evidence_role(evidence_split.get("subjective_evidence"), "subjective", crime)
            },
            "objective": {
                "content": get_evidence_content(evidence_split.get("objective_evidence")),
                "triggers": crime_triggers["objective"],  # 使用罪名特异性触发词
                "role": get_evidence_role(evidence_split.get("objective_evidence"), "objective", crime)
            },
            "sentencing": {
                "content": get_evidence_content(evidence_split.get("sentencing_evidence")),
                "triggers": crime_triggers["sentencing"],  # 使用罪名特异性触发词
                "role": get_evidence_role(evidence_split.get("sentencing_evidence"), "sentencing", crime)
            }
        },
        "ground_truth": {
            "crime": crime,
            "sentence_months": sentence_months,
            "laws": laws,
            "sentence_text": sentence_list[0] if sentence_list else "",
            "required_elements": required_elements
        }
    }

    return rl_case


def validate_rl_case(case: Dict) -> bool:
    """
    验证RL数据格式有效性

    Args:
        case: RL案例数据

    Returns:
        是否有效
    """
    required_fields = ["case_id", "public_info", "hidden_evidence", "ground_truth"]

    for field in required_fields:
        if field not in case:
            return False

    # 检查hidden_evidence结构
    for level in ["subjective", "objective", "sentencing"]:
        if level not in case["hidden_evidence"]:
            return False
        if "content" not in case["hidden_evidence"][level]:
            return False
        if "triggers" not in case["hidden_evidence"][level]:
            return False

    # 检查ground_truth
    for key in ["crime", "sentence_months", "laws", "required_elements"]:
        if key not in case["ground_truth"]:
            return False

    # 罪名不能为空
    if not case["ground_truth"]["crime"]:
        return False

    # public_info不能为空
    if not case["public_info"] or len(case["public_info"]) < 10:
        return False

    return True


def split_train_test(
    cases: List[Dict],
    train_ratio: float = 0.8,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    划分训练/测试集

    Args:
        cases: 全部案例
        train_ratio: 训练集比例
        random_seed: 随机种子

    Returns:
        (train_cases, test_cases)
    """
    random.seed(random_seed)
    random.shuffle(cases)

    train_size = int(len(cases) * train_ratio)
    train_cases = cases[:train_size]
    test_cases = cases[train_size:]

    return train_cases, test_cases


def load_existing_cases(output_dir: str) -> Tuple[Dict[str, Dict], Set[str]]:
    """
    加载已有的处理结果，用于断点续传

    Args:
        output_dir: 输出目录

    Returns:
        (已处理案例字典, 已处理case_id集合)
    """
    output_path = Path(output_dir)

    existing_cases = {}
    processed_ids = set()

    # 尝试加载增量保存的临时文件
    temp_file = output_path / "temp_processed_cases.json"
    if temp_file.exists():
        try:
            with open(temp_file, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
                for case in temp_data:
                    case_id = case.get("case_id", "")
                    if case_id:
                        existing_cases[case_id] = case
                        processed_ids.add(case_id)
            print(f"Loaded {len(processed_ids)} previously processed cases from temp file")
        except Exception as e:
            print(f"Warning: Failed to load temp file: {e}")

    # 也尝试加载最终的训练和测试文件
    for filename in ["train_cases.json", "test_cases.json"]:
        file_path = output_path / filename
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for case in data:
                        case_id = case.get("case_id", "")
                        if case_id and case_id not in processed_ids:
                            existing_cases[case_id] = case
                            processed_ids.add(case_id)
            except Exception as e:
                print(f"Warning: Failed to load {filename}: {e}")

    return existing_cases, processed_ids


def save_temp_cases(output_dir: str, cases: List[Dict], batch_num: int = None):
    """
    增量保存临时处理结果

    Args:
        output_dir: 输出目录
        cases: 当前已处理案例列表
        batch_num: 当前批次号（用于日志）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temp_file = output_path / "temp_processed_cases.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    if batch_num:
        print(f"  [Batch {batch_num}] Saved {len(cases)} cases to temp file")


def build_rl_dataset(
    data_path: str,
    output_dir: str,
    splitter: LLMEvidenceSplitterBase,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
    resume: bool = True,
    save_interval: int = 50
) -> Dict:
    """
    构建完整RL数据集（支持断点续传）

    Args:
        data_path: judge-data路径
        output_dir: 输出目录
        splitter: LLM拆分器实例
        max_samples: 最大样本数（用于测试）
        train_ratio: 训练集比例
        resume: 是否启用断点续传（默认True）
        save_interval: 增量保存间隔（每处理多少条保存一次）

    Returns:
        统计信息
    """
    # 加载原始数据
    raw_data = load_judge_data(data_path)

    if max_samples:
        raw_data = raw_data[:max_samples]

    # 断点续传：加载已处理数据
    existing_cases = {}
    processed_ids = set()
    skipped_count = 0

    if resume:
        existing_cases, processed_ids = load_existing_cases(output_dir)

    # 构建RL数据
    rl_cases = list(existing_cases.values())  # 从已有数据开始
    invalid_count = 0

    print(f"\n{'='*60}")
    print("Building RL dataset...")
    print(f"{'='*60}")
    print(f"Total cases: {len(raw_data)}")
    print(f"Previously processed: {len(processed_ids)}")
    print(f"LLM enabled: {splitter.config.enabled}")
    print(f"Resume mode: {resume}")

    batch_num = 0
    for record in tqdm(raw_data, desc="Processing cases"):
        # 检查是否已处理
        case_id = record.get("CaseId", "")
        if resume and case_id in processed_ids:
            skipped_count += 1
            continue

        case = build_single_rl_case(record, splitter)

        if case and validate_rl_case(case):
            rl_cases.append(case)
        else:
            invalid_count += 1

        # 增量保存
        batch_num += 1
        if batch_num % save_interval == 0:
            save_temp_cases(output_dir, rl_cases, batch_num // save_interval)

    print(f"\nValid cases: {len(rl_cases)}")
    print(f"Invalid cases: {invalid_count}")
    print(f"Skipped (already processed): {skipped_count}")

    # 划分训练/测试集
    train_cases, test_cases = split_train_test(rl_cases, train_ratio)

    print(f"Train cases: {len(train_cases)}")
    print(f"Test cases: {len(test_cases)}")

    # 保存数据
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train_cases.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_cases, f, ensure_ascii=False, indent=2)
    print(f"Saved train cases to: {train_file}")

    test_file = output_path / "test_cases.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_cases, f, ensure_ascii=False, indent=2)
    print(f"Saved test cases to: {test_file}")

    # 清理临时文件
    temp_file = output_path / "temp_processed_cases.json"
    if temp_file.exists():
        temp_file.unlink()
        print(f"Cleaned up temp file: {temp_file}")

    # 保存统计信息
    stats = {
        "timestamp": datetime.now().isoformat(),
        "source": data_path,
        "total_raw": len(raw_data),
        "valid_cases": len(rl_cases),
        "invalid_cases": invalid_count,
        "skipped_cases": skipped_count,
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "llm_enabled": splitter.config.enabled,
        "llm_model": splitter.config.model if splitter.config.enabled else None,
        "train_ratio": train_ratio,
        "resume_mode": resume
    }

    stats_file = output_path / "build_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved stats to: {stats_file}")

    return stats


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Build RL dataset for GRPO training")
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/judge-data/all.json",
        help="Path to judge-data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rl_env",
        help="Output directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/llm_config.yaml",
        help="LLM config file"
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use LLM for evidence splitting"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume mode (start from scratch)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help="Save interval for incremental saving (default: 50)"
    )

    args = parser.parse_args()

    # 加载LLM配置
    llm_config = load_llm_config(args.config)

    # 如果命令行指定use_llm，强制启用
    if args.use_llm:
        llm_config.enabled = True

    # 创建拆分器实例
    splitter = DefaultLLMSplitter(llm_config)

    # 构建数据集（默认启用断点续传）
    stats = build_rl_dataset(
        data_path=args.data,
        output_dir=args.output,
        splitter=splitter,
        max_samples=args.max_samples,
        train_ratio=args.train_ratio,
        resume=not args.no_resume,  # 默认启用断点续传
        save_interval=args.save_interval
    )

    # 打印摘要
    print(f"\n{'='*60}")
    print("Build Summary")
    print(f"{'='*60}")
    print(f"Valid cases: {stats['valid_cases']}")
    print(f"Train cases: {stats['train_cases']}")
    print(f"Test cases: {stats['test_cases']}")
    if stats.get('skipped_cases', 0) > 0:
        print(f"Skipped (resumed): {stats['skipped_cases']}")
    print(f"\nDone!")


if __name__ == "__main__":
    main()