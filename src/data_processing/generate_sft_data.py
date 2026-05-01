"""
SFT数据生成脚本
将SimuCourt和judge-data转换为统一的instruction/input/output格式

运行方式：
    python src/data_processing/generate_sft_data.py
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict
import argparse

# 确保可以导入模板
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_processing.sft_templates import (
    convert_simucourt1_to_sft,
    convert_simucourt2_to_sft,
    convert_judgedata_to_sft,
    convert_judgedata_jsonl_to_sft,
    datapoint_to_json,
    SFTDataPoint
)


def load_json(path: str) -> List[Dict]:
    """加载JSON数据"""
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 尝试作为标准JSON加载
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果失败，尝试作为JSONL格式（每行一个JSON）
            data = []
            for line in content.strip().split('\n'):
                if line.strip():
                    data.append(json.loads(line))
            return data


def process_all_data(output_dir: str = "data/processed"):
    """处理所有数据源"""

    # 数据路径
    simucourt1_path = "data/raw/agentscourt-data/SimuCourt(1).json"
    simucourt2_path = "data/raw/agentscourt-data/SimuCourt(2).json"
    judgedata_path = "data/raw/judge-data/all.json"
    judgedata_train_path = "data/raw/judge-data/train.json"
    judgedata_test_path = "data/raw/judge-data/test.json"

    # 加载数据
    print("=" * 60)
    print("加载原始数据...")

    simucourt1 = load_json(simucourt1_path)
    print(f"SimuCourt(1) 一审数据: {len(simucourt1)} 条")

    simucourt2 = load_json(simucourt2_path)
    print(f"SimuCourt(2) 二审数据: {len(simucourt2)} 条")

    judgedata = load_json(judgedata_path)
    print(f"judge-data 全部数据: {len(judgedata)} 条")

    judgedata_train = load_json(judgedata_train_path)
    print(f"judge-data 训练集: {len(judgedata_train)} 条")

    judgedata_test = load_json(judgedata_test_path)
    print(f"judge-data 测试集: {len(judgedata_test)} 条")

    # 转换数据
    print("\n" + "=" * 60)
    print("转换数据为SFT格式...")

    all_data_points = []

    # 处理SimuCourt(1)
    print("\n处理SimuCourt(1)一审数据...")
    sc1_prosecutor = 0
    sc1_defender = 0
    sc1_judge = 0
    for case in simucourt1:
        dps = convert_simucourt1_to_sft(case)
        for dp in dps:
            if dp.role == "prosecutor":
                sc1_prosecutor += 1
            elif dp.role == "defender":
                sc1_defender += 1
            else:
                sc1_judge += 1
        all_data_points.extend(dps)
    print(f"  控方数据: {sc1_prosecutor} 条")
    print(f"  辩方数据: {sc1_defender} 条")
    print(f"  法官数据: {sc1_judge} 条")

    # 处理SimuCourt(2)
    print("\n处理SimuCourt(2)二审数据...")
    sc2_prosecutor = 0
    sc2_defender = 0
    sc2_judge = 0
    for case in simucourt2:
        dps = convert_simucourt2_to_sft(case)
        for dp in dps:
            if dp.role == "prosecutor":
                sc2_prosecutor += 1
            elif dp.role == "defender":
                sc2_defender += 1
            else:
                sc2_judge += 1
        all_data_points.extend(dps)
    print(f"  控方数据: {sc2_prosecutor} 条")
    print(f"  辩方数据: {sc2_defender} 条")
    print(f"  法官数据: {sc2_judge} 条")

    # 处理judge-data all.json（结构化数据，高质量，作为主要来源）
    print("\n处理judge-data all.json(结构化数据)...")
    jd_all_judge = 0
    for case in judgedata:
        dps = convert_judgedata_to_sft(case)
        jd_all_judge += len(dps)
        all_data_points.extend(dps)
    print(f"  法官数据: {jd_all_judge} 条")

    # 注意：不再使用train.json，因为可能与all.json重叠
    # train.json/test.json是检索任务格式，不适合SFT训练

    # 统计总计
    print("\n" + "=" * 60)
    print("数据统计汇总:")
    print(f"  总数据点: {len(all_data_points)} 条")

    # 按角色统计
    role_counts = {"prosecutor": 0, "defender": 0, "judge": 0}
    for dp in all_data_points:
        role_counts[dp.role] += 1

    print(f"  控方(prosecutor): {role_counts['prosecutor']} 条")
    print(f"  辩方(defender): {role_counts['defender']} 条")
    print(f"  法官(judge): {role_counts['judge']} 条")

    # 按数据源统计
    source_counts = {"simucourt1": 0, "simucourt2": 0, "judgedata": 0}
    for dp in all_data_points:
        if dp.source in source_counts:
            source_counts[dp.source] += 1

    print(f"  SimuCourt(1): {source_counts['simucourt1']} 条")
    print(f"  SimuCourt(2): {source_counts['simucourt2']} 条")
    print(f"  judge-data(all.json): {source_counts['judgedata']} 条")

    # 分角色保存
    print("\n" + "=" * 60)
    print(f"保存数据到 {output_dir}/")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 分角色数据
    prosecutor_data = [datapoint_to_json(dp) for dp in all_data_points if dp.role == "prosecutor"]
    defender_data = [datapoint_to_json(dp) for dp in all_data_points if dp.role == "defender"]
    judge_data = [datapoint_to_json(dp) for dp in all_data_points if dp.role == "judge"]

    # 保存
    with open(f"{output_dir}/prosecutor.json", 'w', encoding='utf-8') as f:
        json.dump(prosecutor_data, f, ensure_ascii=False, indent=2)
    print(f"  prosecutor.json: {len(prosecutor_data)} 条")

    with open(f"{output_dir}/defender.json", 'w', encoding='utf-8') as f:
        json.dump(defender_data, f, ensure_ascii=False, indent=2)
    print(f"  defender.json: {len(defender_data)} 条")

    with open(f"{output_dir}/judge.json", 'w', encoding='utf-8') as f:
        json.dump(judge_data, f, ensure_ascii=False, indent=2)
    print(f"  judge.json: {len(judge_data)} 条")

    # 合并保存（用于混合训练）
    all_data = [datapoint_to_json(dp) for dp in all_data_points]
    with open(f"{output_dir}/all_roles.json", 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"  all_roles.json: {len(all_data)} 条")

    return all_data_points


def show_sample_data(data_points: List[SFTDataPoint], n: int = 2):
    """展示样例数据"""
    print("\n" + "=" * 60)
    print("样例数据展示")
    print("=" * 60)

    # 按角色展示
    roles = ["prosecutor", "defender", "judge"]
    for role in roles:
        role_data = [dp for dp in data_points if dp.role == role]
        if role_data:
            print(f"\n--- {role} 角色样例 ---")
            dp = role_data[0]
            print(f"\n【INSTRUCTION】\n{dp.instruction[:200]}...")
            print(f"\n【INPUT】\n{dp.input[:500]}...")
            print(f"\n【OUTPUT】\n{dp.output[:500]}...")
            print(f"\n来源: {dp.source} | 案件ID: {dp.case_id}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="生成SFT训练数据")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/processed",
        help="输出目录"
    )
    parser.add_argument(
        "--show-sample", "-s",
        action="store_true",
        help="显示样例数据"
    )

    args = parser.parse_args()

    # 处理数据
    data_points = process_all_data(args.output)

    # 显示样例
    if args.show_sample:
        show_sample_data(data_points)

    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()