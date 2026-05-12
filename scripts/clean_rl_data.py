"""
清理RL训练数据
移除或修复"无相关信息"的case

使用方法:
    python scripts/clean_rl_data.py --mode remove  # 移除无效数据
    python scripts/clean_rl_data.py --mode keep    # 只标记，不移除
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List


def analyze_case(case: Dict) -> Dict:
    """
    分析case的数据质量

    Returns:
        {
            'case_id': str,
            'crime': str,
            'subjective_empty': bool,
            'objective_empty': bool,
            'sentencing_empty': bool,
            'quality': 'good'/'partial'/'empty',
            'usable': bool
        }
    """
    hidden = case['hidden_evidence']

    subjective_empty = hidden['subjective']['content'] == '无相关信息' or len(hidden['subjective']['content']) < 10
    objective_empty = hidden['objective']['content'] == '无相关信息' or len(hidden['objective']['content']) < 10
    sentencing_empty = hidden['sentencing']['content'] == '无相关信息' or len(hidden['sentencing']['content']) < 10

    if subjective_empty and objective_empty and sentencing_empty:
        quality = 'empty'
        usable = False
    elif subjective_empty or objective_empty:
        quality = 'partial'
        usable = True  # 部分可用，但质量较差
    else:
        quality = 'good'
        usable = True

    return {
        'case_id': case['case_id'],
        'crime': case['ground_truth']['crime'],
        'subjective_empty': subjective_empty,
        'objective_empty': objective_empty,
        'sentencing_empty': sentencing_empty,
        'quality': quality,
        'usable': usable
    }


def clean_data(train_path: str, test_path: str, output_dir: str, mode: str = 'remove'):
    """
    清理数据

    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        output_dir: 输出目录
        mode: 'remove' 移除无效数据 / 'keep' 只标记不移除
    """
    # 加载数据
    with open(train_path, 'r', encoding='utf-8') as f:
        train_cases = json.load(f)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print("="*60)
    print("RL数据质量分析")
    print("="*60)

    # 分析训练数据
    train_analysis = [analyze_case(c) for c in train_cases]

    good_count = sum(1 for a in train_analysis if a['quality'] == 'good')
    partial_count = sum(1 for a in train_analysis if a['quality'] == 'partial')
    empty_count = sum(1 for a in train_analysis if a['quality'] == 'empty')

    print(f"\n训练数据统计:")
    print(f"  总数: {len(train_cases)}")
    print(f"  高质量(good): {good_count} ({good_count/len(train_cases)*100:.1f}%)")
    print(f"  部分缺失(partial): {partial_count} ({partial_count/len(train_cases)*100:.1f}%)")
    print(f"  完全缺失(empty): {empty_count} ({empty_count/len(train_cases)*100:.1f}%)")

    # 分析测试数据
    test_analysis = [analyze_case(c) for c in test_cases]

    test_good = sum(1 for a in test_analysis if a['quality'] == 'good')
    test_partial = sum(1 for a in test_analysis if a['quality'] == 'partial')
    test_empty = sum(1 for a in test_analysis if a['quality'] == 'empty')

    print(f"\n测试数据统计:")
    print(f"  总数: {len(test_cases)}")
    print(f"  高质量(good): {test_good} ({test_good/len(test_cases)*100:.1f}%)")
    print(f"  部分缺失(partial): {test_partial} ({test_partial/len(test_cases)*100:.1f}%)")
    print(f"  完全缺失(empty): {test_empty} ({test_empty/len(test_cases)*100:.1f}%)")

    # 按罪名统计缺失情况
    print("\n按罪名统计缺失情况:")
    crime_stats = {}
    for a in train_analysis:
        crime = a['crime']
        if crime not in crime_stats:
            crime_stats[crime] = {'good': 0, 'partial': 0, 'empty': 0}
        crime_stats[crime][a['quality']] += 1

    for crime, stats in sorted(crime_stats.items(), key=lambda x: x[1]['empty'], reverse=True):
        total = stats['good'] + stats['partial'] + stats['empty']
        if stats['empty'] > 0:
            print(f"  {crime}: empty={stats['empty']}, partial={stats['partial']}, good={stats['good']} (缺失率:{(stats['empty']+stats['partial'])/total*100:.1f}%)")

    # 处理数据
    if mode == 'remove':
        print("\n" + "="*60)
        print("清理数据...")
        print("="*60)

        # 只保留高质量数据
        clean_train = [c for c, a in zip(train_cases, train_analysis) if a['quality'] == 'good']
        clean_test = [c for c, a in zip(test_cases, test_analysis) if a['quality'] == 'good']

        print(f"清理后训练数据: {len(clean_train)} (移除 {len(train_cases) - len(clean_train)})")
        print(f"清理后测试数据: {len(clean_test)} (移除 {len(test_cases) - len(clean_test)})")

        # 保存清理后的数据
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        clean_train_path = output_path / "train_cases_clean.json"
        clean_test_path = output_path / "test_cases_clean.json"

        with open(clean_train_path, 'w', encoding='utf-8') as f:
            json.dump(clean_train, f, ensure_ascii=False, indent=2)

        with open(clean_test_path, 'w', encoding='utf-8') as f:
            json.dump(clean_test, f, ensure_ascii=False, indent=2)

        print(f"\n清理后数据已保存到:")
        print(f"  {clean_train_path}")
        print(f"  {clean_test_path}")

        # 保存分析报告
        report = {
            'original_train_count': len(train_cases),
            'original_test_count': len(test_cases),
            'clean_train_count': len(clean_train),
            'clean_test_count': len(clean_test),
            'train_quality': {
                'good': good_count,
                'partial': partial_count,
                'empty': empty_count
            },
            'test_quality': {
                'good': test_good,
                'partial': test_partial,
                'empty': test_empty
            },
            'crime_stats': crime_stats
        }

        report_path = output_path / "data_quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"  {report_path}")

    else:
        print("\n[INFO] 只分析，不移除数据")
        print("如需清理，请使用: python scripts/clean_rl_data.py --mode remove")


def main():
    parser = argparse.ArgumentParser(description="清理RL训练数据")
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/rl_env/train_cases.json",
        help="训练数据路径"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/rl_env/test_cases.json",
        help="测试数据路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/rl_env",
        help="输出目录"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="analyze",
        choices=['analyze', 'remove'],
        help="analyze: 只分析; remove: 移除无效数据"
    )

    args = parser.parse_args()

    clean_data(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        mode=args.mode
    )


if __name__ == "__main__":
    main()