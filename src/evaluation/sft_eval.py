"""
SFT综合评估脚本
运行方式: python src/evaluation/sft_eval.py --config configs/eval_config.yaml

功能:
1. 加载SFT训练后的模型
2. 对测试集生成预测
3. 计算各角色评估指标
4. 输出评估报告
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.sft_metrics import (
    compute_role_accuracy,
    compute_legal_term_coverage,
    compute_cot_completeness,
    compute_format_compliance,
    compute_bleu,
    compute_rouge_l,
    compute_all_text_metrics,
    compute_judge_accuracy,
)


class SFTEvaluator:
    """
    SFT模型评估器

    Usage:
        evaluator = SFTEvaluator(config)
        results = evaluator.evaluate(test_data)
        evaluator.save_results(results, output_path)
    """

    def __init__(self, config: Dict):
        """
        初始化评估器

        Args:
            config: 评估配置字典
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/sft_checkpoint')
        self.base_model = config.get('base_model', 'Qwen/Qwen3-8B')

        # 是否加载模型进行生成评估
        self.use_model = config.get('use_model', False)

        if self.use_model:
            self._load_model()

    def _load_model(self):
        """加载模型（如果需要生成评估）"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\n{'='*60}")
        print(f"Loading model from: {self.model_path}")
        print(f"{'='*60}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )

        print(f"Model loaded successfully")

    def generate(self, instruction: str, input_text: str, max_length: int = 512) -> str:
        """
        使用模型生成输出

        Args:
            instruction: 任务指令
            input_text: 输入内容
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        if not self.use_model:
            raise ValueError("Model not loaded. Set use_model=True in config")

        prompt = f"{instruction}\n\n{input_text}\n\n请输出："

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取输出部分（去除prompt）
        if "请输出：" in generated:
            generated = generated.split("请输出：")[-1].strip()

        return generated

    def evaluate_sample(
        self,
        sample: Dict,
        generated_output: Optional[str] = None
    ) -> Dict:
        """
        评估单个样本

        Args:
            sample: 测试样本
            generated_output: 模型生成的输出（如果已生成）

        Returns:
            评估结果字典
        """
        role = sample.get('meta', {}).get('role', 'unknown')
        ground_truth = sample.get('output', '')

        # 获取生成输出
        if generated_output is None and self.use_model:
            generated_output = self.generate(sample['instruction'], sample['input'])

        # 如果没有生成输出，使用ground_truth计算基准指标
        output_to_eval = generated_output or ground_truth

        results = {
            'case_id': sample.get('meta', {}).get('case_id', ''),
            'role': role,
            'generated': generated_output,
            'ground_truth': ground_truth,
        }

        # 计算指标
        results['metrics'] = {
            # 角色指标
            'role_accuracy': compute_role_accuracy(output_to_eval, role),
            'legal_term_coverage': compute_legal_term_coverage(output_to_eval),
            'format_compliance': compute_format_compliance(output_to_eval, role),

            # 文本相似度指标
            'text_metrics': compute_all_text_metrics(ground_truth, output_to_eval)
        }

        # 法官角色额外指标
        if role == 'judge':
            results['metrics']['cot_completeness'] = compute_cot_completeness(output_to_eval)

            # 判决准确性（如果有ground_truth的结构化信息）
            if 'crime' in sample.get('meta', {}):
                judge_accuracy = compute_judge_accuracy(output_to_eval, sample['meta'])
                results['metrics']['judge_accuracy'] = judge_accuracy

        return results

    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """
        评估整个测试集

        Args:
            test_data: 测试数据列表

        Returns:
            完整评估结果（包含样本结果和汇总统计）
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {len(test_data)} samples")
        print(f"{'='*60}")

        results = []

        for sample in tqdm(test_data, desc="Evaluating"):
            result = self.evaluate_sample(sample)
            results.append(result)

        # 计算汇总统计
        summary = self._compute_summary(results)

        return {
            'samples': results,
            'summary': summary,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """
        计算汇总统计

        Args:
            results: 样本评估结果列表

        Returns:
            按角色汇总的统计结果
        """
        role_results = defaultdict(list)

        for r in results:
            role = r.get('role', 'unknown')
            role_results[role].append(r)

        summary = {}

        for role, samples in role_results.items():
            n = len(samples)
            metrics_list = [s['metrics'] for s in samples]

            role_summary = {
                'count': n,
                'role_accuracy': sum(m['role_accuracy'] for m in metrics_list) / n,
                'legal_term_coverage': sum(m['legal_term_coverage'] for m in metrics_list) / n,
                'format_compliance': sum(m['format_compliance'] for m in metrics_list) / n,
                'bleu': sum(m['text_metrics']['bleu'] for m in metrics_list) / n,
                'rouge_l': sum(m['text_metrics']['rouge_l'] for m in metrics_list) / n,
            }

            # 法官角色的额外指标
            if role == 'judge':
                cot_scores = [m.get('cot_completeness', 0) for m in metrics_list]
                role_summary['cot_completeness'] = sum(cot_scores) / n

            summary[role] = role_summary

        # 计算总体平均
        all_samples = []
        for role_data in role_results.values():
            all_samples.extend(role_data)

        total_n = len(all_samples)
        if total_n > 0:
            all_metrics = [s['metrics'] for s in all_samples]

            summary['overall'] = {
                'count': total_n,
                'avg_role_accuracy': sum(m['role_accuracy'] for m in all_metrics) / total_n,
                'avg_legal_term_coverage': sum(m['legal_term_coverage'] for m in all_metrics) / total_n,
                'avg_bleu': sum(m['text_metrics']['bleu'] for m in all_metrics) / total_n,
                'avg_rouge_l': sum(m['text_metrics']['rouge_l'] for m in all_metrics) / total_n,
            }

        return summary

    def save_results(self, results: Dict, output_path: str):
        """
        保存评估结果

        Args:
            results: 评估结果字典
            output_path: 输出文件路径
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to: {output_path}")

    def print_summary(self, results: Dict):
        """
        打印评估摘要

        Args:
            results: 评估结果字典
        """
        summary = results.get('summary', {})

        print(f"\n{'='*60}")
        print("SFT Evaluation Summary")
        print(f"{'='*60}")

        # 各角色结果
        for role in ['prosecutor', 'defender', 'judge']:
            if role in summary:
                print(f"\n[{role}] Role ({summary[role]['count']} samples)")
                print(f"  Role Accuracy:     {summary[role]['role_accuracy']:.4f}")
                print(f"  Legal Term:        {summary[role]['legal_term_coverage']:.4f}")
                print(f"  Format Compliance: {summary[role]['format_compliance']:.4f}")
                print(f"  BLEU:              {summary[role]['bleu']:.4f}")
                print(f"  ROUGE-L:           {summary[role]['rouge_l']:.4f}")

                if role == 'judge':
                    print(f"  CoT Completeness:  {summary[role].get('cot_completeness', 'N/A')}")

        # 总体结果
        if 'overall' in summary:
            print(f"\n[Overall] ({summary['overall']['count']} samples)")
            print(f"  Avg Role Accuracy: {summary['overall']['avg_role_accuracy']:.4f}")
            print(f"  Avg Legal Term:    {summary['overall']['avg_legal_term_coverage']:.4f}")
            print(f"  Avg BLEU:          {summary['overall']['avg_bleu']:.4f}")
            print(f"  Avg ROUGE-L:       {summary['overall']['avg_rouge_l']:.4f}")

        print(f"\n{'='*60}")


def load_test_data(data_path: str) -> List[Dict]:
    """
    加载测试数据

    Args:
        data_path: 测试数据路径

    Returns:
        测试数据列表
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_test_data(all_data: List[Dict], test_ratio: float = 0.1) -> List[Dict]:
    """
    从全部数据中划分测试集（如果还没有单独的测试集）

    Args:
        all_data: 全部数据
        test_ratio: 测试集比例

    Returns:
        测试数据列表
    """
    import random

    random.seed(42)
    random.shuffle(all_data)

    test_size = int(len(all_data) * test_ratio)
    return all_data[:test_size]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SFT Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Evaluation config file"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed/all_roles.json",
        help="Test data path"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/sft_checkpoint",
        help="SFT model checkpoint path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eval_results/sft_eval.json",
        help="Output result path"
    )
    parser.add_argument(
        "--use_model",
        action="store_true",
        help="Whether to load model for generation evaluation"
    )

    args = parser.parse_args()

    # 构建配置
    config = {
        'model_path': args.model_path,
        'use_model': args.use_model
    }

    # 加载yaml配置（如果存在）
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            config.update(yaml_config)

    # 加载测试数据
    test_data = load_test_data(args.test_data)

    # 如果没有单独的测试集，划分一部分作为测试
    if 'test' not in args.test_data:
        test_data = split_test_data(test_data, 0.1)

    # 评估
    evaluator = SFTEvaluator(config)
    results = evaluator.evaluate_dataset(test_data)

    # 保存和打印结果
    evaluator.save_results(results, args.output)
    evaluator.print_summary(results)


if __name__ == "__main__":
    main()