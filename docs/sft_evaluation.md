# SFT阶段评估方案

## 1. 评估目标

SFT阶段的核心目标是让模型学会"像法律人一样说话"，建立三个角色（控方、辩方、法官）的基准行为。评估需要验证：

1. **角色区分度**：模型是否能正确扮演不同角色
2. **法律语言规范**：是否使用正确的法律术语和表达方式
3. **推理完整性**：法官角色的CoT推理链是否完整
4. **判决准确性**：预测的罪名、刑期是否与真实判决一致

---

## 2. 评估维度与指标

### 2.1 角色扮演评估

| 维度 | 指标 | 计算方式 | 说明 |
|------|------|---------|------|
| **角色区分度** | Role Accuracy | 角色特征词匹配率 | 检测输出中是否包含角色特有表述 |
| **语言规范性** | Legal Term Coverage | 法律术语使用率 | 专业术语占比 |
| **格式规范性** | Format Compliance | 结构完整性评分 | 是否包含必要部分（指控/辩护/判决） |

**角色特征词定义**：
```python
ROLE_KEYWORDS = {
    'prosecutor': ['指控', '公诉机关', '应当追究', '构成犯罪', '依法惩处'],
    'defender': ['辩护', '从轻处罚', '减轻处罚', '请求', '恳请', '谅解'],
    'judge': ['本院认为', '依照', '判决如下', '被告人犯', '判处']
}
```

### 2.2 判决准确性评估（法官角色）

| 维度 | 指标 | 计算方式 | 说明 |
|------|------|---------|------|
| **罪名准确率** | Crime Accuracy | 精确匹配率 + F1分数 | 罪名是否正确 |
| **刑期预测** | Sentence MAE | 绝对误差（月） | 刑期预测误差 |
| **法条召回** | Law Recall | 正确法条数/总法条数 | 法条引用完整性 |
| **推理完整性** | CoT Completeness | 4步骤完整度 | 是否包含事实认定、法律适用、量刑考量、判决结论 |

### 2.3 辩护意见质量评估（辩方角色）

| 维度 | 指标 | 计算方式 | 说明 |
|------|------|---------|------|
| **情节识别** | Circumstance Recall | 有效情节识别率 | 是否识别出从轻/减轻情节 |
| **论证强度** | Argument Strength | LLM评分(1-5) | 辩护理由的说服力 |

### 2.4 指控意见质量评估（控方角色）

| 维度 | 指标 | 计算方式 | 说明 |
|------|------|---------|------|
| **定罪要素** | Element Coverage | 定罪要素完整度 | 是否包含所有定罪必要要素 |
| **法律依据** | Law Citation | 法条引用正确率 | 法条引用是否恰当 |

---

## 3. 评估方法

### 3.1 自动评估（主要方法）

#### 3.1.1 文本匹配评估

```python
def compute_role_accuracy(output: str, role: str) -> float:
    """
    计算角色区分度：检测输出中是否包含角色特有表述
    """
    keywords = ROLE_KEYWORDS.get(role, [])
    matches = sum(1 for kw in keywords if kw in output)
    return matches / len(keywords)
```

#### 3.1.2 判决准确性评估

```python
def compute_crime_accuracy(predicted: str, ground_truth: str) -> float:
    """
    罪名准确率计算
    """
    # 精确匹配
    if predicted == ground_truth:
        return 1.0
    
    # F1分数（词语重叠）
    pred_words = set(predicted.split())
    true_words = set(ground_truth.split())
    
    if not pred_words or not true_words:
        return 0.0
    
    intersection = pred_words & true_words
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(true_words)
    
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

def compute_sentence_mae(predicted_months: int, true_months: int) -> float:
    """
    刑期预测绝对误差
    """
    return abs(predicted_months - true_months)
```

#### 3.1.3 CoT完整性评估

```python
COT_REQUIRED_STEPS = ['事实认定', '法律适用', '量刑考量', '判决结论']

def compute_cot_completeness(output: str) -> float:
    """
    CoT推理链完整性评估
    """
    # 检查是否包含必要的推理步骤
    score = 0.0
    for step in COT_REQUIRED_STEPS:
        # 检查步骤关键词
        step_keywords = {
            '事实认定': ['经审理查明', '查明', '事实'],
            '法律适用': ['构成', '罪名', '依照', '触犯'],
            '量刑考量': ['从轻', '从重', '减轻', '加重', '量刑'],
            '判决结论': ['判决如下', '判处', '犯']
        }
        for kw in step_keywords[step]:
            if kw in output:
                score += 0.25
                break
    return min(score, 1.0)
```

#### 3.1.4 LLM辅助评估

使用大模型评估推理质量（无法自动量化的指标）：

```python
LLM_EVAL_PROMPT = """
你是一位资深法律专家，请评估以下法律文书的写作质量。

评估对象：{role}角色的输出
评估内容：
{output}

标准答案：
{ground_truth}

请按以下维度评分（1-5分）：
1. 法律语言规范性：是否使用规范的法律术语？
2. 逻辑推理严密性：推理过程是否逻辑清晰？
3. 内容完整性：是否包含必要内容要素？
4. 专业准确性：法律适用是否正确？

请输出JSON格式评分结果：
{"language": X, "logic": X, "completeness": X, "accuracy": X}
"""
```

### 3.2 人工评估（辅助验证）

#### 3.2.1 专家评审方案

| 项目 | 内容 |
|------|------|
| **评审人员** | 2-3名法律专业人士（法官、检察官、律师或法学研究生） |
| **样本数量** | 从测试集随机抽取100条（每角色33条） |
| **评审标准** | 专业性、规范性、逻辑性、准确性（各1-5分） |
| **一致性检验** | 多评审员评分一致性（Kappa系数） |

#### 3.2.2 评审表设计

```
案件编号: {case_id}
角色: {role}
模型输出: {model_output}
真实参考: {ground_truth}

请评分（1-5分）：
□ 专业性：法律术语使用是否准确、规范
□ 规范性：文书格式是否符合司法规范
□ 逻辑性：论证推理是否清晰、连贯
□ 准确性：法律适用结论是否正确

总评：____/20分
备注：_______________
```

---

## 4. Baseline对比设计

### 4.1 对比方案

| 模型 | 说明 | 对比目的 |
|------|------|---------|
| **Base Model (Qwen3-8B)** | 未经过SFT的原始模型 | 验证SFT训练的必要性 |
| **SFT Model** | 经过SFT训练的模型 | 主实验模型 |
| **Random Role** | 随机角色输出 | 验证角色学习效果 |

### 4.2 对比指标

```
对比实验表格模板：

| Model | Role Accuracy | Legal Term | Crime Acc | Sentence MAE | CoT Complete |
|-------|---------------|------------|-----------|--------------|--------------|
| Base Model | - | - | - | - | - |
| SFT Model | X% | X% | X% | X月 | X% |
| Improvement | +X% | +X% | +X% | -X月 | +X% |
```

---

## 5. 数据划分

### 5.1 训练/验证/测试划分

```
总数据量：3570条
  - 控方：419条
  - 辋方：227条
  - 法官：2924条

建议划分比例：
  - 训练集：80% (~2856条)
  - 验证集：10% (~357条)  - 用于训练过程监控
  - 测试集：10% (~357条)  - 用于最终评估
```

### 5.2 划分策略

```python
def split_dataset(data: List, ratio=(0.8, 0.1, 0.1)):
    """
    按角色分层划分数据集，确保各角色在各集合中比例一致
    """
    from collections import defaultdict
    import random
    
    # 按角色分组
    role_data = defaultdict(list)
    for item in data:
        role_data[item['meta']['role']].append(item)
    
    train, val, test = [], [], []
    
    for role, items in role_data.items():
        random.shuffle(items)
        n = len(items)
        train.extend(items[:int(n * ratio[0])])
        val.extend(items[int(n * ratio[0]):int(n * (ratio[0] + ratio[1]))])
        test.extend(items[int(n * (ratio[0] + ratio[1])):])
    
    return train, val, test
```

---

## 6. 评估代码实现

### 6.1 评估脚本结构

```
src/evaluation/
├── sft_eval.py          # SFT评估主脚本
├── metrics/
│   ├── role_metrics.py   # 角色评估指标
│   ├── judge_metrics.py  # 法官判决评估
│   ├── text_metrics.py   # 文本相似度指标
│   └── llm_eval.py       # LLM辅助评估
└── results/
    ├── eval_report.json  # 评估结果报告
    └── comparison.json   # baseline对比结果
```

### 6.2 主评估脚本

```python
"""
SFT评估主脚本
运行方式：python src/evaluation/sft_eval.py --model models/sft_checkpoint --test_data data/processed/test.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from metrics.role_metrics import compute_role_accuracy, compute_legal_term_coverage
from metrics.judge_metrics import compute_crime_accuracy, compute_sentence_mae, compute_cot_completeness
from metrics.text_metrics import compute_bleu, compute_rouge


class SFTEvaluator:
    """
    SFT模型评估器
    """
    
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen3-8B"):
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
    def generate(self, instruction: str, input_text: str, max_length: int = 512) -> str:
        """
        生成模型输出
        """
        prompt = f"{instruction}\n\n{input_text}\n\n请输出："
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        评估单个样本
        """
        role = sample['meta']['role']
        generated = self.generate(sample['instruction'], sample['input'])
        ground_truth = sample['output']
        
        metrics = {
            'role': role,
            'case_id': sample['meta']['case_id'],
            'generated': generated,
            'ground_truth': ground_truth,
        }
        
        # 通用指标
        metrics['role_accuracy'] = compute_role_accuracy(generated, role)
        metrics['legal_term_coverage'] = compute_legal_term_coverage(generated)
        metrics['bleu'] = compute_bleu(generated, ground_truth)
        metrics['rouge_l'] = compute_rouge(generated, ground_truth)
        
        # 角色特定指标
        if role == 'judge':
            metrics['cot_completeness'] = compute_cot_completeness(generated)
            # 如果有结构化判决信息，计算准确性
            # metrics['crime_accuracy'] = compute_crime_accuracy(...)
            # metrics['sentence_mae'] = compute_sentence_mae(...)
        
        return metrics
    
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """
        评估整个测试集
        """
        results = []
        
        for sample in test_data:
            metrics = self.evaluate_sample(sample)
            results.append(metrics)
        
        # 统计汇总
        summary = self._compute_summary(results)
        
        return {'samples': results, 'summary': summary}
    
    def _compute_summary(self, results: List[Dict]) -> Dict:
        """
        计算汇总统计
        """
        role_results = defaultdict(list)
        
        for r in results:
            role_results[r['role']].append(r)
        
        summary = {}
        for role, role_data in role_results.items():
            n = len(role_data)
            summary[role] = {
                'count': n,
                'role_accuracy': sum(d['role_accuracy'] for d in role_data) / n,
                'legal_term_coverage': sum(d['legal_term_coverage'] for d in role_data) / n,
                'bleu': sum(d['bleu'] for d in role_data) / n,
                'rouge_l': sum(d['rouge_l'] for d in role_data) / n,
            }
            
            if role == 'judge':
                cot_scores = [d.get('cot_completeness', 0) for d in role_data]
                summary[role]['cot_completeness'] = sum(cot_scores) / n
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="SFT Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--test_data", type=str, default="data/processed/test.json")
    parser.add_argument("--output", type=str, default="results/eval_results/sft_eval.json")
    args = parser.parse_args()
    
    # 加载测试数据
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 评估
    evaluator = SFTEvaluator(args.model)
    results = evaluator.evaluate_dataset(test_data)
    
    # 输出结果
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印摘要
    print("\n" + "="*60)
    print("SFT Evaluation Summary")
    print("="*60)
    for role, metrics in results['summary'].items():
        print(f"\n{role} role:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

---

## 7. 论文呈现建议

### 7.1 实验部分结构

```
4. 实验
  4.1 实验设置
      - 数据集描述（来源、规模、划分）
      - 模型配置（基座模型、LoRA参数）
      - 训练参数（学习率、epoch等）
  
  4.2 评估指标
      - 表格列出所有评估指标定义
  
  4.3 SFT阶段结果
      - 各角色评估结果表格
      - 与baseline对比表格
      - 典型案例分析
  
  4.4 讨论
      - 模型学到的能力分析
      - 不足之处分析
```

### 7.2 结果表格模板

**表1：SFT阶段各角色评估结果**

| 角色 | Role Acc | Legal Term | BLEU | ROUGE-L | CoT Complete |
|------|----------|------------|------|---------|--------------|
| 控方 | 0.85 | 0.72 | 0.34 | 0.56 | - |
| 辋方 | 0.81 | 0.68 | 0.31 | 0.52 | - |
| 法官 | 0.89 | 0.76 | 0.38 | 0.61 | 0.82 |

**表2：与Baseline对比**

| Model | Avg Role Acc | Avg Legal Term | Avg BLEU | Avg ROUGE-L |
|-------|--------------|----------------|----------|-------------|
| Qwen3-8B (Base) | 0.23 | 0.15 | 0.08 | 0.21 |
| SFT Model | 0.85 | 0.72 | 0.34 | 0.56 |
| Improvement | +62% | +57% | +26% | +35% |

---

## 8. 实施计划

| 步骤 | 任务 | 预计时间 |
|------|------|---------|
| 1 | 实现评估指标代码 | 1天 |
| 2 | 划分训练/验证/测试数据集 | 0.5天 |
| 3 | 运行SFT训练 | 2天 |
| 4 | 运行自动评估 | 0.5天 |
| 5 | 进行人工评审（如需要） | 1天 |
| 6 | 撰写评估结果部分 | 1天 |

---

*文档创建时间: 2026-05-01*
*作者: Legal Agenttic RL Team*