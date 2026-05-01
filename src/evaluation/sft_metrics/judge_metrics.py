"""
法官判决评估指标模块
专门用于评估法官角色的判决准确性
"""

import re
from typing import Dict, List, Optional, Tuple


def extract_crime(text: str) -> Optional[str]:
    """
    从判决文本中提取罪名

    Args:
        text: 判决文本

    Returns:
        提取的罪名，如"故意伤害罪"
    """
    patterns = [
        r'被告人犯(.+罪)',
        r'构成(.+罪)',
        r'判处(.+罪)',
        r'以(.+罪)追究',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return None


def extract_sentence(text: str) -> Optional[Tuple[int, str]]:
    """
    从判决文本中提取刑期

    Args:
        text: 判决文本

    Returns:
        (刑期月数, 原始文本)，如(36, "有期徒刑三年")
    """
    patterns = [
        r'判处有期徒刑(.+年)',
        r'判处有期徒刑(.+个月)',
        r'判处拘役(.+个月)',
        r'判处有期徒刑(.+)年',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            original = match.group(0)
            value = match.group(1).strip()

            # 转换为月数
            months = 0
            if '年' in value:
                years = re.search(r'(\d+)', value)
                if years:
                    months = int(years.group(1)) * 12
            elif '个月' in value or '月' in value:
                months_match = re.search(r'(\d+)', value)
                if months_match:
                    months = int(months_match.group(1))

            return (months, original)

    return None


def extract_laws(text: str) -> List[str]:
    """
    从判决文本中提取引用的法条

    Args:
        text: 判决文本

    Returns:
        法条列表
    """
    laws = []

    patterns = [
        r'《中华人民共和国刑法》第(\d+)条',
        r'刑法第(\d+)条',
        r'《中华人民共和国刑事诉讼法》第(\d+)条',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            laws.append(match.group(0))

    return laws


def compute_crime_accuracy(predicted: str, ground_truth: str) -> float:
    """
    计算罪名准确率

    Args:
        predicted: 预测的罪名
        ground_truth: 真实罪名

    Returns:
        准确率 (0-1)
    """
    # 精确匹配
    if predicted == ground_truth:
        return 1.0

    # 提取罪名进行比较
    pred_crime = extract_crime(predicted) or predicted
    true_crime = extract_crime(ground_truth) or ground_truth

    if pred_crime == true_crime:
        return 1.0

    # F1分数（词语重叠）
    pred_words = set(pred_crime.split())
    true_words = set(true_crime.split())

    if not pred_words or not true_words:
        return 0.0

    intersection = pred_words & true_words
    precision = len(intersection) / len(pred_words)
    recall = len(intersection) / len(true_words)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_sentence_mae(predicted_months: int, true_months: int) -> float:
    """
    计算刑期预测绝对误差

    Args:
        predicted_months: 预测刑期（月）
        true_months: 真实刑期（月）

    Returns:
        绝对误差（月）
    """
    return abs(predicted_months - true_months)


def compute_sentence_relative_error(predicted_months: int, true_months: int) -> float:
    """
    计算刑期预测相对误差

    Args:
        predicted_months: 预测刑期（月）
        true_months: 真实刑期（月）

    Returns:
        相对误差 (0-1)
    """
    if true_months == 0:
        return 0.0 if predicted_months == 0 else 1.0

    return abs(predicted_months - true_months) / true_months


def compute_law_recall(predicted_laws: List[str], true_laws: List[str]) -> float:
    """
    计算法条召回率

    Args:
        predicted_laws: 预测引用的法条
        true_laws: 应引用的法条

    Returns:
        法条召回率 (0-1)
    """
    if not true_laws:
        return 1.0 if not predicted_laws else 0.0

    # 提取法条编号进行比较
    pred_numbers = set()
    for law in predicted_laws:
        match = re.search(r'第(\d+)条', law)
        if match:
            pred_numbers.add(int(match.group(1)))

    true_numbers = set()
    for law in true_laws:
        if isinstance(law, int):
            true_numbers.add(law)
        else:
            match = re.search(r'第(\d+)条', law)
            if match:
                true_numbers.add(int(match.group(1)))

    if not true_numbers:
        return 1.0

    intersection = pred_numbers & true_numbers
    return len(intersection) / len(true_numbers)


def compute_judge_accuracy(
    predicted_output: str,
    ground_truth: Dict
) -> Dict:
    """
    计算法官判决的综合准确性

    Args:
        predicted_output: 模型预测的判决输出
        ground_truth: 包含crime, sentence_months, laws的真实标签

    Returns:
        包含各项准确性指标的字典
    """
    # 提取预测结果
    pred_crime = extract_crime(predicted_output)
    pred_sentence = extract_sentence(predicted_output)
    pred_laws = extract_laws(predicted_output)

    # 真实值
    true_crime = ground_truth.get('crime', '')
    true_months = ground_truth.get('sentence_months', 0)
    true_laws = ground_truth.get('laws', [])

    # 计算指标
    results = {
        'crime_accuracy': compute_crime_accuracy(pred_crime or '', true_crime) if pred_crime else 0.0,
        'law_recall': compute_law_recall(pred_laws, true_laws),
    }

    # 刑期指标
    if pred_sentence:
        pred_months = pred_sentence[0]
        results['sentence_mae'] = compute_sentence_mae(pred_months, true_months)
        results['sentence_relative_error'] = compute_sentence_relative_error(pred_months, true_months)
    else:
        results['sentence_mae'] = None
        results['sentence_relative_error'] = None

    # 加权综合评分
    weights = {
        'crime_accuracy': 0.4,
        'law_recall': 0.3,
        'sentence_relative_error': 0.3
    }

    total_score = 0.0
    for key, weight in weights.items():
        value = results.get(key)
        if value is not None:
            # 对于relative_error，越小越好，需要转换
            if key == 'sentence_relative_error':
                total_score += weight * (1 - min(value, 1.0))
            else:
                total_score += weight * value

    results['overall_accuracy'] = total_score

    return results


def parse_sentence_to_months(sentence_text: str) -> int:
    """
    将刑期文本转换为月数

    Args:
        sentence_text: 刑期文本，如"有期徒刑三年"、"拘役六个月"

    Returns:
        刑期月数
    """
    if not sentence_text:
        return 0

    # 处理年份
    year_match = re.search(r'(\d+)年', sentence_text)
    if year_match:
        return int(year_match.group(1)) * 12

    # 处理月份
    month_match = re.search(r'(\d+)个?月', sentence_text)
    if month_match:
        return int(month_match.group(1))

    # 处理中文数字
    chinese_nums = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

    for cn, num in chinese_nums.items():
        if cn + '年' in sentence_text:
            return num * 12
        if cn + '个' in sentence_text or cn + '月' in sentence_text:
            return num

    return 0