"""
文本相似度评估指标模块
提供BLEU、ROUGE等文本评估指标
"""

import math
from typing import List, Dict
from collections import Counter
import re


def tokenize(text: str) -> List[str]:
    """
    中文文本分词（简单实现，按字符和标点分割）

    Args:
        text: 输入文本

    Returns:
        词列表
    """
    # 按标点分割
    segments = re.split(r'[，。；：！？、\s]+', text)

    # 按字符分割（中文）
    tokens = []
    for segment in segments:
        if segment:
            # 每2-4个字作为一个token（简单处理）
            for i in range(0, len(segment), 2):
                token = segment[i:i+4]
                if token:
                    tokens.append(token)

    return tokens


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    计算BLEU分数

    Args:
        reference: 参考文本
        hypothesis: 生成的假设文本
        max_n: 最大n-gram阶数

    Returns:
        BLEU分数 (0-1)
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # 计算各阶n-gram精度
    precisions = []

    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])

        if not hyp_ngrams:
            precisions.append(0.0)
            continue

        # 计算 clipped count
        clipped_count = 0
        for ngram, count in hyp_ngrams.items():
            clipped_count += min(count, ref_ngrams.get(ngram, 0))

        precision = clipped_count / sum(hyp_ngrams.values())
        precisions.append(precision)

    # 计算几何平均
    if any(p == 0 for p in precisions):
        return 0.0

    avg_precision = sum(precisions) / len(precisions)

    # BP (Brevity Penalty)
    ref_len = len(ref_tokens)
    hyp_len = len(hyp_tokens)

    if hyp_len >= ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        # BP = exp(1 - ref_len/hyp_len)
        bp = math.exp(1 - ref_len / hyp_len)

    return bp * avg_precision


def compute_rouge_n(reference: str, hypothesis: str, n: int = 1) -> float:
    """
    计算ROUGE-N分数

    Args:
        reference: 参考文本
        hypothesis: 生成的假设文本
        n: n-gram阶数

    Returns:
        ROUGE-N分数 (0-1)
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens:
        return 0.0

    # 提取n-grams
    ref_ngrams = Counter([tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens)-n+1)])
    hyp_ngrams = Counter([tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens)-n+1)])

    # 计算重叠
    overlap = 0
    for ngram, count in hyp_ngrams.items():
        overlap += min(count, ref_ngrams.get(ngram, 0))

    # Recall
    total_ref_ngrams = sum(ref_ngrams.values())
    if total_ref_ngrams == 0:
        return 0.0

    return overlap / total_ref_ngrams


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """
    计算ROUGE-L分数（基于最长公共子序列）

    Args:
        reference: 参考文本
        hypothesis: 生成的假设文本

    Returns:
        ROUGE-L分数 (0-1)
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # 计算LCS长度
    lcs_len = _lcs_length(ref_tokens, hyp_tokens)

    # Recall和Precision
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
    precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0.0

    # F-score
    if recall + precision == 0:
        return 0.0

    return 2 * recall * precision / (recall + precision)


def _lcs_length(seq1: List, seq2: List) -> int:
    """
    计算最长公共子序列长度

    Args:
        seq1: 序列1
        seq2: 序列2

    Returns:
        LCS长度
    """
    m, n = len(seq1), len(seq2)

    # DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]


def compute_all_text_metrics(reference: str, hypothesis: str) -> Dict:
    """
    计算所有文本相似度指标

    Args:
        reference: 参考文本
        hypothesis: 生成的假设文本

    Returns:
        包含所有指标的字典
    """
    return {
        'bleu': compute_bleu(reference, hypothesis),
        'rouge_1': compute_rouge_n(reference, hypothesis, 1),
        'rouge_2': compute_rouge_n(reference, hypothesis, 2),
        'rouge_l': compute_rouge_l(reference, hypothesis),
    }


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """
    计算精确匹配率

    Args:
        reference: 参考文本
        hypothesis: 生成的假设文本

    Returns:
        精确匹配率 (0-1)
    """
    if not reference:
        return 0.0

    # 清理文本
    ref_clean = re.sub(r'[，。；：！？、\s]+', '', reference)
    hyp_clean = re.sub(r'[，。；：！？、\s]+', '', hypothesis)

    if ref_clean == hyp_clean:
        return 1.0

    # 字符级精确匹配
    ref_chars = set(ref_clean)
    hyp_chars = set(hyp_clean)

    if not ref_chars:
        return 0.0

    overlap = len(ref_chars & hyp_chars)
    return overlap / len(ref_chars)