"""
SFT评估指标模块
提供各角色评估指标的实现
"""

# 角色特征关键词
ROLE_KEYWORDS = {
    'prosecutor': [
        '指控', '公诉机关', '应当追究', '构成犯罪', '依法惩处',
        '起诉', '指控罪名', '公诉', '检察院', '追究刑事责任'
    ],
    'defender': [
        '辩护', '从轻处罚', '减轻处罚', '请求', '恳请', '谅解',
        '辩护意见', '辩护人', '酌情', '从宽处罚', '初犯', '偶犯'
    ],
    'judge': [
        '本院认为', '依照', '判决如下', '被告人犯', '判处',
        '经审理查明', '事实清楚', '证据确实', '罪名成立', '量刑'
    ]
}

# 法律术语列表
LEGAL_TERMS = [
    '被告人', '被害人', '公诉机关', '检察院', '法院',
    '本院认为', '依照', '判决如下', '构成犯罪', '罪名',
    '有期徒刑', '拘役', '罚金', '缓刑', '自首',
    '累犯', '坦白', '认罪认罚', '从轻处罚', '从重处罚',
    '故意', '过失', '犯罪事实', '证据', '鉴定',
    '刑事责任', '刑法', '刑事诉讼法', '法定', '酌定'
]

# CoT必要步骤关键词
COT_STEP_KEYWORDS = {
    '事实认定': ['经审理查明', '查明', '事实', '经审理', '认定事实'],
    '法律适用': ['构成', '罪名', '依照', '触犯', '法律适用', '刑法'],
    '量刑考量': ['从轻', '从重', '减轻', '加重', '量刑', '处罚'],
    '判决结论': ['判决如下', '判处', '犯', '判决', '结论']
}


def compute_role_accuracy(output: str, role: str) -> float:
    """
    计算角色区分度：检测输出中是否包含角色特有表述

    Args:
        output: 模型生成的输出文本
        role: 目标角色 (prosecutor, defender, judge)

    Returns:
        角色特征匹配率 (0-1)
    """
    keywords = ROLE_KEYWORDS.get(role, [])
    if not keywords:
        return 0.0

    matches = sum(1 for kw in keywords if kw in output)
    return matches / len(keywords)


def compute_legal_term_coverage(output: str) -> float:
    """
    计算法律术语覆盖率

    Args:
        output: 模型生成的输出文本

    Returns:
        法律术语使用比例 (0-1)
    """
    if not output:
        return 0.0

    used_terms = sum(1 for term in LEGAL_TERMS if term in output)
    return used_terms / len(LEGAL_TERMS)


def compute_cot_completeness(output: str) -> float:
    """
    计算CoT推理链完整性（法官角色专用）

    Args:
        output: 模型生成的输出文本

    Returns:
        推理步骤完整度 (0-1)，满分需包含4个必要步骤
    """
    score = 0.0

    for step, keywords in COT_STEP_KEYWORDS.items():
        # 每个步骤贡献0.25分
        for kw in keywords:
            if kw in output:
                score += 0.25
                break

    return min(score, 1.0)


def compute_format_compliance(output: str, role: str) -> float:
    """
    计算格式规范性评分

    Args:
        output: 模型生成的输出文本
        role: 目标角色

    Returns:
        格式规范性评分 (0-1)
    """
    required_sections = {
        'prosecutor': ['指控', '罪名', '法律依据'],
        'defender': ['辩护', '请求', '理由'],
        'judge': ['本院认为', '依照', '判决']
    }

    sections = required_sections.get(role, [])
    if not sections:
        return 0.0

    matches = sum(1 for s in sections if s in output)
    return matches / len(sections)


def compute_all_role_metrics(output: str, role: str) -> dict:
    """
    计算所有角色相关指标

    Args:
        output: 模型生成的输出文本
        role: 目标角色

    Returns:
        包含所有指标的字典
    """
    return {
        'role_accuracy': compute_role_accuracy(output, role),
        'legal_term_coverage': compute_legal_term_coverage(output),
        'format_compliance': compute_format_compliance(output, role),
        'cot_completeness': compute_cot_completeness(output) if role == 'judge' else None
    }


def get_role_keywords(role: str) -> list:
    """获取指定角色的特征关键词"""
    return ROLE_KEYWORDS.get(role, [])