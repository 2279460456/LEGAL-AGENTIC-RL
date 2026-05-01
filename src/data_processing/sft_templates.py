"""
SFT数据模板定义
针对SimuCourt和judge-data两个数据源构建统一的instruction/input/output格式

核心原则：
1. 充分利用数据中所有有效信息
2. 一个案件生成三条SFT数据（控方、辩方、法官）
3. 根据数据源特点适配不同的input结构
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


# =============================================================================
# 角色Instruction模板
# =============================================================================

INSTRUCTION_TEMPLATES = {
    # 控方（检察官）角色
    "prosecutor": """你是一位专业的检察官，负责对刑事案件提出指控。
请根据提供的案件信息，提出正式的指控意见。

你的任务：
1. 分析案件事实，确定涉嫌罪名
2. 明确指控的法律依据
3. 列出关键定罪要素
4. 提出量刑建议理由

输出要求：
- 使用规范的法律语言
- 明确罪名认定
- 引用相关法律条文
- 论述应简洁有力""",

    # 辩方（辩护律师）角色
    "defender": """你是一位专业的辩护律师，负责为被告人进行辩护。
请根据提供的案件信息，提出辩护意见。

你的任务：
1. 分析案件事实中的有利情节
2. 寻找法定和酌定从轻、减轻情节
3. 指出证据或程序可能存在的问题
4. 提出从宽处罚的具体请求

输出要求：
- 使用规范的法律语言
- 论述有理有据
- 明确辩护请求
- 关注量刑情节""",

    # 法官角色
    "judge": """你是一位资深法官，负责审理刑事案件并作出判决。
请根据提供的案件事实和控辩双方意见，作出公正判决。

你的任务：
1. 事实认定：综合控辩意见，确认案件事实
2. 法律适用：确定罪名，引用法律依据
3. 量刑考量：分析加重和减轻情节
4. 判决结论：明确罪名、刑期、罚金

输出要求：
- 使用规范的法律语言
- 推理逻辑严密
- 明确判决结果
- 引用具体法条"""
}


# =============================================================================
# Input模板构建函数
# =============================================================================

def build_prosecutor_input_simucourt1(case: Dict) -> str:
    """
    从SimuCourt(1)构建控方input
    利用字段：案件名、类别、案由、法院、被告、被告基本情况、基本案情
    """
    parts = []

    # 案件基本信息
    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    # 案件类型
    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    # 法院信息
    if case.get('法院'):
        court_info = f"【审理法院】{case['法院']}"
        if case.get('法院层级'):
            court_info += f"（{case['法院层级']}法院）"
        parts.append(court_info)

    # 当事人信息
    if case.get('被告'):
        parts.append(f"【被告人】{case['被告']}")

    # 被告基本情况（前科劣迹，影响量刑考量）
    if case.get('被告基本情况', '无') != '无':
        parts.append(f"【被告人基本情况】{case['被告基本情况']}")

    # 核心案情
    if case.get('基本案情'):
        parts.append(f"【案情事实】{case['基本案情']}")

    return "\n".join(parts)


def build_defender_input_simucourt1(case: Dict) -> str:
    """
    从SimuCourt(1)构建辩方input
    利用字段：案件名、类别、案由、被告、被告基本情况、基本案情、公诉机关指控（作为控方观点）
    """
    parts = []

    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    if case.get('被告'):
        parts.append(f"【被告人】{case['被告']}（你正在为其辩护）")

    # 被告基本情况（重要辩护背景）
    if case.get('被告基本情况', '无') != '无':
        parts.append(f"【被告人基本情况】{case['被告基本情况']}")

    if case.get('基本案情'):
        parts.append(f"【案情事实】{case['基本案情']}")

    # 控方指控观点（辩方需要回应）
    if case.get('原告诉请判令（公诉机关指控）'):
        parts.append(f"【公诉机关指控】{case['原告诉请判令（公诉机关指控）']}")

    return "\n".join(parts)


def build_judge_input_simucourt1(case: Dict) -> str:
    """
    从SimuCourt(1)构建法官input
    利用字段：案件名、类别、案由、法院、被告、被告基本情况、基本案情、控方指控、辩方意见、被告陈述
    """
    parts = []

    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    if case.get('法院'):
        parts.append(f"【审理法院】{case['法院']}")

    if case.get('被告'):
        parts.append(f"【被告人】{case['被告']}")

    if case.get('被告基本情况', '无') != '无':
        parts.append(f"【被告人基本情况】{case['被告基本情况']}")

    if case.get('基本案情'):
        parts.append(f"【案情事实】{case['基本案情']}")

    # 控方意见
    if case.get('原告诉请判令（公诉机关指控）'):
        parts.append(f"\n【控方指控意见】\n{case['原告诉请判令（公诉机关指控）']}")

    # 辩方意见（如果有）
    if case.get('被告代理人辩护', '无') != '无':
        parts.append(f"\n【辩方辩护意见】\n{case['被告代理人辩护']}")

    # 被告陈述（如果有）
    if case.get('被告陈述', '无') != '无':
        parts.append(f"\n【被告陈述】\n{case['被告陈述']}")

    return "\n".join(parts)


def build_prosecutor_output_simucourt1(case: Dict) -> str:
    """
    从SimuCourt(1)构建控方output
    直接使用：原告诉请判令（公诉机关指控）
    补充：引用法律条文（如果有）
    """
    output = case.get('原告诉请判令（公诉机关指控）', '')

    # 补充法律条文
    laws = []
    for i in range(1, 12):
        law = case.get(f'引用法律条文{i}', '无')
        if law != '无':
            laws.append(law)

    if laws:
        output += f"\n\n【法律依据】\n" + "\n".join(laws)

    return output


def build_defender_output_simucourt1(case: Dict) -> Optional[str]:
    """
    从SimuCourt(1)构建辩方output
    使用：被告代理人辩护、被告陈述
    如果都没有，返回None（不生成该样本）
    """
    output_parts = []

    # 主要辩护意见
    if case.get('被告代理人辩护', '无') != '无':
        output_parts.append(case['被告代理人辩护'])

    # 被告自己陈述
    if case.get('被告陈述', '无') != '无':
        if output_parts:
            output_parts.append(f"\n【被告人陈述】{case['被告陈述']}")
        else:
            output_parts.append(case['被告陈述'])

    if not output_parts:
        return None

    return "\n".join(output_parts)


def build_judge_output_simucourt1(case: Dict) -> str:
    """
    从SimuCourt(1)构建法官output
    使用：法院意见、引用法律条文、刑事_罪名、刑事_刑期、刑事_罚金
    """
    parts = []

    # 法院意见（推理部分）
    if case.get('法院意见'):
        parts.append(f"【本院认为】\n{case['法院意见']}")

    # 法律条文
    laws = []
    for i in range(1, 12):
        law = case.get(f'引用法律条文{i}', '无')
        if law != '无':
            laws.append(law)

    if laws:
        parts.append(f"\n【法律依据】\n依照：" + "、".join(laws))

    # 判决结果
    judgment_parts = []
    if case.get('刑事_罪名'):
        judgment_parts.append(case['刑事_罪名'])
    if case.get('刑事_刑期'):
        judgment_parts.append(case['刑事_刑期'])
    if case.get('刑事_罚金', '无') != '无':
        judgment_parts.append(case['刑事_罚金'])

    if judgment_parts:
        parts.append(f"\n【判决如下】\n" + "，".join(judgment_parts) + "。")

    return "\n".join(parts)


# =============================================================================
# SimuCourt(2) 二审数据模板
# =============================================================================

def build_prosecutor_input_simucourt2(case: Dict) -> str:
    """
    从SimuCourt(2)构建控方input（二审中被上诉人通常是检察院）
    """
    parts = []

    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    if case.get('法院'):
        parts.append(f"【二审法院】{case['法院']}")

    # 上诉人（原审被告）
    if case.get('上诉人'):
        parts.append(f"【上诉人（原审被告）】{case['上诉人']}")

    # 一审结果（重要背景）
    if case.get('一审法院审判结果'):
        parts.append(f"【一审判决结果】{case['一审法院审判结果']}")

    # 上诉请求
    if case.get('上诉请求'):
        parts.append(f"【上诉请求】{case['上诉请求']}")

    # 一审认定事实
    if case.get('一审法院认定事实'):
        parts.append(f"【一审认定事实】{case['一审法院认定事实']}")

    # 二审查明的新事实
    if case.get('本院二审查明事实', '无') != '无':
        parts.append(f"【二审查明事实】{case['本院二审查明事实']}")

    return "\n".join(parts)


def build_defender_input_simucourt2(case: Dict) -> str:
    """
    从SimuCourt(2)构建辩方input（上诉人角度）
    """
    parts = []

    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    if case.get('上诉人'):
        parts.append(f"【上诉人】{case['上诉人']}（你正在为其上诉辩护）")

    # 上诉人基本情况
    if case.get('上诉人基本情况', '无') != '无':
        parts.append(f"【上诉人基本情况】{case['上诉人基本情况']}")

    # 一审判决（需要质疑）
    if case.get('一审法院审判结果'):
        parts.append(f"【一审判决】{case['一审法院审判结果']}")

    if case.get('一审法院意见'):
        parts.append(f"【一审法院意见】{case['一审法院意见']}")

    # 一审事实
    if case.get('一审法院认定事实'):
        parts.append(f"【一审认定事实】{case['一审法院认定事实']}")

    return "\n".join(parts)


def build_judge_input_simucourt2(case: Dict) -> str:
    """
    从SimuCourt(2)构建法官input
    """
    parts = []

    if case.get('案件名'):
        parts.append(f"【案件名称】{case['案件名']}")

    case_type = f"【案件类型】{case.get('类别', '刑事')}案件"
    if case.get('案由'):
        case_type += f" - {case['案由']}"
    parts.append(case_type)

    if case.get('法院'):
        parts.append(f"【二审法院】{case['法院']}")

    if case.get('上诉人'):
        parts.append(f"【上诉人】{case['上诉人']}")

    # 一审情况
    parts.append("\n【一审情况】")
    if case.get('一审法院认定事实'):
        parts.append(f"一审认定事实：{case['一审法院认定事实']}")
    if case.get('一审法院意见'):
        parts.append(f"一审法院意见：{case['一审法院意见']}")
    if case.get('一审法院审判结果'):
        parts.append(f"一审判决：{case['一审法院审判结果']}")

    # 二审新事实
    if case.get('本院二审查明事实', '无') != '无':
        parts.append(f"\n【二审查明事实】{case['本院二审查明事实']}")

    # 上诉请求
    if case.get('上诉请求'):
        parts.append(f"\n【上诉请求】{case['上诉请求']}")

    # 双方意见
    if case.get('上诉人辩护', '无') != '无':
        parts.append(f"\n【上诉人辩护意见】\n{case['上诉人辩护']}")

    if case.get('被上诉人辩护', '无') != '无':
        parts.append(f"\n【被上诉人意见】\n{case['被上诉人辩护']}")

    return "\n".join(parts)


def build_prosecutor_output_simucourt2(case: Dict) -> str:
    """二审控方output：被上诉人辩护"""
    return case.get('被上诉人辩护', '')


def build_defender_output_simucourt2(case: Dict) -> Optional[str]:
    """二审辩方output：上诉人辩护"""
    output = case.get('上诉人辩护', '无')
    if output == '无':
        return None
    return output


def build_judge_output_simucourt2(case: Dict) -> str:
    """二审法官output"""
    parts = []

    if case.get('二审意见'):
        parts.append(f"【本院认为】\n{case['二审意见']}")

    # 法律条文
    laws = []
    for i in range(1, 12):
        law = case.get(f'引用法律条文{i}', '无')
        if law != '无':
            laws.append(law)

    if laws:
        parts.append(f"\n【法律依据】\n依照：" + "、".join(laws))

    # 二审结果
    results = []
    for i in range(1, 4):
        result = case.get(f'刑事_结果{i}', '无')
        if result != '无':
            results.append(result)

    if results:
        parts.append(f"\n【判决如下】\n" + "\n".join(results))

    # 简化标签
    if case.get('刑事_罪名'):
        parts.append(f"\n【罪名认定】{case['刑事_罪名']}")
    if case.get('刑事_刑期'):
        parts.append(f"【刑期】{case['刑事_刑期']}")

    return "\n".join(parts)


# =============================================================================
# judge-data模板（只有法官角色）
# =============================================================================

def build_judge_input_judgedata(case: Dict) -> str:
    """
    从judge-data构建法官input
    利用字段：Fact
    """
    parts = []

    parts.append("【案件事实】")
    parts.append(case.get('Fact', ''))

    return "\n".join(parts)


def build_judge_output_judgedata(case: Dict) -> str:
    """
    从judge-data构建法官output
    利用字段：Reasoning, Judgment, Sentence, Fine, Crime Type, Law Articles
    """
    parts = []

    # 推理部分
    if case.get('Reasoning'):
        parts.append(f"【本院认为】\n{case['Reasoning']}")

    # 判决部分
    if case.get('Judgment'):
        parts.append(f"\n【判决如下】\n{case['Judgment']}")

    # 结构化标签（用于训练模型输出结构化结果）
    if case.get('Crime Type'):
        crimes = ", ".join(case['Crime Type'])
        parts.append(f"\n【罪名】{crimes}")

    if case.get('Sentence'):
        sentences = ", ".join(case['Sentence'])
        parts.append(f"【刑期】{sentences}")

    if case.get('Fine'):
        fines = ", ".join(case['Fine'])
        parts.append(f"【罚金】{fines}")

    if case.get('Law Articles'):
        laws = ", ".join([f"刑法第{a}条" for a in case['Law Articles']])
        parts.append(f"【法条】{laws}")

    return "\n".join(parts)


# =============================================================================
# 统一数据点结构
# =============================================================================

@dataclass
class SFTDataPoint:
    """统一SFT数据格式"""
    instruction: str
    input: str
    output: str
    role: str  # prosecutor, defender, judge
    source: str  # simucourt1, simucourt2, judgedata
    case_id: str  # 案件标识


def convert_simucourt1_to_sft(case: Dict) -> List[SFTDataPoint]:
    """将SimuCourt(1)案例转换为SFT数据点列表"""
    results = []
    case_id = case.get('案号', case.get('序号', ''))

    # 控方数据（100%可用）
    prosecutor_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["prosecutor"],
        input=build_prosecutor_input_simucourt1(case),
        output=build_prosecutor_output_simucourt1(case),
        role="prosecutor",
        source="simucourt1",
        case_id=case_id
    )
    results.append(prosecutor_dp)

    # 辩方数据（仅在有辩护内容时生成）
    defender_output = build_defender_output_simucourt1(case)
    if defender_output:
        defender_dp = SFTDataPoint(
            instruction=INSTRUCTION_TEMPLATES["defender"],
            input=build_defender_input_simucourt1(case),
            output=defender_output,
            role="defender",
            source="simucourt1",
            case_id=case_id
        )
        results.append(defender_dp)

    # 法官数据（100%可用）
    judge_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["judge"],
        input=build_judge_input_simucourt1(case),
        output=build_judge_output_simucourt1(case),
        role="judge",
        source="simucourt1",
        case_id=case_id
    )
    results.append(judge_dp)

    return results


def convert_simucourt2_to_sft(case: Dict) -> List[SFTDataPoint]:
    """将SimuCourt(2)案例转换为SFT数据点列表"""
    results = []
    case_id = case.get('案号', case.get('序号', ''))

    # 控方数据（被上诉人，100%可用）
    prosecutor_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["prosecutor"],
        input=build_prosecutor_input_simucourt2(case),
        output=build_prosecutor_output_simucourt2(case),
        role="prosecutor",
        source="simucourt2",
        case_id=case_id
    )
    results.append(prosecutor_dp)

    # 辩方数据（上诉人辩护，32%可用）
    defender_output = build_defender_output_simucourt2(case)
    if defender_output:
        defender_dp = SFTDataPoint(
            instruction=INSTRUCTION_TEMPLATES["defender"],
            input=build_defender_input_simucourt2(case),
            output=defender_output,
            role="defender",
            source="simucourt2",
            case_id=case_id
        )
        results.append(defender_dp)

    # 法官数据（100%可用）
    judge_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["judge"],
        input=build_judge_input_simucourt2(case),
        output=build_judge_output_simucourt2(case),
        role="judge",
        source="simucourt2",
        case_id=case_id
    )
    results.append(judge_dp)

    return results


def convert_judgedata_to_sft(case: Dict) -> List[SFTDataPoint]:
    """将judge-data案例转换为SFT数据点（仅法官角色）"""
    results = []
    case_id = case.get('CaseId', '')

    # 法官数据
    judge_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["judge"],
        input=build_judge_input_judgedata(case),
        output=build_judge_output_judgedata(case),
        role="judge",
        source="judgedata",
        case_id=case_id
    )
    results.append(judge_dp)

    return results


# =============================================================================
# judge-data train/test JSONL格式模板
# =============================================================================

def build_judge_input_judgedata_jsonl(case: Dict) -> str:
    """
    从judge-data train/test JSONL格式构建法官input
    字段：text_id, text, la, fd
    """
    parts = []

    parts.append("【案件事实】")
    # text字段包含案情事实
    parts.append(case.get('text', ''))

    return "\n".join(parts)


def build_judge_output_judgedata_jsonl(case: Dict) -> str:
    """
    从judge-data train/test JSONL格式构建法官output
    字段：fd (完整判决书), la (法律条文编号)
    """
    parts = []

    # fd包含完整判决书，需要提取"本院认为"和"判决如下"部分
    fd = case.get('fd', '')

    # 提取本院认为部分
    if '本院认为' in fd:
        start = fd.find('本院认为')
        # 找到判决部分的开始
        end_markers = ['判决如下', '依照', '如不服本判决']
        end = len(fd)
        for marker in end_markers:
            if marker in fd[start:]:
                end = min(end, fd.find(marker, start))

        reasoning = fd[start:end].strip()
        parts.append(f"【本院认为】\n{reasoning}")

    # 提取判决部分
    if '判决如下' in fd:
        start = fd.find('判决如下')
        # 找到上诉说明部分
        end_markers = ['如不服本判决', '审判员', '审判长']
        end = len(fd)
        for marker in end_markers:
            if marker in fd[start:]:
                pos = fd.find(marker, start)
                if pos > 0:
                    end = min(end, pos)

        judgment = fd[start:end].strip()
        parts.append(f"\n【判决如下】\n{judgment}")

    # 法律条文
    la = case.get('la', [])
    if la:
        laws = ", ".join([f"刑法第{a}条" for a in la])
        parts.append(f"\n【法条】{laws}")

    return "\n".join(parts)


def convert_judgedata_jsonl_to_sft(case: Dict) -> List[SFTDataPoint]:
    """将judge-data JSONL格式案例转换为SFT数据点（仅法官角色）"""
    results = []
    case_id = case.get('text_id', '')

    # 法官数据
    judge_dp = SFTDataPoint(
        instruction=INSTRUCTION_TEMPLATES["judge"],
        input=build_judge_input_judgedata_jsonl(case),
        output=build_judge_output_judgedata_jsonl(case),
        role="judge",
        source="judgedata_jsonl",
        case_id=case_id
    )
    results.append(judge_dp)

    return results


def datapoint_to_json(dp: SFTDataPoint) -> Dict:
    """将SFTDataPoint转换为JSON格式"""
    return {
        "instruction": dp.instruction,
        "input": dp.input,
        "output": dp.output,
        "meta": {
            "role": dp.role,
            "source": dp.source,
            "case_id": dp.case_id
        }
    }