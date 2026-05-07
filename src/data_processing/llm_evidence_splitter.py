"""
LLM Evidence Splitter Module
模块化的LLM证据拆分接口，用户可自定义LLM调用方式

使用方法：
1. 继承LLMEvidenceSplitterBase类
2. 实现call_llm方法
3. 在build_rl_data.py中使用自定义的splitter

示例：
    class MyLLMSplitter(LLMEvidenceSplitterBase):
        def call_llm(self, prompt: str) -> str:
            # 使用您自己的LLM API
            response = my_api.call(prompt)
            return response

    splitter = MyLLMSplitter()
    evidence = splitter.split(fact)
"""

import os
import json
import re
import yaml
from typing import Dict, Optional, List
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod


@dataclass
class LLMConfig:
    """LLM配置"""
    enabled: bool = False
    model: str = "deepseek-v4-flash"  # DeepSeek推荐模型
    api_key: str = ""
    api_endpoint: str = "https://api.deepseek.com/chat/completions"  # OpenAI兼容格式
    temperature: float = 0.3
    max_tokens: int = 800
    batch_size: int = 10
    retry_times: int = 3
    retry_delay: float = 2.0


def load_llm_config(config_path: str = "configs/llm_config.yaml") -> LLMConfig:
    """
    从配置文件加载LLM配置

    Args:
        config_path: 配置文件路径

    Returns:
        LLMConfig对象
    """
    if not os.path.exists(config_path):
        return LLMConfig()

    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)

    llm_section = yaml_config.get('llm', {})

    # 支持环境变量读取API key
    api_key = llm_section.get('api_key', '')
    if not api_key:
        api_key = os.environ.get('DEEPSEEK_API_KEY', '')
        if not api_key:
            api_key = os.environ.get('OPENAI_API_KEY', '')

    return LLMConfig(
        enabled=llm_section.get('enabled', False),
        model=llm_section.get('model', 'deepseek-chat'),
        api_key=api_key,
        api_endpoint=llm_section.get('api_endpoint', 'https://api.deepseek.com/v1/chat/completions'),
        temperature=float(llm_section.get('temperature', 0.3)),
        max_tokens=int(llm_section.get('max_tokens', 500)),
        batch_size=int(llm_section.get('batch_size', 10)),
        retry_times=int(llm_section.get('retry_times', 3)),
        retry_delay=float(llm_section.get('retry_delay', 2.0))
    )


# ============================================================================
# 改进方案：罪名特异性证据层级框架
# ============================================================================

# 证据片段的通用法律作用定义（适用于所有罪名）
EVIDENCE_ROLES_GENERIC = {
    "subjective": {
        "description": "主观层证据",
        "legal_role_generic": "判断主观故意程度和犯罪动机",
        "key_questions_generic": ["是否有犯罪故意", "作案动机是什么", "是否有预谋"]
    },
    "objective": {
        "description": "客观层证据",
        "legal_role_generic": "证明犯罪行为和危害后果",
        "key_questions_generic": ["具体犯罪行为是什么", "造成什么后果", "涉及金额/伤害程度"]
    },
    "sentencing": {
        "description": "量刑层证据",
        "legal_role_generic": "决定量刑轻重",
        "key_questions_generic": ["是否有自首", "是否赔偿/退赃", "认罪态度", "是否有谅解"]
    }
}

# 罪名特异性证据层级定义
CRIME_SPECIFIC_EVIDENCE_LAYERS = {
    "故意伤害": {
        "subjective": {
            "legal_role": "区分'激情犯罪'与'预谋犯罪'",
            "key_evidence": ["作案动机", "是否有预谋", "工具准备情况"],
            "triggers": ["作案动机", "预谋", "事前准备", "购买工具", "为什么"]
        },
        "objective": {
            "legal_role": "区分'故意伤害'与'故意杀人（未遂）'",
            "key_evidence": ["伤害部位", "打击力度", "使用工具", "伤害次数"],
            "triggers": ["伤害部位", "打击力度", "作案手段", "伤口位置", "使用什么工具", "几下"]
        },
        "sentencing": {
            "legal_role": "决定是否适用从轻处罚",
            "key_evidence": ["自首", "赔偿", "被害人谅解", "认罪态度"],
            "triggers": ["自首", "赔偿", "谅解", "认罪态度", "案后表现"]
        }
    },

    "故意杀人": {
        "subjective": {
            "legal_role": "区分'直接故意'与'间接故意'",
            "key_evidence": ["杀人动机", "是否有预谋", "主观恶性程度"],
            "triggers": ["杀人动机", "预谋", "事前准备", "主观恶性", "为什么"]
        },
        "objective": {
            "legal_role": "判断杀人行为的完成度",
            "key_evidence": ["作案手段", "伤害部位", "后果程度", "被害人是否死亡"],
            "triggers": ["作案手段", "伤害部位", "后果", "死亡", "未遂"]
        },
        "sentencing": {
            "legal_role": "决定是否适用从轻处罚",
            "key_evidence": ["自首", "被害人过错", "赔偿情况"],
            "triggers": ["自首", "被害人过错", "赔偿", "谅解"]
        }
    },

    "诈骗": {
        "subjective": {
            "legal_role": "证明诈骗故意和获利动机",
            "key_evidence": ["诈骗动机", "是否有预谋", "虚构事实的意图"],
            "triggers": ["诈骗动机", "预谋", "事前准备", "虚构内容", "为什么", "获利目的"]
        },
        "objective": {
            "legal_role": "认定诈骗行为和涉案金额",
            "key_evidence": ["诈骗手段", "虚构内容", "涉案金额", "被害人数量"],
            "triggers": ["诈骗手段", "虚构", "欺骗方式", "涉案金额", "骗取", "被害人", "具体行为"]
        },
        "sentencing": {
            "legal_role": "决定量刑档次和从轻情节",
            "key_evidence": ["自首", "退赃情况", "认罪态度", "累犯情况"],
            "triggers": ["自首", "退赃", "认罪态度", "累犯", "案后表现", "赔偿"]
        }
    },

    "盗窃": {
        "subjective": {
            "legal_role": "证明盗窃故意",
            "key_evidence": ["盗窃动机", "是否有预谋", "主观意图"],
            "triggers": ["盗窃动机", "预谋", "主观故意", "为什么", "事前准备"]
        },
        "objective": {
            "legal_role": "认定盗窃行为和涉案金额",
            "key_evidence": ["盗窃手段", "涉案金额", "入户/公共场所", "是否多次"],
            "triggers": ["盗窃手段", "涉案金额", "入户", "公共场所", "多次", "具体行为"]
        },
        "sentencing": {
            "legal_role": "决定量刑档次和从轻情节",
            "key_evidence": ["自首", "退赃情况", "认罪态度"],
            "triggers": ["自首", "退赃", "认罪态度", "案后表现"]
        }
    },

    "抢劫": {
        "subjective": {
            "legal_role": "证明抢劫故意和暴力意图",
            "key_evidence": ["抢劫动机", "是否有预谋", "主观恶性"],
            "triggers": ["抢劫动机", "预谋", "主观故意", "为什么", "暴力意图"]
        },
        "objective": {
            "legal_role": "认定抢劫行为、暴力和涉案金额",
            "key_evidence": ["抢劫手段", "暴力程度", "涉案金额", "是否持械"],
            "triggers": ["抢劫手段", "暴力", "涉案金额", "持械", "具体行为"]
        },
        "sentencing": {
            "legal_role": "决定是否适用加重/从轻情节",
            "key_evidence": ["自首", "被害人伤情", "认罪态度"],
            "triggers": ["自首", "被害人伤情", "认罪态度", "案后表现"]
        }
    },

    "危险驾驶": {
        "subjective": {
            "legal_role": "判断醉酒程度和主观过错",
            "key_evidence": ["是否明知醉酒", "是否故意酒后驾驶"],
            "triggers": ["醉酒程度", "明知", "主观过错", "是否饮酒"]
        },
        "objective": {
            "legal_role": "认定驾驶行为和酒精含量",
            "key_evidence": ["血液酒精含量", "驾驶行为", "是否造成事故", "驾驶路段"],
            "triggers": ["酒精含量", "驾驶行为", "事故", "驾驶路段", "血液", "醉酒"]
        },
        "sentencing": {
            "legal_role": "决定量刑轻重",
            "key_evidence": ["是否造成人员伤亡", "是否有赔偿", "认罪态度"],
            "triggers": ["人员伤亡", "赔偿", "认罪态度", "案后表现"]
        }
    },

    "交通肇事": {
        "subjective": {
            "legal_role": "判断主观过错程度",
            "key_evidence": ["是否违反交通规则", "主观过错类型"],
            "triggers": ["违反交通规则", "主观过错", "是否明知"]
        },
        "objective": {
            "legal_role": "认定事故责任和危害后果",
            "key_evidence": ["事故责任认定", "危害后果", "人员伤亡情况"],
            "triggers": ["事故责任", "危害后果", "人员伤亡", "具体行为"]
        },
        "sentencing": {
            "legal_role": "决定量刑档次",
            "key_evidence": ["自首", "赔偿情况", "被害人谅解"],
            "triggers": ["自首", "赔偿", "谅解", "案后表现"]
        }
    }
}

# ============================================================================
# 罪名分类触发词体系（扩展覆盖100+罪名）
# ============================================================================

# 罪名大类触发词配置（按犯罪类型分类）
CRIME_CATEGORY_TRIGGERS = {
    # 人身伤害类（故意伤害、故意杀人等）
    "人身伤害类": {
        "subjective": [
            "作案动机", "动机", "为什么", "原因", "目的",
            "事前准备", "预谋", "策划", "蓄意", "谋划",
            "购买工具", "工具来源", "报复", "泄愤", "怨恨",
            "故意", "明知", "主观", "是否有预谋"
        ],
        "objective": [
            "作案手段", "手段", "方式", "怎么做的", "怎么实施", "具体行为",
            "伤害部位", "伤口位置", "伤口", "打击部位", "伤情",
            "打击力度", "刺了几刀", "连续", "多次", "几下",
            "伤害程度", "重伤", "轻伤", "伤情鉴定",
            "持刀", "持械", "使用工具", "打击", "捅", "刺", "砍"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案", "报案情况",
            "拨打110", "拨打120", "原地等待",
            "赔偿", "赔偿情况", "医疗费", "损失赔偿", "和解",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "谅解", "取得谅解", "被害人态度", "积极救治"
        ]
    },

    # 财产犯罪类（诈骗、盗窃、抢劫等）
    "财产犯罪类": {
        "subjective": [
            "作案动机", "动机", "获利目的", "非法占有",
            "预谋", "策划", "事前准备", "故意", "明知",
            "为什么", "目的", "主观", "诈骗故意", "盗窃故意"
        ],
        "objective": [
            "作案手段", "手段", "方式", "怎么做的", "具体行为",
            "涉案金额", "金额", "数额", "骗取", "盗窃", "抢劫",
            "合同", "转账", "发票", "虚构事实", "虚构内容",
            "入户", "公共场所", "多次盗窃", "被害人数量",
            "欺骗方式", "诈骗手段", "盗窃手段", "抢劫手段",
            "秘密窃取", "暴力胁迫", "抢夺", "敲诈勒索"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "退赃", "退赔", "返还财物", "赔偿损失",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "累犯", "前科", "谅解", "取得谅解",
            "被害人态度", "退赃情况"
        ]
    },

    # 毒品犯罪类（贩卖、运输、持有毒品等）
    "毒品犯罪类": {
        "subjective": [
            "明知", "贩毒动机", "获利目的", "预谋",
            "故意", "主观", "为什么", "目的",
            "是否明知是毒品", "贩毒故意"
        ],
        "objective": [
            "毒品", "克数", "纯度", "甲基苯丙胺", "冰毒",
            "贩卖", "运输", "持有", "制造毒品",
            "毒品数量", "毒品类型", "海洛因", "鸦片",
            "贩卖毒品", "运输毒品", "持有毒品",
            "交易地点", "交易方式", "包装"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "配合调查", "立功", "检举揭发", "特情介入",
            "控制下交付", "诱惑侦查"
        ]
    },

    # 职务犯罪类（贪污、受贿、挪用公款等）
    "职务犯罪类": {
        "subjective": [
            "职务便利", "贪污故意", "受贿故意", "挪用故意",
            "明知", "故意", "主观", "为什么", "目的",
            "预谋", "策划", "事前准备", "利用职权"
        ],
        "objective": [
            "贪污", "受贿", "挪用", "职权", "公款",
            "账目", "侵吞", "窃取", "骗取",
            "索贿", "收受贿赂", "权钱交易",
            "挪用公款", "归个人使用", "超过三个月",
            "职务便利", "利用职务", "管理权限"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "退赃", "退赔", "上缴违纪所得",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "立功", "配合调查", "检举揭发"
        ]
    },

    # 交通犯罪类（危险驾驶、交通肇事等）
    "交通犯罪类": {
        "subjective": [
            "明知醉酒", "主观过错", "是否饮酒", "是否明知",
            "故意", "过失", "主观", "为什么",
            "违反交通规则", "明知违法"
        ],
        "objective": [
            "酒精含量", "血液酒精", "血液", "驾驶行为",
            "事故责任", "醉酒驾驶", "酒后驾驶",
            "伤亡情况", "财产损失", "事故后果",
            "驾驶路段", "驾驶时间", "车辆类型",
            "交通事故认定", "责任划分"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "赔偿", "赔偿情况", "医疗费", "损失赔偿",
            "谅解", "取得谅解", "被害人态度",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚"
        ]
    },

    # 性犯罪类（强奸、猥亵等）
    "性犯罪类": {
        "subjective": [
            "强行", "违背意志", "明知幼女", "强制故意",
            "胁迫", "故意", "主观", "明知",
            "强奸故意", "猥亵故意", "利用职权"
        ],
        "objective": [
            "强制", "猥亵", "强奸手段", "暴力",
            "胁迫", "幼女", "违背意愿",
            "强行发生性关系", "猥亵行为",
            "被害人反抗", "不敢反抗", "不能反抗",
            "作案地点", "作案时间"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "赔偿", "谅解", "取得谅解",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "被害人态度"
        ]
    },

    # 网络犯罪类（非法侵入信息系统、数据窃取等）
    "网络犯罪类": {
        "subjective": [
            "侵入故意", "控制故意", "获取数据故意",
            "明知", "故意", "主观", "为什么", "目的",
            "预谋", "黑客故意", "窃取故意"
        ],
        "objective": [
            "数据", "信息系统", "侵入", "控制",
            "黑客", "窃取数据", "网络攻击",
            "计算机系统", "服务器", "数据库",
            "非法获取", "非法控制", "破坏系统",
            "传播病毒", "木马程序", "网络入侵"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "退赃", "配合调查", "技术协助"
        ]
    },

    # 妨害社会管理类（介绍卖淫、赌博、传播淫秽物品等）
    "妨害社会管理类": {
        "subjective": [
            "明知", "妨害故意", "获利目的", "组织故意",
            "故意", "主观", "为什么", "目的",
            "预谋", "策划"
        ],
        "objective": [
            "招嫖卡片", "卖淫", "赌博", "传播淫秽",
            "组织卖淫", "介绍卖淫", "联系方式",
            "招嫖电话", "招嫖信息", "卖淫嫖娼",
            "赌博方式", "赌资", "淫秽物品",
            "传播范围", "传播数量", "点击量"
        ],
        "sentencing": [
            "案后表现", "自首", "投案", "主动投案",
            "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
            "配合调查", "退赃", "违法所得"
        ]
    }
}

# 罪名到大类映射字典
CRIME_TO_CATEGORY_MAP = {
    # 人身伤害类
    "故意伤害": "人身伤害类",
    "故意杀人": "人身伤害类",
    "过失致人死亡": "人身伤害类",
    "过失致人重伤": "人身伤害类",
    "聚众斗殴": "人身伤害类",
    "寻衅滋事": "人身伤害类",  # 也可归为妨害社会管理类

    # 财产犯罪类
    "诈骗": "财产犯罪类",
    "盗窃": "财产犯罪类",
    "抢劫": "财产犯罪类",
    "抢夺": "财产犯罪类",
    "敲诈勒索": "财产犯罪类",
    "侵占": "财产犯罪类",
    "职务侵占": "财产犯罪类",
    "挪用资金": "财产犯罪类",
    "信用卡诈骗": "财产犯罪类",
    "合同诈骗": "财产犯罪类",
    "集资诈骗": "财产犯罪类",
    "非法吸收公众存款": "财产犯罪类",
    "组织领导传销": "财产犯罪类",
    "传销": "财产犯罪类",
    "洗钱": "财产犯罪类",
    "贷款诈骗": "财产犯罪类",
    "票据诈骗": "财产犯罪类",
    "保险诈骗": "财产犯罪类",
    "骗取贷款": "财产犯罪类",

    # 毒品犯罪类
    "贩卖毒品": "毒品犯罪类",
    "运输毒品": "毒品犯罪类",
    "制造毒品": "毒品犯罪类",
    "持有毒品": "毒品犯罪类",
    "非法持有毒品": "毒品犯罪类",
    "容留他人吸毒": "毒品犯罪类",
    "引诱教唆欺骗他人吸毒": "毒品犯罪类",
    "非法提供麻醉药品": "毒品犯罪类",

    # 职务犯罪类
    "贪污": "职务犯罪类",
    "受贿": "职务犯罪类",
    "挪用公款": "职务犯罪类",
    "行贿": "职务犯罪类",
    "单位受贿": "职务犯罪类",
    "单位行贿": "职务犯罪类",
    "滥用职权": "职务犯罪类",
    "玩忽职守": "职务犯罪类",
    "徇私枉法": "职务犯罪类",
    "巨额财产来源不明": "职务犯罪类",

    # 交通犯罪类
    "危险驾驶": "交通犯罪类",
    "交通肇事": "交通犯罪类",
    "重大责任事故": "交通犯罪类",

    # 性犯罪类
    "强奸": "性犯罪类",
    "猥亵": "性犯罪类",
    "强制猥亵": "性犯罪类",
    "猥亵儿童": "性犯罪类",
    "强制猥亵侮辱": "性犯罪类",

    # 网络犯罪类
    "非法侵入计算机信息系统": "网络犯罪类",
    "非法获取计算机信息系统数据": "网络犯罪类",
    "破坏计算机信息系统": "网络犯罪类",
    "非法控制计算机信息系统": "网络犯罪类",
    "帮助信息网络犯罪活动": "网络犯罪类",

    # 妨害社会管理类
    "介绍卖淫": "妨害社会管理类",
    "容留卖淫": "妨害社会管理类",
    "组织卖淫": "妨害社会管理类",
    "强迫卖淫": "妨害社会管理类",
    "协助组织卖淫": "妨害社会管理类",
    "赌博": "妨害社会管理类",
    "开设赌场": "妨害社会管理类",
    "传播淫秽物品": "妨害社会管理类",
    "制作淫秽物品": "妨害社会管理类",
    "贩卖淫秽物品": "妨害社会管理类",
    "聚众扰乱社会秩序": "妨害社会管理类",
    "妨害公务": "妨害社会管理类",
    "虚假诉讼": "妨害社会管理类"
}


def get_crime_category(crime_type: str) -> str:
    """
    根据罪名获取犯罪大类

    Args:
        crime_type: 罪名（如"诈骗罪"、"故意伤害罪"等）

    Returns:
        犯罪大类名称，如"财产犯罪类"、"人身伤害类"等
    """
    # 清理罪名格式（去除"罪"字）
    crime_key = crime_type.replace("罪", "").strip()

    # 查找映射
    for key in CRIME_TO_CATEGORY_MAP:
        if key in crime_type or crime_key in key:
            return CRIME_TO_CATEGORY_MAP[key]

    # 未找到映射时，根据关键词推断
    if any(kw in crime_type for kw in ["伤害", "杀人", "殴打", "斗殴", "致死", "重伤", "轻伤"]):
        return "人身伤害类"
    elif any(kw in crime_type for kw in ["诈骗", "盗窃", "抢劫", "抢夺", "侵占", "敲诈", "勒索", "洗钱", "传销"]):
        return "财产犯罪类"
    elif any(kw in crime_type for kw in ["毒品", "贩毒", "运毒", "制毒", "持有毒品", "吸毒"]):
        return "毒品犯罪类"
    elif any(kw in crime_type for kw in ["贪污", "受贿", "挪用公款", "行贿", "职权", "渎职", "职务"]):
        return "职务犯罪类"
    elif any(kw in crime_type for kw in ["交通", "驾驶", "肇事", "醉酒", "酒后驾驶"]):
        return "交通犯罪类"
    elif any(kw in crime_type for kw in ["强奸", "猥亵", "性侵", "幼女"]):
        return "性犯罪类"
    elif any(kw in crime_type for kw in ["计算机", "信息系统", "网络", "数据", "黑客"]):
        return "网络犯罪类"
    elif any(kw in crime_type for kw in ["卖淫", "嫖娼", "赌博", "淫秽", "妨害", "扰乱"]):
        return "妨害社会管理类"

    # 无法分类时返回空字符串
    return ""


# 保留原有的通用触发词（作为最终fallback，扩展覆盖）
TRIGGER_KEYWORDS_GENERIC = {
    "subjective": [
        "作案动机", "动机", "为什么", "原因", "目的",
        "事前准备", "预谋", "策划", "蓄意", "谋划",
        "作案工具来源", "购买行为", "购买工具", "工具来源",
        "是否有预谋", "故意", "明知", "主观",
        "获利目的", "非法占有", "报复", "泄愤"
    ],
    "objective": [
        "作案手段", "手段", "方式", "怎么做的", "怎么实施", "具体行为",
        "伤害部位", "伤口位置", "伤口", "打击部位", "伤情",
        "打击力度", "刺了几刀", "连续", "多次", "几下",
        "伤害程度", "重伤", "轻伤", "伤情鉴定",
        "涉案金额", "金额", "数额", "骗取", "盗窃",
        "持刀", "持械", "使用工具", "合同", "转账", "发票",
        "虚构事实", "毒品", "克数", "纯度", "贩卖",
        "职务便利", "贪污", "受贿", "挪用", "公款",
        "酒精含量", "血液", "驾驶行为", "事故责任",
        "强制", "猥亵", "违背意志", "侵入", "信息系统",
        "招嫖卡片", "卖淫", "赌博", "传播淫秽"
    ],
    "sentencing": [
        "案后表现", "自首", "投案", "主动投案", "报案情况",
        "拨打110", "拨打120", "原地等待",
        "赔偿", "赔偿情况", "医疗费", "损失赔偿", "和解",
        "认罪态度", "悔罪", "坦白", "如实供述", "认罪认罚",
        "谅解", "取得谅解", "被害人态度", "退赃",
        "累犯", "前科", "立功", "配合调查"
    ]
}


# 保留原有的旧定义（向后兼容）
TRIGGER_KEYWORDS = TRIGGER_KEYWORDS_GENERIC

# 向后兼容定义
EVIDENCE_ROLES = EVIDENCE_ROLES_GENERIC

# 必经要素清单模板
REQUIRED_ELEMENTS = {
    "故意伤害": ["主观动机", "作案工具", "打击部位", "案后表现"],
    "故意杀人": ["主观动机", "作案工具", "打击部位", "后果程度"],
    "盗窃": ["主观故意", "作案手段", "涉案金额"],
    "抢劫": ["主观故意", "作案手段", "涉案金额", "危害后果"],
    "诈骗": ["主观故意", "诈骗手段", "涉案金额"],
    "危险驾驶": ["血液酒精含量", "驾驶行为"],
    "交通肇事": ["事故责任", "危害后果", "案后表现"],
    "default": ["主观动机", "作案手段", "案后表现"]
}

# LLM证据拆分Prompt模板（静态版本，向后兼容）
EVIDENCE_SPLIT_PROMPT = """请分析以下刑事案件事实，将其拆分为四个部分。

案件事实：
{fact}

请严格按照以下JSON格式输出（不要添加任何额外文字）：
{{
    "public_info": "中度模糊的案情概述（仅保留当事人、时间、地点、基本冲突类型，去除所有细节）",
    "subjective_evidence": {{
        "content": "精确提取主观层证据片段（如购买工具、预谋行为等具体描述，仅提取关键句子而非长段）",
        "role": "此证据的法律作用：区分'激情犯罪'与'预谋犯罪'"
    }},
    "objective_evidence": {{
        "content": "精确提取客观层证据片段（如伤害部位、打击力度等具体描述，仅提取关键句子而非长段）",
        "role": "此证据的法律作用：区分'故意伤害'与'故意杀人（未遂）'"
    }},
    "sentencing_evidence": {{
        "content": "精确提取量刑层证据片段（如自首、赔偿等具体描述，仅提取关键句子而非长段）",
        "role": "此证据的法律作用：决定是否适用从轻处罚"
    }}
}}

输出要求：
1. public_info示例："2026年3月，被告人张三与被害人李四发生冲突，造成李四受伤"
2. 证据片段应精确提取单一关键事实，例如：
   - 主观："张三案发前一周购买了折叠刀并记录被害人行踪"
   - 客观："刀伤位于左胸部，深达肺部，连续刺了三刀"
   - 量刑："张三现场拨打120并如实供述"
3. 如果某层证据不存在，content输出null，role保留默认描述
4. 不要输出长段落，只提取核心事实句子"""


def build_dynamic_evidence_split_prompt(fact: str, crime_type: str) -> str:
    """
    构建罪名特异性的证据拆分Prompt

    Args:
        fact: 案件事实文本
        crime_type: 罪名

    Returns:
        动态生成的证据拆分Prompt
    """
    # 获取罪名特异性配置
    layers_config = get_crime_specific_evidence_layers(crime_type)

    # 构建动态Prompt
    prompt = f"""请分析以下刑事案件事实，将其拆分为四个部分。

案件事实：
{fact}

罪名类型：{crime_type}

请严格按照以下JSON格式输出（不要添加任何额外文字）：
{{
    "public_info": "中度模糊的案情概述（仅保留当事人、时间、地点、基本冲突类型，去除所有细节）",
    "subjective_evidence": {{
        "content": "精确提取主观层证据片段",
        "role": "{layers_config['subjective']['legal_role']}"
    }},
    "objective_evidence": {{
        "content": "精确提取客观层证据片段",
        "role": "{layers_config['objective']['legal_role']}"
    }},
    "sentencing_evidence": {{
        "content": "精确提取量刑层证据片段",
        "role": "{layers_config['sentencing']['legal_role']}"
    }}
}}

证据提取指导：
1. 主观层证据（{layers_config['subjective']['legal_role']}）：
   - 关键证据类型：{', '.join(layers_config['subjective']['key_evidence'])}
   - 例如：{layers_config['subjective']['key_evidence'][0]}相关的具体描述

2. 客观层证据（{layers_config['objective']['legal_role']}）：
   - 关键证据类型：{', '.join(layers_config['objective']['key_evidence'])}
   - 例如：{layers_config['objective']['key_evidence'][0]}相关的具体描述

3. 量刑层证据（{layers_config['sentencing']['legal_role']}）：
   - 关键证据类型：{', '.join(layers_config['sentencing']['key_evidence'])}
   - 例如：{layers_config['sentencing']['key_evidence'][0]}相关的具体描述

输出要求：
1. public_info应保留基本框架但模糊细节
2. 证据片段应精确提取单一关键事实，仅提取关键句子而非长段落
3. 如果某层证据不存在，content输出"无相关信息"，role保留上述描述
4. 不要输出长段落，只提取核心事实句子"""

    return prompt


class LLMEvidenceSplitterBase(ABC):
    """
    LLM证据拆分器基类

    用户可以继承此类并实现自己的LLM调用方式

    示例自定义实现：

    class MyLLMSplitter(LLMEvidenceSplitterBase):
        def call_llm(self, prompt: str) -> str:
            # 方式1：使用OpenAI
            import openai
            client = openai.Client(api_key="your-key")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

            # 方式2：使用本地模型
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("your-model")
            model = AutoModelForCausalLM.from_pretrained("your-model")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=500)
            return tokenizer.decode(outputs[0])

            # 方式3：使用其他API
            response = requests.post("your-api-endpoint", ...)
            return response.json()["content"]
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化拆分器

        Args:
            config: LLM配置
        """
        self.config = config or LLMConfig()

    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """
        调用LLM获取响应（用户需实现此方法）

        Args:
            prompt: 输入prompt

        Returns:
            LLM响应文本
        """
        pass

    def split(self, fact: str, crime_type: str = "") -> Dict:
        """
        拆分证据层

        Args:
            fact: 案件事实文本
            crime_type: 罪名（用于辅助判断和动态prompt生成）

        Returns:
            拆分结果字典
        """
        # 如果LLM未启用，直接使用规则匹配
        if not self.config.enabled or not self.config.api_key:
            return self._fallback_split(fact, crime_type)

        # 构建动态prompt（根据罪名特异性）
        if crime_type:
            prompt = build_dynamic_evidence_split_prompt(fact, crime_type)
        else:
            # 无罪名时使用静态prompt
            prompt = EVIDENCE_SPLIT_PROMPT.format(fact=fact)

        # 调用LLM
        try:
            response = self.call_llm(prompt)
            evidence_dict = self._parse_response(response)

            if evidence_dict:
                return evidence_dict

        except Exception as e:
            print(f"LLM调用失败，回退到规则匹配: {e}")

        # 回退到规则匹配
        return self._fallback_split(fact, crime_type)

    def _parse_response(self, response: str) -> Optional[Dict]:
        """
        解析LLM响应

        Args:
            response: LLM响应文本

        Returns:
            解析后的字典
        """
        try:
            # 处理可能的markdown包裹
            content = response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            # 移除可能的前缀文字
            if "{" in content:
                content = content[content.index("{"):]

            evidence_dict = json.loads(content)
            return evidence_dict

        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            print(f"原始响应: {response[:200]}")
            return None

    def _fallback_split(self, fact: str, crime_type: str) -> Dict:
        """
        规则匹配回退方案

        Args:
            fact: 案件事实文本
            crime_type: 罪名

        Returns:
            拆分结果字典
        """
        # 主观层关键词
        subjective_patterns = [
            r"(动机|目的|为了|因.*纠纷|因.*矛盾|报复|泄愤|怨恨)",
            r"(事先|事前|预谋|策划|准备|蓄意|谋划|购买.*工具)",
            r"(故意|明知)"
        ]

        # 客观层关键词
        objective_patterns = [
            r"(手段|方式|持.*刀|持.*棍|使用.*工具|捅|刺|打|砍|殴打)",
            r"(头部|胸部|腹部|背部|左.*部|右.*部|伤情|伤口|部位)",
            r"(连续|多次|反复|数刀|几刀|重伤|轻伤)"
        ]

        # 量刑层关键词
        sentencing_patterns = [
            r"(自首|主动投案|拨打110|拨打120|原地等待|如实供述)",
            r"(赔偿|补偿|医疗费|损失|谅解|和解|达成协议)",
            r"(认罪|悔罪|坦白|配合|积极)"
        ]

        sentences = fact.split('。')

        subjective_match = None
        objective_match = None
        sentencing_match = None

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if not subjective_match:
                for pattern in subjective_patterns:
                    if re.search(pattern, sentence):
                        subjective_match = sentence
                        break

            if not objective_match:
                for pattern in objective_patterns:
                    if re.search(pattern, sentence):
                        objective_match = sentence
                        break

            if not sentencing_match:
                for pattern in sentencing_patterns:
                    if re.search(pattern, sentence):
                        sentencing_match = sentence
                        break

        # 构建public_info
        public_info = self._build_public_info(fact, crime_type)

        return {
            "public_info": public_info,
            "subjective_evidence": {
                "content": subjective_match,
                "role": EVIDENCE_ROLES["subjective"]["legal_role_generic"]
            },
            "objective_evidence": {
                "content": objective_match,
                "role": EVIDENCE_ROLES["objective"]["legal_role_generic"]
            },
            "sentencing_evidence": {
                "content": sentencing_match,
                "role": EVIDENCE_ROLES["sentencing"]["legal_role_generic"]
            }
        }

    def _build_public_info(self, fact: str, crime_type: str) -> str:
        """
        构建模糊的public_info

        Args:
            fact: 案件事实文本
            crime_type: 罪名

        Returns:
            模糊化的案情概述
        """
        # 提取当事人
        defendant_match = re.search(r"被告人([^\s，。]+)", fact)
        defendant = defendant_match.group(1) if defendant_match else "被告人"

        victim_match = re.search(r"被害人([^\s，。]+)", fact)
        victim = victim_match.group(1) if victim_match else "被害人"

        # 提取时间
        time_match = re.search(r"(\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d+时许|\d+时\d+分)", fact)
        time_info = time_match.group(1) if time_match else ""

        # 构建基本冲突描述
        if "伤害" in crime_type or "杀人" in crime_type or "故意" in crime_type:
            conflict = f"{defendant}与{victim}发生冲突，造成{victim}受伤"
        elif "盗窃" in crime_type or "抢劫" in crime_type:
            conflict = f"{defendant}实施了盗窃/抢劫行为"
        elif "危险驾驶" in crime_type or "交通肇事" in crime_type:
            conflict = f"{defendant}驾驶车辆发生交通事故"
        elif "诈骗" in crime_type:
            conflict = f"{defendant}实施了诈骗行为"
        else:
            conflict = f"{defendant}实施了违法行为"

        if time_info:
            return f"{time_info}，{conflict}。"
        return f"{conflict}。"


class DefaultLLMSplitter(LLMEvidenceSplitterBase):
    """
    默认LLM拆分器（使用requests调用API）

    用户可以参考此实现自定义自己的拆分器
    """

    def call_llm(self, prompt: str) -> str:
        """
        使用HTTP请求调用LLM API

        Args:
            prompt: 输入prompt

        Returns:
            LLM响应文本
        """
        import requests
        import time
        import urllib3
        from urllib3.util.ssl_ import create_urllib3_context

        # 完全禁用SSL验证（仅用于开发环境）
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        # 重试机制
        for attempt in range(self.config.retry_times):
            try:
                # 使用verify=False跳过SSL验证
                response = requests.post(
                    self.config.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=60,
                    verify=False  # 完全跳过SSL验证
                )

                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]

                elif response.status_code == 429:  # Rate limit
                    print(f"Rate limited, retrying in {self.config.retry_delay}s...")
                    time.sleep(self.config.retry_delay)
                    continue

                else:
                    print(f"API error: {response.status_code} - {response.text[:100]}")
                    if attempt < self.config.retry_times - 1:
                        time.sleep(self.config.retry_delay)
                        continue

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
                if attempt < self.config.retry_times - 1:
                    time.sleep(self.config.retry_delay)
                    continue

        raise Exception("LLM调用失败，已达最大重试次数")


def get_crime_specific_evidence_layers(crime_type: str) -> Dict:
    """
    根据罪名获取特异性证据层级配置

    Args:
        crime_type: 罪名（如"故意伤害罪"、"诈骗罪"等）

    Returns:
        罪名特异性的证据层级配置，包含：
        - subjective: {"legal_role", "key_evidence", "triggers"}
        - objective: {"legal_role", "key_evidence", "triggers"}
        - sentencing: {"legal_role", "key_evidence", "triggers"}
    """
    # 清理罪名格式（去除"罪"字）
    crime_key = crime_type.replace("罪", "")

    # 查找匹配的罪名配置
    for key in CRIME_SPECIFIC_EVIDENCE_LAYERS:
        if key in crime_type or crime_key in key:
            return CRIME_SPECIFIC_EVIDENCE_LAYERS[key]

    # 未找到特异性配置时，使用通用配置
    return {
        "subjective": {
            "legal_role": EVIDENCE_ROLES_GENERIC["subjective"]["legal_role_generic"],
            "key_evidence": ["作案动机", "主观故意", "是否有预谋"],
            "triggers": TRIGGER_KEYWORDS_GENERIC["subjective"]
        },
        "objective": {
            "legal_role": EVIDENCE_ROLES_GENERIC["objective"]["legal_role_generic"],
            "key_evidence": ["作案手段", "危害后果", "涉案金额"],
            "triggers": TRIGGER_KEYWORDS_GENERIC["objective"]
        },
        "sentencing": {
            "legal_role": EVIDENCE_ROLES_GENERIC["sentencing"]["legal_role_generic"],
            "key_evidence": ["自首", "赔偿", "认罪态度"],
            "triggers": TRIGGER_KEYWORDS_GENERIC["sentencing"]
        }
    }


def get_evidence_triggers(crime_type: str) -> Dict[str, List[str]]:
    """
    根据罪名获取证据触发词列表

    三层fallback优先级：
    1. 罪名特异性配置（CRIME_SPECIFIC_EVIDENCE_LAYERS）
    2. 罪名大类配置（CRIME_CATEGORY_TRIGGERS）
    3. 通用配置（TRIGGER_KEYWORDS_GENERIC）

    Args:
        crime_type: 罪名

    Returns:
        三层证据的触发词字典
    """
    # 优先级1：查找罪名特异性配置
    for key in CRIME_SPECIFIC_EVIDENCE_LAYERS:
        if key in crime_type:
            return {
                "subjective": CRIME_SPECIFIC_EVIDENCE_LAYERS[key]["subjective"]["triggers"],
                "objective": CRIME_SPECIFIC_EVIDENCE_LAYERS[key]["objective"]["triggers"],
                "sentencing": CRIME_SPECIFIC_EVIDENCE_LAYERS[key]["sentencing"]["triggers"]
            }

    # 优先级2：查找罪名大类配置
    category = get_crime_category(crime_type)
    if category and category in CRIME_CATEGORY_TRIGGERS:
        return {
            "subjective": CRIME_CATEGORY_TRIGGERS[category]["subjective"],
            "objective": CRIME_CATEGORY_TRIGGERS[category]["objective"],
            "sentencing": CRIME_CATEGORY_TRIGGERS[category]["sentencing"]
        }

    # 优先级3：返回通用配置
    return {
        "subjective": TRIGGER_KEYWORDS_GENERIC["subjective"],
        "objective": TRIGGER_KEYWORDS_GENERIC["objective"],
        "sentencing": TRIGGER_KEYWORDS_GENERIC["sentencing"]
    }


def get_required_elements(crime_type: str) -> List[str]:
    """
    根据罪名获取必经要素清单

    Args:
        crime_type: 罪名

    Returns:
        必经要素列表
    """
    crime_key = crime_type.replace("罪", "")
    for key in REQUIRED_ELEMENTS:
        if key in crime_type or crime_key in key:
            return REQUIRED_ELEMENTS[key]
    return REQUIRED_ELEMENTS["default"]


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例1：使用默认拆分器
    config = load_llm_config()
    splitter = DefaultLLMSplitter(config)

    # 示例事实
    fact = """
    2026年3月，被告人张三因琐事与被害人李四发生争执。
    张三案发前一周在某网店购买了折叠刀，并在日记中写道"要给李四点颜色"。
    案发当日，张三持刀将李四刺伤，刀伤位于左胸部，深达肺部，连续刺了三刀。
    经鉴定，李四构成重伤二级。案发后张三现场拨打120并报警，如实供述罪行。
    法院认定张三犯故意伤害罪，判处有期徒刑三年。
    """

    evidence = splitter.split(fact, "故意伤害罪")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))

    # 示例2：自定义LLM拆分器
    # 用户可以继承LLMEvidenceSplitterBase并实现call_llm方法
    # class MyCustomSplitter(LLMEvidenceSplitterBase):
    #     def call_llm(self, prompt: str) -> str:
    #         # 使用您自己的LLM
    #         ...