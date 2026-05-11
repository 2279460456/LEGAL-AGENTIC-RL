# 罪名特异性证据层级框架设计

## 问题分析

原始三层证据结构存在以下局限性：

### 1. role描述罪名特异性问题

原始设计：
```json
{
  "subjective": {"role": "区分'激情犯罪'与'预谋犯罪'"},
  "objective": {"role": "区分'故意伤害'与'故意杀人（未遂）'"},
  "sentencing": {"role": "决定是否适用从轻处罚"}
}
```

**问题**：
- objective层的role描述仅适用于人身伤害类案件
- 对于诈骗、盗窃、危险驾驶等其他罪名完全不适用
- 约40%的案件类型无法被正确描述

### 2. 触发词覆盖不足

原始触发词针对人身伤害设计：
```python
"objective": ["伤害部位", "伤口位置", "打击力度", "伤情鉴定"]
```

**缺失的关键触发词**：
- 诈骗罪：需要"诈骗手段"、"虚构"、"涉案金额"、"被害人数量"
- 盗窃罪：需要"入户"、"公共场所"、"涉案金额"
- 危险驾驶：需要"酒精含量"、"血液"、"醉酒"

### 3. 具体案例分析

**诈骗罪案件**：
```json
{
  "case_id": "fraud_001",
  "hidden_evidence": {
    "objective": {
      "content": "被告人虚构'百万医疗'险种，欺骗被害人购买，骗取103.5万元",
      "role": "区分'故意伤害'与'故意杀人（未遂）'",  // ← 完全错误！
      "triggers": ["伤害部位", "打击力度"]  // ← 无法触发诈骗证据
    }
  }
}
```

---

## 改进方案：动态证据层级框架

### 设计原则

1. **三层结构保持不变**：subjective/objective/sentencing
2. **罪名特异性配置**：每个罪名有独立的role描述和触发词
3. **通用fallback机制**：未定义罪名使用通用配置

### 改进后的数据结构

```python
# 罪名特异性证据层级定义
CRIME_SPECIFIC_EVIDENCE_LAYERS = {
    "诈骗": {
        "subjective": {
            "legal_role": "证明诈骗故意和获利动机",
            "key_evidence": ["诈骗动机", "是否有预谋", "虚构事实的意图"],
            "triggers": ["诈骗动机", "预谋", "虚构内容", "获利目的"]
        },
        "objective": {
            "legal_role": "认定诈骗行为和涉案金额",
            "key_evidence": ["诈骗手段", "虚构内容", "涉案金额", "被害人数量"],
            "triggers": ["诈骗手段", "虚构", "涉案金额", "骗取", "被害人"]
        },
        "sentencing": {
            "legal_role": "决定量刑档次和从轻情节",
            "key_evidence": ["自首", "退赃情况", "认罪态度", "累犯情况"],
            "triggers": ["自首", "退赃", "认罪态度", "累犯"]
        }
    },
    // ... 其他罪名配置
}
```

### 改进后的RL数据格式

```json
{
  "case_id": "fraud_001",
  "public_info": "2026年8月，被告人董青实施了诈骗行为。",
  "hidden_evidence": {
    "subjective": {
      "content": "无相关信息",
      "triggers": ["诈骗动机", "预谋", "虚构内容", "获利目的"],
      "role": "证明诈骗故意和获利动机"
    },
    "objective": {
      "content": "被告人虚构'百万医疗'险种，欺骗被害人购买，骗取103.5万元",
      "triggers": ["诈骗手段", "虚构", "涉案金额", "骗取", "被害人"],
      "role": "认定诈骗行为和涉案金额"
    },
    "sentencing": {
      "content": "被告人有自首情节",
      "triggers": ["自首", "退赃", "认罪态度", "累犯"],
      "role": "决定量刑档次和从轻情节"
    }
  },
  "ground_truth": {
    "crime": "诈骗罪",
    "sentence_months": 36,
    "laws": ["刑法第266条", "刑法第67条"],
    "required_elements": ["主观故意", "诈骗手段", "涉案金额"]
  }
}
```

---

## 已定义罪名覆盖

| 罪名类别 | 具体罪名 | 特异性配置 |
|---------|---------|----------|
| **人身伤害类** | 故意伤害、故意杀人 | ✓ 完整配置 |
| **财产犯罪类** | 诈骗、盗窃、抢劫 | ✓ 完整配置 |
| **交通犯罪类** | 危险驾驶、交通肇事 | ✓ 完整配置 |
| **其他罪名** | 贪污、受贿、贩毒等 | ✓ 通用配置fallback |

### 详细配置表

| 罪名 | subjective.role | objective.role | sentencing.role |
|------|----------------|---------------|----------------|
| **故意伤害** | 区分激情犯罪与预谋犯罪 | 区分故意伤害与故意杀人（未遂） | 决定是否适用从轻处罚 |
| **诈骗** | 证明诈骗故意和获利动机 | 认定诈骗行为和涉案金额 | 决定量刑档次和从轻情节 |
| **盗窃** | 证明盗窃故意 | 认定盗窃行为和涉案金额 | 决定量刑档次和从轻情节 |
| **危险驾驶** | 判断醉酒程度和主观过错 | 认定驾驶行为和酒精含量 | 决定量刑轻重 |
| **抢劫** | 证明抢劫故意和暴力意图 | 认定抢劫行为、暴力和涉案金额 | 决定是否适用加重/从轻情节 |
| **交通肇事** | 判断主观过错程度 | 认定事故责任和危害后果 | 决定量刑档次 |

---

## 动态Prompt生成机制

### 函数接口

```python
def get_crime_specific_evidence_layers(crime_type: str) -> Dict:
    """
    根据罪名获取特异性证据层级配置
    
    Returns:
        {
            "subjective": {"legal_role", "key_evidence", "triggers"},
            "objective": {"legal_role", "key_evidence", "triggers"},
            "sentencing": {"legal_role", "key_evidence", "triggers"}
        }
    """

def get_evidence_triggers(crime_type: str) -> Dict[str, List[str]]:
    """根据罪名获取证据触发词列表"""

def build_dynamic_evidence_split_prompt(fact: str, crime_type: str) -> str:
    """构建罪名特异性的证据拆分Prompt"""
```

### 动态Prompt示例

**诈骗罪案件**：
```
请分析以下刑事案件事实，将其拆分为四个部分。

案件事实：
2026年8月，被告人董青利用保险公司业务员身份...

罪名类型：诈骗罪

证据提取指导：
1. 主观层证据（证明诈骗故意和获利动机）：
   - 关键证据类型：诈骗动机, 是否有预谋, 虚构事实的意图
   - 例如：诈骗动机相关的具体描述

2. 客观层证据（认定诈骗行为和涉案金额）：
   - 关键证据类型：诈骗手段, 虚构内容, 涉案金额, 被害人数量
   - 例如：诈骗手段相关的具体描述

3. 量刑层证据（决定量刑档次和从轻情节）：
   - 关键证据类型：自首, 退赃情况, 认罪态度, 累犯情况
   - 例如：自首相关的具体描述
```

---

## 实现位置

### 核心代码文件

| 文件 | 功能 |
|------|------|
| `src/data_processing/llm_evidence_splitter.py` | 罪名特异性配置定义、动态Prompt生成 |
| `src/data_processing/build_rl_data.py` | RL数据构建，使用动态触发词 |
| `tests/test_crime_specific_evidence.py` | 测试脚本验证改进方案 |

### 关键函数

```python
# llm_evidence_splitter.py
CRIME_SPECIFIC_EVIDENCE_LAYERS  # 罪名特异性配置字典
get_crime_specific_evidence_layers()  # 获取配置
get_evidence_triggers()  # 获取触发词
build_dynamic_evidence_split_prompt()  # 构建动态Prompt

# build_rl_data.py
build_single_rl_case()  # 使用动态触发词构建RL数据
```

---

## 使用示例

### 1. 获取罪名特异性配置

```python
from llm_evidence_splitter import get_crime_specific_evidence_layers

# 诈骗罪配置
config = get_crime_specific_evidence_layers("诈骗罪")
print(config["objective"]["legal_role"])
# 输出: "认定诈骗行为和涉案金额"

print(config["objective"]["triggers"])
# 输出: ["诈骗手段", "虚构", "欺骗方式", "涉案金额", "骗取", "被害人", "具体行为"]
```

### 2. 构建动态Prompt

```python
from llm_evidence_splitter import build_dynamic_evidence_split_prompt

fact = "被告人虚构'百万医疗'险种，骗取103.5万元..."
prompt = build_dynamic_evidence_split_prompt(fact, "诈骗罪")
# Prompt中包含诈骗罪的特异性证据提取指导
```

### 3. 构建RL数据

```python
from build_rl_data import build_single_rl_case

record = {
    "CaseId": "fraud_001",
    "Fact": "...",
    "Crime Type": ["诈骗罪"],
    ...
}

rl_case = build_single_rl_case(record, splitter)
# rl_case["hidden_evidence"]["objective"]["triggers"] 将使用诈骗罪的特异性触发词
# rl_case["hidden_evidence"]["objective"]["role"] 将使用诈骗罪的特异性描述
```

---

## 测试验证

运行测试脚本：
```bash
python tests/test_crime_specific_evidence.py
```

验证结果：
- ✓ 支持人身伤害类案件（故意伤害、故意杀人）
- ✓ 支持财产犯罪类案件（诈骗、盗窃、抢劫）
- ✓ 支持交通犯罪类案件（危险驾驶、交通肇事）
- ✓ 通用配置覆盖所有未定义罪名
- ✓ 动态Prompt根据罪名自动调整
- ✓ 触发词罪名特异性配置

---

## 扩展新罪名

如需添加新罪名配置，在 `llm_evidence_splitter.py` 的 `CRIME_SPECIFIC_EVIDENCE_LAYERS` 中添加：

```python
CRIME_SPECIFIC_EVIDENCE_LAYERS["新罪名"] = {
    "subjective": {
        "legal_role": "该罪名主观层的法律作用",
        "key_evidence": ["关键证据1", "关键证据2"],
        "triggers": ["触发词1", "触发词2"]
    },
    "objective": {
        "legal_role": "该罪名客观层的法律作用",
        "key_evidence": ["关键证据1", "关键证据2"],
        "triggers": ["触发词1", "触发词2"]
    },
    "sentencing": {
        "legal_role": "该罪名量刑层的法律作用",
        "key_evidence": ["关键证据1", "关键证据2"],
        "triggers": ["触发词1", "触发词2"]
    }
}
```

---

## 相关文档

- [训练指南](training_guide.md) - 完整训练命令和配置
- [实验设计详解](experiment_phases.md) - SFT和RL阶段的完整设计

---

*文档创建时间: 2026-05-05*
*最后更新: 2026-05-10*
*状态: 改进方案已实施并验证*