# 实验阶段设计文档

本文档详细说明SFT阶段和RL阶段的实验设置，包括输入数据格式、训练过程和训练目的。

---

## 一、SFT阶段（监督微调）

### 1. 输入数据格式

**数据来源**：SimuCourt + judge-data

**数据结构**（`data/processed/` 目录）：
```json
{
  "instruction": "角色任务指令",
  "input": "案情信息",
  "output": "期望输出"
}
```

**三角色数据**：

| 角色 | instruction | input | output |
|------|------------|-------|--------|
| **控方** | 检察官指控犯罪指令 | 案情摘要（被告、被害人、基本案情） | 指控意见（定罪要素、法条引用） |
| **辩方** | 律师辩护指令 | 案情摘要 | 辩护意见（减轻情节、从轻理由） |
| **法官** | 法官审判指令 | 案情 + 控辩陈述 | CoT推理链 + 判决结论 |

**当前生成规模**：
- prosecutor.json: 419条
- defender.json: 227条
- judge.json: 2924条
- **总计**: 3570条

**数据示例**（法官角色）：
```json
{
  "instruction": "你是一位资深法官，正在审理案件...",
  "input": "被告人张三与被害人李四因纠纷发生冲突...\n控方指控：...\n辩方辩护：...",
  "output": "【判决书】\n一、事实认定：经审理查明...\n二、法律适用：本院认为...\n三、量刑考量：...\n四、判决结论：被告人犯故意伤害罪，判处有期徒刑三年"
}
```

### 2. 训练过程

**配置文件**：`configs/sft_config.yaml`

```yaml
基座模型: Qwen/Qwen3-8B
方法: LoRA（QLoRA可选，关闭4bit即LoRA，开启为QLoRA）

LoRA参数:
  - rank: 64
  - alpha: 16
  - target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

训练参数:
  - learning_rate: 2e-4
  - batch_size: 4, gradient_accumulation: 4 (effective_batch=16)
  - epochs: 3
  - max_seq_length: 2048
```

**训练流程**：
```
1. 加载Qwen3-8B基座模型
2. 配置LoRA适配器（仅训练 adapters 参数）
3. 加载三角色SFT数据（3570条）
4. 使用SFTTrainer进行监督微调
5. 格式化函数将样本转为Qwen对话格式：
   <|im_start|>user\n{instruction}\n\n{input}<|im_end|>
   <|im_start|>assistant\n{output}<|im_end|>
6. 训练完成后保存LoRA权重
```

**运行命令**：
```bash
python src/sft/train_sft.py --config configs/sft_config.yaml
```

### 3. 训练目的

**SFT阶段目标**：让模型学会"像法律人一样说话"

| 能力 | 说明 |
|------|------|
| **角色区分** | 同一模型能扮演控方、辩方、法官三个角色 |
| **语言规范** | 使用正确的法律术语（被告人、本院认为、依照...） |
| **格式规范** | 输出符合法律文书格式（指控/辩护/判决结构） |
| **CoT推理** | 法官角色能输出完整的推理链（事实→法律→量刑→判决） |

**SFT是RL的初始化**：为RL阶段提供能理解法律任务的基础模型

---

## 二、RL阶段（策略优化）

### 1. 输入数据格式

**数据来源**：judge-data → `build_rl_data.py` 构建

**输出路径**：
- `data/rl_env/train_cases.json` (训练集)
- `data/rl_env/test_cases.json` (测试集)

**数据结构**（符合实验设计）：

```json
{
  "case_id": "唯一标识",
  
  "public_info": "初始公开信息（模糊案情）",
  
  "hidden_evidence": {
    "subjective": {
      "content": "主观层证据片段",
      "triggers": ["作案动机", "预谋", "购买行为"],
      "role": "区分'激情犯罪'与'预谋犯罪'"
    },
    "objective": {
      "content": "客观层证据片段",
      "triggers": ["作案手段", "伤害部位", "打击力度"],
      "role": "区分'故意伤害'与'故意杀人（未遂）'"
    },
    "sentencing": {
      "content": "量刑层证据片段",
      "triggers": ["自首", "赔偿", "认罪态度"],
      "role": "决定是否适用从轻处罚"
    }
  },
  
  "ground_truth": {
    "crime": "故意伤害罪",
    "sentence_months": 36,
    "laws": ["刑法第234条", "刑法第67条"],
    "sentence_text": "有期徒刑三年",
    "required_elements": ["主观动机", "作案工具", "打击部位", "案后表现"]
  }
}
```

**数据示例**：
```json
{
  "case_id": "xxx",
  "public_info": "2017年8月，董青利用保险公司业务员身份实施了诈骗行为。",
  "hidden_evidence": {
    "subjective": {
      "content": "无相关信息",
      "triggers": ["作案动机", "预谋", ...],
      "role": "区分'激情犯罪'与'预谋犯罪'"
    },
    "objective": {
      "content": "被告人虚构'百万医疗'险种，欺骗被害人购买，骗取103.5万元",
      "triggers": ["作案手段", "具体行为", ...],
      "role": "区分'故意伤害'与'故意杀人（未遂）'"
    },
    "sentencing": {
      "content": "被告人有自首情节",
      "triggers": ["自首", "案后表现", ...],
      "role": "决定是否适用从轻处罚"
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

**当前规模**：
- 训练集: 479条
- 测试集: 120条

### 2. 训练过程

**配置文件**：`configs/rl_config.yaml`

```yaml
GRPO参数:
  - group_size: 4 (每组采样4条轨迹)
  - temperature: 0.9 (保证多样性)
  - max_turns: 10 (最大对话轮次)
  - max_prompt_length: 2048 (Prompt最大长度)
  - max_completion_length: 512 (生成最大长度)

训练参数（2026-05-16优化）:
  - learning_rate: 5e-6 (更稳定)
  - max_grad_norm: 0.1 (更严格裁剪)
  - gradient_accumulation_steps: 4 (等效batch=4)
  - weight_decay: 0.1 (防止过拟合)
  - lr_scheduler_type: "cosine" (cosine衰减)
  - warmup_ratio: 0.1 (warmup比例)
  - optim: "paged_adamw_8bit" (节省显存)
  - adam_beta2: 0.99 (适应RL梯度波动)

奖励权重:
  - λ₁=0.5 (准确性)
  - λ₂=0.3 (信息效率)
  - λ₃=0.2 (程序合规)
```

**训练流程**：

```
For each episode:
    1. 【状态初始化】
       - 输入: public_info（模糊案情）
       - 已触发证据: 空
       
    2. 【组采样】对同一案件生成G=4条不同轨迹
    
    3. 【轨迹执行】每条轨迹：
       while not terminal:
         a. 法官智能体决策:
            - 提问(a_query): 询问证据细节
            - 或判决(a_judge): 给出最终判决
         
         b. 环境响应:
            - 若提问触发关键词 → 释放隐藏证据
            - 计算step_reward (仅P_irrelevant，证据触发奖励统一在最终R_info中计算)
         
         c. 记录: (state, action, log_prob, reward)
         
       最终奖励计算:
         R_total = λ₁·R_accuracy + λ₂·R_information + λ₃·R_compliance
       
    4. 【优势值计算】
       A_i = (R_i - mean(R_group)) / std(R_group)
       
    5. 【梯度更新】
       Loss = -Σ A_i · Σ log_prob(action)
       让高奖励轨迹概率↑，低奖励轨迹概率↓
```

**信息隐藏-触发机制（对话回复格式）**：

法官提问 → 触发证据层级 → 控辩双方回复

| 证据层 | 触发词示例 | 回复角色 | 回复格式示例 |
|--------|-----------|---------|-------------|
| **主观层** | "作案动机"、"预谋情况"、"工具来源" | **公诉人** | "公诉人补充说明：经调查，被告人案发前购买了折叠刀..." |
| **客观层** | "伤害部位"、"打击力度"、"伤口位置" | **公诉人** | "公诉人补充说明：关于作案情况，刀伤位于左胸部..." |
| **量刑层** | "是否自首"、"报案情况"、"赔偿" | **辩护人** | "辩护人补充说明：被告人案发后主动投案并如实供述..." |

**交互示例**：

```
法官提问："我想了解被告人的作案动机"
→ 触发主观层
→ 回复："公诉人补充说明：经调查，被告人预谋作案"

法官提问："是否有自首情节"
→ 触发量刑层
→ 回复："辩护人补充说明：被告人有自首情节"
```

**奖励计算公式（v5 - 最新版本）**：

```
R_total = 0.3·R_accuracy + 0.4·R_info + R_process

# R_accuracy (准确性奖励 - 混合相似度)
R_accuracy = 
  0.4·ROUGE(预测罪名, 真实罪名) +       # 罪名ROUGE相似度
  0.3·(1 - |刑期误差|/真实刑期) +        # 刑期相对误差
  0.3·法条召回率                         # 法条集合匹配

# R_info (信息收集奖励)
R_info = 
  证据发现率 + 0.1·已发现层级数           # 调查深度奖励

# R_process (过程奖励/惩罚)
R_process = 
  -0.5 (第1轮判决且无调查) +             # 早判重惩罚
  -0.2·(1-发现率) (证据不足就判决) +     # 过早判决惩罚
  -0.2 (罪名缺失) +                      # 格式检查
  +0.1 (3轮内且有调查) +                 # 效率奖励
  -0.2 (达到最大轮次)                    # 轮次惩罚
```

**ROUGE相似度计算**（用于罪名评估）：

```python
def compute_rouge_similarity(text1, text2):
    """
    基于F1分数的文本相似度计算
    适用场景：罪名相似度评估
    """
    # 使用jieba分词（或fallback到字符分割）
    words1 = jieba.lcut(text1)  # 如"故意伤害" → ["故意", "伤害"]
    words2 = jieba.lcut(text2)  # 如"故意伤害致死" → ["故意", "伤害", "致死"]
    
    # 计算词频交集
    common = Counter(words1) & Counter(words2)
    num_common = sum(common.values())
    
    precision = num_common / len(words1)
    recall = num_common / len(words2)
    
    f1 = 2·precision·recall / (precision + recall)
    return f1
```

**动作追踪机制**（用于人工审核和训练分析）：

```python
# 追踪变量
_all_actions = []      # 所有动作历史
_valid_actions = []    # 有效格式动作
_effective_actions = [] # 触发证据的动作
_invalid_actions = []  # 无效动作记录

# 统计报告
stats = env.get_action_statistics()
# 返回: {
#   "valid_rate": 有效动作比例,
#   "effective_rate": 触发证据比例,
#   "invalid_details": 无效动作详情（用于人工审核）
# }
```

**第1轮禁止判决规则**：

```
if current_round == 0 and action.type == "judge":
    reward = -0.3  # 早判惩罚
    is_terminal = False  # 不终止，强制继续调查
```

### 3. 训练目的

**RL阶段目标**：让法官智能体学会"调查性思维"

| 能力 | 说明 |
|------|------|
| **主动调查** | 学会在不确定案情中提问获取证据 |
| **策略决策** | 知道何时提问、何时判决 |
| **证据触发** | 通过关键词触发隐藏证据层 |
| **高效调查** | 少轮次获取关键证据（信息效率奖励） |
| **准确判决** | 基于完整证据做出正确判决（准确性奖励） |

**与SFT的区别**：

| 阶段 | 输入 | 输出 | 学习目标 |
|------|------|------|---------|
| **SFT** | 完整案情 | 静态判决 | "像法律人说话"（行为克隆） |
| **RL** | 模糊案情（需调查） | 动态推理过程 | "调查性思维"（策略优化） |

---

## 三、两个阶段的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    完整实验流程                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【Phase 1: SFT】                                           │
│     输入: 3570条完整法律文书                                  │
│     方法: LoRA监督微调                                       │
│     目的: 模型学会扮演控方/辩方/法官，理解法律任务             │
│     输出: SFT模型（法律语言能力）                             │
│                                                             │
│                         ↓ 初始化策略模型                      │
│                                                             │
│  【Phase 2: RL】                                            │
│     输入: 600条模糊案情（含隐藏证据）                         │
│     方法: GRPO策略优化                                       │
│     目的: 模型学会主动调查、高效获取证据、准确判决            │
│     输出: RL模型（调查推理策略）                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、实验假设验证设计

### 4.1 Baseline对比

| Baseline | 对比目的 |
|---------|---------|
| **原始Qwen模型** | 验证SFT训练的必要性 |
| **仅SFT模型** | 验证RL策略优化的必要性 |
| **静态输入** | 验证动态调查过程的有效性 |

### 4.2 评估指标

**SFT阶段评估**（见 `docs/sft_evaluation.md`）：
- Role Accuracy: 角色区分度
- Legal Term Coverage: 法律术语覆盖率
- CoT Completeness: 推理链完整度
- Crime Accuracy: 罪名准确率

**RL阶段评估**：
- 判决准确性: 罪名F1、刑期MAE、法条召回率
- 调查效率: 平均轮次、证据召回率、无效提问率
- 程序合规: 术语规范度、推理完整性

---

## 五、关键文件位置

```
LEGAL-AGENTIC-RL/
├── configs/
│   ├── sft_config.yaml          # SFT配置
│   ├── rl_config.yaml           # RL配置
│   └── llm_config.yaml          # LLM证据拆分配置
│
├── data/
│   ├── processed/               # SFT数据（3570条）
│   │   ├── prosecutor.json
│   │   ├── defender.json
│   │   ├── judge.json
│   │   └── all_roles.json
│   └── rl_env/                  # RL数据（600条）
│       ├── train_cases.json
│       ├── test_cases.json
│       └── build_stats.json
│
├── src/
│   ├── sft/train_sft.py         # SFT训练脚本
│   ├── rl/
│   │   ├── environment.py       # RL环境（信息隐藏机制）
│   │   ├── reward.py            # 奖励函数
│   │   ├── grpo.py              # GRPO算法
│   │   └── train_rl.py          # RL训练脚本
│   └── data_processing/
│       ├── build_rl_data.py     # RL数据构建
│       └── llm_evidence_splitter.py  # LLM证据拆分模块
│
└── docs/
    ├── sft_evaluation.md        # SFT评估方案
    └── experiment_phases.md     # 本文档
```

---

## 六、运行命令

**SFT阶段**：
```bash
# SFT训练
python src/sft/train_sft.py --config configs/sft_config.yaml

# SFT评估
python src/evaluation/sft_eval.py --model models/sft_checkpoint --test_data data/processed/all_roles.json
```

**RL阶段**：
```bash
# 构建RL数据（不使用LLM）
python src/data_processing/build_rl_data.py --max_samples 600

# 构建RL数据（使用LLM，需先配置llm_config.yaml）
python src/data_processing/build_rl_data.py --config configs/llm_config.yaml --use_llm --max_samples 600

# RL训练
python src/rl/train_rl.py --config configs/rl_config.yaml
```

---

*文档创建时间: 2026-05-05*
*最后更新: 2026-05-17*
*状态: 完整实验阶段说明（含对话交互 + 训练稳定性优化 + 奖励函数优化）*

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-05-17 | 奖励函数优化：移除步级奖励中的B_trigger和P_repeat，避免重复奖励/惩罚 |
| 2026-05-17 | 防重复提问机制：新增asked_questions记录和prompt中的已调查方向摘要 |
| 2026-05-16 | 训练稳定性优化：weight_decay、cosine lr scheduler、paged_adamw_8bit、gradient_accumulation_steps=4 |
| 2026-05-16 | 参数调整：learning_rate 5e-6、max_grad_norm 0.1、adam_beta2 0.99 |
| 2026-05-13 | 对话交互设计：证据释放格式化为控辩双方回复格式 |
| 2026-05-13 | 上下文长度控制：防止多轮对话导致 token 爆炸 |
| 2026-05-13 | 配置更新：新增 max_prompt_tokens、max_history_rounds 参数 |

---

## 相关文档

- [训练指南](training_guide.md) - 完整训练命令和配置
- [证据层级设计](crime_specific_evidence_design.md) - 罪名特异性触发词系统
- [SFT评估方案](sft_evaluation.md) - SFT阶段的评估指标和方法