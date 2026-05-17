# LEGAL-AGENTIC-RL 强化学习模块文档

## 概述

本项目使用基于 GRPO (Group Relative Policy Optimization) 的强化学习框架来训练法律判决 Agent。基于 SFT 模型进行策略优化，让模型学会"调查性思维"——在不确定案情中主动提问获取证据，然后做出准确判决。

**核心改进（2026-05-16）**：
- **训练稳定性优化**：新增 weight_decay、cosine lr scheduler、paged_adamw_8bit 优化器
- **梯度控制优化**：max_grad_norm 从 1.0 降到 0.1，gradient_accumulation_steps 增加到 4
- **检测规则修复**：放宽 invalid 检测规则，避免正常输出被误判
- **对话交互设计**：证据释放以控辩双方回复格式呈现
- **上下文长度控制**：防止多轮对话导致 token 爆炸

---

## 1. 强化学习策略 (GRPO算法)

### 1.1 算法架构

项目采用 **GRPO算法** 作为核心强化学习框架，这是PPO的简化版本，不需要Critic网络。

**核心文件路径：**
- `src/rl/environment.py` - RL环境（信息隐藏机制）
- `src/rl/grpo.py` - GRPO训练器实现
- `src/rl/train_rl.py` - RL训练入口

### 1.2 GRPO优势计算

**GRPO公式**：
```
A_i = (R_i - mean(R_group)) / std(R_group)
Loss = -Σ A_i · Σ log_prob(action)
```

相比PPO+GAE的优势：
- 不需要Critic网络，节省计算资源
- 使用组内相对比较作为baseline
- 适合outcome-only reward场景

---

## 2. 训练稳定性优化（2026-05-16新增）

### 2.1 优化器配置

```python
# GRPOConfig 新增参数
weight_decay: float = 0.1       # 防止过拟合
adam_beta1: float = 0.9         # Adam动量参数
adam_beta2: float = 0.99        # 比默认0.999更小，适应RL梯度波动
optim: str = "paged_adamw_8bit" # 节省显存的优化器
gradient_accumulation_steps: int = 4  # 等效batch_size=4，梯度更平滑
```

**优化器选择**：
```python
if config.optim == "paged_adamw_8bit":
    import bitsandbytes as bnb
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
else:
    optimizer = AdamW(trainable_params, ...)
```

### 2.2 学习率调度

```python
# GRPOConfig 新增参数
warmup_ratio: float = 0.1       # warmup比例
lr_scheduler_type: str = "cosine"  # cosine衰减

# Scheduler初始化
warmup_steps = int(num_training_steps * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)
```

### 2.3 梯度控制

```python
# GRPOConfig
max_grad_norm: float = 0.1  # 更严格的梯度裁剪（从1.0降到0.1）

# 梯度裁剪 + scheduler更新
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
optimizer.step()
scheduler.step()  # 更新学习率
```

### 2.4 参数对比（优化前后）

| 参数 | 优化前 | 优化后 | 说明 |
|------|--------|--------|------|
| `learning_rate` | 2e-5 | **5e-6** | 更稳定，不破坏SFT知识 |
| `max_grad_norm` | 1.0 | **0.1** | 更严格裁剪，防止梯度爆炸 |
| `weight_decay` | 无 | **0.1** | 防止RL过拟合 |
| `gradient_accumulation_steps` | 1 | **4** | 等效batch=4，梯度平滑 |
| `lr_scheduler` | 无 | **cosine** | 平滑衰减lr |
| `warmup_ratio` | 无 | **0.1** | RL训练初期需要warmup |
| `optim` | AdamW | **paged_adamw_8bit** | 节省显存 |
| `adam_beta2` | 0.999 | **0.99** | 适应RL梯度波动 |

---

## 3. 环境设计（信息隐藏-触发机制）

### 3.1 EvidenceEnvironment类

**文件路径：** `src/rl/environment.py`

**核心机制**：
- 初始状态：只有模糊案情（public_info）
- 隐藏证据：三层证据（主观层、客观层、量刑层）
- 触发方式：关键词匹配 → 释放证据

### 3.2 证据触发机制（对话回复格式）

**证据层级与控辩对应关系**：

| 层级 | 触发词示例 | 回复角色 |
|------|-----------|---------|
| **主观层** | "动机"、"预谋"、"事前" | **公诉人**（定罪要素） |
| **客观层** | "伤害部位"、"手段"、"伤口" | **公诉人**（定罪要素） |
| **量刑层** | "自首"、"赔偿"、"认罪" | **辩护人**（从轻情节） |

---

## 4. Reward设计（v5版本）

### 4.1 Reward组成

```python
R_total = 0.3·R_accuracy + 0.4·R_info + R_process

# 组成：
# - 30% 准确性reward（混合相似度）
# - 40% 信息收集reward（调查深度）
# - 30% 过程reward/惩罚
```

### 4.2 过程Reward详细设计

| 情况 | Reward | 说明 |
|------|--------|------|
| 第1轮判决且无调查 | **-0.5** | 早判重惩罚 |
| 证据不足就判决 | -0.2×(1-发现率) | 过早判决惩罚 |
| 罪名缺失 | -0.2 | 格式检查 |
| 刑期缺失 | -0.1 | 格式检查 |
| 3轮内且有调查 | +0.1 | 效率奖励 |
| 达到最大轮次 | -0.2 | 轮次惩罚 |

### 4.3 步级Reward（精简版）

```python
R_step = P_irrelevant  # 仅保留不相关问题惩罚
```

| 情况 | Reward | 说明 |
|------|--------|------|
| 提问无法律关键词 | -0.02 | 不相关问题惩罚 |

**设计说明**：
- **证据触发奖励**已统一在R_info（信息收集奖励）中计算，避免重复
- **重复提问惩罚**已统一在无效输出处理机制中处理，避免重复惩罚

### 4.4 防重复提问机制

为避免上下文丢失导致模型重复提问，系统实现了以下机制：

```python
# environment.py
class EvidenceEnvironment:
    _asked_questions: List[str] = []  # 记录已问问题

    def get_asked_questions(self) -> List[str]:
        return self._asked_questions.copy()

# grpo.py _build_prompt()
asked_questions = state.get('asked_questions', [])
if asked_questions:
    # 提取关键词显示已调查方向
    asked_section = "\n已调查方向：" + "、".join(keywords)
```

---

## 5. 无效动作检测（2026-05-16修复）

### 5.1 检测规则优化

**问题**：原检测规则过于严格，正常输出被误判为 invalid。

**修复内容**：

| 函数 | 修复内容 |
|------|----------|
| `_has_template_residue` | 精简关键词列表，移除"至少"、"例如"等常见词 |
| `_has_repetition` | 移除子串检测，提高阈值（20字、15字短语） |
| `_is_query_semantic` | 扩展关键词：新增"介绍"、"说明"、"描述"、"涉案金额"等 |

### 5.2 Invalid惩罚机制

```python
# 3轮invalid后强制终止
if invalid_count >= 3:
    return StepResult(
        reward=repetition_penalty - 0.3,  # 额外惩罚
        is_terminal=True
    )
```

---

## 6. 上下文长度控制

### 6.1 控制策略

```python
# GRPOConfig 参数
max_prompt_length: int = 2048       # Prompt最大长度
max_completion_length: int = 512    # 生成最大长度
max_history_rounds: int = 3         # 保留最近N轮对话
max_evidence_preview: int = 50      # 每个证据预览最大字数
```

---

## 7. GRPO训练配置

**配置文件**：`configs/rl_config.yaml`

```yaml
# GRPO Configuration
grpo:
  group_size: 4
  temperature: 0.9
  max_new_tokens: 128
  max_prompt_length: 2048       # NEW
  max_completion_length: 512    # NEW

# Training Configuration  
training:
  learning_rate: 5e-6           # MODIFY: 更稳定
  max_grad_norm: 0.1            # MODIFY: 更严格裁剪
  gradient_accumulation_steps: 4  # NEW
  weight_decay: 0.1             # NEW
  lr_scheduler_type: "cosine"   # NEW
  warmup_ratio: 0.1             # NEW
  optim: "paged_adamw_8bit"     # NEW
  adam_beta1: 0.9               # NEW
  adam_beta2: 0.99              # NEW
  num_episodes: 2000
  save_interval: 200
```

---

## 8. 关键文件位置

```
src/rl/
├── environment.py     # RL环境（信息隐藏-触发机制）
│   ├── _asked_questions      # NEW: 已问问题记录（防重复）
│   ├── _calculate_step_reward()  # MODIFY: 移除重复奖励
│   └── get_asked_questions()  # NEW: 获取已问问题
│   └── AgentState.asked_questions  # NEW: 状态包含已问问题
├── grpo.py            # GRPO训练器（含检测规则）
│   ├── _has_template_residue()  # 模板残留检测（已修复）
│   ├── _has_repetition()        # 重复检测（已修复）
│   ├── _is_query_semantic()     # 提问识别（已修复）
│   ├── _build_prompt()          # MODIFY: 显示已调查方向摘要
│   ├── _state_to_dict()         # MODIFY: 传递asked_questions
│   └── setup_scheduler()        # NEW: 学习率调度器
└── train_rl.py        # RL训练入口

configs/
└── rl_config.yaml     # RL训练配置（已更新）
```

---

## 9. 运行命令

```bash
# SFT训练（预训练）
python -m src.sft.train_sft --config configs/sft_config.yaml

# RL训练（策略优化）
python -m src.rl.train_rl --config configs/rl_config.yaml

# 测试验证
python scripts/test_rl_flow.py
```

---

*文档创建时间: 2026-05-13*
*最后更新: 2026-05-17*
*状态: LEGAL-AGENTIC-RL项目RL模块文档（奖励函数优化版本）*

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-05-17 | 奖励函数优化：移除步级奖励中的B_trigger和P_repeat，避免重复奖励/惩罚 |
| 2026-05-17 | 防重复提问机制：新增_asked_questions记录和prompt中的已调查方向摘要 |
| 2026-05-16 | 训练稳定性优化：weight_decay、cosine lr scheduler、paged_adamw_8bit |
| 2026-05-16 | 检测规则修复：放宽 invalid 检测，避免正常输出被误判 |
| 2026-05-16 | 参数调整：learning_rate 5e-6、max_grad_norm 0.1、gradient_accumulation_steps 4 |
| 2026-05-13 | 对话交互设计：证据释放格式化为控辩双方回复 |
| 2026-05-13 | 上下文长度控制：max_prompt_tokens、max_history_rounds |