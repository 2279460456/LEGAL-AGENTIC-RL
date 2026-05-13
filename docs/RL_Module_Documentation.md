# LEGAL-AGENTIC-RL 强化学习模块文档

## 概述

本项目使用基于 GRPO (Group Relative Policy Optimization) 的强化学习框架来训练法律判决 Agent。基于 SFT 模型进行策略优化，让模型学会"调查性思维"——在不确定案情中主动提问获取证据，然后做出准确判决。

**核心改进（2026-05-13）**：
- **对话交互设计**：证据释放以控辩双方回复格式呈现，法官提问有对话上下文
- **上下文长度控制**：防止多轮对话导致 token 爆炸，Qwen3-8B 训练时控制在 1200 tokens
- **证据层级与控辩对应**：主观层/客观层 → 公诉人回复，量刑层 → 辽护人回复

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

**与GAE对比**：

| 特性 | GAE (γ=1, λ=1) | GRPO |
|------|----------------|------|
| 需要Critic | 是 | 否 |
| Advantage计算 | R - V(s_t) | R - mean(R_group) |
| 每token的A不同 | 是（取决于V(s_t)） | 否（所有token相同） |
| 理论基础 | TD学习，降低方差 | 组相对比较 |

### 1.3 策略更新流程

```python
def train_step(self, case_data):
    # 1. 生成G条轨迹（group_size=4）
    trajectories = self._generate_trajectories(case_data, G)
    
    # 2. 计算每条轨迹的total_reward
    rewards = [traj.total_reward for traj in trajectories]
    
    # 3. 计算advantage（组标准化）
    mean_r = np.mean(rewards)
    std_r = np.std(rewards)
    if std_r == 0:
        advantages = [0.0] * len(rewards)  # 无区分度
    else:
        advantages = [(r - mean_r) / std_r for r in rewards]
    
    # 4. 计算GRPO loss
    loss = 0
    for traj, adv in zip(trajectories, advantages):
        traj_log_prob = self._compute_trajectory_log_prob_with_grad(traj)
        loss -= adv * traj_log_prob
    loss /= len(trajectories)
    
    # 5. 反向传播更新
    loss.backward()
    self.optimizer.step()
```

---

## 2. 环境设计（信息隐藏-触发机制）

### 2.1 EvidenceEnvironment类

**文件路径：** `src/rl/environment.py`

**核心机制**：
- 初始状态：只有模糊案情（public_info）
- 隐藏证据：三层证据（主观层、客观层、量刑层）
- 触发方式：关键词匹配 → 释放证据

```python
class EvidenceEnvironment:
    def __init__(self, trigger_method="keyword", max_rounds=10):
        self.hidden_evidence = {}  # level -> evidence dict
        self.revealed_levels = set()  # 已揭示层级
        self.final_prediction = None
    
    def step(self, action):
        if action["type"] == "query":
            # 检查是否触发隐藏证据
            new_evidence = self._check_trigger(action["content"])
            return StepResult(reward=step_reward, new_evidence=new_evidence)
        
        elif action["type"] == "judge":
            # 第1轮禁止判决规则
            if self.current_round == 0:
                return StepResult(reward=-0.3, is_terminal=False)
            self.final_prediction = action["content"]
            return StepResult(is_terminal=True)
```

### 2.2 第1轮禁止判决规则

```python
def _handle_judge(self, judgment):
    # 【关键规则】第1轮禁止判决，强制先调查
    if self.current_round == 0:
        self.current_round += 1
        self.conversation_history.append(("judge", "[拒绝判决-第1轮]"))
        return StepResult(
            reward=-0.3,  # 早判惩罚
            is_terminal=False,  # 不终止，强制继续
            info={"rejected": True, "reason": "第1轮禁止判决"}
        )
```

### 2.3 证据触发机制（对话回复格式）

**改进说明**：证据释放以对话回复格式呈现，模拟控辩双方回答法官提问。

```python
def _check_trigger(self, query):
    for level, evidence in self.hidden_evidence.items():
        if level not in self.revealed_levels:
            triggers = evidence.get("triggers", [])
            if self._match_triggers(query, triggers):
                self.revealed_levels.add(level)
                raw_content = evidence.get("content")
                # 格式化为对话回复
                return self._format_evidence_as_dialogue(level, raw_content)
    return None

def _format_evidence_as_dialogue(self, level, content):
    # 主观层/客观层 → 控方回复（定罪要素）
    if level == "subjective":
        return f"公诉人补充说明：经调查，{content}"
    elif level == "objective":
        return f"公诉人补充说明：关于作案情况，{content}"
    # 量刑层 → 辽方回复（从轻情节）
    elif level == "sentencing":
        return f"辩护人补充说明：{content}"
```

**证据层级与控辩对应关系**：

| 层级 | 触发词示例 | 证据内容示例 | 回复角色 |
|------|-----------|-------------|---------|
| **主观层** | "动机"、"预谋"、"事前" | 作案动机、预谋证据 | **公诉人**（定罪要素） |
| **客观层** | "伤害部位"、"手段"、"伤口" | 作案手段、后果证据 | **公诉人**（定罪要素） |
| **量刑层** | "自首"、"赔偿"、"认罪" | 案后表现、减轻情节 | **辩护人**（从轻情节） |

**交互示例**：

```
法官提问："我想了解被告人的作案动机"
→ 触发主观层证据
→ 回复："公诉人补充说明：经调查，被告人案发前购买了折叠刀并记录被害人行踪"

法官提问："是否有自首情节"
→ 触发量刑层证据
→ 回复："辩护人补充说明：被告人案发后主动投案并如实供述犯罪事实"
```

**设计说明**：当前阶段不区分"向控方提问"vs"向辩方提问"，证据层级隐式对应控辩双方。后续可扩展为更精细的庭审交互设计。

---

## 3. Reward设计（v5版本）

### 3.1 Reward组成

```python
def get_final_reward(self):
    """
    R_total = 0.3·R_accuracy + 0.4·R_info + R_process
    
    组成：
    - 30% 准确性reward（混合相似度）
    - 40% 信息收集reward（调查深度）
    - 30% 过程reward/惩罚
    """
```

### 3.2 混合相似度计算

**各组件计算方式**：

| 组件 | 计算方式 | 原因 |
|------|----------|------|
| 罪名 | ROUGE F1 | 文本相似度，如"故意伤害"vs"故意伤害致死" |
| 刑期 | 相对误差 | 数值型，1 - |pred-true|/true |
| 法条 | 集合召回率 | 列表匹配，只有完全匹配才算 |

```python
def _calculate_accuracy_reward(self, prediction, ground_truth):
    # 1. 罪名相似度（ROUGE）
    crime_sim = compute_rouge_similarity(
        prediction.get("crime", ""),
        ground_truth.get("crime", "")
    )
    crime_reward = 0.4 * crime_sim
    
    # 2. 刑期准确度（相对误差）
    if true_months > 0:
        sentence_error = abs(pred_months - true_months) / true_months
        sentence_reward = 0.3 * (1 - min(sentence_error, 1))
    
    # 3. 法条召回率（集合匹配）
    law_recall = len(pred_laws & true_laws) / len(true_laws)
    law_reward = 0.3 * law_recall
    
    return crime_reward + sentence_reward + law_reward
```

### 3.3 ROUGE相似度函数

```python
def compute_rouge_similarity(text1, text2):
    """
    基于F1分数的ROUGE相似度计算
    
    示例：
    - "故意伤害罪" vs "故意伤害罪" → 1.0
    - "故意伤害" vs "故意伤害致死" → 0.71
    - "盗窃罪" vs "抢劫罪" → 0.17
    """
    try:
        import jieba
        words1 = jieba.lcut(text1)
        words2 = jieba.lcut(text2)
    except ImportError:
        # Fallback: 字符分割
        words1 = [text1[i:i+l] for i in range(len(text1)) for l in [4,3,2,1]]
        words2 = [text2[i:i+l] for i in range(len(text2)) for l in [4,3,2,1]]
    
    count1 = Counter(words1)
    count2 = Counter(words2)
    common = count1 & count2
    
    precision = sum(common.values()) / len(words1)
    recall = sum(common.values()) / len(words2)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1
```

### 3.4 过程Reward详细设计

```python
process_reward = 0.0

# 早判惩罚（第1轮判决且无调查）
if current_round == 1 and discovered_levels == 0:
    process_reward -= 0.5

# 过早判决惩罚（证据不足）
elif discovered_ratio < 0.5:
    process_reward -= 0.2 * (1 - discovered_ratio)

# 格式检查
if not prediction.get("crime") or len(crime) < 2:
    process_reward -= 0.2  # 罪名缺失

if prediction.get("sentence_months", 0) <= 0:
    process_reward -= 0.1  # 刑期缺失

# 效率奖励
if current_round <= 3 and discovered_levels >= 1:
    process_reward += 0.1

# 最大轮次惩罚
if current_round >= max_rounds:
    process_reward -= 0.2
```

---

## 4. 动作追踪机制

### 4.1 追踪变量

```python
# 动作追踪（用于人工审核和训练分析）
self._all_actions = []          # 所有动作历史
self._valid_actions = []        # 有效格式动作
self._effective_actions = []    # 触发证据的动作
self._invalid_actions = []      # 无效动作记录
```

### 4.2 追踪方法

```python
def _track_action(self, action, result):
    record = {
        "round": self.current_round,
        "type": action.get("type"),
        "content_preview": str(action.get("content", ""))[:50],
        "reward": result.reward,
        "triggered": result.new_evidence is not None,
        "is_valid": not result.info.get("invalid", False),
        "reason": result.info.get("reason", "")
    }
    
    self._all_actions.append(record)
    if record["is_valid"]:
        self._valid_actions.append(record)
        if record["triggered"]:
            self._effective_actions.append(record)
    else:
        self._invalid_actions.append(record)
```

### 4.3 统计报告

```python
def get_action_statistics(self):
    """生成动作统计报告"""
    return {
        "total_actions": len(self._all_actions),
        "valid_rate": len(self._valid_actions) / max(total, 1),
        "effective_rate": len(self._effective_actions) / max(total, 1),
        "invalid_details": self._invalid_actions,  # 用于人工审核
        "evidence_discovered": list(self.revealed_levels),
        "final_reward": self.get_final_reward()
    }
```

---

## 5. 无效动作处理

### 5.1 无效格式检测

```python
def _parse_action(self, action_text):
    # 检测prompt模板残留
    invalid_patterns = [
        "提问：... 或 判决：",
        "你的决定",
        "【"  # 模板格式符号
    ]
    
    # 检测必须以"提问"或"判决"开头
    if not action_text.startswith("提问") and not action_text.startswith("判决"):
        return {"type": "invalid", "repetition_penalty": -0.15}
    
    # 检测重复输出
    if self._has_repetition(action_text):
        return {"type": "invalid", "repetition_penalty": -0.3}
```

### 5.2 无效动作处理

```python
def _handle_invalid(self, content, repetition_penalty):
    self.current_round += 1
    reward = repetition_penalty  # 如 -0.2
    
    # 连续无效输出超过3次 → 强制终止
    invalid_count = sum(1 for msg in self.conversation_history if "无效输出" in msg)
    if invalid_count >= 3:
        self.final_prediction = {"crime": "", "sentence_months": 0}
        return StepResult(
            reward=reward - 0.3,  # 额外惩罚
            is_terminal=True
        )
    
    return StepResult(reward=reward, is_terminal=False)
```

---

## 6. 上下文长度控制

### 6.1 问题背景

**Qwen3-8B 的限制**：
- 最大上下文长度：约8192 tokens（但训练时建议控制在2048-4096）
- 每轮对话增长：约200-400 tokens（法官提问+控辩回复）
- 10轮对话可能导致上下文爆炸 → 显存不足、模型崩溃

### 6.2 控制策略

```python
# GRPOConfig 新增参数
max_prompt_tokens: int = 1200    # Prompt最大token数
max_history_rounds: int = 3      # 保留最近N轮对话
max_evidence_preview: int = 50   # 每个证据预览最大字数
```

**三层截断机制**：

| 层级 | 控制方式 | 说明 |
|------|----------|------|
| **对话历史** | 保留最近3轮 | 每轮约200-400 tokens → 总约600-1200 |
| **证据内容** | 截断到50字 | 避免单个证据过长 |
| **总长度** | 预估+压缩 | 超过1200 tokens时自动压缩 |

### 6.3 Prompt构建示例

```python
def _build_prompt(self, state: Dict) -> str:
    # 1. 截断对话历史
    recent_history = conversation_history[-(max_history_rounds * 2):]

    # 2. 压缩历史为摘要
    for role, msg in recent_history:
        msg_preview = msg[:50] + "..." if len(msg) > 50 else msg
        history_items.append(f"[{role}]: {msg_preview}")

    # 3. 压缩证据内容
    evidence_summary = [ev[:50] + "..." for ev in revealed[-2:]]

    # 4. 预估token数并压缩
    estimated_tokens = len(prompt) / 1.5
    if estimated_tokens > max_prompt_tokens:
        prompt = self._compress_prompt_further(prompt)
```

**实际Prompt示例**（约800 tokens）：

```
案情摘要：被告人杨×以虚假票据骗取他人财物...

近期对话：
[法官]: 我想了解作案动机
[公诉人]: 公诉人补充说明：经调查，被告人预谋作案...
[法官]: 是否有自首情节
[辩护人]: 辩护人补充说明：被告人有自首情节...

已获取证据：
公诉人补充说明：经调查，被告人预谋作案...
辩护人补充说明：被告人有自首情节...

【可以判决】
如果证据充分，可给出判决。
输出格式：判决：罪名：XXX，刑期：XXX个月

直接输出（不要解释）：
```

---

## 7. GRPO训练配置

**配置文件**：`configs/rl_config.yaml`

```yaml
GRPO参数:
  group_size: 4          # 每组采样4条轨迹
  temperature: 0.7       # 降低重复概率
  max_turns: 10          # 最大对话轮次
  max_new_tokens: 128    # 最大生成token数

  # 上下文控制（新增）
  max_prompt_tokens: 1200      # Prompt最大token数
  max_history_rounds: 3        # 保留最近N轮对话
  max_evidence_preview: 50     # 每个证据预览最大字数

Reward权重:
  - accuracy: 30%
  - information: 40%
  - process: 30%

训练参数:
  learning_rate: 2e-5
  num_episodes: 2000
  eval_interval: 100
  save_interval: 200

模型参数:
  base_model: Qwen/Qwen3-8B
  sft_checkpoint: models/sft_checkpoint
  lora_rank: 64
  use_4bit: true         # QLoRA量化
```

---

## 8. 训练流程

```
For each episode:
    1. 【状态初始化】
       env.reset(case_data)
       state = {public_info, revealed_evidence=[], current_round=0}
       
    2. 【组采样】生成G=4条轨迹
       for i in range(group_size):
           while not env.is_terminal():
               action, log_prob = model.sample(state)
               result = env.step(action)
               record(state, action, log_prob, result.reward)
       
    3. 【最终奖励计算】
       final_reward = env.get_final_reward()
       total_reward = final_reward + sum(step_rewards)
       
    4. 【优势值计算】
       advantages = (rewards - mean) / std
       
    5. 【梯度更新】
       Loss = -Σ A_i · Σ log_prob(action)
       optimizer.step()
```

---

## 9. Reward区分度测试

**测试结果**（v5版本）：

| 调查程度 | Reward |
|---------|--------|
| 不调查(0层) | 0.10 |
| 调查1层 | 0.44 |
| 调查3层 | 0.82 |
| **区分度** | **0.72** |

**关键改进**：
- 调查深度奖励：每层+0.1
- 早判惩罚：第1轮判决 → -0.5
- ROUGE相似度：罪名评估更合理

---

## 10. 关键文件位置

```
src/rl/
├── environment.py     # RL环境（信息隐藏-触发机制）
│   ├── EvidenceEnvironment类
│   ├── compute_rouge_similarity()
│   ├── get_final_reward() - v5版本
│   ├── _track_action()
│   └── get_action_statistics()
│
├── grpo.py            # GRPO训练器
│   ├── GRPOTrainer类
│   ├── _parse_action() - 无效格式检测
│   ├── _build_prompt() - 简化prompt
│   └── _compute_grpo_loss()
│
└── train_rl.py        # RL训练入口

configs/
└── rl_config.yaml     # RL训练配置

docs/
├── RL_Module_Documentation.md  # 本文档
└── experiment_phases.md        # 实验阶段设计
```

---

## 11. 运行命令

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
*最后更新: 2026-05-13*
*状态: LEGAL-AGENTIC-RL项目RL模块文档（对话交互 + 上下文控制版本）*

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-05-13 | 添加对话交互设计：证据释放格式化为控辩双方回复 |
| 2026-05-13 | 添加上下文长度控制：max_prompt_tokens=1200, max_history_rounds=3 |
| 2026-05-13 | 对话历史记录控辩回复：conversation_history 包含完整交互 |
| 2026-05-13 | 更新配置文件和训练入口：新增 GRPOConfig 上下文参数 |