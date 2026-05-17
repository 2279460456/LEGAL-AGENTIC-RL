# LEGAL-AGENTIC-RL 训练指南

> 最后更新: 2026-05-17

## 项目概述

本项目实现法律案件的多智能体强化学习系统，使用GRPO算法训练法官Agent。

**核心机制**：
- SFT阶段：让模型学会"像法律人一样说话"
- RL阶段：让模型学会"主动调查、策略决策"

**最新改进（2026-05-17）**：
- 奖励函数优化：移除步级奖励中的重复奖励/惩罚（B_trigger、P_repeat）
- 防重复提问机制：新增asked_questions记录和prompt中的已调查方向摘要

**最新改进（2026-05-16）**：
- 训练稳定性优化：weight_decay、cosine lr scheduler、paged_adamw_8bit
- 梯度控制优化：max_grad_norm=0.1、gradient_accumulation_steps=4
- 检测规则修复：放宽 invalid 检测，避免正常输出被误判

---

## 一、完整训练流程

### 流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    完整训练流程                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: SFT数据准备                                        │
│     python src/data_processing/generate_sft_data.py         │
│     → data/processed/ (控方/辩方/法官角色数据)               │
│                                                             │
│  Step 2: SFT训练                                            │
│     python -m src.sft.train_sft --config configs/sft_config.yaml │
│     → models/sft_checkpoint (LoRA adapters)                 │
│                                                             │
│  Step 3: RL数据构建                                         │
│     python src/data_processing/build_rl_data.py --use_llm   │
│     → data/rl_env/train_cases.json, test_cases.json         │
│                                                             │
│  Step 4: GRPO训练                                           │
│     python -m src.rl.train_rl --config configs/rl_config.yaml │
│     → models/rl_checkpoint                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、SFT阶段

### 2.1 数据准备

```bash
# 生成训练数据
python src/data_processing/generate_sft_data.py

# 查看样例数据
python src/data_processing/generate_sft_data.py --show-sample
```

数据保存到 `data/processed/`：
- `prosecutor.json` - 检察官角色数据
- `defender.json` - 辩护律师角色数据  
- `judge.json` - 法官角色数据
- `all_roles.json` - 合并数据

### 2.2 SFT训练

```bash
# 使用默认配置
python -m src.sft.train_sft --config configs/sft_config.yaml

# 从checkpoint继续训练
python -m src.sft.train_sft --config configs/sft_config.yaml \
    --resume models/sft_checkpoint/checkpoint-500

# 训练完成后合并LoRA（可选）
python -m src.sft.train_sft --config configs/sft_config.yaml \
    --merge --merge-output models/sft_merged
```

### 2.3 显存需求

| 方案 | 模型权重 | 总显存 | 适用显卡 |
|------|----------|--------|----------|
| 标准LoRA | ~16GB | >=20GB | RTX 3090/4090 |
| QLoRA | ~4GB | >=12GB | RTX 3060/4060/4070 |

---

## 三、RL数据构建

### 3.1 构建命令

```bash
# 使用LLM构建（推荐，证据拆分质量更高）
python src/data_processing/build_rl_data.py --use_llm

# 禁用断点续传（从头开始）
python src/data_processing/build_rl_data.py --use_llm --no_resume

# 不使用LLM（快速构建）
python src/data_processing/build_rl_data.py --max_samples 600
```

### 3.2 数据格式

```json
{
  "case_id": "xxx",
  "public_info": "被告人杨×于2014年以虚假票据骗取他人财物。",
  "hidden_evidence": {
    "subjective": {
      "content": "以玩牌还钱要套取现金为由...",
      "triggers": ["诈骗动机", "预谋", "虚构内容"],
      "role": "证明诈骗故意和获利动机"
    },
    "objective": {...},
    "sentencing": {...}
  },
  "ground_truth": {
    "crime": "票据诈骗罪",
    "sentence_months": 12,
    "laws": ["刑法第194条", "刑法第67条"]
  }
}
```

---

## 四、GRPO训练

### 4.1 训练命令

```bash
# 基本训练（必须先完成SFT）
python -m src.rl.train_rl --config configs/rl_config.yaml

# 指定SFT checkpoint路径
python -m src.rl.train_rl --config configs/rl_config.yaml \
    --sft_checkpoint models/sft_checkpoint

# 从RL checkpoint继续训练
python -m src.rl.train_rl --config configs/rl_config.yaml \
    --resume models/rl_checkpoint/checkpoint_episode_100

# 禁用4bit量化
python -m src.rl.train_rl --config configs/rl_config.yaml --no_quantization
```

### 4.2 GRPO配置说明（最新版）

`configs/rl_config.yaml`：

```yaml
# GRPO参数
grpo:
  group_size: 4            # 每组采样轨迹数
  temperature: 0.9         # 采样温度
  max_new_tokens: 128      # 每步最大生成token数
  max_prompt_length: 2048  # Prompt最大长度（NEW）
  max_completion_length: 512  # 生成最大长度（NEW）

# 训练参数（已优化）
training:
  learning_rate: 5e-6      # 更稳定（从2e-5降到5e-6）
  max_grad_norm: 0.1       # 更严格裁剪（从1.0降到0.1）
  gradient_accumulation_steps: 4  # 等效batch=4（NEW）
  weight_decay: 0.1        # 防止过拟合（NEW）
  lr_scheduler_type: "cosine"     # cosine衰减（NEW）
  warmup_ratio: 0.1        # warmup比例（NEW）
  optim: "paged_adamw_8bit"       # 节省显存（NEW）
  adam_beta1: 0.9          # Adam动量（NEW）
  adam_beta2: 0.99         # 比默认0.999更小（NEW）
  num_episodes: 2000
  save_total_limit: 2      # checkpoint数量限制
  save_interval: 200       # 保存间隔
```

### 4.3 参数优化对比

| 参数 | 旧值 | 新值 | 优化原因 |
|------|------|------|----------|
| `learning_rate` | 2e-5 | **5e-6** | RL微调更稳定，不破坏SFT知识 |
| `max_grad_norm` | 1.0 | **0.1** | GRPO梯度波动大，需要严格裁剪 |
| `weight_decay` | 无 | **0.1** | RL容易过拟合到特定reward pattern |
| `gradient_accumulation_steps` | 1 | **4** | 等效batch=4，梯度更平滑 |
| `lr_scheduler` | 无 | **cosine** | 平滑衰减，避免训练不稳定 |
| `warmup_ratio` | 无 | **0.1** | RL训练初期模型不稳定 |
| `optim` | AdamW | **paged_adamw_8bit** | 节省显存，防止OOM |
| `adam_beta2` | 0.999 | **0.99** | 适应RL的梯度波动特性 |

### 4.4 显存需求（24GB配置）

| 组件 | 显存占用 |
|------|----------|
| 模型(4bit) | ~4GB |
| KV Cache | ~4GB |
| 训练开销 | ~3GB |
| **总需求** | ~11GB |

---

## 五、Checkpoint管理

### 5.1 续训机制

```bash
# SFT续训
python -m src.sft.train_sft --resume models/sft_checkpoint/checkpoint-500

# RL续训（数据不会重头开始）
python -m src.rl.train_rl --resume models/rl_checkpoint/checkpoint_episode_200
```

### 5.2 Checkpoint限制

为节省磁盘空间，可配置保留数量：

```yaml
training:
  save_total_limit: 2  # 最多保留2个checkpoint
```

---

## 六、测试命令

```bash
# 测试RL训练流程（无需模型）
python scripts/test_rl_flow.py

# 测试触发词覆盖
python tests/test_crime_specific_evidence.py

# 测试模型加载（需要SFT checkpoint）
python src/rl/model_loader.py --lora_path models/sft_checkpoint --test
```

---

## 七、关键文件

| 文件 | 说明 |
|------|------|
| `src/sft/train_sft.py` | SFT训练入口 |
| `src/rl/train_rl.py` | GRPO训练入口 |
| `src/rl/grpo.py` | GRPO算法实现（含检测规则、scheduler） |
| `src/rl/environment.py` | RL环境（信息隐藏机制） |
| `src/rl/reward.py` | 奖励函数 |
| `src/rl/model_loader.py` | SFT模型加载 |
| `src/data_processing/build_rl_data.py` | RL数据构建 |
| `src/data_processing/llm_evidence_splitter.py` | LLM证据拆分 |
| `configs/rl_config.yaml` | RL配置文件（已更新） |

---

## 八、注意事项

1. **GRPO必须在SFT完成后运行**
2. **数据构建支持断点续传**
3. **触发词覆盖8大罪名类型、69个罪名映射**
4. **新配置已优化训练稳定性**：weight_decay、cosine lr scheduler、paged_adamw_8bit

---

## 更多文档

- [RL模块文档](RL_Module_Documentation.md) - GRPO算法详细说明（已更新）
- [实验设计详解](experiment_phases.md) - SFT和RL阶段的完整设计
- [证据层级设计](crime_specific_evidence_design.md) - 罪名特异性触发词系统
- [SFT评估方案](sft_evaluation.md) - SFT阶段的评估指标和方法

---

*文档创建时间: 2026-05-05*
*最后更新: 2026-05-16*