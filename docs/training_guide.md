# LEGAL-AGENTIC-RL 训练指南

> 最后更新: 2026-05-10

## 项目概述

本项目实现法律案件的多智能体强化学习系统，使用GRPO算法训练法官Agent。

**核心机制**：
- SFT阶段：让模型学会"像法律人一样说话"
- RL阶段：让模型学会"主动调查、策略决策"

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

### 2.3 SFT配置说明

修改 `configs/sft_config.yaml`：

```yaml
# 方案一：标准LoRA（FP16，显存>=20GB）
model:
  base_model: "Qwen/Qwen3-8B"
  use_4bit: false  # 关闭量化

# 方案二：QLoRA（4-bit，显存>=12GB）
model:
  base_model: "Qwen/Qwen3-8B"
  use_4bit: true
  bnb_4bit_quant_type: "nf4"
```

### 2.4 显存需求

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

# 调整保存间隔
python src/data_processing/build_rl_data.py --use_llm --save_interval 100

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

### 4.2 GRPO配置说明

`configs/rl_config.yaml`：

```yaml
grpo:
  group_size: 4        # 每组采样轨迹数
  temperature: 1.0     # 采样温度
  max_new_tokens: 128  # 每步最大生成token数

training:
  learning_rate: 2e-5
  num_episodes: 2000
  save_total_limit: 2  # checkpoint数量限制
  save_interval: 200   # 保存间隔
```

### 4.3 显存需求（24GB配置）

| 组件 | 显存占用 |
|------|----------|
| 模型(4bit) | ~4GB |
| KV Cache | ~4GB |
| 训练开销 | ~3GB |
| **总需求** | ~11GB |

---

## 五、Checkpoint管理

### 5.1 续训机制

训练会自动保存checkpoint，支持续训：

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

| save_total_limit | 磁盘占用 |
|------------------|----------|
| 1 | ~320MB |
| 2 | ~640MB |
| 3 | ~960MB |

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
| `src/rl/grpo.py` | GRPO算法实现 |
| `src/rl/environment.py` | RL环境（信息隐藏机制） |
| `src/rl/reward.py` | 奖励函数 |
| `src/rl/model_loader.py` | SFT模型加载 |
| `src/data_processing/build_rl_data.py` | RL数据构建 |
| `src/data_processing/llm_evidence_splitter.py` | LLM证据拆分 |

---

## 八、注意事项

1. **GRPO必须在SFT完成后运行**
2. **数据构建支持断点续传**
3. **触发词覆盖8大罪名类型、69个罪名映射**
4. **Gradient checkpointing已禁用以加速生成**

---

## 更多文档

- [实验设计详解](experiment_phases.md) - SFT和RL阶段的完整设计
- [证据层级设计](crime_specific_evidence_design.md) - 罪名特异性触发词系统
- [SFT评估方案](sft_evaluation.md) - SFT阶段的评估指标和方法