# LEGAL-AGENTIC-RL 训练指南

## 项目概述

本项目实现法律案件的多智能体强化学习系统，使用GRPO算法进行RL训练。

## 训练流程

### Step 1: 数据准备

```bash
# 构建RL训练数据（支持断点续传）
python src/data_processing/build_rl_data.py --use_llm

# 禁用断点续传（从头开始）
python src/data_processing/build_rl_data.py --use_llm --no_resume

# 调整增量保存间隔
python src/data_processing/build_rl_data.py --use_llm --save_interval 100
```

数据会保存到：
- `data/rl_env/train_cases.json` - 训练集
- `data/rl_env/test_cases.json` - 测试集

### Step 2: SFT训练（必须先完成）

```bash
# SFT训练
python -m src.sft.train_sft --config configs/sft_config.yaml

# 从checkpoint继续训练
python -m src.sft.train_sft --config configs/sft_config.yaml --resume models/sft_checkpoint/checkpoint-XXX

# 训练完成后合并LoRA（可选）
python -m src.sft.train_sft --config configs/sft_config.yaml --merge --merge-output models/sft_merged
```

SFT模型保存到：`models/sft_checkpoint` (LoRA adapters)

### Step 3: GRPO训练（基于SFT模型）

```bash
# GRPO训练（必须先完成SFT）
python -m src.rl.train_rl --config configs/rl_config.yaml

# 指定SFT checkpoint路径
python -m src.rl.train_rl --config configs/rl_config.yaml --sft_checkpoint models/sft_checkpoint

# 从RL checkpoint继续训练
python -m src.rl.train_rl --config configs/rl_config.yaml --resume models/rl_checkpoint/checkpoint_episode_100

# 禁用4bit量化（使用FP16）
python -m src.rl.train_rl --config configs/rl_config.yaml --no_quantization
```

GRPO模型保存到：`models/rl_checkpoint`

## 配置文件说明

### configs/sft_config.yaml
- 模型配置：基座模型、量化方式
- LoRA配置：rank、alpha、target_modules
- 训练配置：学习率、batch_size、epochs

### configs/rl_config.yaml
- GRPO配置：group_size、temperature、max_new_tokens
- 环境配置：evidence_levels、trigger_method、max_rounds
- 奖励配置：accuracy、information、compliance权重
- 模型配置：SFT checkpoint路径、LoRA参数

## 显存需求估算（24GB配置）

| 配置 | 模型 | KV Cache | 训练开销 | 总需求 |
|------|------|----------|----------|--------|
| 4bit + LoRA(r=64) | 4GB | 4GB | 1.3GB | ~11GB |

24GB显存配置：
- group_size: 8
- LoRA rank: 64
- 使用4bit量化

## 测试命令

```bash
# 测试RL训练流程（无需模型）
python scripts/test_rl_flow.py

# 测试触发词覆盖
python tests/test_crime_specific_evidence.py

# 测试模型加载（需要SFT checkpoint）
python src/rl/model_loader.py --lora_path models/sft_checkpoint --test
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `src/rl/grpo.py` | GRPO算法实现 |
| `src/rl/train_rl.py` | RL训练入口 |
| `src/rl/model_loader.py` | SFT模型加载 |
| `src/rl/environment.py` | RL环境（信息隐藏机制） |
| `src/rl/reward.py` | 奔励函数 |
| `src/data_processing/build_rl_data.py` | RL数据构建 |
| `src/data_processing/llm_evidence_splitter.py` | 证据拆分 |

## 注意事项

1. **GRPO必须在SFT完成后运行**：会自动加载SFT的LoRA adapters
2. **数据构建支持断点续传**：程序中断后重新运行会跳过已处理数据
3. **增量保存**：每50条数据保存临时文件，防止丢失
4. **触发词覆盖**：支持8大罪名类型，69个罪名映射