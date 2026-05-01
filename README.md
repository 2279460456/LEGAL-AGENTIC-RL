# LEGAL-AGENTIC-RL

基于多智能体强化学习的法律判决预测(LJP)系统

## 项目概述

本项目通过"信息隐藏-触发"机制模拟法律调查过程，实现动态的法律判决预测。

### 创新点

1. **信息隐藏-触发机制**：动态法律模拟器，建模事实发现过程
2. **结构化CoT驱动的角色行为模型**：从"文本生成"到"法律论证"
3. **多维协同奖励下的GRPO框架**：首次将GRPO应用于法律多智能体环境

## 项目结构

```
LEGAL-AGENTIC-RL/
├── data/                      # 数据目录
│   ├── raw/                   # 原始判决书数据
│   ├── processed/             # SFT训练数据
│   └── rl_env/                # RL环境数据
│
├── src/                       # 源代码
│   ├── data_processing/       # 数据处理模块
│   ├── sft/                   # SFT训练模块
│   ├── rl/                    # RL训练模块
│   └── evaluation/            # 评估模块
│
├── configs/                   # 配置文件
│   ├── sft_config.yaml        # SFT配置
│   ├── rl_config.yaml         # RL配置
│   └── eval_config.yaml       # 评估配置
│
├── scripts/                   # 运行脚本
│   ├── run_sft.sh             # SFT训练脚本
│   ├── run_rl.sh              # RL训练脚本
│   └── run_eval.sh            # 评估脚本
│
├── models/                    # 模型存储
│   ├── sft_checkpoint/        # SFT检查点
│   ├── rl_checkpoint/         # RL检查点
│   └── baseline/              # baseline模型
│
├── results/                   # 结果目录
│   ├── logs/                  # 训练日志
│   └── eval_results/          # 评估结果
│
├── requirements.txt           # 依赖文件
└── README.md                  # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 准备SFT数据
python -m src.data_processing.dataset_builder --output data/processed

# 准备RL环境数据
python -m src.data_processing.evidence_split --output data/rl_env
```

### 3. SFT训练

```bash
bash scripts/run_sft.sh
```

### 4. RL训练

```bash
bash scripts/run_rl.sh
```

### 5. 评估

```bash
bash scripts/run_eval.sh
```

## 核心模块说明

### 数据处理 (`src/data_processing/`)

- `cot_distill.py`: CoT思维链蒸馏
- `evidence_split.py`: 证据拆分（隐藏-触发机制）
- `dataset_builder.py`: 数据集构建

### RL训练 (`src/rl/`)

- `environment.py`: 法律多智能体环境
- `reward.py`: 多维奖励函数
- `grpo.py`: GRPO算法实现

### 评估 (`src/evaluation/`)

- `metrics.py`: 评估指标计算

## 技术配置

| 项目 | 配置 |
|-----|-----|
| 基座模型 | Qwen-7B |
| 微调方法 | QLoRA (4-bit量化) |
| RL算法 | GRPO (组相对策略优化) |
| GPU需求 | <24GB显存 |

## 实验流程

1. **阶段一**: 数据构建 + SFT行为克隆
2. **阶段二**: GRPO策略优化
3. **阶段三**: 评估与对比实验

## 对比实验

| Baseline | 对比目的 |
|---------|---------|
| 传统LJP模型 | 验证动态调查有效性 |
| 仅SFT模型 | 验证RL必要性 |
| 单智能体版本 | 验证多智能体优势 |

## 联系方式

项目开发中，欢迎讨论和反馈。

---

*创建时间: 2026-05-01*