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
│   │   ├── agentscourt-data/  # SimuCourt数据集
│   │   │   ├── SimuCourt(1).json  # 一审案例
│   │   │   └── SimuCourt(2).json  # 二审案例
│   │   └── judge-data/        # Judge-Data数据集
│   │       ├── all.json       # 结构化判决数据
│   │       ├── train.json     # 训练集
│   │       └── test.json      # 测试集
│   ├── processed/             # SFT训练数据（生成后）
│   │   ├── prosecutor.json    # 检察官角色数据
│   │   ├── defender.json      # 辩护律师数据
│   │   ├── judge.json         # 法官角色数据
│   │   └── all_roles.json     # 合并数据
│   └── rl_env/                # RL环境数据
│
├── src/                       # 源代码
│   ├── data_processing/       # 数据处理模块
│   │   ├── sft_templates.py   # SFT数据模板定义
│   │   ├── generate_sft_data.py  # SFT数据生成脚本
│   │   ├── llm_evidence_splitter.py  # LLM证据拆分（罪名特异性）
│   │   └── build_rl_data.py   # GRPO训练数据构建
│   ├── sft/                   # SFT训练模块
│   │   └── train_sft.py       # LoRA/QLoRA训练实现
│   ├── rl/                    # RL训练模块
│   │   ├── environment.py     # 法律多智能体环境
│   │   ├── reward.py          # 多维奖励函数
│   │   ├── grpo.py            # GRPO算法实现
│   │   ├── train_rl.py        # RL训练入口
│   │   └── model_loader.py    # SFT模型加载器
│   └── evaluation/            # 评估模块
│       ├── metrics.py         # 评估指标计算
│       ├── sft_eval.py        # SFT评估脚本
│       └── sft_metrics/       # SFT评估指标包
│
├── configs/                   # 配置文件
│   ├── sft_config.yaml        # SFT配置（LoRA/QLoRA）
│   ├── rl_config.yaml         # RL配置
│   └── eval_config.yaml       # 评估配置
│
├── docs/                      # 文档目录
│   ├── experiment_phases.md   # 实验阶段设计（SFT+RL）
│   ├── crime_specific_evidence_design.md  # 罪名特异性证据层级
│   ├── sft_training_guide.md  # SFT训练指南
│   └── sft_evaluation.md      # SFT评估文档
│
├── scripts/                   # 运行脚本
│   ├── run_sft.sh             # SFT训练脚本
│   ├── run_rl.sh              # RL训练脚本
│   └── run_eval.sh            # 评估脚本
│
├── models/                    # 模型存储（训练后生成）
│   ├── sft_checkpoint/        # SFT检查点（LoRA适配器）
│   ├── sft_merged/            # 合并后的完整模型
│   └── rl_checkpoint/         # RL检查点
│
├── results/                   # 结果目录
│   ├── logs/                  # 训练日志
│   └── eval_results/          # 评估结果
│
├── pyproject.toml             # 项目依赖配置（uv）
└── README.md                  # 本文件
```

## 快速开始

### 1. 环境准备

本项目使用 [uv](https://docs.astral.sh/uv/) 进行依赖管理。

```bash
# 安装uv（如果未安装）
pip install uv

# PyTorch需要手动安装（根据CUDA版本选择）
# CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
uv sync
```

### 2. 数据准备

```bash
# 生成SFT训练数据（从原始判决书转换）
python src/data_processing/generate_sft_data.py

# 查看样例数据
python src/data_processing/generate_sft_data.py --show-sample
```

### 3. SFT训练

```bash
# 基本训练
python src/sft/train_sft.py --config configs/sft_config.yaml

# 从checkpoint恢复
python src/sft/train_sft.py --resume models/sft_checkpoint/checkpoint-500

# 训练后合并LoRA
python src/sft/train_sft.py --merge --merge-output models/sft_merged

# 或使用脚本
bash scripts/run_sft.sh
```

详细说明见 [SFT训练指南](docs/sft_training_guide.md)。

### 4. RL训练

```bash
bash scripts/run_rl.sh
```

### 5. 评估

```bash
bash scripts/run_eval.sh
```

## 技术配置

| 项目 | 配置 |
|-----|-----|
| 基座模型 | Qwen3-8B |
| 微调方法 | LoRA / QLoRA（可选） |
| RL算法 | GRPO (组相对策略优化) |
| 训练框架 | transformers + peft + trl |

### 显存需求（Qwen3-8B）

| 方案 | 显存需求 | 适用显卡 |
|------|----------|----------|
| 标准LoRA (FP16) | >=20GB | RTX 3090/4090 |
| QLoRA (4-bit) | >=12GB | RTX 3060/4060/4070 |

### LoRA/QLoRA切换

在 `configs/sft_config.yaml` 中修改：

```yaml
# 标准LoRA（注释量化配置）
model:
  base_model: "Qwen/Qwen3-8B"
  # 不使用量化

# QLoRA（启用量化）
model:
  base_model: "Qwen/Qwen3-8B"
  use_4bit: true
  use_double_quant: true
  quant_type: "nf4"
```

## 数据来源

- **SimuCourt(1)**: 一审刑事案例，包含控辩审三方意见
- **SimuCourt(2)**: 二审刑事案例，包含上诉辩护意见
- **judge-data**: 结构化判决数据，高质量法官视角

每个案例可生成多个角色的SFT数据点（检察官、辩护律师、法官）。

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