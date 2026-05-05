# SFT Training Guide

## 数据准备

训练前需要先生成SFT数据：

```bash
# 生成训练数据（从原始数据转换）
python src/data_processing/generate_sft_data.py

# 查看样例数据
python src/data_processing/generate_sft_data.py --show-sample
```

数据将保存到 `data/processed/` 目录：
- `prosecutor.json` - 检察官角色数据
- `defender.json` - 辩护律师角色数据
- `judge.json` - 法官角色数据
- `all_roles.json` - 合并数据

## 训练运行方式

### 基本训练

```bash
# 使用默认配置
python src/sft/train_sft.py

# 指定配置文件
python src/sft/train_sft.py --config configs/sft_config.yaml
```

### 从Checkpoint恢复

```bash
# 从指定checkpoint继续训练
python src/sft/train_sft.py --resume models/sft_checkpoint/checkpoint-500
```

### 完整参数

```bash
python src/sft/train_sft.py \
    --config configs/sft_config.yaml \
    --resume models/sft_checkpoint/checkpoint-500
```

## LoRA/QLoRA切换

修改 `configs/sft_config.yaml`：

### 方案一：标准LoRA（FP16，显存>=20GB）

```yaml
model:
  base_model: "Qwen/Qwen3-8B"
  # 无量化配置
```

### 方案二：QLoRA（4-bit，显存>=12GB）

```yaml
model:
  base_model: "Qwen/Qwen3-8B"
  use_4bit: true
  use_double_quant: true
  quant_type: "nf4"
```

切换方式：在配置文件中注释/取消注释相应配置块。

## 显存估算（Qwen3-8B）

| 方案 | 模型权重 | 总显存需求 | 适用显卡 |
|------|----------|------------|----------|
| 标准LoRA | ~16GB | >=20GB | RTX 3090/4090 |
| QLoRA | ~4GB | >=12GB | RTX 3060/4060/4070 |

## 输出文件

训练完成后：

- `models/sft_checkpoint/` - LoRA适配器权重

**注意**：LoRA合并步骤已跳过。RL训练可直接使用LoRA模型。

## 使用LoRA模型

加载SFT后的模型（用于推理或RL训练）：

```python
from src.rl.model_loader import load_sft_model

# 加载带LoRA的模型
model, tokenizer = load_sft_model(
    base_model_path="Qwen/Qwen3-8B",
    lora_path="models/sft_checkpoint"
)

# 直接用于推理或作为RL的policy model
```

## 测试模型

```bash
# 测试SFT模型生成能力
python scripts/test_sft_model.py
```