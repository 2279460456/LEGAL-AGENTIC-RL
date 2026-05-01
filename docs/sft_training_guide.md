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

### 训练后合并LoRA

```bash
# 训练完成后将LoRA合并到基础模型
python src/sft/train_sft.py --merge --merge-output models/sft_merged
```

### 完整参数

```bash
python src/sft/train_sft.py \
    --config configs/sft_config.yaml \
    --resume models/sft_checkpoint/checkpoint-500 \
    --merge \
    --merge-output models/sft_merged
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

- `models/sft_checkpoint/` - LoRA适配器
- `models/sft_merged/` - 合并后的完整模型（可选）

## 推理测试

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载LoRA模型
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
model = PeftModel.from_pretrained(base, "models/sft_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 或加载合并后的模型
model = AutoModelForCausalLM.from_pretrained("models/sft_merged")
tokenizer = AutoTokenizer.from_pretrained("models/sft_merged")
```