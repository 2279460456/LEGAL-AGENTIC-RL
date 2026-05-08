"""
SFT Model Loader for RL Training
加载SFT后的模型（LoRA适配器）用于RL训练

使用方法:
    from src.rl.model_loader import load_sft_model

    model, tokenizer = load_sft_model(
        base_model="Qwen/Qwen3-8B",
        lora_path="models/sft_checkpoint"
    )
"""

import os
import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_sft_model(
    base_model_path: str = "Qwen/Qwen3-8B",
    lora_path: str = "models/sft_checkpoint",
    use_quantization: bool = True,
    device_map: str = "auto",
    max_memory: Optional[dict] = None,
    enable_training: bool = False  # 新增：是否启用训练模式
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载SFT后的模型（带LoRA适配器）

    Args:
        base_model_path: 基座模型路径
        lora_path: LoRA checkpoint路径
        use_quantization: 是否使用4bit量化（节省显存）
        device_map: 设备映射策略
        max_memory: 最大内存限制（如 {0: "20GB", "cpu": "30GB"}）
        enable_training: 是否启用训练模式（用于RL训练）

    Returns:
        (model, tokenizer) 元组
    """
    print("="*60)
    print("Loading SFT Model...")
    print("="*60)
    print(f"Base model: {base_model_path}")
    print(f"LoRA path: {lora_path}")
    print(f"Quantization: {use_quantization}")
    print(f"Enable training: {enable_training}")

    # 配置量化参数
    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 加载基座模型
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config if use_quantization else None,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not use_quantization else None
    )

    # 加载LoRA适配器
    print("\nLoading LoRA adapters...")

    if enable_training:
        # 训练模式：需要先准备基座模型，再加载LoRA
        if use_quantization:
            from peft import prepare_model_for_kbit_training
            print("Preparing model for k-bit training...")
            base_model = prepare_model_for_kbit_training(base_model)

        # 加载LoRA，设置为可训练
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=True  # 显式设置为可训练
        )

        # 强制解冻LoRA参数（某些peft版本需要手动解冻）
        print("Unfreezing LoRA parameters...")
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True

        # 尝试调用enable_adapter_layers（如果存在）
        if hasattr(model, 'enable_adapter_layers'):
            model.enable_adapter_layers()
    else:
        # 推理模式
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            is_trainable=False
        )

    # 验证可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("\nModel loaded successfully!")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Trainable params: {trainable_params} / {total_params} ({100*trainable_params/total_params:.4f}%)")

    if enable_training and trainable_params == 0:
        raise RuntimeError(
            "训练模式启用但可训练参数为0！这可能是peft版本问题。\n"
            "请尝试:\n"
            "  1. pip install peft>=0.5.0\n"
            "  2. 或手动设置: model.enable_adapter_layers()"
        )

    return model, tokenizer


def load_merged_model(
    merged_path: str = "models/sft_merged",
    use_quantization: bool = False,
    device_map: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载已合并的模型（如果存在）

    Args:
        merged_path: 合并后的模型路径
        use_quantization: 是否使用量化
        device_map: 设备映射

    Returns:
        (model, tokenizer) 元组
    """
    if not os.path.exists(merged_path):
        raise FileNotFoundError(f"Merged model not found at {merged_path}")

    print(f"Loading merged model from {merged_path}")

    tokenizer = AutoTokenizer.from_pretrained(merged_path)

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.float16 if not use_quantization else None
    )

    return model, tokenizer


def test_model_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str = None,
    max_new_tokens: int = 100
) -> str:
    """
    测试模型生成能力

    Args:
        model: 加载的模型
        tokenizer: tokenizer
        prompt: 测试prompt
        max_new_tokens: 最大生成token数

    Returns:
        生成的文本
    """
    if prompt is None:
        prompt = """你是一位资深法官，正在审理案件。

当前案情：2026年8月，被告人张三与被害人李四因琐事发生冲突，造成李四受伤。

请分析案情并给出判决思路。"""

    print("\n" + "="*60)
    print("Testing Model Generation...")
    print("="*60)
    print(f"Prompt: {prompt[:100]}...")

    # 构建输入
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 生成
    print("\nGenerating...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    # 解码
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"\nGenerated text:\n{generated_text}")

    return generated_text


def get_model_info(model) -> dict:
    """
    获取模型信息

    Args:
        model: 加载的模型

    Returns:
        模型信息字典
    """
    info = {
        "model_type": type(model).__name__,
        "is_peft_model": isinstance(model, PeftModel),
        "device": str(next(model.parameters()).device),
        "dtype": str(next(model.parameters()).dtype),
    }

    if isinstance(model, PeftModel):
        info["peft_config"] = str(model.peft_config)
        info["active_adapter"] = model.active_adapter

    # 计算参数数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    info["trainable_params"] = trainable_params
    info["total_params"] = total_params
    info["trainable_ratio"] = trainable_params / total_params

    return info


def main():
    """测试模型加载"""
    import argparse

    parser = argparse.ArgumentParser(description="Load and test SFT model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model path"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="models/sft_checkpoint",
        help="LoRA checkpoint path"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 4bit quantization"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run generation test"
    )

    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_sft_model(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        use_quantization=not args.no_quantization
    )

    # 显示模型信息
    info = get_model_info(model)
    print("\n" + "="*60)
    print("Model Information")
    print("="*60)
    for key, value in info.items():
        print(f"{key}: {value}")

    # 测试生成
    if args.test:
        test_model_generation(model, tokenizer)


if __name__ == "__main__":
    main()