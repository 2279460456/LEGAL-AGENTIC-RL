"""
SFT模型测试脚本
验证SFT后的模型是否能正常输出法律问答

使用方法:
    python scripts/test_sft_model.py --sft_checkpoint models/sft_checkpoint
    python scripts/test_sft_model.py --sft_checkpoint models/sft_checkpoint --test_interactive
"""

import argparse
import torch
from pathlib import Path

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.model_loader import load_sft_model, get_model_info, test_model_generation


def test_basic_generation(model, tokenizer, device):
    """测试基础生成能力"""
    print("\n" + "="*60)
    print("测试1: 基础生成能力")
    print("="*60)

    test_prompts = [
        "你好，请介绍一下你自己。",
        "请解释什么是故意伤害罪。",
        "被告人张三因纠纷打伤李四，请分析案情。"
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n[测试 {i+1}] Prompt: {prompt}")

        # 构建对话格式
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 检查重复
        has_repetition = check_repetition(generated)

        print(f"输出: {generated[:200]}...")
        print(f"重复检测: {has_repetition}")

        if has_repetition:
            print("⚠️ 警告: 输出有重复，可能需要调整生成参数")


def test_legal_question_format(model, tokenizer, device):
    """测试法律问答格式"""
    print("\n" + "="*60)
    print("测试2: 法律问答格式（RL训练格式）")
    print("="*60)

    test_cases = [
        {
            "prompt": """案情摘要：被告人杨某以虚假票据骗取他人财物共计2.6万元。

近期对话：
[法官]: 我想了解作案动机
[公诉人]: 公诉人补充说明：经调查，被告人预谋作案...

已获取证据：
公诉人补充说明：经调查，被告人预谋作案...

【优先提问】
请继续提问获取更多证据。
示例：提问：被告人是否有自首情节？

请直接输出你的提问或判决：""",
            "expected_type": "提问",
            "description": "测试提问格式"
        },
        {
            "prompt": """案情摘要：被告人杨某以虚假票据骗取他人财物共计2.6万元。

已获取证据：
公诉人补充说明：经调查，被告人预谋作案，购买了作案工具
辩护人补充说明：被告人案发后主动投案并如实供述犯罪事实

【可以判决】
如果证据充分，可给出判决。
示例：判决：罪名：诈骗罪，刑期：36个月

请直接输出你的提问或判决：""",
            "expected_type": "判决",
            "description": "测试判决格式"
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\n[测试 {i+1}] {case['description']}")
        print(f"期望输出类型: {case['expected_type']}")

        # 直接使用prompt（不使用chat_template，因为已经是完整prompt）
        inputs = tokenizer(case['prompt'], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,  # 使用较高的temperature减少重复
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.1  # 添加重复惩罚
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # 检查输出格式
        output_type = "提问" if generated.startswith("提问") else ("判决" if generated.startswith("判决") else "其他")
        has_repetition = check_repetition(generated)

        print(f"\n输出: {generated[:300]}")
        print(f"实际类型: {output_type}")
        print(f"重复检测: {has_repetition}")

        # 判断结果
        if output_type == case['expected_type']:
            print("✅ 格式正确")
        else:
            print(f"❌ 格式错误: 期望 '{case['expected_type']}'，实际 '{output_type}'")

        if has_repetition:
            print("⚠️ 警告: 输出有重复")
        else:
            print("✅ 无重复")


def test_multiple_samples(model, tokenizer, device, num_samples=4):
    """测试多次采样（模拟GRPO的group_size）"""
    print("\n" + "="*60)
    print(f"测试3: 多次采样（模拟group_size={num_samples}）")
    print("="*60)

    prompt = """案情摘要：被告人杨某以虚假票据骗取他人财物共计2.6万元。

【第1轮：必须提问】
请针对案情提问，获取关键证据细节（如作案动机、作案手段、案后表现等）。
示例：提问：被告人的作案动机是什么？

请直接输出你的提问或判决："""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs_list = []
    for i in range(num_samples):
        print(f"\n[采样 {i+1}/{num_samples}]")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.1
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        outputs_list.append(generated)
        has_repetition = check_repetition(generated)

        print(f"输出: {generated[:150]}...")
        print(f"重复: {has_repetition}")

    # 分析多样性
    unique_outputs = len(set([o[:50] for o in outputs_list]))
    print(f"\n多样性分析: {unique_outputs}/{num_samples} 个不同的输出开头")

    if unique_outputs >= num_samples * 0.5:
        print("✅ 输出多样性良好")
    else:
        print("⚠️ 警告: 输出多样性不足")


def check_repetition(text: str) -> bool:
    """检测文本是否有重复"""
    import re

    if len(text) < 20:
        return False

    # 1. 检查句子重复
    sentences = re.split(r'[。，]', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) >= 3:
        for i in range(len(sentences) - 1):
            if sentences[i] == sentences[i + 1] and len(sentences[i]) > 10:
                return True

    # 2. 检查短语重复
    pattern = r'(.{15,}?)[。，\s]*\1'
    if re.search(pattern, text):
        return True

    # 3. 检查单词重复比例
    words = text.split()
    if len(words) >= 10:
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count / len(words) > 0.3:
                return True

    return False


def test_interactive(model, tokenizer, device):
    """交互式测试"""
    print("\n" + "="*60)
    print("交互式测试（输入'quit'退出）")
    print("="*60)

    while True:
        user_input = input("\n请输入prompt: ").strip()
        if user_input.lower() == 'quit':
            break

        messages = [{"role": "user", "content": user_input}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        generated = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        print(f"\n模型输出:\n{generated}")


def main():
    parser = argparse.ArgumentParser(description="Test SFT model")
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default="models/sft_checkpoint",
        help="SFT checkpoint path"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model path"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 4bit quantization"
    )
    parser.add_argument(
        "--test_interactive",
        action="store_true",
        help="Run interactive test"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Max new tokens to generate"
    )

    args = parser.parse_args()

    # 检查checkpoint是否存在
    if not Path(args.sft_checkpoint).exists():
        print(f"错误: SFT checkpoint不存在: {args.sft_checkpoint}")
        print("请先完成SFT训练:")
        print("  python -m src.sft.train_sft --config configs/sft_config.yaml")
        return

    # 加载模型
    print("="*60)
    print("加载SFT模型...")
    print("="*60)
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.sft_checkpoint}")
    print(f"Quantization: {not args.no_quantization}")

    model, tokenizer = load_sft_model(
        base_model_path=args.base_model,
        lora_path=args.sft_checkpoint,
        use_quantization=not args.no_quantization,
        enable_training=False  # 测试时不需要训练模式
    )

    device = next(model.parameters()).device
    print(f"Device: {device}")

    # 显示模型信息
    info = get_model_info(model)
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 运行测试
    test_basic_generation(model, tokenizer, device)
    test_legal_question_format(model, tokenizer, device)
    test_multiple_samples(model, tokenizer, device)

    # 交互测试
    if args.test_interactive:
        test_interactive(model, tokenizer, device)

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()