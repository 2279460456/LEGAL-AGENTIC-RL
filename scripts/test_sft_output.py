"""
测试SFT模型输出质量

使用方法:
    python scripts/test_sft_output.py --lora_path models/sft_checkpoint

或者直接运行:
    python scripts/test_sft_output.py
"""

import argparse
import torch
from transformers import AutoTokenizer

# 添加项目路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.model_loader import load_sft_model


# 测试用例
TEST_CASES = [
    {
        "name": "故意伤害案",
        "prompt": """你是一位资深法官，正在审理案件。

当前案情：2024年3月，被告人张三与被害人李四因琐事发生冲突，张三用拳头击打李四面部，致李四鼻骨骨折。经鉴定，李四的损伤程度为轻伤二级。

请分析案情并给出判决结论。"""
    },
    {
        "name": "诈骗案",
        "prompt": """你是一位资深法官，正在审理案件。

当前案情：被告人王五于2023年虚构投资项目，骗取被害人赵六人民币50万元。

请分析案情并给出判决结论。"""
    },
    {
        "name": "危险驾驶案",
        "prompt": """你是一位资深法官，正在审理案件。

当前案情：2024年5月，被告人陈某醉酒驾驶机动车，血液酒精含量为180mg/100ml，被交警查获。

请分析案情并给出判决结论。"""
    }
]


def test_sft_model(model_path: str = "models/sft_checkpoint",
                   base_model: str = "Qwen/Qwen3-8B",
                   num_samples: int = 3):
    """
    测试SFT模型输出

    Args:
        model_path: SFT checkpoint路径
        base_model: 基座模型路径
        num_samples: 每个case生成几个样本（测试多样性）
    """
    print("="*60)
    print("测试SFT模型输出质量")
    print("="*60)

    # 加载模型（推理模式，不需要训练）
    model, tokenizer = load_sft_model(
        base_model_path=base_model,
        lora_path=model_path,
        use_quantization=True,
        enable_training=False  # 推理模式
    )

    print("\n开始测试...")
    print("="*60)

    for case in TEST_CASES:
        print(f"\n【测试案件: {case['name']}】")
        print("-"*40)

        for i in range(num_samples):
            print(f"\n样本 {i+1}:")

            # 构建输入
            messages = [{"role": "user", "content": case['prompt']}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
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

            print(f"输出: {generated_text}")

            # 检查输出质量
            quality_check(generated_text, case['name'])

    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


def quality_check(output: str, case_name: str):
    """
    检查输出质量
    """
    issues = []

    # 1. 检查是否有重复
    if check_repetition(output):
        issues.append("⚠️ 输出有重复模式")

    # 2. 检查是否包含判决关键词
    if "判决" not in output and "罪名" not in output:
        issues.append("⚠️ 缺少判决关键词")

    # 3. 检查是否有法律术语
    legal_terms = ["被告人", "被害人", "本院认为", "依照", "刑法"]
    term_count = sum(1 for term in legal_terms if term in output)
    if term_count < 2:
        issues.append("⚠️ 法律术语使用不足")

    # 4. 检查输出长度
    if len(output) < 50:
        issues.append("⚠️ 输出过短")
    elif len(output) > 500:
        issues.append("⚠️ 输出过长")

    # 打印检查结果
    if issues:
        print(f"质量检查: {', '.join(issues)}")
    else:
        print("质量检查: ✅ 正常")


def check_repetition(text: str, threshold: float = 0.3) -> bool:
    """
    检查文本是否有重复模式

    Args:
        text: 输出文本
        threshold: 重复比例阈值

    Returns:
        True if repetition detected
    """
    # 检查连续重复
    words = text.split()
    if len(words) < 10:
        return False

    # 统计重复词
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # 如果某个词出现超过30%的比例，认为有重复
    for word, count in word_counts.items():
        if len(word) > 4:  # 只检查较长词
            if count / len(words) > threshold:
                return True

    # 检查短语重复
    for i in range(len(words) - 5):
        phrase = ' '.join(words[i:i+5])
        if text.count(phrase) > 2:
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="测试SFT模型输出")
    parser.add_argument(
        "--lora_path",
        type=str,
        default="models/sft_checkpoint",
        help="SFT checkpoint路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="基座模型路径"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="每个case生成几个样本"
    )

    args = parser.parse_args()

    test_sft_model(
        model_path=args.lora_path,
        base_model=args.base_model,
        num_samples=args.samples
    )


if __name__ == "__main__":
    main()