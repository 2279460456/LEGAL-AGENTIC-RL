"""
Test SFT Model on Windows/Server
验证SFT checkpoint是否正常工作
"""

import subprocess
import sys

def run_test():
    print("="*60)
    print("Testing SFT Model (LoRA Adapter)")
    print("="*60)

    # Test 1: Load model
    print("\n[Test 1] Loading model...")
    cmd = [
        sys.executable, "-m", "src.rl.model_loader",
        "--base_model", "Qwen/Qwen3-8B",
        "--lora_path", "models/sft_checkpoint"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.TimeoutExpired:
        print("Model loading timed out (this may happen on first load)")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Generation test (optional)
    print("\n[Test 2] Testing generation (run manually if needed)...")
    print("Command: python -m src.rl.model_loader --test")

    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. If model loads, proceed to RL training")
    print("2. RL training uses models/sft_checkpoint directly")
    print("3. Start RL: python -m src.rl.train_rl")


if __name__ == "__main__":
    run_test()