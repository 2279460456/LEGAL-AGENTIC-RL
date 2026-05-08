"""
RL Training Main Script
Main entry point for GRPO reinforcement learning training.
基于SFT模型进行RL训练。

Usage:
    python -m src.rl.train_rl --config configs/rl_config.yaml

    # 从SFT checkpoint开始训练
    python -m src.rl.train_rl --config configs/rl_config.yaml --sft_checkpoint models/sft_checkpoint

    # 从RL checkpoint继续训练
    python -m src.rl.train_rl --config configs/rl_config.yaml --resume models/rl_checkpoint/checkpoint_episode_100
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from datetime import datetime

from .environment import EvidenceEnvironment
from .reward import RewardCalculator
from .grpo import GRPOTrainer, GRPOConfig
from .model_loader import load_sft_model, load_merged_model, get_model_info


def load_config(config_path: str) -> Dict:
    """Load YAML configuration"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_cases(data_path: str, max_cases: Optional[int] = None) -> List[Dict]:
    """Load training cases"""
    with open(data_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)

    if max_cases:
        cases = cases[:max_cases]

    print(f"Loaded {len(cases)} cases from {data_path}")
    return cases


def setup_logging(log_dir: str, run_name: str) -> str:
    """Setup logging directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}_{timestamp}"
    log_path.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def create_grpo_config(yaml_config: Dict) -> GRPOConfig:
    """Create GRPOConfig from YAML config"""
    grpo_section = yaml_config.get("grpo", {})
    training_section = yaml_config.get("training", {})

    # 确保learning_rate是float类型
    lr_value = training_section.get("learning_rate", 1e-5)
    if isinstance(lr_value, str):
        lr_value = float(lr_value)

    return GRPOConfig(
        group_size=grpo_section.get("group_size", 4),
        temperature=grpo_section.get("temperature", 1.0),
        top_p=grpo_section.get("top_p", 0.9),
        learning_rate=lr_value,
        max_grad_norm=training_section.get("max_grad_norm", 1.0),
        max_new_tokens=grpo_section.get("max_new_tokens", 256),
        do_sample=grpo_section.get("do_sample", True),
        save_total_limit=training_section.get("save_total_limit", 2)
    )


def train_grpo(
    config: Dict,
    train_cases: List[Dict],
    policy_model,
    tokenizer,
    num_episodes: int,
    log_dir: str,
    resume_episode: int = 0
):
    """
    Actual GRPO training function.

    Args:
        config: Training configuration
        train_cases: Training case data
        policy_model: Policy model (SFT模型)
        tokenizer: Tokenizer
        num_episodes: Number of training episodes
        log_dir: Log directory
        resume_episode: Episode to resume from
    """
    # Setup environment
    env_config = config.get("environment", {})
    env = EvidenceEnvironment(
        trigger_method=env_config.get("trigger_method", "keyword"),
        max_rounds=env_config.get("max_rounds", 10)
    )

    # Setup reward calculator
    reward_config = config.get("reward", {})
    reward_calc = RewardCalculator(
        weights=reward_config.get("weights"),
        info_params=reward_config.get("information"),
        compliance_params=reward_config.get("compliance")
    )

    # Setup GRPO trainer
    grpo_config = create_grpo_config(config)

    trainer = GRPOTrainer(
        policy_model=policy_model,
        tokenizer=tokenizer,
        config=grpo_config,
        reward_calculator=reward_calc,
        environment=env
    )

    # Training loop
    training_config = config.get("training", {})
    eval_interval = training_config.get("eval_interval", 100)
    save_interval = training_config.get("save_interval", 200)

    output_dir = config.get("output", {}).get("output_dir", "models/rl_checkpoint")

    print(f"\n{'='*60}")
    print("Starting GRPO Training...")
    print(f"{'='*60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Resume from episode: {resume_episode}")
    print(f"Training cases: {len(train_cases)}")
    print(f"Group size: {grpo_config.group_size}")

    for episode in range(resume_episode, num_episodes):
        # Sample case
        case_idx = episode % len(train_cases)
        case_data = train_cases[case_idx]

        # Training step
        try:
            loss, rewards = trainer.train_step(case_data)
        except Exception as e:
            print(f"Episode {episode} failed: {e}")
            continue

        # Logging
        if episode % eval_interval == 0 or episode == resume_episode:
            stats = trainer.get_training_stats()
            mean_reward = np.mean(rewards) if rewards else 0.0
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            print(f"Episode {episode}: Loss={loss_val:.4f}, Mean Reward={mean_reward:.4f}")

        # Save checkpoint
        if episode % save_interval == 0 and episode > resume_episode:
            trainer.save_checkpoint(output_dir, episode)

    # Save final checkpoint
    trainer.save_checkpoint(output_dir, num_episodes - 1)

    # Save training log
    final_stats = trainer.get_training_stats()
    log_file = Path(log_dir) / "training_log.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(final_stats, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete!")
    print(f"Final checkpoint saved to: {output_dir}")
    print(f"Training log saved to: {log_file}")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="GRPO RL Training (基于SFT模型)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--sft_checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint (LoRA adapters)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to RL checkpoint to resume from"
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable 4bit quantization"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("=" * 60)
    print("LEGAL-AGENTIC-RL: GRPO Training (基于SFT模型)")
    print("=" * 60)
    print(f"Config: {args.config}")

    # Setup directories
    output_dir = Path(config.get("output", {}).get("output_dir", "models/rl_checkpoint"))
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = setup_logging(
        config.get("output", {}).get("log_dir", "results/logs"),
        config.get("output", {}).get("run_name", "grpo_legal")
    )

    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")

    # Determine SFT checkpoint path
    sft_checkpoint = args.sft_checkpoint
    if sft_checkpoint is None:
        sft_checkpoint = config.get("model", {}).get("sft_checkpoint", "models/sft_checkpoint")

    # Check if SFT checkpoint exists
    if not Path(sft_checkpoint).exists():
        print(f"\n[ERROR] SFT checkpoint not found: {sft_checkpoint}")
        print("请先完成SFT训练，或使用 --sft_checkpoint 指定正确的路径")
        print("\nSFT训练命令:")
        print("  python -m src.sft.train_sft --config configs/sft_config.yaml")
        return

    # Load SFT model
    print(f"\n{'='*60}")
    print("Loading SFT Model...")
    print(f"{'='*60}")
    print(f"SFT checkpoint: {sft_checkpoint}")

    base_model = config.get("model", {}).get("base_model", "Qwen/Qwen3-8B")
    use_4bit = not args.no_quantization

    # 加载SFT模型，启用训练模式（用于GRPO训练）
    model, tokenizer = load_sft_model(
        base_model_path=base_model,
        lora_path=sft_checkpoint,
        use_quantization=use_4bit,
        enable_training=True  # 关键：启用LoRA训练
    )

    # Show model info
    model_info = get_model_info(model)
    print(f"\nModel Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Load training cases
    data_config = config.get("data", {})
    train_cases = load_cases(
        data_config.get("train_cases_path", "data/rl_env/train_cases.json"),
        max_cases=data_config.get("num_train_cases")
    )

    # Determine resume episode
    resume_episode = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            # Try to load training state
            state_file = resume_path / "training_state.json"
            if state_file.exists():
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    resume_episode = state.get("episode", 0) + 1
                print(f"Resuming from episode {resume_episode}")
            else:
                # Try to parse from directory name
                import re
                match = re.search(r"checkpoint_episode_(\d+)", resume_path.name)
                if match:
                    resume_episode = int(match.group(1)) + 1
                    print(f"Resuming from episode {resume_episode}")

    # Start training
    num_episodes = config.get("training", {}).get("num_episodes", 1000)

    train_grpo(
        config=config,
        train_cases=train_cases,
        policy_model=model,
        tokenizer=tokenizer,
        num_episodes=num_episodes,
        log_dir=log_dir,
        resume_episode=resume_episode
    )


if __name__ == "__main__":
    main()