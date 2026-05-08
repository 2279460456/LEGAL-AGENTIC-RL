"""
GRPO (Group Relative Policy Optimization) Algorithm Module

Implements the GRPO algorithm for legal multi-agent reinforcement learning.
基于SFT模型进行RL训练。

Core formula:
    L_GRPO = -E[∑_{i=1}^G (A_i / σ_R) · log π(a_i | s_i)]

    where:
    - G: Group size (number of trajectories per case)
    - A_i: Advantage of trajectory i = R_i - mean(R_group)
    - σ_R: Standard deviation of rewards in group

使用方法:
    from src.rl.grpo import GRPOTrainer, GRPOConfig
    from src.rl.model_loader import load_sft_model

    model, tokenizer = load_sft_model(
        base_model_path="Qwen/Qwen3-8B",
        lora_path="models/sft_checkpoint"
    )

    trainer = GRPOTrainer(model, tokenizer, config)
    loss, rewards = trainer.train_step(case_data)
"""

import json
import shutil
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW


@dataclass
class GRPOConfig:
    """GRPO training configuration"""
    group_size: int = 4  # Number of trajectories per group (G)
    temperature: float = 1.0  # Sampling temperature
    top_p: float = 0.9  # Top-p sampling threshold
    learning_rate: float = 1e-5  # Policy learning rate
    max_grad_norm: float = 1.0  # Gradient clipping
    value_loss_coef: float = 0.5  # Value loss coefficient (if using value network)
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_new_tokens: int = 256  # Maximum tokens to generate per action
    do_sample: bool = True  # Whether to sample (vs greedy)
    save_total_limit: int = 2  # 最多保留几个checkpoint（节省磁盘空间）


@dataclass
class Trajectory:
    """Single trajectory (sequence of actions)"""
    case_id: str
    states: List[Dict]  # Sequence of states
    actions: List[Dict]  # Sequence of actions
    action_texts: List[str] = field(default_factory=list)  # Sequence of action texts
    log_probs: List[float] = field(default_factory=list)  # Log probabilities
    total_reward: float = 0.0  # Final total reward
    step_rewards: List[float] = field(default_factory=list)  # Per-step rewards


class GRPOTrainer:
    """
    GRPO training algorithm implementation.

    基于SFT模型进行RL训练，使用LoRA适配器。

    Usage:
        # 先加载SFT模型
        model, tokenizer = load_sft_model(
            base_model_path="Qwen/Qwen3-8B",
            lora_path="models/sft_checkpoint"
        )

        # 创建训练器
        trainer = GRPOTrainer(model, tokenizer, config)
        for episode in range(num_episodes):
            loss, rewards = trainer.train_step(case_data)
    """

    def __init__(
        self,
        policy_model,  # SFT模型（带LoRA adapters）
        tokenizer,     # Tokenizer for the model
        config: Optional[GRPOConfig] = None,
        reward_calculator=None,
        environment=None,
        optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """
        Initialize GRPO Trainer.

        Args:
            policy_model: Language model (SFT模型，带LoRA adapters)
            tokenizer: Tokenizer for the model
            config: GRPO configuration
            reward_calculator: Reward calculator module
            environment: RL environment
            optimizer: Optimizer (如果None，自动创建)
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.config = config or GRPOConfig()
        self.reward_calculator = reward_calculator
        self.environment = environment

        # 创建优化器（只优化LoRA参数）
        if optimizer is None:
            trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
            if len(trainable_params) == 0:
                raise ValueError(
                    "模型没有可训练参数！请确保加载模型时启用训练模式:\n"
                    "  load_sft_model(..., enable_training=True)"
                )
            trainable_param_count = sum(p.numel() for p in trainable_params)
            print(f"Creating optimizer for {trainable_param_count} trainable parameters")
            self.optimizer = AdamW(trainable_params, lr=self.config.learning_rate)
        else:
            self.optimizer = optimizer

        # Training statistics
        self.training_stats = defaultdict(list)
        self.global_step = 0

    def enable_training_mode(self):
        """启用训练模式"""
        self.policy_model.train()

    def enable_eval_mode(self):
        """启用评估模式"""
        self.policy_model.eval()

    def train_step(
        self,
        case_data: Dict,
        group_size: Optional[int] = None
    ) -> Tuple[float, List[float]]:
        """
        Execute one GRPO training step.

        Args:
            case_data: Case data for training
            group_size: Override config group size

        Returns:
            Tuple of (loss, list of rewards)
        """
        G = group_size or self.config.group_size

        # 启用训练模式
        self.enable_training_mode()

        # Step 1: Generate G trajectories for same case
        trajectories = self._generate_trajectories(case_data, G)

        # Step 2: Compute total rewards for each trajectory
        rewards = [traj.total_reward for traj in trajectories]

        # Step 3: Compute advantages (relative to group mean)
        advantages = self._compute_advantages(rewards)

        # Step 4: Compute GRPO loss (基于实际log_probs)
        loss = self._compute_grpo_loss(trajectories, advantages)

        # Step 5: Backpropagate and update
        self._update_policy(loss)

        self.global_step += 1

        # Record statistics
        self.training_stats["mean_reward"].append(np.mean(rewards))
        self.training_stats["std_reward"].append(np.std(rewards))
        self.training_stats["loss"].append(loss.item() if hasattr(loss, 'item') else float(loss))

        return loss, rewards

    def _generate_trajectories(
        self,
        case_data: Dict,
        num_trajectories: int
    ) -> List[Trajectory]:
        """
        Generate multiple trajectories for same case.

        Args:
            case_data: Case data
            num_trajectories: Number of trajectories (G)

        Returns:
            List of Trajectory objects
        """
        trajectories = []

        for _ in range(num_trajectories):
            traj = self._sample_trajectory(case_data)
            trajectories.append(traj)

        return trajectories

    def _sample_trajectory(self, case_data: Dict) -> Trajectory:
        """
        Sample a single trajectory using current policy.

        Args:
            case_data: Case data

        Returns:
            Single Trajectory
        """
        # Reset environment
        state = self.environment.reset(case_data)

        states = []
        actions = []
        action_texts = []
        log_probs = []
        step_rewards = []

        while not self.environment.is_terminal():
            # Sample action from policy (使用实际模型)
            action, action_text, log_prob = self._sample_action(state)

            # Execute action
            result = self.environment.step(action)

            # Record
            states.append(self._state_to_dict(state))
            actions.append(action)
            action_texts.append(action_text)
            log_probs.append(log_prob)
            step_rewards.append(result.reward)

            # Update state
            state = result.state

        # Compute final reward
        final_reward = self.environment.get_final_reward()

        return Trajectory(
            case_id=case_data.get("case_id", ""),
            states=states,
            actions=actions,
            action_texts=action_texts,
            log_probs=log_probs,
            total_reward=final_reward + sum(step_rewards),
            step_rewards=step_rewards
        )

    def _sample_action(self, state) -> Tuple[Dict, str, float]:
        """
        Sample action from policy model (实际模型推理).

        Args:
            state: Current state (AgentState 或 Dict)

        Returns:
            Tuple of (action dict, action text, log probability)
        """
        # 将 AgentState 转换为字典
        state_dict = self._state_to_dict(state)

        # Build prompt for model
        prompt = self._build_prompt(state_dict)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.policy_model.device)

        # Generate with sampling
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        action_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse action from generated text
        action = self._parse_action(action_text)

        # Calculate log probability (简化版本，使用最后一个token的score)
        if outputs.scores:
            last_score = outputs.scores[-1][0]  # 最后一步的logits
            probs = torch.softmax(last_score, dim=-1)
            # 取生成的token的概率
            if len(generated_ids) > 0:
                last_token_id = generated_ids[-1]
                log_prob = torch.log(probs[last_token_id] + 1e-10).item()
            else:
                log_prob = -1.0
        else:
            log_prob = -1.0

        return action, action_text, log_prob

    def _parse_action(self, action_text: str) -> Dict:
        """
        Parse action from generated text.

        Args:
            action_text: Generated text from model

        Returns:
            Action dict with 'type' and 'content'
        """
        action_text = action_text.strip()

        # 判断是判决还是提问
        if "判决" in action_text or "裁定" in action_text or "罪名" in action_text:
            # 尝试解析判决内容
            action = {
                "type": "judge",
                "content": self._parse_judgment(action_text)
            }
        else:
            # 默认为查询
            action = {
                "type": "query",
                "content": action_text
            }

        return action

    def _parse_judgment(self, text: str) -> Dict:
        """解析判决内容"""
        # 简化解析，返回基本结构
        judgment = {
            "crime": "",
            "sentence_months": 0,
            "laws": []
        }

        # 尝试提取罪名
        import re
        crime_match = re.search(r"罪名[：:]\s*([^\n，。]+)", text)
        if crime_match:
            judgment["crime"] = crime_match.group(1).strip()

        # 尝试提取刑期
        sentence_match = re.search(r"有期徒刑\s*(\d+)\s*年", text)
        if sentence_match:
            judgment["sentence_months"] = int(sentence_match.group(1)) * 12
        else:
            sentence_match = re.search(r"有期徒刑\s*(\d+)\s*个?月", text)
            if sentence_match:
                judgment["sentence_months"] = int(sentence_match.group(1))

        return judgment

    def _build_prompt(self, state: Dict) -> str:
        """Build prompt for model based on state"""
        public_info = state.get('public_info', '')
        revealed = state.get('revealed_evidence', [])
        current_round = state.get('current_round', 0)
        max_rounds = state.get('max_rounds', 10)

        revealed_text = "\n".join(revealed) if revealed else "暂无"

        prompt = f"""你是一位资深法官，正在审理案件。你的任务是：
1. 分析案情和证据
2. 通过提问获取必要的证据细节
3. 当证据充分时给出判决

当前案情：{public_info}

已获取证据：
{revealed_text}

当前轮次：第{current_round}轮（最多{max_rounds}轮）

请决定下一步行动：
- 如果需要更多证据，请提问（格式：提问：...）
- 如果证据充分，请判决（格式：判决：罪名...刑期...）

请直接输出你的决定："""

        return prompt

    def _state_to_dict(self, state) -> Dict:
        """Convert state object to dictionary"""
        if hasattr(state, '__dict__'):
            return {
                "public_info": state.public_info,
                "revealed_evidence": state.revealed_evidence,
                "current_round": state.current_round,
                "max_rounds": state.max_rounds
            }
        return state

    def _compute_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute advantages relative to group mean.

        A_i = (R_i - mean(R)) / std(R)

        Args:
            rewards: List of trajectory rewards

        Returns:
            List of advantage values
        """
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        if std_reward == 0:
            return [0.0] * len(rewards)

        advantages = [(r - mean_reward) / std_reward for r in rewards]
        return advantages

    def _compute_grpo_loss(
        self,
        trajectories: List[Trajectory],
        advantages: List[float]
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        L = -∑_{i=1}^G A_i · ∑_{t} log π(a_t | s_t)

        Args:
            trajectories: List of trajectories
            advantages: List of advantage values

        Returns:
            Loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.policy_model.device)

        for traj, adv in zip(trajectories, advantages):
            # Sum of log probs for trajectory
            traj_log_prob = sum(traj.log_probs)

            # Weighted by advantage (转换为tensor)
            loss_contribution = -adv * traj_log_prob
            total_loss = total_loss + loss_contribution

        # Normalize by group size
        total_loss /= len(trajectories)

        return total_loss

    def _update_policy(self, loss: torch.Tensor):
        """
        Update policy model with computed loss.

        Args:
            loss: Computed loss tensor
        """
        self.optimizer.zero_grad()

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy_model.parameters() if p.requires_grad],
            self.config.max_grad_norm
        )

        # 更新参数
        self.optimizer.step()

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "mean_rewards": self.training_stats["mean_reward"],
            "std_rewards": self.training_stats["std_reward"],
            "losses": self.training_stats["loss"],
            "global_step": self.global_step
        }

    def save_checkpoint(self, output_dir: str, episode: int):
        """
        Save model checkpoint and remove old checkpoints if exceeding limit.

        Args:
            output_dir: Output directory
            episode: Current episode number
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = output_path / f"checkpoint_episode_{episode}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        self.policy_model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))

        # Save training state
        training_state = {
            "episode": episode,
            "global_step": self.global_step,
            "training_stats": dict(self.training_stats),
            "config": {
                "group_size": self.config.group_size,
                "temperature": self.config.temperature,
                "learning_rate": self.config.learning_rate,
                "save_total_limit": self.config.save_total_limit
            }
        }

        with open(checkpoint_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(training_state, f, ensure_ascii=False, indent=2)

        print(f"Checkpoint saved to: {checkpoint_dir}")

        # 删除旧的checkpoint（超过数量限制时）
        if self.config.save_total_limit > 0:
            self._cleanup_old_checkpoints(output_path)

    def _cleanup_old_checkpoints(self, output_path: Path):
        """
        清理旧checkpoint，只保留最新的N个

        Args:
            output_path: checkpoint目录
        """

        # 查找所有checkpoint目录
        checkpoints = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_episode_"):
                match = re.search(r"checkpoint_episode_(\d+)", item.name)
                if match:
                    episode_num = int(match.group(1))
                    checkpoints.append((episode_num, item))

        # 按episode排序（大的在前，即最新的）
        checkpoints.sort(key=lambda x: x[0], reverse=True)

        # 删除超过限制的旧checkpoint
        if len(checkpoints) > self.config.save_total_limit:
            to_remove = checkpoints[self.config.save_total_limit:]
            for episode, path in to_remove:
                print(f"Removing old checkpoint: {path}")
                shutil.rmtree(path)


def main():
    """Example usage"""
    config = GRPOConfig(
        group_size=4,
        temperature=1.0,
        learning_rate=1e-5
    )

    print(f"GRPO Config: group_size={config.group_size}")
    print(f"Temperature: {config.temperature}")
    print(f"Max new tokens: {config.max_new_tokens}")
    print("\n使用方法:")
    print("1. 先完成SFT训练，保存到 models/sft_checkpoint")
    print("2. 使用 load_sft_model 加载SFT模型")
    print("3. 创建 GRPOTrainer 并开始训练")


if __name__ == "__main__":
    main()