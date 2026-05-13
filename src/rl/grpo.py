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
    temperature: float = 0.9  # Sampling temperature（提高到0.9减少重复）
    top_p: float = 0.9  # Top-p sampling threshold
    learning_rate: float = 1e-5  # Policy learning rate
    max_grad_norm: float = 1.0  # Gradient clipping
    value_loss_coef: float = 0.5  # Value loss coefficient (if using value network)
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    max_new_tokens: int = 256  # Maximum tokens to generate per action
    do_sample: bool = True  # Whether to sample (vs greedy)
    save_total_limit: int = 2  # 最多保留几个checkpoint（节省磁盘空间）

    # ========== 新增：上下文长度控制 ==========
    max_prompt_tokens: int = 1200  # Prompt最大token数（为输出留空间）
    max_history_rounds: int = 3  # 保留最近N轮对话历史
    max_evidence_preview: int = 50  # 每个证据预览最大字数

    # ========== 新增：生成质量控制 ==========
    repetition_penalty: float = 1.1  # 重复惩罚系数（>1减少重复输出）


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

        # 清理显存缓存
        torch.cuda.empty_cache()
        print(f"[Memory cleared before episode, group_size={G}]")

        # 启用训练模式
        self.enable_training_mode()

        # Step 1: Generate G trajectories for same case (no_grad模式)
        self.policy_model.eval()  # 生成时用eval模式
        trajectories = self._generate_trajectories(case_data, G)

        # 清理显存后再计算loss
        torch.cuda.empty_cache()
        print("[Memory cleared before loss computation]")

        # 切回训练模式计算loss
        self.policy_model.train()

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

        for i in range(num_trajectories):
            print(f"  [Trajectory {i+1}/{num_trajectories}] Generating...")
            traj = self._sample_trajectory(case_data)
            trajectories.append(traj)
            print(f"  [Trajectory {i+1}/{num_trajectories}] Reward={traj.total_reward:.4f}")

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

        traj_round = 0
        while not self.environment.is_terminal():
            traj_round += 1
            # 进度显示（显示所有轮次）
            print(f"    [Round {traj_round}/{self.environment.max_rounds}] Generating...", end="", flush=True)

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

        print(f"    [Trajectory] Completed in {traj_round} rounds, action={actions[-1].get('type', 'unknown')}")

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

        # Generate with sampling - 优化生成参数
        with torch.no_grad():
            outputs = self.policy_model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,  # 使用配置中的重复惩罚
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                use_cache=True,  # 启用KV cache加速生成
                return_dict_in_generate=True,
                output_scores=True
            )
        print(" done")

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        action_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse action from generated text
        action = self._parse_action(action_text)

        # 显示模型输出（截断过长文本）
        display_text = action_text[:100] + "..." if len(action_text) > 100 else action_text
        print(f"      → [{action.get('type', 'unknown')}]: {display_text}")

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

        新增检测：
        1. 重复输出 → invalid
        2. prompt模板残留 → invalid
        3. 格式不正确 → invalid

        Args:
            action_text: Generated text from model

        Returns:
            Action dict with 'type' and 'content'
        """
        action_text = action_text.strip()

        # 检测1：重复输出
        if self._has_repetition(action_text):
            return {
                "type": "invalid",
                "content": action_text,
                "repetition_penalty": -0.3
            }

        # 检测2：prompt模板残留（如"提问：... 或 判决：..."）
        invalid_patterns = [
            "提问：... 或 判决：",
            "提问：... 或：判决",
            "或 判决：",
            "或：判决",
            "你的决定",
            "你的行动",
            "【",  # 模板格式符号残留
        ]
        for pattern in invalid_patterns:
            if pattern in action_text:
                return {
                    "type": "invalid",
                    "content": action_text,
                    "repetition_penalty": -0.2
                }

        # 检测3：必须以"提问："或"判决："开头
        if not action_text.startswith("提问") and not action_text.startswith("判决"):
            # 不是有效格式，视为invalid
            return {
                "type": "invalid",
                "content": action_text,
                "repetition_penalty": -0.15
            }

        # 有效格式解析
        if action_text.startswith("判决"):
            # 尝试解析判决内容
            return {
                "type": "judge",
                "content": self._parse_judgment(action_text)
            }
        else:
            # 提问格式
            return {
                "type": "query",
                "content": action_text
            }

    def _has_repetition(self, text: str) -> bool:
        """
        检测输出是否有重复循环

        Args:
            text: 输出文本

        Returns:
            True if repetition detected
        """
        if len(text) < 20:
            return False

        # 1. 检查句子重复（以句号分隔）
        sentences = re.split(r'[。，]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 3:
            # 检查是否有连续2个以上相同句子
            for i in range(len(sentences) - 1):
                if sentences[i] == sentences[i + 1] and len(sentences[i]) > 10:
                    return True

        # 2. 检查短语重复（如"判决：罪名：..."重复出现）
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
        """
        Build prompt for model based on state.

        改进版本：
        1. 控制对话历史长度（避免上下文爆炸）
        2. 压缩证据内容为摘要
        3. 总token数预估（控制在1500以内，为输出留空间）

        Qwen3-8B 建议总长度 < 2048（训练时）
        """
        public_info = state.get('public_info', '')
        revealed = state.get('revealed_evidence', [])
        current_round = state.get('current_round', 0)
        max_rounds = state.get('max_rounds', 10)
        conversation_history = state.get('conversation_history', [])

        # ========== 1. 对话历史截断（保留最近N轮）==========
        # 每轮包含法官提问+控辩回复，约200-400 tokens
        max_history_rounds = self.config.max_history_rounds
        recent_history = conversation_history[-(max_history_rounds * 2):]  # 每轮2条消息

        # ========== 2. 压缩对话历史为摘要格式 ==========
        history_summary = ""
        max_preview = self.config.max_evidence_preview
        if recent_history:
            # 按轮次组织
            history_items = []
            for role, msg in recent_history:
                # 截断过长内容
                msg_preview = msg[:max_preview] + "..." if len(msg) > max_preview else msg
                role_label = {"judge": "法官", "prosecutor": "公诉人", "defender": "辩护人"}.get(role, role)
                history_items.append(f"[{role_label}]: {msg_preview}")
            history_summary = "\n".join(history_items[-6:])  # 最多6条

        # ========== 3. 压缩已收集证据 ==========
        # 证据内容可能很长，需要压缩
        if revealed:
            # 只保留最近2个证据，每个截断
            evidence_summary = []
            for ev in revealed[-2:]:
                ev_short = ev[:max_preview] + "..." if len(ev) > max_preview else ev
                evidence_summary.append(ev_short)
            revealed_text = "\n".join(evidence_summary)
        else:
            revealed_text = "暂无"

        # ========== 4. 根据轮次构建指令 ==========
        if current_round == 0:
            # 第1轮：强制提问
            instruction = """【第1轮：必须提问】
请针对案情提问，获取关键证据细节（如作案动机、作案手段、案后表现等）。
示例：提问：被告人的作案动机是什么？"""
        elif current_round < 3:
            # 前3轮：优先提问
            instruction = """【优先提问】
请继续提问获取更多证据。
示例：提问：被告人是否有自首情节？"""
        else:
            # 后续轮：可以判决
            instruction = """【可以判决】
如果证据充分，可给出判决。
示例：判决：罪名：诈骗罪，刑期：36个月"""

        # ========== 5. 构建prompt（预估token数）==========
        # 案情部分（约100-200 tokens）
        case_section = f"案情摘要：{public_info[:150]}..." if len(public_info) > 150 else f"案情摘要：{public_info}"

        # 对话历史部分（约200-400 tokens）
        if history_summary:
            history_section = f"\n\n近期对话：\n{history_summary}"
        else:
            history_section = ""

        # 证据部分（约100 tokens）
        evidence_section = f"\n\n已获取证据：\n{revealed_text}"

        prompt = f"""{case_section}{history_section}{evidence_section}

{instruction}

请直接输出你的提问或判决："""

        # 预估prompt长度（粗略估算：1 token ≈ 1.5 中文字符）
        estimated_tokens = len(prompt) / 1.5
        if estimated_tokens > self.config.max_prompt_tokens:
            # 如果超出限制，进一步压缩
            prompt = self._compress_prompt_further(prompt, max_tokens=self.config.max_prompt_tokens)

        return prompt

    def _compress_prompt_further(self, prompt: str, max_tokens: int = 1200) -> str:
        """
        进一步压缩prompt，确保不超过token限制。

        Args:
            prompt: 原始prompt
            max_tokens: 目标最大token数

        Returns:
            压缩后的prompt
        """
        max_chars = max_tokens * 1.5  # 粗略估算

        # 提取关键部分
        parts = prompt.split("\n\n")

        # 保留指令部分（最后部分）
        instruction_part = parts[-1] if parts else ""

        # 压缩其他部分
        compressed_parts = []
        remaining_chars = max_chars - len(instruction_part)

        for part in parts[:-1]:
            if remaining_chars > 0:
                if len(part) > remaining_chars:
                    compressed_parts.append(part[:int(remaining_chars)] + "...")
                    remaining_chars = 0
                else:
                    compressed_parts.append(part)
                    remaining_chars -= len(part)

        return "\n\n".join(compressed_parts) + "\n\n" + instruction_part

    def _state_to_dict(self, state) -> Dict:
        """Convert state object to dictionary"""
        if hasattr(state, '__dict__'):
            return {
                "public_info": state.public_info,
                "revealed_evidence": state.revealed_evidence,
                "conversation_history": getattr(state, 'conversation_history', []),
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
        Compute GRPO loss with gradient-enabled log probabilities.

        L = -∑_{i=1}^G A_i · ∑_{t} log π(a_t | s_t)

        Args:
            trajectories: List of trajectories
            advantages: List of advantage values

        Returns:
            Loss tensor (with gradients)
        """
        print("  [Computing GRPO loss with gradients...]")
        total_loss = torch.tensor(0.0, device=self.policy_model.device, requires_grad=True)

        for i, (traj, adv) in enumerate(zip(trajectories, advantages)):
            # 重新计算带梯度的log概率
            traj_log_prob = self._compute_trajectory_log_prob_with_grad(traj)

            # Weighted by advantage
            loss_contribution = -adv * traj_log_prob
            total_loss = total_loss + loss_contribution

            if i == 0:
                print(f"    Trajectory 0: log_prob={traj_log_prob.item():.4f}, advantage={adv:.4f}")

        # Normalize by group size
        total_loss = total_loss / len(trajectories)
        print(f"  [Loss computed: {total_loss.item():.4f}]")

        return total_loss

    def _compute_trajectory_log_prob_with_grad(self, traj: Trajectory) -> torch.Tensor:
        """
        Compute log probability of trajectory with gradients enabled.

        优化版本：只计算最后几个token的log prob，减少显存占用。

        Args:
            traj: Trajectory object

        Returns:
            Log probability tensor with gradients
        """
        total_log_prob = torch.tensor(0.0, device=self.policy_model.device, requires_grad=True)

        for state_dict, action_text in zip(traj.states, traj.action_texts):
            # 构建完整prompt
            prompt = self._build_prompt(state_dict)
            full_text = prompt + action_text

            # Tokenize
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024  # 减少最大长度
            ).to(self.policy_model.device)

            # 获取prompt长度
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.policy_model.device)
            prompt_length = prompt_inputs.input_ids.shape[1]

            # 只取最后32个action tokens计算（减少计算量）
            action_ids = inputs.input_ids[0, prompt_length:]
            if len(action_ids) == 0:
                continue

            # 限制计算的token数量
            max_tokens_to_compute = min(32, len(action_ids))
            start_idx = max(0, len(action_ids) - max_tokens_to_compute)

            # Forward pass (带梯度) - 只计算需要的部分
            with torch.cuda.amp.autocast():  # 使用混合精度减少显存
                outputs = self.policy_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True
                )
            logits = outputs.logits

            # 只计算最后几个token的log概率
            pred_logits = logits[0, prompt_length-1+start_idx:prompt_length-1+len(action_ids), :]
            target_ids = action_ids[start_idx:]

            log_probs = torch.log_softmax(pred_logits, dim=-1)
            token_log_probs = log_probs[range(len(target_ids)), target_ids]

            # 总和
            step_log_prob = token_log_probs.sum()
            total_log_prob = total_log_prob + step_log_prob

            # 清理中间变量
            del outputs, logits, pred_logits, log_probs
            torch.cuda.empty_cache()

        return total_log_prob

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