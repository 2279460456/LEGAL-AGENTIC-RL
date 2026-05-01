"""
GRPO (Group Relative Policy Optimization) Algorithm Module

Implements the GRPO algorithm for legal multi-agent reinforcement learning.

Core formula:
    L_GRPO = -E[∑_{i=1}^G (A_i / σ_R) · log π(a_i | s_i)]

    where:
    - G: Group size (number of trajectories per case)
    - A_i: Advantage of trajectory i = R_i - mean(R_group)
    - σ_R: Standard deviation of rewards in group
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import torch


@dataclass
class Trajectory:
    """Single trajectory (sequence of actions)"""
    case_id: str
    states: List[Dict]  # Sequence of states
    actions: List[Dict]  # Sequence of actions
    log_probs: List[float]  # Log probabilities of actions
    total_reward: float  # Final total reward
    step_rewards: List[float]  # Per-step rewards


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


class GRPOTrainer:
    """
    GRPO training algorithm implementation.

    GRPO is a variant of policy gradient that uses group-relative advantages:
    - Sample multiple trajectories for same case
    - Compute advantages relative to group mean
    - Optimize to favor trajectories with higher-than-average rewards

    Usage:
        trainer = GRPOTrainer(policy_model, config)
        for episode in range(num_episodes):
            loss, rewards = trainer.train_step(case_data)
    """

    def __init__(
        self,
        policy_model,  # The language model to train
        config: Optional[GRPOConfig] = None,
        reward_calculator=None,
        environment=None
    ):
        """
        Initialize GRPO Trainer.

        Args:
            policy_model: Language model (with LoRA adapters)
            config: GRPO configuration
            reward_calculator: Reward calculator module
            environment: RL environment
        """
        self.policy_model = policy_model
        self.config = config or GRPOConfig()
        self.reward_calculator = reward_calculator
        self.environment = environment

        # Training statistics
        self.training_stats = defaultdict(list)

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

        # Step 1: Generate G trajectories for same case
        trajectories = self._generate_trajectories(case_data, G)

        # Step 2: Compute total rewards for each trajectory
        rewards = [traj.total_reward for traj in trajectories]

        # Step 3: Compute advantages (relative to group mean)
        advantages = self._compute_advantages(rewards)

        # Step 4: Compute GRPO loss
        loss = self._compute_grpo_loss(trajectories, advantages)

        # Step 5: Backpropagate and update
        self._update_policy(loss)

        # Record statistics
        self.training_stats["mean_reward"].append(np.mean(rewards))
        self.training_stats["std_reward"].append(np.std(rewards))
        self.training_stats["loss"].append(loss)

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
        log_probs = []
        step_rewards = []

        while not self.environment.is_terminal():
            # Sample action from policy
            action, log_prob = self._sample_action(state)

            # Execute action
            result = self.environment.step(action)

            # Record
            states.append(self._state_to_dict(state))
            actions.append(action)
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
            log_probs=log_probs,
            total_reward=final_reward + sum(step_rewards),
            step_rewards=step_rewards
        )

    def _sample_action(self, state: Dict) -> Tuple[Dict, float]:
        """
        Sample action from policy model.

        Args:
            state: Current state

        Returns:
            Tuple of (action dict, log probability)
        """
        # Build prompt for model
        prompt = self._build_prompt(state)

        # Sample from model with temperature
        # TODO: Implement actual model sampling
        # For placeholder, return mock action

        # Mock implementation
        if np.random.random() < 0.3:  # 30% chance to judge
            action = {
                "type": "judge",
                "content": {
                    "crime": "故意伤害罪",
                    "sentence_months": 36,
                    "laws": ["刑法第234条"]
                }
            }
        else:
            action = {
                "type": "query",
                "content": "请说明被告人的作案动机"
            }

        log_prob = np.log(0.5)  # Placeholder log prob

        return action, log_prob

    def _build_prompt(self, state: Dict) -> str:
        """Build prompt for model based on state"""
        prompt = f"""当前案情：{state.get('public_info', '')}

已获取证据：{state.get('revealed_evidence', [])}

当前轮次：第{state.get('current_round', 0)}轮

请决定下一步行动：提问或判决。"""
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
    ) -> float:
        """
        Compute GRPO loss.

        L = -∑_{i=1}^G A_i · ∑_{t} log π(a_t | s_t)

        Args:
            trajectories: List of trajectories
            advantages: List of advantage values

        Returns:
            Loss value
        """
        loss = 0.0

        for traj, adv in zip(trajectories, advantages):
            # Sum of log probs for trajectory
            traj_log_prob = sum(traj.log_probs)

            # Weighted by advantage
            loss -= adv * traj_log_prob

        # Normalize by group size
        loss /= len(trajectories)

        return loss

    def _update_policy(self, loss: float):
        """
        Update policy model with computed loss.

        Args:
            loss: Computed loss value
        """
        # TODO: Implement actual gradient update
        # Placeholder for actual backpropagation

        # In actual implementation:
        # self.policy_model.backward(loss)
        # self.optimizer.step()
        pass

    def get_training_stats(self) -> Dict:
        """Get training statistics"""
        return {
            "mean_rewards": self.training_stats["mean_reward"],
            "std_rewards": self.training_stats["std_reward"],
            "losses": self.training_stats["loss"]
        }


def grpo_train_step(
    policy_model,
    env,
    case_data: Dict,
    reward_calculator,
    G: int = 4
) -> Tuple[float, List[float], List[Trajectory]]:
    """
    Standalone GRPO training step function.

    Args:
        policy_model: Policy model
        env: Environment
        case_data: Case data
        reward_calculator: Reward calculator
        G: Group size

    Returns:
        Tuple of (loss, rewards, trajectories)
    """
    trajectories = []
    rewards = []

    # Generate G trajectories
    for g in range(G):
        state = env.reset(case_data)
        traj_states = []
        traj_actions = []
        traj_log_probs = []
        total_reward = 0

        while not env.is_terminal():
            # Sample action (placeholder)
            action, log_prob = _sample_action_placeholder(state)
            traj_states.append(state)
            traj_actions.append(action)
            traj_log_probs.append(log_prob)

            # Step environment
            result = env.step(action)
            total_reward += result.reward
            state = result.state

        # Final reward
        final_reward = env.get_final_reward() or 0
        total_reward += final_reward

        trajectories.append(Trajectory(
            case_id=case_data.get("case_id", ""),
            states=traj_states,
            actions=traj_actions,
            log_probs=traj_log_probs,
            total_reward=total_reward,
            step_rewards=[]
        ))
        rewards.append(total_reward)

    # Compute advantages
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) or 1.0
    advantages = [(r - mean_r) / std_r for r in rewards]

    # Compute loss
    loss = 0
    for traj, adv in zip(trajectories, advantages):
        loss -= adv * sum(traj.log_probs)
    loss /= G

    return loss, rewards, trajectories


def _sample_action_placeholder(state) -> Tuple[Dict, float]:
    """Placeholder action sampling"""
    if np.random.random() < 0.3:
        return {"type": "judge", "content": {"crime": "placeholder"}}, -0.5
    return {"type": "query", "content": "placeholder query"}, -0.5


def main():
    """Example usage"""
    config = GRPOConfig(
        group_size=4,
        temperature=1.0,
        learning_rate=1e-5
    )

    trainer = GRPOTrainer(
        policy_model=None,  # Placeholder
        config=config
    )

    # Simulate training step
    case_data = {
        "case_id": "test_001",
        "public_info": "Test case",
        "hidden_evidence": {},
        "ground_truth": {"crime": "test"}
    }

    # Would need actual environment and model
    print(f"GRPO Config: group_size={config.group_size}")
    print(f"Temperature: {config.temperature}")


if __name__ == "__main__":
    main()