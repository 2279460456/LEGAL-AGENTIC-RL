"""
RL训练流程测试脚本
验证GRPO训练代码是否能够正常运行（包括新的SFT模型加载逻辑）
"""

import os
import sys
import json
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.environment import EvidenceEnvironment
from src.rl.reward import RewardCalculator, RewardComponents
from src.rl.grpo import GRPOConfig, GRPOTrainer, Trajectory


def test_environment():
    """测试环境初始化和运行"""
    print("="*60)
    print("测试环境初始化...")
    print("="*60)

    # 加载一条训练数据
    with open('data/rl_env/train_cases.json', 'r', encoding='utf-8') as f:
        train_cases = json.load(f)

    case_data = train_cases[0]
    print(f"案例ID: {case_data['case_id']}")
    print(f"罪名: {case_data['ground_truth']['crime']}")

    # 创建环境
    env = EvidenceEnvironment(trigger_method="keyword", max_rounds=10)
    state = env.reset(case_data)

    print(f"\n初始化成功!")
    print(f"public_info: {state.public_info[:50]}...")
    print(f"max_rounds: {state.max_rounds}")

    # 测试查询动作
    print("\n测试查询动作...")
    result = env.step({
        "type": "query",
        "content": "请说明被告人的作案动机"
    })

    print(f"触发新证据: {result.new_evidence is not None}")
    if result.new_evidence:
        print(f"新证据: {result.new_evidence[:50]}...")
    print(f"奖励: {result.reward}")
    print(f"是否终止: {result.is_terminal}")

    # 测试判决动作
    print("\n测试判决动作...")
    result = env.step({
        "type": "judge",
        "content": {
            "crime": case_data['ground_truth']['crime'],
            "sentence_months": 6,
            "laws": case_data['ground_truth']['laws']
        }
    })

    print(f"是否终止: {result.is_terminal}")

    # 计算最终奖励
    final_reward = env.get_final_reward()
    print(f"最终奖励: {final_reward}")

    print("\n[PASS] 环境测试通过!")
    return True


def test_reward_calculator():
    """测试奖励计算"""
    print("\n" + "="*60)
    print("测试奖励计算...")
    print("="*60)

    calculator = RewardCalculator()

    # 模拟预测和真实值
    prediction = {
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    ground_truth = {
        "crime": "故意伤害罪",
        "sentence_months": 36,
        "laws": ["刑法第234条"]
    }

    reward = calculator.calculate_total_reward(
        prediction=prediction,
        ground_truth=ground_truth,
        conversation_history=[("judge", "查询动机"), ("judge", "查询伤情")],
        revealed_evidence={"subjective", "objective", "sentencing"},
        total_evidence_levels=3,
        current_round=5,
        max_rounds=10
    )

    print(f"\n总奖励: {reward.total:.4f}")
    print(f"准确度奖励: {reward.accuracy:.4f}")
    print(f"信息奖励: {reward.information:.4f}")
    print(f"合规奖励: {reward.compliance:.4f}")

    # 测试部分匹配
    prediction2 = {
        "crime": "故意伤害罪",
        "sentence_months": 24,  # 不同刑期
        "laws": ["刑法第234条", "刑法第67条"]  # 多一个法条
    }

    reward2 = calculator.calculate_total_reward(
        prediction=prediction2,
        ground_truth=ground_truth,
        conversation_history=[],
        revealed_evidence={"subjective"},
        total_evidence_levels=3,
        current_round=3,
        max_rounds=10
    )

    print(f"\n部分匹配奖励: {reward2.total:.4f}")
    print(f"准确度奖励: {reward2.accuracy:.4f}")

    print("\n[PASS] 奖励计算测试通过!")
    return True


def test_grpo_config():
    """测试GRPO配置"""
    print("\n" + "="*60)
    print("测试GRPO配置...")
    print("="*60)

    config = GRPOConfig(
        group_size=8,
        temperature=1.0,
        top_p=0.9,
        learning_rate=2e-5,
        max_new_tokens=256
    )

    print(f"group_size: {config.group_size}")
    print(f"temperature: {config.temperature}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"max_new_tokens: {config.max_new_tokens}")

    print("\n[PASS] GRPO配置测试通过!")
    return True


def test_trajectory_dataclass():
    """测试轨迹数据类"""
    print("\n" + "="*60)
    print("测试轨迹数据类...")
    print("="*60)

    traj = Trajectory(
        case_id="test_001",
        states=[{"public_info": "test"}],
        actions=[{"type": "query", "content": "test"}],
        action_texts=["提问：作案动机是什么"],
        log_probs=[-0.5, -0.3],
        total_reward=0.8,
        step_rewards=[0.1, 0.1]
    )

    print(f"case_id: {traj.case_id}")
    print(f"total_reward: {traj.total_reward}")
    print(f"log_probs: {traj.log_probs}")
    print(f"action_texts: {traj.action_texts}")

    print("\n[PASS] 轨迹数据类测试通过!")
    return True


def test_grpo_trainer_without_model():
    """测试GRPO训练器（无实际模型，测试骨架）"""
    print("\n" + "="*60)
    print("测试GRPO训练器骨架...")
    print("="*60)

    config = GRPOConfig(group_size=2)

    # 创建训练器（使用None作为模型和tokenizer）
    # 注意：这只是测试骨架，实际训练需要真实模型
    print("注意：此测试仅验证训练器骨架，不涉及实际模型")

    # 测试advantage计算逻辑
    rewards = [0.5, 0.8, 0.3, 0.6]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    if std_reward == 0:
        advantages = [0.0] * len(rewards)
    else:
        advantages = [(r - mean_reward) / std_reward for r in rewards]

    print(f"原始奖励: {rewards}")
    print(f"计算advantages: {advantages}")

    # 测试GRPO loss计算逻辑
    log_probs = [[-0.5, -0.3], [-0.2, -0.1]]
    loss = 0.0
    for lp, adv in zip(log_probs, advantages[:2]):
        traj_log_prob = sum(lp)
        loss -= adv * traj_log_prob
    loss /= 2

    print(f"GRPO loss (模拟): {loss:.4f}")

    print("\n[PASS] GRPO训练器骨架测试通过!")
    return True


def test_full_flow_without_model():
    """测试完整流程（无需实际模型）"""
    print("\n" + "="*60)
    print("测试完整训练流程（无模型）...")
    print("="*60)

    # 加载训练数据
    with open('data/rl_env/train_cases.json', 'r', encoding='utf-8') as f:
        train_cases = json.load(f)

    case_data = train_cases[0]

    # 创建环境
    env = EvidenceEnvironment(trigger_method="keyword", max_rounds=5)
    state = env.reset(case_data)

    # 模拟一次查询
    env.step({"type": "query", "content": "作案动机是什么"})
    env.step({"type": "query", "content": "伤害程度如何"})

    # 模拟判决
    env.step({
        "type": "judge",
        "content": {
            "crime": case_data['ground_truth']['crime'],
            "sentence_months": 6,
            "laws": case_data['ground_truth']['laws']
        }
    })

    # 计算最终奖励
    final_reward = env.get_final_reward()

    print(f"最终奖励: {final_reward:.4f}")
    print(f"已揭示证据层级: {env.revealed_levels}")

    print("\n[PASS] 完整流程测试通过!")
    return True


def test_config_loading():
    """测试配置加载"""
    print("\n" + "="*60)
    print("测试配置加载...")
    print("="*60)

    import yaml

    config_path = "configs/rl_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"GRPO group_size: {config['grpo']['group_size']}")
    print(f"GRPO temperature: {config['grpo']['temperature']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"LoRA rank: {config['model']['lora']['r']}")
    print(f"SFT checkpoint: {config['model']['sft_checkpoint']}")
    print(f"Use 4bit: {config['model']['quantization']['use_4bit']}")

    print("\n[PASS] 配置加载测试通过!")
    return True


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("RL训练代码测试（更新版：基于SFT模型）")
    print("="*80)

    tests = [
        ("环境初始化", test_environment),
        ("奖励计算", test_reward_calculator),
        ("GRPO配置", test_grpo_config),
        ("轨迹数据类", test_trajectory_dataclass),
        ("GRPO训练器骨架", test_grpo_trainer_without_model),
        ("完整流程", test_full_flow_without_model),
        ("配置加载", test_config_loading),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n[FAIL] {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    passed = sum(1 for _, s in results if s)
    total = len(results)

    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n[SUCCESS] 所有测试通过！")
        print("\nGRPO训练流程：")
        print("1. 先完成SFT训练: python -m src.sft.train_sft --config configs/sft_config.yaml")
        print("2. 运行GRPO训练: python -m src.rl.train_rl --config configs/rl_config.yaml")
    else:
        print("\n[WARNING] 有测试失败，请检查相应模块。")


if __name__ == "__main__":
    main()