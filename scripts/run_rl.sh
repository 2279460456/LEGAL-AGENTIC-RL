#!/bin/bash
# RL Training Script (GRPO)
# Usage: ./scripts/run_rl.sh

echo "========================================"
echo "LEGAL-AGENTIC-RL: GRPO Training"
echo "========================================"

CONFIG_FILE="configs/rl_config.yaml"
SFT_CHECKPOINT="models/sft_checkpoint"
TRAIN_DATA="data/rl_env/train_cases.json"
OUTPUT_DIR="models/rl_checkpoint"

# Check if SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "Error: SFT checkpoint not found: $SFT_CHECKPOINT"
    echo "Please run SFT training first:"
    echo "  bash scripts/run_sft.sh"
    exit 1
fi

# Check if RL training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Warning: Training data not found: $TRAIN_DATA"
    echo "Please prepare RL environment data first:"
    echo "  python src/data_processing/build_rl_data.py --max_samples 600"
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run GRPO training
echo "Starting GRPO training..."
python src/rl/train_rl.py --config "$CONFIG_FILE"

echo "GRPO training completed."
echo "Model saved to: $OUTPUT_DIR"