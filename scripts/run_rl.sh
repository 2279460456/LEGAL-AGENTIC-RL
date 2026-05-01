#!/bin/bash
# RL Training Script (GRPO)
# Usage: ./scripts/run_rl.sh

echo "========================================"
echo "LEGAL-AGENTIC-RL: GRPO Training"
echo "========================================"

CONFIG_FILE="configs/rl_config.yaml"
TRAIN_DATA="data/rl_env/train_cases.json"
OUTPUT_DIR="models/rl_checkpoint"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Warning: Training data not found: $TRAIN_DATA"
    echo "Please prepare RL environment data first."
fi

# Run GRPO training
python -m src.rl.train_rl \
    --config "$CONFIG_FILE"

echo "GRPO training completed."