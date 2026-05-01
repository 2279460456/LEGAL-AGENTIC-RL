#!/bin/bash
# Evaluation Script
# Usage: ./scripts/run_eval.sh

echo "========================================"
echo "LEGAL-AGENTIC-RL: Evaluation"
echo "========================================"

CONFIG_FILE="configs/eval_config.yaml"
TEST_DATA="data/rl_env/test_cases.json"
RESULTS_DIR="results/eval_results"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Run evaluation
python -m src.evaluation.auto_eval \
    --config "$CONFIG_FILE" \
    --test "$TEST_DATA" \
    --output "$RESULTS_DIR"

echo "Evaluation completed."