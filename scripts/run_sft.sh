#!/bin/bash
# SFT Training Script
# Usage: ./scripts/run_sft.sh

echo "========================================"
echo "LEGAL-AGENTIC-RL: SFT Training"
echo "========================================"

CONFIG_FILE="configs/sft_config.yaml"
DATA_DIR="data/processed"
OUTPUT_DIR="models/sft_checkpoint"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Warning: Data directory not found: $DATA_DIR"
    echo "Please run data processing first."
fi

# Run SFT training
python -m src.sft.train_sft \
    --config "$CONFIG_FILE" \
    --data "$DATA_DIR" \
    --output "$OUTPUT_DIR"

echo "SFT training completed."