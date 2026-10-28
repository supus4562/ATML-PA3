#!/usr/bin/env bash
# ==============================================================================
# ATML PA3 - Task 1: Large Language Model Fine-Tuning Execution Script
# Optimized for Kaggle GPU Environment (NVIDIA Tesla T4 / P100 16GB VRAM)
# ==============================================================================

set -e

echo "=============================================================================="
echo " Starting ATML PA3 Task 1 Pipeline on Kaggle GPU (Tesla T4 / P100)"
echo "=============================================================================="

# 1. Kaggle-safe environment settings
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 2. Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "[Info] Detected GPU Hardware:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "[Info] Dual-GPU acceleration active ($GPU_COUNT GPUs available): parallel variant training & evaluation enabled."
    else
        echo "[Info] Single GPU active ($GPU_COUNT GPU available): sequential execution mode."
    fi
else
    echo "[Warning] nvidia-smi not found. Verifying PyTorch CUDA..."
fi

# 3. Create required output directories
mkdir -p configs task1 data_cache/task1 checkpoints/task1 results/task1

# 4. Run master orchestrator with Kaggle configuration
CONFIG_PATH="configs/kaggle_config.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    CONFIG_PATH="configs/task1_config.yaml"
fi

python3 -m task1.run_all --config "$CONFIG_PATH" "$@"

echo "=============================================================================="
echo " Task 1 Pipeline Completed Successfully! Results saved to results/task1/"
echo "=============================================================================="
