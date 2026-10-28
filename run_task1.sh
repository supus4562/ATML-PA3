#!/usr/bin/env bash
# ==============================================================================
# ATML PA3 - Task 1: Large Language Model Fine-Tuning Execution Script
# Optimized for NVIDIA RTX 6000 Pro (Ada Lovelace / Blackwell) Workstation Architecture
# ==============================================================================

set -e

echo "=============================================================================="
echo " Starting ATML PA3 Task 1 Pipeline on NVIDIA RTX 6000 Pro (Ada/Blackwell)"
echo "=============================================================================="

# 1. Performance environment settings
export TOKENIZERS_PARALLELISM="false"
export PYTHONUNBUFFERED=1

# 2. Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "[Info] Detected NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "[Warning] nvidia-smi not found. Checking PyTorch device in Python..."
fi

# 3. Create required directories
mkdir -p configs task1 data_cache/task1 checkpoints/task1 results/task1

# 4. Run master orchestrator
# Pass any extra arguments directly (e.g. --stages, --config)
python3 -m task1.run_all "$@"

echo "=============================================================================="
echo " Task 1 Pipeline Completed! Results saved to results/task1/"
echo "=============================================================================="
