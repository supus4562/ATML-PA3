#!/usr/bin/env bash
# ==============================================================================
# ATML PA3 - Task 2: Vision-Language Models & Visual Grounding
# Runs end-to-end VLM pipeline:
# 1. Split isolation & manifest generation
# 2. SigLIP patch feature caching
# 3. Connector calibration
# 4. Stage 1: Caption alignment (2,250 updates)
# 5. Stage 2: Visual instruction tuning (1,500 updates)
# 6. Stage 3: Natural VQA specialization (750 updates)
# 7. Final 4-condition grounding evaluation with 4,000 pair bootstrap CIs
# 8. Report synthesis (metrics_summary.json, qualitative_annotations.md)
# ==============================================================================
set -euo pipefail

echo "================================================================================"
echo "LAUNCHING ATML PA3 TASK 2: VLM & VISUAL GROUNDING"
echo "================================================================================"

CONFIG_PATH="${1:-configs/task2_config.yaml}"
STAGES="${2:-all}"

python3 -u -m task2.run_all \
    --config "${CONFIG_PATH}" \
    --stages "${STAGES}"

echo "================================================================================"
echo "TASK 2 COMPLETED! Results saved in results/task2/"
echo "================================================================================"
