#!/usr/bin/env bash
# ==============================================================================
# ATML PA3 - Task 3: Test-Time Compute Shell Runner
# Executes offline evaluation, signal analysis, live refinement, and adaptive inference.
# ==============================================================================

set -eo pipefail

CONFIG_FILE="${1:-configs/task3_config.yaml}"
STAGES="${2:-all}"

echo "================================================================================"
echo "LAUNCHING ATML PA3 TASK 3 PIPELINE"
echo "Config: ${CONFIG_FILE} | Stages: ${STAGES}"
echo "================================================================================"

# Verify python environment
PYTHON_BIN="python3"
if [ -f "venv/bin/python" ]; then
    PYTHON_BIN="venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

${PYTHON_BIN} -u -m task3.run_all \
    --config "${CONFIG_FILE}" \
    --stages ${STAGES}

echo ""
echo "================================================================================"
echo "TASK 3 EXECUTION FINISHED SUCCESSFULLY"
echo "Results available in results/task3/"
echo "================================================================================"
