#!/usr/bin/env bash
# Complete evaluation pipeline

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
GEN_DIR="${SRC_DIR}/train/gerador_charts"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NUM_CHARTS="${NUM_CHARTS:-100}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${SRC_DIR}/analysis_results}"
EVAL_DIR="${EVAL_DIR:-${SRC_DIR}/evaluation}"
MODELS_DIR="${MODELS_DIR:-${SRC_DIR}/models}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${GEN_DIR}/output}"
GEN_IMAGES_DIR="${GEN_IMAGES_DIR:-${GEN_OUTPUT_DIR}/images}"
GEN_GT_DIR="${GEN_GT_DIR:-${GEN_OUTPUT_DIR}/ground_truth}"

echo "========================================="
echo "Chart Analysis Evaluation Pipeline"
echo "========================================="

mkdir -p "${ANALYSIS_DIR}"
mkdir -p "${EVAL_DIR}"

# Step 1: Generate charts with ground truth
echo -e "\n[1/6] Generating ${NUM_CHARTS} charts with ground truth..."
cd "${GEN_DIR}"
"${PYTHON_BIN}" generator.py --num "${NUM_CHARTS}"
cd "${SRC_DIR}"

# Step 2: Run analysis
echo -e "\n[2/6] Running chart analysis..."
"${PYTHON_BIN}" analysis.py \
    --input "${GEN_IMAGES_DIR}" \
    --output "${ANALYSIS_DIR}" \
    --ocr Paddle \
    --calibration PROSAC \
    --annotated \
    --models-dir "${MODELS_DIR}"

# Step 3: Evaluate accuracy
echo -e "\n[3/6] Computing accuracy metrics..."
"${PYTHON_BIN}" scripts/run_evaluation.py \
    --gt "${GEN_GT_DIR}" \
    --pred "${ANALYSIS_DIR}" \
    --output "${EVAL_DIR}/evaluation_results.json"

# Step 4: Analyze handler performance
echo -e "\n[4/6] Analyzing handler performance..."
"${PYTHON_BIN}" evaluation/handler_analyzer.py "${EVAL_DIR}/evaluation_results.json"

# Step 5: Test calibration methods
echo -e "\n[5/6] Comparing calibration methods..."
"${PYTHON_BIN}" evaluation/calibration_tester.py \
    --images "${GEN_IMAGES_DIR}" \
    --gt "${GEN_GT_DIR}" \
    --models "${MODELS_DIR}" \
    --output "${EVAL_DIR}/calibration_tests"

# Step 6: Optimize confidence thresholds
echo -e "\n[6/6] Optimizing confidence thresholds..."
"${PYTHON_BIN}" evaluation/confidence_tuner.py "${ANALYSIS_DIR}" "${GEN_GT_DIR}"

echo -e "\n========================================="
echo "Pipeline complete! Results in ${EVAL_DIR}/"
echo "========================================="
