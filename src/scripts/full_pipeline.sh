#!/bin/bash
# Complete evaluation pipeline

set -e  # Exit on error

echo "========================================="
echo "Chart Analysis Evaluation Pipeline"
echo "========================================="

# Configuration
NUM_CHARTS=100
OUTPUT_DIR="output"
ANALYSIS_DIR="analysis_results"
EVAL_DIR="evaluation"

# Step 1: Generate charts with ground truth
echo -e "\n[1/6] Generating $NUM_CHARTS charts with ground truth..."
cd ../gerador_charts
python generator.py --num $NUM_CHARTS
cd ../src

# Step 2: Run analysis
echo -e "\n[2/6] Running chart analysis..."
python analysis.py \
    --input ../gerador_charts/output/images \
    --output $ANALYSIS_DIR \
    --ocr Paddle \
    --calibration PROSAC \
    --annotated True \
    --models-dir models

# Step 3: Evaluate accuracy
echo -e "\n[3/6] Computing accuracy metrics..."
python scripts/run_evaluation.py \
    --gt ../gerador_charts/output/ground_truth \
    --pred $ANALYSIS_DIR \
    --output $EVAL_DIR/evaluation_results.json

# Step 4: Analyze handler performance
echo -e "\n[4/6] Analyzing handler performance..."
python evaluation/handler_analyzer.py $EVAL_DIR/evaluation_results.json

# Step 5: Test calibration methods
echo -e "\n[5/6] Comparing calibration methods..."
python evaluation/calibration_tester.py \
    --images ../gerador_charts/output/images \
    --gt ../gerador_charts/output/ground_truth \
    --models models \
    --output $EVAL_DIR/calibration_tests

# Step 6: Optimize confidence thresholds
echo -e "\n[6/6] Optimizing confidence thresholds..."
python evaluation/confidence_tuner.py $ANALYSIS_DIR ../gerador_charts/output/ground_truth

echo -e "\n========================================="
echo "Pipeline complete! Results in $EVAL_DIR/"
echo "========================================="
