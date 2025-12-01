#!/bin/bash
#SBATCH --job-name=sensitivity_analysis
#SBATCH --output=logs/sensitivity_analysis_%j.log
#SBATCH --error=logs/sensitivity_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=general

# Reward Weight Sensitivity Analysis for CAMEO Results
# This script reanalyzes existing MCTS results with different weight configurations

set -e

echo "=========================================="
echo "REWARD WEIGHT SENSITIVITY ANALYSIS"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate environment (prefer named env, fall back to prefix)
source ~/.bashrc
ENV_PREFIX="${DPLM_ENV_PREFIX:-/net/scratch/caom/dplm_env}"
if conda env list | awk '{print $1}' | grep -qx "dplm_env"; then
    conda activate dplm_env
elif [ -d "$ENV_PREFIX" ]; then
    conda activate "$ENV_PREFIX"
else
    echo "ERROR: Could not find conda env 'dplm_env' or prefix: $ENV_PREFIX"
    echo "Set DPLM_ENV_PREFIX to your environment path if different."
    exit 1
fi

# Project root
PROJECT_ROOT="/home/caom/AID3/dplm/mcts_diffusion_finetune"
cd $PROJECT_ROOT

# Create logs directory
mkdir -p logs

# Configuration
RESULTS_DIR="${1:-/net/scratch/caom/mcts_results/cameo2022}"
OUTPUT_DIR="${2:-./sensitivity_analysis_cameo_$(date +%Y%m%d_%H%M%S)}"

echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    echo "Please provide a valid results directory as first argument"
    exit 1
fi

# Count result files
NUM_JSON=$(find "$RESULTS_DIR" -name "*.json" | wc -l)
NUM_LOG=$(find "$RESULTS_DIR" -name "*.log" | wc -l)

echo "Found $NUM_JSON JSON files and $NUM_LOG log files"
echo ""

# Run analysis
if [ $NUM_JSON -gt 0 ]; then
    echo "Running analysis on JSON files..."
    python analysis/reanalyze_existing_results.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --file_pattern "*.json"
elif [ $NUM_LOG -gt 0 ]; then
    echo "Running analysis on log files (will convert to JSON first)..."
    python analysis/reanalyze_existing_results.py \
        --results_dir "$RESULTS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --convert_logs
else
    echo "ERROR: No JSON or log files found in $RESULTS_DIR"
    exit 1
fi

# Check if analysis succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ANALYSIS COMPLETE"
    echo "=========================================="
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUTPUT_DIR"
    echo ""
    echo "View results:"
    echo "  - Pareto fronts: $OUTPUT_DIR/pareto_fronts.png"
    echo "  - 3D Pareto: $OUTPUT_DIR/pareto_front_3d.png"
    echo "  - Report: $OUTPUT_DIR/sensitivity_analysis_report.txt"
    echo "  - Detailed data: $OUTPUT_DIR/sensitivity_analysis_detailed.json"
else
    echo ""
    echo "=========================================="
    echo "❌ ANALYSIS FAILED"
    echo "=========================================="
    echo "Check the log file for errors"
    exit 1
fi

echo ""
echo "End time: $(date)"
