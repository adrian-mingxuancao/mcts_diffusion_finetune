#!/bin/bash
# Weight sensitivity sweep for MCTS (reward weight configs)
# Saves results under /net/scratch/caom by default.

# SLURM resources (adjust partition/gres/time if needed)
#SBATCH --job-name=weight_sensitivity
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-18%4          # 9 configs * 2 batches (10 structures each) ≈ full CAMEO; cap concurrency
#SBATCH --output=logs/weight_sensitivity_%A_%a.out
#SBATCH --error=logs/weight_sensitivity_%A_%a.err

set -euo pipefail

echo "=========================================="
echo "MCTS Reward Weight Sensitivity Sweep"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo ""

# ---- Environment setup ----
source ~/.bashrc
ENV_PREFIX="${DPLM_ENV_PREFIX:-/net/scratch/caom/dplm_env}"
if [ -d "$ENV_PREFIX" ]; then
    conda activate "$ENV_PREFIX"
else
    conda activate dplm_env || {
        echo "ERROR: Could not find conda env 'dplm_env' or prefix: $ENV_PREFIX"
        exit 1
    }
fi

# Caches on scratch to avoid $HOME pressure
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TRANSFORMERS_CACHE="/net/scratch/caom/.cache/huggingface/transformers"
export TORCH_HOME="/net/scratch/caom/.cache/torch"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# ---- Paths and arguments ----
PROJECT_ROOT="/home/caom/AID3/dplm/mcts_diffusion_finetune/rebuttal sensitivity analysis"
cd "$PROJECT_ROOT"
mkdir -p logs

DATA_DIR="${1:-/home/caom/AID3/dplm/data-bin/cameo2022}"
OUTPUT_BASE="${2:-/net/scratch/caom/mcts_weight_sensitivity}"
TOTAL_STRUCTURES="${3:-20}"
BATCH_SIZE="${4:-10}"
MAX_DEPTH="${5:-5}"
NUM_ITERATIONS="${6:-25}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIGS=("baseline" "sequence_dominant" "sequence_focused" "structure_dominant" "structure_focused" "equal_balance" "structure_slight_edge" "biophysical_aware" "biophysical_strict")
NUM_CONFIGS=${#CONFIGS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
BATCH_INDEX=$(( TASK_ID / NUM_CONFIGS ))
CONFIG_INDEX=$(( TASK_ID % NUM_CONFIGS ))
START_INDEX=$(( BATCH_INDEX * BATCH_SIZE ))
if [ "$START_INDEX" -ge "$TOTAL_STRUCTURES" ]; then
  echo "Array index ${TASK_ID} maps outside dataset (start_index=${START_INDEX} >= ${TOTAL_STRUCTURES}); skipping."
  exit 0
fi
CONFIG_NAME=${CONFIGS[$CONFIG_INDEX]}

OUTPUT_DIR="${OUTPUT_BASE}/weight_sensitivity_${TIMESTAMP}_${CONFIG_NAME}_batch${BATCH_INDEX}"

mkdir -p "$OUTPUT_DIR"
mkdir -p /net/scratch/caom/logs

echo "Data dir:      $DATA_DIR"
echo "Output base:   $OUTPUT_BASE"
echo "Output dir:    $OUTPUT_DIR"
echo "Total structures: $TOTAL_STRUCTURES"
echo "Batch size: $BATCH_SIZE"
echo "Start index: $START_INDEX"
echo "Max depth:      $MAX_DEPTH"
echo "Iterations:     $NUM_ITERATIONS"
echo "Config:         $CONFIG_NAME"
echo ""

# ---- Run sweep ----
python run_weight_sensitivity_experiments.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --num_structures "$BATCH_SIZE" \
  --start_index "$START_INDEX" \
  --max_depth "$MAX_DEPTH" \
  --num_iterations "$NUM_ITERATIONS" \
  --configs "$CONFIG_NAME"

STATUS=$?
echo ""
echo "Run status: $STATUS"
echo "Results saved to: $OUTPUT_DIR"
echo "Next: reanalyze with:"
echo "  python reanalyze_existing_results.py \\"
echo "    --results_dir \"$OUTPUT_DIR\" \\"
echo "    --output_dir \"${OUTPUT_DIR}_analysis\""
echo ""
echo "End time: $(date)"
exit $STATUS
