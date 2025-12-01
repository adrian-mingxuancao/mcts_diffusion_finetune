#!/bin/bash
#SBATCH --job-name=mcts_entropy_recompute
#SBATCH --output=logs/entropy_recompute_%A_%a.out
#SBATCH --error=logs/entropy_recompute_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --array=0-36%3      # 37 tasks * 5 structures each covers ~185 structures (full CAMEO)

# MCTS Entropy Recompute Ablation
# Tests MCTS with dynamic entropy recomputation at each selection step
# vs cached entropy (computed once during expansion)

set -e

echo "=========================================="
echo "MCTS ENTROPY RECOMPUTE ABLATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

# Activate environment and caches on scratch
source ~/.bashrc
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH="/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:${PYTHONPATH:-}"
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TRANSFORMERS_CACHE="/net/scratch/caom/.cache/huggingface/transformers"
export TORCH_HOME="/net/scratch/caom/.cache/torch"
conda activate /net/scratch/caom/dplm_env

# Project root
PROJECT_ROOT="/home/caom/AID3/dplm/mcts_diffusion_finetune"
cd $PROJECT_ROOT

# Create logs directory
mkdir -p logs
mkdir -p /net/scratch/caom/mcts_entropy_recompute_results

# Configuration
MODE="${1:-multi_expert}"
NUM_STRUCTURES="${2:-183}"
NUM_ITERATIONS="${3:-25}"
MAX_DEPTH="${4:-5}"
OUTPUT_DIR="/net/scratch/caom/mcts_entropy_recompute_results"
BATCH_SIZE="${BATCH_SIZE:-5}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START_INDEX=$((TASK_ID * BATCH_SIZE))

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Structures per task: $BATCH_SIZE"
echo "  Start index: $START_INDEX"
echo "  Iterations: $NUM_ITERATIONS"
echo "  Max Depth: $MAX_DEPTH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run ablation based on mode
if [ "$MODE" == "multi_expert" ]; then
    echo "Running multi-expert entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode multi_expert \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "single_expert_0" ]; then
    echo "Running single expert 0 (650M) entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 0 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "single_expert_1" ]; then
    echo "Running single expert 1 (150M) entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 1 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "single_expert_2" ]; then
    echo "Running single expert 2 (3B) entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 2 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "single_expert_proteinmpnn" ]; then
    echo "Running single expert ProteinMPNN entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 3 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "random_no_expert" ]; then
    echo "Running random no expert entropy recompute ablation..."
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode random_no_expert \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

elif [ "$MODE" == "all" ]; then
    echo "Running all entropy recompute ablations..."
    
    # Multi-expert
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode multi_expert \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR
    
    # Single expert 0 (650M)
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 0 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR
    
    # Single expert 1 (150M)
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode single_expert \
        --single_expert_id 1 \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR
    
    # Random no expert
    python tests/ablations/mcts_entropy_recompute_ablation.py \
        --mode random_no_expert \
        --num_structures $BATCH_SIZE \
        --start_index $START_INDEX \
        --num_iterations $NUM_ITERATIONS \
        --max_depth $MAX_DEPTH \
        --output_dir $OUTPUT_DIR

else
    echo "ERROR: Unknown mode '$MODE'"
    echo "Valid modes: multi_expert, single_expert_0, single_expert_1, single_expert_2, single_expert_proteinmpnn, random_no_expert, all"
    exit 1
fi

# Check if ablation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ ABLATION COMPLETE"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Summary files:"
    ls -lh $OUTPUT_DIR/summary_*.json | tail -5
    echo ""
    echo "Compare with cached entropy results:"
    echo "  Cached: /net/scratch/caom/mcts_results/cameo2022"
    echo "  Recompute: $OUTPUT_DIR"
else
    echo ""
    echo "=========================================="
    echo "❌ ABLATION FAILED"
    echo "=========================================="
    echo "Check the log file for errors"
    exit 1
fi

echo ""
echo "End time: $(date)"
