#!/bin/bash
#SBATCH --job-name=mcts_fold_resume
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132%12
#SBATCH --output=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_%A_%a.out
#SBATCH --error=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_%A_%a.err

set -euo pipefail

echo "🚀 Resuming MCTS Folding Ablation Study"
echo "📅 $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔢 GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "🔢 Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo ""

# Environment setup
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:${PYTHONPATH:-}
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TRANSFORMERS_CACHE="/net/scratch/caom/.cache/huggingface/transformers"
export TORCH_HOME="/net/scratch/caom/.cache/torch"
export TRITON_CACHE_DIR="/net/scratch/caom/.cache/triton"

WORKDIR="/home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding"
cd "${WORKDIR}"

# Configuration matrix (same as original)
STRUCTURES_PER_TASK=10
NUM_CONFIGS=7
NUM_STRUCTURE_BATCHES=19

# Calculate batch and config from task ID
BATCH_ID=$((SLURM_ARRAY_TASK_ID / NUM_CONFIGS))
CONFIG_ID=$((SLURM_ARRAY_TASK_ID % NUM_CONFIGS))

# Calculate structure range
START_IDX=$((BATCH_ID * STRUCTURES_PER_TASK))
END_IDX=$((START_IDX + STRUCTURES_PER_TASK))

# Fixed parameters
TEMPERATURE=1.0
NUM_ITERATIONS=25

# Determine configuration
case ${CONFIG_ID} in
    0)
        CONFIG_NAME="Random (MCTS-0)"
        MODE="random_no_expert"
        EXPERT_ID=""
        MAX_DEPTH=5
        UCT_FLAG=""
        ;;
    1)
        CONFIG_NAME="DPLM-2 150M (Single-Expert)"
        MODE="single_expert"
        EXPERT_ID="1"
        MAX_DEPTH=5
        UCT_FLAG=""
        ;;
    2)
        CONFIG_NAME="DPLM-2 650M (Single-Expert)"
        MODE="single_expert"
        EXPERT_ID="0"
        MAX_DEPTH=5
        UCT_FLAG=""
        ;;
    3)
        CONFIG_NAME="DPLM-2 3B (Single-Expert)"
        MODE="single_expert"
        EXPERT_ID="2"
        MAX_DEPTH=5
        UCT_FLAG=""
        ;;
    4)
        CONFIG_NAME="Sampling (Multi-Expert)"
        MODE="multi_expert"
        EXPERT_ID=""
        MAX_DEPTH=1
        UCT_FLAG=""
        ;;
    5)
        CONFIG_NAME="MCTS-PH (Multi-Expert)"
        MODE="multi_expert"
        EXPERT_ID=""
        MAX_DEPTH=5
        UCT_FLAG=""
        ;;
    6)
        CONFIG_NAME="MCTS-UCT (Multi-Expert)"
        MODE="multi_expert"
        EXPERT_ID=""
        MAX_DEPTH=5
        UCT_FLAG="--use_standard_uct"
        ;;
    *)
        echo "❌ Invalid config ID: ${CONFIG_ID}"
        exit 1
        ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo "🔬 ${CONFIG_NAME}"
echo "════════════════════════════════════════════════════════════════"
echo "📊 Configuration:"
echo "   Config ID: ${CONFIG_ID}"
echo "   Batch ID: ${BATCH_ID}"
echo "   Mode: ${MODE}"
if [ -n "${EXPERT_ID}" ]; then
    echo "   Expert: ${EXPERT_ID}"
fi
echo "   Structures: ${START_IDX}-${END_IDX}"
echo "   Iterations: ${NUM_ITERATIONS}"
echo "   Max Depth: ${MAX_DEPTH}"
echo "   Temperature: ${TEMPERATURE}"

# Print UCT selection type
if [ "${MAX_DEPTH}" -eq 1 ]; then
    echo "   Selection: Sampling (no tree)"
elif [ -n "${UCT_FLAG}" ]; then
    echo "   Selection: Standard UCT"
else
    echo "   Selection: PH-UCT (default)"
fi
echo "════════════════════════════════════════════════════════════════"
echo ""

# Build command
CMD="python test_mcts_folding_ablation.py ${START_IDX} ${END_IDX} --mode ${MODE}"
CMD="${CMD} --num_iterations ${NUM_ITERATIONS}"
CMD="${CMD} --max_depth ${MAX_DEPTH}"
CMD="${CMD} --temperature ${TEMPERATURE}"

if [ -n "${EXPERT_ID}" ]; then
    CMD="${CMD} --single_expert_id ${EXPERT_ID}"
fi

if [ -n "${UCT_FLAG}" ]; then
    CMD="${CMD} ${UCT_FLAG}"
fi

echo "🚀 Running: ${CMD}"
echo ""

# Run the command
eval ${CMD}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}, batch ${BATCH_ID}) completed successfully in ${SECONDS}s"
else
    echo ""
    echo "❌ Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}, batch ${BATCH_ID}) failed with exit code ${EXIT_CODE}"
fi

echo "📅 $(date)"
exit ${EXIT_CODE}
