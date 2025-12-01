#!/bin/bash
#SBATCH --job-name=mcts_fold_complete
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-132%12
#SBATCH --output=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_%A_%a.out
#SBATCH --error=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_%A_%a.err

set -euo pipefail

echo "🚀 Starting MCTS Folding Complete Ablation Study"
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

# Configuration matrix:
# 7 configurations × 19 structure batches = 133 total tasks
# REORGANIZED: Each structure batch gets all 7 configs before moving to next batch
# This ensures fair comparison even if jobs don't complete
#
# Task organization:
#   Tasks 0-6: Structures 0-10 (all 7 configs)
#   Tasks 7-13: Structures 10-20 (all 7 configs)
#   Tasks 14-20: Structures 20-30 (all 7 configs)
#   ...
#   Tasks 126-132: Structures 180-190 (all 7 configs)
#
# Configurations:
#   0: Random (MCTS-0)
#   1: DPLM-2 150M (Single-Expert)
#   2: DPLM-2 650M (Single-Expert)
#   3: DPLM-2 3B (Single-Expert)
#   4: Sampling (Multi-Expert, depth=1, PH-UCT)
#   5: MCTS-PH (Multi-Expert, depth=5, PH-UCT with entropy) - DEFAULT
#   6: MCTS-UCT (Multi-Expert, depth=5, standard UCT)

STRUCTURES_PER_TASK=10
NUM_CONFIGS=7
NUM_STRUCTURE_BATCHES=19

# REORGANIZED CALCULATION: Structure batch first, then config
# This ensures all configs for a structure batch complete before moving to next batch
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
echo "📊 Structure range: ${START_IDX}-${END_IDX}"
echo "🎯 Mode: ${MODE}"
if [ -n "${EXPERT_ID}" ]; then
    echo "🤖 Expert ID: ${EXPERT_ID}"
fi
echo "🌳 Max depth: ${MAX_DEPTH}"
if [ "${MODE}" == "random_no_expert" ]; then
    echo "🧮 Selection: Random"
elif [ "${MAX_DEPTH}" -eq 1 ]; then
    echo "🧮 Selection: Sampling (no tree, PH-UCT)"
elif [ -n "${UCT_FLAG}" ]; then
    echo "🧮 UCT: Standard UCT"
else
    echo "🧮 UCT: PH-UCT (Entropy-based) - DEFAULT"
fi
echo "🌡️  Temperature: ${TEMPERATURE}"
echo "🔄 Iterations: ${NUM_ITERATIONS}"
echo ""

# Build command
CMD="python test_mcts_folding_ablation.py ${START_IDX} ${END_IDX} --mode ${MODE} --num_iterations ${NUM_ITERATIONS} --max_depth ${MAX_DEPTH} --temperature ${TEMPERATURE} ${UCT_FLAG}"
if [ -n "${EXPERT_ID}" ]; then
    CMD="${CMD} --single_expert_id ${EXPERT_ID}"
fi

# Run MCTS folding ablation
echo "🔄 Running command:"
echo "   ${CMD}"
echo ""

START_TIME=$(date +%s)

eval ${CMD}

EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✅ Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}, batch ${BATCH_ID}) completed successfully in ${DURATION}s"
else
    echo "❌ Task ${SLURM_ARRAY_TASK_ID} (${CONFIG_NAME}, batch ${BATCH_ID}) failed with exit code ${EXIT_CODE}"
fi

echo "📅 $(date)"
echo ""

exit ${EXIT_CODE}
