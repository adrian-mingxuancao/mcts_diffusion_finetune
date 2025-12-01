#!/bin/bash
#SBATCH --job-name=mcts_fold_pdb_cont
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-180%12
#SBATCH --output=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_pdb_cont_%A_%a.out
#SBATCH --error=/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_pdb_cont_%A_%a.err

set -euo pipefail

echo "🚀 Starting MCTS Folding PDB Ablation Study (Continuation)"
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
# Continuing from structure 190 to 449 (259 structures remaining)
# 7 configurations × 26 structure batches = 182 total tasks (last batch has 9 structures)
# REORGANIZED: Each structure batch gets all 7 configs before moving to next batch
#
# Task organization:
#   Tasks 0-6: Structures 190-200 (all 7 configs)
#   Tasks 7-13: Structures 200-210 (all 7 configs)
#   Tasks 14-20: Structures 210-220 (all 7 configs)
#   ...
#   Tasks 175-181: Structures 440-449 (all 7 configs, only 9 structures)
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
START_STRUCTURE=190
TOTAL_STRUCTURES=449

# REORGANIZED CALCULATION: Structure batch first, then config
BATCH_ID=$((SLURM_ARRAY_TASK_ID / NUM_CONFIGS))
CONFIG_ID=$((SLURM_ARRAY_TASK_ID % NUM_CONFIGS))

# Calculate structure range for this batch
START_IDX=$((START_STRUCTURE + BATCH_ID * STRUCTURES_PER_TASK))
END_IDX=$((START_IDX + STRUCTURES_PER_TASK))

# Cap at total structures
if [ ${END_IDX} -gt ${TOTAL_STRUCTURES} ]; then
    END_IDX=${TOTAL_STRUCTURES}
fi

echo "📊 Configuration Matrix:"
echo "   Batch ID: ${BATCH_ID}"
echo "   Config ID: ${CONFIG_ID}"
echo "   Structure range: ${START_IDX}-${END_IDX}"
echo ""

# Configuration names
CONFIG_NAMES=(
    "Random (MCTS-0)"
    "Single-Expert (150M)"
    "Single-Expert (650M)"
    "Single-Expert (3B)"
    "Sampling (depth=1)"
    "MCTS-PH (depth=5)"
    "MCTS-UCT (depth=5)"
)

echo "🎯 Running: ${CONFIG_NAMES[${CONFIG_ID}]}"
echo "   Structures: ${START_IDX} to ${END_IDX}"
echo ""

# Build command based on configuration
case ${CONFIG_ID} in
    0)
        # Random (MCTS-0)
        MODE="random_no_expert"
        EXTRA_ARGS=""
        ;;
    1)
        # Single-Expert (150M)
        MODE="single_expert"
        EXTRA_ARGS="--single_expert_id 1"
        ;;
    2)
        # Single-Expert (650M)
        MODE="single_expert"
        EXTRA_ARGS="--single_expert_id 0"
        ;;
    3)
        # Single-Expert (3B)
        MODE="single_expert"
        EXTRA_ARGS="--single_expert_id 2"
        ;;
    4)
        # Sampling (Multi-Expert, depth=1)
        MODE="multi_expert"
        EXTRA_ARGS="--max_depth 1"
        ;;
    5)
        # MCTS-PH (Multi-Expert, depth=5, PH-UCT)
        MODE="multi_expert"
        EXTRA_ARGS="--max_depth 5"
        ;;
    6)
        # MCTS-UCT (Multi-Expert, depth=5, standard UCT)
        MODE="multi_expert"
        EXTRA_ARGS="--max_depth 5 --use_standard_uct"
        ;;
    *)
        echo "❌ Invalid config ID: ${CONFIG_ID}"
        exit 1
        ;;
esac

# Run the ablation
echo "🔬 Executing ablation..."
echo "Command: python mcts_folding_ablation_pdb.py ${START_IDX} ${END_IDX} --mode ${MODE} ${EXTRA_ARGS}"
echo ""

python mcts_folding_ablation_pdb.py \
    ${START_IDX} \
    ${END_IDX} \
    --mode ${MODE} \
    --num_iterations 25 \
    --temperature 1.0 \
    ${EXTRA_ARGS}

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ Task completed successfully"
    echo "📅 $(date)"
else
    echo ""
    echo "❌ Task failed with exit code ${EXIT_CODE}"
    echo "📅 $(date)"
    exit ${EXIT_CODE}
fi
