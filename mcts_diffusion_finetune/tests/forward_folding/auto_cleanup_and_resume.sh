#!/bin/bash
# Automatically cleanup failed tasks and create resume script

LOG_DIR="/home/caom/AID3/dplm/logs/mcts_folding_ablation"
JOB_ID="569308"

echo "🔍 Analyzing job ${JOB_ID} logs..."
echo ""

# Arrays to track task status
declare -a completed
declare -a failed
declare -a missing

# Check each task
for task_id in {0..132}; do
    out_file="${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.out"
    err_file="${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.err"
    
    if [ ! -f "$out_file" ]; then
        missing+=($task_id)
    elif grep -q "SyntaxError" "$out_file" 2>/dev/null || grep -q "SyntaxError" "$err_file" 2>/dev/null; then
        failed+=($task_id)
    elif grep -q "MCTS search failed" "$out_file" 2>/dev/null; then
        failed+=($task_id)
    elif grep -q "Traceback (most recent call last)" "$out_file" 2>/dev/null; then
        failed+=($task_id)
    elif grep -q "completed successfully" "$out_file" 2>/dev/null && ! grep -q "MCTS search failed" "$out_file" 2>/dev/null; then
        completed+=($task_id)
    else
        failed+=($task_id)
    fi
done

echo "═══════════════════════════════════════════"
echo "📊 Summary:"
echo "═══════════════════════════════════════════"
echo "✅ Completed: ${#completed[@]} tasks"
echo "❌ Failed: ${#failed[@]} tasks"
echo "⚠️  Missing: ${#missing[@]} tasks"
echo ""

# Show some completed tasks
if [ ${#completed[@]} -gt 0 ]; then
    echo "✅ Sample completed tasks: ${completed[@]:0:10}"
    echo ""
fi

# Combine failed and missing for rerun
all_rerun=("${failed[@]}" "${missing[@]}")
total_rerun=${#all_rerun[@]}

if [ $total_rerun -eq 0 ]; then
    echo "🎉 All tasks completed successfully!"
    exit 0
fi

echo "📋 Tasks to rerun: $total_rerun"

# Sort and create comma-separated list
rerun_list=$(printf '%s\n' "${all_rerun[@]}" | sort -n | tr '\n' ',' | sed 's/,$//')
echo ""

# Delete failed task logs
echo "🗑️  Deleting $total_rerun failed/incomplete log files..."
for task_id in "${all_rerun[@]}"; do
    rm -f "${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.out"
    rm -f "${LOG_DIR}/mcts_fold_complete_${JOB_ID}_${task_id}.err"
done
echo "✅ Cleanup complete!"
echo ""

# Create resume submission script
RESUME_SCRIPT="/home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding/submit_mcts_folding_resume.sh"

cat > "$RESUME_SCRIPT" << 'EOFSCRIPT'
#!/bin/bash
#SBATCH --job-name=mcts_fold_resume
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=TASK_ARRAY_PLACEHOLDER%12
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
EOFSCRIPT

# Replace placeholder with actual task list
sed -i "s/TASK_ARRAY_PLACEHOLDER/$rerun_list/" "$RESUME_SCRIPT"

chmod +x "$RESUME_SCRIPT"

echo "📝 Created resume script: $RESUME_SCRIPT"
echo ""
echo "🚀 To resume, run:"
echo "   cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding"
echo "   sbatch submit_mcts_folding_resume.sh"
echo ""
echo "📊 This will rerun $total_rerun tasks with the fixed code"
