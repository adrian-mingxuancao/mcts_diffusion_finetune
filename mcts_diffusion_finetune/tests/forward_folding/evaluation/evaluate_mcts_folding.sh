#!/bin/bash
#SBATCH --job-name=eval_mcts_folding
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/caom/AID3/dplm/logs/eval_mcts_folding_%j.out
#SBATCH --error=/home/caom/AID3/dplm/logs/eval_mcts_folding_%j.err

set -euo pipefail

echo "🚀 Starting MCTS Folding Evaluation"
echo "📅 $(date)"
echo "🖥️  Node: $(hostname)"
echo ""

# Environment setup
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH="/home/caom/AID3/dplm:${PYTHONPATH:-}"
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TRANSFORMERS_CACHE="/net/scratch/caom/.cache/huggingface/transformers"
export TORCH_HOME="/net/scratch/caom/.cache/torch"

WORKDIR="/home/caom/AID3/dplm"
cd "${WORKDIR}"

FASTA_DIR="/home/caom/AID3/dplm/generation-results/mcts-me/folding"

echo "════════════════════════════════════════════════════════════════"
echo "🔬 Evaluating MCTS Multi-Expert Folding Results"
echo "════════════════════════════════════════════════════════════════"
echo "📁 FASTA directory: $FASTA_DIR"

# Count FASTA files
FASTA_COUNT=$(find "$FASTA_DIR" -name "*.fasta" | wc -l)
echo "📊 Found $FASTA_COUNT FASTA files"

if [ $FASTA_COUNT -eq 0 ]; then
    echo "❌ No FASTA files found!"
    exit 1
fi

echo ""
echo "🔄 Running official DPLM-2 evaluation..."
echo "   Task: forward_folding"
echo "   Input directory: $FASTA_DIR"
echo ""

START_TIME=$(date +%s)

python src/byprot/utils/protein/evaluator_dplm2.py -cn forward_folding \
    inference.input_fasta_dir="$FASTA_DIR" \
    2>&1 | tee "/home/caom/AID3/dplm/generation-results/mcts-me/eval_mcts-me_log.txt"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ Evaluation completed in ${DURATION}s"
echo "📁 Results saved to: /home/caom/AID3/dplm/generation-results/mcts-me/eval_mcts-me/"
echo ""

# Check for metrics file
METRICS_FILE="/home/caom/AID3/dplm/generation-results/mcts-me/folding/forward_folding/forward_fold_metrics.csv"
if [ -f "$METRICS_FILE" ]; then
    echo "📊 Metrics file found:"
    echo "   $METRICS_FILE"
    echo ""
    echo "📈 Metrics preview:"
    head -20 "$METRICS_FILE"
fi

echo ""
echo "🎉 MCTS Folding Evaluation Complete!"
echo "📅 $(date)"
