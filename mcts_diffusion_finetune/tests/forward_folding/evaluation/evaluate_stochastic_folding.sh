#!/bin/bash
#SBATCH --job-name=eval_stochastic_folding
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/caom/AID3/dplm/logs/eval_stochastic_folding_%j.out
#SBATCH --error=/home/caom/AID3/dplm/logs/eval_stochastic_folding_%j.err

set -euo pipefail

echo "🚀 Starting Stochastic Folding Evaluation for All Models"
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

# List of stochastic folding result directories
# Note: FASTA files are in folding/folding/ subdirectory
FOLDING_DIRS=(
    "/home/caom/AID3/dplm/generation-results/dplm2_150m_stochastic/folding/folding"
    "/home/caom/AID3/dplm/generation-results/dplm2_650m_stochastic/folding/folding"
    "/home/caom/AID3/dplm/generation-results/dplm2_3b_stochastic/folding/folding"
)

echo "📊 Evaluating stochastic folding results from ${#FOLDING_DIRS[@]} models"
echo ""

TOTAL_EVALUATED=0
TOTAL_FAILED=0

for FASTA_DIR in "${FOLDING_DIRS[@]}"; do
    echo "════════════════════════════════════════════════════════════════"
    MODEL_NAME=$(basename $(dirname $FASTA_DIR))
    echo "🔬 Processing: ${MODEL_NAME}"
    echo "════════════════════════════════════════════════════════════════"
    echo "📁 FASTA directory: $FASTA_DIR"
    
    if [ ! -d "$FASTA_DIR" ]; then
        echo "⚠️  Directory not found, skipping..."
        echo ""
        continue
    fi
    
    FASTA_COUNT=$(find "$FASTA_DIR" -name "*.fasta" | wc -l)
    echo "📊 Found $FASTA_COUNT FASTA files"
    
    if [ $FASTA_COUNT -eq 0 ]; then
        echo "⚠️  No FASTA files found, skipping..."
        echo ""
        continue
    fi
    
    echo "🔄 Running official DPLM-2 evaluation..."
    echo "   Task: forward_folding"
    echo "   Input directory: $FASTA_DIR"
    echo ""
    
    START_TIME=$(date +%s)
    
    if python src/byprot/utils/protein/evaluator_dplm2.py -cn forward_folding \
        inference.input_fasta_dir="$FASTA_DIR" \
        2>&1 | tee "${FASTA_DIR}/../eval_${MODEL_NAME}_log.txt"; then
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo "✅ Evaluation completed in ${DURATION}s"
        echo "📁 Results saved to: ${FASTA_DIR}/forward_folding/"
        
        TOTAL_EVALUATED=$((TOTAL_EVALUATED + FASTA_COUNT))
    else
        echo "❌ Evaluation failed for $MODEL_NAME"
        TOTAL_FAILED=$((TOTAL_FAILED + FASTA_COUNT))
    fi
    
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "🎉 All Stochastic Evaluations Complete!"
echo "════════════════════════════════════════════════════════════════"
echo "📊 Summary:"
echo "   Total structures evaluated: $TOTAL_EVALUATED"
echo "   Failed: $TOTAL_FAILED"
echo ""
echo "📅 $(date)"
