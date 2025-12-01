#!/bin/bash
#SBATCH --job-name=dplm2_folding_stochastic
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/caom/AID3/dplm/logs/dplm2_folding_stochastic_%A_%a.out
#SBATCH --error=/home/caom/AID3/dplm/logs/dplm2_folding_stochastic_%A_%a.err

set -euo pipefail

echo "🚀 Starting DPLM-2 Stochastic Folding Generation"
echo "📅 $(date)"
echo "🖥️  Node: $(hostname)"
echo "🔢 GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "🔢 Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo ""

# Environment setup
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH="/home/caom/AID3/dplm:${PYTHONPATH:-}"
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TRANSFORMERS_CACHE="/net/scratch/caom/.cache/huggingface/transformers"
export TORCH_HOME="/net/scratch/caom/.cache/torch"

WORKDIR="/home/caom/AID3/dplm"
cd "${WORKDIR}"

# Model selection based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    0)
        MODEL_NAME="dplm2_150m"
        MODEL_PATH="airkingbd/dplm2_150m"
        ;;
    1)
        MODEL_NAME="dplm2_650m"
        MODEL_PATH="airkingbd/dplm2_650m"
        ;;
    2)
        MODEL_NAME="dplm2_3b"
        MODEL_PATH="airkingbd/dplm2_3b"
        ;;
    *)
        echo "❌ Invalid array task ID: ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

echo "════════════════════════════════════════════════════════════════"
echo "🔬 Model: ${MODEL_NAME}"
echo "════════════════════════════════════════════════════════════════"
echo "📦 Model path: ${MODEL_PATH}"
echo ""

# Input and output paths
# For folding: use aatype.fasta (amino acid sequences)
INPUT_FASTA="/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
OUTPUT_DIR="/home/caom/AID3/dplm/generation-results/${MODEL_NAME}_stochastic/folding"

echo "📁 Input FASTA: ${INPUT_FASTA}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run stochastic generation for folding (matching MCTS config)
echo "🔄 Running stochastic generation..."
echo "   Task: forward_folding"
echo "   Sampling: stochastic (matching MCTS)"
echo "   Temperature: 1.0"
echo "   Unmasking: stochastic1.0"
echo "   Sampling strategy: annealing@2.2:1.0"
echo "   Max iterations: 150"
echo ""

START_TIME=$(date +%s)

python generate_dplm2_stochastic.py \
    --model_name "${MODEL_PATH}" \
    --task folding \
    --input_fasta_path "${INPUT_FASTA}" \
    --saveto "${OUTPUT_DIR}" \
    --max_iter 150 \
    --temperature 1.0 \
    --unmasking_strategy stochastic1.0 \
    --sampling_strategy "annealing@2.2:1.0" \
    2>&1 | tee "${OUTPUT_DIR}/../${MODEL_NAME}_folding_stochastic_log.txt"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ Generation completed in ${DURATION}s"
echo "📁 Results saved to: ${OUTPUT_DIR}"
echo ""

# Count generated FASTA files
FASTA_COUNT=$(find "${OUTPUT_DIR}" -name "*.fasta" | wc -l)
echo "📊 Generated ${FASTA_COUNT} FASTA files"

echo ""
echo "🎉 ${MODEL_NAME} Stochastic Folding Complete!"
echo "📅 $(date)"
