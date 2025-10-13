#!/bin/bash
#SBATCH --job-name=motif_scaffolding_real
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --partition=
#SBATCH --output=/net/scratch/caom/logs/motif_scaffolding_real_%j.out
#SBATCH --error=/net/scratch/caom/logs/motif_scaffolding_real_%j.err

set -euo pipefail

echo "==> Activating DPLM environment"
export PATH="/net/scratch/caom/dplm_env/bin:${PATH}"

echo "==> Configuring CUDA/torch caches"
export PYTHONPATH="/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:${PYTHONPATH:-}"
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch

echo "==> Setting workspace"
cd /home/caom/AID3/dplm

RESULTS_DIR=${RESULTS_DIR:-/net/scratch/caom/motif_scaffolding_results}
mkdir -p "${RESULTS_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_JSON="${RESULTS_DIR}/baseline_all_real_${TIMESTAMP}.json"

NUM_MOTIFS=${NUM_MOTIFS:-17}

echo "==> Launching real-model motif scaffolding baseline sweep"
python mcts_diffusion_finetune/tests/test_motif_scaffolding_ablation.py \
  --mode baseline_all \
  --num_motifs "${NUM_MOTIFS}" \
  --mcts_iterations 0 \
  --use_real_models \
  --save_results "${RESULTS_JSON}"

echo "==> Completed baseline sweep. Results: ${RESULTS_JSON}"
