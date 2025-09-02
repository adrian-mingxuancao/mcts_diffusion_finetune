#!/bin/bash
#SBATCH --job-name=cameo_mcts_batch2
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/caom/cameo_batch2_%j.out
#SBATCH --error=/net/scratch/caom/cameo_batch2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=caom@example.com

# Batch 2: Structures 4-7
echo "CAMEO MCTS Ablation - Batch 2 (Structures 4-7)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Environment setup
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/huggingface_cache
export TRITON_CACHE_DIR=/net/scratch/caom/triton_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR

cd /home/caom/AID3/dplm

# Run batch 2 (structures 4-7)
timeout 28800 python -m mcts_diffusion_finetune.tests.mcts_tree_search_ablation 4 8

echo "Batch 2 completed at: $(date)"
