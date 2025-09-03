#!/bin/bash
#SBATCH --job-name=cameo_mcts_mode_single_e0
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/caom/cameo_mode_single_e0_%j.out
#SBATCH --error=/net/scratch/caom/cameo_mode_single_e0_%j.err

echo "CAMEO MCTS Ablation - Mode: single_expert (id=0)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Environment setup
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/huggingface_cache
export TRITON_CACHE_DIR=/net/scratch/caom/triton_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR

cd /home/caom/AID3/dplm/mcts_diffusion_finetune

timeout 28800 python tests/mcts_tree_search_ablation.py --mode single_expert --single_expert_id 0

echo "Mode single_expert (id=0) completed at: $(date)"

