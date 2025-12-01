#!/bin/bash
# Interactive test script for PDB folding ablation

srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash -c '

# Activate environment  
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"

# Set CUDA path
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Set cache directories to /net/scratch/ to avoid disk quota issues
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch
export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton

cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding

# Test with first 2 structures, single expert mode (DPLM-2 150M)
echo "🧪 Testing PDB folding ablation with 2 structures..."
python mcts_folding_ablation_pdb.py 0 2 --mode single_expert --single_expert_id 1 --num_iterations 10 --max_depth 3 '
