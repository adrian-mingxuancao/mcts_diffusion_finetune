#!/bin/bash

# Comprehensive Test Script for Both Folding and Inverse Folding MCTS Pipelines
# Usage: ./run_comprehensive_test.sh

echo "🧬 Starting Comprehensive MCTS Pipeline Test"
echo "============================================="

# Set environment variables
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch

# Change to project directory
cd /home/caom/AID3/dplm/mcts_diffusion_finetune

echo "🔧 Environment configured:"
echo "  Python: $(which python)"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

echo ""
echo "🧪 Test 1: Both Pipelines Integration Test (2 structures)"
echo "========================================================="
python tests/test_both_pipelines.py --task both --structures 0 2

echo ""
echo "🧪 Test 2: Folding Pipeline Test (3 structures)"
echo "==============================================="
python tests/test_mcts_folding_ablation.py 0 3 --mode single_expert --single_expert_id 1

echo ""
echo "🧪 Test 3: Inverse Folding Pipeline Test (3 structures)"
echo "======================================================="
python tests/mcts_tree_search_ablation.py 0 3 --mode single_expert --single_expert_id 1

echo ""
echo "✅ All tests completed! Check /net/scratch/caom/cameo_evaluation_results/ for results."
