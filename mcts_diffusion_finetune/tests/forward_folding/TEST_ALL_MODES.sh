#!/bin/bash
# Test all 7 configurations interactively to ensure they work

BASE_CMD='
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch
export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
'

echo "🧪 Interactive Test Commands for All 7 Configurations"
echo "======================================================"
echo ""

echo "1️⃣  Random (MCTS-0) - Config 0"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode random_no_expert --num_iterations 5 --max_depth 5'"
echo ""

echo "2️⃣  DPLM-2 150M (Single-Expert) - Config 1"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode single_expert --single_expert_id 1 --num_iterations 5 --max_depth 5'"
echo ""

echo "3️⃣  DPLM-2 650M (Single-Expert) - Config 2"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode single_expert --single_expert_id 0 --num_iterations 5 --max_depth 5'"
echo ""

echo "4️⃣  DPLM-2 3B (Single-Expert) - Config 3"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode single_expert --single_expert_id 2 --num_iterations 5 --max_depth 5'"
echo ""

echo "5️⃣  Sampling (Multi-Expert, depth=1) - Config 4"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --num_iterations 5 --max_depth 1'"
echo ""

echo "6️⃣  MCTS-PH (Multi-Expert, PH-UCT, depth=5) - Config 5 ⭐ DEFAULT"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --num_iterations 5 --max_depth 5'"
echo ""

echo "7️⃣  MCTS-UCT (Multi-Expert, Standard UCT, depth=5) - Config 6"
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '$BASE_CMD
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --num_iterations 5 --max_depth 5 --use_standard_uct'"
echo ""

echo "======================================================"
echo "💡 Quick Test: Multi-Expert (MCTS-PH)"
echo "======================================================"
echo ""
echo "srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '"
echo "export PATH=\"/net/scratch/caom/dplm_env/bin:\$PATH\""
echo "export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:\$PYTHONPATH"
echo "export HF_HOME=/net/scratch/caom/.cache/huggingface"
echo "export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers"
echo "export TORCH_HOME=/net/scratch/caom/.cache/torch"
echo "export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton"
echo "cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding"
echo "python test_mcts_folding_ablation.py 0 1 --mode multi_expert --num_iterations 5 --max_depth 5"
echo "'"
