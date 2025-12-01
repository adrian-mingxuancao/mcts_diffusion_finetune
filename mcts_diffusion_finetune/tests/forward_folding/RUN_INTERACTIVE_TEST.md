# Interactive Testing Commands

## Setup Environment (run once)
```bash
srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash

# Inside the interactive session:
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch
export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton

cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
```

## Test 1: Sampling (depth=1, no tree growing)
```bash
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --max_depth 1 --num_iterations 5
```
**Expected**: Only depth 0 and 1 nodes, PH-UCT by default

## Test 2: MCTS-PH (depth=10, PH-UCT - DEFAULT)
```bash
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --max_depth 10 --num_iterations 5
```
**Expected**: Multiple depths (0,1,2,3...), PH-UCT with entropy

## Test 3: MCTS-UCT (depth=10, standard UCT)
```bash
python test_mcts_folding_ablation.py 0 1 --mode multi_expert --max_depth 10 --num_iterations 5 --use_standard_uct
```
**Expected**: Multiple depths (0,1,2,3...), standard UCT without entropy

## What to Look For

### Sampling (depth=1):
- All "Selected node at depth" messages should show depth 0 or 1 ONLY
- No tree growing beyond root level

### MCTS-PH (default):
- "Selected node at depth" messages show various depths: 0, 1, 2, 3, etc.
- Tree grows progressively deeper
- Uses PH-UCT with entropy bonuses

### MCTS-UCT:
- "Selected node at depth" messages show various depths: 0, 1, 2, 3, etc.
- Tree grows progressively deeper
- Uses standard UCT (no entropy bonuses)
