# FINAL FIX - Random Mode Doing Inverse Folding Instead of Folding

## 🐛 The Bug

Random mode was generating **TWO batches** of candidates:
1. **4 structure candidates** (folding - CORRECT) ✅
2. **2 sequence candidates** (inverse folding - WRONG!) ❌

This caused the folding task to evaluate sequences with AAR/scTM instead of structure with RMSD/TM-score.

## 🔍 Root Cause

There were **TWO** random mode implementations in the code:

1. **Lines 335-412**: NEW random mode (folding)
   - Generates structure token perturbations
   - Evaluates with `_evaluate_structure_reward` (RMSD/TM-score)
   - Prints: "🎲 Random mode: generating 4 random candidates"

2. **Lines 477-503**: OLD random mode (inverse folding)  
   - Generates random amino acid sequences
   - Evaluates with `_evaluate_sequence_aar` (AAR/scTM)
   - Prints: "🎲 Random mode: generating 2 random candidates"

The OLD code was in an `else` block that ran AFTER skipping DPLM-2 experts for random mode.

## ✅ The Fix

**Deleted the OLD random mode code (lines 477-503)**

Now random mode only uses the NEW folding implementation:
- Generates 4 structure token perturbations
- Evaluates ONLY with RMSD/TM-score
- No sequence evaluation (AAR/scTM)
- Sequence stays FIXED (as it should for folding)

## 🧪 Test Command

```bash
srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=1:00:00 bash -c '
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch
export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
python test_mcts_folding_ablation.py 0 1 --mode random_no_expert --num_iterations 1 --max_depth 5
'
```

**Expected output:**
- ✅ Only ONE "🎲 Random mode: generating 4 random candidates"
- ✅ Only "📊 Folding metrics: RMSD=X.XXXÅ, TM=X.XXX"
- ❌ NO "🔍 REWARD EVAL: Sequence"
- ❌ NO "Computing real scTM with ESMFold"
- ✅ "📊 Selected 2 from 4 total candidates" (not 6!)

## 🚀 Ready to Submit

After testing, submit the full job:

```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
sbatch submit_mcts_folding_resume.sh
```

This will run all 129 tasks with proper folding evaluation!
