# MCTS Folding Ablation - Resume Status

## 🔧 Bug Fixed
**Issue**: Syntax error in `sequence_level_mcts.py` line 368
- Changed `elif self.ablation_mode != "random_no_expert":` → `if self.ablation_mode != "random_no_expert":`
- Also fixed pLDDT estimation crash when coords is None

## 📊 Job 569308 Analysis

### Completed Tasks: 4
- Task 1: DPLM-2 150M (Single-Expert), batch 0
- Task 2: DPLM-2 650M (Single-Expert), batch 0  
- Task 3: DPLM-2 3B (Single-Expert), batch 0
- Task 4: Sampling (Multi-Expert), batch 0

### Failed/Missing Tasks: 129
- Task 0 had "MCTS search failed" error (old random mode bug)
- Tasks 5-132: Either syntax errors or never started

### Cleanup Actions
✅ Deleted all 129 failed/incomplete log files (.out and .err)
✅ Created resume script with fixed code

## 🚀 Resume Instructions

```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
sbatch submit_mcts_folding_resume.sh
```

### Resume Script Details
- **Script**: `submit_mcts_folding_resume.sh`
- **Tasks**: 129 (0,5-132)
- **Array**: `--array=0,5,6,7,...,132%12`
- **Concurrent**: 12 jobs at a time
- **Time**: 12 hours per task
- **Same config**: 7 configs × 19 batches, organized for fair comparison

## 📋 What Will Run

The resume script will rerun:
- **Task 0**: Random (MCTS-0), structures 0-10 ✨ **With fixed random mode**
- **Tasks 5-6**: Remaining configs for batch 0 (structures 0-10)
- **Tasks 7-132**: All configs for batches 1-18 (structures 10-190)

## ✅ Expected Results

After completion, you should have:
- **133 total tasks** completed
- **7 configurations** × **19 structure batches**
- **190 structures** per configuration
- **Fair comparison** data for all configs

## 🔍 Monitor Progress

```bash
# Check running jobs
squeue -u $USER

# Check completion status
for i in {0..132}; do 
  if [ -f "/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_*_${i}.out" ]; then
    if grep -q "completed successfully" "/home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_*_${i}.out" 2>/dev/null; then
      echo "✅ Task $i"
    fi
  fi
done | wc -l
```

## 📁 Results Location

Results will be saved to:
- `/net/scratch/caom/cameo_evaluation_results/mcts_folding_ablation_results_*.json`
- `/net/scratch/caom/cameo_evaluation_results/mcts_folding_ablation_summary_*.txt`
