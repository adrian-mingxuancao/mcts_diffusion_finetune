# MCTS Folding Ablation Submission

## 📋 Configuration Summary
- **Iterations**: 25
- **Max Depth**: 5 (sampling uses depth=1)
- **Temperature**: 1.0
- **Time Limit**: 12 hours per task
- **Total Tasks**: 133 (7 configs × 19 batches)
- **Structures**: 190 total (10 per batch)
- **Concurrent Jobs**: 12
- **Logs**: `/home/caom/AID3/dplm/logs/mcts_folding_ablation/`

## ⚠️ IMPORTANT: Fair Comparison Design

**Tasks are organized to ensure fair comparison even if jobs don't complete!**

- Each structure batch (10 structures) gets all 7 configs before moving to next batch
- Example: Tasks 0-6 all run on structures 0-10 (one task per config)
- This means if you only complete 70 tasks, you'll have all 7 configs for 10 structure batches
- **Fair comparison guaranteed** for completed structure batches

## 🚀 Submit Command

```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
sbatch submit_mcts_folding_complete.sh
```

## 📊 Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <JOB_ID>

# View logs (replace JOB_ID and ARRAY_ID)
tail -f /home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_complete_<JOB_ID>_<ARRAY_ID>.out

# Count completed tasks
ls /home/caom/AID3/dplm/logs/mcts_folding_ablation/*.out | wc -l
```

## 🔍 Results Location

Results saved to: `/net/scratch/caom/cameo_evaluation_results/`

File pattern: `mcts_folding_ablation_results_<timestamp>.json`
