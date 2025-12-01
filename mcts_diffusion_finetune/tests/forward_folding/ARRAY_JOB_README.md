# MCTS Folding Ablation Study

## 📋 Overview

Complete MCTS folding ablation study with 7 configurations on CAMEO 2022 dataset.

## 🚀 Quick Start - Complete Ablation (133 tasks) ⭐ **RECOMMENDED**
```bash
sbatch submit_mcts_folding_complete.sh
```
- **Tasks**: 133 (0-132)
- **Configurations**: 7 complete ablation conditions
- **Temperature**: 1.0
- **Structures**: 10 per task × 19 batches = 190 structures per configuration
- **Concurrent**: 12 tasks at a time
- **Time**: 24 hours per task

**7 Ablation Configurations:**
1. **Random (MCTS-0)** - Random selection baseline
2. **DPLM-2 150M (Single-Expert)** - Single model baseline
3. **DPLM-2 650M (Single-Expert)** - Single model baseline
4. **DPLM-2 3B (Single-Expert)** - Single model baseline
5. **Sampling (Multi-Expert)** - Pure sampling (depth=1, no tree, PH-UCT)
6. **MCTS-PH (Multi-Expert)** - PH-UCT with entropy (depth=5) **DEFAULT**
7. **MCTS-UCT (Multi-Expert)** - Standard UCT (depth=5)

## 📊 Task Breakdown

**⚠️ IMPORTANT: Tasks organized for fair comparison!**

Tasks are organized so each structure batch gets all 7 configs before moving to the next batch.
This ensures you have complete data for comparison even if jobs don't finish.

| Task ID | Structure Range | Configs Included |
|---------|----------------|------------------|
| 0-6     | 0-10           | All 7 configs |
| 7-13    | 10-20          | All 7 configs |
| 14-20   | 20-30          | All 7 configs |
| 21-27   | 30-40          | All 7 configs |
| 28-34   | 40-50          | All 7 configs |
| 35-41   | 50-60          | All 7 configs |
| 42-48   | 60-70          | All 7 configs |
| 49-55   | 70-80          | All 7 configs |
| 56-62   | 80-90          | All 7 configs |
| 63-69   | 90-100         | All 7 configs |
| 70-76   | 100-110        | All 7 configs |
| 77-83   | 110-120        | All 7 configs |
| 84-90   | 120-130        | All 7 configs |
| 91-97   | 130-140        | All 7 configs |
| 98-104  | 140-150        | All 7 configs |
| 105-111 | 150-160        | All 7 configs |
| 112-118 | 160-170        | All 7 configs |
| 119-125 | 170-180        | All 7 configs |
| 126-132 | 180-190        | All 7 configs |

**Config mapping within each batch:**
- Task X+0: Random (MCTS-0)
- Task X+1: DPLM-2 150M
- Task X+2: DPLM-2 650M
- Task X+3: DPLM-2 3B
- Task X+4: Sampling (depth=1)
- Task X+5: MCTS-PH (depth=5)
- Task X+6: MCTS-UCT (depth=5)

## 🔧 Configuration

### MCTS Parameters
- **Iterations**: 25
- **Max Depth**: 5 (sampling uses depth=1)
- **Baseline**: DPLM-2 150M with fixed seed (42)
- **Experts**: 3 DPLM-2 models (650M, 150M, 3B)
- **Temperature**: 1.0 (stochastic sampling)
- **Logs**: `/home/caom/AID3/dplm/logs/mcts_folding_ablation/`

### UCT Selection Strategies

#### Standard UCT (Default)
```
UCB = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
```
- Pure exploitation-exploration tradeoff
- No model uncertainty consideration
- Faster convergence
- May miss high-uncertainty regions

#### PH-UCT (Entropy-based) (`--use_ph_uct`)
```
UCB = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a)) + α * H(s,a) + β * novelty(s,a)
```
- Adds entropy bonus for model uncertainty
- Adds novelty bonus for exploration
- Better exploration of uncertain regions
- May find better solutions in complex landscapes

**When to use PH-UCT:**
- Complex protein structures with high uncertainty
- When baseline quality is poor
- When you want thorough exploration
- Research/discovery mode

**When to use Standard UCT:**
- Fast optimization needed
- Baseline already good quality
- Production/deployment mode
- Limited computational budget

### Resource Allocation
- **GPU**: 1 per task
- **CPUs**: 4 per task
- **Memory**: 32GB per task
- **Walltime**: 24 hours per task

### Concurrent Execution
- **Array script**: 6 tasks at a time (`%6`)
- **Modes script**: 12 tasks at a time (`%12`)

## 📁 Output Locations

### Log Files
```
/home/caom/AID3/dplm/logs/mcts_folding_ablation_<JOB_ID>_<TASK_ID>.out
/home/caom/AID3/dplm/logs/mcts_folding_ablation_<JOB_ID>_<TASK_ID>.err
```

### Results Files
```
/net/scratch/caom/cameo_evaluation_results/mcts_folding_ablation_results_<TIMESTAMP>.json
/net/scratch/caom/cameo_evaluation_results/mcts_folding_ablation_summary_<TIMESTAMP>.txt
```

## 🎯 Expected Results

Each task will generate:
- ✅ Baseline metrics (RMSD, TM-score, Reward)
- ✅ Final optimized metrics
- ✅ Improvement deltas
- ✅ Search time and tree statistics
- ✅ JSON results file
- ✅ Summary table

## 📈 Monitoring Progress

### Check job status
```bash
squeue -u $USER | grep mcts_folding
```

### Check specific task output
```bash
tail -f /home/caom/AID3/dplm/logs/mcts_folding_ablation_<JOB_ID>_<TASK_ID>.out
```

### Count completed tasks
```bash
grep "✅.*completed successfully" /home/caom/AID3/dplm/logs/mcts_folding_ablation_*.out | wc -l
```

### Check for failures
```bash
grep "❌.*failed" /home/caom/AID3/dplm/logs/mcts_folding_ablation_*.out
```

## 🔄 Resubmitting Failed Tasks

If specific tasks fail, resubmit only those:
```bash
# Resubmit task 5 only
sbatch --array=5 submit_mcts_folding_ablation_array.sh

# Resubmit multiple specific tasks
sbatch --array=5,7,12 submit_mcts_folding_ablation_array.sh
```

## 📊 Post-Processing

After all tasks complete, aggregate results:
```bash
# Combine all JSON results
python aggregate_folding_results.py \
    --results_dir /net/scratch/caom/cameo_evaluation_results \
    --pattern "mcts_folding_ablation_results_*.json" \
    --output final_folding_ablation_summary.json
```

## ⚙️ Customization

### Modify MCTS parameters
Edit the `python test_mcts_folding_ablation.py` command in the script:
```bash
python test_mcts_folding_ablation.py \
    ${START_IDX} ${END_IDX} \
    --mode multi_expert \
    --num_iterations 100 \      # Increase iterations
    --max_depth 15 \            # Increase depth
    --temperature 0.8 \         # Lower temperature (more deterministic)
    --use_ph_uct                # Enable PH-UCT (entropy-based)
```

### Available Command-Line Options
```bash
python test_mcts_folding_ablation.py START END [OPTIONS]

Positional Arguments:
  START                    Start structure index (inclusive)
  END                      End structure index (exclusive)

Options:
  --mode {random_no_expert,single_expert,multi_expert,all}
                          Ablation mode (default: all)
  --single_expert_id {0,1,2}
                          Expert ID for single_expert mode
                          0=650M, 1=150M, 2=3B
  --num_iterations INT    Number of MCTS iterations (default: 50)
  --max_depth INT         Maximum tree depth (default: 10)
  --use_ph_uct           Enable PH-UCT with entropy bonuses
  --temperature FLOAT     Generation temperature (default: 1.0)
                          Lower = more deterministic
                          Higher = more stochastic
```

### Adjust resource allocation
Modify SBATCH directives:
```bash
#SBATCH --mem=64G              # More memory
#SBATCH --time=48:00:00        # More time
#SBATCH --cpus-per-task=8      # More CPUs
```

### Change concurrent task limit
```bash
#SBATCH --array=0-18%10        # Run 10 tasks at once
```

## 🎉 Complete Ablation Study

To run the full comprehensive study with all modes:
```bash
# Submit all modes
sbatch submit_mcts_folding_ablation_modes.sh

# Expected completion time: ~24-48 hours
# Total GPU hours: 76 tasks × 24 hours = 1,824 GPU hours
# With 12 concurrent: ~152 hours wall time (~6.3 days)
```

## 📝 Notes

- Each task processes 10 structures independently
- Results are saved incrementally (safe for interruptions)
- Failed tasks can be resubmitted individually
- Baseline uses DPLM-2 150M with fixed seed for reproducibility
- pLDDT scores extracted from structure tokenizer detokenization
