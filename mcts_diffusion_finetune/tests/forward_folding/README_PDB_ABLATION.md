# MCTS Folding Ablation - PDB Dataset

## Overview

This directory contains scripts for running MCTS folding ablation studies on the **PDB dataset**, parallel to the CAMEO 2022 evaluation.

## Files Created

### 1. **`mcts_folding_ablation_pdb.py`**
Main ablation script adapted from `test_mcts_folding_ablation.py` with PDB data loading:

**Key Changes:**
- Uses `PDBDataLoader` instead of `CAMEODataLoader`
- Loads sequences from PDB data loader's structure objects
- Data path: `/home/caom/AID3/dplm/data-bin/PDB_date`
- Results saved to: `/net/scratch/caom/pdb_evaluation_results/`

**Everything else identical to CAMEO version:**
- Same 7 ablation configurations
- Same MCTS parameters
- Same evaluation metrics (RMSD, TM-score)
- Same baseline generation (DPLM-2 150M)

### 2. **`submit_mcts_folding_pdb.sh`**
SLURM batch submission script for running all 7 configurations:

**Configuration Matrix:**
```
7 configurations × 19 structure batches = 133 total tasks
Each batch processes 10 structures (0-10, 10-20, ..., 180-190)
```

**7 Configurations:**
- **Config 0:** Random (MCTS-0) - Random selection baseline
- **Config 1:** DPLM-2 150M (Single-Expert)
- **Config 2:** DPLM-2 650M (Single-Expert)
- **Config 3:** DPLM-2 3B (Single-Expert)
- **Config 4:** Sampling (Multi-Expert, depth=1)
- **Config 5:** MCTS-PH (Multi-Expert, depth=5, PH-UCT) ⭐ **DEFAULT**
- **Config 6:** MCTS-UCT (Multi-Expert, depth=5, standard UCT)

**Resource Allocation:**
- 1 GPU per task
- 4 CPUs per task
- 32GB RAM
- 12 hours walltime
- Max 12 concurrent jobs

### 3. **`test_pdb_interactive.sh`**
Quick interactive test script for development/debugging.

## Usage

### Interactive Testing

```bash
# Request interactive GPU node
srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash

# Run test script
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding
./test_pdb_interactive.sh
```

Or manually:
```bash
# Activate environment  
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch
export TRITON_CACHE_DIR=/net/scratch/caom/.cache/triton

cd /home/caom/AID3/dplm/mcts_diffusion_finetune/tests/forward_folding

# Test with 2 structures, single expert mode
python mcts_folding_ablation_pdb.py 0 2 --mode single_expert --single_expert_id 1 --num_iterations 10 --max_depth 3
```

### Batch Job Submission

```bash
# Submit all 133 tasks (7 configs × 19 batches)
sbatch submit_mcts_folding_pdb.sh

# Check job status
squeue -u $USER

# Monitor specific job
tail -f /home/caom/AID3/dplm/logs/mcts_folding_ablation/mcts_fold_pdb_<JOB_ID>_<TASK_ID>.out
```

### Custom Runs

```bash
# Single configuration, specific structure range
python mcts_folding_ablation_pdb.py 0 50 --mode multi_expert --num_iterations 25 --max_depth 5

# Test different expert
python mcts_folding_ablation_pdb.py 10 20 --mode single_expert --single_expert_id 2

# Use standard UCT instead of PH-UCT
python mcts_folding_ablation_pdb.py 0 10 --mode multi_expert --use_standard_uct
```

## Output Files

### Results Directory
```
/net/scratch/caom/pdb_evaluation_results/
├── mcts_folding_ablation_pdb_results_<timestamp>.json
└── mcts_folding_ablation_pdb_summary_<timestamp>.txt
```

### Log Files
```
/home/caom/AID3/dplm/logs/mcts_folding_ablation/
├── mcts_fold_pdb_<JOB_ID>_<TASK_ID>.out
└── mcts_fold_pdb_<JOB_ID>_<TASK_ID>.err
```

## Expected Results Format

### JSON Results
```json
{
  "structure_name": "1abc_A",
  "sequence_length": 150,
  "ablation_mode": "multi_expert",
  "baseline_rmsd": 12.5,
  "baseline_tmscore": 0.45,
  "final_rmsd": 8.3,
  "final_tmscore": 0.62,
  "rmsd_improvement": 4.2,
  "tmscore_improvement": 0.17,
  "improved": true,
  "search_time": 245.3
}
```

### Summary Table
```
Structure            Len   Base RMSD  Final RMSD  ΔRMSD    Base TM  Final TM  ΔTM      Improved  Time(s)
1abc_A               150   12.500     8.300       4.200    0.450    0.620     0.170    True      245.3
```

## Comparison with CAMEO Version

| Aspect | CAMEO Version | PDB Version |
|--------|---------------|-------------|
| **Script** | `test_mcts_folding_ablation.py` | `mcts_folding_ablation_pdb.py` |
| **Data Loader** | `CAMEODataLoader` | `PDBDataLoader` |
| **Data Path** | `/home/caom/AID3/dplm/data-bin/cameo2022` | `/home/caom/AID3/dplm/data-bin/PDB_date` |
| **Sequence Loading** | From `aatype.fasta` | From structure objects |
| **Results Dir** | `/net/scratch/caom/cameo_evaluation_results/` | `/net/scratch/caom/pdb_evaluation_results/` |
| **Batch Script** | `submit_mcts_folding_complete.sh` | `submit_mcts_folding_pdb.sh` |
| **Everything Else** | ✅ Identical | ✅ Identical |

## Notes

- **Data Loading:** Only difference is PDB uses `PDBDataLoader` and loads sequences from structure objects instead of separate FASTA file
- **All MCTS logic:** Completely identical to CAMEO version
- **Evaluation metrics:** Same RMSD/TM-score calculations
- **Baseline generation:** Same DPLM-2 150M baseline approach
- **7 configurations:** Identical ablation study design

## Troubleshooting

### PDB Data Loader Issues
```python
# Check if PDB data loader is working
from utils.pdb_data_loader import PDBDataLoader
loader = PDBDataLoader(data_path="/home/caom/AID3/dplm/data-bin/PDB_date")
print(f"Loaded {len(loader.structures)} structures")
```

### Sequence Loading Issues
```python
# Verify sequences are being loaded
structure = loader.get_structure_by_index(0)
print(f"Structure keys: {structure.keys()}")
print(f"Sequence: {structure.get('sequence', 'NOT FOUND')}")
```

### Memory Issues
- Reduce `--num_iterations` (default: 25)
- Reduce `--max_depth` (default: 5)
- Process fewer structures per batch

## Next Steps

1. **Test interactively** with 1-2 structures to verify PDB data loading
2. **Submit small batch** (e.g., tasks 0-6) to test all 7 configurations
3. **Monitor results** and check output files
4. **Submit full job** once validated
