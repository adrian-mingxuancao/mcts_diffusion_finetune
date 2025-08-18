# MCTS Framework Tests

This directory contains the main test scripts for the MCTS-guided DPLM-2 framework.

## 📋 Available Tests

### 🟢 **Primary Test**
- **`test_mcts_with_real_data.py`** - Main comprehensive test
  - Uses real CAMEO 2022 protein structures
  - Includes DPLM-2 baseline comparison
  - Evaluates scTM-score, AAR, RMSD, pLDDT
  - Shows MCTS vs baseline improvements
  - **Usage**: `python tests/test_mcts_with_real_data.py`

### 🟡 **Specialized Tests**  
- **`test_masking_strategies.py`** - Masking strategy comparison
  - Compares baseline, random masking, plDDT masking
  - Tests different masking percentages
  - Useful for masking strategy research
  - **Usage**: `python tests/test_masking_strategies.py`

## 🚀 Quick Start

For most users, run the main comprehensive test:

```bash
# Get compute node with GPU
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=2:00:00 --pty bash

# Run main test
cd /home/caom/AID3/dplm/mcts_diffusion_finetune
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH
python tests/test_mcts_with_real_data.py
```

## 📊 Expected Output

The main test provides:
- DPLM-2 baseline performance (scTM, AAR, RMSD)
- MCTS optimized performance  
- Direct improvement comparisons
- Individual structure analysis
- Performance summary statistics

## 🧬 Test Data

Tests use real protein structures from:
- **CAMEO 2022**: 17 high-quality protein structures
- **Size range**: 50-200 residues for optimal testing
- **Location**: `/net/scratch/caom/dplm_datasets/`

## 🔧 Environment

Tests require:
- DPLM environment: `/net/scratch/caom/dplm_env/`
- CUDA support for ESMFold evaluation
- Real protein datasets downloaded



