# MCTS-DPLM2 Test Suite

Organized test scripts for MCTS-guided protein generation experiments.

## 📁 Directory Structure

```
tests/
├── forward_folding/        # Sequence → Structure (folding)
├── inverse_folding/        # Structure → Sequence (design)
├── motif_scaffolding/      # Motif-constrained design
├── ablations/              # Ablation studies
├── comprehensive/          # Full evaluation scripts
└── README.md               # This file
```

## 🧪 Test Categories

### 1. Forward Folding (7 scripts)
**Task:** Predict structure from sequence

**Main scripts:**
- `test_mcts_folding_ablation.py` - Lead optimization ablation studies 
- `test_denovo_folding_mcts_fixed.py` - MCTS folding with multi-expert search
- `generate_dplm2_stochastic.py` - Stochastic sampling baseline
- `uct_mcts_folding.py` - UCT-based MCTS folding
- `evaluate_*_folding.sh` - Evaluation scripts
- `extract_*_folding_results.py` - Results extraction

**Dataset:** CAMEO2022 (183 sequences)

### 2. Inverse Folding (7 scripts)
**Task:** Design sequence from structure

**Main scripts:**
- `test_denovo_mcts_corrected.py` - De novo inverse folding
- `inverse_folding_sampling*.py` - Sampling baselines
- `uct_mcts_inverse_folding*.py` - UCT-based design

**Dataset:** CAMEO2022, PDB structures

### 3. Motif Scaffolding (2 scripts)
**Task:** Design around fixed motif

**Main scripts:**
- `test_mcts_scaffold_ablation.py` - Lead optimization ablation studies
- `test_motif_scaffolding_ablation.py` - Full motif scaffolding pipeline
- `uct_mcts_motif_scaffolding.py` - UCT-based scaffolding

**Dataset:** Motif-scaffolding benchmarks

### 4. Ablations (3 scripts)
**Task:** Ablation studies

**Main scripts:**
- `test_mcts_folding_ablation.py` - Lead optimization ablation studies 
- `mcts_tree_search_ablation*.py` - MCTS parameter ablations
- Various depth/simulation studies

### 5. Comprehensive (3 scripts)
**Task:** Full evaluations

**Main scripts:**
- `comprehensive_cameo_evaluation.py` - Full CAMEO evaluation
- `test_fixed_baseline_concept.py` - Baseline testing
- `test_pregenerated_baselines_simple.py` - Pregenerated baseline tests

## 🚀 Quick Start

### Forward Folding Example
```bash
cd /home/caom/AID3/dplm/mcts_diffusion_finetune
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"

# MCTS folding
python tests/forward_folding/test_denovo_folding_mcts_fixed.py

# Stochastic baseline
bash tests/forward_folding/submit_dplm2_folding_stochastic.sh

# Evaluate
bash tests/forward_folding/evaluate_all_folding_results.sh
python tests/forward_folding/extract_all_folding_results.py
```

### Inverse Folding Example
```bash
# De novo inverse folding
python tests/inverse_folding/test_denovo_mcts_corrected.py

# UCT-based inverse folding
python tests/inverse_folding/uct_mcts_inverse_folding.py
```

## 📊 Latest Results

**Forward Folding (CAMEO2022, 163 proteins):**
- DPLM-2 650M: TM=0.754, RMSD=7.83Å, Reward=0.418
- MCTS Multi-Expert: TM=0.732, RMSD=8.53Å, Reward=0.391
- Results in: `/home/caom/AID3/dplm/folding_evaluation_summary/`

## 🔧 Environment

```bash
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"
export PYTHONPATH="/home/caom/AID3/dplm:${PYTHONPATH:-}"
export HF_HOME="/net/scratch/caom/.cache/huggingface"
```

## 📝 Notes

- All scripts use ESM-patched DPLM-2 models
- Reward formula: R = 0.6·TM + 0.4·(1-min(RMSD/10,1))
- Results saved to `/home/caom/AID3/dplm/generation-results/`



