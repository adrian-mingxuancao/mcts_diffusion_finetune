# MCTS-Guided DPLM-2 Performance Improvement Framework

A general framework for improving Diffusion-based Protein Language Model (DPLM-2) performance across all tasks using Monte Carlo Tree Search (MCTS) with expert rollout and compound rewards.

## 🎯 **Project Overview**

This project provides a **task-agnostic MCTS layer** on top of DPLM-2 that we use for the paper experiments across:

- **Inverse Folding** (structure → sequence)
- **Forward Folding** (sequence → structure)
- **Motif Scaffolding** (motif-conditioned design)
- **Unconditional / Conditional Generation** (used for ablations)

### 🚀 **Key Innovations**

1. **plDDT-aware masking** that targets low-confidence regions while backing off to random masking when needed.
2. **Simultaneous diffusion sampling** so multiple residues can be regenerated in one rollout.
3. **Multi-expert reward shaping** that combines DPLM-2, ProteinMPNN, and external motif experts under a unified score.
4. **UCT + Sampling planners** that we can swap per task (see the updated experiment suite below).

### 🧠 **Core Loop**

```
1. Select a frontier node with PH-UCT (entropy + novelty bonuses)
2. Mask residues using pLDDT guidance and task-specific heuristics
3. Roll out candidates via DPLM-2 (and optional external experts)
4. Score with compound rewards (TM, RMSD, biophysics, motif preservation)
5. Backpropagate rewards and repeat
```

## 🏗️ **Framework Structure**

```
mcts_diffusion_finetune/
├── core/
│   ├── sequence_level_mcts.py              # General sequence MCTS planner
│   ├── folding_mcts.py                     # Folding-specific orchestration
│   ├── motif_scaffolding_mcts.py           # Motif scaffolding search
│   ├── external_models/                    # Integrations (ProteinMPNN, RFdiffusion, etc.)
│   └── archive/                            # Historical variants kept for reference
├── utils/
│   ├── structure_converter.py              # Token ↔ coordinate helpers
│   ├── cameo_data_loader.py                # CAMEO dataset utilities
│   ├── pdb_data_loader.py                  # PDB_Date dataset helpers
│   └── reward_computation.py               # Composite reward functions
├── tests/
│   ├── uct_mcts_folding.py                 # Folding UCT campaign runner
│   ├── uct_sampling_folding.py             # Folding sampling baseline
│   ├── uct_mcts_inverse_folding.py         # Inverse folding UCT runner
│   ├── inverse_folding_sampling.py         # Inverse folding sampling baseline
│   ├── uct_mcts_motif_scaffolding.py       # Motif scaffolding UCT runner
│   ├── test_mcts_folding_ablation.py       # Folding CAMEO ablation
│   ├── mcts_folding_ablation_pdb.py        # Folding PDB_Date ablation
│   ├── mcts_tree_search_ablation.py        # Inverse folding CAMEO ablation
│   ├── mcts_tree_search_ablation_pdb.py    # Inverse folding PDB_Date ablation
│   └── test_motif_scaffolding_ablation.py  # Motif scaffolding PDB ablation
└── scripts/                                # Slurm submission helpers, data prep, visualisation
```

## 🔬 **Updated Experiment Suite (2025 Q4)**

Each core task now has four canonical experiment entry points:

| Task | UCT Planner | Sampling Baseline | CAMEO Ablation | PDB Ablation |
|------|-------------|-------------------|----------------|--------------|
| Folding | `python tests/uct_mcts_folding.py` | `python tests/uct_sampling_folding.py` | `python tests/test_mcts_folding_ablation.py` | `python tests/mcts_folding_ablation_pdb.py` |
| Inverse Folding | `python tests/uct_mcts_inverse_folding.py` | `python tests/inverse_folding_sampling.py` | `python tests/mcts_tree_search_ablation.py` | `python tests/mcts_tree_search_ablation_pdb.py` |
| Motif Scaffolding | `python tests/uct_mcts_motif_scaffolding.py` | *python tests/test_motif_scaffolding_ablation.py -mode sampling* | -- | `python tests/test_motif_scaffolding_ablation.py` |

Key takeaways from the latest runs:

- **UCT vs. Sampling**: UCT planners consistently outperform sampling-only baselines on TM-score and composite rewards across folding and inverse folding.
- **CAMEO vs. PDB_Date**: Ablations on curated CAMEO targets emphasize model generalisation, while PDB_Date highlights behaviour on larger backbones.
- **Motif Scaffolding**: PDB motif library is now wired into the ablation runner; we are finalising the sampling baseline once the non-contiguous motif templates stabilise.

## 📦 Data Sources

- **CAMEO 2022** (`data-bin/cameo2022/`): sequences and structural tokens for benchmarking inverse folding and folding. Access requires the internal preprocessing script; see `utils/cameo_data_loader.py`.
- **PDB_Date Split** (`data-bin/pdb_date/` via loaders): curated to evaluate generalisation beyond CAMEO, used in the `_pdb` ablation scripts.
- **Motif Scaffolding Library** (`data-bin/scaffolding-pdbs/`): includes motif, reference, and scaffold target files plus supporting FASTA metadata. Required for `test_motif_scaffolding_ablation.py` and the UCT motif runner.
- **Generated artefacts** (`generation-results/`, `mcts_diffusion_finetune/results/`): cached rollouts, inverse folding summaries, and TM-score traces used for paper figures.

All datasets are already referenced in the repo via absolute paths under `/home/caom/AID3/dplm/`. Update the loaders if you relocate the project.

## 🔧 Environment Setup

```bash
srun --partition=general --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=2:00:00 --pty bash

# Activate environment  
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"

# Set CUDA path
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Set cache directories to /net/scratch/ to avoid disk quota issues
export HF_HOME=/net/scratch/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/net/scratch/caom/.cache/huggingface/transformers
export TORCH_HOME=/net/scratch/caom/.cache/torch

# Test all server health endpoints
# curl http://localhost:8080/health
# curl http://localhost:8081/health
# curl http://localhost:8082/health

# Quick smoke tests
cd /home/caom/AID3/dplm/mcts_diffusion_finetune
python tests/uct_mcts_folding.py --mode single_expert --expert_id 0 --start 0 --end 1
python tests/mcts_tree_search_ablation.py --mode random_no_expert --start 0 --end 1
```

## 📊 Evaluation Protocol

Evaluations now span the full multi-model ecosystem we use in the paper:

1. **Candidate Generation**  
   - DPLM-2 variants (150M / 650M / 3B) for sequence rollouts.  
   - External experts (ProteinMPNN, RFdiffusion, FlowFlow, Proteina) optionally contribute proposals for ensemble rollouts, especially in motif scaffolding.

2. **Structure Realisation**  
   - Sequences are folded with **ESMFold** (via `EvalRunner`) for forward folding and inverse-folding validation.  
   - RFdiffusion / FlowFlow outputs are converted back into sequence-space baselines when used as comparison models.

3. **Scoring & Metrics**  
   - **Structural**: TM-score, RMSD, scTM, motif RMSD (for scaffolding).  
   - **Confidence**: pLDDT, entropy/novelty bonuses, ensemble disagreement.  
   - **Biophysical**: heuristics for charge/hydrophobicity plus ProteinMPNN log-probabilities when available.

4. **Result Tracking**  
   - All runs emit JSON summaries under `mcts_diffusion_finetune/results/` (one-shot analyses, UCT/sampling comparisons, ablation tables).  
   - Slurm wrappers gather per-task CSVs for the paper figures and ablation plots.

As a trend, UCT planners give the highest composite scores, while sampling baselines serve as ablation anchors to show the contribution of search and multi-expert rollouts.
