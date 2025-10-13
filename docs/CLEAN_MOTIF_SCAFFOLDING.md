# Clean Motif Scaffolding with MCTS

This document describes the clean, focused implementation of motif scaffolding with MCTS optimization.

## Overview

The clean implementation follows the correct approach:

1. **Baseline Generation**: Use `_motif.pdb` files for proper DPLM-2 tokenization
2. **MCTS Optimization**: pLDDT masking + multi-expert rollout
3. **Correct Evaluation**: motif-RMSD + scTM scoring
4. **pH-UCT Selection**: Entropy-aware selection for diversity

## Template Format

```
<cls>[scaffold_struct(partial_mask)][Motif_struct][scaffold_struct(partial_mask)]<sep>[scaffold_aa(partial_mask)][Motif_aa][scaffold_aa(partial_mask)]<eos>
```

## Key Features

### 1. Proper Data Loading
- Uses `_motif.pdb` files from `data-bin/scaffolding-pdbs/`
- Extracts motif sequence and coordinates
- Converts coordinates to DPLM-2 structure tokens
- Finds motif positions in reference sequences

### 2. Clean Baseline Generation
- Template: `<cls>[masked_struct][MOTIF_STRUCT][masked_struct]<sep>[masked_aa][MOTIF_AA][masked_aa]<eos>`
- Uses DPLM-2 150M for initial scaffold generation
- Preserves motif structure tokens from PDB

### 3. MCTS with pLDDT Masking
- **Selection**: pH-UCT with entropy awareness
- **Expansion**: pLDDT masking with quantile fallback
- **Simulation**: Multi-expert rollout
- **Backpropagation**: Standard MCTS update

### 4. Correct Reward Function
- **Motif Preservation**: Strict requirement (0 if not preserved)
- **Motif-RMSD**: Between predicted and reference motif coordinates
- **scTM**: Structure comparison between generated and reference
- **Combined**: `0.4 * rmsd_score + 0.6 * sctm_score`

## Usage

### Basic Usage
```bash
python test_clean_motif_scaffolding.py
```

### Full Usage
```bash
python tests/test_motif_scaffolding_ablation.py --mode clean --num_motifs 3 --mcts_iterations 10
```

### With External Experts
```bash
python tests/test_motif_scaffolding_ablation.py --mode clean --experts dplm2,proteinea,flowflow
```

## Architecture

### MotifScaffoldingData
Clean data structure containing:
- `motif_sequence`: AA sequence from `_motif.pdb`
- `motif_structure_tokens`: Structure tokens from coordinates
- `motif_coordinates`: 3D coordinates (L, 3, 3) format
- `reference_sequence`: Full sequence from `_clean.pdb`
- `reference_coordinates`: Full coordinates for scTM
- `motif_positions`: Where motif appears in reference

### MCTSNode
MCTS node for motif scaffolding:
- `sequence`: Current sequence
- `structure_tokens`: Current structure tokens
- `masked_positions`: Positions masked for improvement
- `expert_entropies`: For pH-UCT selection

### MotifScaffoldingMCTS
Main MCTS class:
- `load_motif_data()`: Load from `_motif.pdb` files
- `generate_baseline()`: Create initial scaffold
- `search()`: Run MCTS optimization
- `_calculate_reward()`: Motif-RMSD + scTM evaluation

## Key Improvements

1. **Separation of Concerns**: Dedicated class for motif scaffolding
2. **Correct Data Format**: Uses actual PDB files instead of mock data
3. **Proper Template**: Follows DPLM-2 format exactly
4. **Real Evaluation**: Uses actual structural metrics
5. **Clean Code**: No confusion with other tasks

## Comparison with Legacy

| Aspect | Legacy Implementation | Clean Implementation |
|--------|----------------------|---------------------|
| Data Source | Mock/simplified data | Real `_motif.pdb` files |
| Template | Mixed with inverse folding | Dedicated motif format |
| Masking | Random or fake pLDDT | Real pLDDT with quantile fallback |
| Evaluation | AAR (wrong for motif) | Motif-RMSD + scTM (correct) |
| Selection | Standard UCB1 | pH-UCT with entropy |
| Code Structure | Mixed with other tasks | Dedicated clean class |

## File Structure

```
mcts_diffusion_finetune/
├── core/
│   ├── motif_scaffolding_mcts.py    # Clean implementation
│   └── sequence_level_mcts.py       # Inverse folding only
├── tests/
│   └── test_motif_scaffolding_ablation.py  # Updated test
├── test_clean_motif_scaffolding.py  # Simple test script
└── CLEAN_MOTIF_SCAFFOLDING.md      # This file
```

## Future Enhancements

1. **More External Experts**: Add RFDiffusion, ProteinMPNN when available
2. **Better Structure Prediction**: Use AlphaFold3 or other advanced models
3. **Multi-motif Support**: Handle multiple motifs in single scaffold
4. **Design Constraints**: Add additional biological constraints
5. **Parallel Processing**: Optimize for multiple motifs simultaneously

## Troubleshooting

### Common Issues

1. **ESMFold Loading**: If ESMFold fails to load, pLDDT masking falls back to random
2. **Structure Conversion**: If coordinate→token conversion fails, uses mask tokens
3. **External Experts**: If not available, MCTS runs with DPLM-2 only
4. **Memory Issues**: Reduce `num_motifs` or `mcts_iterations` if OOM

### Debug Mode
Add `--save_results results.json` to save detailed results for analysis.

## Citation

If using this implementation, please cite:
- DPLM-2 paper for the base model
- Any external expert models used
- This implementation for the MCTS motif scaffolding approach





