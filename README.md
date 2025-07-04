# MCTS-Guided Diffusion Finetuning for Inverse Folding

This repository implements Monte Carlo Tree Search (MCTS) guided finetuning for inverse folding using diffusion-based protein language models (DPLM-2).

## Overview

The project implements two different MCTS approaches for inverse folding:

1. **Sequence-Level MCTS** (`sequence_level_mcts.py`): Each node represents a complete candidate sequence
2. **Position-Level MCTS** (`position_level_mcts.py`): Each node represents a partial sequence with masked positions, using plDDT scores to guide unmasking

## Key Components

### Core MCTS Implementations

- `sequence_level_mcts.py` - Sequence-level MCTS where each node is a complete sequence
- `position_level_mcts.py` - Position-level MCTS with plDDT masking guidance
- `compare_mcts_approaches.py` - Comparison script to evaluate both approaches

### Supporting Modules

- `protein_utils.py` - Protein structure utilities and metrics computation
- `dplm_inverse_folding.py` - DPLM-2 model integration for inverse folding
- `prototype_invfold_finetune.py` - Main prototype pipeline
- `test_initial_rewards.py` - Testing initial reward distributions

### Legacy Files (Kept for Reference)

- `mcts_search.py` - Original MCTS implementation (superseded by new approaches)

## Installation

```bash
# Create conda environment
conda create -n mcts_dplm python=3.9
conda activate mcts_dplm

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Sequence-Level MCTS

```python
from sequence_level_mcts import SequenceLevelMCTS
from protein_utils import create_mock_structure_no_sequence

# Create test structure
structure = create_mock_structure_no_sequence(length=50)

# Initialize MCTS
mcts = SequenceLevelMCTS(
    model=model,
    tokenizer=tokenizer,
    max_depth=5,
    num_simulations=50,
    exploration_constant=1.414,
    temperature=1.0
)

# Run search
best_sequence, best_reward = mcts.search(structure, target_length=50)
```

### Running Position-Level MCTS

```python
from position_level_mcts import PositionLevelMCTS

# Initialize MCTS with plDDT masking
mcts = PositionLevelMCTS(
    model=model,
    tokenizer=tokenizer,
    max_depth=10,
    num_simulations=100,
    exploration_constant=1.414,
    temperature=1.0,
    plddt_threshold=0.7,
    max_unmask_per_step=3
)

# Run search
best_sequence, best_reward = mcts.search(structure, target_length=50)
```

### Comparing Both Approaches

```python
from compare_mcts_approaches import MCTSComparison

# Run comparison
comparison = MCTSComparison()
result = comparison.run_comparison(target_length=50)
```

## Key Differences Between Approaches

### Sequence-Level MCTS
- **Node representation**: Complete sequences
- **Search strategy**: Generate full sequences and explore variations
- **Advantages**: Simpler implementation, good for global optimization
- **Use case**: When you want to explore complete sequence space

### Position-Level MCTS
- **Node representation**: Partial sequences with masked positions
- **Search strategy**: Unmask positions based on plDDT confidence scores
- **Advantages**: More interpretable, leverages structural confidence
- **Use case**: When you want fine-grained control over sequence generation

## Architecture

```
MCTS-Guided Inverse Folding Pipeline
├── Structure Input (PDB/mmCIF)
├── MCTS Search
│   ├── Selection (UCB1)
│   ├── Expansion (Generate candidates)
│   ├── Simulation (Evaluate sequences)
│   └── Backpropagation (Update statistics)
├── Reward Computation
│   ├── Structure-sequence compatibility
│   ├── Biophysical properties
│   └── Sequence diversity
└── Model Update (Imitation Learning)
```

## Reward Function

The reward function combines multiple factors:
- Structure-sequence compatibility (placeholder for TM-score)
- Hydrophobicity balance
- Charge neutrality
- Sequence diversity
- Length constraints

## Future Work

- [ ] Integrate actual DPLM-2 model for sequence generation
- [ ] Implement proper plDDT computation
- [ ] Add TM-score and other structure metrics
- [ ] Implement model finetuning with imitation learning
- [ ] Add support for real protein structures
- [ ] Optimize hyperparameters for different protein types

## References

- arXiv:2506.00925 - MCTS-guided protein design
- arXiv:2406.07025 - Entropy-guided drug design
- DPLM-2: Diffusion-based protein language model 