# MCTS-Guided Diffusion Finetuning for Inverse Folding

This repository implements Monte Carlo Tree Search (MCTS) guided finetuning for inverse folding using diffusion-based protein language models (DPLM-2).

## 📁 Project Structure

```
mcts_diffusion_finetune/
├── core/                           # Core MCTS algorithms and model integration
│   ├── __init__.py                 # Core module exports
│   ├── sequence_level_mcts.py      # Main sequence-level MCTS implementation
│   └── dplm2_integration.py        # DPLM-2 model integration
├── utils/                          # Utility functions and helpers
│   ├── __init__.py                 # Utils module exports
│   ├── protein_utils.py            # Protein structure manipulation
│   ├── plddt_computation.py        # plDDT scoring functions
│   └── reward_computation.py       # Enhanced length-aware reward system
├── examples/                       # Example scripts and demonstrations
│   ├── demo_mcts_simulations.py    # Clear demonstration script
│   └── detailed_simulations.py     # Comprehensive simulation analysis
├── tests/                          # Unit tests (for future development)
├── docs/                           # Documentation (for future development)
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🎯 Focus on Sequence-Level MCTS

The codebase emphasizes **sequence-level MCTS** as the primary approach for inverse folding:

- **Each node represents a complete protein sequence**
- **Tree structure**: Root → DPLM-2 initial sequences → Point mutations/variations
- **Search process**: Selection → Expansion → Simulation → Backpropagation
- **Optimized for different protein lengths** (50, 200, 500 residues)

## 🧬 MCTS Node Structure and Organization

### Node Representation
```python
@dataclass
class MCTSNode:
    sequence: str           # Complete amino acid sequence
    reward: float          # Computed reward score
    visit_count: int       # Number of visits
    total_value: float     # Cumulative reward
    children: List[MCTSNode]  # Child nodes (sequence variations)
```

### Actual Tree Structure (Based on Implementation)

```
Root Node (empty sequence "")
├── Child 1 (DPLM-2 generated sequence A)
│   ├── Grandchild 1.1 (Point mutation of A: pos 5 A→V)
│   ├── Grandchild 1.2 (Point mutation of A: pos 12 L→F)
│   └── Grandchild 1.3 (Point mutation of A: pos 23 K→R)
├── Child 2 (DPLM-2 generated sequence B)
│   ├── Grandchild 2.1 (Point mutation of B: pos 8 D→E)
│   └── Grandchild 2.2 (Point mutation of B: pos 15 G→A)
└── Child 3 (DPLM-2 generated sequence C)
    ├── Grandchild 3.1 (Point mutation of C: pos 3 M→I)
    └── ...
```

**Key Process:**
1. **Root**: Empty sequence (starting point)
2. **Level 1**: DPLM-2 generates initial complete sequences
3. **Level 2+**: Point mutations of existing sequences (1-3 amino acid changes)

## 🔄 MCTS Simulation Process

### 1. **Initialization**
- Create root node with empty sequence
- Use DPLM-2 to generate initial complete sequences
- Add these as children of root node

### 2. **Selection Phase**
- Start from root node
- Use **Upper Confidence Bound (UCB1)** to select child nodes
- UCB1 formula: `value + c * sqrt(ln(parent_visits) / node_visits)`
- Traverse down the tree until reaching a leaf node

### 3. **Expansion Phase**
- Generate sequence variations from selected node
- **Point mutations**: Randomly mutate 1-3 amino acid positions
- **Fallback**: Random sequences if node is empty
- Create new child nodes for each unique sequence

### 4. **Simulation Phase**
- Evaluate the selected node using the reward function
- Reward components:
  - Structure-sequence compatibility
  - Biophysical properties (hydrophobicity, charge)
  - Sequence diversity
  - Length constraints

### 5. **Backpropagation Phase**
- Update statistics for the selected node
- Increment visit counts and update cumulative values
- Recalculate average values for future selections

## 🏆 Enhanced Reward System

### Length-Aware Reward Components

#### Small Proteins (<100 residues)
- **Structure compatibility**: 30%
- **Hydrophobicity balance**: 30%
- **Charge balance**: 25%
- **Sequence diversity**: 10%
- **Stability score**: 5%

#### Medium Proteins (100-300 residues)
- **Structure compatibility**: 40%
- **Hydrophobicity balance**: 25%
- **Charge balance**: 20%
- **Sequence diversity**: 10%
- **Stability score**: 5%

#### Large Proteins (>300 residues)
- **Structure compatibility**: 50%
- **Hydrophobicity balance**: 20%
- **Charge balance**: 15%
- **Sequence diversity**: 10%
- **Stability score**: 5%

### Reward Calculation Details

1. **Structure Compatibility**: Mock compatibility based on sequence properties (placeholder for TM-score)
2. **Hydrophobicity Balance**: Optimal hydrophobic/hydrophilic ratio with length scaling
3. **Charge Balance**: Neutrality with length-specific tolerance
4. **Sequence Diversity**: Shannon entropy + natural amino acid distribution
5. **Stability Score**: Avoids problematic patterns, promotes balanced secondary structure

## 🚀 Installation

```bash
# Create conda environment
conda create -n mcts_dplm python=3.9
conda activate mcts_dplm

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Basic Sequence-Level MCTS

```python
from core import SequenceLevelMCTS
from utils import create_mock_structure_no_sequence

# Create test structure (no sequence)
structure = create_mock_structure_no_sequence(length=200)

# Initialize MCTS
mcts = SequenceLevelMCTS(
    model=model,
    tokenizer=tokenizer,
    max_depth=6,
    num_simulations=100,
    exploration_constant=1.414
)

# Run search
best_sequence, best_reward = mcts.search(structure, target_length=200)
```

### Running Demonstrations

```python
# Run educational demonstration
python examples/demo_mcts_simulations.py

# Run comprehensive analysis
python examples/detailed_simulations.py
```

## 🧪 Simulation Examples

### Demo Script (`examples/demo_mcts_simulations.py`)
- **Purpose**: Clear, educational demonstration
- **Features**: 
  - Node structure explanation
  - Step-by-step simulation process
  - Protein length comparison (50, 200, 500)
  - Reward calculation explanation

### Detailed Simulations (`examples/detailed_simulations.py`)
- **Purpose**: Comprehensive analysis for research
- **Features**:
  - Convergence tracking
  - Diversity analysis
  - Performance metrics
  - Visualization plots
  - JSON result export

### Key Simulation Parameters

| Length | Simulations | Max Depth | Exploration | Focus |
|--------|-------------|-----------|-------------|-------|
| 50     | 30-100      | 4         | 1.8         | Local optimization |
| 200    | 50-150      | 6         | 1.414       | Balanced approach |
| 500    | 80-200      | 8         | 1.0         | Global structure |

## 📊 Key Insights

### Performance Characteristics
- **Small proteins**: Fast convergence (30-100 simulations), local optimization
- **Medium proteins**: Balanced performance (50-150 simulations), good exploration
- **Large proteins**: Thorough search (80-200 simulations), global structure focus

### MCTS Advantages
- **Adaptive exploration**: UCB1 balances exploration vs exploitation
- **Length-aware optimization**: Reward function adapts to protein size
- **Interpretable process**: Clear node structure and selection criteria
- **Scalable**: Can handle proteins from 50 to 500+ residues

## 🔧 Technical Implementation

### Core Classes
- `SequenceLevelMCTS`: Main MCTS algorithm
- `MCTSNode`: Tree node representation
- `LengthAwareRewardComputation`: Sophisticated reward system

### Key Methods
- `search()`: Main MCTS search loop
- `_select()`: UCB1-based node selection
- `_expand()`: Sequence variation generation
- `_simulate()`: Reward computation
- `_backpropagate()`: Statistics update

## 🔬 Future Enhancements

- [ ] Integration with real DPLM-2 model
- [ ] Proper plDDT computation from structure
- [ ] TM-score and GDT-TS integration
- [ ] Model finetuning with imitation learning
- [ ] Support for real protein structures (PDB/mmCIF)
- [ ] Parallel MCTS implementation
- [ ] Advanced reward functions with learned components

## 📚 References

- arXiv:2506.00925 - MCTS-guided protein design
- arXiv:2406.07025 - Entropy-guided drug design
- DPLM-2: Diffusion-based protein language model

## 🤝 Contributing

This is a research prototype. For questions or contributions, please refer to the examples and test scripts for usage patterns. 