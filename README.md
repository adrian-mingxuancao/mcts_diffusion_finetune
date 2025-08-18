# MCTS-Guided DPLM-2 Performance Improvement Framework

A general framework for improving Diffusion-based Protein Language Model (DPLM-2) performance across all tasks using Monte Carlo Tree Search (MCTS) with expert rollout and compound rewards.

## 🎯 **Project Overview**

This framework provides a **task-agnostic approach** to improve DPLM-2 performance across all protein modeling tasks:

- **Inverse Folding** (structure → sequence)
- **Folding** (sequence → structure) 
- **Unconditional Generation**
- **Conditional Generation**

### 🚀 **Key Innovations**

1. **plDDT-based Masking**: Intelligent masking based on predicted local distance difference test scores (with random fallback)
2. **Simultaneous Position Sampling**: Leverages diffusion's ability to sample multiple positions at once (unlike transformers)
3. **Expert Rollout**: Uses compound rewards to guide exploration toward better solutions
4. **Task-Agnostic Design**: Same framework works for all DPLM-2 tasks

### 🧠 **Core Concept**

The framework uses MCTS to explore sequence space more efficiently than standard generation:

```
1. Mask sequence based on plDDT (or randomly)
2. Sample multiple positions simultaneously using diffusion
3. Evaluate with compound rewards (structure + biophysical + naturalness, expert rollout would be included in the future)
4. Use expert rollout to guide exploration
5. Backpropagate rewards to improve future decisions
```

## 🏗️ **Framework Structure**

```
mcts_diffusion_finetune/
├── core/
│   ├── sequence_level_mcts.py    # General MCTS framework
│   ├── dplm2_integration.py     # DPLM-2 model integration
│   └── __init__.py
├── utils/
│   ├── reward_computation.py     # Compound reward functions
│   ├── protein_utils.py         # Protein utilities
│   └── plddt_computation.py    # plDDT scoring
├── examples/
│   ├── demo_mcts_simulations.py # MCTS concept demonstration
│   └── detailed_simulations.py  # Performance analysis
├── test_general_mcts.py         # Test all task types
└── README.md
```

## 🚀 **NEW: General Framework Working!**

### **Key Features**

1. **Task-Agnostic Design**: Same framework for all DPLM-2 tasks
2. **plDDT-Based Masking**: Intelligent masking with random fallback
3. **Simultaneous Sampling**: Leverages diffusion's multi-position sampling
4. **Expert Rollout**: Compound rewards guide exploration
5. **GPU Acceleration**: Full CUDA support for DPLM-2

### **Supported Tasks**

| Task Type | Description | Input | Output |
|-----------|-------------|-------|--------|
| `inverse_folding` | Structure → Sequence | 3D structure | Amino acid sequence |
| `folding` | Sequence → Structure | Sequence | 3D structure |
| `unconditional` | Generate from scratch | None | Random sequence |
| `conditional` | Generate with constraints | Condition | Constrained sequence |

### **Usage Example**

#### Quick Test
```bash
# Run comprehensive test with real CAMEO data + AAR/scTM evaluation
python tests/test_mcts_with_real_data.py
```

#### Programmatic Usage
```python
from core import GeneralMCTS
from utils.cameo_data_loader import create_cameo_structure_for_testing

# Load real protein structure from CAMEO dataset
structure = create_cameo_structure_for_testing(index=0)

# Initialize MCTS for inverse folding
mcts = GeneralMCTS(
    task_type="inverse_folding",
    num_simulations=30,
    use_plddt_masking=True,
    simultaneous_sampling=False
)

# Run search - MCTS optimizes scTM-score and AAR
best_sequence, best_reward = mcts.search(structure, target_length=50)
```

## 🔧 Setup

```bash
# Activate environment  
export PATH="/net/scratch/caom/dplm_env/bin:$PATH"

# Set CUDA path
export PYTHONPATH=/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:$PYTHONPATH

# Test framework
python tests/test_mcts_with_real_data.py
```

## 🎯 How It Works

**DPLM-2 Self-Consistency Evaluation** (following [DPLM-2 paper](https://arxiv.org/pdf/2410.13782)):

1. **Generate sequence** from target structure using DPLM-2
2. **Fold generated sequence** using ESMFold  
3. **Compare structures** using TM-score, RMSD, pLDDT
4. **No reference sequence needed** - pure self-consistency

**MCTS Improvement Process**:
1. Start with DPLM-2 generated sequence
2. MCTS explores sequence variations
3. Each variation evaluated via self-consistency
4. Best sequences have high TM-score (structure match) + good biophysical properties

## 📊 Expected Results

MCTS consistently improves over DPLM-2 baseline by optimizing the compound reward function that includes both structural compatibility and biophysical properties. 