# MCTS Hallucination Pipeline

## Overview

This pipeline implements hallucination-based protein design using MCTS to guide the exploration process. Unlike the discrete diffusion approach (DPLM), this method uses:

1. **Structure Hallucination**: AlphaFold3 (via ABCFold) generates structures from masked sequences
2. **Inverse Folding**: ProteinMPNN designs sequences for hallucinated structures
3. **MCTS Guidance**: Tree search explores the hallucination space efficiently

## Pipeline Flow

```
Start with all-mask sequence
    ↓
AF3 (ABCFold) → Hallucinated Structure
    ↓
ProteinMPNN → Designed Sequence
    ↓
Create MCTS Node
    ↓
MCTS Tree Search:
  - Selection: PH-UCT with entropy/novelty
  - Expansion: Partial masking + AF3 + ProteinMPNN
  - Evaluation: Structure quality + sequence metrics
  - Backpropagation: Update tree statistics
```

## Key Differences from Discrete Diffusion MCTS

| Aspect | Discrete Diffusion | Hallucination |
|--------|-------------------|---------------|
| **Structure Generation** | DPLM structure tokens | AF3 structure prediction |
| **Sequence Generation** | DPLM sequence tokens | ProteinMPNN inverse folding |
| **Starting Point** | Baseline structure/sequence | All-mask sequence |
| **Masking Strategy** | pLDDT-based progressive | Confidence-based progressive |
| **Expert Models** | DPLM-2 (650M, 150M, 3B) | AF3 + ProteinMPNN |


## Installation

### 1. ABCFold (AF3 wrapper)

ABCFold is already cloned in this directory. To install:

```bash
cd ABCFold
pip install -e .
```

**About AlphaFold3 Setup:**

ABCFold is a **wrapper** that calls AlphaFold3 - you don't need to download AF3 code yourself. ABCFold handles it via Docker/Singularity containers.

**What you need:**
1. **Docker or Singularity** installed on your system
2. **AF3 model parameters** - Download from [AlphaFold3 GitHub](https://github.com/google-deepmind/alphafold3)
3. **Optional: MMseqs2** for faster MSA generation (recommended)

**How ABCFold works:**
- You provide: sequence + path to AF3 model parameters
- ABCFold: Creates AF3 input JSON, runs AF3 via Docker, parses output
- You get: Structure coordinates + confidence scores

**For now:** Use mock mode for testing. Switch to real mode when you have AF3 parameters.

See [ABCFold README](ABCFold/README.md) for detailed setup instructions.

### 2. ProteinMPNN

Already available in the denovo-protein-server directory. No additional installation needed.

## Usage

```python
from mcts_hallucination.core.hallucination_mcts_simple import HallucinationMCTS
from mcts_hallucination.core.abcfold_integration import ABCFoldIntegration
from mcts_hallucination.core.proteinmpnn_integration import ProteinMPNNIntegration

# Initialize integrations
abcfold = ABCFoldIntegration()
proteinmpnn = ProteinMPNNIntegration()

# Initialize MCTS
mcts = HallucinationMCTS(
    target_length=100,
    abcfold_integration=abcfold,
    proteinmpnn_integration=proteinmpnn,
    max_depth=5,
    num_iterations=10,
    use_ph_uct=True,
    initial_mask_ratio=1.0  # Start with all-mask
)

# Run hallucination-guided search
result = mcts.search()

print(f"Best sequence: {result.sequence}")
print(f"Best structure quality: {result.reward}")
```

## Testing the Pipeline

### Quick Start: Mock Mode (No AF3 Required)

Test the pipeline structure without needing AF3 installed:

```bash
python test_integration.py
```

This runs in **mock mode** and will:
1. Create a test sequence with masked positions
2. Simulate AF3 structure hallucination (random but reasonable structures)
3. Simulate ProteinMPNN sequence design
4. Show the pipeline flow and data structures

### Real Mode: With ABCFold/AF3

To use real AF3 predictions:

```python
from mcts_hallucination.core.abcfold_integration import ABCFoldIntegration

# Initialize with real AF3
abcfold = ABCFoldIntegration(
    model_params="/path/to/af3/params",
    use_mmseqs=True,  # Faster MSA
    use_mock=False    # Use real AF3
)

# Test prediction
result = abcfold.predict_structure("ACDEFGHIKLMNPQRSTVWY")
print(f"Coordinates: {result['coordinates'].shape}")
print(f"Mean pLDDT: {result['confidence'].mean():.1f}")
```

**Requirements for real mode:**
- ABCFold installed: `cd ABCFold && pip install -e .`
- AlphaFold3 Docker/Singularity container
- AF3 model parameters downloaded
- Optional: MMseqs2 for faster MSA generation

### Option 2: Test with Existing MCTS

To integrate with the existing MCTS framework:

```python
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS

# Create hallucination expert
hallucination_expert = create_hallucination_expert()

# Add to MCTS as external expert
mcts = GeneralMCTS(
    dplm2_integration=dplm2,
    external_experts=[hallucination_expert],
    ablation_mode="single_expert",
    single_expert_id=3  # Use external expert
)

# Run search
result = mcts.search(initial_sequence=seq, num_iterations=5)
```

**Note:** This requires adding the external expert handling code to `sequence_level_mcts.py` (see `test_integration.py` for the code snippet).

## Components

- `core/hallucination_mcts.py`: Copy of original MCTS (for reference)
- `core/hallucination_expert.py`: **Main expert class** - AF3 + ProteinMPNN pipeline
- `core/abcfold_integration.py`: AF3 wrapper (supports mock and real modes)
- `core/proteinmpnn_integration.py`: ProteinMPNN interface
- `test_integration.py`: **Start here** - Integration tests and usage examples
- `ABCFold/`: ABCFold repository for AF3 access
- `SUMMARY.txt`: Overview of the integration approach

## Next Steps

1. **Test in mock mode**: `python test_integration.py`
2. **Install ABCFold**: `cd ABCFold && pip install -e .`
3. **Set up AF3**: Follow [ABCFold installation guide](ABCFold/README.md)
4. **Switch to real mode**: Set `use_mock=False` in `ABCFoldIntegration`
5. **Integrate with MCTS**: Add external expert handling to `sequence_level_mcts.py`

## How It Works

The hallucination expert plugs into the existing MCTS as an external expert:

```
MCTS Iteration:
├─ Selection (UCT/PH-UCT)
├─ Expansion
│  ├─ DPLM-2 rollouts (650M, 150M, 3B)
│  ├─ ProteinMPNN rollouts
│  └─ Hallucination rollouts ← NEW!
│     └─ For each rollout:
│        1. Mask low-confidence positions
│        2. AF3 hallucinates structure
│        3. ProteinMPNN designs sequence
│        4. Return candidate
├─ Evaluation (structure quality / AAR)
└─ Backpropagation (max rule)
```

The key difference: Instead of starting from a baseline and refining it (DPLM approach), hallucination starts from masked positions and generates novel structures de novo.
