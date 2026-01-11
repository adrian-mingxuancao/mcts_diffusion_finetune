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

create virtual env
```bash
python3 -m venv ~/venvs/abcfold
source ~/venvs/abcfold/bin/activate
python -m pip install -U pip setuptools wheel
```

```bash
cd ABCFold
pip install -e .
```

**About AlphaFold3 Setup:**

ABCFold is a **wrapper** that calls AlphaFold3 - you don't need to download AF3 code yourself. ABCFold handles it via Docker/Podman/Singularity containers.

**What you need:**
1. **Podman** (or Docker/Singularity) - Container runtime
2. **AF3 model parameters** (~200GB) - Download from [AlphaFold3 GitHub](https://github.com/google-deepmind/alphafold3)
3. **Optional: MMseqs2** for faster MSA generation (recommended)

**How ABCFold works:**
- You provide: sequence + path to AF3 model parameters
- ABCFold: Creates AF3 input JSON, runs AF3 container, parses output
- You get: Structure coordinates + confidence scores

By default the hallucination expert now assumes real inference. Use mock mode only when explicitly testing (pass `use_mock=True` or `--use-mock`).

**Setup guides:**
- **Podman/Docker setup**: See [AF3_SETUP.md](AF3_SETUP.md) for detailed instructions
- **ABCFold usage**: See [ABCFold README](ABCFold/README.md)

### 2. ProteinMPNN

Set the `PROTEINMPNN_PATH` environment variable to the directory that contains `third_party/proteinpmnn` from your denovo-protein-server checkout. The expert will fail immediately if the real weights cannot be found.

```bash
export PROTEINMPNN_PATH=/lus/grand/projects/CompBioAffin/caom/denovo-protein-server/third_party/proteinpmnn
```

### 3. ESMFold Dependencies

Install GPU-enabled PyTorch and transformers inside `.venv` so the ESMFold backend can download the `facebook/esmfold_v1` weights:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.44.2
```

## Usage

```python
from mcts_hallucination.core.hallucination_mcts_simple import HallucinationMCTS
from mcts_hallucination.core.abcfold_integration import ABCFoldIntegration
from mcts_hallucination.core.proteinmpnn_integration import ProteinMPNNIntegration

# Initialize integrations in REAL mode
abcfold = ABCFoldIntegration(
    model_params="/path/to/af3_params",
    use_mock=False,
)
proteinmpnn = ProteinMPNNIntegration(
    use_real=True,
    device="cuda",
)

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

### Switching Structure Backends

The hallucination expert can fold structures with different engines:

```python
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert

# 1) ABCFold + Boltz (real Boltz predictions)
boltz_expert = create_hallucination_expert(
    structure_backend="abcfold",
    abcfold_engine="boltz",
)

# 2) ABCFold + Chai-1
chai_expert = create_hallucination_expert(
    structure_backend="abcfold",
    abcfold_engine="chai1",
)

# 3) Pure ESMFold (uses facebook/esmfold_v1 from HuggingFace)
esmfold_expert = create_hallucination_expert(
    structure_backend="esmfold",
    esmfold_device="cuda",  # or "cpu" if needed
)

# 4) Enable the real ProteinMPNN inverse-folding model
# (set $PROTEINMPNN_PATH to the directory containing third_party/proteinpmnn)
real_mpnn_expert = create_hallucination_expert(
    structure_backend="esmfold",
    use_real_proteinmpnn=True,
    proteinmpnn_device="cuda",
)

# Mock/testing mode (skip all real models)
mock_expert = create_hallucination_expert(
    use_mock=True,
    use_real_proteinmpnn=False,
)
```

When `abcfold_engine="af3"` (default) you must follow the official AlphaFold3 setup. For Boltz/Chai-1 you can skip AF3 weights and still use the ABCFold CLI. The ESMFold backend bypasses ABCFold entirely and relies on the HuggingFace `transformers` implementation. Real ProteinMPNN inverse folding is enabled by default (set `$PROTEINMPNN_PATH` accordingly); pass `use_real_proteinmpnn=False` or `--no-real-proteinmpnn` if you explicitly want to disable it. Mock mode is available only if you set both `use_mock=True` and `use_real_proteinmpnn=False`.

### Hallucination Tree Demo

Use `examples/hallucination_tree_branching_demo.py` to grow a small hallucination tree without running the full MCTS stack:

```bash
python examples/hallucination_tree_branching_demo.py \
  --length 50 --depth 3 --branching 2 --mask_ratio 0.5 \
  --structure-backend abcfold --abcfold-engine chai1 \
  --output-json branching_nodes.json
```

Switch `--structure-backend` to `esmfold` (and install PyTorch + transformers in `.venv`) to run against the real HuggingFace model, or change `--abcfold-engine` to `boltz`/`chai1` for lighter ABCFold predictors. Real ProteinMPNN runs by default; pass `--no-real-proteinmpnn` (and/or `--use-mock`) only if you explicitly want synthetic inverse folding. Adjust `--depth`/`--branching` to control the total number of nodes and use `--refold-designed` plus `--output-json nodes.json` to capture validation stats for each node.

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

### Option 2: Use with MCTS (Integration Complete!)

The hallucination expert is now integrated into the MCTS framework:

```python
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS

# Create hallucination expert
hallucination_expert = create_hallucination_expert(
    model_params="/path/to/af3_params",
    use_real_proteinmpnn=True,
)

# Add to MCTS as external expert
mcts = GeneralMCTS(
    dplm2_integration=dplm2,
    external_experts=[hallucination_expert],
    ablation_mode="single_expert",
    single_expert_id=3  # Use external expert
)

# Run search - hallucination expert will be called during expansion!
result = mcts.search(initial_sequence=seq, num_iterations=5)
```

**Test the integration:**
```bash
python test_integration.py
```

## Components

### Files You'll Use:
- `core/hallucination_expert.py`: **Main expert class** - AF3 + ProteinMPNN pipeline
- `core/abcfold_integration.py`: AF3 wrapper (supports mock and real modes)
- `test_integration.py`: **Start here** - Integration tests and usage examples
- `AF3_SETUP.md`: Guide for setting up AF3 with Podman/Docker

### Reference Files:
- `core/hallucination_mcts.py`: Copy of MCTS showing what was added (for reference only)
- `core/proteinmpnn_integration.py`: ProteinMPNN interface (mock version)
- `ABCFold/`: ABCFold repository for AF3 access
- `SUMMARY.txt`: Overview of the integration approach

### Important Note:
When actually using the hallucination expert, you'll import from the **original** MCTS:
```python
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS  # Original!
```

The `hallucination_mcts.py` file is kept as a reference to show what code was added.

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
