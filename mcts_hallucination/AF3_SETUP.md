# AlphaFold3 Setup Guide for ABCFold

This guide explains how to set up AlphaFold3 with ABCFold, specifically for Podman (Docker alternative).

## Overview

ABCFold is a **wrapper** that calls AlphaFold3. You don't download AF3 code directly - instead, you:
1. Install Docker/Podman (container runtime)
2. Download AF3 model parameters (~200GB)
3. ABCFold automatically pulls and runs the AF3 container

## Option 1: Using Podman (Recommended for HPC)

Podman is a Docker alternative that doesn't require root privileges.

### Step 1: Check if Podman is Available

```bash
podman --version
```

If not installed, contact your system administrator or install via:
```bash
# On RHEL/CentOS/Fedora
sudo yum install podman

# On Ubuntu/Debian
sudo apt-get install podman
```

### Step 2: Configure ABCFold to Use Podman

ABCFold uses Docker by default, but you can make it use Podman by creating a Docker alias:

```bash
# Add to your ~/.bashrc or ~/.bash_profile
alias docker=podman

# Or set up podman-docker package
sudo yum install podman-docker  # RHEL/CentOS
```

### Step 3: Download AlphaFold3 Model Parameters

The AF3 model parameters are available from Google DeepMind:

**Official source:** https://github.com/google-deepmind/alphafold3

1. Go to the AlphaFold3 GitHub repository
2. Follow their instructions to download model parameters
3. You'll need to agree to their terms of use
4. Download size: ~200GB

```bash
# Work from the hallucination repo root
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination
source .venv/bin/activate
cd ABCFold

# Decompress the AF3 weights that were downloaded as af3.bin.zst
mkdir -p af3_params
zstd -d af3.bin.zst -o af3_params/af3.bin

# Record or print the absolute path for later use
readlink -f af3_params
```

### Step 4: Clone the Official AlphaFold3 Repo and Build the Container

Google DeepMind’s instructions (see `docs/installation.md` in the official repo) have you build the container locally:

```bash
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination
git clone https://github.com/google-deepmind/alphafold3.git alphafold3_official
cd alphafold3_official
podman build -t alphafold3 -f docker/Dockerfile .
```

Replace `podman` with `docker` if that is your runtime. Once this finishes you will have a local image/tag named `alphafold3` that ABCFold can call.

### Step 5: Download the AlphaFold3 Genetic Databases

AlphaFold3 expects the full set of genetic databases described in the paper. The official repo provides `fetch_databases.sh` to download and unpack them (~630 GB uncompressed):

```bash
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/alphafold3_official
./fetch_databases.sh /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/alphafold3_databases
```

Keep the database directory outside the git checkout if you prefer a different location; just remember the absolute path so you can mount it when running AlphaFold3.

### Step 6: (Optional) Prepare MMseqs2 Databases for ABCFold

If you want to run MMseqs2 locally for faster MSAs inside ABCFold (instead of letting AF3 perform JACKHMMER searches), set up the databases using the helper script shipped with ABCFold:

```bash
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/ABCFold
MMSEQS_NO_INDEX=1 ./setup_mmseqs_databases.sh /path/to/mmseqs_db
```

Point `--mmseqs_database /path/to/mmseqs_db` when running `abcfold` if you do not rely on the remote ColabFold server.

### Step 7: Install and Test ABCFold with the Real Weights

```bash
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination
source .venv/bin/activate

cd ABCFold
pip install -e .

# Smoke test (run from inside ABCFold so relative paths resolve)
abcfold examples/protein_example.json output_test -a --mmseqs2 \
  --model_params af3_params
```

If the command completes without errors and produces `output_test/model_0.cif`, the container runtime, AF3 weights, and MMseqs2 path are configured correctly.

### Step 8: Using Boltz or Chai-1 Without AF3

If AlphaFold3 is not available, you can still leverage the lighter models included with ABCFold:

```bash
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination
source .venv/bin/activate
cd ABCFold

# Boltz run
abcfold examples/protein_example.json boltz_out -b --mmseqs2

# Chai-1 run
abcfold examples/protein_example.json chai_out -c --mmseqs2
```

These commands install Boltz/Chai-1 automatically—no AF3 weights or container builds required. In the hallucination expert, set `structure_backend="abcfold"` and `abcfold_engine="boltz"` (or `"chai1"`) with `use_mock=False` to read those predictions.

## Option 2: Using Singularity (Alternative for HPC)

If your HPC system uses Singularity instead of Docker/Podman:

```bash
# ABCFold supports Singularity via --sif_path flag
abcfold input.json output -a --mmseqs2 \
  --model_params /path/to/af3_params \
  --sif_path /path/to/alphafold3.sif
```

## Integration with Hallucination Expert

Once AF3 is set up, update the hallucination expert:

```python
from mcts_hallucination.core.abcfold_integration import ABCFoldIntegration

# Switch from mock to real mode
abcfold = ABCFoldIntegration(
    model_params="/path/to/af3_params",
    use_mmseqs=True,  # Faster MSA generation
    use_mock=False    # Use real AF3!
)

# Test it
result = abcfold.predict_structure("ACDEFGHIKLMNPQRSTVWY")
print(f"Mean pLDDT: {result['confidence'].mean():.1f}")
```

To bypass ABCFold entirely, instantiate the expert with `structure_backend="esmfold"` (optionally `esmfold_device="cuda"`) and ensure PyTorch plus `transformers` are installed so the HuggingFace `facebook/esmfold_v1` weights can be downloaded.

## Troubleshooting

### Issue: "Cannot connect to Docker daemon"

**Solution:** Make sure Podman is running and Docker alias is set:
```bash
alias docker=podman
# Or
sudo systemctl start podman
```

### Issue: "Permission denied" errors

**Solution:** Podman runs rootless by default, but you may need to configure:
```bash
podman system migrate
```

### Issue: AF3 container build fails

**Solution:** Inspect the build logs from `podman build -t alphafold3 -f docker/Dockerfile .` inside the official repo. Common causes are insufficient disk space, missing CUDA 12.6 on the host, or network hiccups while installing dependencies. Rerun the build after fixing the underlying issue.

### Issue: Out of memory

**Solution:** AF3 requires significant GPU memory. Use smaller sequences or reduce batch size.

## Quick Reference

### File Locations
- **Project root**: `/lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination`
- **ABCFold**: `/lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/ABCFold`
- **Model params**: `/lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/ABCFold/af3_params`
- **ABCFold README**: `ABCFold/README.md`

### Key Commands
```bash
# Activate env and install ABCFold
cd /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination
source .venv/bin/activate
cd ABCFold && pip install -e .

# Run with MMseqs2 (recommended)
abcfold input.json output -a --mmseqs2 --model_params /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/ABCFold/af3_params

# Run with Singularity (after building alphafold3.sif from the official repo)
abcfold input.json output -a --sif_path /path/to/alphafold3.sif \
  --model_params /lus/grand/projects/CompBioAffin/caom/mcts_diffusion_finetune/mcts_hallucination/ABCFold/af3_params
```

### For Mock Mode (Testing)
If you don't have AF3 set up yet, keep using mock mode:
```python
abcfold = ABCFoldIntegration(use_mock=True)  # Default
```

## Additional Resources

- **AlphaFold3 GitHub**: https://github.com/google-deepmind/alphafold3
- **ABCFold GitHub**: https://github.com/rigdenlab/ABCFold
- **Podman Documentation**: https://podman.io/
- **AlphaFold3 Installation Guide**: https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md
