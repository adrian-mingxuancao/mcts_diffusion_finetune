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
# Example download location
mkdir -p /path/to/af3_params
cd /path/to/af3_params
# Follow official download instructions from GitHub
```

### Step 4: Test ABCFold with Podman

```bash
# Install ABCFold
cd /home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination/ABCFold
pip install -e .

# Test with example
abcfold examples/protein_example.json output_test -a --mmseqs2 \
  --model_params /path/to/af3_params
```

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

### Issue: AF3 container download fails

**Solution:** The AF3 container is pulled automatically. If it fails:
```bash
# Manually pull the container
podman pull ghcr.io/google-deepmind/alphafold3:latest
```

### Issue: Out of memory

**Solution:** AF3 requires significant GPU memory. Use smaller sequences or reduce batch size.

## Quick Reference

### File Locations
- **ABCFold**: `/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination/ABCFold`
- **Model params**: Download from https://github.com/google-deepmind/alphafold3
- **ABCFold README**: `ABCFold/README.md`

### Key Commands
```bash
# Install ABCFold
cd ABCFold && pip install -e .

# Run with MMseqs2 (recommended)
abcfold input.json output -a --mmseqs2 --model_params /path/to/params

# Run with Singularity
abcfold input.json output -a --sif_path /path/to/af3.sif --model_params /path/to/params
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
