#!/bin/bash
# MCTS Hallucination Pipeline Setup Script
# This script sets up the environment for the mcts_hallucination module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/net/scratch/$USER}"

echo "=============================================="
echo "MCTS Hallucination Pipeline Setup"
echo "=============================================="

# 1. Create virtual environment if it doesn't exist
VENV_DIR="${SCRATCH_DIR}/hallucination_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Activated virtual environment: $VENV_DIR"

# 2. Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# 3. Install ABCFold (optional)
if [ -d "$SCRIPT_DIR/ABCFold" ]; then
    echo ""
    echo "Installing ABCFold..."
    pip install -e "$SCRIPT_DIR/ABCFold"
fi

# 4. Install DSSP via micromamba
DSSP_ENV="${SCRATCH_DIR}/dssp_env"
DSSP_BINARY="${DSSP_ENV}/bin/mkdssp"

if [ ! -f "$DSSP_BINARY" ]; then
    echo ""
    echo "Installing DSSP..."
    
    # Check if micromamba is available
    MICROMAMBA="${SCRATCH_DIR}/bin/micromamba"
    if [ ! -f "$MICROMAMBA" ]; then
        echo "Downloading micromamba..."
        mkdir -p "${SCRATCH_DIR}/bin"
        curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
            tar -xvj -C "${SCRATCH_DIR}/bin" --strip-components=1 bin/micromamba
    fi
    
    echo "Creating DSSP environment..."
    "$MICROMAMBA" create -y -p "$DSSP_ENV" -c conda-forge dssp
fi

if [ -f "$DSSP_BINARY" ]; then
    echo "DSSP installed at: $DSSP_BINARY"
    "$DSSP_BINARY" --version
else
    echo "WARNING: DSSP installation failed. SS guidance will not work."
fi

# 5. Set environment variables
echo ""
echo "Setting environment variables..."

# ProteinMPNN path (adjust as needed)
export PROTEINMPNN_PATH="${PROTEINMPNN_PATH:-/home/$USER/AID3/dplm/denovo-protein-server/third_party/proteinmpnn}"

# Add to PATH
export PATH="${SCRATCH_DIR}/bin:$PATH"
export PATH="${DSSP_ENV}/bin:$PATH"

# 6. Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

# Test Python imports
python -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR')

print('Testing imports...')

# Core modules
from core.esmfold_integration import ESMFoldIntegration
print('  ✓ ESMFoldIntegration')

from core.hallucination_expert import create_hallucination_expert
print('  ✓ HallucinationExpert')

from core.abcfold_integration import ABCFoldIntegration
print('  ✓ ABCFoldIntegration')

from core.proteinmpnn_integration import ProteinMPNNIntegration
print('  ✓ ProteinMPNNIntegration')

from core.ss_guidance import SSGuidance, SSGuidanceConfig, DSSP_BINARY_PATH
print(f'  ✓ SSGuidance (DSSP: {DSSP_BINARY_PATH})')

print('')
print('All imports successful!')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run tests:"
echo "  python test_backends.py --mock"
echo "  python test_integration.py --use-mock"
echo ""
echo "Environment variables to set:"
echo "  export PROTEINMPNN_PATH=$PROTEINMPNN_PATH"
echo "  export PATH=${SCRATCH_DIR}/bin:\$PATH"
