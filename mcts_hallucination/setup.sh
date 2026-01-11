#!/bin/bash
# MCTS Hallucination Pipeline Setup Script
# This script sets up the environment for the mcts_hallucination module

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="${SCRATCH_DIR:-/net/scratch/$USER}"

echo "=============================================="
echo "MCTS Hallucination Pipeline Setup"
echo "=============================================="

# Parse arguments
INSTALL_OPTIONAL=false
SKIP_DSSP=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --full) INSTALL_OPTIONAL=true; shift ;;
        --skip-dssp) SKIP_DSSP=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 1. Create virtual environment if it doesn't exist
VENV_DIR="${SCRATCH_DIR}/hallucination_env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Activated virtual environment: $VENV_DIR"

# 2. Install core Python dependencies
echo ""
echo "Installing core Python dependencies..."
pip install --upgrade pip
pip install numpy torch transformers biopython matplotlib pandas scipy pytest

# 3. Install structure prediction backends
echo ""
echo "Installing structure prediction backends..."

# Boltz (diffusion-based structure prediction)
echo "  Installing Boltz..."
pip install boltz || echo "  Warning: Boltz installation failed (optional)"

# Chai-1 (structure prediction with ligand support)
echo "  Installing Chai-1..."
pip install chai-lab || echo "  Warning: Chai-1 installation failed (optional)"

# 4. Install ABCFold (unified AF3/Boltz/Chai interface)
if [ -d "$SCRIPT_DIR/ABCFold" ]; then
    echo ""
    echo "Installing ABCFold..."
    pip install -e "$SCRIPT_DIR/ABCFold"
fi

# 5. Install DSSP via micromamba
if [ "$SKIP_DSSP" = false ]; then
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
        "$DSSP_BINARY" --version 2>/dev/null || true
    else
        echo "WARNING: DSSP installation failed. SS guidance will not work."
    fi
fi

# 6. Install optional tools (if --full flag)
if [ "$INSTALL_OPTIONAL" = true ]; then
    echo ""
    echo "Installing optional tools..."
    
    # BoltzDesign1
    if [ -d "$SCRIPT_DIR/extra/BoltzDesign1" ]; then
        echo "  Installing BoltzDesign1..."
        pip install -e "$SCRIPT_DIR/extra/BoltzDesign1" || echo "  Warning: BoltzDesign1 failed"
    fi
fi

# 7. Set environment variables
echo ""
echo "Setting environment variables..."

# ProteinMPNN path (adjust as needed)
export PROTEINMPNN_PATH="${PROTEINMPNN_PATH:-/home/$USER/AID3/dplm/denovo-protein-server/third_party/proteinmpnn}"

# Add to PATH
export PATH="${SCRATCH_DIR}/bin:$PATH"
export PATH="${SCRATCH_DIR}/dssp_env/bin:$PATH"

# 8. Verify installation
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
try:
    from core.esmfold_integration import ESMFoldIntegration
    print('  ✓ ESMFoldIntegration')
except Exception as e:
    print(f'  ✗ ESMFoldIntegration: {e}')

try:
    from core.hallucination_expert import create_hallucination_expert
    print('  ✓ HallucinationExpert')
except Exception as e:
    print(f'  ✗ HallucinationExpert: {e}')

try:
    from core.abcfold_integration import ABCFoldIntegration
    print('  ✓ ABCFoldIntegration')
except Exception as e:
    print(f'  ✗ ABCFoldIntegration: {e}')

try:
    from core.hallucination_mcts import HallucinationMCTS, HallucinationNode
    print('  ✓ HallucinationMCTS')
except Exception as e:
    print(f'  ✗ HallucinationMCTS: {e}')

try:
    from core.ss_guidance import SSGuidance, SSGuidanceConfig, DSSP_BINARY_PATH
    print(f'  ✓ SSGuidance (DSSP: {DSSP_BINARY_PATH})')
except Exception as e:
    print(f'  ✗ SSGuidance: {e}')

# Optional backends
try:
    import boltz
    print('  ✓ Boltz')
except:
    print('  - Boltz (not installed)')

try:
    from chai_lab.chai1 import run_inference
    print('  ✓ Chai-1')
except:
    print('  - Chai-1 (not installed)')

print('')
print('Setup verification complete!')
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
echo ""
echo "For full installation with optional tools:"
echo "  ./setup.sh --full"
