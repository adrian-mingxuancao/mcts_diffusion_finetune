#!/bin/bash
#SBATCH -J aptamer_6922
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 12:00:00

set -euo pipefail

# ============================================================================
# ABCFold DNA Aptamer Structure Prediction & Complex Docking Pipeline
# ============================================================================
# Scientific task:
#   - Aptamer 6922 (binder) and 6927 (non-binder) tertiary structure prediction
#   - Complex prediction: aptamer 6922 + TIRR (6D0L), Nudt16TI (6CO2), Nudt16 (3COU)
#   - Goal: Show 6922 binds TIRR/Nudt16TI but NOT Nudt16 (low confidence)
#
# Biological context:
#   - TIRR blocks 53BP1 recruitment to DNA damage sites
#   - Aptamer 6922 binds TIRR, potentially disrupting TIRR-53BP1 interaction
#   - Nudt16TI shares domain with TIRR; Nudt16 is parent (no binding expected)
#
# TODO (future): Design optimization of aptamer sequence for improved binding
# ============================================================================

# -------------------------
# Configuration
# -------------------------
CONDA_ENV="${CONDA_ENV:-/net/scratch/caom/hallucination_env}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(dirname $(realpath $0))}"

# Create timestamped results directory with backend name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/results/abcfold_aptamer_6922_vs_6927/${BACKEND}_${TIMESTAMP}"

# Aptamer sequences (80-nt DNA)
APTAMER_6922="TGCGTATTGACACATGCGTGCATACGGACATGATGGCGAACCCGGCTTGAGTTCTTTGGGTGAGTCCGAATTTTGGAGCG"
APTAMER_6927="TGCGTATTGACACATGCGTGTAGGGGTACCAACTGTCACTGCCAATTTTGCCACTTGCTCTGAGTCCGAATTTTGGAGCG"

# Protein targets (PDB IDs)
PROTEINS=("6D0L" "6CO2" "3COU")
PROTEIN_NAMES=("TIRR" "Nudt16TI" "Nudt16")

# Backend selection: "boltz" or "chai"
BACKEND="${BACKEND:-boltz}"

# Prediction parameters
RECYCLING_STEPS="${RECYCLING_STEPS:-3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"

# -------------------------
# Setup directories
# -------------------------
mkdir -p "${RESULTS_DIR}"/{inputs,aptamer_only/6922,aptamer_only/6927,summary,logs}
for pdb in "${PROTEINS[@]}"; do
    mkdir -p "${RESULTS_DIR}/complex/6922_${pdb}"
done

# Redirect logs to results directory
exec > >(tee -a "${RESULTS_DIR}/logs/job_${SLURM_JOB_ID:-local}.out") 2>&1

echo "============================================================================"
echo "ABCFold DNA Aptamer Pipeline"
echo "============================================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "============================================================================"

# GPU info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv || true

# -------------------------
# Environment
# -------------------------
if [[ -f "${CONDA_ENV}/bin/activate" ]]; then
    source "${CONDA_ENV}/bin/activate"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate hallucination_env 2>/dev/null || true
fi

echo "Python: $(which python)"
python -V

# Verify backend is available
if [[ "${BACKEND}" == "boltz" ]]; then
    if ! command -v boltz &>/dev/null; then
        echo "ERROR: boltz command not found. Please install boltz or activate correct environment."
        exit 1
    fi
    echo "Backend: Boltz ($(which boltz))"
elif [[ "${BACKEND}" == "chai" ]]; then
    if ! python3 -c "from chai_lab.chai1 import run_inference" 2>/dev/null; then
        echo "ERROR: chai_lab not found. Please install chai-lab or activate correct environment."
        exit 1
    fi
    echo "Backend: Chai-1 (Python API)"
else
    echo "ERROR: Unknown backend '${BACKEND}'. Use 'boltz' or 'chai'."
    exit 1
fi

# Set cache directories
export XDG_CACHE_HOME="/net/scratch/caom/.cache"
export HF_HOME="/net/scratch/caom/.cache/huggingface"
export TORCH_HOME="/net/scratch/caom/.cache/torch"

# -------------------------
# Write aptamer FASTA files
# -------------------------
echo ""
echo "Writing aptamer FASTA files..."

cat > "${RESULTS_DIR}/inputs/aptamer_6922.fasta" << EOF
>aptamer_6922|dna
${APTAMER_6922}
EOF

cat > "${RESULTS_DIR}/inputs/aptamer_6927.fasta" << EOF
>aptamer_6927|dna
${APTAMER_6927}
EOF

echo "  Created: aptamer_6922.fasta (${#APTAMER_6922} nt)"
echo "  Created: aptamer_6927.fasta (${#APTAMER_6927} nt)"

# -------------------------
# Download protein PDBs
# -------------------------
echo ""
echo "Downloading protein structures..."

for i in "${!PROTEINS[@]}"; do
    pdb="${PROTEINS[$i]}"
    name="${PROTEIN_NAMES[$i]}"
    pdb_file="${RESULTS_DIR}/inputs/${pdb}.pdb"
    
    if [[ -f "${pdb_file}" ]] && [[ $(head -c 4 "${pdb_file}") != "<!DO" ]]; then
        echo "  ${pdb} (${name}): already exists"
    else
        echo "  Downloading ${pdb} (${name})..."
        curl -sL "https://files.rcsb.org/download/${pdb}.pdb" -o "${pdb_file}"
        
        # Check if we got HTML error page
        if [[ ! -s "${pdb_file}" ]] || [[ $(head -c 4 "${pdb_file}") == "<!DO" ]]; then
            echo "    PDB format unavailable, trying CIF..."
            cif_file="${RESULTS_DIR}/inputs/${pdb}.cif"
            curl -sL "https://files.rcsb.org/download/${pdb}.cif" -o "${cif_file}"
            
            if [[ -s "${cif_file}" ]] && [[ $(head -c 5 "${cif_file}") == "data_" ]]; then
                # Convert CIF to PDB
                python3 << PYEOF
from Bio.PDB import MMCIFParser, PDBIO
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("${pdb}", "${cif_file}")
chain_map = {}
next_chain = ord('A')
for model in structure:
    for chain in model:
        old_id = chain.id
        if old_id not in chain_map:
            chain_map[old_id] = chr(next_chain)
            next_chain += 1
        chain.id = chain_map[old_id]
io = PDBIO()
io.set_structure(structure)
io.save("${pdb_file}")
print(f"    Converted CIF to PDB: {chain_map}")
PYEOF
            fi
        fi
        
        if [[ ! -s "${pdb_file}" ]] || [[ $(head -c 4 "${pdb_file}") == "<!DO" ]]; then
            echo "    ERROR: Failed to download ${pdb}"
            exit 1
        fi
        echo "    Downloaded: ${pdb_file}"
    fi
done

# -------------------------
# Initialize metrics CSV
# -------------------------
METRICS_FILE="${RESULTS_DIR}/summary/metrics.csv"
echo "task_name,target,top_model_confidence,runtime_seconds,notes" > "${METRICS_FILE}"

# -------------------------
# Helper function: Run Boltz prediction
# -------------------------
run_boltz_prediction() {
    local fasta_file="$1"
    local output_dir="$2"
    local task_name="$3"
    local target="${4:-none}"
    
    local start_time=$(date +%s)
    local notes=""
    local confidence="0.0"
    
    echo ""
    echo "--- ${task_name} (Boltz) ---"
    echo "  FASTA: ${fasta_file}"
    echo "  Output: ${output_dir}"
    
    # Run Boltz
    local boltz_tmp=$(mktemp -d)
    local boltz_out="${boltz_tmp}/output"
    
    set +e
    timeout "${TIMEOUT_SECONDS}" boltz predict \
        "${fasta_file}" \
        --out_dir "${boltz_out}" \
        --use_msa_server \
        --override \
        --recycling_steps "${RECYCLING_STEPS}" \
        2>&1 | tee "${output_dir}/boltz.log"
    local exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))
    
    if [[ ${exit_code} -ne 0 ]]; then
        notes="FAILED: exit_code=${exit_code}"
        echo "  ERROR: Boltz failed (exit ${exit_code})"
    else
        # Find and copy output structure
        local cif_file=$(find "${boltz_out}" -name "*.cif" -type f 2>/dev/null | head -1)
        local plddt_file=$(find "${boltz_out}" -name "plddt*.npz" -type f 2>/dev/null | head -1)
        
        if [[ -n "${cif_file}" ]]; then
            cp "${cif_file}" "${output_dir}/best_model.cif"
            echo "  Saved: best_model.cif"
            
            # Extract confidence (pLDDT)
            if [[ -n "${plddt_file}" ]]; then
                cp "${plddt_file}" "${output_dir}/plddt.npz"
                confidence=$(python3 -c "
import numpy as np
data = np.load('${plddt_file}')
plddt = data['plddt']
if plddt.max() <= 1.0:
    plddt = plddt * 100.0
print(f'{np.mean(plddt):.2f}')
" 2>/dev/null || echo "0.0")
            else
                # Try to extract from CIF B-factors
                confidence=$(python3 -c "
from Bio.PDB import MMCIFParser
import numpy as np
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure('s', '${cif_file}')
bfactors = [atom.bfactor for atom in structure.get_atoms()]
print(f'{np.mean(bfactors):.2f}' if bfactors else '0.0')
" 2>/dev/null || echo "0.0")
            fi
            
            notes="success"
            echo "  Mean pLDDT: ${confidence}"
        else
            notes="no_output_structure"
            echo "  WARNING: No output structure found"
        fi
    fi
    
    # Cleanup
    rm -rf "${boltz_tmp}"
    
    # Append to metrics
    echo "${task_name},${target},${confidence},${runtime},${notes}" >> "${METRICS_FILE}"
    
    echo "  Runtime: ${runtime}s"
}

# -------------------------
# Helper function: Run Chai-1 prediction
# -------------------------
run_chai_prediction() {
    local fasta_file="$1"
    local output_dir="$2"
    local task_name="$3"
    local target="${4:-none}"
    
    local start_time=$(date +%s)
    local notes=""
    local confidence="0.0"
    
    echo ""
    echo "--- ${task_name} (Chai-1) ---"
    echo "  FASTA: ${fasta_file}"
    echo "  Output: ${output_dir}"
    
    # Run Chai-1 via Python
    set +e
    python3 << PYEOF
import sys
import os
from pathlib import Path

# Set output directory
output_dir = Path("${output_dir}")
fasta_file = Path("${fasta_file}")

try:
    from chai_lab.chai1 import run_inference
    
    # Run inference
    print("  Running Chai-1 inference...")
    candidates = run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir / "chai_output",
        num_trunk_recycles=${RECYCLING_STEPS},
        num_diffn_timesteps=200,
        seed=42,
        device='cuda:0',
        use_esm_embeddings=True,
    )
    
    # Copy best model
    if candidates.cif_paths:
        import shutil
        best_cif = candidates.cif_paths[0]
        shutil.copy(best_cif, output_dir / "best_model.cif")
        print(f"  Saved: best_model.cif")
        
        # Extract pLDDT from structure
        from Bio.PDB import MMCIFParser
        import numpy as np
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('s', best_cif)
        bfactors = [atom.bfactor for atom in structure.get_atoms()]
        if bfactors:
            mean_plddt = np.mean(bfactors)
            print(f"  Mean pLDDT: {mean_plddt:.2f}")
            with open(output_dir / "plddt.txt", 'w') as f:
                f.write(f"{mean_plddt:.2f}")
        
        # Save scores if available
        if hasattr(candidates, 'scores') and candidates.scores:
            import json
            with open(output_dir / "scores.json", 'w') as f:
                json.dump({'scores': [float(s) for s in candidates.scores]}, f)
        
        sys.exit(0)
    else:
        print("  ERROR: Chai-1 produced no structures")
        sys.exit(1)
        
except Exception as e:
    print(f"  ERROR: Chai-1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
    local exit_code=$?
    set -e
    
    local end_time=$(date +%s)
    local runtime=$((end_time - start_time))
    
    if [[ ${exit_code} -ne 0 ]]; then
        notes="FAILED: exit_code=${exit_code}"
        echo "  ERROR: Chai-1 failed (exit ${exit_code})"
    else
        # Read pLDDT from file
        if [[ -f "${output_dir}/plddt.txt" ]]; then
            confidence=$(cat "${output_dir}/plddt.txt")
        fi
        notes="success"
    fi
    
    # Append to metrics
    echo "${task_name},${target},${confidence},${runtime},${notes}" >> "${METRICS_FILE}"
    
    echo "  Runtime: ${runtime}s"
}

# -------------------------
# Unified prediction function (selects backend)
# -------------------------
run_prediction() {
    if [[ "${BACKEND}" == "boltz" ]]; then
        run_boltz_prediction "$@"
    elif [[ "${BACKEND}" == "chai" ]]; then
        run_chai_prediction "$@"
    fi
}

# -------------------------
# Task 1: Aptamer-only folding
# -------------------------
echo ""
echo "============================================================================"
echo "Task 1: Aptamer-only tertiary structure prediction"
echo "============================================================================"

# Create FASTA for aptamer-only (single chain)
# Boltz format: >CHAIN_ID|entity_type
# Chai format: >entity_type|name=...
if [[ "${BACKEND}" == "boltz" ]]; then
    cat > "${RESULTS_DIR}/aptamer_only/6922/input.fasta" << EOF
>A|dna
${APTAMER_6922}
EOF
    cat > "${RESULTS_DIR}/aptamer_only/6927/input.fasta" << EOF
>A|dna
${APTAMER_6927}
EOF
elif [[ "${BACKEND}" == "chai" ]]; then
    cat > "${RESULTS_DIR}/aptamer_only/6922/input.fasta" << EOF
>dna|name=aptamer_6922
${APTAMER_6922}
EOF
    cat > "${RESULTS_DIR}/aptamer_only/6927/input.fasta" << EOF
>dna|name=aptamer_6927
${APTAMER_6927}
EOF
fi

run_prediction \
    "${RESULTS_DIR}/aptamer_only/6922/input.fasta" \
    "${RESULTS_DIR}/aptamer_only/6922" \
    "aptamer_6922_only" \
    "none"

run_prediction \
    "${RESULTS_DIR}/aptamer_only/6927/input.fasta" \
    "${RESULTS_DIR}/aptamer_only/6927" \
    "aptamer_6927_only" \
    "none"

# -------------------------
# Task 2-4: Complex prediction (aptamer 6922 + each protein)
# -------------------------
echo ""
echo "============================================================================"
echo "Task 2-4: Complex prediction (aptamer 6922 + proteins)"
echo "============================================================================"

# Helper: Extract protein sequence from PDB
extract_protein_seq() {
    local pdb_file="$1"
    python3 << PYEOF
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

parser = PDBParser(QUIET=True)
structure = parser.get_structure('s', '${pdb_file}')

# Get first chain's sequence
for model in structure:
    for chain in model:
        residues = [r for r in chain.get_residues() if r.id[0] == ' ']
        seq = ''.join([seq1(r.resname) for r in residues])
        if len(seq) > 10:  # Skip very short chains
            print(seq)
            break
    break
PYEOF
}

for i in "${!PROTEINS[@]}"; do
    pdb="${PROTEINS[$i]}"
    name="${PROTEIN_NAMES[$i]}"
    complex_dir="${RESULTS_DIR}/complex/6922_${pdb}"
    
    echo ""
    echo "--- Complex: aptamer_6922 + ${pdb} (${name}) ---"
    
    # Extract protein sequence
    protein_seq=$(extract_protein_seq "${RESULTS_DIR}/inputs/${pdb}.pdb")
    
    if [[ -z "${protein_seq}" ]]; then
        echo "  ERROR: Could not extract protein sequence from ${pdb}.pdb"
        echo "aptamer_6922_${pdb},${pdb},0.0,0,failed_seq_extraction" >> "${METRICS_FILE}"
        continue
    fi
    
    echo "  Protein sequence length: ${#protein_seq}"
    
    # Create multi-chain FASTA (protein + DNA)
    # Boltz format: >CHAIN_ID|entity_type
    # Chai format: >entity_type|name=...
    if [[ "${BACKEND}" == "boltz" ]]; then
        cat > "${complex_dir}/input.fasta" << EOF
>A|protein
${protein_seq}
>B|dna
${APTAMER_6922}
EOF
    elif [[ "${BACKEND}" == "chai" ]]; then
        cat > "${complex_dir}/input.fasta" << EOF
>protein|name=${pdb}
${protein_seq}
>dna|name=aptamer_6922
${APTAMER_6922}
EOF
    fi
    
    run_prediction \
        "${complex_dir}/input.fasta" \
        "${complex_dir}" \
        "aptamer_6922_${pdb}" \
        "${pdb}"
done

# -------------------------
# Generate README
# -------------------------
cat > "${RESULTS_DIR}/summary/README.txt" << EOF
ABCFold DNA Aptamer Structure Prediction Results
================================================

Date: $(date)
Job ID: ${SLURM_JOB_ID:-local}

Scientific Context:
- Aptamer 6922 is a known binder to TIRR protein
- Aptamer 6927 is a non-binder control
- TIRR (6D0L) blocks 53BP1 recruitment to DNA damage sites
- Nudt16TI (6CO2) shares domain with TIRR
- Nudt16 (3COU) is parent protein (no binding expected)

Tasks Completed:
1. Aptamer-only folding: 6922 and 6927 tertiary structure prediction
2. Complex prediction: aptamer 6922 + TIRR (6D0L)
3. Complex prediction: aptamer 6922 + Nudt16TI (6CO2)
4. Complex prediction: aptamer 6922 + Nudt16 (3COU)

Expected Results:
- 6922 + TIRR/Nudt16TI: High confidence (good binding)
- 6922 + Nudt16: Low confidence (no binding)

Directory Structure:
- inputs/: FASTA files and downloaded PDBs
- aptamer_only/: Single-chain aptamer structure predictions
- complex/: Protein-aptamer complex predictions
- summary/: Metrics CSV and this README
- logs/: Job output logs

Metrics File: summary/metrics.csv
- task_name: Description of the prediction task
- target: Protein PDB ID (or "none" for aptamer-only)
- top_model_confidence: Mean pLDDT score (0-100)
- runtime_seconds: Wall-clock time for prediction
- notes: Success/failure status

TODO (Future Work):
- Aptamer sequence optimization for improved binding
- Interface analysis and contact mapping
- Comparison with experimental binding data
EOF

# -------------------------
# Final summary
# -------------------------
echo ""
echo "============================================================================"
echo "Pipeline Complete"
echo "============================================================================"
echo "Results directory: ${RESULTS_DIR}"
echo ""
echo "Metrics summary:"
cat "${METRICS_FILE}"
echo ""
echo "End: $(date)"
