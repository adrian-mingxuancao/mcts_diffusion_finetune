#!/bin/bash
#SBATCH --job-name=boltzdesign_8eo4_dna
#SBATCH --output=/net/scratch/caom/logs/boltzdesign_8eo4_%j.out
#SBATCH --error=/net/scratch/caom/logs/boltzdesign_8eo4_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# BoltzDesign 8EO4 DNA-Binding Test
# This script downloads PDB 8EO4, detects DNA chains, and runs BoltzDesign
# to design a DNA binder protein.

set -e

# Configuration
PDB_ID="8EO4"
WORK_DIR="/net/scratch/caom/boltzdesign_8eo4"
BOLTZDESIGN_DIR="/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination/extra/BoltzDesign1"
BOLTZ_CKPT="${BOLTZ_CKPT:-/net/scratch/caom/.boltz/boltz1_conf.ckpt}"
BOLTZ_CCD="${BOLTZ_CCD:-/net/scratch/caom/.boltz/ccd.pkl}"

# Create directories
mkdir -p "${WORK_DIR}"
mkdir -p "/net/scratch/caom/logs"
cd "${WORK_DIR}"

echo "=============================================="
echo "BoltzDesign 8EO4 DNA-Binding Test"
echo "=============================================="
echo "PDB ID: ${PDB_ID}"
echo "Work directory: ${WORK_DIR}"
echo "BoltzDesign directory: ${BOLTZDESIGN_DIR}"
echo "Start time: $(date)"
echo "=============================================="

# Step 1: Download PDB if not exists
PDB_FILE="${WORK_DIR}/${PDB_ID}.pdb"
if [ ! -f "${PDB_FILE}" ]; then
    echo "Downloading PDB ${PDB_ID}..."
    curl -sL "https://files.rcsb.org/download/${PDB_ID}.pdb" -o "${PDB_FILE}"
    if [ ! -s "${PDB_FILE}" ]; then
        echo "Failed to download PDB, trying CIF format..."
        curl -sL "https://files.rcsb.org/download/${PDB_ID}.cif" -o "${WORK_DIR}/${PDB_ID}.cif"
        PDB_FILE="${WORK_DIR}/${PDB_ID}.cif"
    fi
    echo "Downloaded: ${PDB_FILE}"
else
    echo "PDB file already exists: ${PDB_FILE}"
fi

# Step 2: Analyze PDB to detect DNA and protein chains
echo ""
echo "Analyzing PDB structure..."
python3 << 'PYTHON_SCRIPT'
import os
import sys

pdb_file = os.environ.get('PDB_FILE', '8EO4.pdb')
work_dir = os.environ.get('WORK_DIR', '.')

# Parse PDB to find chain types
dna_chains = []
protein_chains = []
chain_lengths = {}

# DNA residue names
dna_residues = {'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C'}
# Standard amino acids
aa_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
               'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
               'TYR', 'VAL'}

chain_residue_types = {}

with open(pdb_file, 'r') as f:
    for line in f:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            chain_id = line[21]
            res_name = line[17:20].strip()
            
            if chain_id not in chain_residue_types:
                chain_residue_types[chain_id] = {'dna': 0, 'protein': 0, 'other': 0}
            
            if res_name in dna_residues:
                chain_residue_types[chain_id]['dna'] += 1
            elif res_name in aa_residues:
                chain_residue_types[chain_id]['protein'] += 1
            else:
                chain_residue_types[chain_id]['other'] += 1

# Classify chains
for chain_id, counts in chain_residue_types.items():
    total = counts['dna'] + counts['protein']
    if total == 0:
        continue
    if counts['dna'] > counts['protein']:
        dna_chains.append(chain_id)
    else:
        protein_chains.append(chain_id)

print(f"DNA chains: {','.join(sorted(dna_chains))}")
print(f"Protein chains: {','.join(sorted(protein_chains))}")

# Select target (DNA) and estimate binder length
# For DNA binding, we typically want a protein of 80-150 residues
if dna_chains:
    target_chains = ','.join(sorted(dna_chains))
else:
    print("ERROR: No DNA chains found in PDB!")
    sys.exit(1)

# Estimate binder length based on existing protein chains
if protein_chains:
    # Count residues in first protein chain as reference
    ref_chain = sorted(protein_chains)[0]
    ref_length = chain_residue_types[ref_chain]['protein'] // 10  # Rough estimate
    binder_min = max(50, ref_length - 20)
    binder_max = min(200, ref_length + 50)
else:
    binder_min = 80
    binder_max = 150

# Write config for BoltzDesign
config = {
    'pdb_id': os.environ.get('PDB_ID', '8EO4'),
    'dna_chains': target_chains,
    'protein_chains': ','.join(sorted(protein_chains)) if protein_chains else '',
    'binder_min': binder_min,
    'binder_max': binder_max,
}

config_file = os.path.join(work_dir, 'chain_config.txt')
with open(config_file, 'w') as f:
    for k, v in config.items():
        f.write(f"{k}={v}\n")

print(f"Binder length range: {binder_min}-{binder_max}")
print(f"Config saved to: {config_file}")
PYTHON_SCRIPT

# Read the config
source "${WORK_DIR}/chain_config.txt"
echo ""
echo "Configuration:"
echo "  DNA chains (target): ${dna_chains}"
echo "  Protein chains: ${protein_chains}"
echo "  Binder length: ${binder_min}-${binder_max}"

# Step 3: Run BoltzDesign
echo ""
echo "=============================================="
echo "Running BoltzDesign..."
echo "=============================================="

cd "${BOLTZDESIGN_DIR}"

# Activate environment if needed
if [ -d "/net/scratch/caom/hallucination_env" ]; then
    source /net/scratch/caom/hallucination_env/bin/activate
fi

# Run BoltzDesign for DNA binder design
python boltzdesign.py \
    --target_name "${PDB_ID}" \
    --target_type dna \
    --input_type pdb \
    --pdb_path "${PDB_FILE}" \
    --pdb_target_ids "${dna_chains}" \
    --binder_id A \
    --length_min "${binder_min}" \
    --length_max "${binder_max}" \
    --design_samples 3 \
    --num_designs 2 \
    --gpu_id 0 \
    --work_dir "${WORK_DIR}" \
    --boltz_checkpoint "${BOLTZ_CKPT}" \
    --ccd_path "${BOLTZ_CCD}" \
    --run_boltz_design True \
    --run_ligandmpnn True \
    --run_alphafold False \
    --run_rosetta False \
    --suffix "dna_binder" \
    --use_msa False \
    --pre_iteration 30 \
    --soft_iteration 75 \
    --temp_iteration 50 \
    --hard_iteration 5

echo ""
echo "=============================================="
echo "BoltzDesign completed!"
echo "=============================================="
echo "Results saved to: ${WORK_DIR}"
echo "End time: $(date)"

# List output files
echo ""
echo "Output files:"
find "${WORK_DIR}" -name "*.pdb" -o -name "*.csv" -o -name "*.json" 2>/dev/null | head -20
