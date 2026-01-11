#!/bin/bash
#SBATCH -J aptamer_boltz
#SBATCH -p general
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH --output=/net/scratch/%u/logs/%x_%A_%a.out
#SBATCH --error=/net/scratch/%u/logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

# -------------------------
# User-configurable knobs
# -------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
TARGET_DIR="${PROJECT_ROOT}/data/targets"
OUT_ROOT="${PROJECT_ROOT}/results/aptamer"

# Aptamer search parameters (tune as needed)
LENGTH="${LENGTH:-60}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ITERS="${ITERS:-5}"
ELITE="${ELITE:-8}"
SEED="${SEED:-0}"
BOLTZ_TIMEOUT="${BOLTZ_TIMEOUT:-1800}"

# Scoring weights
W_CONF="${W_CONF:-1.0}"
W_IFACE="${W_IFACE:-1.0}"
W_INVALID="${W_INVALID:-10.0}"

# -------------------------
# Targets via job array
# -------------------------
TARGETS=( "4AD7" "4MZV" "6Y8K" )
TARGET_ID="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
TARGET_PDB="${TARGET_DIR}/${TARGET_ID}.pdb"

# CD137 special handling: remove ligands by default (binding independent of BCY10916)
EXTRA_ARGS=""
if [[ "${TARGET_ID}" == "6Y8K" ]]; then
  if [[ "${KEEP_BCY_6Y8K:-false}" == "true" ]]; then
    EXTRA_ARGS="--keep_bcy"
  else
    EXTRA_ARGS="--remove_ligands"
  fi
else
  EXTRA_ARGS="--remove_ligands"
fi

# -------------------------
# Logging / diagnostics
# -------------------------
LOG_DIR="/net/scratch/${USER}/logs"
mkdir -p "${LOG_DIR}"

echo "===== SLURM JOB INFO ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Target: ${TARGET_ID}"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"
echo "Start: $(date)"
echo "=========================="

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

# Git metadata (best-effort)
if command -v git >/dev/null 2>&1 && [[ -d "${PROJECT_ROOT}/.git" ]]; then
  echo "Git commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo 'unknown')"
fi

# -------------------------
# Environment activation
# -------------------------
# Try conda first
if [[ -n "${CONDA_ENV_NAME:-}" ]] && [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
elif [[ -f "/net/scratch/caom/hallucination_env/bin/activate" ]]; then
  source /net/scratch/caom/hallucination_env/bin/activate
fi

echo "Python: $(which python)"
python -V

# -------------------------
# Download PDB if missing
# -------------------------
mkdir -p "${TARGET_DIR}"
if [[ ! -f "${TARGET_PDB}" ]] || [[ $(head -1 "${TARGET_PDB}" 2>/dev/null) == "<!doctype"* ]]; then
  echo "Downloading PDB ${TARGET_ID} from RCSB..."
  curl -sL "https://files.rcsb.org/download/${TARGET_ID}.pdb" -o "${TARGET_PDB}"
  
  # Check if we got HTML instead of PDB (some structures only have CIF)
  if [[ ! -s "${TARGET_PDB}" ]] || [[ $(head -1 "${TARGET_PDB}") == "<!doctype"* ]]; then
    echo "PDB format not available, trying CIF and converting..."
    CIF_FILE="${TARGET_DIR}/${TARGET_ID}.cif"
    curl -sL "https://files.rcsb.org/download/${TARGET_ID}.cif" -o "${CIF_FILE}"
    
    if [[ -s "${CIF_FILE}" ]] && [[ $(head -1 "${CIF_FILE}") == "data_"* ]]; then
      # Convert CIF to PDB using BioPython
      python3 << PYEOF
from Bio.PDB import MMCIFParser, PDBIO
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("${TARGET_ID}", "${CIF_FILE}")
# Rename long chain IDs to single letters for PDB format
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
io.save("${TARGET_PDB}")
print(f"Converted CIF to PDB. Chain mapping: {chain_map}")
PYEOF
    fi
  fi
  
  if [[ ! -s "${TARGET_PDB}" ]] || [[ $(head -1 "${TARGET_PDB}") == "<!doctype"* ]]; then
    echo "ERROR: Failed to download/convert ${TARGET_ID}"
    exit 1
  fi
  echo "Downloaded: ${TARGET_PDB}"
else
  echo "PDB already exists: ${TARGET_PDB}"
fi

# Log target-specific notes
if [[ "${TARGET_ID}" == "4AD7" ]]; then
  echo "NOTE: Glypican-1 (4AD7) has glycosylation sites noted but not fully resolved in crystal structure"
elif [[ "${TARGET_ID}" == "6Y8K" ]]; then
  echo "NOTE: CD137 (6Y8K) crystal is complexed with agonist BCY10916. Designing binding independent of BCY."
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUT_ROOT}/${TARGET_ID}"
mkdir -p "${OUTDIR}"

echo "Output dir: ${OUTDIR}/${RUN_ID}"

# -------------------------
# Run
# -------------------------
set -x
python -u "${PROJECT_ROOT}/aptamer_boltzdesign.py" \
  --target_pdb "${TARGET_PDB}" \
  --target_id "${TARGET_ID}" \
  --outdir "${OUT_ROOT}" \
  --length "${LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --iters "${ITERS}" \
  --elite "${ELITE}" \
  --seed "${SEED}" \
  --w_conf "${W_CONF}" \
  --w_iface "${W_IFACE}" \
  --w_invalid "${W_INVALID}" \
  --boltz_timeout "${BOLTZ_TIMEOUT}" \
  ${EXTRA_ARGS}
set +x

echo "Done: $(date)"
