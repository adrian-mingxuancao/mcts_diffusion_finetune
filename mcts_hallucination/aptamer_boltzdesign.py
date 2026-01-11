#!/usr/bin/env python
"""
DNA Aptamer Design using BoltzDesign/Boltz Structure Backend.

This script runs an iterative design loop to generate de novo DNA aptamer sequences
(40-80 nt) that bind to a target protein, producing all-atom docked complex structures.

How to run:
-----------
# Basic usage with EpCAM target
python aptamer_boltzdesign.py \
    --target_pdb path/to/4MZV.pdb \
    --target_id 4MZV \
    --outdir results \
    --length 60 \
    --batch_size 64 \
    --iters 10

# CD137 with BCY ligand removed (default)
python aptamer_boltzdesign.py \
    --target_pdb path/to/6Y8K.pdb \
    --target_id 6Y8K \
    --remove_ligands

# CD137 keeping BCY ligand
python aptamer_boltzdesign.py \
    --target_pdb path/to/6Y8K.pdb \
    --target_id 6Y8K \
    --keep_bcy

Supported targets:
- Glypican-1: PDB 4AD7 (glycosylation sites unresolved - warning logged)
- EpCAM: PDB 4MZV
- CD137: PDB 6Y8K (--keep_bcy or --remove_bcy modes)
"""

import os
import sys
import json
import random
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Add project paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "mcts_diffusion_finetune"))

# DNA alphabet
DNA_BASES = "ACGT"

# Boltz checkpoint paths
BOLTZ_CKPT = os.environ.get("BOLTZ_CKPT", "/net/scratch/caom/.boltz/boltz1_conf.ckpt")
BOLTZ_CCD = os.environ.get("BOLTZ_CCD", "/net/scratch/caom/.boltz/ccd.pkl")


@dataclass
class CandidateResult:
    """Result for a single aptamer candidate."""
    sequence: str
    conf_score: float
    iface_score: float
    total_score: float
    mean_plddt: float
    contact_count: int
    min_distance: float
    structure_path: Optional[str]
    failed: bool
    error_msg: str = ""


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DNA Aptamer Design using Boltz Structure Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--target_pdb", type=str, required=True,
                        help="Path to target protein PDB file")
    parser.add_argument("--target_id", type=str, required=True,
                        help="Target identifier (e.g., 4MZV)")
    parser.add_argument("--outdir", type=str, default="results",
                        help="Output directory (default: results)")
    
    # Sequence parameters
    parser.add_argument("--length", type=int, default=60,
                        help="DNA aptamer length (default: 60)")
    parser.add_argument("--min_len", type=int, default=40,
                        help="Minimum aptamer length (default: 40)")
    parser.add_argument("--max_len", type=int, default=80,
                        help="Maximum aptamer length (default: 80)")
    parser.add_argument("--fixed_length", action="store_true",
                        help="Keep length fixed (no indels)")
    
    # Design parameters
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of candidates per iteration (default: 64)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Number of iterations (default: 10)")
    parser.add_argument("--elite", type=int, default=8,
                        help="Number of elite candidates to keep (default: 8)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stop patience (default: 3)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--seed_sequences", type=str, default=None,
                        help="Path to file with seed sequences (one per line)")
    
    # Mutation parameters
    parser.add_argument("--p_mut", type=float, default=0.05,
                        help="Point mutation probability per position (default: 0.05)")
    
    # Scoring weights
    parser.add_argument("--w_conf", type=float, default=1.0,
                        help="Weight for confidence score (default: 1.0)")
    parser.add_argument("--w_iface", type=float, default=0.5,
                        help="Weight for interface score (default: 0.5)")
    parser.add_argument("--w_invalid", type=float, default=100.0,
                        help="Penalty for invalid structures (default: 100.0)")
    
    # Ligand handling (for CD137/6Y8K)
    parser.add_argument("--remove_ligands", action="store_true", default=True,
                        help="Remove non-protein hetero atoms/ligands (default: True)")
    parser.add_argument("--keep_bcy", action="store_true",
                        help="Keep BCY ligand for CD137 (mutually exclusive with --remove_ligands)")
    
    # Output options
    parser.add_argument("--save_all_structures", action="store_true",
                        help="Save all candidate structures (default: only top-K)")
    
    # Boltz options
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock mode for testing (no real Boltz calls)")
    parser.add_argument("--recycling_steps", type=int, default=3,
                        help="Boltz recycling steps (default: 3)")
    
    args = parser.parse_args()
    
    # Handle mutually exclusive options
    if args.keep_bcy:
        args.remove_ligands = False
    
    # Validate length constraints
    if args.length < args.min_len or args.length > args.max_len:
        parser.error(f"--length {args.length} must be between --min_len {args.min_len} and --max_len {args.max_len}")
    
    return args


def get_git_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=SCRIPT_DIR
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


def generate_random_dna(length: int) -> str:
    """Generate a random DNA sequence."""
    return ''.join(random.choice(DNA_BASES) for _ in range(length))


def mutate_sequence(
    sequence: str,
    p_mut: float,
    min_len: int,
    max_len: int,
    fixed_length: bool = True
) -> str:
    """
    Mutate a DNA sequence with point mutations.
    
    Args:
        sequence: Input DNA sequence
        p_mut: Probability of mutation per position
        min_len: Minimum allowed length
        max_len: Maximum allowed length
        fixed_length: If True, no indels allowed
    
    Returns:
        Mutated sequence
    """
    seq_list = list(sequence)
    
    # Point mutations
    for i in range(len(seq_list)):
        if random.random() < p_mut:
            # Mutate to a different base
            current = seq_list[i]
            other_bases = [b for b in DNA_BASES if b != current]
            seq_list[i] = random.choice(other_bases)
    
    result = ''.join(seq_list)
    
    # Ensure length constraints
    if len(result) < min_len:
        result += generate_random_dna(min_len - len(result))
    elif len(result) > max_len:
        result = result[:max_len]
    
    return result


def strip_ligands(pdb_path: str, output_path: str, keep_bcy: bool = False) -> str:
    """
    Strip non-protein hetero atoms/ligands from PDB file.
    
    Args:
        pdb_path: Input PDB path
        output_path: Output PDB path
        keep_bcy: If True, keep BCY ligand
    
    Returns:
        Path to cleaned PDB file
    """
    kept_lines = []
    removed_hetero = []
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("HETATM"):
                res_name = line[17:20].strip()
                # Keep water (HOH) and optionally BCY
                if res_name == "HOH":
                    kept_lines.append(line)
                elif keep_bcy and res_name == "BCY":
                    kept_lines.append(line)
                else:
                    removed_hetero.append(res_name)
            elif line.startswith("ATOM") or line.startswith("TER") or line.startswith("END"):
                kept_lines.append(line)
            elif line.startswith("HEADER") or line.startswith("TITLE") or line.startswith("CRYST"):
                kept_lines.append(line)
    
    with open(output_path, 'w') as f:
        f.writelines(kept_lines)
    
    if removed_hetero:
        unique_removed = list(set(removed_hetero))
        print(f"   Removed hetero atoms: {', '.join(unique_removed)}")
    
    return output_path


def check_glycosylation_warning(pdb_path: str, target_id: str) -> None:
    """Check for glycosylation sites and log warning if needed."""
    if target_id.upper() == "4AD7":
        print("   WARNING: Glypican-1 (4AD7) may have unresolved glycosylation sites.")
        print("            Glycan modeling is not attempted - results may be affected.")


def parse_pdb_atoms(pdb_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Parse PDB file to extract atom coordinates.
    
    Returns:
        coords: (N, 3) array of coordinates
        chain_ids: List of chain IDs for each atom
        atom_types: List of atom types ('protein' or 'dna')
    """
    coords = []
    chain_ids = []
    atom_types = []
    
    # DNA residue names
    dna_residues = {'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C'}
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                chain_id = line[21]
                res_name = line[17:20].strip()
                
                coords.append([x, y, z])
                chain_ids.append(chain_id)
                
                if res_name in dna_residues:
                    atom_types.append('dna')
                else:
                    atom_types.append('protein')
    
    return np.array(coords), chain_ids, atom_types


def compute_interface_score(
    structure_path: str,
    contact_cutoff: float = 4.5
) -> Tuple[int, float]:
    """
    Compute interface score based on DNA-protein contacts.
    
    Args:
        structure_path: Path to complex structure (PDB/CIF)
        contact_cutoff: Distance cutoff for contacts (Angstroms)
    
    Returns:
        contact_count: Number of DNA-protein contacts
        min_distance: Minimum distance between DNA and protein atoms
    """
    try:
        coords, chain_ids, atom_types = parse_pdb_atoms(structure_path)
        
        # Separate DNA and protein atoms
        dna_mask = np.array([t == 'dna' for t in atom_types])
        protein_mask = np.array([t == 'protein' for t in atom_types])
        
        if not np.any(dna_mask) or not np.any(protein_mask):
            return 0, float('inf')
        
        dna_coords = coords[dna_mask]
        protein_coords = coords[protein_mask]
        
        # Compute pairwise distances (vectorized for efficiency)
        # For large structures, use chunked computation
        contact_count = 0
        min_dist = float('inf')
        
        chunk_size = 1000
        for i in range(0, len(dna_coords), chunk_size):
            dna_chunk = dna_coords[i:i+chunk_size]
            # Compute distances: (N_dna, N_protein)
            diff = dna_chunk[:, np.newaxis, :] - protein_coords[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            
            contact_count += np.sum(distances < contact_cutoff)
            chunk_min = np.min(distances)
            if chunk_min < min_dist:
                min_dist = chunk_min
        
        return int(contact_count), float(min_dist)
        
    except Exception as e:
        print(f"   Warning: Interface scoring failed: {e}")
        return 0, float('inf')


def predict_complex_boltz(
    protein_pdb: str,
    dna_sequence: str,
    output_dir: Path,
    candidate_id: str,
    recycling_steps: int = 3,
    use_mock: bool = False,
) -> Dict:
    """
    Predict protein-DNA complex structure using Boltz.
    
    Args:
        protein_pdb: Path to protein PDB file
        dna_sequence: DNA aptamer sequence
        output_dir: Directory to save outputs
        candidate_id: Unique identifier for this candidate
        recycling_steps: Number of Boltz recycling steps
        use_mock: Use mock mode for testing
    
    Returns:
        Dict with prediction results
    """
    if use_mock:
        # Mock prediction for testing
        return _mock_predict_complex(dna_sequence, output_dir, candidate_id)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create multi-chain FASTA for Boltz
        # Format: >CHAIN_ID|ENTITY_TYPE
        fasta_path = tmpdir / "input.fasta"
        
        # Read protein sequence from PDB
        protein_seq = extract_protein_sequence(protein_pdb)
        
        with open(fasta_path, 'w') as f:
            # Protein chain
            f.write(f">A|protein\n{protein_seq}\n")
            # DNA chain
            f.write(f">B|dna\n{dna_sequence}\n")
        
        boltz_output = tmpdir / "output"
        
        # Set environment for Boltz
        env = os.environ.copy()
        env['XDG_CACHE_HOME'] = '/net/scratch/caom/.cache'
        env['HF_HOME'] = '/net/scratch/caom/.cache/huggingface'
        env['TORCH_HOME'] = '/net/scratch/caom/.cache/torch'
        
        cmd = [
            "boltz", "predict",
            str(fasta_path),
            "--out_dir", str(boltz_output),
            "--use_msa_server",
            "--override",
            "--recycling_steps", str(recycling_steps),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        
        if result.returncode != 0:
            return {
                'success': False,
                'error': result.stderr[-500:] if result.stderr else "Unknown error",
                'structure_path': None,
                'mean_plddt': 0.0,
            }
        
        # Find output structure
        cif_files = list(boltz_output.glob("**/predictions/**/*.cif"))
        pdb_files = list(boltz_output.glob("**/predictions/**/*.pdb"))
        
        structure_files = cif_files + pdb_files
        if not structure_files:
            return {
                'success': False,
                'error': "No output structure found",
                'structure_path': None,
                'mean_plddt': 0.0,
            }
        
        # Copy structure to output directory
        src_structure = structure_files[0]
        dst_structure = output_dir / f"{candidate_id}{src_structure.suffix}"
        shutil.copy(src_structure, dst_structure)
        
        # Extract confidence
        mean_plddt = extract_plddt_from_structure(src_structure)
        
        return {
            'success': True,
            'error': "",
            'structure_path': str(dst_structure),
            'mean_plddt': mean_plddt,
        }


def _mock_predict_complex(dna_sequence: str, output_dir: Path, candidate_id: str) -> Dict:
    """Mock complex prediction for testing."""
    # Generate mock PDB
    mock_pdb = output_dir / f"{candidate_id}.pdb"
    
    # Create minimal mock structure
    with open(mock_pdb, 'w') as f:
        f.write("HEADER    MOCK STRUCTURE\n")
        atom_num = 1
        # Mock protein atoms
        for i in range(50):
            f.write(f"ATOM  {atom_num:5d}  CA  ALA A{i+1:4d}    {i*3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 70.00           C\n")
            atom_num += 1
        # Mock DNA atoms
        for i, base in enumerate(dna_sequence[:20]):
            res_name = f"D{base}"
            f.write(f"ATOM  {atom_num:5d}  C1' {res_name} B{i+1:4d}    {i*3.4:8.3f}{10.0:8.3f}{0.0:8.3f}  1.00 65.00           C\n")
            atom_num += 1
        f.write("END\n")
    
    return {
        'success': True,
        'error': "",
        'structure_path': str(mock_pdb),
        'mean_plddt': 65.0 + random.random() * 20,
    }


def extract_protein_sequence(pdb_path: str) -> str:
    """Extract protein sequence from PDB file."""
    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }
    
    sequence = []
    seen_residues = set()
    
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                res_name = line[17:20].strip()
                chain_id = line[21]
                res_num = line[22:26].strip()
                
                key = (chain_id, res_num)
                if key not in seen_residues and res_name in aa_map:
                    sequence.append(aa_map[res_name])
                    seen_residues.add(key)
    
    return ''.join(sequence)


def extract_plddt_from_structure(structure_path: Path) -> float:
    """Extract mean pLDDT from structure file (B-factor column)."""
    plddts = []
    
    try:
        with open(structure_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        bfactor = float(line[60:66])
                        plddts.append(bfactor)
                    except:
                        pass
        
        if plddts:
            return np.mean(plddts)
    except:
        pass
    
    return 50.0  # Default


def evaluate_candidate(
    protein_pdb: str,
    dna_sequence: str,
    output_dir: Path,
    candidate_id: str,
    w_conf: float,
    w_iface: float,
    w_invalid: float,
    recycling_steps: int = 3,
    use_mock: bool = False,
) -> CandidateResult:
    """
    Evaluate a single aptamer candidate.
    
    Args:
        protein_pdb: Path to target protein PDB
        dna_sequence: DNA aptamer sequence
        output_dir: Directory for outputs
        candidate_id: Unique candidate identifier
        w_conf: Weight for confidence score
        w_iface: Weight for interface score
        w_invalid: Penalty for invalid structures
        recycling_steps: Boltz recycling steps
        use_mock: Use mock mode
    
    Returns:
        CandidateResult with scores and metadata
    """
    # Predict complex structure
    pred_result = predict_complex_boltz(
        protein_pdb, dna_sequence, output_dir, candidate_id,
        recycling_steps=recycling_steps, use_mock=use_mock
    )
    
    if not pred_result['success']:
        return CandidateResult(
            sequence=dna_sequence,
            conf_score=0.0,
            iface_score=0.0,
            total_score=-w_invalid,
            mean_plddt=0.0,
            contact_count=0,
            min_distance=float('inf'),
            structure_path=None,
            failed=True,
            error_msg=pred_result['error'],
        )
    
    # Compute confidence score (normalize pLDDT to 0-1)
    mean_plddt = pred_result['mean_plddt']
    conf_score = mean_plddt / 100.0
    
    # Compute interface score
    contact_count, min_distance = compute_interface_score(
        pred_result['structure_path']
    )
    
    # Normalize interface score (log scale for contact count)
    iface_score = np.log1p(contact_count) / 10.0  # Normalize roughly to 0-1
    
    # Compute total score
    total_score = w_conf * conf_score + w_iface * iface_score
    
    return CandidateResult(
        sequence=dna_sequence,
        conf_score=conf_score,
        iface_score=iface_score,
        total_score=total_score,
        mean_plddt=mean_plddt,
        contact_count=contact_count,
        min_distance=min_distance,
        structure_path=pred_result['structure_path'],
        failed=False,
    )


def score_candidate(result: CandidateResult) -> float:
    """Get sortable score from candidate result."""
    return result.total_score


def run_design_loop(args) -> None:
    """Run the main aptamer design loop."""
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory structure
    run_id = f"{args.target_id}_{args.seed}_{get_git_hash()}"
    run_dir = Path(args.outdir) / args.target_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DNA Aptamer Design - {args.target_id}")
    print(f"{'='*60}")
    print(f"Output directory: {run_dir}")
    print(f"Target PDB: {args.target_pdb}")
    print(f"Aptamer length: {args.length} (range: {args.min_len}-{args.max_len})")
    print(f"Batch size: {args.batch_size}, Iterations: {args.iters}, Elite: {args.elite}")
    print(f"Mutation rate: {args.p_mut}")
    print(f"Scoring weights: conf={args.w_conf}, iface={args.w_iface}, invalid={args.w_invalid}")
    print(f"Ligand mode: {'keep_bcy' if args.keep_bcy else 'remove_ligands'}")
    
    # Save config
    config = {
        **vars(args),
        'git_hash': get_git_hash(),
        'run_id': run_id,
    }
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare target PDB
    target_pdb = Path(args.target_pdb)
    if not target_pdb.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_pdb}")
    
    # Check for glycosylation warning
    check_glycosylation_warning(str(target_pdb), args.target_id)
    
    # Strip ligands if requested
    if args.remove_ligands:
        cleaned_pdb = run_dir / f"{args.target_id}_cleaned.pdb"
        print(f"\nStripping ligands from target PDB...")
        strip_ligands(str(target_pdb), str(cleaned_pdb), keep_bcy=args.keep_bcy)
        target_pdb = cleaned_pdb
        print(f"   Ligand mode: {'keep_bcy' if args.keep_bcy else 'remove_ligands'}")
    
    # Initialize population
    print(f"\nInitializing population of {args.batch_size} candidates...")
    population = []
    
    # Load seed sequences if provided
    if args.seed_sequences and Path(args.seed_sequences).exists():
        with open(args.seed_sequences, 'r') as f:
            for line in f:
                seq = line.strip().upper()
                if seq and all(b in DNA_BASES for b in seq):
                    if args.min_len <= len(seq) <= args.max_len:
                        population.append(seq)
        print(f"   Loaded {len(population)} seed sequences")
    
    # Fill remaining with random sequences
    while len(population) < args.batch_size:
        population.append(generate_random_dna(args.length))
    
    # Tracking
    best_score_overall = float('-inf')
    best_sequence_overall = None
    best_result_overall = None
    no_improvement_count = 0
    summary_rows = []
    
    # Main iteration loop
    for iteration in range(args.iters):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration:03d}")
        print(f"{'='*60}")
        
        # Create iteration directory
        iter_dir = run_dir / f"iter_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)
        
        if args.save_all_structures:
            all_structures_dir = iter_dir / "all_structures"
            all_structures_dir.mkdir(exist_ok=True)
        else:
            all_structures_dir = iter_dir
        
        topk_dir = iter_dir / "topk"
        topk_dir.mkdir(exist_ok=True)
        
        # Evaluate all candidates
        results = []
        for i, seq in enumerate(population):
            candidate_id = f"cand_{i:04d}"
            print(f"   Evaluating {candidate_id}: {seq[:20]}...")
            
            result = evaluate_candidate(
                str(target_pdb), seq, all_structures_dir, candidate_id,
                w_conf=args.w_conf, w_iface=args.w_iface, w_invalid=args.w_invalid,
                recycling_steps=args.recycling_steps, use_mock=args.use_mock,
            )
            results.append(result)
            
            status = "FAILED" if result.failed else f"score={result.total_score:.3f}"
            print(f"      {status} (pLDDT={result.mean_plddt:.1f}, contacts={result.contact_count})")
        
        # Sort by score
        results.sort(key=score_candidate, reverse=True)
        
        # Save candidates CSV
        candidates_data = []
        for i, r in enumerate(results):
            candidates_data.append({
                'rank': i + 1,
                'sequence': r.sequence,
                'total_score': r.total_score,
                'conf_score': r.conf_score,
                'iface_score': r.iface_score,
                'mean_plddt': r.mean_plddt,
                'contact_count': r.contact_count,
                'min_distance': r.min_distance,
                'structure_path': r.structure_path,
                'failed': r.failed,
                'error_msg': r.error_msg,
            })
        
        candidates_df = pd.DataFrame(candidates_data)
        candidates_df.to_csv(iter_dir / "candidates.csv", index=False)
        
        # Copy top-K structures to topk directory
        elites = results[:args.elite]
        for i, r in enumerate(elites):
            if r.structure_path and Path(r.structure_path).exists():
                src = Path(r.structure_path)
                dst = topk_dir / f"top{i+1}_{src.name}"
                if src != dst:
                    shutil.copy(src, dst)
        
        # Update best overall
        best_this_iter = results[0]
        if best_this_iter.total_score > best_score_overall:
            best_score_overall = best_this_iter.total_score
            best_sequence_overall = best_this_iter.sequence
            best_result_overall = best_this_iter
            no_improvement_count = 0
            print(f"\n   NEW BEST: score={best_score_overall:.3f}, seq={best_sequence_overall[:30]}...")
        else:
            no_improvement_count += 1
            print(f"\n   No improvement ({no_improvement_count}/{args.patience})")
        
        # Summary row
        summary_rows.append({
            'iteration': iteration,
            'best_score': best_this_iter.total_score,
            'best_plddt': best_this_iter.mean_plddt,
            'best_contacts': best_this_iter.contact_count,
            'mean_score': np.mean([r.total_score for r in results]),
            'num_failed': sum(1 for r in results if r.failed),
            'best_overall_score': best_score_overall,
            'best_overall_sequence': best_sequence_overall,
        })
        
        # Early stopping
        if no_improvement_count >= args.patience:
            print(f"\nEarly stopping: no improvement for {args.patience} iterations")
            break
        
        # Generate next population via mutation
        if iteration < args.iters - 1:
            print(f"\n   Generating next population via mutation...")
            new_population = []
            
            # Keep elite sequences
            for r in elites:
                new_population.append(r.sequence)
            
            # Generate mutants from elites
            while len(new_population) < args.batch_size:
                parent = random.choice(elites)
                child = mutate_sequence(
                    parent.sequence, args.p_mut,
                    args.min_len, args.max_len, args.fixed_length
                )
                new_population.append(child)
            
            population = new_population
    
    # Save final summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    
    # Final report
    print(f"\n{'='*60}")
    print(f"DESIGN COMPLETE")
    print(f"{'='*60}")
    print(f"Best score: {best_score_overall:.3f}")
    print(f"Best sequence: {best_sequence_overall}")
    print(f"Best pLDDT: {best_result_overall.mean_plddt:.1f}")
    print(f"Best contacts: {best_result_overall.contact_count}")
    print(f"Results saved to: {run_dir}")


def main():
    args = parse_args()
    run_design_loop(args)


if __name__ == "__main__":
    main()
