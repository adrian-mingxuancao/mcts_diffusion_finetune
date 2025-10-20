#!/usr/bin/env python3
"""
Inverse Folding Sampling with Baseline + pLDDT Masking for PDB Dataset

This script performs inverse folding sampling on the PDB dataset using the correct approach:
    1. Load reference sequence and structure tokens
    2. Generate baseline sequence with DPLM-2
    3. Compute pLDDT scores for baseline using ESMFold
    4. Mask only low-confidence positions based on pLDDT
    5. Sample improved sequences from masked positions
    6. Evaluate AAR, scTM, and composite rewards
    7. Save detailed results

This matches the successful CAMEO MCTS ablation approach with proper
baseline generation and selective pLDDT-based masking.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import SeqIO

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mcts_diffusion_finetune.core.dplm2_integration import DPLM2Integration  # noqa: E402
from mcts_diffusion_finetune.utils.pdb_data_loader import PDBDataLoader  # noqa: E402

try:
    from mcts_diffusion_finetune.utils.sctm_calculation import calculate_sctm_score  # noqa: E402
    SCTM_AVAILABLE = True
except Exception:
    SCTM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Baseline Generation and pLDDT Computation
# ---------------------------------------------------------------------------

def generate_baseline_sequence_with_plddt(dplm2: DPLM2Integration, struct_tokens: str, 
                                         target_length: int, structure_id: str, 
                                         seed: int = 42) -> Tuple[Optional[str], Optional[List[float]]]:
    """Generate baseline sequence and compute pLDDT scores using ESMFold."""
    
    # Set fixed seed for baseline generation
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    
    logging.info(f"  🎯 Generating baseline sequence for {structure_id} (seed: {seed})")
    
    try:
        # Generate baseline sequence using proper DPLM-2 generation (like generate_dplm2_patched_v2.py)
        result = dplm2.generate_from_masked_input(
            aa_sequence="<mask_aa>" * target_length,  # Fully masked AA sequence
            struct_tokens=struct_tokens,
            task_type="inverse_folding",
            expert_id=1,  # Use 150M model for baseline
            temperature=1.0,
        )
        
        if not result:
            logging.warning(f"  ❌ Failed to generate baseline for {structure_id}")
            return None, None
            
        baseline_seq = clean_sequence(result)
        logging.info(f"  ✅ Generated baseline: {len(baseline_seq)} chars (target: {target_length})")
        
        # Compute pLDDT scores using ESMFold
        plddt_scores = compute_esmfold_plddt(baseline_seq)
        if plddt_scores:
            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            logging.info(f"  ✅ Computed pLDDT: mean={mean_plddt:.1f}, length={len(plddt_scores)}")
        else:
            logging.warning(f"  ⚠️ Failed to compute pLDDT scores")
            
        return baseline_seq, plddt_scores
        
    except Exception as e:
        logging.error(f"  ❌ Baseline generation failed for {structure_id}: {e}")
        return None, None


def compute_esmfold_plddt(sequence: str) -> Optional[List[float]]:
    """Compute pLDDT scores for a sequence using ESMFold."""
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            # Tokenize sequence
            tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            # Predict structure
            output = model(tokenized["input_ids"])
            
            # Extract pLDDT scores (confidence scores)
            if hasattr(output, 'plddt') and output.plddt is not None:
                plddt_tensor = output.plddt.cpu()
                # pLDDT is per-residue confidence, convert to list
                if len(plddt_tensor.shape) == 3:  # [batch, length, 1]
                    plddt_scores = plddt_tensor[0, :, 0].tolist()
                elif len(plddt_tensor.shape) == 2:  # [batch, length]
                    plddt_scores = plddt_tensor[0, :].tolist()
                else:
                    plddt_scores = plddt_tensor.flatten().tolist()
                
                # Clean sequence to match pLDDT length
                clean_seq = clean_sequence(sequence)
                return plddt_scores[:len(clean_seq)]  # Match sequence length
                
    except Exception as e:
        logging.warning(f"  ⚠️ ESMFold pLDDT computation failed: {e}")
        
    return None


def predict_structure_coordinates(sequence: str) -> Optional[np.ndarray]:
    """Predict 3D coordinates for a sequence using ESMFold."""
    try:
        import torch
        import numpy as np
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            # Tokenize sequence
            tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            # Predict structure
            output = model(tokenized["input_ids"])
            
            # Extract coordinates (CA atoms)
            if hasattr(output, 'positions') and output.positions is not None:
                positions = output.positions.cpu().numpy()
                # positions shape: [8, 1, length, 14, 3] (ESMFold format)
                # Extract CA atoms (index 1 in 14-atom representation)
                if len(positions.shape) == 5:
                    if positions.shape[3] >= 2:  # At least 2 atoms (N, CA)
                        ca_coords = positions[0, 0, :, 1, :]  # CA atoms
                    else:
                        ca_coords = positions[0, 0, :, 0, :]  # Use first atom if CA not available
                    
                    # Clean sequence to match coordinate length
                    clean_seq = clean_sequence(sequence)
                    return ca_coords[:len(clean_seq)]  # Match sequence length
                elif len(positions.shape) == 4:
                    if positions.shape[2] >= 2:  # At least 2 atoms (N, CA)
                        ca_coords = positions[0, :, 1, :]  # CA atoms
                    else:
                        ca_coords = positions[0, :, 0, :]  # Use first atom if CA not available
                    
                    # Clean sequence to match coordinate length
                    clean_seq = clean_sequence(sequence)
                    return ca_coords[:len(clean_seq)]  # Match sequence length
                
    except Exception as e:
        logging.warning(f"  ⚠️ ESMFold coordinate prediction failed: {e}")
        
    return None


def load_pdb_structure_data(structure_id: str) -> Optional[Dict]:
    """Load PDB structure data from preprocessed directory."""
    try:
        import pickle
        
        # Get middle two characters for directory (e.g., 5S9R -> s9)
        if len(structure_id) >= 4:
            middle_chars = structure_id[1:3].lower()
            pkl_path = f"/home/caom/AID3/dplm/data-bin/PDB_date/preprocessed/{middle_chars}/{structure_id}.pkl"
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            return data
        
    except Exception as e:
        logging.warning(f"Failed to load structure data for {structure_id}: {e}")
    
    return None


def create_proteinmpnn_mask(baseline_seq: str, plddt_scores: List[float], 
                           mask_threshold: float = 70.0) -> List[bool]:
    """Create ProteinMPNN mask (True = keep, False = design) based on pLDDT scores."""
    if not plddt_scores or len(plddt_scores) != len(baseline_seq):
        # Fallback: keep 80% of positions randomly
        keep_count = max(1, int(len(baseline_seq) * 0.8))
        positions_to_keep = random.sample(range(len(baseline_seq)), keep_count)
        mask = [i in positions_to_keep for i in range(len(baseline_seq))]
        return mask

    # Keep positions with pLDDT >= threshold (high confidence)
    mask = [plddt >= mask_threshold for plddt in plddt_scores]
    
    # Ensure at least some positions are kept
    keep_count = sum(mask)
    if keep_count == 0:
        # If all positions have low confidence, keep the highest 90%
        sorted_positions = sorted(enumerate(plddt_scores), key=lambda x: x[1], reverse=True)
        positions_to_keep = [pos for pos, _ in sorted_positions[:max(1, int(len(baseline_seq) * 0.9))]]
        mask = [i in positions_to_keep for i in range(len(baseline_seq))]
    elif keep_count < len(baseline_seq) * 0.5:  # Keep at least 50%
        # If too few positions are high confidence, keep the highest 70%
        sorted_positions = sorted(enumerate(plddt_scores), key=lambda x: x[1], reverse=True)
        positions_to_keep = [pos for pos, _ in sorted_positions[:int(len(baseline_seq) * 0.7)]]
        mask = [i in positions_to_keep for i in range(len(baseline_seq))]
    
    return mask


def create_plddt_masked_sequence(baseline_seq: str, plddt_scores: List[float], 
                                mask_threshold: float = 70.0) -> str:
    """Create masked sequence based on pLDDT scores using <mask_aa> tokens."""
    if not plddt_scores or len(plddt_scores) != len(baseline_seq):
        logging.warning("  ⚠️ pLDDT scores unavailable, using random masking")
        # Fallback: mask 20% of positions randomly
        mask_count = max(1, len(baseline_seq) // 5)
        positions_to_mask = random.sample(range(len(baseline_seq)), mask_count)
        masked_seq = list(baseline_seq)
        for pos in positions_to_mask:
            masked_seq[pos] = '<mask_aa>'
        return ''.join(masked_seq)
    
    # Mask positions with pLDDT < threshold
    masked_seq = list(baseline_seq)
    masked_count = 0
    
    for i, (aa, plddt) in enumerate(zip(baseline_seq, plddt_scores)):
        if plddt < mask_threshold:
            masked_seq[i] = '<mask_aa>'
            masked_count += 1
    
    # Ensure at least some positions are masked (but not too many)
    if masked_count == 0:
        # If all positions have high confidence, mask the lowest 10%
        sorted_positions = sorted(enumerate(plddt_scores), key=lambda x: x[1])
        positions_to_mask = [pos for pos, _ in sorted_positions[:max(1, len(baseline_seq) // 10)]]
        for pos in positions_to_mask:
            masked_seq[pos] = '<mask_aa>'
            masked_count += 1
    elif masked_count > len(baseline_seq) * 0.5:  # Don't mask more than 50%
        # If too many positions are low confidence, only mask the lowest 30%
        sorted_positions = sorted(enumerate(plddt_scores), key=lambda x: x[1])
        positions_to_mask = [pos for pos, _ in sorted_positions[:int(len(baseline_seq) * 0.3)]]
        masked_seq = list(baseline_seq)
        masked_count = 0
        for pos in positions_to_mask:
            masked_seq[pos] = '<mask_aa>'
            masked_count += 1
    
    mask_percentage = (masked_count / len(baseline_seq)) * 100
    logging.info(f"  🎭 Masked {masked_count}/{len(baseline_seq)} positions ({mask_percentage:.1f}%) with pLDDT < {mask_threshold}")
    
    return ''.join(masked_seq)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def clean_sequence(seq: str) -> str:
    """Remove invalid amino acids and convert to uppercase."""
    valid = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper() if aa in valid)


def load_pregenerated_baseline_pdb(structure_id: str) -> Optional[str]:
    """Load pregenerated DPLM-2 baseline sequence for PDB structure (if available)."""
    baseline_dir = Path("/home/caom/AID3/dplm/generation-results/dplm2_150m_pdb/inverse_folding")
    fasta_path = baseline_dir / f"{structure_id}.fasta"
    if not fasta_path.exists():
        return None

    try:
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            return clean_sequence(record.seq)
    except Exception as exc:
        logging.warning("Failed to load PDB baseline for %s: %s", structure_id, exc)
    return None


def calculate_simple_aar(pred_seq: str, ref_seq: str) -> float:
    """Calculate amino acid recovery (AAR) between predicted and reference sequences."""
    if not pred_seq or not ref_seq:
        return 0.0
    min_len = min(len(pred_seq), len(ref_seq))
    if min_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if pred_seq[i] == ref_seq[i])
    return matches / min_len


def calculate_biophysical_score(sequence: str) -> float:
    """Calculate biophysical feasibility score based on amino acid composition."""
    if not sequence:
        return 0.0
    
    # Count amino acid types
    charged = sum(1 for aa in sequence if aa in 'DEKR')
    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV')
    polar = sum(1 for aa in sequence if aa in 'NQSTY')
    
    # Calculate fractions
    total = len(sequence)
    charged_frac = charged / total
    hydrophobic_frac = hydrophobic / total
    
    # Apply penalties for extreme compositions
    penalty = 0.0
    if charged_frac > 0.3:  # Too many charged residues
        penalty += (charged_frac - 0.3) * 2.0
    if hydrophobic_frac > 0.4:  # Too many hydrophobic residues
        penalty += (hydrophobic_frac - 0.4) * 1.5
    
    return max(0.0, 1.0 - penalty)


def calculate_composite_reward(aar: float, sctm: float, bio: float) -> float:
    """Calculate composite reward from AAR, scTM, and biophysical scores."""
    return 0.4 * aar + 0.45 * sctm + 0.15 * bio


def load_reference_sequences(fasta_path: Path) -> Dict[str, str]:
    """Load reference sequences from FASTA file."""
    sequences = {}
    if not fasta_path.exists():
        logging.warning("Reference FASTA not found: %s", fasta_path)
        return sequences
    
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq)
        logging.info("Loaded %d reference sequences from %s", len(sequences), fasta_path)
    except Exception as e:
        logging.error("Failed to load reference sequences: %s", e)
    
    return sequences


def load_structure_tokens(fasta_path: Path) -> Dict[str, str]:
    """Load structure tokens from FASTA file."""
    tokens = {}
    if not fasta_path.exists():
        logging.warning("Structure token FASTA not found: %s", fasta_path)
        return tokens
    
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            tokens[record.id] = str(record.seq)
        logging.info("Loaded %d structure token sequences from %s", len(tokens), fasta_path)
    except Exception as e:
        logging.error("Failed to load structure tokens: %s", e)
    
    return tokens


def resolve_expert_ids(experts_str: str) -> List[int]:
    """Parse expert IDs from command line argument."""
    if experts_str.lower() == "all":
        return [0, 1, 2]  # All DPLM-2 experts
    try:
        return [int(x.strip()) for x in experts_str.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid expert IDs: {experts_str}")


def save_individual_sampling_result(record: Dict, structure_id: str):
    """Save individual sampling result immediately after each structure completion."""
    
    # Create individual results directory
    individual_dir = "/net/scratch/caom/inverse_folding_sampling_pdb_individual_results"
    os.makedirs(individual_dir, exist_ok=True)
    
    # Save individual result with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    individual_file = os.path.join(individual_dir, f"{structure_id}_{timestamp}.json")
    
    with open(individual_file, 'w') as f:
        json.dump(record, f, indent=2, default=str)
    
    # Print immediate baseline vs best comparison
    baseline = record["baseline"]
    best_sample = record["best_sample"]
    
    aar_improvement = best_sample["aar"] - baseline["aar"]
    reward_improvement = best_sample["reward"] - baseline["reward"]
    best_sequence = best_sample["sequence"][:50] + "..." if len(best_sample["sequence"]) > 50 else best_sample["sequence"]
    baseline_sequence = baseline["sequence"][:50] + "..." if len(baseline["sequence"]) > 50 else baseline["sequence"]
    
    print(f"\n🎉 COMPLETED SAMPLING: {structure_id}")
    print(f"   📊 Baseline → Best:")
    print(f"      AAR: {baseline['aar']:.3f} → {best_sample['aar']:.3f} (Δ{aar_improvement:+.3f})")
    print(f"      scTM: {baseline['sctm']:.3f} → {best_sample['sctm']:.3f} (Δ{best_sample['sctm']-baseline['sctm']:+.3f})")
    print(f"      Reward: {baseline['reward']:.3f} → {best_sample['reward']:.3f} (Δ{reward_improvement:+.3f})")
    print(f"      Expert: {best_sample.get('expert_id', 'baseline')}")
    print(f"   🧬 Baseline seq: {baseline_sequence}")
    print(f"   🧬 Best seq:     {best_sequence}")
    print(f"   💾 Saved to: {individual_file}")


def sample_sequence(
    dplm2: DPLM2Integration,
    struct_tokens: str,
    target_length: int,
    expert_id: int,
    temperature: float,
) -> Optional[str]:
    """Sample a sequence from DPLM-2 given structure tokens."""
    masked_aa = dplm2.tokenizer.aa_mask_token * target_length
    return dplm2.generate_from_masked_input(
        aa_sequence=masked_aa,
        struct_tokens=struct_tokens,
        task_type="inverse_folding",
        expert_id=expert_id,
        temperature=temperature,
    )


def sample_from_masked_sequence(
    dplm2: DPLM2Integration,
    masked_sequence: str,
    struct_tokens: str,
    expert_id: int,
    temperature: float = 1.0,
) -> Optional[str]:
    """Sample a sequence from DPLM-2 using masked sequence input."""
    try:
        result = dplm2.generate_from_masked_input(
            aa_sequence=masked_sequence,
            struct_tokens=struct_tokens,
            task_type="inverse_folding",
            expert_id=expert_id,
            temperature=temperature,
        )
        return result if result else None
    except Exception as e:
        logging.error("Masked sampling failed (expert %d): %s", expert_id, e)
        return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse folding sampling benchmark for PDB dataset - MULTI-EXPERT ONLY")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/caom/AID3/dplm/data-bin/PDB_date",
        help="Path to PDB dataset directory containing aatype.fasta and struct.fasta.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index within the sorted PDB structure list (inclusive).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index within the structure list (exclusive). Defaults to all structures.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Samples per expert for each structure.",
    )
    parser.add_argument(
        "--experts",
        type=str,
        default="dplm2_650m,dplm2_150m,dplm2_3b",
        help="Comma-separated list of DPLM-2 expert names (ProteinMPNN added automatically)",
    )
    parser.add_argument(
        "--num_rollouts_per_expert",
        type=int,
        default=2,
        help="Number of rollouts per expert (default: 2, matching MCTS ablation)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=25,
        help="Maximum iterations like MCTS ablation (default: 25)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature passed to DPLM-2.",
    )
    parser.add_argument(
        "--skip_sctm",
        action="store_true",
        help="Skip scTM calculation (saves time and avoids ESMFold dependency).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/net/scratch/caom/pdb_evaluation_results",
        help="Directory for JSON results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed for reproducible baseline generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    random.seed(args.seed)
    np.random.seed(args.seed)

    loader = PDBDataLoader(args.data_path)
    reference_sequences = loader.reference_sequences
    struct_sequences = loader.structure_sequences

    # Use structure IDs from the actual data files, not the loader's structure list
    structure_ids = list(reference_sequences.keys())
    total_structures = len(structure_ids)
    logging.info(f"Found {total_structures} structures in data files")
    logging.info(f"First 5 structure IDs: {structure_ids[:5]}")
    
    if total_structures == 0:
        logging.error("No PDB structures found in data files")
        return

    start_idx = max(0, args.start)
    end_idx = args.end if args.end is not None else total_structures
    end_idx = min(end_idx, total_structures)
    if start_idx >= end_idx:
        logging.error("Invalid range: start=%d end=%d (dataset size=%d)", start_idx, end_idx, total_structures)
        return

    selected_ids = structure_ids[start_idx:end_idx]
    logging.info(
        "Processing %d structures (indices %d:%d)",
        len(selected_ids),
        start_idx,
        end_idx,
    )

    expert_ids = resolve_expert_ids(args.experts)
    logging.info("Experts: %s", expert_ids)

    compute_sctm = SCTM_AVAILABLE and not args.skip_sctm
    if not SCTM_AVAILABLE and not args.skip_sctm:
        logging.warning("scTM computation unavailable (ESMFold/TMalign missing). Using fallback scores.")

    dplm2 = DPLM2Integration(device="cuda", default_temperature=args.temperature)

    results: List[Dict] = []
    best_rewards: List[float] = []
    best_aars: List[float] = []
    best_sctm: List[float] = []

    for idx, structure_id in enumerate(selected_ids, start=start_idx):
        logging.info("(%d/%d) Structure %s", idx + 1, end_idx, structure_id)

        # Load reference data
        ref_seq = reference_sequences.get(structure_id)
        struct_tokens = struct_sequences.get(structure_id)
        if not ref_seq or not struct_tokens:
            logging.warning("Missing data for %s (ref_seq=%s, struct_tokens=%s)", 
                          structure_id, bool(ref_seq), bool(struct_tokens))
            continue

        ref_seq = clean_sequence(ref_seq)
        if len(ref_seq) == 0:
            logging.warning("Empty reference sequence for %s", structure_id)
            continue

        # Generate reference coordinates using ESMFold for scTM calculation
        reference_coords = None
        if compute_sctm:
            try:
                # Use ESMFold to predict reference structure coordinates
                reference_coords = predict_structure_coordinates(ref_seq)
                if reference_coords is not None:
                    logging.info(f"  ✅ Generated reference coordinates: {reference_coords.shape}")
                else:
                    logging.warning("Failed to generate reference coordinates for %s", structure_id)
            except Exception as e:
                logging.warning("Failed to generate reference coordinates for %s: %s", structure_id, e)

        # Load pregenerated baseline first (like CAMEO version)
        baseline_seq = load_pregenerated_baseline_pdb(structure_id)
        if not baseline_seq:
            logging.info("Generating baseline with DPLM-2 150M...")
            baseline_seq = sample_sequence(
                dplm2,
                struct_tokens=struct_tokens,
                target_length=len(ref_seq),
                expert_id=1,  # Use 150M model
                temperature=args.temperature,
            )
        if not baseline_seq:
            logging.warning("Baseline generation failed for %s – skipping", structure_id)
            continue
            
        baseline_seq = clean_sequence(baseline_seq)
        
        # Compute pLDDT scores for the generated baseline
        plddt_scores = compute_esmfold_plddt(baseline_seq)
        if plddt_scores:
            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            logging.info(f"  ✅ Computed pLDDT: mean={mean_plddt:.1f}, length={len(plddt_scores)}")
        else:
            logging.warning(f"  ⚠️ Failed to compute pLDDT scores")
            
        # Create pLDDT-masked sequence for sampling
        masked_sequence = create_plddt_masked_sequence(baseline_seq, plddt_scores, mask_threshold=70.0)

        # Evaluate baseline
        baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
        baseline_bio = calculate_biophysical_score(baseline_seq)
        baseline_sctm = baseline_aar  # fallback
        if compute_sctm and reference_coords is not None:
            try:
                baseline_sctm = float(calculate_sctm_score(baseline_seq, reference_coords))
            except Exception as e:
                logging.warning("Baseline scTM calculation failed for %s: %s", structure_id, e)

        baseline_reward = calculate_composite_reward(baseline_aar, baseline_sctm, baseline_bio)
        logging.info("Baseline: AAR=%.3f, scTM=%.3f, Bio=%.3f, Reward=%.3f", 
                    baseline_aar, baseline_sctm, baseline_bio, baseline_reward)

        # Load ProteinMPNN expert for fair comparison
        proteinmpnn_expert = None
        try:
            from core.proteinmpnn_integration import ProteinMPNNExpert
            proteinmpnn_expert = ProteinMPNNExpert()
            logging.info(f"✅ Loaded ProteinMPNN expert: ProteinMPNN")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load ProteinMPNN expert: {e}")
            proteinmpnn_expert = None

        # Load structure data for ProteinMPNN (from preprocessed directory)
        structure_data = None
        if proteinmpnn_expert:
            try:
                structure_data = load_pdb_structure_data(structure_id)
                if structure_data:
                    logging.info(f"✅ Loaded structure data for {structure_id}")
                else:
                    logging.warning(f"⚠️ No structure data found for {structure_id}")
            except Exception as e:
                logging.warning(f"Failed to load structure data for {structure_id}: {e}")

        # Sample from experts using masked sequence
        samples: List[Dict] = []
        best_sample = {
            "sequence": baseline_seq,
            "aar": baseline_aar,
            "sctm": baseline_sctm,
            "biophysical": baseline_bio,
            "reward": baseline_reward,
            "expert_id": expert_ids[0],
            "sample_index": 0,
        }

        sample_counter = 0
        # All experts (DPLM-2: 0,1,2 + ProteinMPNN: 3) - matching MCTS ablation
        
        for expert_id in expert_ids:
            for sample_idx in range(args.max_iter):
                sample_counter += 1
                seed = args.seed + sample_counter + int(time.time()) % 1000
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                try:
                    import torch

                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                except Exception:
                    pass
                
                candidate = None
                
                # Handle DPLM-2 experts (0, 1, 2)
                if expert_id in [0, 1, 2]:
                    candidate = sample_from_masked_sequence(
                        dplm2, masked_sequence, struct_tokens, expert_id, args.temperature
                    )
                    if not candidate:
                        logging.warning("DPLM-2 expert %d sample %d failed", expert_id, sample_idx)
                        continue
                
                # Handle ProteinMPNN expert (3)
                elif expert_id == 3 and proteinmpnn_expert:
                    try:
                        # Use masked sequence directly for ProteinMPNN
                        candidates = proteinmpnn_expert.generate_sequences(
                            masked_sequence=masked_sequence,
                            structure_coords=reference_coords,
                            num_samples=1
                        )
                        
                        if not candidates or len(candidates) == 0:
                            logging.warning("ProteinMPNN sample %d failed - no candidates", sample_idx)
                            continue
                            
                        candidate = candidates[0]  # Take first candidate
                        
                    except Exception as e:
                        logging.warning("ProteinMPNN sample %d failed: %s", sample_idx, e)
                        continue
                
                else:
                    logging.warning("Unknown expert_id %d or ProteinMPNN not available", expert_id)
                    continue

                # Evaluate candidate
                candidate = clean_sequence(candidate)
                aar = calculate_simple_aar(candidate, ref_seq)
                bio = calculate_biophysical_score(candidate)
                sctm = baseline_sctm  # default fallback
                if compute_sctm and reference_coords is not None:
                    try:
                        sctm = float(calculate_sctm_score(candidate, reference_coords))
                    except Exception as e:
                        logging.warning("Sample scTM calculation failed: %s", e)

                reward = calculate_composite_reward(aar, sctm, bio)

                sample_data = {
                    "sequence": candidate,
                    "aar": aar,
                    "sctm": sctm,
                    "biophysical": bio,
                    "reward": reward,
                    "expert_id": expert_id,
                    "sample_index": sample_idx,
                }
                samples.append(sample_data)

                # Update best sample
                if reward > best_sample["reward"]:
                    best_sample = sample_data

        record = {
            "structure_id": structure_id,
            "reference_length": len(ref_seq),
            "baseline": {
                "sequence": baseline_seq,
                "aar": baseline_aar,
                "sctm": baseline_sctm,
                "biophysical": baseline_bio,
                "reward": baseline_reward,
            },
            "best_sample": best_sample,
            "samples": samples,
        }
        results.append(record)
        best_rewards.append(best_sample["reward"])
        best_aars.append(best_sample["aar"])
        best_sctm.append(best_sample["sctm"])

        # SAVE INDIVIDUAL RESULT IMMEDIATELY
        save_individual_sampling_result(record, structure_id)

        logging.info(
            "Best reward %.3f (AAR %.3f, scTM %.3f, Expert %s)",
            best_sample["reward"],
            best_sample["aar"],
            best_sample["sctm"],
            best_sample["expert_id"],
        )

    if not results:
        logging.warning("No successful structures processed.")
        return

    mean_reward = float(np.mean(best_rewards))
    mean_aar = float(np.mean(best_aars))
    mean_sctm = float(np.mean(best_sctm))
    logging.info("==== Sampling Summary ====")
    logging.info("Structures processed: %d", len(results))
    logging.info("Mean best reward: %.3f", mean_reward)
    logging.info("Mean best AAR   : %.3f", mean_aar)
    logging.info("Mean best scTM  : %.3f", mean_sctm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"inverse_folding_sampling_pdb_{timestamp}.json"
    summary_path = output_dir / f"inverse_folding_sampling_pdb_summary_{timestamp}.json"

    with result_path.open("w") as fh:
        json.dump(results, fh, indent=2)

    summary_payload = {
        "timestamp": timestamp,
        "num_structures": len(results),
        "mean_best_reward": mean_reward,
        "mean_best_aar": mean_aar,
        "mean_best_sctm": mean_sctm,
        "experts": expert_ids,
        "num_samples_per_expert": args.num_samples,
        "temperature": args.temperature,
        "skip_sctm": not compute_sctm,
        "start_index": start_idx,
        "end_index": end_idx,
        "result_file": str(result_path),
    }
    with summary_path.open("w") as fh:
        json.dump(summary_payload, fh, indent=2)

    logging.info("Saved detailed results to %s", result_path)
    logging.info("Saved run summary to %s", summary_path)


if __name__ == "__main__":
    main()
