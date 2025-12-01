#!/usr/bin/env python3
"""
De Novo Folding with MCTS - CORRECTED VERSION
Generates 3D structure from sequence using MCTS with multiple experts including ESMFold.

Task: FOLDING (sequence → structure)
- Input: Full amino acid sequence (NOT MASKED)
- Output: 3D structure tokens/coordinates (MASKED initially, decoded by MCTS)
- Evaluation: RMSD and TM-score (NOT AAR)
- Experts: DPLM-2 (150M, 650M, 3B), ProteinMPNN, ESMFold

Key differences from inverse folding:
- Inverse folding: structure given → decode sequence
- Folding: sequence given → decode structure

Usage:
    python test_denovo_folding_mcts_fixed.py --sample-index 0
    python test_denovo_folding_mcts_fixed.py --pdb-id 7dz2_C
    python test_denovo_folding_mcts_fixed.py --max-samples 10 --full-mcts
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

import torch
import numpy as np
from Bio import SeqIO

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'mcts_diffusion_finetune'))

try:
    from core.dplm2_integration import DPLM2Integration
    from core.sequence_level_mcts import GeneralMCTS, MCTSNode
    print("✅ Successfully imported working MCTS components")
except ImportError as e:
    print(f"❌ Failed to import MCTS components: {e}")
    sys.exit(1)

DEFAULT_MCTS_PARAMS: Dict[str, int] = {
    "max_depth": 3,
    "num_simulations": 3,
    "num_iterations": 3,
    "num_rollouts_per_expert": 2,
    "top_k_candidates": 4,
}

FOLDING_PARAMS: Dict[str, int] = {
    "max_depth": 4,
    "num_simulations": 25,
    "num_iterations": 25,
    "num_rollouts_per_expert": 2,
    "top_k_candidates": 4,
}


def resolve_param(args: argparse.Namespace, key: str) -> int:
    """Resolve an MCTS hyperparameter from CLI overrides or defaults."""
    user_value = getattr(args, key, None)
    if user_value is not None:
        return user_value
    base = FOLDING_PARAMS if getattr(args, "full_mcts", False) else DEFAULT_MCTS_PARAMS
    return base[key]


def load_cameo2022_data(max_samples: Optional[int] = 2, target_ids: Optional[Set[str]] = None) -> List[Tuple[str, str, str]]:
    """Load CAMEO2022 data for folding task."""
    cameo_dir = "/home/caom/AID3/dplm/data-bin/cameo2022"
    aa_file = os.path.join(cameo_dir, "aatype.fasta")
    struct_file = os.path.join(cameo_dir, "struct.fasta")

    if not os.path.exists(aa_file) or not os.path.exists(struct_file):
        raise FileNotFoundError(f"CAMEO2022 files not found: {aa_file}, {struct_file}")

    aa_sequences: Dict[str, str] = {}
    for record in SeqIO.parse(aa_file, "fasta"):
        aa_sequences[record.id] = str(record.seq).strip()

    struct_sequences: Dict[str, str] = {}
    for record in SeqIO.parse(struct_file, "fasta"):
        struct_sequences[record.id] = str(record.seq).strip()

    print(f"   📊 Loaded {len(aa_sequences)} AA sequences, {len(struct_sequences)} structure sequences")

    pdb_ids: List[str] = list(aa_sequences.keys())
    if target_ids:
        pdb_ids = [pid for pid in pdb_ids if pid in target_ids]

    if max_samples is not None and max_samples > 0:
        pdb_ids = pdb_ids[:max_samples]

    samples: List[Tuple[str, str, str]] = []
    for pdb_id in pdb_ids:
        if pdb_id not in struct_sequences:
            continue

        aa_seq = aa_sequences[pdb_id]
        struct_tokens = struct_sequences[pdb_id]
        struct_tokens_spaced = struct_tokens.replace(',', ' ') if ',' in struct_tokens else struct_tokens

        samples.append((pdb_id, aa_seq, struct_tokens_spaced))
        print(f"   🔍 {pdb_id}: {len(aa_seq)} residues")
        print(f"      Sequence: {aa_seq[:50]}{'...' if len(aa_seq) > 50 else ''}")
        print(f"      Ground truth structure: {struct_tokens_spaced.split()[:10]}...")

    print(f"✅ Loaded {len(samples)} CAMEO2022 samples for folding")
    return samples


def create_masked_structure(sequence_length: int) -> str:
    """Create all-masked structure tokens for de novo folding.
    
    For FOLDING task:
    - Sequence is GIVEN (full, not masked)
    - Structure is UNKNOWN (all masked with '#')
    """
    masked_tokens = ['#'] * sequence_length
    return ' '.join(masked_tokens)


def calculate_rmsd(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """Calculate RMSD between predicted and reference coordinates."""
    try:
        pred = np.array(pred_coords)
        ref = np.array(ref_coords)
        
        # Handle length mismatches
        min_len = min(len(pred), len(ref))
        pred = pred[:min_len]
        ref = ref[:min_len]
        
        if len(pred) == 0:
            return float('inf')
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred - ref) ** 2, axis=1)))
        return float(rmsd)
    except Exception as e:
        print(f"   ⚠️ RMSD calculation failed: {e}")
        return float('inf')


def calculate_tm_score(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """Calculate TM-score between predicted and reference structures."""
    try:
        pred = np.array(pred_coords)
        ref = np.array(ref_coords)
        
        # Handle length mismatches
        min_len = min(len(pred), len(ref))
        pred = pred[:min_len]
        ref = ref[:min_len]
        
        if len(pred) == 0:
            return 0.0
        
        # Calculate TM-score
        L_target = len(ref)
        d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
        
        distances = np.sqrt(np.sum((pred - ref) ** 2, axis=1))
        tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
        
        return float(tm_score)
    except Exception as e:
        print(f"   ⚠️ TM-score calculation failed: {e}")
        return 0.0


def save_structure_to_fasta(pdb_id: str, sequence: str, struct_tokens: str, output_dir: str):
    """Save predicted structure tokens to FASTA file (same format as generate_dplm2_patched_v2.py)."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Format: >pdb_id\n<cls_struct> token1 token2 ... <eos_struct> <cls_aa> A A S ... <eos_aa>
        # Structure tokens should be space-separated
        if ',' in struct_tokens:
            struct_tokens_formatted = struct_tokens.replace(',', ' ')
        else:
            struct_tokens_formatted = struct_tokens
        
        # Build FASTA content
        fasta_content = f">{pdb_id}\n"
        fasta_content += f"<cls_struct> {struct_tokens_formatted} <eos_struct> "
        fasta_content += f"<cls_aa> {' '.join(list(sequence))} <eos_aa>\n"
        
        # Save to file
        output_file = os.path.join(output_dir, f"{pdb_id}.fasta")
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"   💾 Saved structure tokens to: {output_file}")
        return output_file
    except Exception as e:
        print(f"   ⚠️ Failed to save FASTA: {e}")
        return None


def load_reference_coordinates(pdb_id: str) -> Optional[np.ndarray]:
    """Load ground truth coordinates for evaluation."""
    try:
        from utils.cameo_data_loader import CAMEODataLoader
        
        coord_loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
        structure_idx = None
        
        for idx, structure_file in enumerate(coord_loader.structures):
            if isinstance(structure_file, str):
                base_name = structure_file.replace('.pkl', '')
                if base_name == pdb_id:
                    structure_idx = idx
                    break
        
        if structure_idx is not None:
            cameo_structure = coord_loader.get_structure_by_index(structure_idx)
            if cameo_structure and isinstance(cameo_structure, dict):
                # Try different coordinate keys
                if 'backbone_coords' in cameo_structure and cameo_structure['backbone_coords'] is not None:
                    coords = cameo_structure['backbone_coords']
                    if hasattr(coords, 'shape') and len(coords.shape) == 3 and coords.shape[1] == 3:
                        return coords[:, 1, :]  # CA atoms
                    else:
                        return coords
                elif 'coordinates' in cameo_structure and cameo_structure['coordinates'] is not None:
                    return cameo_structure['coordinates']
                elif 'atom_positions' in cameo_structure and cameo_structure['atom_positions'] is not None:
                    coords = cameo_structure['atom_positions']
                    if hasattr(coords, 'shape') and len(coords.shape) == 3 and coords.shape[1] >= 2:
                        return coords[:, 1, :]  # CA atoms
                    else:
                        return coords
        
        return None
    except Exception as e:
        print(f"   ⚠️ Failed to load coordinates: {e}")
        return None


def generate_esmfold_baseline(sequence: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate baseline structure using ESMFold."""
    try:
        print(f"   🔄 Generating ESMFold baseline for sequence length {len(sequence)}")
        
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            tokenized = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            
            output = model(tokenized['input_ids'])
            positions = output.positions  # Shape: [batch, length, atoms, 3]
            plddt = output.plddt if hasattr(output, 'plddt') else None
            
            # Extract CA coordinates
            if len(positions.shape) == 5:
                ca_coords = positions[0, 0, :, 1, :].cpu().numpy()
            elif len(positions.shape) == 4:
                ca_coords = positions[0, :, 1, :].cpu().numpy()
            else:
                raise ValueError(f"Unexpected ESMFold output shape: {positions.shape}")
            
            # Extract pLDDT scores
            plddt_scores = None
            if plddt is not None:
                plddt_scores = plddt[0].cpu().numpy()
            
            print(f"   ✅ ESMFold baseline generated: {ca_coords.shape}")
            return ca_coords, plddt_scores
            
    except Exception as e:
        print(f"   ❌ ESMFold baseline generation failed: {e}")
        return None, None


def test_denovo_folding_mcts(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], bool]:
    """Run de novo folding using MCTS to decode structure from sequence."""
    if args.sample_index is not None and args.pdb_id:
        raise ValueError("Provide either --sample-index or --pdb-id, not both.")

    target_ids: Optional[Set[str]] = {args.pdb_id} if args.pdb_id else None
    samples = load_cameo2022_data(max_samples=None, target_ids=target_ids)
    total_samples = len(samples)

    if total_samples == 0:
        print("❌ No CAMEO2022 samples loaded")
        return [], False

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= total_samples:
            raise IndexError(f"Sample index {args.sample_index} out of range (0-{total_samples - 1})")
        samples = [samples[args.sample_index]]
    elif args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"   ▶ Running on {len(samples)} sample(s) (out of {total_samples})")

    try:
        print("\n🔄 Loading DPLM-2 integration...")
        dplm2_integration = DPLM2Integration()
        print("✅ DPLM-2 integration loaded")
    except Exception as e:
        print(f"❌ DPLM-2 loading FAILED: {e}")
        raise Exception(f"DPLM-2 integration required but failed to load: {e}")

    results: List[Dict[str, object]] = []

    for i, (pdb_id, sequence, ground_truth_struct) in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"🧪 Processing sample {i + 1}/{len(samples)}: {pdb_id}")
        print(f"{'='*80}")
        print(f"   📏 Sequence length: {len(sequence)} residues")
        print(f"   🧬 Sequence (GIVEN, NOT MASKED): {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        print(f"   🎯 Task: FOLDING (sequence → structure)")

        # Load ground truth coordinates for evaluation
        print(f"\n   🔄 Loading ground truth coordinates for {pdb_id}...")
        reference_coords = load_reference_coordinates(pdb_id)
        if reference_coords is not None:
            print(f"   ✅ Loaded reference coordinates: {reference_coords.shape}")
        else:
            print(f"   ⚠️ No reference coordinates found for {pdb_id}")
            continue

        # Generate ESMFold baseline
        print(f"\n   🔄 Generating ESMFold baseline...")
        baseline_coords, baseline_plddt = generate_esmfold_baseline(sequence)
        if baseline_coords is None:
            print(f"   ❌ ESMFold baseline generation failed, skipping {pdb_id}")
            continue
        
        # Calculate baseline metrics using PROPER alignment
        from utils.folding_metrics import evaluate_folding_metrics
        baseline_rmsd, baseline_tm, baseline_reward = evaluate_folding_metrics(
            baseline_coords, reference_coords, sequence
        )
        print(f"   📊 Baseline ESMFold: RMSD={baseline_rmsd:.3f}Å, TM-score={baseline_tm:.4f}, Reward={baseline_reward:.4f}")

        # Create all-masked structure (de novo folding starts from scratch)
        # CRITICAL: For folding, sequence is GIVEN, structure is MASKED
        masked_struct = create_masked_structure(len(sequence))
        print(f"\n   🎭 Created masked structure: {masked_struct.split()[:10]}... (all masked)")
        print(f"   ✅ Sequence is GIVEN (not masked): {sequence[:30]}...")

        # Prepare baseline structure for MCTS
        baseline_structure = {
            'struct_seq': masked_struct,  # All masked - to be decoded by MCTS
            'sequence': sequence,  # GIVEN sequence (NOT MASKED) - this is the input!
            'coordinates': reference_coords,  # For evaluation only
            'plddt_scores': baseline_plddt,  # From ESMFold baseline
            'baseline_reward': 0.0,
            'baseline_rmsd': baseline_rmsd,
            'baseline_tm': baseline_tm,
            'task': 'folding',  # FOLDING task (not inverse_folding)
            'task_type': 'folding'
        }

        # Resolve MCTS parameters
        max_depth = resolve_param(args, "max_depth")
        num_simulations = resolve_param(args, "num_simulations")
        num_iterations = resolve_param(args, "num_iterations")
        num_rollouts = resolve_param(args, "num_rollouts_per_expert")
        top_k = resolve_param(args, "top_k_candidates")

        print(f"\n🌲 MCTS Parameters:")
        print(f"   Max depth: {max_depth}")
        print(f"   Simulations: {num_simulations}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Rollouts per expert: {num_rollouts}")
        print(f"   Top-k candidates: {top_k}")

        # Initialize MCTS
        mcts = GeneralMCTS(
            baseline_structure=baseline_structure,
            dplm2_integration=dplm2_integration,
            max_depth=max_depth,
            num_simulations=num_simulations,
            top_k_candidates=top_k,
            num_rollouts_per_expert=num_rollouts,
            task_type='folding',  # CRITICAL: Use task_type parameter for folding task
            reference_sequence=sequence  # For evaluation
        )

        # Run MCTS
        print(f"\n🚀 Running MCTS folding for {pdb_id}...")
        start_time = time.time()
        
        try:
            # For folding: sequence is fixed (initial_sequence), structure is decoded
            # CRITICAL: Pass the FULL sequence, not None!
            best_node = mcts.search(
                initial_sequence=sequence,  # GIVEN sequence as input (NOT MASKED!)
                num_iterations=num_iterations,
                reference_sequence=sequence  # Same as initial for folding
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract results directly from MCTSNode fields
            best_reward = best_node.reward if hasattr(best_node, 'reward') else 0.0
            predicted_sequence = best_node.sequence if hasattr(best_node, 'sequence') else sequence
            predicted_struct = best_node.structure_tokens if hasattr(best_node, 'structure_tokens') else masked_struct
            predicted_coords = best_node.coordinates if hasattr(best_node, 'coordinates') else None
            node_rmsd = best_node.rmsd if hasattr(best_node, 'rmsd') else None
            node_tm = best_node.tm_score if hasattr(best_node, 'tm_score') else None
            
            print(f"\n✅ MCTS completed in {elapsed_time:.2f}s")
            print(f"   Best node reward: {best_reward:.4f}")
            print(f"   Best node has coordinates: {predicted_coords is not None}")
            if predicted_coords is not None:
                print(f"   Predicted coordinates shape: {predicted_coords.shape}")
            print(f"   Best node has structure_tokens: {predicted_struct is not None}")
            print(f"   Best node has RMSD/TM: {node_rmsd is not None}/{node_tm is not None}")
            
            print(f"   Predicted structure tokens: {predicted_struct.split()[:10]}...")
            
            # Evaluate against ground truth
            # Use stored values from node if available, otherwise calculate
            if node_rmsd is not None and node_tm is not None:
                rmsd = node_rmsd
                tm_score = node_tm
                print(f"\n   ✅ Using stored RMSD/TM from best node")
            elif predicted_coords is not None and reference_coords is not None:
                # Use proper alignment for recalculation
                rmsd, tm_score, _ = evaluate_folding_metrics(
                    predicted_coords, reference_coords, sequence
                )
                print(f"\n   🔄 Calculated RMSD/TM from coordinates (with alignment)")
            else:
                rmsd = float('inf')
                tm_score = 0.0
                print(f"\n   ⚠️ Cannot calculate metrics (missing coordinates)")
            
            if rmsd != float('inf'):
                # Calculate improvements
                rmsd_improvement = baseline_rmsd - rmsd
                tm_improvement = tm_score - baseline_tm
                
                print(f"\n   📊 Final Results:")
                print(f"      RMSD:     {baseline_rmsd:.3f}Å → {rmsd:.3f}Å (Δ {rmsd_improvement:+.3f}Å)")
                print(f"      TM-score: {baseline_tm:.4f} → {tm_score:.4f} (Δ {tm_improvement:+.4f})")
                print(f"      Improved: {rmsd_improvement > 0 or tm_improvement > 0}")
            
            # Save structure tokens to FASTA file (same format as generate_dplm2_patched_v2.py)
            fasta_output_dir = os.path.join(args.output_dir, "fasta")
            fasta_file = save_structure_to_fasta(pdb_id, sequence, predicted_struct, fasta_output_dir)
            
            # Store results
            result = {
                'pdb_id': pdb_id,
                'sequence': sequence,
                'sequence_length': len(sequence),
                'ground_truth_struct': ground_truth_struct,
                'predicted_struct': predicted_struct,
                'reward': best_reward,
                'baseline_rmsd': baseline_rmsd,
                'baseline_tm': baseline_tm,
                'final_rmsd': rmsd,
                'final_tm': tm_score,
                'rmsd_improvement': baseline_rmsd - rmsd if rmsd != float('inf') else 0.0,
                'tm_improvement': tm_score - baseline_tm,
                'improved': (baseline_rmsd - rmsd > 0) or (tm_score - baseline_tm > 0),
                'elapsed_time': elapsed_time,
                'fasta_file': fasta_file,
                'mcts_params': {
                    'max_depth': max_depth,
                    'num_simulations': num_simulations,
                    'num_iterations': num_iterations,
                    'num_rollouts_per_expert': num_rollouts,
                    'top_k_candidates': top_k
                }
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"\n❌ MCTS failed for {pdb_id}: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'pdb_id': pdb_id,
                'sequence': sequence,
                'error': str(e),
                'baseline_rmsd': baseline_rmsd if 'baseline_rmsd' in locals() else float('inf'),
                'baseline_tm': baseline_tm if 'baseline_tm' in locals() else 0.0,
                'final_rmsd': float('inf'),
                'final_tm': 0.0
            }
            results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: De Novo Folding Results")
    print(f"{'='*80}")
    
    successful = [r for r in results if 'error' not in r]
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_baseline_rmsd = np.mean([r['baseline_rmsd'] for r in successful if r['baseline_rmsd'] != float('inf')])
        avg_final_rmsd = np.mean([r['final_rmsd'] for r in successful if r['final_rmsd'] != float('inf')])
        avg_baseline_tm = np.mean([r['baseline_tm'] for r in successful])
        avg_final_tm = np.mean([r['final_tm'] for r in successful])
        avg_reward = np.mean([r['reward'] for r in successful])
        improved_count = sum(1 for r in successful if r.get('improved', False))
        
        print(f"\nBaseline (ESMFold):")
        print(f"  Average RMSD: {avg_baseline_rmsd:.3f} Å")
        print(f"  Average TM-score: {avg_baseline_tm:.4f}")
        
        print(f"\nMCTS Optimized:")
        print(f"  Average RMSD: {avg_final_rmsd:.3f} Å")
        print(f"  Average TM-score: {avg_final_tm:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
        
        print(f"\nImprovement:")
        print(f"  RMSD: {avg_baseline_rmsd - avg_final_rmsd:+.3f} Å")
        print(f"  TM-score: {avg_final_tm - avg_baseline_tm:+.4f}")
        print(f"  Structures improved: {improved_count}/{len(successful)} ({improved_count/len(successful)*100:.1f}%)")
    
    return results, len(successful) == len(results)


def main():
    parser = argparse.ArgumentParser(description="De Novo Folding with MCTS (CORRECTED)")
    
    # Sample selection
    parser.add_argument("--sample-index", type=int, default=None,
                       help="Index of specific sample to process")
    parser.add_argument("--pdb-id", type=str, default=None,
                       help="Specific PDB ID to process")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    
    # MCTS parameters
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Maximum depth for MCTS tree")
    parser.add_argument("--num-simulations", type=int, default=None,
                       help="Number of simulations per iteration")
    parser.add_argument("--num-iterations", type=int, default=None,
                       help="Number of MCTS iterations")
    parser.add_argument("--num-rollouts-per-expert", type=int, default=None,
                       help="Number of rollouts per expert")
    parser.add_argument("--top-k-candidates", type=int, default=None,
                       help="Number of top candidates to keep")
    parser.add_argument("--full-mcts", action="store_true",
                       help="Use full MCTS parameters (more intensive)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="folding-results/mcts",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"DE NOVO FOLDING WITH MCTS - CORRECTED VERSION")
    print(f"{'='*80}")
    print(f"Task: FOLDING (sequence → structure)")
    print(f"Input: Full amino acid sequence (GIVEN, NOT MASKED)")
    print(f"Output: 3D structure (MASKED initially, decoded by MCTS)")
    print(f"Evaluation: RMSD and TM-score (NOT AAR)")
    print(f"Experts: DPLM-2 (150M, 650M, 3B), ProteinMPNN, ESMFold")
    print(f"{'='*80}\n")
    
    results, success = test_denovo_folding_mcts(args)
    
    # Save results
    if results:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"folding_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
