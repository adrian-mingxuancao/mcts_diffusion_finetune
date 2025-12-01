#!/usr/bin/env python3
"""
MCTS Folding Ablation Study - PDB Dataset

This script tests the MCTS folding pipeline with PDB structures:
- ESMFold baseline for comparison
- MCTS optimization using 3 DPLM experts (650M, 150M, 3B)
- Proper structure token to coordinate conversion
- RMSD and TM-score evaluation

Usage:
python mcts_folding_ablation_pdb.py 0 10 --mode single_expert --single_expert_id 1
python mcts_folding_ablation_pdb.py 0 10 --mode multi_expert
"""

import os, sys, json, time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from Bio import SeqIO

# Import MCTS and DPLM-2 components
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration
from utils.structure_converter import get_structure_converter

# Import evaluation utilities
try:
    from utils.pdb_data_loader import PDBDataLoader
except ImportError:
    class PDBDataLoader:
        def __init__(self, *args, **kwargs):
            self.structures = []
        def get_test_structure(self, index=0):
            return {
                "name": f"test_structure_{index}",
                "sequence": "MKLLVLGLGAGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHQYREQIKRVKDSDDVPMVLVGNKCDLAARTVESRQAQDLARSYGIPYIETSAKTRQGVEDAFYTLVREIRQHKLRKLNPPDESGPGCMNCKCVIS",
                "length": 166
            }

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def load_pdb_sequences(pdb_loader) -> Dict[str, str]:
    """Load PDB sequences from the PDB data loader"""
    sequences = {}
    
    try:
        # Use reference_sequences from the loader (loaded from aatype.fasta)
        for structure_file in pdb_loader.structures:
            # Extract structure name from file path (e.g., "a0/5S9R.pkl" -> "5S9R")
            structure_name = os.path.splitext(os.path.basename(structure_file))[0]
            
            # Get sequence from reference_sequences dict
            if structure_name in pdb_loader.reference_sequences:
                sequences[structure_name] = pdb_loader.reference_sequences[structure_name]
        
        print(f"✅ Loaded {len(sequences)} PDB sequences from data loader")
    except Exception as e:
        print(f"⚠️ PDB sequence loading failed: {e}")
        import traceback
        traceback.print_exc()
        sequences = {}
    
    return sequences

def kabsch_rmsd(pred_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    """Calculate RMSD after Kabsch alignment"""
    # Center both structures
    pred_center = pred_coords.mean(axis=0)
    ref_center = ref_coords.mean(axis=0)
    pred_centered = pred_coords - pred_center
    ref_centered = ref_coords - ref_center
    
    # Compute optimal rotation using SVD
    H = pred_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_aligned = (R @ pred_centered.T).T + ref_center
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum((pred_aligned - ref_coords) ** 2, axis=1)))
    return rmsd

def calculate_rmsd_and_tmscore(predicted_coords: np.ndarray, reference_coords: np.ndarray, 
                                pred_seq: str = None, ref_seq: str = None) -> Tuple[float, float]:
    """
    Calculate RMSD and TM-score between predicted and reference structures.
    
    Uses:
    - RMSD: Kabsch alignment (optimal superposition)
    - TM-score: tmtools.tm_align if sequences provided; otherwise length-scaled fallback
    """
    try:
        pred_coords = np.array(predicted_coords, dtype=np.float64)  # tmtools needs float64
        ref_coords = np.array(reference_coords, dtype=np.float64)
        
        # Handle length mismatches
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        if len(pred_coords) == 0:
            return float('inf'), 0.0
        
        # Calculate RMSD using Kabsch alignment
        rmsd = kabsch_rmsd(pred_coords.astype(np.float32), ref_coords.astype(np.float32))
        
        # Calculate TM-score using tmtools if sequences are provided
        tm_score = None
        if pred_seq is not None and ref_seq is not None:
            try:
                import tmtools
                
                # Truncate sequences to match coordinates
                pred_seq_truncated = pred_seq[:min_len]
                ref_seq_truncated = ref_seq[:min_len]
                
                # tmtools.tm_align(mobile_coords, target_coords, mobile_seq, target_seq)
                tm_results = tmtools.tm_align(pred_coords, ref_coords, pred_seq_truncated, ref_seq_truncated)
                tm_score = tm_results.tm_norm_chain1  # TM-score normalized by chain 1 length
                
            except Exception as e:
                print(f"  ⚠️ tmtools tm_align failed: {e}, using fallback TM-score")
                tm_score = None
        
        # Fallback TM-score if tmtools failed or no sequences provided
        if tm_score is None:
            L_target = len(ref_coords)
            d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
            
            # Use Kabsch-aligned coordinates for TM-score
            pred_center = pred_coords.mean(axis=0)
            ref_center = ref_coords.mean(axis=0)
            pred_centered = pred_coords - pred_center
            ref_centered = ref_coords - ref_center
            
            H = pred_centered.T @ ref_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            pred_aligned = (R @ pred_centered.T).T + ref_center
            
            distances = np.sqrt(np.sum((pred_aligned - ref_coords) ** 2, axis=1))
            tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
        
        return float(rmsd), float(tm_score)
        
    except Exception as e:
        print(f"⚠️ RMSD/TM-score calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0

def predict_structure_with_esmfold(sequence: str) -> Optional[np.ndarray]:
    """Predict structure using ESMFold and return CA coordinates"""
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model (cached)
        if not hasattr(predict_structure_with_esmfold, 'model'):
            print("🔄 Loading ESMFold model...")
            predict_structure_with_esmfold.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            predict_structure_with_esmfold.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            predict_structure_with_esmfold.model.eval()
            print("✅ ESMFold model loaded")
        
        model = predict_structure_with_esmfold.model
        tokenizer = predict_structure_with_esmfold.tokenizer
        
        # Tokenize and predict
        tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        
        with torch.no_grad():
            output = model(tokenized['input_ids'])
        
        # Extract CA coordinates - fix shape handling
        positions = output['positions']
        print(f"🔧 ESMFold output shape: {positions.shape}")
        
        # Handle different ESMFold output formats
        if positions.dim() == 5:  # (batch, extra_dim, seq_len, atoms, 3) - e.g., [8, 1, 65, 14, 3]
            ca_coords = positions[0, 0, :, 1, :].cpu().numpy()  # CA atoms (index 1) - FIXED: added missing middle dimension
        elif positions.dim() == 4:  # (batch, residue, atom, 3) 
            ca_coords = positions[0, :, 1, :].cpu().numpy()
        elif positions.dim() == 3:  # (residue, atom, 3)
            ca_coords = positions[:, 1, :].cpu().numpy()
        else:
            # Fallback - try to extract CA coordinates
            ca_coords = positions.cpu().numpy()
            if ca_coords.ndim > 2:
                ca_coords = ca_coords.reshape(-1, ca_coords.shape[-1])
        
        print(f"🔧 Extracted CA coordinates shape: {ca_coords.shape}")
        
        # Extract pLDDT scores from ESMFold output
        plddt_scores = None
        if 'plddt' in output:
            plddt_raw = output['plddt'].cpu().numpy()
            print(f"🔧 ESMFold pLDDT shape: {plddt_raw.shape}, range: {plddt_raw.min():.3f}-{plddt_raw.max():.3f}")
            
            # Handle batch dimensions and extract per-residue confidence
            if plddt_raw.ndim > 1:
                plddt_scores = plddt_raw.flatten()[:len(sequence)]
            else:
                plddt_scores = plddt_raw[:len(sequence)]
                
            # ESMFold pLDDT is typically 0-100 already, but check scale
            if plddt_scores.max() <= 1.0:
                plddt_scores = plddt_scores * 100.0  # Convert 0-1 to 0-100
                print(f"🔧 Converted ESMFold pLDDT from 0-1 to 0-100 scale")
            
            print(f"✅ ESMFold pLDDT extracted: mean={plddt_scores.mean():.1f}, range={plddt_scores.min():.1f}-{plddt_scores.max():.1f}")
        else:
            print("⚠️ No pLDDT scores in ESMFold output")
        
        # Validate coordinate extraction
        if len(ca_coords) != len(sequence):
            print(f"⚠️ Coordinate length mismatch: got {len(ca_coords)}, expected {len(sequence)}")
            # Try to fix by padding or truncating
            if len(ca_coords) < len(sequence):
                # Pad with linear extension
                padding_needed = len(sequence) - len(ca_coords)
                if len(ca_coords) > 0:
                    last_coord = ca_coords[-1]
                    padding = np.array([last_coord + i * np.array([3.8, 0, 0]) for i in range(1, padding_needed + 1)])
                    ca_coords = np.vstack([ca_coords, padding])
                else:
                    # Generate synthetic coordinates
                    ca_coords = np.array([[i * 3.8, 0, 0] for i in range(len(sequence))])
            else:
                # Truncate to sequence length
                ca_coords = ca_coords[:len(sequence)]
        
        print(f"🔧 Final CA coordinates shape: {ca_coords.shape}")
        return ca_coords, plddt_scores
        
    except Exception as e:
        print(f"❌ ESMFold prediction failed: {e}")
        return None, None

def generate_baseline_structure(sequence: str, dplm2: DPLM2Integration, structure_idx: int = None, seed: int = 42) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Generate baseline structure using DPLM-2 150M with fixed seed.
    
    Returns:
        Tuple of (structure_tokens, plddt_scores)
        - structure_tokens: comma-separated structure tokens from DPLM-2
        - plddt_scores: None (DPLM-2 doesn't provide pLDDT)
    """
    try:
        print(f"🎯 Generating DPLM-2 150M baseline structure for sequence length {len(sequence)}")
        print(f"   Using fixed seed: {seed}")
        
        # Set seed for reproducibility
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        # Generate structure tokens using DPLM-2 150M (expert_id=1)
        structure_tokens = dplm2.generate_baseline_structure(
            sequence=sequence,
            expert_id=1  # DPLM-2 150M
        )
        
        if structure_tokens:
            token_count = len(structure_tokens.split(','))
            print(f"✅ DPLM-2 150M baseline generation successful: {token_count} structure tokens")
            return structure_tokens, None  # Return structure tokens, no pLDDT
        else:
            print(f"❌ DPLM-2 150M baseline generation failed")
            return None, None
        
    except Exception as e:
        print(f"❌ Baseline structure generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_folding_ablation(sequence: str, structure_name: str, reference_coords: np.ndarray,
                        dplm2: DPLM2Integration, ablation_mode: str, single_expert_id: int = None,
                        structure_idx: int = None, num_iterations: int = 50, max_depth: int = 10,
                        use_ph_uct: bool = False) -> Optional[Dict]:
    """
    Run single folding ablation experiment.
    
    Args:
        sequence: Input amino acid sequence
        structure_name: Structure identifier
        reference_coords: Reference CA coordinates
        dplm2: DPLM-2 integration
        ablation_mode: "random_no_expert", "single_expert", or "multi_expert"
        single_expert_id: Expert ID for single expert mode
        structure_idx: Structure index for data loading
        
    Returns:
        Results dictionary or None if failed
    """
    print(f"\n🧬 [{ablation_mode}{'' if single_expert_id is None else f'_{single_expert_id}'}] {structure_name}")
    print(f"  📊 Sequence length: {len(sequence)}")
    print(f"  🎯 Task: Folding optimization")
    
    # Generate baseline structure using DPLM-2 150M with fixed seed
    baseline_struct_tokens, baseline_plddt = generate_baseline_structure(sequence, dplm2, structure_idx, seed=42)
    if baseline_struct_tokens is None:
        print("  ❌ Baseline structure generation failed")
        return None
    
    # Use the DPLM-2 generated structure tokens directly
    struct_seq_str = baseline_struct_tokens
    print(f"  ✅ Using DPLM-2 150M baseline: {len(struct_seq_str.split(','))} structure tokens")
    
    # For baseline metrics, we need to convert structure tokens to coordinates
    # This is needed to calculate RMSD/TM-score against reference
    baseline_coords = None
    baseline_plddt = None
    try:
        # Convert structure tokens to coordinates and pLDDT using structure tokenizer
        baseline_coords, baseline_plddt = dplm2._structure_tokens_to_coords(baseline_struct_tokens, len(sequence))
        if baseline_coords is not None:
            # Calculate baseline metrics (with sequences for proper TM-score)
            baseline_rmsd, baseline_tmscore = calculate_rmsd_and_tmscore(
                baseline_coords, reference_coords, 
                pred_seq=sequence, ref_seq=sequence  # Same sequence for folding task
            )
            print(f"  ✅ Baseline metrics: RMSD={baseline_rmsd:.3f}Å, TM-score={baseline_tmscore:.3f}")
            if baseline_plddt is not None:
                print(f"  ✅ Baseline pLDDT: mean={baseline_plddt.mean():.1f}, range={baseline_plddt.min():.1f}-{baseline_plddt.max():.1f}")
        else:
            print(f"  ⚠️ Could not convert baseline structure tokens to coordinates")
            baseline_rmsd, baseline_tmscore = None, None
    except Exception as e:
        print(f"  ⚠️ Baseline metric calculation failed: {e}")
        baseline_rmsd, baseline_tmscore = None, None
        baseline_plddt = None
    
    # Prepare baseline structure data for MCTS
    baseline_structure = {
        'sequence': sequence,
        'coordinates': baseline_coords,
        'length': len(sequence),
        'plddt_scores': baseline_plddt,  # Use ESMFold pLDDT for initial masking
        'structure_idx': structure_idx,
        'struct_seq': struct_seq_str,  # Use converted tokens or FASTA fallback
        'name': structure_name,  # Add name for fallback loading
        'pdb_id': structure_name.split('_')[0] if '_' in structure_name else structure_name,
        'chain_id': structure_name.split('_')[1] if '_' in structure_name else 'A'
    }
    
    # Configure MCTS based on ablation mode
    mcts_kwargs = {
        'dplm2_integration': dplm2,
        'reference_sequence': sequence,
        'baseline_structure': baseline_structure,
        'reference_coords': reference_coords,  # Add reference coordinates for folding evaluation
        'max_depth': max_depth,
        'backup_rule': "max",
        'use_ph_uct': use_ph_uct,  # Control PH-UCT vs standard UCT
        'task_type': "folding"  # Critical: Set task type for proper reward calculation
    }
    
    if ablation_mode == "random_no_expert":
        mcts_kwargs.update({
            'ablation_mode': "random_no_expert",
            'num_children_select': 4
        })
    elif ablation_mode == "single_expert":
        mcts_kwargs.update({
            'ablation_mode': "single_expert",
            'single_expert_id': int(single_expert_id or 0),
            'k_rollouts_per_expert': 3,
            'num_children_select': 3
        })
    else:  # multi_expert
        mcts_kwargs.update({
            'ablation_mode': "multi_expert",
            'k_rollouts_per_expert': 2,
            'num_children_select': 2
        })
    
    try:
        mcts = GeneralMCTS(**mcts_kwargs)
        
        # Compute baseline reward for comparison using proper structure reward
        baseline_reward = 0.0
        if baseline_rmsd is not None and baseline_tmscore is not None:
            try:
                # For folding tasks, use structure reward based on baseline RMSD/TM-score
                baseline_reward = mcts._compute_structure_reward(baseline_rmsd, baseline_tmscore)
                print(f"  📈 Baseline reward: {baseline_reward:.3f}")
            except Exception as e:
                print(f"  ⚠️ Baseline reward computation failed: {e}")
                baseline_reward = 0.0
        else:
            print(f"  ⚠️ Baseline metrics unavailable, using reward=0.0")
        
        # Run MCTS search
        start_time = time.time()
        root_node = mcts.search(initial_sequence=sequence, num_iterations=num_iterations)
        search_time = time.time() - start_time
        
        # Find best node in tree
        def find_best_node(node):
            best_node, best_score = node, getattr(node, "reward", 0.0)
            for child in node.children:
                child_best = find_best_node(child)
                child_score = getattr(child_best, "reward", 0.0)
                if child_score > best_score:
                    best_node, best_score = child_best, child_score
            return best_node
        
        best_node = find_best_node(root_node)
        best_sequence = best_node.sequence
        best_reward = getattr(best_node, "reward", 0.0)
        
        # Use stored results from best node if available, otherwise regenerate
        if hasattr(best_node, 'rmsd') and hasattr(best_node, 'tm_score') and best_node.rmsd is not None and best_node.tm_score is not None:
            print(f"  ✅ Using stored results from best node: RMSD={best_node.rmsd:.3f}Å, TM-score={best_node.tm_score:.3f}")
            final_rmsd = best_node.rmsd
            final_tmscore = best_node.tm_score
        else:
            print(f"  ⚠️ Best node missing stored results, regenerating structure...")
            # Generate final structure for best sequence using ESMFold for consistency
            final_coords, _ = predict_structure_with_esmfold(best_sequence)
            if final_coords is None:
                print(f"  ❌ Final structure prediction failed")
                return None
            
            print(f"  ✅ Final structure generated: {final_coords.shape}")
            
            # Calculate final metrics
            final_rmsd, final_tmscore = calculate_rmsd_and_tmscore(final_coords, reference_coords)
        
        # Calculate improvements (handle None values)
        if baseline_rmsd is not None and final_rmsd is not None:
            rmsd_improvement = baseline_rmsd - final_rmsd  # Lower RMSD is better
        else:
            rmsd_improvement = None
            
        if baseline_tmscore is not None and final_tmscore is not None:
            tmscore_improvement = final_tmscore - baseline_tmscore  # Higher TM-score is better
        else:
            tmscore_improvement = None
            
        structure_improved = (rmsd_improvement is not None and rmsd_improvement > 0) or \
                           (tmscore_improvement is not None and tmscore_improvement > 0)
        
        # Store results (convert numpy types to Python types for JSON serialization)
        result = {
            'structure_name': structure_name,
            'sequence_length': len(sequence),
            'ablation_mode': ablation_mode,
            'single_expert_id': single_expert_id,
            'baseline_rmsd': float(baseline_rmsd) if baseline_rmsd is not None else None,
            'baseline_tmscore': float(baseline_tmscore) if baseline_tmscore is not None else None,
            'baseline_reward': float(baseline_reward),
            'final_rmsd': float(final_rmsd),
            'final_tmscore': float(final_tmscore),
            'final_reward': float(best_reward),
            'rmsd_improvement': float(rmsd_improvement) if rmsd_improvement is not None else None,
            'tmscore_improvement': float(tmscore_improvement) if tmscore_improvement is not None else None,
            'reward_improvement': float(best_reward - baseline_reward),
            'improved': bool(structure_improved),  # Convert to Python bool
            'search_time': float(search_time),
            'total_time': float(time.time() - start_time),
            'num_iterations': int(num_iterations),
            'best_depth': int(getattr(best_node, "depth", 0)),
            'tree_size': int(len([n for n in [root_node] + getattr(root_node, 'children', []) if hasattr(n, 'sequence')]))
        }
        
        print("  ─────────────────────────────────────────────────────────")
        print(f"  Summary [{ablation_mode}{'' if single_expert_id is None else f'/{single_expert_id}'}] {structure_name}")
        
        # Print metrics with None handling
        if baseline_rmsd is not None and rmsd_improvement is not None:
            print(f"    RMSD    : {baseline_rmsd:.3f}Å → {final_rmsd:.3f}Å (Δ {rmsd_improvement:+.3f}Å)")
        else:
            print(f"    RMSD    : N/A → {final_rmsd:.3f}Å")
            
        if baseline_tmscore is not None and tmscore_improvement is not None:
            print(f"    TM-score: {baseline_tmscore:.3f} → {final_tmscore:.3f} (Δ {tmscore_improvement:+.3f})")
        else:
            print(f"    TM-score: N/A → {final_tmscore:.3f}")
            
        print(f"    Reward  : {baseline_reward:.3f} → {best_reward:.3f} (Δ {best_reward - baseline_reward:+.3f})")
        print(f"    Improved: {structure_improved}")
        
        # Add structure_improved to result for summary
        result['structure_improved'] = bool(structure_improved)
        print(f"    Time    : {search_time:.1f}s")
        print("  ─────────────────────────────────────────────────────────")
        
        return result
        
    except Exception as e:
        print(f"  ❌ MCTS search failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main ablation study function"""
    print("🧬 MCTS Folding Ablation Study - PDB Dataset Evaluation")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs='?', type=int, default=0, help="start index (inclusive)")
    parser.add_argument("end", nargs='?', type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--mode", choices=["random_no_expert", "single_expert", "multi_expert", "all"], default="all")
    parser.add_argument("--single_expert_id", type=int, default=None, help="single expert id (0/1/2)")
    parser.add_argument("--num_iterations", type=int, default=50, help="number of MCTS iterations")
    parser.add_argument("--max_depth", type=int, default=10, help="maximum MCTS tree depth")
    parser.add_argument("--use_standard_uct", action="store_true", help="use standard UCT instead of PH-UCT (default: PH-UCT)")
    parser.add_argument("--temperature", type=float, default=1.0, help="generation temperature for DPLM-2")
    args = parser.parse_args()
    
    start_idx = args.start
    end_idx = args.end
    print(f"🎯 Structure range: {start_idx}-{end_idx if end_idx is not None else 'end'} | Mode: {args.mode}")
    
    # Load PDB data
    try:
        loader = PDBDataLoader(data_path="/home/caom/AID3/dplm/data-bin/PDB_date")
        print(f"✅ Loaded PDB structure loader")
    except Exception as e:
        print(f"❌ Failed to load PDB structures: {e}")
        return
    
    # Load PDB sequences
    pdb_sequences = load_pdb_sequences(loader)
    if not pdb_sequences:
        print("❌ Failed to load PDB sequences")
        return
    
    # Initialize DPLM-2 with multi-expert support (650M, 150M, 3B)
    try:
        dplm2 = DPLM2Integration(device="cuda", default_temperature=args.temperature)  # Use consolidated integration
        print(f"✅ DPLM-2 multi-expert integration initialized (supports 650M, 150M, 3B)")
        print(f"   Temperature: {args.temperature}")
        print(f"   UCT mode: {'Standard UCT' if args.use_standard_uct else 'PH-UCT (entropy-based)'}")
    except Exception as e:
        print(f"❌ Failed to initialize DPLM-2: {e}")
        return
    
    # Create test pairs (sequence + reference structure)
    test_pairs = []
    skipped_no_seq = 0
    skipped_no_coords = 0
    skipped_errors = 0
    
    for idx, structure_file in enumerate(loader.structures):
        # Extract structure name from file path (e.g., "a0/5S9R.pkl" -> "5S9R")
        structure_name = os.path.splitext(os.path.basename(structure_file))[0]
        
        # Get sequence from PDB data loader
        if structure_name in pdb_sequences:
            sequence = pdb_sequences[structure_name]
            
            # Get reference structure coordinates
            try:
                structure = loader.get_structure_by_index(idx)
                if structure:
                    # Extract reference coordinates
                    ref_coords = None
                    if 'backbone_coords' in structure and structure['backbone_coords'] is not None:
                        coords = structure['backbone_coords']
                        if len(coords.shape) == 3 and coords.shape[1] >= 2:
                            ref_coords = coords[:, 1, :]  # CA atoms
                        else:
                            ref_coords = coords
                    elif 'coordinates' in structure and structure['coordinates'] is not None:
                        ref_coords = structure['coordinates']
                    elif 'atom_positions' in structure and structure['atom_positions'] is not None:
                        coords = structure['atom_positions']
                        if len(coords.shape) == 3 and coords.shape[1] >= 2:
                            ref_coords = coords[:, 1, :]  # CA atoms
                        else:
                            ref_coords = coords
                    
                    if ref_coords is not None:
                        test_pairs.append((idx, structure_name, sequence, ref_coords))
                    else:
                        skipped_no_coords += 1
                else:
                    skipped_errors += 1
            except Exception as e:
                skipped_errors += 1
        else:
            skipped_no_seq += 1
    
    print(f"📊 Structure matching summary:")
    print(f"   ✅ Matched: {len(test_pairs)} structures with sequences and coordinates")
    print(f"   ⚠️  Skipped: {skipped_no_seq} (no sequence), {skipped_no_coords} (no coords), {skipped_errors} (errors)")
    
    # Select test range
    if end_idx is not None:
        test_pairs = test_pairs[start_idx:end_idx]
    else:
        test_pairs = test_pairs[start_idx:]
    
    print(f"📊 Selected {len(test_pairs)} structure pairs to process")
    
    # Run ablation experiments
    results = []
    
    for idx, structure_name, sequence, ref_coords in test_pairs:
        print(f"\n{'='*70}")
        print(f"PROCESSING: {structure_name}")
        print(f"{'='*70}")
        
        if args.mode == "random_no_expert":
            result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2, 
                                        "random_no_expert", structure_idx=idx,
                                        num_iterations=args.num_iterations, max_depth=args.max_depth,
                                        use_ph_uct=not args.use_standard_uct)
            if result:
                results.append(result)
                
        elif args.mode == "single_expert":
            expert_ids = [args.single_expert_id] if args.single_expert_id is not None else [0, 1, 2]
            for expert_id in expert_ids:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            "single_expert", expert_id, structure_idx=idx,
                                            num_iterations=args.num_iterations, max_depth=args.max_depth,
                                            use_ph_uct=not args.use_standard_uct)
                if result:
                    results.append(result)
                    
        elif args.mode == "multi_expert":
            result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                        "multi_expert", structure_idx=idx,
                                        num_iterations=args.num_iterations, max_depth=args.max_depth,
                                        use_ph_uct=not args.use_standard_uct)
            if result:
                results.append(result)
                
        else:  # args.mode == "all"
            # Run all ablation modes
            for mode in ["random_no_expert", "multi_expert"]:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            mode, structure_idx=idx,
                                            num_iterations=args.num_iterations, max_depth=args.max_depth,
                                            use_ph_uct=not args.use_standard_uct)
                if result:
                    results.append(result)
            
            # Run single expert modes
            for expert_id in [0, 1, 2]:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            "single_expert", expert_id, structure_idx=idx,
                                            num_iterations=args.num_iterations, max_depth=args.max_depth,
                                            use_ph_uct=not args.use_standard_uct)
                if result:
                    results.append(result)
    
    # Save results
    output_dir = "/net/scratch/caom/pdb_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"mcts_folding_ablation_pdb_results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved ablation results → {json_file}")
    
    # Create summary table
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["ablation_mode"]].append(r)
    
    summary_file = os.path.join(output_dir, f"mcts_folding_ablation_pdb_summary_{timestamp}.txt")
    try:
        with open(summary_file, "w") as f:
            f.write("MCTS Folding Ablation Summary\n")
            f.write("=" * 80 + "\n\n")
            
            for mode, rows in grouped.items():
                f.write(f"Mode: {mode}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Structure':<20} {'Len':<5} {'Base RMSD':<10} {'Final RMSD':<11} {'ΔRMSD':<8} "
                       f"{'Base TM':<8} {'Final TM':<9} {'ΔTM':<8} {'Improved':<8} {'Time(s)':<8}\n")
                
                for r in rows:
                    name = r['structure_name'][:19]
                    # Handle None values in formatting
                    base_rmsd = f"{r['baseline_rmsd']:<10.3f}" if r['baseline_rmsd'] is not None else "N/A       "
                    rmsd_imp = f"{r['rmsd_improvement']:<8.3f}" if r['rmsd_improvement'] is not None else "N/A     "
                    base_tm = f"{r['baseline_tmscore']:<8.3f}" if r['baseline_tmscore'] is not None else "N/A     "
                    tm_imp = f"{r['tmscore_improvement']:<8.3f}" if r['tmscore_improvement'] is not None else "N/A     "
                    
                    f.write(f"{name:<20} {r['sequence_length']:<5} {base_rmsd} "
                           f"{r['final_rmsd']:<11.3f} {rmsd_imp} "
                           f"{base_tm} {r['final_tmscore']:<9.3f} "
                           f"{tm_imp} {str(r['structure_improved']):<8} "
                           f"{r['search_time']:<8.1f}\n")
                
                # Statistics (skip None values)
                if rows:
                    valid_rmsd = [x['rmsd_improvement'] for x in rows if x['rmsd_improvement'] is not None]
                    valid_tm = [x['tmscore_improvement'] for x in rows if x['tmscore_improvement'] is not None]
                    avg_rmsd_improvement = sum(valid_rmsd) / len(valid_rmsd) if valid_rmsd else 0.0
                    avg_tmscore_improvement = sum(valid_tm) / len(valid_tm) if valid_tm else 0.0
                    improved_count = sum(1 for x in rows if x['structure_improved'])
                    
                    f.write(f"\nAvg RMSD Improvement: {avg_rmsd_improvement:+.3f}Å\n")
                    f.write(f"Avg TM-score Improvement: {avg_tmscore_improvement:+.3f}\n")
                    f.write(f"Structures Improved: {improved_count}/{len(rows)} ({improved_count/len(rows)*100:.1f}%)\n\n")
        
        print(f"📊 Summary table saved → {summary_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to write summary: {e}")

if __name__ == "__main__":
    main()
