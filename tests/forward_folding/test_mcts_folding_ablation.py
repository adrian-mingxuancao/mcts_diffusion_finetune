#!/usr/bin/env python3
"""
MCTS Folding Ablation Study

This script tests the MCTS folding pipeline with:
- ESMFold baseline for comparison
- MCTS optimization using 3 DPLM experts (650M, 150M, 3B)
- Proper structure token to coordinate conversion
- RMSD and TM-score evaluation

Usage:
python test_mcts_folding_ablation.py 0 1 --mode single_expert --single_expert_id 1
python test_mcts_folding_ablation.py 0 1 --mode multi_expert
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
    from utils.cameo_data_loader import CAMEODataLoader
except ImportError:
    class CAMEODataLoader:
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



def load_cameo_sequences() -> Dict[str, str]:
    # Load CAMEO sequences from FASTA file (same approach as test_both_pipelines.py)
    fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    sequences = {}
    
    try:
        from Bio import SeqIO
        with open(fasta_path, 'r') as f:
            for record in SeqIO.parse(f, 'fasta'):
                sequences[record.id] = str(record.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(sequences)} CAMEO sequences from {fasta_path}")
    except Exception as e:
        print(f"⚠️ CAMEO sequence loading failed: {e}")
        # Try alternative path
        alt_path = "/home/caom/AID3/dplm/data-bin/cameo2022/preprocessed/aatype.fasta"
        try:
            with open(alt_path, 'r') as f:
                for record in SeqIO.parse(f, 'fasta'):
                    sequences[record.id] = str(record.seq).replace(" ", "").upper()
            print(f"✅ Loaded {len(sequences)} CAMEO sequences from {alt_path}")
        except Exception as e2:
            print(f"⚠️ Alternative CAMEO path also failed: {e2}")
            sequences = {}
    
    return sequences

def calculate_rmsd_and_tmscore(predicted_coords: np.ndarray, reference_coords: np.ndarray) -> Tuple[float, float]:
    """Calculate RMSD and TM-score between predicted and reference structures"""
    try:
        pred_coords = np.array(predicted_coords)
        ref_coords = np.array(reference_coords)
        
        # Handle length mismatches
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        if len(pred_coords) == 0:
            return float('inf'), 0.0
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords) ** 2, axis=1)))
        
        # Calculate TM-score
        L_target = len(ref_coords)
        d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
        
        distances = np.sqrt(np.sum((pred_coords - ref_coords) ** 2, axis=1))
        tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
        
        return rmsd, tm_score
        
    except Exception as e:
        print(f"⚠️ RMSD/TM-score calculation failed: {e}")
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

def generate_baseline_structure(sequence: str, dplm2: DPLM2Integration, structure_idx: int = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Generate baseline structure using ESMFold (consistent baseline for comparison)"""
    try:
        print(f"🎯 Generating ESMFold baseline structure for sequence length {len(sequence)}")
        
        # Use ESMFold as primary baseline for consistent comparison
        baseline_coords, baseline_plddt = predict_structure_with_esmfold(sequence)
        if baseline_coords is not None:
            print(f"✅ ESMFold baseline generation successful: {baseline_coords.shape}")
            return baseline_coords, baseline_plddt
        
        # Fallback to DPLM-2 if ESMFold fails
        print(f"🔄 ESMFold failed, trying DPLM-2 150M as fallback...")
        try:
            structure_coords = dplm2.generate_structure_from_sequence(
                sequence=sequence,
                expert_id=1,  # 150M model
                temperature=1.0
            )
            
            if structure_coords is not None:
                print(f"✅ DPLM-2 fallback structure generation successful")
                return structure_coords, None  # No pLDDT from DPLM-2 fallback
                
        except Exception as dplm2_e:
            print(f"⚠️ DPLM-2 fallback also failed: {dplm2_e}")
        
        print(f"❌ All baseline structure generation methods failed")
        return None, None
        
    except Exception as e:
        print(f"❌ Baseline structure generation failed: {e}")
        return None, None

def run_folding_ablation(sequence: str, structure_name: str, reference_coords: np.ndarray,
                        dplm2: DPLM2Integration, ablation_mode: str, single_expert_id: int = None,
                        structure_idx: int = None) -> Optional[Dict]:
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
    
    # Generate baseline structure
    baseline_coords, baseline_plddt = generate_baseline_structure(sequence, dplm2, structure_idx)
    if baseline_coords is None:
        print("  ❌ Baseline structure generation failed")
        return None
    
    # Calculate baseline metrics
    baseline_rmsd, baseline_tmscore = calculate_rmsd_and_tmscore(baseline_coords, reference_coords)
    print(f"  ✅ Baseline: RMSD={baseline_rmsd:.3f}Å, TM-score={baseline_tmscore:.3f}")
    
    # Initialize MCTS with ablation configuration
    # Load structure sequence from FASTA to match DPLM-2 training format
    try:
        from utils.struct_loader import load_struct_seq_from_fasta
        struct_seq_str = load_struct_seq_from_fasta("/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta", structure_name)
        print(f"  🔍 Loaded struct_seq: {len(struct_seq_str.split(','))} tokens")
    except Exception as e:
        print(f"  ⚠️ FASTA loading failed, using dummy tokens: {e}")
        # Fallback to dummy structure tokens
        struct_seq_str = ','.join(['159'] * len(sequence))
    
    # Convert ESMFold coordinates to REAL DPLM structure tokens for baseline
    converted_struct_tokens = None
    if baseline_coords is not None:
        try:
            from byprot.models.utils import get_struct_tokenizer
            import torch
            
            print(f"  🔄 Converting ESMFold coordinates to REAL structure tokens...")
            struct_tokenizer = get_struct_tokenizer()
            
            # Convert CA coordinates to full atom37 format
            seq_len = len(sequence)
            full_coords = torch.zeros((1, seq_len, 37, 3), dtype=torch.float32)
            coords_tensor = torch.from_numpy(baseline_coords).float()
            
            # Place CA coordinates at atom index 1 (standard CA position)
            full_coords[0, :, 1, :] = coords_tensor
            
            # Create residue mask (all positions valid)
            res_mask = torch.ones((1, seq_len), dtype=torch.float32)  # Use float32, not bool
            
            # Create seq_length tensor (required by tokenizer)
            seq_length = torch.tensor([seq_len], dtype=torch.long)
            
            # Tokenize coordinates to get REAL structure tokens
            struct_tokens = struct_tokenizer.tokenize(full_coords, res_mask, seq_length)
            
            if struct_tokens is not None and isinstance(struct_tokens, torch.Tensor):
                struct_tokens = struct_tokens.squeeze(0)  # Remove batch dim
                token_list = [str(int(token.item())) for token in struct_tokens]
                struct_seq_str = ','.join(token_list)
                print(f"  ✅ Generated REAL baseline structure tokens: {len(token_list)} tokens")
            else:
                print(f"  ⚠️ Structure tokenizer returned None, using mask tokens")
                struct_seq_str = ','.join(['<mask_struct>'] * seq_len)
                struct_seq_str = f"<cls_struct>,{struct_seq_str},<eos_struct>"
                
        except Exception as e:
            print(f"  ⚠️ Structure tokenization failed: {e}, using mask tokens")
            struct_seq_str = ','.join(['<mask_struct>'] * len(sequence))
            struct_seq_str = f"<cls_struct>,{struct_seq_str},<eos_struct>"
    
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
        'max_depth': 4,
        'backup_rule': "max",
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
        try:
            # For folding tasks, use structure reward based on baseline RMSD/TM-score
            baseline_reward = mcts._compute_structure_reward(baseline_rmsd, baseline_tmscore)
            print(f"  📈 Baseline reward: {baseline_reward:.3f}")
        except Exception as e:
            print(f"  ⚠️ Baseline reward computation failed: {e}")
        
        # Run MCTS search
        num_iterations = 25
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
        if hasattr(best_node, 'rmsd') and hasattr(best_node, 'tm_score'):
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
        
        # Calculate improvements
        rmsd_improvement = baseline_rmsd - final_rmsd  # Lower RMSD is better
        tmscore_improvement = final_tmscore - baseline_tmscore  # Higher TM-score is better
        structure_improved = rmsd_improvement > 0 or tmscore_improvement > 0
        
        # Store results (convert numpy types to Python types for JSON serialization)
        result = {
            'structure_name': structure_name,
            'sequence_length': len(sequence),
            'ablation_mode': ablation_mode,
            'single_expert_id': single_expert_id,
            'baseline_rmsd': float(baseline_rmsd),
            'baseline_tmscore': float(baseline_tmscore),
            'baseline_reward': float(baseline_reward),
            'final_rmsd': float(final_rmsd),
            'final_tmscore': float(final_tmscore),
            'final_reward': float(best_reward),
            'rmsd_improvement': float(rmsd_improvement),
            'tmscore_improvement': float(tmscore_improvement),
            'reward_improvement': float(best_reward - baseline_reward),
            'improved': bool(rmsd_improvement > 0),  # Convert numpy bool_ to Python bool
            'search_time': float(search_time),
            'total_time': float(time.time() - start_time),
            'num_iterations': int(num_iterations),
            'best_depth': int(getattr(best_node, "depth", 0)),
            'tree_size': int(len([n for n in [root_node] + getattr(root_node, 'children', []) if hasattr(n, 'sequence')]))
        }
        
        print("  ─────────────────────────────────────────────────────────")
        print(f"  Summary [{ablation_mode}{'' if single_expert_id is None else f'/{single_expert_id}'}] {structure_name}")
        print(f"    RMSD    : {baseline_rmsd:.3f}Å → {final_rmsd:.3f}Å (Δ {rmsd_improvement:+.3f}Å)")
        print(f"    TM-score: {baseline_tmscore:.3f} → {final_tmscore:.3f} (Δ {tmscore_improvement:+.3f})")
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
    print("🧬 MCTS Folding Ablation Study - CAMEO 2022 Evaluation")
    print("=" * 60)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs='?', type=int, default=0, help="start index (inclusive)")
    parser.add_argument("end", nargs='?', type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--mode", choices=["random_no_expert", "single_expert", "multi_expert", "all"], default="all")
    parser.add_argument("--single_expert_id", type=int, default=None, help="single expert id (0/1/2)")
    args = parser.parse_args()
    
    start_idx = args.start
    end_idx = args.end
    print(f"🎯 Structure range: {start_idx}-{end_idx if end_idx is not None else 'end'} | Mode: {args.mode}")
    
    # Load CAMEO data
    cameo_sequences = load_cameo_sequences()
    if not cameo_sequences:
        print("❌ Failed to load CAMEO sequences")
        return
    
    # Load reference structures
    try:
        loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
        print(f"✅ Loaded CAMEO structure loader")
    except Exception as e:
        print(f"❌ Failed to load CAMEO structures: {e}")
        return
    
    # Initialize DPLM-2 with multi-expert support (650M, 150M, 3B)
    try:
        dplm2 = DPLM2Integration(device="cuda")  # Use consolidated integration
        print("✅ DPLM-2 multi-expert integration initialized (supports 650M, 150M, 3B)")
    except Exception as e:
        print(f"❌ Failed to initialize DPLM-2: {e}")
        return
    
    # Create test pairs (sequence + reference structure)
    test_pairs = []
    for idx, structure_file in enumerate(loader.structures):
        structure_name = structure_file.replace('.pkl', '')
        
        # Get sequence from aatype.fasta
        if structure_name in cameo_sequences:
            sequence = cameo_sequences[structure_name]
            
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
                        print(f"✅ Added test pair {idx}: {structure_name} (seq_len: {len(sequence)}, coords: {ref_coords.shape})")
                    else:
                        print(f"⚠️ No coordinates found for {structure_name}")
                else:
                    print(f"⚠️ Could not load structure {idx}: {structure_file}")
            except Exception as e:
                print(f"⚠️ Error loading structure {idx}: {e}")
        else:
            print(f"⚠️ No sequence found for {structure_name}")
    
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
                                        "random_no_expert", structure_idx=idx)
            if result:
                results.append(result)
                
        elif args.mode == "single_expert":
            expert_ids = [args.single_expert_id] if args.single_expert_id is not None else [0, 1, 2]
            for expert_id in expert_ids:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            "single_expert", expert_id, structure_idx=idx)
                if result:
                    results.append(result)
                    
        elif args.mode == "multi_expert":
            result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                        "multi_expert", structure_idx=idx)
            if result:
                results.append(result)
                
        else:  # args.mode == "all"
            # Run all ablation modes
            for mode in ["random_no_expert", "multi_expert"]:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            mode, structure_idx=idx)
                if result:
                    results.append(result)
            
            # Run single expert modes
            for expert_id in [0, 1, 2]:
                result = run_folding_ablation(sequence, structure_name, ref_coords, dplm2,
                                            "single_expert", expert_id, structure_idx=idx)
                if result:
                    results.append(result)
    
    # Save results
    output_dir = "/net/scratch/caom/cameo_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"mcts_folding_ablation_results_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved ablation results → {json_file}")
    
    # Create summary table
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["ablation_mode"]].append(r)
    
    summary_file = os.path.join(output_dir, f"mcts_folding_ablation_summary_{timestamp}.txt")
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
                    f.write(f"{name:<20} {r['sequence_length']:<5} {r['baseline_rmsd']:<10.3f} "
                           f"{r['final_rmsd']:<11.3f} {r['rmsd_improvement']:<8.3f} "
                           f"{r['baseline_tmscore']:<8.3f} {r['final_tmscore']:<9.3f} "
                           f"{r['tmscore_improvement']:<8.3f} {str(r['structure_improved']):<8} "
                           f"{r['search_time']:<8.1f}\n")
                
                # Statistics
                if rows:
                    avg_rmsd_improvement = sum(x['rmsd_improvement'] for x in rows) / len(rows)
                    avg_tmscore_improvement = sum(x['tmscore_improvement'] for x in rows) / len(rows)
                    improved_count = sum(1 for x in rows if x['structure_improved'])
                    
                    f.write(f"\nAvg RMSD Improvement: {avg_rmsd_improvement:+.3f}Å\n")
                    f.write(f"Avg TM-score Improvement: {avg_tmscore_improvement:+.3f}\n")
                    f.write(f"Structures Improved: {improved_count}/{len(rows)} ({improved_count/len(rows)*100:.1f}%)\n\n")
        
        print(f"📊 Summary table saved → {summary_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to write summary: {e}")

if __name__ == "__main__":
    main()
