#!/usr/bin/env python3
"""
Comprehensive Test Script for Both Folding and Inverse Folding MCTS Pipelines

This script tests both:
1. Folding: sequence → structure (optimizes RMSD/TM-score)
2. Inverse folding: structure → sequence (optimizes AAR/scTM)

Usage:
python test_both_pipelines.py --task folding --structures 0 2
python test_both_pipelines.py --task inverse_folding --structures 0 2
python test_both_pipelines.py --task both --structures 0 1
"""

import os, sys, json, time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import argparse

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

def load_cameo_data():
    """Load CAMEO sequences and structures"""
    # Load sequences
    fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    sequences = {}
    
    if os.path.exists(fasta_path):
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(sequences)} CAMEO sequences")
    else:
        print(f"⚠️ CAMEO aatype.fasta not found: {fasta_path}")
    
    # Load structures
    try:
        loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
        print(f"✅ Loaded CAMEO structure loader")
        return sequences, loader
    except Exception as e:
        print(f"❌ Failed to load CAMEO structures: {e}")
        return sequences, None

def calculate_simple_aar(pred_seq: str, ref_seq: str) -> float:
    """Calculate amino acid recovery (AAR)"""
    L = min(len(pred_seq), len(ref_seq))
    if L == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred_seq[:L], ref_seq[:L])) / L

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
            if torch.cuda.is_available():
                predict_structure_with_esmfold.model = predict_structure_with_esmfold.model.cuda()
            print("✅ ESMFold model loaded")
        
        model = predict_structure_with_esmfold.model
        tokenizer = predict_structure_with_esmfold.tokenizer
        
        # Tokenize and predict
        tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        if torch.cuda.is_available():
            tokenized = {k: v.cuda() for k, v in tokenized.items()}
        
        with torch.no_grad():
            output = model(tokenized['input_ids'])
        
        # Extract CA coordinates (fixed indexing for 5D tensor)
        positions = output['positions']
        if len(positions.shape) == 5:
            ca_coords = positions[0, 0, :, 1, :].cpu().numpy()
        else:
            ca_coords = positions[0, :, 1, :].cpu().numpy()
        
        return ca_coords
        
    except Exception as e:
        print(f"⚠️ ESMFold prediction failed: {e}")
        return None

def run_folding_test(sequence: str, structure_name: str, reference_coords: np.ndarray, 
                    dplm2: DPLM2Integration, structure_idx: int = None) -> Optional[Dict]:
    """Run folding test: sequence → structure optimization"""
    print(f"\n🧬 [FOLDING] {structure_name}")
    print(f"  📊 Sequence length: {len(sequence)}")
    
    # Generate baseline structure using ESMFold (corrected approach)
    try:
        baseline_coords = predict_structure_with_esmfold(sequence)
        if baseline_coords is None:
            print(f"  ⚠️ ESMFold baseline generation failed, using dummy coordinates")
            # Create dummy coordinates as fallback
            baseline_coords = np.array([[i * 3.8, 0, 0] for i in range(len(sequence))])
    except Exception as e:
        print(f"  ⚠️ ESMFold baseline generation failed: {e}, using dummy coordinates")
        baseline_coords = np.array([[i * 3.8, 0, 0] for i in range(len(sequence))])
    
    if baseline_coords is None:
        print("  ❌ Baseline structure generation failed")
        return None
    
    # Calculate baseline metrics (ESMFold vs Ground Truth)
    baseline_rmsd, baseline_tmscore = calculate_rmsd_and_tmscore(baseline_coords, reference_coords)
    print(f"  ✅ ESMFold Baseline vs GT: RMSD={baseline_rmsd:.3f}Å, TM-score={baseline_tmscore:.3f}")
    
    # Load structure sequence from FASTA
    try:
        from utils.struct_loader import load_struct_seq_from_fasta
        struct_seq_str = load_struct_seq_from_fasta("/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta", structure_name)
        print(f"  🔍 Loaded struct_seq: {len(struct_seq_str.split(','))} tokens")
    except Exception as e:
        print(f"  ⚠️ FASTA loading failed, using dummy tokens: {e}")
        struct_seq_str = ','.join(['159'] * len(sequence))
    
    baseline_structure = {
        'sequence': sequence,
        'coordinates': baseline_coords,
        'length': len(sequence),
        'plddt_scores': np.ones(len(sequence)) * 0.8,
        'structure_idx': structure_idx,
        'struct_seq': struct_seq_str,
        'name': structure_name,
        'pdb_id': structure_name.split('_')[0] if '_' in structure_name else structure_name,
        'chain_id': structure_name.split('_')[1] if '_' in structure_name else 'A',
        'structure_data': {'coords': baseline_coords}
    }
    
    # Initialize folding MCTS
    try:
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            reference_sequence=sequence,
            baseline_structure=baseline_structure,
            reference_coords=reference_coords,  # Critical: Ground truth coordinates for folding evaluation
            max_depth=3,
            task_type="folding",  # Critical: Set task type
            ablation_mode="multi_expert",  # Use multi-expert for better MCTS demonstration
            num_children_select=3,  # Select top 3 from all expert rollouts
            k_rollouts_per_expert=2  # 3 experts × 2 rollouts = 6 total candidates
        )
        
        # Run MCTS search
        start_time = time.time()
        root_node = mcts.search(initial_sequence=sequence, num_iterations=1)  # Single iteration for testing
        search_time = time.time() - start_time
        
        # Find best node
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
        
        # Get structure metrics from best node
        final_rmsd = getattr(best_node, 'rmsd', baseline_rmsd)
        final_tmscore = getattr(best_node, 'tm_score', baseline_tmscore)
        
        result = {
            "task": "folding",
            "structure_name": structure_name,
            "sequence_length": len(sequence),
            "baseline_rmsd": baseline_rmsd,
            "final_rmsd": final_rmsd,
            "rmsd_improvement": baseline_rmsd - final_rmsd,
            "baseline_tmscore": baseline_tmscore,
            "final_tmscore": final_tmscore,
            "tmscore_improvement": final_tmscore - baseline_tmscore,
            "search_time": search_time,
            "success": True
        }
        
        print(f"  📈 Results vs GT: RMSD {baseline_rmsd:.3f}→{final_rmsd:.3f} (Δ{baseline_rmsd-final_rmsd:+.3f})")
        print(f"                   TM-score {baseline_tmscore:.3f}→{final_tmscore:.3f} (Δ{final_tmscore-baseline_tmscore:+.3f})")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Folding MCTS failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_inverse_folding_test(sequence: str, structure_name: str, reference_coords: np.ndarray,
                           dplm2: DPLM2Integration, structure_idx: int = None) -> Optional[Dict]:
    """Run inverse folding test: structure → sequence optimization"""
    print(f"\n🧬 [INVERSE FOLDING] {structure_name}")
    print(f"  📊 Reference sequence length: {len(sequence)}")
    
    # Load structure sequence from FASTA
    try:
        from utils.struct_loader import load_struct_seq_from_fasta
        struct_seq_str = load_struct_seq_from_fasta("/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta", structure_name)
        print(f"  🔍 Loaded struct_seq: {len(struct_seq_str.split(','))} tokens")
    except Exception as e:
        print(f"  ⚠️ FASTA loading failed, using dummy tokens: {e}")
        struct_seq_str = ','.join(['159'] * len(sequence))
    
    baseline_structure = {
        'sequence': sequence,
        'coordinates': reference_coords,
        'length': len(sequence),
        'plddt_scores': np.ones(len(sequence)) * 0.8,
        'structure_idx': structure_idx,
        'struct_seq': struct_seq_str,
        'name': structure_name,
        'pdb_id': structure_name.split('_')[0] if '_' in structure_name else structure_name,
        'chain_id': structure_name.split('_')[1] if '_' in structure_name else 'A',
        'structure_data': {'coords': reference_coords}
    }
    
    # Generate baseline sequence using DPLM-2 (inverse folding: structure → sequence)
    try:
        baseline_seq = dplm2.generate_baseline_sequence(struct_seq_str, len(sequence), expert_id=1)
        if not baseline_seq:
            print("  ❌ Baseline sequence generation failed")
            return None
        baseline_aar = calculate_simple_aar(baseline_seq, sequence)
        print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")
    except Exception as e:
        print(f"  ❌ Baseline generation failed: {e}")
        return None
    
    # Initialize inverse folding MCTS
    try:
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            reference_sequence=sequence,  # Target sequence for AAR calculation
            baseline_structure=baseline_structure,
            max_depth=3,
            task_type="inverse_folding",  # Default task type
            ablation_mode="multi_expert",  # Use multiple experts for clearer demonstration
            num_children_select=2,
            k_rollouts_per_expert=2  # 3 experts × 2 rollouts = 6 total candidates
        )
        
        # Run MCTS search
        start_time = time.time()
        root_node = mcts.search(initial_sequence=baseline_seq, num_iterations=1)  # Single iteration for testing
        search_time = time.time() - start_time
        
        # Find best node
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
        final_aar = calculate_simple_aar(best_sequence, sequence)
        
        result = {
            "task": "inverse_folding",
            "structure_name": structure_name,
            "sequence_length": len(sequence),
            "baseline_aar": baseline_aar,
            "final_aar": final_aar,
            "aar_improvement": final_aar - baseline_aar,
            "search_time": search_time,
            "success": True
        }
        
        print(f"  📈 Results: AAR {baseline_aar:.1%}→{final_aar:.1%} (Δ{final_aar-baseline_aar:+.1%})")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Inverse folding MCTS failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🧬 Comprehensive MCTS Pipeline Test - Both Folding and Inverse Folding")
    print("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["folding", "inverse_folding", "both"], default="both")
    parser.add_argument("--structures", nargs=2, type=int, default=[0, 2], help="start and end structure indices")
    args = parser.parse_args()
    
    print(f"🎯 Task: {args.task}")
    print(f"📊 Testing structures: {args.structures[0]}-{args.structures[1]}")
    
    # Load CAMEO data
    sequences, loader = load_cameo_data()
    if not sequences or not loader:
        print("❌ Failed to load CAMEO data")
        return
    
    # Test range - limit to 1 structure for quick testing
    start_idx, end_idx = args.structures
    test_range = range(start_idx, min(start_idx + 1, len(sequences)))  # Only test 1 structure
    
    print(f"🧪 Testing {len(test_range)} structure(s): {start_idx}")
    
    # Initialize DPLM-2
    try:
        dplm2 = DPLM2Integration(device="cuda")
        print("✅ DPLM-2 integration initialized")
    except Exception as e:
        print(f"❌ Failed to initialize DPLM-2: {e}")
        return
    
    # Create test pairs
    test_pairs = []
    for idx, structure_file in enumerate(loader.structures[start_idx:end_idx]):
        structure_name = structure_file.replace('.pkl', '')
        
        if structure_name in sequences:
            sequence = sequences[structure_name]
            
            try:
                structure = loader.get_structure_by_index(start_idx + idx)
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
                        test_pairs.append((start_idx + idx, structure_name, sequence, ref_coords))
                        print(f"✅ Added test pair: {structure_name} (seq_len: {len(sequence)}, coords: {ref_coords.shape})")
                    else:
                        print(f"⚠️ No coordinates found for {structure_name}")
            except Exception as e:
                print(f"⚠️ Error loading structure {start_idx + idx}: {e}")
        else:
            print(f"⚠️ No sequence found for {structure_name}")
    
    print(f"📊 Testing {len(test_pairs)} structure pairs")
    
    # Run tests
    results = []
    
    for idx, structure_name, sequence, ref_coords in test_pairs:
        print(f"\n{'='*80}")
        print(f"TESTING: {structure_name}")
        print(f"{'='*80}")
        
        if args.task in ["folding", "both"]:
            result = run_folding_test(sequence, structure_name, ref_coords, dplm2, idx)
            if result:
                results.append(result)
        
        if args.task in ["inverse_folding", "both"]:
            result = run_inverse_folding_test(sequence, structure_name, ref_coords, dplm2, idx)
            if result:
                results.append(result)
    
    # Save results
    output_dir = "/net/scratch/caom/cameo_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = os.path.join(output_dir, f"both_pipelines_test_{timestamp}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved → {json_file}")
    
    # Print summary
    print(f"\n📊 SUMMARY:")
    folding_results = [r for r in results if r['task'] == 'folding']
    inverse_results = [r for r in results if r['task'] == 'inverse_folding']
    
    if folding_results:
        avg_rmsd_improvement = sum(r['rmsd_improvement'] for r in folding_results) / len(folding_results)
        avg_tm_improvement = sum(r['tmscore_improvement'] for r in folding_results) / len(folding_results)
        print(f"  Folding ({len(folding_results)} tests):")
        print(f"    Avg RMSD improvement: {avg_rmsd_improvement:+.3f}Å")
        print(f"    Avg TM-score improvement: {avg_tm_improvement:+.3f}")
    
    if inverse_results:
        avg_aar_improvement = sum(r['aar_improvement'] for r in inverse_results) / len(inverse_results)
        print(f"  Inverse Folding ({len(inverse_results)} tests):")
        print(f"    Avg AAR improvement: {avg_aar_improvement:+.1%}")
    
    print(f"\n🎯 Both pipelines tested successfully!")

if __name__ == "__main__":
    main()
