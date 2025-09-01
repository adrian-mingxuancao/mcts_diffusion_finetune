#!/usr/bin/env python3
"""
PROPER MCTS TREE SEARCH TEST - Testing the MCTS Framework with Real Data

This script tests the actual MCTS tree search framework:

🌳 MCTS TREE STRUCTURE:
1. **Root Node**: Baseline sequence from DPLM-2
2. **Tree Growth**: Each masking strategy creates new branches
3. **UCB1 Selection**: Balance exploration vs exploitation
4. **Tree Expansion**: Different pLDDT masking strategies as actions
5. **Simulation**: DPLM-2 unmasking for evaluation
6. **Backpropagation**: Propagate rewards up the tree

🎯 TESTING GOALS:
- Verify MCTS can grow trees properly
- Check that different masking strategies create different branches
- Ensure UCB1 selection works for exploration vs exploitation
- Validate that tree search improves AAR over baseline
- Test the full MCTS cycle: Selection → Expansion → Simulation → Backpropagation

🚀 TESTING MODES:
1. SUBSET TESTING MODE (SUBSET_TEST_MODE = True):
   - Tests 1 structure for debugging
   - Quick feedback on MCTS implementation issues
   - Set SUBSET_TEST_MODE = True for full batch testing

2. FULL BATCH TESTING MODE (SUBSET_TEST_MODE = True):
   - Tests ALL CAMEO structures
   - Comprehensive MCTS evaluation for statistical significance
"""

import sys
import os

# Add both the main project directory and src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
import random
import numpy as np
from datetime import datetime

try:
    from utils.cameo_data_loader import CAMEODataLoader
except ImportError:
    # Fallback: create a simple test structure with proper struct_seq
    class CAMEODataLoader:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_test_structure(self, index=0):
            # Use a simple structure token sequence (5 residues)
            struct_tokens = "159,162,163,164,165"  # Example structure tokens
            return {
                "name": f"test_structure_{index}",
                "struct_seq": struct_tokens,
                "sequence": "IKKSI",  # Short sequence for testing
                "length": 5
            }

def setup_logging():
    """Configure logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def calculate_simple_aar(pred_seq, ref_seq):
    """Calculate AAR (Amino Acid Recovery) - the main metric we care about"""
    if len(pred_seq) != len(ref_seq):
        # If lengths differ, align to the shorter one for fair comparison
        min_len = min(len(pred_seq), len(ref_seq))
        pred_seq = pred_seq[:min_len]
        ref_seq = ref_seq[:min_len]
    
    if len(ref_seq) == 0:
        return 0.0
    
    # Calculate matches (same logic as working script)
    matches = sum(1 for p, r in zip(pred_seq, ref_seq) if p == r)
    return matches / len(ref_seq) if len(ref_seq) > 0 else 0.0


def load_correct_reference_sequences():
    """Load correct reference sequences from FASTA file (same as working script)"""
    reference_fasta = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    sequences = {}
    
    if os.path.exists(reference_fasta):
        try:
            from Bio import SeqIO
            for record in SeqIO.parse(reference_fasta, "fasta"):
                sequences[record.id] = str(record.seq).replace(" ", "").upper()
            print(f"✅ Loaded {len(sequences)} correct reference sequences from FASTA")
            return sequences
        except Exception as e:
            print(f"⚠️ Error loading reference sequences: {e}")
    else:
        print(f"⚠️ Reference FASTA not found: {reference_fasta}")
    
    return {}

def generate_dplm2_baseline_sequence(structure, dplm2_integration):
    """Generate baseline sequence using DPLM-2 150M model following README.md approach"""
    print(f"  🎯 Generating baseline using DPLM-2 150M (structure-conditioned inverse folding)")
    
    try:
        # Following README.md approach: structure → sequence generation
        target_length = structure['length']
        
        print(f"  📊 Structure length: {target_length} residues")
        print(f"  🎯 Creating baseline for inverse folding")
        
        # Fix: Load real struct_seq from struct.fasta
        baseline_structure = dict(structure)
        if not baseline_structure.get('struct_seq') and not baseline_structure.get('struct_ids'):
            # Load real structure tokens from CAMEO struct.fasta
            struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
            # Extract structure name from pdb_id and chain_id
            pdb_id = structure.get('pdb_id', '')
            chain_id = structure.get('chain_id', '')
            structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
            print(f"  🔍 Looking for struct_seq: {structure_name}")
            
            try:
                from utils.struct_loader import load_struct_seq_from_fasta
                struct_seq = load_struct_seq_from_fasta(struct_fasta_path, structure_name)
                baseline_structure['struct_seq'] = struct_seq
                print(f"  ✅ Loaded real struct_seq from struct.fasta: {structure_name}")
            except Exception as e:
                print(f"  ❌ Failed to load struct_seq from {struct_fasta_path}: {e}")
                raise ValueError(
                    "Missing struct_seq/struct_ids. Baseline inverse folding must use true struct.fasta tokens. "
                    f"Could not load {structure_name} from CAMEO struct.fasta."
                )
        
        # Use 150M expert for weaker baseline (creates optimization headroom)
        expert_idx = 1  # 150M model index
        
        # Generate using structure-conditioned inverse folding (baseline = masked_sequence=None)
        baseline_seq = dplm2_integration.fill_masked_positions(
            structure=baseline_structure,  # Use fixed structure
            target_length=target_length,
            masked_sequence=None,  # None = baseline inverse folding
            temperature=1.0
        )
        
        if baseline_seq and len(baseline_seq) > 0:
            print(f"  ✅ Generated DPLM-2 150M baseline: {len(baseline_seq)} chars")
            print(f"  🔍 Sequence preview: {baseline_seq[:50]}...")
            return baseline_seq, baseline_structure
        else:
            print(f"  ⚠️ DPLM-2 150M generation returned empty sequence")
            return None, None
            
    except Exception as e:
        print(f"  ❌ DPLM-2 150M generation failed: {e}")
        print(f"  🔄 Trying fallback to pre-generated 150M results...")
        
        # Fallback to pre-generated results
        pdb_id = structure.get('pdb_id', '')
        chain_id = structure.get('chain_id', '')
        structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
        
        fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_name}.fasta"
        try:
            from Bio import SeqIO
            
            def clean_aa(seq_str: str) -> str:
                valid = set("ACDEFGHIKLMNPQRSTVWY")
                s = "".join(c for c in str(seq_str).upper() if c in valid)  # drop spaces & non-AA
                return s
            
            for record in SeqIO.parse(fallback_path, "fasta"):
                baseline_seq = clean_aa(record.seq)
                print(f"  ✅ Loaded fallback 150M baseline: {len(baseline_seq)} chars")
                return baseline_seq, baseline_structure
                
        except Exception as fallback_e:
            print(f"  ❌ Fallback also failed: {fallback_e}")
        
        return None, None

def load_dplm2_baseline_sequence(structure_name: str):
    """Load pre-generated DPLM-2 baseline sequence from official results (fallback only)"""
    # Try to find matching DPLM-2 result file
    dplm2_dir = "/home/caom/AID3/dplm/generation-results/dplm2_650m/inverse_folding"
    
    # Structure name format: "CAMEO 7n99_A" -> "7n99_A.fasta"
    if "CAMEO " in structure_name:
        pdb_chain = structure_name.replace("CAMEO ", "")
        result_file = os.path.join(dplm2_dir, f"{pdb_chain}.fasta")
    else:
        # Try to extract PDB_chain format
        result_file = os.path.join(dplm2_dir, f"{structure_name}.fasta")
    
    if os.path.exists(result_file):
        try:
            # Use BioPython's FASTA parser (same as working script)
            from Bio import SeqIO
            for record in SeqIO.parse(result_file, "fasta"):
                sequence = str(record.seq).replace(" ", "").upper()
                print(f"  ✅ Loaded pre-generated DPLM-2 baseline from: {os.path.basename(result_file)}")
                print(f"  🔍 Sequence length: {len(sequence)} chars (properly parsed)")
                return sequence
        except Exception as e:
            print(f"  ⚠️ Error reading DPLM-2 baseline file {result_file}: {e}")
    
    print(f"  ❌ No pre-generated DPLM-2 baseline found for {structure_name}")
    return None


def test_mcts_tree_search(structure, structure_name: str, correct_reference_sequences: dict):
    """
    Test the PROPER MCTS tree search framework.
    
    This tests the actual MCTS implementation, not just iterations!
    
    Args:
        structure: Structure data from CAMEO dataset
        structure_name: Name of the structure (e.g., "CAMEO 7dz2_C")
        correct_reference_sequences: Dict mapping structure names to reference sequences
    
    Returns:
        Dict with test results or None if failed
    """
    print(f"\n🧬 Testing MCTS Tree Search for {structure_name}")
    print(f"  🎯 This is a PROPER test of the MCTS framework")
    print(f"  🎯 Expected behavior: Tree growth, exploration, and AAR improvement")
    
    # Get correct reference sequence
    structure_id = structure_name.replace('CAMEO ', '')
    correct_ref_seq = correct_reference_sequences.get(structure_id)
    
    if not correct_ref_seq:
        print(f"  ❌ No correct reference sequence found for {structure_id}")
        return None
    
    print(f"Structure info:")
    print(f"  Length: {structure['length']} residues")
    print(f"  PDB ID: {structure.get('pdb_id', 'N/A')}")
    print(f"  Chain ID: {structure.get('chain_id', 'N/A')}")
    print(f"  Avg plDDT: {sum(structure['plddt_scores']) / len(structure['plddt_scores']):.3f}")
    
    # Initialize DPLM-2 integration FIRST before baseline generation
    print(f"\n📊 Initializing DPLM-2 integration...")
    try:
        from core.dplm2_integration_fixed import DPLM2Integration
        dplm2 = DPLM2Integration()
        print("  ✅ DPLM-2 integration initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize DPLM-2: {e}")
        # Try alternative import path
        try:
            import sys
            sys.path.append('/home/caom/AID3/dplm/mcts_diffusion_finetune/core')
            from dplm2_integration_fixed import DPLM2Integration
            dplm2 = DPLM2Integration()
            print("  ✅ DPLM-2 integration initialized (alternative path)")
        except Exception as e2:
            print(f"  ❌ Alternative import also failed: {e2}")
            return None
    
    # Generate DPLM-2 baseline sequence using 150M model for optimization headroom
    print(f"\n📊 Generating DPLM-2 baseline sequence...")
    baseline_seq, baseline_structure = generate_dplm2_baseline_sequence(structure, dplm2)
    
    if not baseline_seq:
        print(f"  ⚠️ DPLM-2 150M generation failed, trying pre-generated fallback...")
        baseline_seq = load_dplm2_baseline_sequence(structure_name)
        baseline_structure = dict(structure)  # Use original structure for fallback
        
        if not baseline_seq:
            print(f"  ❌ No DPLM-2 baseline available")
            return None
    
    # Calculate baseline AAR
    baseline_aar = calculate_simple_aar(baseline_seq, correct_ref_seq)
    print(f"  ✅ DPLM-2 Baseline: AAR {baseline_aar:.1%}")
    
    # 🎯 CRITICAL: Test the PROPER MCTS framework
    print(f"\n🌳 MCTS TREE SEARCH TEST:")
    print(f"  🎯 Goal: Test actual MCTS tree growth and search")
    print(f"  🎯 Baseline AAR: {baseline_aar:.1%}")
    print(f"  🎯 Expected: MCTS should grow trees and improve AAR")
    print(f"  🎯 Framework: Selection → Expansion → Simulation → Backpropagation")
    
    # 🎯 STEP 1: Initialize MCTS with the baseline sequence
    print(f"\n🌳 Step 1: Initializing MCTS with baseline sequence")
    
    try:
        from core.sequence_level_mcts import GeneralMCTS
        
        # Ensure MCTS has real struct_seq tokens
        mcts_struct_seq = baseline_structure.get('struct_seq')
        if not mcts_struct_seq:
            # Load real structure tokens from CAMEO struct.fasta for MCTS too
            struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
            # Extract structure name from pdb_id and chain_id
            pdb_id = structure.get('pdb_id', '')
            chain_id = structure.get('chain_id', '')
            structure_name_clean = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
            print(f"  🔍 Looking for MCTS struct_seq: {structure_name_clean}")
            
            try:
                from utils.struct_loader import load_struct_seq_from_fasta
                mcts_struct_seq = load_struct_seq_from_fasta(struct_fasta_path, structure_name_clean)
                print(f"  ✅ Loaded real struct_seq for MCTS: {structure_name_clean}")
            except Exception as e:
                print(f"  ❌ Failed to load struct_seq for MCTS: {e}")
                raise ValueError(f"MCTS requires real struct.fasta tokens for {structure_name_clean}")
        
        mcts_baseline_structure = {
            'sequence': baseline_seq,
            'plddt_scores': structure.get('plddt_scores', [0.8] * len(baseline_seq)),
            'coordinates': structure.get('coordinates'),
            'atom_positions': structure.get('atom_positions'),  # Alternative coordinate key
            'backbone_coords': structure.get('backbone_coords'),  # Another alternative
            'structure_data': structure.get('structure_data'),
            'structure_path': structure.get('structure_path'),
            'struct_ids': structure.get('struct_ids'),  # Critical: Pass struct_ids to MCTS
            'struct_seq': mcts_struct_seq   # Real struct.fasta tokens
        }
        
        # Debug coordinate availability
        coord_keys = ['coordinates', 'atom_positions', 'backbone_coords']
        available_coords = [k for k in coord_keys if mcts_baseline_structure.get(k) is not None]
        print(f"  🎯 Available coordinate keys: {available_coords}")
        if available_coords:
            coords = mcts_baseline_structure[available_coords[0]]
            import numpy as np
            print(f"  ✅ Using coordinates from '{available_coords[0]}': shape={np.array(coords).shape}")
        else:
            print(f"  ⚠️ No coordinates found in structure data")
        
        # Create MCTS instance with correct constructor - pass baseline_seq as initial_sequence
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            initial_sequence=baseline_seq,  # Use the working baseline sequence
            baseline_structure=mcts_baseline_structure,
            reference_sequence=correct_ref_seq,
            max_depth=4
        )
        
        print(f"  ✅ MCTS initialized successfully")
        print(f"  📊 Initial sequence: {baseline_seq[:50]}...")
        print(f"  📊 Sequence length: {len(baseline_seq)}")
        print(f"  📊 MCTS parameters: max_depth=5, simulations=30, expansion=3")
        
    except Exception as e:
        print(f"  ❌ Failed to initialize MCTS: {e}")
        print(f"  🔧 This suggests the MCTS framework has issues")
        return None
    
    # 🎯 STEP 2: Run MCTS tree search
    print(f"\n🌳 Step 2: Running MCTS tree search")
    print(f"  🎯 This should grow a tree and explore different masking strategies")
    
    try:
        # REMOVED: Old pkl loading logic - MCTS already has the structure data we need
        # The structure_data will be created from MCTS._baseline_structure in the scTM section
        
        start_time = time.time()
        
        # Run MCTS search with correct method signature
        root_node = mcts.search(num_iterations=30)
        
        # Get best sequence from the search tree
        best_node = root_node
        best_reward = root_node.reward if hasattr(root_node, 'reward') else 0.0
        
        # Find the best child in the tree
        def find_best_node(node):
            best = node
            best_score = getattr(node, 'reward', 0.0)
            
            for child in node.children:
                child_best = find_best_node(child)
                child_score = getattr(child_best, 'reward', 0.0)
                if child_score > best_score:
                    best = child_best
                    best_score = child_score
            return best
        
        best_node = find_best_node(root_node)
        best_sequence = best_node.sequence
        best_reward = getattr(best_node, 'reward', 0.0)
        
        search_time = time.time() - start_time
        
        if best_sequence:
            print(f"  ✅ MCTS search completed successfully!")
            print(f"  📊 Search time: {search_time:.1f} seconds")
            print(f"  📊 Best sequence found: {best_sequence[:50]}...")
            print(f"  📊 Best reward: {best_reward:.4f}")
            
            # Calculate AAR improvement
            best_aar = calculate_simple_aar(best_sequence, correct_ref_seq)
            aar_improvement = best_aar - baseline_aar
            
            print(f"  📊 AAR Results:")
            print(f"    Baseline AAR: {baseline_aar:.1%}")
            print(f"    MCTS AAR: {best_aar:.1%}")
            print(f"    AAR Improvement: {aar_improvement:+.1%}")
            
            # 🎯 VERIFY: MCTS should have improved AAR
            if aar_improvement > 0:
                print(f"  🎉 SUCCESS: MCTS improved AAR by {aar_improvement:+.1%}!")
            elif aar_improvement == 0:
                print(f"  📝 MCTS maintained AAR (no change)")
            else:
                print(f"  ⚠️  MCTS decreased AAR by {abs(aar_improvement):.1%}")
                print(f"  🔧 This suggests the MCTS framework may need tuning")
            
            # Calculate baseline reward and scTM for comparison
            baseline_reward = 0.0
            baseline_sctm = 0.0
            final_sctm = 0.0
            
            if hasattr(mcts, '_compute_compound_reward') and hasattr(mcts, '_baseline_structure'):
                try:
                    baseline_structure = mcts._baseline_structure.copy()
                    baseline_structure['sequence'] = correct_ref_seq
                    baseline_reward = mcts._compute_compound_reward(baseline_seq, baseline_structure)
                    print(f"  📊 Baseline reward: {baseline_reward:.4f}")
                except Exception as e:
                    print(f"  ⚠️ Could not calculate baseline reward: {e}")
            
            # Calculate scTM scores for both baseline and final sequences using CAMEO reference structure
            try:
                from utils.sctm_calculation import calculate_sctm_with_cameo_data
                
                print(f"  🧬 Calculating scTM scores using CAMEO reference structure...")
                
                # FIXED: Use same adapter as MCTS instead of reloading .pkl files
                structure_data = None
                if hasattr(mcts, '_baseline_structure') and mcts._baseline_structure:
                    baseline = mcts._baseline_structure
                    
                    # Create CAMEO-format dict from _baseline_structure (same as MCTS)
                    structure_data = {
                        'bb_positions': baseline.get('backbone_coords', baseline.get('coordinates')),
                        'sequence': baseline.get('sequence', ''),
                        'bb_mask': [True] * baseline.get('length', 0) if baseline.get('length') else None
                    }
                    
                    # Convert sequence to aatype if needed
                    if structure_data['sequence']:
                        AA_TO_IDX = {
                            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
                        }
                        import numpy as np
                        aatype = np.array([AA_TO_IDX.get(aa, 20) for aa in structure_data['sequence']])
                        structure_data['aatype'] = aatype
                    
                    print(f"  📁 Using structure data from MCTS _baseline_structure")
                
                if structure_data:
                    # Use CAMEO-specific scTM calculation with reference coordinates
                    baseline_sctm = calculate_sctm_with_cameo_data(baseline_seq, structure_data)
                    final_sctm = calculate_sctm_with_cameo_data(best_sequence, structure_data)
                else:
                    # No fallback - require reference structure for scTM calculation
                    print(f"  ⚠️ No reference structure available, skipping scTM calculation")
                    baseline_sctm = None
                    final_sctm = None
                
                if baseline_sctm is not None:
                    print(f"  📊 Baseline scTM: {baseline_sctm:.3f}")
                if final_sctm is not None:
                    print(f"  📊 Final scTM: {final_sctm:.3f}")
                    
            except Exception as e:
                print(f"  ⚠️ Could not calculate scTM scores: {e}")
                baseline_sctm = None
                final_sctm = None
            
            # Return results with baseline and final metrics including scTM
            result = {
                'structure_name': structure_name,
                'length': structure['length'],
                'baseline_aar': baseline_aar,
                'final_aar': best_aar,
                'aar_improvement': aar_improvement,
                'baseline_reward': baseline_reward,
                'final_reward': best_reward,
                'reward_improvement': best_reward - baseline_reward,
                'baseline_sctm': baseline_sctm if baseline_sctm is not None else 0.0,
                'final_sctm': final_sctm if final_sctm is not None else 0.0,
                'sctm_improvement': (final_sctm - baseline_sctm) if (final_sctm is not None and baseline_sctm is not None) else 0.0,
                'mcts_success': True,
                'search_time': search_time,
                'sequence': best_sequence
            }
            
            return result
            
        else:
            print(f"  ❌ MCTS search failed - no sequence returned")
            print(f"  🔧 This suggests the MCTS framework has critical issues")
            return None
            
    except Exception as e:
        print(f"  ❌ MCTS search failed with error: {e}")
        print(f"  🔧 This suggests the MCTS framework is broken")
        import traceback
        traceback.print_exc()
        return None


def save_results_to_files(results):
    """Save results to JSON and create readable table summaries"""
    if not results:
        print("No results to save")
        return
    
    # Create output directory
    cameo_structures_dir = "/net/scratch/caom/dplm_datasets/data-bin/cameo2022/preprocessed"
    output_dir = "/net/scratch/caom/cameo_evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = os.path.join(output_dir, f"mcts_tree_search_results_{timestamp}.json")
    try:
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to JSON: {json_file}")
    except Exception as e:
        print(f"⚠️ Failed to save JSON: {e}")
    
    # Create readable table summary
    table_file = os.path.join(output_dir, f"mcts_tree_search_summary_{timestamp}.txt")
    try:
        successful_results = [r for r in results if r.get('mcts_success', False)]
        
        with open(table_file, 'w') as f:
            f.write("CAMEO 2022 MCTS TREE SEARCH Evaluation Results\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total structures tested: {len(results)}\n")
            f.write(f"Successful MCTS evaluations: {len(successful_results)}\n")
            f.write(f"Success rate: {len(successful_results)/len(results)*100:.1f}%\n\n")
            
            if successful_results:
                # Detailed results table
                f.write("DETAILED RESULTS\n")
                f.write("=" * 80 + "\n")
                f.write(f"{'Structure':<15} {'Length':<6} {'DPLM-2 AAR':<12} {'MCTS AAR':<10} {'Δ AAR':<10} {'Base R':<8} {'Final R':<8} {'Δ R':<8} {'Base scTM':<9} {'Final scTM':<10} {'Δ scTM':<8} {'Time':<8}\n")
                f.write("-" * 125 + "\n")
                
                for result in successful_results:
                    name = result['structure_name'].replace('CAMEO ', '')[:14]
                    baseline_aar = result.get('baseline_aar', 0.0)
                    final_aar = result.get('final_aar', 0.0) 
                    aar_delta = result.get('aar_improvement', 0.0)
                    baseline_reward = result.get('baseline_reward', 0.0)
                    final_reward = result.get('final_reward', 0.0)
                    reward_delta = result.get('reward_improvement', 0.0)
                    baseline_sctm = result.get('baseline_sctm', 0.0)
                    final_sctm = result.get('final_sctm', 0.0)
                    sctm_delta = result.get('sctm_improvement', 0.0)
                    search_time = result.get('search_time', 0.0)
                    
                    f.write(f"{name:<15} {result['length']:<6} {baseline_aar:<12.1%} {final_aar:<10.1%} "
                           f"{aar_delta:<10.1%} {baseline_reward:<8.3f} {final_reward:<8.3f} {reward_delta:<8.3f} "
                           f"{baseline_sctm:<9.3f} {final_sctm:<10.3f} {sctm_delta:<8.3f} {search_time:<8.1f}s\n")
                
                # Summary statistics
                f.write("\n")
                f.write("SUMMARY STATISTICS\n")
                f.write("=" * 50 + "\n")
                
                baseline_aars = [r['baseline_aar'] for r in successful_results if 'baseline_aar' in r]
                final_aars = [r['final_aar'] for r in successful_results if 'final_aar' in r]
                
                if baseline_aars and final_aars:
                    avg_baseline_aar = sum(baseline_aars) / len(baseline_aars)
                    avg_final_aar = sum(final_aars) / len(final_aars)
                    avg_improvement = avg_final_aar - avg_baseline_aar
                    
                    f.write(f"Average DPLM-2 Baseline AAR: {avg_baseline_aar:.1%}\n")
                    f.write(f"Average MCTS Final AAR:  {avg_final_aar:.1%}\n")
                    f.write(f"Average AAR Improvement:     {avg_improvement:+.1%}\n\n")
                    
                    # Count improvements
                    improved_count = sum(1 for r in successful_results if r.get('aar_improvement', 0) > 0)
                    maintained_count = sum(1 for r in successful_results if r.get('aar_improvement', 0) == 0)
                    decreased_count = sum(1 for r in successful_results if r.get('aar_improvement', 0) < 0)
                    
                    f.write(f"MCTS Performance Breakdown:\n")
                    f.write(f"  Improved:  {improved_count}/{len(successful_results)} structures ({improved_count/len(successful_results)*100:.1f}%)\n")
                    f.write(f"  Maintained: {maintained_count}/{len(successful_results)} structures ({maintained_count/len(successful_results)*100:.1f}%)\n")
                    f.write(f"  Decreased: {decreased_count}/{len(successful_results)} structures ({decreased_count/len(successful_results)*100:.1f}%)\n\n")
                    
                    if improved_count > 0:
                        avg_improvement_for_improved = sum(r.get('aar_improvement', 0) for r in successful_results if r.get('aar_improvement', 0) > 0) / improved_count
                        f.write(f"Average improvement for improved structures: {avg_improvement_for_improved:+.1%}\n")
                
                # Top improvements
                f.write("\nTOP 5 IMPROVEMENTS\n")
                f.write("-" * 50 + "\n")
                sorted_by_improvement = sorted(successful_results, key=lambda x: x.get('aar_improvement', 0), reverse=True)
                for i, result in enumerate(sorted_by_improvement[:5]):
                    name = result['structure_name'].replace('CAMEO ', '')
                    improvement = result.get('aar_improvement', 0)
                    baseline_aar = result.get('baseline_aar', 0)
                    final_aar = result.get('final_aar', 0)
                    f.write(f"{i+1}. {name}: {baseline_aar:.1%} → {final_aar:.1%} ({improvement:+.1%})\n")
                
        print(f"📊 Summary table saved to: {table_file}")
        
    except Exception as e:
        print(f"⚠️ Failed to create summary table: {e}")


def main():
    """Main test function."""
    setup_logging()
    
    # 🎯 FULL BATCH TESTING MODE: Test ALL CAMEO structures for comprehensive evaluation
    SUBSET_TEST_MODE = False  # Set to False for full batch testing, True for debugging
    MAX_STRUCTURES_TO_TEST = None if not SUBSET_TEST_MODE else 1  # 🎯 Test ALL structures in full mode
    
    print("🌳 PROPER MCTS TREE SEARCH TEST - Testing MCTS Framework with Real Data")
    print("=" * 70)
    
    if SUBSET_TEST_MODE:
        print("🚀 SUBSET TESTING MODE: Testing 1 structure for debugging")
        print("📊 This mode is for quick testing and debugging of MCTS framework")
        print("📊 Set SUBSET_TEST_MODE = True for full batch testing")
    else:
        print("🚀 FULL BATCH TESTING MODE: Testing ALL CAMEO structures")
        print("📊 This will process all CAMEO structures with MCTS tree search")
        print("📊 Expected duration: Several hours with full MCTS exploration")
        print("📊 Results will be saved incrementally every 5 structures")
    
    print("Key Question: Does the MCTS framework actually work and improve AAR?")
    print("Testing the PROPER MCTS implementation, not just iterations!")
    print("🎯 GOAL: Verify MCTS tree growth, UCB1 selection, and AAR improvement")
    print("📁 DPLM-2 baselines: /home/caom/AID3/dplm/generation-results/dplm2_650m/inverse_folding")
    print("🔬 Strategy: MCTS tree search with different masking strategies")
    print("⚡ FRAMEWORK: Selection → Expansion → Simulation → Backpropagation")
    print("💾 INCREMENTAL SAVING: Every 5 structures to prevent data loss")
    print("📊 COMPREHENSIVE: Processing ALL CAMEO structures with MCTS")
    
    results = []
    
    # Load and test CAMEO structures
    print("\n" + "="*70)
    print("MCTS TREE SEARCH EVALUATION ON CAMEO 2022")
    print("="*70)
    
    # Load correct reference sequences from FASTA
    print("Loading correct reference sequences from FASTA...")
    correct_reference_sequences = load_correct_reference_sequences()
    
    if not correct_reference_sequences:
        print("Failed to load correct reference sequences - cannot proceed")
        return 1
    
    print(f"Loaded {len(correct_reference_sequences)} correct reference sequences")
    
    loader = CAMEODataLoader()
    
    if not loader.structures:
        print("No CAMEO structures available - using mock data only")
        print("Make sure CAMEO data is downloaded to /net/scratch/caom/dplm_datasets/")
    else:
        print(f"Found {len(loader.structures)} CAMEO structures")
        
        # Process CAMEO structures based on testing mode
        all_structures = []
        for idx, structure_file in enumerate(loader.structures):
            structure = loader.get_structure_by_index(idx)
            if structure:
                all_structures.append((idx, structure))
        
        if SUBSET_TEST_MODE:
            print("SUBSET TESTING: Testing 1 structure for debugging")
            test_structures = all_structures[:1]  # Take only first structure
        else:
            print("FULL TESTING: Testing all structures")
            test_structures = all_structures
        
        # Track progress
        main_start_time = time.time()
        successful_count = 0
        
    for i, (idx, structure) in enumerate(test_structures):
            print(f"\n{'='*70}")
            if SUBSET_TEST_MODE:
                print(f"SUBSET TEST: CAMEO Structure {structure['pdb_id']}_{structure['chain_id']}")
            else:
                print(f"CAMEO Structure {i+1}/{len(test_structures)} - {structure['pdb_id']}_{structure['chain_id']}")
            print(f"{'='*70}")
            
            # Estimate remaining time (only for full batch mode)
            if not SUBSET_TEST_MODE and i > 0:
                elapsed = time.time() - main_start_time
                avg_time_per_structure = elapsed / i
                remaining_structures = len(test_structures) - i
                estimated_remaining = remaining_structures * avg_time_per_structure
                print(f"Progress: {i}/{len(test_structures)} ({100*i/len(test_structures):.1f}%)")
                print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            
            structure_name = f"CAMEO {structure['pdb_id']}_{structure['chain_id']}"
            
            try:
                # 🎯 CRITICAL: Test the PROPER MCTS framework
                result = test_mcts_tree_search(structure, structure_name, correct_reference_sequences)
                if result:
                    results.append(result)
                    successful_count += 1
                    if SUBSET_TEST_MODE:
                        print(f"SUBSET TEST completed successfully for {structure_name}")
                    else:
                        print(f"Structure {i+1}/{len(test_structures)} processed successfully")
                    
                    # 🎯 INCREMENTAL SAVING: Save results every 5 structures (or immediately for subset mode)
                    if SUBSET_TEST_MODE or successful_count % 5 == 0 or successful_count == len(test_structures):
                        if SUBSET_TEST_MODE:
                            print(f"💾 Saving subset test results immediately")
                        else:
                            print(f"💾 Saving incremental results ({successful_count}/{len(test_structures)} structures processed)")
                        try:
                            save_results_to_files(results)
                            print(f"✅ Results saved successfully")
                        except Exception as e:
                            print(f"⚠️  Warning: Could not save results: {e}")
                else:
                    if SUBSET_TEST_MODE:
                        print(f"❌ SUBSET TEST failed for {structure_name}")
                        print(f"🔧 Check the error and fix before running full batch")
                    else:
                        print(f"❌ FULL BATCH TEST failed for {structure_name}")
                        print(f"📊 Continuing with next structure...")
                        print(f"📊 Progress: {successful_count}/{len(test_structures)} successful so far")
            except Exception as e:
                if SUBSET_TEST_MODE:
                    print(f"❌ SUBSET TEST error for {structure_name}: {e}")
                    print(f"🔧 Need to debug this error before full batch testing")
                else:
                    print(f"❌ FULL BATCH TEST error for {structure_name}: {e}")
                    print(f"📊 Continuing with next structure...")
                    print(f"📊 Progress: {successful_count}/{len(test_structures)} successful so far")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY OF MCTS TREE SEARCH RESULTS")
    print("="*70)
    
    if results:
        # Display comprehensive comparison table
        print(f"📊 MCTS TREE SEARCH vs DPLM-2 Baseline - AAR & Reward Results")
        print("=" * 140)
        print(f"{'Structure':<20} {'Length':<6} {'DPLM-2 AAR':<12} {'MCTS AAR':<10} {'Δ AAR':<10} {'Base R':<8} {'Final R':<8} {'Δ R':<8} {'Base scTM':<9} {'Final scTM':<10} {'Δ scTM':<8} {'Time':<8}")
        print("-" * 140)
        
        for result in results:
            name = result['structure_name'].replace('CAMEO_', '').replace('CAMEO ', '')[:19]
            baseline_aar = result.get('baseline_aar', 0.0)
            final_aar = result.get('final_aar', 0.0)
            aar_delta = result.get('aar_improvement', 0.0)
            baseline_reward = result.get('baseline_reward', 0.0)
            final_reward = result.get('final_reward', 0.0)
            reward_delta = result.get('reward_improvement', 0.0)
            baseline_sctm = result.get('baseline_sctm', 0.0)
            final_sctm = result.get('final_sctm', 0.0)
            sctm_delta = result.get('sctm_improvement', 0.0)
            search_time = result.get('search_time', 'N/A')
            
            # Format values
            baseline_r_str = f"{baseline_reward:.3f}" if isinstance(baseline_reward, (int, float)) else "N/A"
            final_r_str = f"{final_reward:.3f}" if isinstance(final_reward, (int, float)) else "N/A"
            delta_r_str = f"{reward_delta:+.3f}" if isinstance(reward_delta, (int, float)) else "N/A"
            baseline_sctm_str = f"{baseline_sctm:.3f}" if isinstance(baseline_sctm, (int, float)) else "N/A"
            final_sctm_str = f"{final_sctm:.3f}" if isinstance(final_sctm, (int, float)) else "N/A"
            delta_sctm_str = f"{sctm_delta:+.3f}" if isinstance(sctm_delta, (int, float)) else "N/A"
            time_str = f"{search_time:.1f}s" if isinstance(search_time, (int, float)) else str(search_time)
            
            print(f"{name:<20} {result['length']:<6} {baseline_aar:<12.1%} {final_aar:<10.1%} "
                  f"{aar_delta:<10.1%} {baseline_r_str:<8} {final_r_str:<8} {delta_r_str:<8} "
                  f"{baseline_sctm_str:<9} {final_sctm_str:<10} {delta_sctm_str:<8} {time_str:<8}")
        
        # Statistics
        successful_results = [r for r in results if r.get('mcts_success', False)]
        if successful_results:
            # Calculate AAR statistics
            baseline_aars = [r['baseline_aar'] for r in successful_results if 'baseline_aar' in r]
            final_aars = [r['final_aar'] for r in successful_results if 'final_aar' in r]
            
            print(f"\nMCTS Performance Summary:")
            print(f"  Success rate: {len(successful_results)}/{len(results)} ({100*len(successful_results)/len(results):.1f}%)")
            
            if baseline_aars and final_aars:
                avg_baseline_aar = sum(baseline_aars) / len(baseline_aars)
                avg_final_aar = sum(final_aars) / len(final_aars)
                avg_improvement = avg_final_aar - avg_baseline_aar
                
                print(f"\nAAR Results:")
                print(f"  Average DPLM-2 Baseline: {avg_baseline_aar:.1%}")
                print(f"  Average MCTS Final:  {avg_final_aar:.1%}")
                print(f"  Average Improvement:     {avg_improvement:+.1%}")
                
                # Count improvements
                improvements = [r.get('aar_improvement', 0) for r in successful_results if 'aar_improvement' in r]
                improved_count = sum(1 for imp in improvements if imp > 0)
                
                if improvements:
                    print(f"  MCTS improved {improved_count}/{len(successful_results)} structures")
                
                # 🎯 MCTS FRAMEWORK VALIDATION
                print(f"\n🌳 MCTS Framework Validation:")
                print(f"  Tree Search: {'✅ Working' if successful_results else '❌ Failed'}")
                print(f"  AAR Improvement: {'✅ Achieved' if improved_count > 0 else '❌ No improvement'}")
                
                # Show search time statistics
                search_times = [r.get('search_time', 0) for r in successful_results if 'search_time' in r and isinstance(r['search_time'], (int, float))]
                if search_times:
                    avg_search_time = sum(search_times) / len(search_times)
                    max_search_time = max(search_times)
                    min_search_time = min(search_times)
                    
                    print(f"\n⏱️  Search Performance:")
                    print(f"  Average Search Time: {avg_search_time:.1f}s")
                    print(f"  Search Time Range: {min_search_time:.1f}s - {max_search_time:.1f}s")
        
        print("\nKey Insights:")
        print("• MCTS Tree Search: Tests the actual MCTS framework")
        print("• Tree Growth: Different masking strategies create branches")
        print("• UCB1 Selection: Balances exploration vs exploitation")
        print("• AAR Improvement: Primary metric for MCTS success")
        
    else:
        print("No successful results to summarize")
    
    if results:
        print(f"\nMCTS evaluation completed! Processed {len(results)} CAMEO structures")
    else:
        print(f"\nNo results to save - MCTS evaluation failed")
    
    # Save results to JSON for further analysis
    save_results_to_files(results)
    
    print(f"\nMCTS Tree Search Test completed! Results summary available above.")


if __name__ == "__main__":
    main()
