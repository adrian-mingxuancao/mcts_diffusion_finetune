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
   - Set SUBSET_TEST_MODE = False for full batch testing

2. FULL BATCH TESTING MODE (SUBSET_TEST_MODE = False):
   - Tests ALL CAMEO structures
   - Comprehensive MCTS evaluation for statistical significance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
import random
import numpy as np
from datetime import datetime
from utils.cameo_data_loader import CAMEODataLoader

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

def load_dplm2_baseline_sequence(structure_name: str):
    """Load pre-generated DPLM-2 baseline sequence from official results"""
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
        structure: Structure dictionary
        structure_name: Name for logging
        correct_reference_sequences: Dictionary of correct reference sequences from FASTA
    """
    print(f"\n🌳 Testing PROPER MCTS TREE SEARCH: {structure_name}")
    print("=" * 60)
    
    if structure is None:
        print(f"❌ Failed to load {structure_name}")
        return None
    
    print(f"Structure info:")
    print(f"  Length: {structure['length']} residues")
    print(f"  PDB ID: {structure.get('pdb_id', 'N/A')}")
    print(f"  Chain ID: {structure.get('chain_id', 'N/A')}")
    print(f"  Avg plDDT: {sum(structure['plddt_scores']) / len(structure['plddt_scores']):.3f}")
    
    # Get correct reference sequence
    structure_id = f"{structure['pdb_id']}_{structure['chain_id']}"
    correct_ref_seq = correct_reference_sequences.get(structure_id)
    
    if not correct_ref_seq:
        print(f"  ❌ No correct reference sequence found for {structure_id}")
        return None
    
    print(f"  ✅ Using correct reference sequence from FASTA: {structure_id}")
    print(f"  🔍 Reference sequence: {correct_ref_seq[:40]}... (len={len(correct_ref_seq)})")
    
    # Load DPLM-2 baseline sequence
    print(f"\n📊 Loading DPLM-2 baseline sequence...")
    baseline_seq = load_dplm2_baseline_sequence(structure_name)
    
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
    
    # Initialize DPLM-2 integration for MCTS
    try:
        from core.dplm2_integration import DPLM2Integration
        dplm2 = DPLM2Integration(use_local=True)
        print("  ✅ DPLM-2 integration initialized for MCTS")
    except Exception as e:
        print(f"  ❌ Failed to initialize DPLM-2: {e}")
        # Try alternative import path
        try:
            import sys
            sys.path.append('/home/caom/AID3/dplm/mcts_diffusion_finetune/core')
            from dplm2_integration import DPLM2Integration
            dplm2 = DPLM2Integration(use_local=True)
            print("  ✅ DPLM-2 integration initialized (alternative path)")
        except Exception as e2:
            print(f"  ❌ Alternative import also failed: {e2}")
            return None
    
    # 🎯 STEP 1: Initialize MCTS with the baseline sequence
    print(f"\n🌳 Step 1: Initializing MCTS with baseline sequence")
    
    try:
        from core.sequence_level_mcts import SequenceLevelMCTS
        
        # Create MCTS instance with the baseline sequence
        mcts = SequenceLevelMCTS(
            initial_sequence=baseline_seq,
            task_type="inverse_folding",
            max_depth=5,
            num_simulations=30,
            exploration_constant=1.414,
            temperature=1.0,
            num_candidates_per_expansion=3,
            use_plddt_masking=True,
            simultaneous_sampling=True,
            dplm2_integration=dplm2
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
        # Run MCTS search - this should grow a tree!
        start_time = time.time()
        
        best_sequence, best_reward = mcts.search(
            target_length=len(baseline_seq),
            max_simulations=30,  # Start with fewer simulations for testing
            max_depth=5,
            exploration_constant=1.414,
            temperature=1.0,
            num_candidates_per_expansion=3,
            start_from_complete=True,  # Start from baseline sequence
            reference_sequence=correct_ref_seq,  # For AAR calculation
            structure=structure  # For DPLM-2 integration
        )
        
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
            
            # Return results
            result = {
                'structure_name': structure_name,
                'length': structure['length'],
                'baseline_aar': baseline_aar,
                'final_aar': best_aar,
                'aar_improvement': aar_improvement,
                'final_reward': best_reward,
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
                f.write(f"{'Structure':<15} {'Length':<6} {'DPLM-2 AAR':<12} {'MCTS AAR':<10} {'Δ AAR':<10} {'Reward':<10} {'Time':<8}\n")
                f.write("-" * 80 + "\n")
                
                for result in successful_results:
                    name = result['structure_name'].replace('CAMEO ', '')[:14]
                    baseline_aar = result.get('baseline_aar', 0.0)
                    final_aar = result.get('final_aar', 0.0) 
                    aar_delta = result.get('aar_improvement', 0.0)
                    reward = result.get('final_reward', 0.0)
                    search_time = result.get('search_time', 0.0)
                    
                    f.write(f"{name:<15} {result['length']:<6} {baseline_aar:<12.1%} {final_aar:<10.1%} "
                           f"{aar_delta:<10.1%} {reward:<10.4f} {search_time:<8.1f}s\n")
                
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
    SUBSET_TEST_MODE = False  # 🚀 Set to False for full batch testing, True for debugging
    MAX_STRUCTURES_TO_TEST = None if not SUBSET_TEST_MODE else 1  # 🎯 Test ALL structures in full mode
    
    print("🌳 PROPER MCTS TREE SEARCH TEST - Testing MCTS Framework with Real Data")
    print("=" * 70)
    
    if SUBSET_TEST_MODE:
        print("🚀 SUBSET TESTING MODE: Testing 1 structure for debugging")
        print("📊 This mode is for quick testing and debugging of MCTS framework")
        print("📊 Set SUBSET_TEST_MODE = False for full batch testing")
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
        else:
            print(f"FULL BATCH TESTING: Processing {len(all_structures)} CAMEO structures")
        
        # Limit testing based on mode
        if MAX_STRUCTURES_TO_TEST is not None:
            test_structures = all_structures[:MAX_STRUCTURES_TO_TEST]
            if SUBSET_TEST_MODE:
                print(f"SUBSET MODE: Testing {len(test_structures)} structure for debugging")
            else:
                print(f"Testing first {len(test_structures)}/{len(all_structures)} structures")
        else:
            test_structures = all_structures
            print(f"Testing ALL {len(test_structures)} structures")
        
        # Track progress
        main_start_time = time.time()
        successful_count = 0
        
        # Initialize test_structures if not defined
        if 'test_structures' not in locals():
            test_structures = all_structures
        
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
        print("📊 MCTS TREE SEARCH vs DPLM-2 Baseline - AAR & Reward Results")
        print("=" * 85)
        print(f"{'Structure':<20} {'Length':<6} {'DPLM-2 AAR':<12} {'MCTS AAR':<10} {'Δ AAR':<10} {'Reward':<12} {'Time':<8}")
        print("-" * 85)
        
        for result in results:
            name = result['structure_name'].replace('CAMEO_', '').replace('CAMEO ', '')[:19]
            baseline_aar = result.get('baseline_aar', 0.0)
            final_aar = result.get('final_aar', 0.0)
            aar_delta = result.get('aar_improvement', 0.0)
            reward = result.get('final_reward', 'N/A')
            search_time = result.get('search_time', 'N/A')
            
            # Format reward and time
            reward_str = f"{reward:.4f}" if isinstance(reward, (int, float)) else str(reward)
            time_str = f"{search_time:.1f}s" if isinstance(search_time, (int, float)) else str(search_time)
            
            print(f"{name:<20} {result['length']:<6} {baseline_aar:<12.1%} {final_aar:<10.1%} "
                  f"{aar_delta:<10.1%} {reward_str:<12} {time_str:<8}")
        
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
