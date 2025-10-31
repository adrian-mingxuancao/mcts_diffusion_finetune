#!/usr/bin/env python3
"""
Comprehensive CAMEO 2022 Evaluation Script

This script evaluates all 17 CAMEO structures with both DPLM-2 baseline
and MCTS optimization, providing detailed AAR and scTM-score comparisons.
"""

import sys
import os
import time
import json
import argparse
import traceback
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cameo_data_loader import CAMEODataLoader
from core.sequence_level_mcts import GeneralMCTS
from utils.structure_evaluation import StructureEvaluator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_single_structure(structure, structure_name: str, max_time_minutes: int = 30):
    """
    Evaluate a single structure with both DPLM-2 baseline and MCTS.
    
    Args:
        structure: CAMEO structure dictionary
        structure_name: Structure identifier
        max_time_minutes: Maximum time per structure
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"🧬 Evaluating {structure_name}")
    print(f"{'='*70}")
    
    if structure is None:
        return {'error': 'Failed to load structure'}
    
    length = structure['length']
    print(f"Structure info:")
    print(f"  Length: {length} residues")
    print(f"  PDB ID: {structure.get('pdb_id', 'N/A')}")
    print(f"  Chain ID: {structure.get('chain_id', 'N/A')}")
    print(f"  Has reference sequence: {structure.get('sequence') is not None}")
    
    if not structure.get('sequence'):
        print("⚠️ No reference sequence - skipping AAR evaluation")
        return {'error': 'No reference sequence for AAR calculation'}
    
    # Check for valid reference sequence (not all X)
    ref_seq = structure['sequence']
    valid_positions = sum(1 for aa in ref_seq if aa != 'X')
    if valid_positions == 0:
        print("⚠️ Reference sequence contains only X tokens - skipping")
        return {'error': 'Reference sequence has no valid positions'}
    
    print(f"  Valid reference positions: {valid_positions}/{len(ref_seq)} ({100*valid_positions/len(ref_seq):.1f}%)")
    
    results = {
        'structure_name': structure_name,
        'pdb_id': structure.get('pdb_id'),
        'chain_id': structure.get('chain_id'), 
        'length': length,
        'valid_positions': valid_positions,
        'total_positions': len(ref_seq)
    }
    
    try:
        # Initialize components
        evaluator = StructureEvaluator(use_cuda=True)
        
        # 🎯 AGGRESSIVE MCTS PARAMETERS for protein sequence optimization
        if length <= 100:
            max_depth, num_sims, exploration = 12, 2000, 4.0  # 🚀 AGGRESSIVE: Was 3, 50, 1.5
            category = "small"
        elif length <= 300:
            max_depth, num_sims, exploration = 15, 3000, 5.0  # 🚀 AGGRESSIVE: Was 8, 100, 2.0
            category = "medium"
        else:
            max_depth, num_sims, exploration = 18, 5000, 6.0  # 🚀 AGGRESSIVE: Was 5, 20, 1.0
            category = "large"
        
        mcts = GeneralMCTS(
            task_type="inverse_folding",
            max_depth=max_depth,
            num_simulations=num_sims,
            exploration_constant=exploration,
            temperature=2.0,  # 🚀 AGGRESSIVE: More stochastic exploration
            num_candidates_per_expansion=15,  # 🚀 AGGRESSIVE: More diverse candidates
            use_plddt_masking=True,  # ✅ KEEP: Still masking pLDDT low positions
            simultaneous_sampling=False
        )
        
        results['category'] = category
        results['mcts_config'] = {
            'max_depth': max_depth,
            'num_simulations': num_sims,
            'exploration': exploration
        }
        
        # 1. DPLM-2 Baseline Evaluation
        print(f"\n📊 Computing DPLM-2 baseline performance...")
        baseline_start = time.time()
        
        try:
            baseline_seq = mcts.dplm2_integration.generate_sequence(structure, target_length=length)
            
            if baseline_seq and structure.get('sequence'):
                # Remove any X tokens from baseline sequence
                baseline_seq_clean = "".join([aa for aa in baseline_seq if aa in "ACDEFGHIKLMNPQRSTVWY"])
                
                baseline_designability = evaluator.evaluate_designability(baseline_seq_clean, structure)
                baseline_aar = evaluator.compute_sequence_recovery(baseline_seq_clean, structure['sequence'])
                
                results['baseline'] = {
                    'sequence': baseline_seq_clean,
                    'sequence_length': len(baseline_seq_clean),
                    'sc_tmscore': baseline_designability.get('sc_tmscore', 0.0),
                    'bb_rmsd': baseline_designability.get('bb_rmsd', 999.0),
                    'plddt': baseline_designability.get('plddt', 0.0),
                    'aar': baseline_aar,
                    'time': time.time() - baseline_start
                }
                
                print(f"  DPLM-2 Baseline Results:")
                print(f"    Sequence length: {len(baseline_seq_clean)}")
                print(f"    scTM-score: {results['baseline']['sc_tmscore']:.3f}")
                print(f"    RMSD: {results['baseline']['bb_rmsd']:.2f} Å")
                print(f"    AAR: {results['baseline']['aar']:.1%}")
                print(f"    Time: {results['baseline']['time']:.1f}s")
                
        except Exception as e:
            print(f"  ⚠️ Baseline evaluation failed: {e}")
            results['baseline'] = {'error': str(e)}
        
        # 2. MCTS Optimization with pLDDT-based masking
        # 🎯 STRATEGY: 
        # 1. Start with DPLM-2 baseline sequence
        # 2. Mask positions with low pLDDT (structure quality based)
        # 3. Fill masked positions with random amino acids  
        # 4. Use AGGRESSIVE MCTS to optimize the random fills
        # 5. Target: Improve AAR through intelligent sequence modifications
        print(f"\n🚀 Running MCTS optimization...")
        mcts_start = time.time()
        
        try:
            best_sequence, best_reward = mcts.search(structure, target_length=length)
            
            if best_sequence:
                # Clean MCTS sequence - remove X tokens
                mcts_seq_clean = "".join([aa for aa in best_sequence if aa in "ACDEFGHIKLMNPQRSTVWY"])
                
                if mcts_seq_clean and structure.get('sequence'):
                    mcts_designability = evaluator.evaluate_designability(mcts_seq_clean, structure)
                    mcts_aar = evaluator.compute_sequence_recovery(mcts_seq_clean, structure['sequence'])
                    
                    results['mcts'] = {
                        'sequence': mcts_seq_clean,
                        'sequence_length': len(mcts_seq_clean),
                        'sc_tmscore': mcts_designability.get('sc_tmscore', 0.0),
                        'bb_rmsd': mcts_designability.get('bb_rmsd', 999.0),
                        'plddt': mcts_designability.get('plddt', 0.0),
                        'aar': mcts_aar,
                        'reward': best_reward,
                        'time': time.time() - mcts_start
                    }
                    
                    print(f"  MCTS Results:")
                    print(f"    Sequence length: {len(mcts_seq_clean)}")
                    print(f"    scTM-score: {results['mcts']['sc_tmscore']:.3f}")
                    print(f"    RMSD: {results['mcts']['bb_rmsd']:.2f} Å")
                    print(f"    AAR: {results['mcts']['aar']:.1%}")
                    print(f"    Reward: {results['mcts']['reward']:.3f}")
                    print(f"    Time: {results['mcts']['time']:.1f}s")
                    
                    # Calculate improvements
                    if 'baseline' in results and 'error' not in results['baseline']:
                        results['improvements'] = {
                            'sctm_delta': results['mcts']['sc_tmscore'] - results['baseline']['sc_tmscore'],
                            'rmsd_delta': results['baseline']['bb_rmsd'] - results['mcts']['bb_rmsd'],  # Lower is better
                            'aar_delta': results['mcts']['aar'] - results['baseline']['aar']
                        }
                        
                        print(f"\n🔄 MCTS vs DPLM-2 Comparison:")
                        print(f"    scTM improvement: {results['improvements']['sctm_delta']:+.3f}")
                        print(f"    RMSD improvement: {results['improvements']['rmsd_delta']:+.2f} Å")
                        print(f"    AAR improvement: {results['improvements']['aar_delta']:+.1%}")
                else:
                    results['mcts'] = {'error': 'No valid sequence generated'}
            else:
                results['mcts'] = {'error': 'MCTS failed to generate sequence'}
                
        except Exception as e:
            print(f"  ⚠️ MCTS evaluation failed: {e}")
            results['mcts'] = {'error': str(e)}
        
        results['total_time'] = time.time() - baseline_start
        results['success'] = 'baseline' in results and 'mcts' in results and \
                            'error' not in results.get('baseline', {}) and \
                            'error' not in results.get('mcts', {})
        
        print(f"\n✅ Evaluation completed in {results['total_time']:.1f}s")
        return results
        
    except Exception as e:
        error_msg = f"Evaluation failed: {e}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}")
        return {'error': error_msg, 'structure_name': structure_name}


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive CAMEO evaluation')
    parser.add_argument('--output', '-o', default='cameo_evaluation_results.json',
                       help='Output JSON file for results')
    parser.add_argument('--max-structures', '-n', type=int, default=17,
                       help='Maximum number of structures to evaluate')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum structure length')
    parser.add_argument('--max-length', type=int, default=300,
                       help='Maximum structure length')
    parser.add_argument('--time-limit', type=int, default=30,
                       help='Time limit per structure (minutes)')
    
    args = parser.parse_args()
    
    print("🧬 Comprehensive CAMEO 2022 Evaluation")
    print("=" * 70)
    print(f"Evaluating DPLM-2 baseline vs MCTS on real protein structures")
    print(f"Output file: {args.output}")
    print(f"Length range: {args.min_length}-{args.max_length} residues")
    print(f"Max structures: {args.max_structures}")
    
    # Load CAMEO data
    loader = CAMEODataLoader()
    
    if not loader.structures:
        print("❌ No CAMEO structures available!")
        print("Make sure CAMEO data is downloaded to /net/scratch/caom/dplm_datasets/")
        return
    
    print(f"\nFound {len(loader.structures)} CAMEO structures")
    
    # Filter structures by length and get reference sequences
    suitable_structures = []
    for idx, structure_file in enumerate(loader.structures):
        structure = loader.get_structure_by_index(idx)
        if structure and args.min_length <= structure['length'] <= args.max_length:
            if structure.get('sequence'):
                # Check if sequence has valid positions
                valid_pos = sum(1 for aa in structure['sequence'] if aa != 'X')
                if valid_pos > 0:
                    suitable_structures.append((idx, structure, structure_file))
    
    print(f"Found {len(suitable_structures)} suitable structures with reference sequences")
    
    # Limit number of structures
    test_structures = suitable_structures[:args.max_structures]
    print(f"Evaluating {len(test_structures)} structures")
    
    # Run evaluations
    all_results = []
    start_time = time.time()
    
    for i, (idx, structure, structure_file) in enumerate(test_structures):
        structure_name = f"CAMEO_{structure['pdb_id']}_{structure['chain_id']}"
        
        print(f"\n{'-'*70}")
        print(f"Progress: {i+1}/{len(test_structures)} - {structure_name}")
        print(f"Elapsed: {(time.time() - start_time)/60:.1f} min")
        
        result = evaluate_single_structure(structure, structure_name, args.time_limit)
        all_results.append(result)
        
        # Save intermediate results
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Final summary
    print(f"\n{'='*70}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*70}")
    
    successful_results = [r for r in all_results if r.get('success', False)]
    print(f"Successful evaluations: {len(successful_results)}/{len(all_results)}")
    
    if successful_results:
        # Calculate average metrics
        baseline_sctm = [r['baseline']['sc_tmscore'] for r in successful_results]
        baseline_aar = [r['baseline']['aar'] for r in successful_results]
        mcts_sctm = [r['mcts']['sc_tmscore'] for r in successful_results]
        mcts_aar = [r['mcts']['aar'] for r in successful_results]
        
        print(f"\n📊 Average Performance:")
        print(f"  DPLM-2 Baseline:")
        print(f"    Average scTM-score: {sum(baseline_sctm)/len(baseline_sctm):.3f}")
        print(f"    Average AAR: {sum(baseline_aar)/len(baseline_aar):.1%}")
        print(f"  MCTS Optimized:")
        print(f"    Average scTM-score: {sum(mcts_sctm)/len(mcts_sctm):.3f}")
        print(f"    Average AAR: {sum(mcts_aar)/len(mcts_aar):.1%}")
        
        avg_sctm_improvement = sum(mcts_sctm)/len(mcts_sctm) - sum(baseline_sctm)/len(baseline_sctm)
        avg_aar_improvement = sum(mcts_aar)/len(mcts_aar) - sum(baseline_aar)/len(baseline_aar)
        
        print(f"\n🔄 Average Improvements:")
        print(f"  scTM-score: {avg_sctm_improvement:+.3f}")
        print(f"  AAR: {avg_aar_improvement:+.1%}")
        
        # Individual results table
        print(f"\n📋 Individual Results:")
        print(f"{'Structure':<20} {'Length':<6} {'Baseline AAR':<12} {'MCTS AAR':<10} {'Δ AAR':<8} {'Baseline scTM':<12} {'MCTS scTM':<10} {'Δ scTM':<8}")
        print("-" * 100)
        
        for r in successful_results:
            name = r['structure_name'].replace('CAMEO_', '')
            baseline = r['baseline']
            mcts = r['mcts']
            improvements = r.get('improvements', {})
            
            print(f"{name:<20} {r['length']:<6} {baseline['aar']:<12.1%} {mcts['aar']:<10.1%} "
                  f"{improvements.get('aar_delta', 0):<8.1%} {baseline['sc_tmscore']:<12.3f} "
                  f"{mcts['sc_tmscore']:<10.3f} {improvements.get('sctm_delta', 0):<8.3f}")
    
    total_time = (time.time() - start_time) / 3600  # hours
    print(f"\n✅ Evaluation completed in {total_time:.2f} hours")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()



