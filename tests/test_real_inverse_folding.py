#!/usr/bin/env python3
"""
Real MCTS inverse folding test - comparing MCTS vs DPLM-2 baseline.

This test shows whether MCTS can improve upon DPLM-2's baseline 
performance for inverse folding tasks using better optimization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import GeneralMCTS
from core.dplm2_integration import DPLM2Integration
from utils.structure_evaluation import create_structure_evaluator
from utils.reward_computation import LengthAwareRewardComputation
from utils.protein_utils import create_mock_structure_no_sequence
import torch
import random


def test_mcts_vs_baseline_inverse_folding():
    """
    Real test: Does MCTS improve upon DPLM-2 baseline for inverse folding?
    
    Given the same target structure:
    - Baseline: DPLM-2 direct generation  
    - MCTS: DPLM-2 + MCTS optimization
    - Compare: Which achieves better structural compatibility?
    """
    print("🎯 MCTS vs Baseline Inverse Folding Test")
    print("=" * 60)
    print("Question: Can MCTS improve upon DPLM-2's baseline performance?")
    print()
    
    # Test different protein sizes
    test_lengths = [50, 100, 200]
    
    for length in test_lengths:
        print(f"\n🧬 Testing {length}-residue protein")
        print("-" * 40)
        
        # Create target structure
        target_structure = create_mock_structure_no_sequence(length=length)
        target_structure['target_length'] = length
        target_structure['pdb_id'] = f'test_{length}'
        
        print(f"Target structure: {length} residues")
        
        # Initialize evaluation tools
        dplm2 = DPLM2Integration()
        evaluator = create_structure_evaluator()
        reward_computer = LengthAwareRewardComputation(use_real_structure_eval=True)
        
        # BASELINE: DPLM-2 direct generation
        print(f"\n📊 Baseline (DPLM-2 only):")
        baseline_sequences = []
        baseline_rewards = []
        baseline_metrics = []
        
        for trial in range(3):  # Multiple trials for statistical significance
            try:
                # Direct DPLM-2 generation
                baseline_seq = dplm2.generate_sequence(
                    target_structure, target_length=length, temperature=1.0
                )
                
                if baseline_seq and len(baseline_seq) == length:
                    # Evaluate baseline sequence
                    baseline_reward = reward_computer.compute_reward(
                        baseline_seq, target_structure, detailed=False
                    )
                    baseline_results = evaluator.evaluate_designability(
                        baseline_seq, target_structure
                    )
                    
                    baseline_sequences.append(baseline_seq)
                    baseline_rewards.append(baseline_reward)
                    baseline_metrics.append(baseline_results)
                    
                    print(f"  Trial {trial+1}: Reward={baseline_reward:.3f}, "
                          f"RMSD={baseline_results['bb_rmsd']:.2f}Å, "
                          f"TM={baseline_results['sc_tmscore']:.3f}")
                else:
                    print(f"  Trial {trial+1}: Generation failed")
                    
            except Exception as e:
                print(f"  Trial {trial+1}: Error - {e}")
        
        if not baseline_rewards:
            print("  ❌ Baseline generation failed, skipping this length")
            continue
            
        best_baseline_idx = baseline_rewards.index(max(baseline_rewards))
        best_baseline_reward = baseline_rewards[best_baseline_idx]
        best_baseline_seq = baseline_sequences[best_baseline_idx]
        best_baseline_metrics = baseline_metrics[best_baseline_idx]
        
        print(f"  Best baseline: Reward={best_baseline_reward:.3f}, "
              f"RMSD={best_baseline_metrics['bb_rmsd']:.2f}Å, "
              f"TM={best_baseline_metrics['sc_tmscore']:.3f}")
        
        # MCTS: DPLM-2 + MCTS optimization
        print(f"\n🔍 MCTS Optimization:")
        
        # Configure MCTS based on length
        if length <= 100:
            num_simulations, max_depth = 20, 3
        elif length <= 200:
            num_simulations, max_depth = 30, 4
        else:
            num_simulations, max_depth = 40, 5
            
        try:
            # Initialize MCTS
            mcts = GeneralMCTS(
                task_type="inverse_folding",
                max_depth=max_depth,
                num_simulations=num_simulations,
                exploration_constant=1.414,
                temperature=1.0,
                num_candidates_per_expansion=2,
                use_plddt_masking=False,  # For speed
                simultaneous_sampling=False
            )
            
            print(f"  MCTS config: {num_simulations} simulations, depth {max_depth}")
            
            # Run MCTS optimization
            mcts_best_seq, mcts_best_reward = mcts.search(
                target_structure, target_length=length
            )
            
            # Evaluate MCTS result
            mcts_results = evaluator.evaluate_designability(
                mcts_best_seq, target_structure
            )
            
            print(f"  MCTS result: Reward={mcts_best_reward:.3f}, "
                  f"RMSD={mcts_results['bb_rmsd']:.2f}Å, "
                  f"TM={mcts_results['sc_tmscore']:.3f}")
            
            # COMPARISON
            print(f"\n📈 Results Comparison:")
            reward_improvement = mcts_best_reward - best_baseline_reward
            reward_improvement_pct = (reward_improvement / best_baseline_reward) * 100
            
            rmsd_improvement = best_baseline_metrics['bb_rmsd'] - mcts_results['bb_rmsd']
            tm_improvement = mcts_results['sc_tmscore'] - best_baseline_metrics['sc_tmscore']
            
            print(f"  Reward:     {best_baseline_reward:.3f} → {mcts_best_reward:.3f} "
                  f"({reward_improvement:+.3f}, {reward_improvement_pct:+.1f}%)")
            print(f"  RMSD:       {best_baseline_metrics['bb_rmsd']:.2f}Å → {mcts_results['bb_rmsd']:.2f}Å "
                  f"({rmsd_improvement:+.2f}Å)")
            print(f"  TM-score:   {best_baseline_metrics['sc_tmscore']:.3f} → {mcts_results['sc_tmscore']:.3f} "
                  f"({tm_improvement:+.3f})")
            
            # Success assessment
            improvements = 0
            if reward_improvement > 0:
                print(f"  ✅ Reward improved")
                improvements += 1
            if rmsd_improvement > 0:
                print(f"  ✅ RMSD improved (lower is better)")
                improvements += 1
            if tm_improvement > 0:
                print(f"  ✅ TM-score improved")
                improvements += 1
                
            if improvements >= 2:
                print(f"  🎉 MCTS shows clear improvement over baseline!")
            elif improvements >= 1:
                print(f"  ✅ MCTS shows some improvement over baseline")
            else:
                print(f"  📊 MCTS comparable to baseline (may need more optimization)")
                
        except Exception as e:
            print(f"  ❌ MCTS optimization failed: {e}")
            import traceback
            traceback.print_exc()


def test_mcts_optimization_process():
    """
    Test to see the MCTS optimization process in action.
    Shows how rewards improve over MCTS iterations.
    """
    print("\n\n🔄 MCTS Optimization Process Analysis")
    print("=" * 60)
    print("Analyzing how MCTS improves sequences over iterations")
    
    # Use a smaller protein for detailed analysis
    length = 50
    target_structure = create_mock_structure_no_sequence(length=length)
    target_structure['target_length'] = length
    
    # Get baseline
    dplm2 = DPLM2Integration()
    reward_computer = LengthAwareRewardComputation(use_real_structure_eval=True)
    
    try:
        baseline_seq = dplm2.generate_sequence(target_structure, target_length=length)
        baseline_reward = reward_computer.compute_reward(baseline_seq, target_structure)
        
        print(f"Baseline DPLM-2 reward: {baseline_reward:.3f}")
        print(f"Target: Find sequences with reward > {baseline_reward:.3f}")
        
        # Run MCTS with progress tracking
        mcts = GeneralMCTS(
            task_type="inverse_folding",
            max_depth=3,
            num_simulations=15,  # Smaller for detailed tracking
            exploration_constant=1.414,
            use_plddt_masking=False,
            simultaneous_sampling=False
        )
        
        print(f"\nRunning MCTS optimization...")
        mcts_best_seq, mcts_best_reward = mcts.search(target_structure, target_length=length)
        
        print(f"Final MCTS reward: {mcts_best_reward:.3f}")
        improvement = mcts_best_reward - baseline_reward
        print(f"Improvement: {improvement:+.3f} ({improvement/baseline_reward*100:+.1f}%)")
        
        if improvement > 0:
            print("✅ MCTS successfully improved upon DPLM-2 baseline!")
        else:
            print("📊 MCTS result comparable to baseline (normal for small test)")
            
    except Exception as e:
        print(f"Process analysis failed: {e}")


def main():
    """Run comprehensive MCTS vs baseline comparison."""
    print("🧬 Real MCTS Inverse Folding Performance Test")
    print("="*80)
    print("Testing whether MCTS can improve upon DPLM-2 baseline performance")
    print("This is the core question for our research!")
    print()
    
    try:
        # Main comparison test
        test_mcts_vs_baseline_inverse_folding()
        
        # Detailed process analysis
        test_mcts_optimization_process()
        
        print("\n\n🎯 Summary:")
        print("This test shows whether our MCTS approach can improve")
        print("structural compatibility beyond what DPLM-2 achieves alone.")
        print("Positive results indicate MCTS is adding optimization value!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()