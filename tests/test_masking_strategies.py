#!/usr/bin/env python3
"""
Comprehensive test comparing different masking strategies for MCTS-guided inverse folding.

This test compares:
1. Baseline DPLM-2 (no MCTS)
2. MCTS with no masking (full sequence optimization)
3. MCTS with random masking (random 15% positions masked)
4. MCTS with plDDT masking (intelligent 15% positions masked)

Uses real protein structures from PDB for validation.
"""

import sys
import os
import numpy as np
import random
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import GeneralMCTS
from utils.simple_protein_loader import get_test_proteins
from utils.reward_computation import compute_detailed_reward_analysis, LengthAwareRewardComputation
from utils.real_plddt_computation import compute_plddt_from_structure


def get_baseline_performance(structure: Dict, length: int) -> tuple:
    """Get baseline DPLM-2 performance without MCTS."""
    print(f"📊 Evaluating baseline DPLM-2...")
    
    from core.dplm2_integration import DPLM2Integration
    reward_computer = LengthAwareRewardComputation(use_real_structure_eval=True)
    
    dplm2 = DPLM2Integration()
    baseline_sequences = []
    baseline_rewards = []
    
    for i in range(3):  # Get 3 baseline sequences
        try:
            sequence = dplm2.generate_sequence(structure, target_length=length)
            if sequence and len(sequence) == length:
                reward = reward_computer.compute_reward(sequence, structure)
                baseline_sequences.append(sequence)
                baseline_rewards.append(reward)
                print(f"  Baseline {i+1}: {reward:.4f}")
        except Exception as e:
            print(f"  Warning: Baseline {i+1} failed: {e}")
            # Use random fallback
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            sequence = ''.join(random.choices(amino_acids, k=length))
            reward = reward_computer.compute_reward(sequence, structure)
            baseline_sequences.append(sequence)
            baseline_rewards.append(reward)
            print(f"  Fallback {i+1}: {reward:.4f}")
    
    best_idx = baseline_rewards.index(max(baseline_rewards))
    return baseline_rewards[best_idx], baseline_sequences[best_idx]


def create_masked_sequence(sequence: str, mask_positions: List[int]) -> str:
    """Create a masked sequence with X tokens at specified positions."""
    masked_seq = list(sequence)
    for pos in mask_positions:
        if 0 <= pos < len(masked_seq):
            masked_seq[pos] = 'X'
    return ''.join(masked_seq)


def test_masking_strategy(structure: Dict, strategy: str, num_simulations: int = 20) -> Dict:
    """Test a specific masking strategy."""
    length = structure['target_length']
    
    print(f"\n🧬 Testing {strategy} strategy")
    print("=" * 50)
    
    if strategy == "baseline":
        # Just return baseline performance
        reward, sequence = get_baseline_performance(structure, length)
        analysis = compute_detailed_reward_analysis(sequence, structure)
        return {
            'strategy': strategy,
            'reward': reward,
            'sequence': sequence,
            'analysis': analysis,
            'mask_percentage': 0.0,
            'masked_positions': []
        }
    
    # For MCTS strategies, determine masking
    if strategy == "no_masking":
        use_masking = False
        mask_positions = []
        
    elif strategy == "random_masking":
        use_masking = True
        # Random 15% masking
        num_mask = max(1, int(length * 0.15))
        mask_positions = random.sample(range(length), num_mask)
        
    elif strategy == "plddt_masking":
        use_masking = True
        # plDDT-based masking
        plddt_scores = compute_plddt_from_structure(structure)
        low_conf_positions = [i for i, score in enumerate(plddt_scores) if score < 0.7]
        mask_positions = low_conf_positions[:max(1, int(length * 0.15))]  # Limit to 15%
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    mask_percentage = len(mask_positions) / length * 100
    
    # Configure MCTS
    if length < 60:
        max_depth, exploration, actual_sims = 3, 1.8, min(15, num_simulations)
    elif length < 120:
        max_depth, exploration, actual_sims = 4, 1.414, min(20, num_simulations)
    else:
        max_depth, exploration, actual_sims = 5, 1.0, min(25, num_simulations)
    
    mcts = GeneralMCTS(
        task_type="inverse_folding",
        max_depth=max_depth,
        num_simulations=actual_sims,
        exploration_constant=exploration,
        temperature=1.0,
        num_candidates_per_expansion=2,
        use_plddt_masking=False,  # We handle masking manually
        simultaneous_sampling=False
    )
    
    print(f"Strategy: {strategy}")
    print(f"Masking: {mask_percentage:.1f}% ({len(mask_positions)}/{length} positions)")
    print(f"MCTS sims: {actual_sims}")
    
    # Create initial masked sequence if needed
    if use_masking and mask_positions:
        # Create a base sequence and mask it
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        base_sequence = ''.join(random.choices(amino_acids, k=length))
        initial_sequence = create_masked_sequence(base_sequence, mask_positions)
        print(f"Initial: {initial_sequence[:40]}{'...' if len(initial_sequence) > 40 else ''}")
        
        # For manual masking, we need to modify the structure
        modified_structure = structure.copy()
        modified_structure['mask_positions'] = mask_positions
        modified_structure['initial_sequence'] = initial_sequence
    else:
        modified_structure = structure
    
    # Run MCTS
    best_sequence, best_reward = mcts.search(modified_structure, target_length=length)
    analysis = compute_detailed_reward_analysis(best_sequence, structure)
    
    print(f"Best reward: {best_reward:.4f}")
    print(f"Best sequence: {best_sequence[:40]}{'...' if len(best_sequence) > 40 else ''}")
    
    return {
        'strategy': strategy,
        'reward': best_reward,
        'sequence': best_sequence,
        'analysis': analysis,
        'mask_percentage': mask_percentage,
        'masked_positions': mask_positions
    }


def test_protein_with_all_strategies(protein: Dict) -> Dict:
    """Test one protein with all masking strategies."""
    print(f"\n🧬 Testing {protein['name']} ({protein['target_length']} residues)")
    print("=" * 80)
    
    # Add plDDT scores to structure
    plddt_scores = compute_plddt_from_structure(protein)
    protein['plddt_scores'] = plddt_scores
    avg_plddt = np.mean(plddt_scores)
    low_conf_count = sum(1 for score in plddt_scores if score < 0.7)
    print(f"plDDT: avg={avg_plddt:.3f}, low_conf={low_conf_count}/{protein['target_length']} ({low_conf_count/protein['target_length']*100:.1f}%)")
    
    strategies = ["baseline", "no_masking", "random_masking", "plddt_masking"]
    results = {}
    
    for strategy in strategies:
        try:
            result = test_masking_strategy(protein, strategy, num_simulations=15)
            results[strategy] = result
        except Exception as e:
            print(f"❌ {strategy} failed: {e}")
            results[strategy] = None
    
    return results


def main():
    """Run comprehensive masking strategy comparison."""
    print("🧬 MCTS Masking Strategy Comparison")
    print("=" * 80)
    print("Testing: Baseline vs No Masking vs Random Masking vs plDDT Masking")
    print("Using real protein structures from PDB")
    
    # Load real proteins
    proteins = get_test_proteins()
    
    # Select a few proteins for testing (to keep runtime reasonable)
    test_proteins = proteins[:3]  # Test first 3 proteins
    
    all_results = {}
    
    for protein in test_proteins:
        results = test_protein_with_all_strategies(protein)
        all_results[protein['name']] = results
    
    # Summary comparison
    print(f"\n📈 Summary Comparison")
    print("=" * 100)
    
    print(f"{'Protein':<15} {'Strategy':<15} {'Reward':<10} {'Improvement':<12} {'Masking %':<10} {'Notes':<20}")
    print("-" * 100)
    
    for protein_name, results in all_results.items():
        baseline_reward = results['baseline']['reward'] if results['baseline'] else 0.0
        
        for strategy in ["baseline", "no_masking", "random_masking", "plddt_masking"]:
            if results[strategy]:
                result = results[strategy]
                reward = result['reward']
                improvement = ((reward - baseline_reward) / baseline_reward * 100) if baseline_reward > 0 else 0
                mask_pct = result['mask_percentage']
                
                if strategy == "baseline":
                    notes = "Reference"
                elif strategy == "no_masking":
                    notes = "Full sequence"
                elif strategy == "random_masking":
                    notes = "Random positions"
                else:  # plddt_masking
                    notes = "Low confidence"
                
                print(f"{protein_name:<15} {strategy:<15} {reward:<10.4f} {improvement:<12.1f}% {mask_pct:<10.1f}% {notes:<20}")
            else:
                print(f"{protein_name:<15} {strategy:<15} {'FAILED':<10} {'-':<12} {'-':<10} {'-':<20}")
    
    # Strategy-wise analysis
    print(f"\n🎯 Strategy Performance Analysis")
    print("=" * 60)
    
    strategy_improvements = {strategy: [] for strategy in ["no_masking", "random_masking", "plddt_masking"]}
    
    for protein_name, results in all_results.items():
        if results['baseline']:
            baseline_reward = results['baseline']['reward']
            for strategy in strategy_improvements.keys():
                if results[strategy]:
                    reward = results[strategy]['reward']
                    improvement = ((reward - baseline_reward) / baseline_reward * 100)
                    strategy_improvements[strategy].append(improvement)
    
    for strategy, improvements in strategy_improvements.items():
        if improvements:
            avg_imp = np.mean(improvements)
            max_imp = np.max(improvements)
            min_imp = np.min(improvements)
            print(f"{strategy:>15}: Avg={avg_imp:+6.1f}%, Max={max_imp:+6.1f}%, Min={min_imp:+6.1f}%, Proteins={len(improvements)}")
    
    print(f"\n✅ Masking Strategy Comparison Completed!")


if __name__ == "__main__":
    main()
