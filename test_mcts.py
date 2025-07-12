#!/usr/bin/env python3
"""
Simple test script to demonstrate MCTS-guided inverse folding.

This script shows how the sequence-level MCTS works with different protein lengths
and provides detailed analysis of the results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sequence_level_mcts import SequenceLevelMCTS, MCTSNode
from utils.protein_utils import create_mock_structure_no_sequence
from utils.reward_computation import compute_detailed_reward_analysis
import torch


def create_mock_components():
    """Create mock model and tokenizer for testing."""
    class MockModel:
        def __init__(self):
            self.device = torch.device('cpu')
        def to(self, device):
            return self
        def eval(self):
            return self

    class MockTokenizer:
        def __init__(self):
            self.vocab = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    return MockModel(), MockTokenizer()


def test_mcts_for_length(length: int, num_simulations: int = 30):
    """Test MCTS for a specific protein length."""
    print(f"\n🧬 Testing {length}-residue protein")
    print("=" * 50)
    
    # Create structure
    structure = create_mock_structure_no_sequence(length=length)
    print(f"Created structure for {length} residues")
    
    # Create components
    model, tokenizer = create_mock_components()
    
    # Configure MCTS based on length
    if length < 100:
        max_depth, exploration = 3, 1.8
        focus = "Local optimization"
    elif length < 300:
        max_depth, exploration = 5, 1.414
        focus = "Balanced approach"
    else:
        max_depth, exploration = 7, 1.0
        focus = "Global structure"
    
    # Initialize MCTS
    mcts = SequenceLevelMCTS(
        model=model,
        tokenizer=tokenizer,
        max_depth=max_depth,
        num_simulations=num_simulations,
        exploration_constant=exploration,
        temperature=1.0,
        num_candidates_per_expansion=3
    )
    
    print(f"MCTS Configuration:")
    print(f"  Max depth: {max_depth}")
    print(f"  Simulations: {num_simulations}")
    print(f"  Exploration: {exploration}")
    print(f"  Focus: {focus}")
    
    # Run search
    print(f"\n🚀 Running MCTS search...")
    best_sequence, best_reward = mcts.search(structure, target_length=length)
    
    # Analyze results
    analysis = compute_detailed_reward_analysis(best_sequence, structure)
    
    print(f"\n✅ Results:")
    print(f"  Best sequence: {best_sequence[:40]}{'...' if len(best_sequence) > 40 else ''}")
    print(f"  Sequence length: {len(best_sequence)}")
    print(f"  Best reward: {best_reward:.4f}")
    
    print(f"\n📊 Detailed Analysis:")
    print(f"  Total reward: {analysis['total_reward']:.4f}")
    print(f"  Structure compatibility: {analysis['structure_compatibility']:.4f}")
    print(f"  Hydrophobicity balance: {analysis['hydrophobicity_balance']:.4f}")
    print(f"  Charge balance: {analysis['charge_balance']:.4f}")
    print(f"  Sequence diversity: {analysis['sequence_diversity']:.4f}")
    print(f"  Length category: {analysis['length_category']}")
    print(f"  Reward weights: {analysis['weights']}")
    
    return best_sequence, best_reward, analysis


def demonstrate_tree_structure():
    """Demonstrate the MCTS tree structure."""
    print("\n🌳 MCTS Tree Structure Explanation")
    print("=" * 60)
    
    print("Actual Tree Structure (based on implementation):")
    print("```")
    print("Root Node (empty sequence \"\")")
    print("├── Child 1 (DPLM-2 generated sequence A)")
    print("│   ├── Grandchild 1.1 (Point mutation: pos 5 A→V)")
    print("│   ├── Grandchild 1.2 (Point mutation: pos 12 L→F)")
    print("│   └── Grandchild 1.3 (Point mutation: pos 23 K→R)")
    print("├── Child 2 (DPLM-2 generated sequence B)")
    print("│   ├── Grandchild 2.1 (Point mutation: pos 8 D→E)")
    print("│   └── Grandchild 2.2 (Point mutation: pos 15 G→A)")
    print("└── Child 3 (DPLM-2 generated sequence C)")
    print("    ├── Grandchild 3.1 (Point mutation: pos 3 M→I)")
    print("    └── ...")
    print("```")
    
    print("\n🔄 MCTS Process:")
    print("1. **Initialization**: Root with empty sequence")
    print("2. **Level 1**: DPLM-2 generates initial complete sequences")
    print("3. **Level 2+**: Point mutations (1-3 amino acid changes)")
    print("4. **Selection**: UCB1 chooses most promising nodes")
    print("5. **Expansion**: Generate sequence variations")
    print("6. **Simulation**: Evaluate using length-aware reward")
    print("7. **Backpropagation**: Update node statistics")


def main():
    """Run the complete MCTS demonstration."""
    print("🧬 MCTS-Guided Inverse Folding Test")
    print("=" * 70)
    print("This test demonstrates sequence-level MCTS for different protein lengths")
    
    # Demonstrate tree structure
    demonstrate_tree_structure()
    
    # Test different protein lengths
    test_lengths = [50, 200, 500]
    results = {}
    
    for length in test_lengths:
        # Adjust simulation count based on length
        num_sims = min(50, max(20, length // 10))
        
        try:
            seq, reward, analysis = test_mcts_for_length(length, num_sims)
            results[length] = {
                'sequence': seq,
                'reward': reward,
                'analysis': analysis
            }
        except Exception as e:
            print(f"❌ Error testing {length}-residue protein: {e}")
            continue
    
    # Summary
    print(f"\n📈 Summary Comparison:")
    print(f"{'Length':<8} {'Reward':<8} {'Category':<10} {'Time Focus':<20}")
    print("-" * 55)
    
    for length in test_lengths:
        if length in results:
            result = results[length]
            category = result['analysis']['length_category']
            reward = result['reward']
            
            if length < 100:
                focus = "Local optimization"
            elif length < 300:
                focus = "Balanced approach"
            else:
                focus = "Global structure"
            
            print(f"{length:<8} {reward:<8.4f} {category:<10} {focus:<20}")
    
    print(f"\n🎯 Key Insights:")
    print("• Small proteins: Fast convergence, local stability focus")
    print("• Medium proteins: Balanced exploration, moderate complexity")
    print("• Large proteins: Thorough search, global structure emphasis")
    print("• Tree structure: Root → DPLM-2 → Point mutations")
    print("• Each node represents a complete amino acid sequence")
    
    print(f"\n✅ MCTS Test Completed Successfully!")


if __name__ == "__main__":
    main() 