"""
Demonstration of MCTS simulations for different protein lengths.

This script provides a clear, easy-to-understand demonstration of how
MCTS performs on proteins of different sizes, with detailed explanations
of the node structure and simulation process.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Tuple
import time

from core.sequence_level_mcts import SequenceLevelMCTS, MCTSNode
from utils.protein_utils import create_mock_structure_no_sequence
from utils.reward_computation import compute_detailed_reward_analysis


def demonstrate_mcts_node_structure():
    """Demonstrate how MCTS nodes are structured and organized."""
    
    print("🧬 MCTS Node Structure Demonstration")
    print("=" * 50)
    
    # Create example nodes
    root = MCTSNode(sequence="", reward=0.0)
    
    # Add some example children
    child1 = MCTSNode(sequence="ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL", reward=0.65)
    child1.visit_count = 5
    child1.total_value = 3.25
    
    child2 = MCTSNode(sequence="MLKIHGFEDCBAPQRSTVWYMLKIHGFEDCBAPQRSTVWYMLKIHGFEDC", reward=0.72)
    child2.visit_count = 8
    child2.total_value = 5.76
    
    root.children = [child1, child2]
    
    print("Root Node:")
    print(f"  Sequence: '{root.sequence}' (empty - starting point)")
    print(f"  Reward: {root.reward}")
    print(f"  Children: {len(root.children)}")
    
    print("\nChild Nodes:")
    for i, child in enumerate(root.children, 1):
        print(f"  Child {i}:")
        print(f"    Sequence: {child.sequence[:20]}... (length: {len(child.sequence)})")
        print(f"    Reward: {child.reward:.3f}")
        print(f"    Visit count: {child.visit_count}")
        print(f"    Total value: {child.total_value:.3f}")
        print(f"    Average value: {child.average_value:.3f}")
        print(f"    UCB score: {child.ucb_score:.3f}")
        print()
    
    print("🔍 Node Selection Process:")
    print("1. Start at root node")
    print("2. Calculate UCB scores for all children")
    print("3. Select child with highest UCB score")
    print("4. Repeat until reaching a leaf node")
    
    best_child = max(root.children, key=lambda x: x.ucb_score)
    print(f"\nSelected node: Child with UCB score {best_child.ucb_score:.3f}")


def demonstrate_simulation_process(protein_length: int, num_simulations: int = 20):
    """Demonstrate the MCTS simulation process step by step."""
    
    print(f"\n🔄 MCTS Simulation Process - {protein_length} Residues")
    print("=" * 60)
    
    # Create structure and mock components
    structure = create_mock_structure_no_sequence(length=protein_length)
    
    class MockModel:
        def to(self, device): return self
        def eval(self): return self
    
    class MockTokenizer:
        def encode(self, sequence): return list(range(len(sequence)))
    
    # Initialize MCTS
    mcts = SequenceLevelMCTS(
        model=MockModel(),
        tokenizer=MockTokenizer(),
        max_depth=3,
        num_simulations=num_simulations,
        exploration_constant=1.414,
        temperature=1.0,
        num_candidates_per_expansion=3
    )
    
    print(f"📋 MCTS Configuration:")
    print(f"  Max depth: {mcts.max_depth}")
    print(f"  Num simulations: {mcts.num_simulations}")
    print(f"  Exploration constant: {mcts.exploration_constant}")
    print(f"  Temperature: {mcts.temperature}")
    
    print(f"\n🎯 Target: Generate {protein_length}-residue protein sequence")
    print(f"Structure info: {protein_length} backbone coordinates (no sequence)")
    
    # Demonstrate the 4 phases of MCTS
    print("\n🔍 MCTS Phases:")
    print("1. SELECTION: Use UCB1 to select most promising node")
    print("   Formula: value + c * sqrt(ln(parent_visits) / node_visits)")
    print("2. EXPANSION: Generate new candidate sequences")
    print("   Methods: Point mutations, DPLM-2 generation, variations")
    print("3. SIMULATION: Evaluate sequence using reward function")
    print("   Components: Structure compatibility, biophysical properties")
    print("4. BACKPROPAGATION: Update node statistics")
    print("   Updates: Visit counts, total values, average values")
    
    # Run a few simulations with detailed output
    print(f"\n🚀 Running {num_simulations} simulations...")
    
    start_time = time.time()
    best_sequence, best_reward = mcts.search(structure, protein_length)
    end_time = time.time()
    
    print(f"\n✅ Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Best sequence found: {best_sequence[:30]}...")
    print(f"Best reward: {best_reward:.4f}")
    
    # Analyze the best sequence
    detailed_analysis = compute_detailed_reward_analysis(best_sequence, structure)
    
    print(f"\n📊 Detailed Analysis of Best Sequence:")
    print(f"  Length: {detailed_analysis['length']}")
    print(f"  Total reward: {detailed_analysis['total_reward']:.4f}")
    print(f"  Structure compatibility: {detailed_analysis['structure_compatibility']:.4f}")
    print(f"  Hydrophobicity balance: {detailed_analysis['hydrophobicity_balance']:.4f}")
    print(f"  Charge balance: {detailed_analysis['charge_balance']:.4f}")
    print(f"  Sequence diversity: {detailed_analysis['sequence_diversity']:.4f}")
    print(f"  Length category: {detailed_analysis['length_category']}")
    
    return best_sequence, detailed_analysis


def compare_protein_lengths():
    """Compare MCTS performance across different protein lengths."""
    
    print("\n🔬 Protein Length Comparison")
    print("=" * 50)
    
    lengths = [50, 200, 500]
    results = {}
    
    for length in lengths:
        print(f"\n📏 Testing {length}-residue protein:")
        
        # Adjust simulation count based on length
        if length <= 50:
            num_sims = 30
            focus = "Local structure optimization"
        elif length <= 200:
            num_sims = 50
            focus = "Balanced local/global optimization"
        else:
            num_sims = 80
            focus = "Global structure compatibility"
        
        print(f"  Focus: {focus}")
        print(f"  Simulations: {num_sims}")
        
        # Run simulation
        start_time = time.time()
        best_seq, analysis = demonstrate_simulation_process(length, num_sims)
        end_time = time.time()
        
        results[length] = {
            'best_sequence': best_seq,
            'analysis': analysis,
            'simulation_time': end_time - start_time,
            'focus': focus
        }
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Reward: {analysis['total_reward']:.4f}")
    
    # Summary comparison
    print(f"\n📈 Summary Comparison:")
    print(f"{'Length':<8} {'Time(s)':<8} {'Reward':<8} {'Focus':<30}")
    print("-" * 60)
    
    for length in lengths:
        result = results[length]
        print(f"{length:<8} {result['simulation_time']:<8.2f} {result['analysis']['total_reward']:<8.4f} {result['focus']:<30}")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    print(f"• Small proteins (50): Fast convergence, emphasis on local stability")
    print(f"• Medium proteins (200): Balanced approach, good exploration")
    print(f"• Large proteins (500): Slower but thorough, global structure focus")
    
    return results


def explain_reward_calculation():
    """Explain how rewards are calculated for different protein lengths."""
    
    print("\n🏆 Reward Calculation Explanation")
    print("=" * 50)
    
    # Create example sequences
    sequences = {
        50: "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKL",
        200: "ACDEFGHIKLMNPQRSTVWY" * 10,  # 200 residues
        500: "ACDEFGHIKLMNPQRSTVWY" * 25   # 500 residues
    }
    
    print("Reward components and their weights by protein length:")
    print()
    
    for length, sequence in sequences.items():
        structure = create_mock_structure_no_sequence(length=length)
        analysis = compute_detailed_reward_analysis(sequence, structure)
        
        print(f"📏 {length}-residue protein:")
        print(f"  Length category: {analysis['length_category']}")
        print(f"  Reward weights: {analysis['weights']}")
        print(f"  Component scores:")
        print(f"    Structure compatibility: {analysis['structure_compatibility']:.4f}")
        print(f"    Hydrophobicity balance: {analysis['hydrophobicity_balance']:.4f}")
        print(f"    Charge balance: {analysis['charge_balance']:.4f}")
        print(f"    Sequence diversity: {analysis['sequence_diversity']:.4f}")
        print(f"    Stability score: {analysis['stability_score']:.4f}")
        print(f"  Total reward: {analysis['total_reward']:.4f}")
        print()
    
    print("💡 Key Principles:")
    print("• Small proteins: Higher weight on local properties (hydrophobicity, charge)")
    print("• Medium proteins: Balanced weighting across all components")
    print("• Large proteins: Higher weight on global structure compatibility")
    print("• Length scaling: Penalties/bonuses adjusted for protein size")


def main():
    """Run the complete MCTS demonstration."""
    
    print("🧬 MCTS-Guided Inverse Folding Demonstration")
    print("=" * 70)
    print("This demonstration shows how MCTS works for protein inverse folding")
    print("with detailed explanations of node structure and simulation process.")
    
    # 1. Demonstrate node structure
    demonstrate_mcts_node_structure()
    
    # 2. Demonstrate simulation process
    demonstrate_simulation_process(protein_length=100, num_simulations=25)
    
    # 3. Compare different protein lengths
    compare_protein_lengths()
    
    # 4. Explain reward calculation
    explain_reward_calculation()
    
    print("\n🎉 Demonstration Complete!")
    print("=" * 50)
    print("Summary of MCTS Process:")
    print("1. Initialize tree with root node (empty sequence)")
    print("2. Generate initial candidate sequences as children")
    print("3. For each simulation:")
    print("   a. SELECT: Choose node with highest UCB score")
    print("   b. EXPAND: Generate new sequence variations")
    print("   c. SIMULATE: Evaluate sequence with reward function")
    print("   d. BACKPROPAGATE: Update node statistics")
    print("4. Return best sequence found")
    print()
    print("Key Adaptations by Protein Length:")
    print("• Small (50): Fast, local optimization, higher exploration")
    print("• Medium (200): Balanced approach, moderate depth")
    print("• Large (500): Deeper search, global structure focus")


if __name__ == "__main__":
    main() 