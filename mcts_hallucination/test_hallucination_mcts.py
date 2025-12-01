"""
Test script for hallucination-based MCTS.

This demonstrates the key difference from discrete diffusion:
- Start with all-mask (not baseline)
- AF3 hallucinates structures
- ProteinMPNN designs sequences
- MCTS guides exploration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.hallucination_mcts_simple import HallucinationMCTS, HallucinationNode
from core.abcfold_integration import ABCFoldIntegration
from core.proteinmpnn_integration import ProteinMPNNIntegration


def test_hallucination_mcts():
    """Test hallucination MCTS pipeline."""
    print("\n" + "="*80)
    print("Testing Hallucination-based MCTS")
    print("="*80 + "\n")
    
    # Initialize integrations (will use mock mode for now)
    abcfold = ABCFoldIntegration()
    proteinmpnn = ProteinMPNNIntegration()
    
    # Initialize MCTS
    mcts = HallucinationMCTS(
        target_length=50,  # Small test protein
        abcfold_integration=abcfold,
        proteinmpnn_integration=proteinmpnn,
        max_depth=3,
        num_iterations=5,
        num_rollouts_per_iteration=2,
        top_k_candidates=2,
        use_ph_uct=True,
        initial_mask_ratio=1.0,  # Start with all-mask
        min_mask_ratio=0.2,
        confidence_threshold=70.0
    )
    
    # Run search
    best_node = mcts.search()
    
    if best_node:
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Best sequence: {best_node.sequence[:50]}...")
        print(f"Structure quality: {best_node.structure_quality:.3f}")
        print(f"Reward: {best_node.reward:.3f}")
        print(f"Depth: {best_node.depth}")
        print(f"Visits: {best_node.visits}")
        print(f"Mean confidence: {sum(best_node.confidence_scores)/len(best_node.confidence_scores):.1f}")
        print("="*80 + "\n")
        
        return True
    else:
        print("\n❌ Search failed\n")
        return False


if __name__ == "__main__":
    success = test_hallucination_mcts()
    sys.exit(0 if success else 1)
