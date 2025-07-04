"""
Test script to show initial DPLM-2 sequence generation and rewards.
This helps us understand the improvement from MCTS refinement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from protein_utils import create_mock_structure_no_sequence
from dplm_inverse_folding import generate_sequence_from_structure, evaluate_sequence_structure_compatibility
from protein_utils import compute_structure_metrics

def test_initial_rewards():
    """Test initial DPLM-2 sequence generation and their rewards."""
    
    # Create mock structure (same as in main pipeline)
    print("=== Creating Mock Structure ===")
    structure = create_mock_structure_no_sequence(length=50)
    print(f"Structure: {structure['length']} residues, backbone coords shape: {structure['backbone_coords'].shape}")
    print()
    
    # Generate initial sequences using DPLM-2 (placeholder)
    print("=== Initial DPLM-2 Sequence Generation ===")
    initial_sequences = generate_sequence_from_structure(
        model=None,  # Placeholder
        tokenizer=None,  # Placeholder
        structure=structure,
        num_samples=10,
        temperature=1.0
    )
    print()
    
    # Evaluate each sequence
    print("=== Initial Sequence Rewards ===")
    rewards = []
    
    for i, sequence in enumerate(initial_sequences):
        # Compute structure-sequence compatibility
        compatibility = evaluate_sequence_structure_compatibility(sequence, structure)
        
        # Compute biophysical metrics
        metrics = compute_structure_metrics(sequence, structure)
        
        # Calculate total reward (same as in MCTS)
        reward = 0.0
        reward += compatibility * 0.5  # Structure compatibility
        if 10 <= len(sequence) <= 500:
            reward += 0.1  # Length reward
        if -2.0 <= metrics['hydrophobicity'] <= 2.0:
            reward += 0.2  # Hydrophobicity reward
        if abs(metrics['charge']) <= 10:
            reward += 0.1  # Charge reward
        
        # Diversity reward
        unique_aas = len(set(sequence))
        diversity_score = unique_aas / len(sequence)
        reward += diversity_score * 0.1
        
        rewards.append(reward)
        
        print(f"Sequence {i+1}: {sequence[:30]}...")
        print(f"  Compatibility: {compatibility:.3f}")
        print(f"  Hydrophobicity: {metrics['hydrophobicity']:.3f}")
        print(f"  Charge: {metrics['charge']}")
        print(f"  Diversity: {diversity_score:.3f}")
        print(f"  Total Reward: {reward:.3f}")
        print()
    
    # Summary
    print("=== Summary ===")
    print(f"Average initial reward: {sum(rewards)/len(rewards):.3f}")
    print(f"Best initial reward: {max(rewards):.3f}")
    print(f"Worst initial reward: {min(rewards):.3f}")
    print(f"Reward range: {max(rewards) - min(rewards):.3f}")
    
    return initial_sequences, rewards

if __name__ == "__main__":
    test_initial_rewards() 