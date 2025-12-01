"""
Test integration of hallucination expert with existing MCTS.

This demonstrates how to plug the AF3+ProteinMPNN hallucination expert
into the existing GeneralMCTS framework.
"""

import sys
import os

# Add paths
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_diffusion_finetune')
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination')

from core.hallucination_expert import create_hallucination_expert


def test_hallucination_expert_standalone():
    """Test hallucination expert standalone (without MCTS)."""
    print("\n" + "="*80)
    print("Test 1: Hallucination Expert Standalone")
    print("="*80 + "\n")
    
    # Create expert
    expert = create_hallucination_expert()
    
    # Test sequence
    test_sequence = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues
    masked_positions = {10, 11, 12, 20, 21, 22}  # Mask 6 positions
    
    print(f"Test sequence length: {len(test_sequence)}")
    print(f"Masked positions: {masked_positions}")
    
    # Generate candidate
    candidate = expert.generate_candidate(
        sequence=test_sequence,
        masked_positions=masked_positions
    )
    
    if candidate:
        print(f"\n✅ Candidate generated:")
        print(f"   Sequence length: {len(candidate['sequence'])}")
        print(f"   Sequence: {candidate['sequence'][:50]}...")
        print(f"   Mean pLDDT: {candidate['mean_plddt']:.1f}")
        print(f"   Entropy: {candidate['entropy']:.3f}")
        print(f"   Coordinates shape: {candidate['coordinates'].shape}")
        return True
    else:
        print(f"\n❌ Failed to generate candidate")
        return False


def test_hallucination_expert_with_mcts():
    """
    Test hallucination expert integrated with MCTS.
    
    NOTE: This requires the existing MCTS to be modified to call external experts.
    Currently, the MCTS only calls DPLM-2 and ProteinMPNN (hardcoded).
    
    To fully integrate, we need to modify _expand_with_multi_expert_rollouts
    to iterate over self.external_experts and call their generate_candidate method.
    """
    print("\n" + "="*80)
    print("Test 2: Hallucination Expert with MCTS Integration")
    print("="*80 + "\n")
    
    print("⚠️  Integration requires modification to GeneralMCTS._expand_with_multi_expert_rollouts")
    print("    to call external expert.generate_candidate() method.")
    print("\nProposed integration code:")
    print("""
    # In _expand_with_multi_expert_rollouts, after DPLM-2 and ProteinMPNN:
    
    # External experts (e.g., hallucination)
    for expert in self.external_experts:
        if hasattr(expert, 'generate_candidate'):
            print(f"      🤖 {expert.get_name()}: generating {self.num_rollouts_per_expert} rollouts")
            
            for rollout in range(self.num_rollouts_per_expert):
                try:
                    candidate_data = expert.generate_candidate(
                        sequence=node.sequence,
                        masked_positions=node.masked_positions,
                        coordinates=node.coordinates
                    )
                    
                    if candidate_data:
                        # Evaluate reward
                        if self.task_type == "folding":
                            reward = self._evaluate_structure_reward(
                                candidate_data['coordinates'], 
                                candidate_data['sequence']
                            )
                        else:
                            reward = self._evaluate_sequence_aar(candidate_data['sequence'])
                        
                        candidate_data['reward'] = reward
                        candidate_data['rollout_id'] = rollout
                        all_candidates.append(candidate_data)
                        
                        print(f"         ✅ {expert.get_name()} rollout {rollout+1}: reward={reward:.3f}")
                    
                except Exception as e:
                    print(f"         ❌ {expert.get_name()} rollout {rollout+1}: error {e}")
    """)
    
    return True


def show_usage_example():
    """Show how to use hallucination expert with MCTS."""
    print("\n" + "="*80)
    print("Usage Example")
    print("="*80 + "\n")
    
    print("""
# 1. Create hallucination expert
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert

hallucination_expert = create_hallucination_expert(
    abcfold_path="/path/to/ABCFold",
    proteinmpnn_path="/path/to/proteinmpnn"
)

# 2. Initialize MCTS with hallucination expert
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS
from mcts_diffusion_finetune.core.dplm2_integration import DPLM2Integration

dplm2 = DPLM2Integration(...)

mcts = GeneralMCTS(
    dplm2_integration=dplm2,
    external_experts=[hallucination_expert],  # Add hallucination expert
    ablation_mode="single_expert",
    single_expert_id=3,  # Use external expert (index 3)
    num_rollouts_per_expert=2,
    top_k_candidates=2
)

# 3. Run MCTS search
result = mcts.search(
    initial_sequence=baseline_sequence,
    num_iterations=5
)

# The hallucination expert will:
# - Take the current sequence with masked positions
# - Run AF3 to hallucinate structure
# - Run ProteinMPNN to design sequence
# - Return candidate for MCTS evaluation
    """)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Hallucination Expert Integration Tests")
    print("="*80)
    
    # Test 1: Standalone expert
    success1 = test_hallucination_expert_standalone()
    
    # Test 2: MCTS integration (shows what's needed)
    success2 = test_hallucination_expert_with_mcts()
    
    # Show usage
    show_usage_example()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Standalone test: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Integration guide: {'✅ SHOWN' if success2 else '❌ FAIL'}")
    print("="*80 + "\n")
    
    sys.exit(0 if success1 else 1)
