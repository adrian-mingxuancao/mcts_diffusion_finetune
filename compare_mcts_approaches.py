"""
Comparison of Sequence-Level vs Position-Level MCTS for Inverse Folding

This script runs both MCTS approaches on the same protein structure
and compares their performance and results.
"""

import time
import random
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from sequence_level_mcts import SequenceLevelMCTS
from position_level_mcts import PositionLevelMCTS
from protein_utils import create_mock_structure_no_sequence, compute_structure_metrics


class MCTSComparison:
    """Compare different MCTS approaches for inverse folding."""
    
    def __init__(self):
        # Create stub model and tokenizer for both approaches
        self.model = self._create_stub_model()
        self.tokenizer = self._create_stub_tokenizer()
    
    def _create_stub_model(self):
        """Create a stub model for testing."""
        class StubModel:
            def __init__(self):
                self.device = torch.device('cpu')
            
            def to(self, device):
                self.device = device
                return self
            
            def eval(self):
                return self
        
        return StubModel()
    
    def _create_stub_tokenizer(self):
        """Create a stub tokenizer for testing."""
        class StubTokenizer:
            def __init__(self):
                self.vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        return StubTokenizer()
    
    def run_sequence_level_mcts(self, structure: Dict, target_length: int) -> Tuple[str, float, float]:
        """Run sequence-level MCTS and return results with timing."""
        print("\n" + "="*60)
        print("RUNNING SEQUENCE-LEVEL MCTS")
        print("="*60)
        
        start_time = time.time()
        
        mcts = SequenceLevelMCTS(
            model=self.model,
            tokenizer=self.tokenizer,
            max_depth=5,
            num_simulations=50,
            exploration_constant=1.414,
            temperature=1.0
        )
        
        best_sequence, best_reward = mcts.search(structure, target_length)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Sequence-Level MCTS Results:")
        print(f"  Best sequence: {best_sequence}")
        print(f"  Best reward: {best_reward:.3f}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        
        return best_sequence, best_reward, elapsed_time
    
    def run_position_level_mcts(self, structure: Dict, target_length: int) -> Tuple[str, float, float]:
        """Run position-level MCTS and return results with timing."""
        print("\n" + "="*60)
        print("RUNNING POSITION-LEVEL MCTS WITH PLDDT MASKING")
        print("="*60)
        
        start_time = time.time()
        
        mcts = PositionLevelMCTS(
            model=self.model,
            tokenizer=self.tokenizer,
            max_depth=10,
            num_simulations=100,
            exploration_constant=1.414,
            temperature=1.0,
            plddt_threshold=0.7,
            max_unmask_per_step=3
        )
        
        best_sequence, best_reward = mcts.search(structure, target_length)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Position-Level MCTS Results:")
        print(f"  Best sequence: {best_sequence}")
        print(f"  Best reward: {best_reward:.3f}")
        print(f"  Time elapsed: {elapsed_time:.2f} seconds")
        
        return best_sequence, best_reward, elapsed_time
    
    def analyze_sequences(self, seq1: str, seq2: str, structure: Dict):
        """Analyze and compare the generated sequences."""
        print("\n" + "="*60)
        print("SEQUENCE ANALYSIS")
        print("="*60)
        
        # Compute metrics for both sequences
        metrics1 = compute_structure_metrics(seq1, structure)
        metrics2 = compute_structure_metrics(seq2, structure)
        
        print(f"Sequence-Level MCTS Sequence:")
        print(f"  Sequence: {seq1}")
        print(f"  Length: {len(seq1)}")
        print(f"  Hydrophobicity: {metrics1['hydrophobicity']:.3f}")
        print(f"  Charge: {metrics1['charge']:.3f}")
        print(f"  Diversity: {metrics1['diversity']:.3f}")
        
        print(f"\nPosition-Level MCTS Sequence:")
        print(f"  Sequence: {seq2}")
        print(f"  Length: {len(seq2)}")
        print(f"  Hydrophobicity: {metrics2['hydrophobicity']:.3f}")
        print(f"  Charge: {metrics2['charge']:.3f}")
        print(f"  Diversity: {metrics2['diversity']:.3f}")
        
        # Compare sequences
        print(f"\nComparison:")
        print(f"  Sequence similarity: {self._compute_sequence_similarity(seq1, seq2):.3f}")
        print(f"  Length difference: {abs(len(seq1) - len(seq2))}")
        
        return metrics1, metrics2
    
    def _compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute similarity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def run_comparison(self, target_length: int = 50) -> Dict:
        """Run a complete comparison of both MCTS approaches."""
        print("MCTS APPROACHES COMPARISON")
        print("="*60)
        
        # Create test structure
        structure = create_mock_structure_no_sequence(length=target_length)
        print(f"Created test structure with {target_length} residues")
        
        # Run both approaches
        seq1, reward1, time1 = self.run_sequence_level_mcts(structure, target_length)
        seq2, reward2, time2 = self.run_position_level_mcts(structure, target_length)
        
        # Analyze results
        metrics1, metrics2 = self.analyze_sequences(seq1, seq2, structure)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"Sequence-Level MCTS:")
        print(f"  Reward: {reward1:.3f}")
        print(f"  Time: {time1:.2f}s")
        print(f"  Approach: Full sequence generation and variation")
        
        print(f"\nPosition-Level MCTS:")
        print(f"  Reward: {reward2:.3f}")
        print(f"  Time: {time2:.2f}s")
        print(f"  Approach: Position-by-position unmasking with plDDT guidance")
        
        print(f"\nPerformance Comparison:")
        if reward1 > reward2:
            print(f"  Sequence-Level MCTS achieved higher reward (+{reward1 - reward2:.3f})")
        elif reward2 > reward1:
            print(f"  Position-Level MCTS achieved higher reward (+{reward2 - reward1:.3f})")
        else:
            print(f"  Both approaches achieved similar rewards")
        
        if time1 < time2:
            print(f"  Sequence-Level MCTS was faster (-{time2 - time1:.2f}s)")
        elif time2 < time1:
            print(f"  Position-Level MCTS was faster (-{time1 - time2:.2f}s)")
        else:
            print(f"  Both approaches took similar time")
        
        return {
            'sequence_level': {
                'sequence': seq1,
                'reward': reward1,
                'time': time1,
                'metrics': metrics1
            },
            'position_level': {
                'sequence': seq2,
                'reward': reward2,
                'time': time2,
                'metrics': metrics2
            },
            'structure': structure
        }


def run_multiple_comparisons(num_trials: int = 3):
    """Run multiple comparison trials to get more robust results."""
    print(f"Running {num_trials} comparison trials...")
    
    comparison = MCTSComparison()
    results = []
    
    for trial in range(num_trials):
        print(f"\n{'='*20} TRIAL {trial + 1} {'='*20}")
        
        # Set random seed for reproducibility
        random.seed(42 + trial)
        np.random.seed(42 + trial)
        
        result = comparison.run_comparison(target_length=50)
        results.append(result)
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS")
    print(f"{'='*60}")
    
    seq_rewards = [r['sequence_level']['reward'] for r in results]
    pos_rewards = [r['position_level']['reward'] for r in results]
    seq_times = [r['sequence_level']['time'] for r in results]
    pos_times = [r['position_level']['time'] for r in results]
    
    print(f"Sequence-Level MCTS (avg over {num_trials} trials):")
    print(f"  Average reward: {np.mean(seq_rewards):.3f} ± {np.std(seq_rewards):.3f}")
    print(f"  Average time: {np.mean(seq_times):.2f}s ± {np.std(seq_times):.2f}s")
    
    print(f"\nPosition-Level MCTS (avg over {num_trials} trials):")
    print(f"  Average reward: {np.mean(pos_rewards):.3f} ± {np.std(pos_rewards):.3f}")
    print(f"  Average time: {np.mean(pos_times):.2f}s ± {np.std(pos_times):.2f}s")
    
    # Statistical comparison
    reward_diff = np.mean(seq_rewards) - np.mean(pos_rewards)
    time_diff = np.mean(seq_times) - np.mean(pos_times)
    
    print(f"\nStatistical Comparison:")
    print(f"  Reward difference: {reward_diff:.3f} (positive = sequence-level better)")
    print(f"  Time difference: {time_diff:.2f}s (negative = sequence-level faster)")
    
    return results


if __name__ == "__main__":
    # Import torch for the stub model
    import torch
    
    # Run single comparison
    comparison = MCTSComparison()
    result = comparison.run_comparison(target_length=50)
    
    # Optionally run multiple trials
    # results = run_multiple_comparisons(num_trials=3) 