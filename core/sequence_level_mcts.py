"""
Sequence-Level MCTS for Inverse Folding with DPLM-2

This module implements Monte Carlo Tree Search at the sequence level,
where each node represents a complete candidate sequence.
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a complete sequence."""
    sequence: str
    reward: float
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_value(self) -> float:
        """Average value of this node."""
        return self.total_value / max(self.visit_count, 1)
    
    @property
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula: value + exploration_constant * sqrt(ln(parent_visits) / visits)
        parent_visits = sum(child.visit_count for child in self.children) if self.children else 1
        if parent_visits <= 0:
            return self.average_value
        
        log_term = math.log(parent_visits)
        if log_term <= 0:
            return self.average_value
            
        return self.average_value + exploration_constant * math.sqrt(log_term / self.visit_count)


class SequenceLevelMCTS:
    """
    Monte Carlo Tree Search for sequence-level inverse folding.
    
    Each node represents a complete candidate sequence. The search explores
    variations of sequences by generating new candidates from the model.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_depth: int = 5,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        num_candidates_per_expansion: int = 5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.num_candidates_per_expansion = num_candidates_per_expansion
        
        # Cache for generated sequences to avoid duplicates
        self.sequence_cache = set()
    
    def search(self, structure: Dict, target_length: int) -> Tuple[str, float]:
        """
        Perform MCTS search to find the best sequence for the given structure.
        
        Args:
            structure: Protein structure data
            target_length: Target sequence length
            
        Returns:
            Tuple of (best_sequence, best_reward)
        """
        print(f"Starting sequence-level MCTS search with {self.num_simulations} simulations...")
        
        # Initialize root node with empty sequence
        root = MCTSNode(sequence="", reward=0.0)
        
        # Generate initial candidates using DPLM-2
        initial_sequences = self._generate_initial_sequences(structure, target_length)
        
        # Create initial children
        for seq in initial_sequences:
            reward = self._compute_reward(seq, structure)
            child = MCTSNode(sequence=seq, reward=reward)
            root.children.append(child)
            self.sequence_cache.add(seq)
        
        # Perform MCTS simulations
        for i in range(self.num_simulations):
            if i % 10 == 0:
                print(f"  Simulation {i+1}/{self.num_simulations}")
            
            # Selection
            selected_node = self._select(root)
            
            # Expansion
            if selected_node.visit_count > 0 and len(selected_node.children) < self.max_depth:
                self._expand(selected_node, structure, target_length)
            
            # Simulation (evaluate the selected node)
            value = self._simulate(selected_node, structure)
            
            # Backpropagation
            self._backpropagate(selected_node, value)
        
        # Find best sequence
        best_child = max(root.children, key=lambda x: x.average_value)
        return best_child.sequence, best_child.average_value
    
    def _generate_initial_sequences(self, structure: Dict, target_length: int) -> List[str]:
        """Generate initial candidate sequences using DPLM-2."""
        sequences = []
        
        try:
            # Try to use the actual DPLM-2 model for generation
            for _ in range(self.num_candidates_per_expansion):
                # This is a placeholder - in practice, you'd use the actual DPLM-2 generation
                seq = self._generate_sequence_from_model(structure, target_length)
                if seq and seq not in self.sequence_cache:
                    sequences.append(seq)
        except Exception as e:
            print(f"Model generation failed: {e}")
            # Fallback: generate random sequences
            sequences = self._generate_random_sequences(target_length, self.num_candidates_per_expansion)
        
        return sequences
    
    def _generate_sequence_from_model(self, structure: Dict, target_length: int) -> str:
        """Generate a sequence using the DPLM-2 model."""
        # Placeholder implementation - replace with actual DPLM-2 generation
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        return ''.join(random.choices(amino_acids, k=target_length))
    
    def _generate_random_sequences(self, length: int, num_sequences: int) -> List[str]:
        """Generate random amino acid sequences as fallback."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequences = []
        
        for _ in range(num_sequences):
            seq = ''.join(random.choices(amino_acids, k=length))
            if seq not in self.sequence_cache:
                sequences.append(seq)
        
        return sequences
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using UCB1."""
        if not node.children:
            return node
        
        # Select child with highest UCB score
        best_child = max(node.children, key=lambda x: x.ucb_score)
        return self._select(best_child)
    
    def _expand(self, node: MCTSNode, structure: Dict, target_length: int):
        """Expand a node by generating new candidate sequences."""
        # Generate variations of the current sequence
        variations = self._generate_sequence_variations(node.sequence, target_length)
        
        for variation in variations:
            if variation not in self.sequence_cache:
                reward = self._compute_reward(variation, structure)
                child = MCTSNode(sequence=variation, reward=reward)
                node.children.append(child)
                self.sequence_cache.add(variation)
    
    def _generate_sequence_variations(self, sequence: str, target_length: int) -> List[str]:
        """Generate variations of a sequence."""
        variations = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Method 1: Point mutations
        for i in range(min(3, len(sequence))):  # Mutate up to 3 positions
            for aa in random.sample(amino_acids, 3):  # Try 3 different AAs
                if aa != sequence[i]:
                    new_seq = sequence[:i] + aa + sequence[i+1:]
                    variations.append(new_seq)
        
        # Method 2: Random generation (for empty sequences)
        if not sequence:
            for _ in range(2):
                variations.append(''.join(random.choices(amino_acids, k=target_length)))
        
        return variations[:self.num_candidates_per_expansion]
    
    def _simulate(self, node: MCTSNode, structure: Dict) -> float:
        """Simulate by evaluating the node's sequence."""
        return node.reward
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the simulation result up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            # Move to parent (simplified - in practice you'd need parent pointers)
            break
    
    def _compute_reward(self, sequence: str, structure: Dict) -> float:
        """Compute reward for a sequence given the structure."""
        from utils.protein_utils import compute_structure_metrics
        
        if not sequence:
            return 0.0
        
        # Compute structure-based metrics
        metrics = compute_structure_metrics(sequence, structure)
        
        # Simple reward based on sequence properties
        reward = 0.0
        
        # Reward for reasonable length
        if 20 <= len(sequence) <= 200:
            reward += 0.3
        
        # Reward for balanced hydrophobicity
        if -1.0 <= metrics['hydrophobicity'] <= 1.0:
            reward += 0.3
        
        # Reward for reasonable charge
        if abs(metrics['charge']) <= 5:
            reward += 0.2
        
        # Add some randomness for exploration
        reward += random.uniform(0, 0.2)
        
        return reward


def run_sequence_level_mcts_example():
    """Example usage of sequence-level MCTS."""
    from utils.protein_utils import create_mock_structure_no_sequence
    
    # Create mock structure
    structure = create_mock_structure_no_sequence(length=50)
    
    # Create stub model and tokenizer
    class StubModel:
        def __init__(self):
            self.device = torch.device('cpu')
        
        def to(self, device):
            self.device = device
            return self
        
        def eval(self):
            return self
    
    class StubTokenizer:
        def __init__(self):
            self.vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    
    model = StubModel()
    tokenizer = StubTokenizer()
    
    # Initialize MCTS
    mcts = SequenceLevelMCTS(
        model=model,
        tokenizer=tokenizer,
        max_depth=5,
        num_simulations=50,
        exploration_constant=1.414,
        temperature=1.0
    )
    
    # Run search
    best_sequence, best_reward = mcts.search(structure, target_length=50)
    
    print(f"Best sequence found: {best_sequence}")
    print(f"Best reward: {best_reward:.3f}")
    
    return best_sequence, best_reward


if __name__ == "__main__":
    run_sequence_level_mcts_example() 