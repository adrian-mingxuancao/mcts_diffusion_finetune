"""
Position-Level MCTS with plDDT Masking for Inverse Folding with DPLM-2

This module implements Monte Carlo Tree Search at the position level,
where each node represents a partial sequence with masked positions.
The search uses plDDT scores to guide which positions to unmask next.
"""

import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class PositionNode:
    """Node in the position-level MCTS tree representing a partial sequence."""
    sequence: str  # Partial sequence with masked positions
    masked_positions: Set[int]  # Set of masked position indices
    plddt_scores: List[float]  # plDDT scores for each position
    visit_count: int = 0
    total_value: float = 0.0
    children: List['PositionNode'] = None
    parent: Optional['PositionNode'] = None
    
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
        
        if self.parent is None:
            return self.average_value
        
        # UCB1 formula: value + exploration_constant * sqrt(ln(parent_visits) / visits)
        parent_visits = self.parent.visit_count
        return self.average_value + exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
    
    @property
    def is_complete(self) -> bool:
        """Check if all positions are unmasked."""
        return len(self.masked_positions) == 0
    
    @property
    def completion_ratio(self) -> float:
        """Ratio of unmasked positions."""
        total_positions = len(self.sequence)
        unmasked_positions = total_positions - len(self.masked_positions)
        return unmasked_positions / total_positions


class PositionLevelMCTS:
    """
    Monte Carlo Tree Search for position-level inverse folding with plDDT masking.
    
    Each node represents a partial sequence with some positions masked.
    The search uses plDDT scores to determine which positions to unmask next.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        max_depth: int = 10,
        num_simulations: int = 100,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        plddt_threshold: float = 0.7,
        max_unmask_per_step: int = 3
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.plddt_threshold = plddt_threshold
        self.max_unmask_per_step = max_unmask_per_step
        
        # Cache for generated sequences to avoid duplicates
        self.sequence_cache = set()
    
    def search(self, structure: Dict, target_length: int) -> Tuple[str, float]:
        """
        Perform position-level MCTS search to find the best sequence.
        
        Args:
            structure: Protein structure data
            target_length: Target sequence length
            
        Returns:
            Tuple of (best_sequence, best_reward)
        """
        print(f"Starting position-level MCTS search with {self.num_simulations} simulations...")
        
        # Initialize root node with all positions masked
        initial_sequence = "X" * target_length  # X represents masked positions
        masked_positions = set(range(target_length))
        plddt_scores = [0.0] * target_length  # Initial plDDT scores
        
        root = PositionNode(
            sequence=initial_sequence,
            masked_positions=masked_positions,
            plddt_scores=plddt_scores
        )
        
        # Perform MCTS simulations
        for i in range(self.num_simulations):
            if i % 20 == 0:
                print(f"  Simulation {i+1}/{self.num_simulations}")
            
            # Selection
            selected_node = self._select(root)
            
            # Expansion
            if selected_node.visit_count > 0 and not selected_node.is_complete:
                self._expand(selected_node, structure)
            
            # Simulation
            value = self._simulate(selected_node, structure)
            
            # Backpropagation
            self._backpropagate(selected_node, value)
        
        # Find best complete sequence
        best_sequence = self._find_best_complete_sequence(root)
        best_reward = self._compute_reward(best_sequence, structure)
        
        return best_sequence, best_reward
    
    def _select(self, node: PositionNode) -> PositionNode:
        """Select a node using UCB1."""
        if not node.children:
            return node
        
        # Select child with highest UCB score
        best_child = max(node.children, key=lambda x: x.ucb_score)
        return self._select(best_child)
    
    def _expand(self, node: PositionNode, structure: Dict):
        """Expand a node by unmasking positions based on plDDT scores."""
        if node.is_complete:
            return
        
        # Get plDDT scores for masked positions
        plddt_scores = self._compute_plddt_scores(node, structure)
        
        # Select positions to unmask based on plDDT scores
        positions_to_unmask = self._select_positions_to_unmask(
            node.masked_positions, plddt_scores
        )
        
        # Generate children by unmasking different combinations of positions
        for positions in positions_to_unmask:
            child = self._create_child_node(node, positions, structure)
            if child and child.sequence not in self.sequence_cache:
                node.children.append(child)
                self.sequence_cache.add(child.sequence)
    
    def _compute_plddt_scores(self, node: PositionNode, structure: Dict) -> List[float]:
        """Compute plDDT scores for masked positions."""
        # This is a placeholder - in practice, you'd use the actual model to compute plDDT
        # For now, we'll use a simple heuristic based on position and randomness
        
        scores = []
        for pos in node.masked_positions:
            # Simple heuristic: positions near the middle tend to have higher confidence
            center_distance = abs(pos - len(node.sequence) / 2) / len(node.sequence)
            base_score = 1.0 - center_distance * 0.5
            # Add some randomness
            score = base_score + random.uniform(-0.2, 0.2)
            scores.append(max(0.0, min(1.0, score)))
        
        return scores
    
    def _select_positions_to_unmask(
        self, masked_positions: Set[int], plddt_scores: List[float]
    ) -> List[Set[int]]:
        """Select which positions to unmask based on plDDT scores."""
        if not masked_positions:
            return []
        
        # Convert to list for easier handling
        positions_list = list(masked_positions)
        
        # Strategy 1: Unmask high-confidence positions
        high_confidence_positions = []
        for i, pos in enumerate(positions_list):
            if plddt_scores[i] > self.plddt_threshold:
                high_confidence_positions.append(pos)
        
        # Strategy 2: Unmask a few random positions for exploration
        random_positions = random.sample(positions_list, min(2, len(positions_list)))
        
        # Combine strategies
        all_candidates = set(high_confidence_positions + random_positions)
        
        # Generate different combinations
        combinations = []
        candidates_list = list(all_candidates)
        
        # Single position unmasking
        for pos in candidates_list:
            combinations.append({pos})
        
        # Multiple position unmasking (up to max_unmask_per_step)
        if len(candidates_list) >= 2:
            for i in range(min(self.max_unmask_per_step - 1, len(candidates_list) - 1)):
                for j in range(i + 1, min(i + 3, len(candidates_list))):
                    combinations.append({candidates_list[i], candidates_list[j]})
        
        return combinations[:5]  # Limit number of children
    
    def _create_child_node(
        self, parent: PositionNode, positions_to_unmask: Set[int], structure: Dict
    ) -> Optional[PositionNode]:
        """Create a child node by unmasking specific positions."""
        if not positions_to_unmask:
            return None
        
        # Generate amino acids for the positions to unmask
        new_sequence = list(parent.sequence)
        new_masked_positions = parent.masked_positions.copy()
        new_plddt_scores = parent.plddt_scores.copy()
        
        for pos in positions_to_unmask:
            if pos in new_masked_positions:
                # Generate amino acid for this position
                aa = self._generate_amino_acid_for_position(pos, parent, structure)
                new_sequence[pos] = aa
                new_masked_positions.remove(pos)
                new_plddt_scores[pos] = 1.0  # Assume high confidence for unmasked positions
        
        child = PositionNode(
            sequence=''.join(new_sequence),
            masked_positions=new_masked_positions,
            plddt_scores=new_plddt_scores,
            parent=parent
        )
        
        return child
    
    def _generate_amino_acid_for_position(
        self, position: int, parent: PositionNode, structure: Dict
    ) -> str:
        """Generate an amino acid for a specific position."""
        # This is a placeholder - in practice, you'd use the DPLM-2 model
        # to generate the most likely amino acid for this position
        
        try:
            # Try to use the actual model
            aa = self._generate_from_model(position, parent, structure)
        except Exception as e:
            # Fallback: use heuristics
            aa = self._generate_with_heuristics(position, parent, structure)
        
        return aa
    
    def _generate_from_model(self, position: int, parent: PositionNode, structure: Dict) -> str:
        """Generate amino acid using the DPLM-2 model."""
        # Placeholder - replace with actual model generation
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        return random.choice(amino_acids)
    
    def _generate_with_heuristics(self, position: int, parent: PositionNode, structure: Dict) -> str:
        """Generate amino acid using heuristics."""
        # Simple heuristics based on position and context
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Prefer common amino acids
        common_aas = "ACDEFGHIKLMNPQRSTVWY"
        weights = [1.0] * len(common_aas)
        
        # Adjust weights based on position
        if position < 10:  # N-terminus
            weights[common_aas.index('M')] *= 1.5  # Prefer methionine at start
        elif position > len(parent.sequence) - 10:  # C-terminus
            weights[common_aas.index('K')] *= 1.3  # Prefer lysine at end
        
        # Choose based on weights
        return random.choices(common_aas, weights=weights)[0]
    
    def _simulate(self, node: PositionNode, structure: Dict) -> float:
        """Simulate by completing the sequence and evaluating it."""
        if node.is_complete:
            return self._compute_reward(node.sequence, structure)
        
        # Complete the sequence randomly
        completed_sequence = self._complete_sequence_randomly(node)
        return self._compute_reward(completed_sequence, structure)
    
    def _complete_sequence_randomly(self, node: PositionNode) -> str:
        """Complete a partial sequence by filling masked positions randomly."""
        sequence = list(node.sequence)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for pos in node.masked_positions:
            sequence[pos] = random.choice(amino_acids)
        
        return ''.join(sequence)
    
    def _backpropagate(self, node: PositionNode, value: float):
        """Backpropagate the simulation result up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent
    
    def _find_best_complete_sequence(self, root: PositionNode) -> str:
        """Find the best complete sequence in the tree."""
        best_sequence = ""
        best_value = -float('inf')
        
        def search_node(node: PositionNode):
            nonlocal best_sequence, best_value
            
            if node.is_complete and node.average_value > best_value:
                best_sequence = node.sequence
                best_value = node.average_value
            
            for child in node.children:
                search_node(child)
        
        search_node(root)
        
        # If no complete sequence found, return the most complete one
        if not best_sequence:
            best_sequence = self._find_most_complete_sequence(root)
        
        return best_sequence
    
    def _find_most_complete_sequence(self, root: PositionNode) -> str:
        """Find the most complete sequence in the tree."""
        best_sequence = ""
        best_completion = -1
        
        def search_node(node: PositionNode):
            nonlocal best_sequence, best_completion
            
            completion = node.completion_ratio
            if completion > best_completion:
                best_sequence = node.sequence
                best_completion = completion
            
            for child in node.children:
                search_node(child)
        
        search_node(root)
        return best_sequence
    
    def _compute_reward(self, sequence: str, structure: Dict) -> float:
        """Compute reward for a sequence given the structure."""
        from protein_utils import compute_structure_metrics
        
        if not sequence or "X" in sequence:
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


def run_position_level_mcts_example():
    """Example usage of position-level MCTS."""
    from protein_utils import create_mock_structure_no_sequence
    
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
    mcts = PositionLevelMCTS(
        model=model,
        tokenizer=tokenizer,
        max_depth=10,
        num_simulations=100,
        exploration_constant=1.414,
        temperature=1.0,
        plddt_threshold=0.7,
        max_unmask_per_step=3
    )
    
    # Run search
    best_sequence, best_reward = mcts.search(structure, target_length=50)
    
    print(f"Best sequence found: {best_sequence}")
    print(f"Best reward: {best_reward:.3f}")
    
    return best_sequence, best_reward


if __name__ == "__main__":
    run_position_level_mcts_example() 