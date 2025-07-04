"""
Monte Carlo Tree Search (MCTS) for protein sequence generation.

This module implements MCTS to guide the diffusion model in generating
high-quality protein sequences for inverse folding tasks.

Key components:
- MCTSNode: Tree node representing a sequence state
- MCTS: Main search algorithm with UCB1 selection
- Sequence generation with diffusion model guidance
"""

import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a sequence state."""
    sequence: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = None
    visits: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'MCTSNode'):
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def update_reward(self, reward: float):
        """Update node statistics after simulation."""
        self.visits += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.visits
    
    def ucb1_score(self, exploration_constant: float = 1.414) -> float:
        """Compute UCB1 score for node selection."""
        if self.visits == 0:
            return float('inf')
        
        # UCB1 formula: mean_reward + C * sqrt(ln(parent_visits) / visits)
        parent_visits = self.parent.visits if self.parent else 1
        exploration_term = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return self.mean_reward + exploration_term

class MCTS:
    """Monte Carlo Tree Search for protein sequence generation."""
    
    def __init__(self, 
                 model, 
                 tokenizer,
                 max_depth: int = 10,
                 num_simulations: int = 100,
                 exploration_constant: float = 1.414,
                 temperature: float = 1.0):
        """
        Initialize MCTS.
        
        Args:
            model: DPLM model for sequence generation
            tokenizer: Model tokenizer
            max_depth: Maximum tree depth
            num_simulations: Number of MCTS simulations
            exploration_constant: UCB1 exploration constant
            temperature: Sampling temperature for sequence generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
    
    def search(self, structure: Dict, target_length: int = 50) -> Tuple[str, float]:
        """
        Perform MCTS search to find the best sequence.
        
        Args:
            structure: Protein structure dictionary
            target_length: Target sequence length
            
        Returns:
            Tuple of (best_sequence, best_reward)
        """
        print(f"Starting MCTS search with {self.num_simulations} simulations...")
        
        # Initialize root node
        root = MCTSNode(sequence="")
        
        # Run MCTS simulations
        for i in range(self.num_simulations):
            if i % 20 == 0:
                print(f"  Simulation {i+1}/{self.num_simulations}")
            
            # Selection: traverse tree to find leaf node
            leaf = self._select(root)
            
            # Expansion: expand leaf if not terminal
            if leaf.depth < self.max_depth and len(leaf.sequence) < target_length:
                self._expand(leaf, structure, target_length)
            
            # Simulation: simulate from leaf to terminal
            reward = self._simulate(leaf, structure, target_length)
            
            # Backpropagation: update node statistics
            self._backpropagate(leaf, reward)
        
        # Get search statistics
        stats = self.get_search_statistics(root)
        print(f"MCTS Statistics:")
        print(f"  Total nodes explored: {stats['total_nodes']}")
        print(f"  Max tree depth: {stats['max_depth']}")
        print(f"  Root visits: {stats['root_visits']}")
        print(f"  Best reward found: {stats['best_reward']:.3f}")
        
        # Return best child of root
        if root.children:
            # Sort children by mean reward and show top 3
            sorted_children = sorted(root.children, key=lambda c: c.mean_reward, reverse=True)
            print(f"  Top 3 candidates:")
            for i, child in enumerate(sorted_children[:3]):
                print(f"    {i+1}. Reward: {child.mean_reward:.3f}, Visits: {child.visits}, Seq: {child.sequence[:30]}...")
            
            best_child = sorted_children[0]
            print(f"  Best sequence length: {len(best_child.sequence)}")
            return best_child.sequence, best_child.mean_reward
        else:
            print("  No children found - returning empty sequence")
            return "", 0.0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCB1."""
        while node.children:
            # If all children have been visited, select by UCB1
            if all(child.visits > 0 for child in node.children):
                node = max(node.children, key=lambda c: c.ucb1_score(self.exploration_constant))
            else:
                # If some children haven't been visited, select an unvisited one
                unvisited = [child for child in node.children if child.visits == 0]
                if unvisited:
                    node = random.choice(unvisited)
                    break
                else:
                    # Fallback: select by UCB1
                    node = max(node.children, key=lambda c: c.ucb1_score(self.exploration_constant))
        
        return node
    
    def _expand(self, node: MCTSNode, structure: Dict, target_length: int):
        """Expand a leaf node by generating child sequences."""
        if len(node.sequence) >= target_length:
            return
        
        # Generate candidate sequences using diffusion model
        candidates = self._generate_candidates(node.sequence, structure, target_length)
        
        # Create child nodes
        for candidate in candidates[:5]:  # Limit to top 5 candidates
            child = MCTSNode(sequence=candidate)
            node.add_child(child)
        
        # Debug: show expansion
        if node.depth == 0 and random.random() < 0.2:
            print(f"    Expanded root with {len(node.children)} children")
            for i, child in enumerate(node.children[:2]):
                print(f"      Child {i+1}: {child.sequence[:20]}...")
    
    def _generate_candidates(self, current_seq: str, structure: Dict, target_length: int) -> List[str]:
        """Generate candidate sequences using the diffusion model."""
        from dplm_inverse_folding import generate_sequence_from_structure, generate_sequence_variations
        
        candidates = []
        
        if len(current_seq) == 0:
            # Root node: generate initial sequences using DPLM-2
            print("    Using DPLM-2 for initial sequence generation...")
            initial_sequences = generate_sequence_from_structure(
                self.model, self.tokenizer, structure, 
                num_samples=10, temperature=self.temperature
            )
            candidates.extend(initial_sequences)
        else:
            # Non-root node: generate variations of current sequence
            print(f"    Generating variations of current sequence (length: {len(current_seq)})...")
            variations = generate_sequence_variations(
                current_seq, structure, 
                num_variations=10, mutation_rate=0.1
            )
            candidates.extend(variations)
        
        return candidates
    
    def _simulate(self, node: MCTSNode, structure: Dict, target_length: int) -> float:
        """Simulate from node to terminal state and compute reward."""
        from protein_utils import compute_structure_metrics
        from dplm_inverse_folding import evaluate_sequence_structure_compatibility
        
        # Complete the sequence if needed
        sequence = node.sequence
        if len(sequence) < target_length:
            remaining_length = target_length - len(sequence)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            sequence += ''.join(random.choices(amino_acids, k=remaining_length))
        
        # Compute structure-sequence compatibility (main reward)
        compatibility_score = evaluate_sequence_structure_compatibility(sequence, structure)
        
        # Compute additional metrics
        metrics = compute_structure_metrics(sequence, structure)
        
        # Combined reward function
        reward = 0.0
        
        # Structure compatibility (primary reward)
        reward += compatibility_score * 0.5
        
        # Sequence quality metrics
        if 10 <= len(sequence) <= 500:
            reward += 0.1
        
        if -2.0 <= metrics['hydrophobicity'] <= 2.0:
            reward += 0.2
        
        if abs(metrics['charge']) <= 10:
            reward += 0.1
        
        # Diversity reward
        unique_aas = len(set(sequence))
        diversity_score = unique_aas / len(sequence)
        reward += diversity_score * 0.1
        
        # Add small randomness for exploration
        reward += random.uniform(0, 0.05)
        
        # Debug output for first few simulations
        if node.depth == 0 and random.random() < 0.1:  # 10% of root simulations
            print(f"    Debug - Seq: {sequence[:20]}... | Compat: {compatibility_score:.2f} | Hydro: {metrics['hydrophobicity']:.2f} | Reward: {reward:.3f}")
        
        return reward
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        # Debug: print reward for first few backpropagations
        if node.depth == 0 and random.random() < 0.1:
            print(f"    Backpropagating reward: {reward:.3f}")
        
        while node is not None:
            old_mean = node.mean_reward
            node.update_reward(reward)
            
            # Debug: show reward update for root node
            if node.depth == 0 and random.random() < 0.1:
                print(f"    Root reward update: {old_mean:.3f} -> {node.mean_reward:.3f} (visits: {node.visits})")
            
            node = node.parent
    
    def get_search_statistics(self, root: MCTSNode) -> Dict:
        """Get statistics about the MCTS search."""
        def count_nodes(node: MCTSNode) -> int:
            return 1 + sum(count_nodes(child) for child in node.children)
        
        def max_depth(node: MCTSNode) -> int:
            if not node.children:
                return node.depth
            return max(max_depth(child) for child in node.children)
        
        return {
            'total_nodes': count_nodes(root),
            'max_depth': max_depth(root),
            'root_visits': root.visits,
            'best_reward': max((child.mean_reward for child in root.children), default=0.0)
        } 