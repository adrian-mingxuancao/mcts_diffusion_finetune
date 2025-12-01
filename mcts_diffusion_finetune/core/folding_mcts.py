"""
Folding-specific MCTS implementation for protein structure prediction.
Given amino acid sequences, generates structure tokens and optimizes structural metrics.
"""

import math
import random
import hashlib
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np

from .mcts_utils import (
    compute_sequence_hash, SequenceCache
)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class FoldingMCTSNode:
    """
    MCTS Node for folding task that stores amino acid sequences and generated structure tokens.
    """
    sequence: str  # Amino acid sequence (fixed input)
    structure_tokens: Optional[str] = None  # Generated structure tokens (comma-separated)
    depth: int = 0
    parent: Optional['FoldingMCTSNode'] = None
    children: Optional[List['FoldingMCTSNode']] = None
    visit_count: int = 0
    value_sum: float = 0.0
    
    # Folding-specific metrics
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_value(self) -> float:
        """Average reward value."""
        return self.value_sum / max(self.visit_count, 1)
    
    def ucb_score(self, exploration_constant: float = 1.414, parent_visits: int = 1) -> float:
        """UCB1 score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        return exploitation + exploration


class FoldingMCTS:
    """
    MCTS for protein folding task: given amino acid sequence, generate structure tokens
    and optimize structural quality metrics (RMSD, TM-score).
    """
    
    def __init__(self, dplm2_integration, reference_coords: np.ndarray, 
                 max_depth: int = 3, exploration_constant: float = 1.414,
                 num_children_select: int = 2, k_rollouts_per_expert: int = 2,
                 ablation_mode: str = "multi_expert", single_expert_id: int = 0):
        """
        Initialize Folding MCTS.
        
        Args:
            dplm2_integration: DPLM-2 model integration
            reference_coords: Reference structure coordinates for evaluation
            max_depth: Maximum search depth
            exploration_constant: UCB exploration parameter
            num_children_select: Number of children to keep per expansion
            k_rollouts_per_expert: Number of rollouts per expert
            ablation_mode: "random_no_expert", "single_expert", or "multi_expert"
            single_expert_id: Expert ID for single expert mode
        """
        self.dplm2_integration = dplm2_integration
        self.reference_coords = reference_coords
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.num_children_select = num_children_select
        self.k_rollouts_per_expert = k_rollouts_per_expert
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        
        # Cache for structure evaluation
        self.structure_cache = SequenceCache()
        
        print(f"🧬 FoldingMCTS initialized:")
        print(f"   Task type: folding")
        print(f"   Max depth: {max_depth}")
        print(f"   Ablation mode: {ablation_mode}")
        print(f"   Reference coords shape: {reference_coords.shape}")
    
    def search(self, initial_sequence: str, num_iterations: int = 25) -> 'FoldingMCTSNode':
        """
        Run MCTS search for folding optimization.
        
        Args:
            initial_sequence: Input amino acid sequence
            num_iterations: Number of MCTS iterations
            
        Returns:
            Root node with search tree
        """
        print(f"🎯 Folding MCTS search starting from sequence (length: {len(initial_sequence)})")
        
        # Create root node with initial sequence
        root = FoldingMCTSNode(sequence=initial_sequence)
        
        # Generate initial structure tokens for root
        root.structure_tokens = self._generate_structure_tokens(initial_sequence, expert_id=1)  # Use 150M for baseline
        if root.structure_tokens:
            root.rmsd, root.tm_score = self._evaluate_structure_quality(root.structure_tokens)
            print(f"  📈 Root structure quality: RMSD={root.rmsd:.3f}Å, TM-score={root.tm_score:.3f}")
        
        best_node = root
        best_reward = self._compute_folding_reward(root)
        
        for iteration in range(num_iterations):
            # Selection: traverse tree to find leaf node
            selected_node = self._select_node(root)
            
            # Expansion: generate children if not at max depth
            if selected_node.depth < self.max_depth:
                children = self._expand_node(selected_node)
                if children:
                    selected_node = random.choice(children)  # Select random child for evaluation
            
            # Evaluation: compute reward for selected node
            reward = self._compute_folding_reward(selected_node)
            
            # Backpropagation: update values up the tree
            self._backpropagate(selected_node, reward)
            
            # Track best node
            if reward > best_reward:
                best_reward = reward
                best_node = selected_node
                print(f"[iter {iteration+1}] New best: depth={selected_node.depth}, "
                      f"RMSD={selected_node.rmsd:.3f}Å, TM-score={selected_node.tm_score:.3f}, reward={reward:.3f}")
        
        print(f"🏆 Folding search completed. Best reward: {best_reward:.3f}")
        return root
    
    def _select_node(self, root: FoldingMCTSNode) -> FoldingMCTSNode:
        """Select node using UCB1 policy."""
        current = root
        
        while current.children and current.depth < self.max_depth:
            # Select child with highest UCB score
            ucb_scores = [child.ucb_score(self.exploration_constant, current.visit_count) 
                         for child in current.children]
            best_child_idx = np.argmax(ucb_scores)
            current = current.children[best_child_idx]
        
        return current
    
    def _expand_node(self, node: FoldingMCTSNode) -> List[FoldingMCTSNode]:
        """
        Expand node by generating multiple structure predictions using different experts.
        """
        if node.depth >= self.max_depth:
            return []
        
        print(f"🌱 Expanding folding node at depth {node.depth}")
        
        children = []
        
        if self.ablation_mode == "single_expert":
            # Single expert mode
            expert_ids = [self.single_expert_id]
            k_rollouts = 3
        elif self.ablation_mode == "multi_expert":
            # Multi-expert mode
            expert_ids = [0, 1, 2]  # 650M, 150M, 3B
            k_rollouts = self.k_rollouts_per_expert
        else:
            # Random mode - just use ESMFold fallback
            expert_ids = [1]  # Use 150M
            k_rollouts = self.num_children_select
        
        # Generate structure candidates using different experts
        candidates = []
        for expert_id in expert_ids:
            for rollout in range(k_rollouts):
                structure_tokens = self._generate_structure_tokens(node.sequence, expert_id)
                if structure_tokens:
                    candidates.append((expert_id, structure_tokens))
        
        # Evaluate and rank candidates
        evaluated_candidates = []
        for expert_id, structure_tokens in candidates:
            rmsd, tm_score = self._evaluate_structure_quality(structure_tokens)
            reward = self._compute_structure_reward(rmsd, tm_score)
            evaluated_candidates.append((reward, expert_id, structure_tokens, rmsd, tm_score))
        
        # Select top candidates
        evaluated_candidates.sort(reverse=True, key=lambda x: x[0])
        top_candidates = evaluated_candidates[:self.num_children_select]
        
        # Create child nodes
        for reward, expert_id, structure_tokens, rmsd, tm_score in top_candidates:
            child = FoldingMCTSNode(
                sequence=node.sequence,
                structure_tokens=structure_tokens,
                depth=node.depth + 1,
                parent=node,
                rmsd=rmsd,
                tm_score=tm_score
            )
            children.append(child)
            node.children.append(child)
        
        print(f"   ✅ Generated {len(children)} folding children")
        return children
    
    def _generate_structure_tokens(self, sequence: str, expert_id: int) -> Optional[str]:
        """Generate structure tokens from amino acid sequence using specified expert."""
        try:
            return self.dplm2_integration.generate_structure_tokens_from_sequence(
                expert_id=expert_id,
                sequence=sequence,
                temperature=1.0
            )
        except Exception as e:
            print(f"⚠️ Structure generation failed with expert {expert_id}: {e}")
            return None
    
    def _evaluate_structure_quality(self, structure_tokens: str) -> Tuple[float, float]:
        """
        Evaluate structure quality by converting tokens to coordinates and computing metrics.
        
        Args:
            structure_tokens: Comma-separated structure tokens
            
        Returns:
            Tuple of (rmsd, tm_score)
        """
        try:
            # Convert structure tokens to coordinates
            predicted_coords = self._structure_tokens_to_coords(structure_tokens)
            if predicted_coords is None:
                return 10.0, 0.0  # Bad RMSD, no TM-score
            
            # Calculate RMSD and TM-score against reference
            rmsd, tm_score = self._calculate_rmsd_and_tmscore(predicted_coords, self.reference_coords)
            return rmsd, tm_score
            
        except Exception as e:
            print(f"⚠️ Structure evaluation failed: {e}")
            return 10.0, 0.0
    
    def _structure_tokens_to_coords(self, structure_tokens: str) -> Optional[np.ndarray]:
        """
        Convert structure tokens to 3D coordinates.
        
        TODO: Implement proper structure token to coordinate conversion.
        For now, use ESMFold as fallback.
        """
        try:
            # TODO: Implement DPLM-2 structure token decoder
            # For now, fallback to ESMFold prediction from sequence
            print("⚠️ Using ESMFold fallback for structure token conversion")
            
            # This is a placeholder - we need to implement proper token-to-coord conversion
            return self.dplm2_integration._predict_structure_coords("A" * 100)  # Dummy
            
        except Exception as e:
            print(f"❌ Structure token conversion failed: {e}")
            return None
    
    def _calculate_rmsd_and_tmscore(self, pred_coords: np.ndarray, ref_coords: np.ndarray) -> Tuple[float, float]:
        """Calculate RMSD and TM-score between predicted and reference coordinates."""
        try:
            # Ensure same length
            min_len = min(len(pred_coords), len(ref_coords))
            pred_coords = pred_coords[:min_len]
            ref_coords = ref_coords[:min_len]
            
            if len(pred_coords) == 0:
                return float('inf'), 0.0
            
            # Calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords) ** 2, axis=1)))
            
            # Calculate TM-score
            L_target = len(ref_coords)
            d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
            
            distances = np.sqrt(np.sum((pred_coords - ref_coords) ** 2, axis=1))
            tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
            
            return rmsd, tm_score
            
        except Exception as e:
            print(f"⚠️ RMSD/TM-score calculation failed: {e}")
            return float('inf'), 0.0
    
    def _compute_folding_reward(self, node: FoldingMCTSNode) -> float:
        """Compute reward for folding node based on structural quality."""
        if node.rmsd is None or node.tm_score is None:
            return 0.0
        
        return self._compute_structure_reward(node.rmsd, node.tm_score)
    
    def _compute_structure_reward(self, rmsd: float, tm_score: float) -> float:
        """Compute structure quality reward from RMSD and TM-score."""
        # Convert RMSD to reward (lower is better, so invert)
        rmsd_reward = max(0.0, 1.0 - min(rmsd / 10.0, 1.0))  # Normalize RMSD
        
        # Combine metrics
        rmsd_weight = 0.40
        tm_weight = 0.60
        
        reward = (rmsd_reward * rmsd_weight) + (tm_score * tm_weight)
        return reward
    
    def _backpropagate(self, node: FoldingMCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent
