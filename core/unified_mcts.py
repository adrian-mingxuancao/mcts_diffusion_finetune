"""
Unified Task-Agnostic MCTS Framework for DPLM-2
===============================================

This module provides a clean, unified MCTS implementation that works across all protein modeling tasks:
- Folding: sequence -> structure
- Inverse Folding: structure -> sequence  
- Motif Scaffolding: motif + scaffold -> complete structure

Key Design Principles:
1. Task-agnostic: Same MCTS logic for all tasks
2. Clean separation: MCTS logic separate from task-specific evaluation
3. pLDDT-based masking: Intelligent masking strategy
4. Multi-expert rollouts: Use different DPLM-2 models for diversity
5. Proper tokenization: Handle both sequence and structure tokens correctly
"""

import math
import random
import hashlib
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import numpy as np
import logging

from .mcts_utils import (
    compute_sequence_hash, SequenceCache, ph_uct_score
)
from .dplm2_integration import DPLM2Integration

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMCTSNode:
    """
    Unified MCTS Node that works for all tasks.
    
    For different tasks, different fields are used:
    - Folding: sequence (input), structure_tokens (generated)
    - Inverse Folding: structure_tokens (input), sequence (generated)
    - Motif Scaffolding: partial_structure_tokens (input), structure_tokens (generated)
    """
    # Core data (task-dependent usage)
    sequence: Optional[str] = None  # Amino acid sequence
    structure_tokens: Optional[str] = None  # Structure tokens (comma-separated)
    
    # Task metadata
    task_type: str = "inverse_folding"  # "folding", "inverse_folding", "motif_scaffolding"
    
    # MCTS tree structure
    depth: int = 0
    parent: Optional['UnifiedMCTSNode'] = None
    children: Optional[List['UnifiedMCTSNode']] = None
    visit_count: int = 0
    value_sum: float = 0.0
    
    # Task-specific evaluation metrics (cached)
    metrics: Optional[Dict[str, float]] = None
    
    # PH-UCT enhancements
    entropy_bonus: float = 0.0
    novelty_bonus: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def average_value(self) -> float:
        """Average reward value."""
        return self.value_sum / max(self.visit_count, 1)
    
    def ucb_score(self, exploration_constant: float = 1.414, parent_visits: int = 1) -> float:
        """UCB1 score with PH-UCT enhancements."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.average_value
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visit_count)
        ph_bonus = self.entropy_bonus + self.novelty_bonus
        
        return exploitation + exploration + ph_bonus
    
    def get_primary_output(self) -> str:
        """Get the primary output for this task."""
        if self.task_type == "folding":
            return self.structure_tokens or ""
        elif self.task_type == "inverse_folding":
            return self.sequence or ""
        elif self.task_type == "motif_scaffolding":
            return self.structure_tokens or ""
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def get_input_data(self) -> Tuple[str, str]:
        """Get (sequence, structure_tokens) input pair for DPLM-2."""
        if self.task_type == "folding":
            return self.sequence or "", self.structure_tokens or ""
        elif self.task_type == "inverse_folding":
            return self.sequence or "", self.structure_tokens or ""
        elif self.task_type == "motif_scaffolding":
            return self.sequence or "", self.structure_tokens or ""
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")


class UnifiedMCTS:
    """
    Unified MCTS implementation for all DPLM-2 tasks.
    
    This class provides a task-agnostic MCTS framework that can optimize:
    - Folding: Given sequence, find best structure
    - Inverse Folding: Given structure, find best sequence  
    - Motif Scaffolding: Given partial structure, complete it
    """
    
    def __init__(self, 
                 task_type: str,
                 evaluator,
                 max_depth: int = 3,
                 exploration_constant: float = 1.414,
                 num_children_select: int = 3,
                 k_rollouts_per_expert: int = 2,
                 use_plddt_masking: bool = True,
                 use_ph_uct: bool = True,
                 device: str = "cuda"):
        """
        Initialize Unified MCTS.
        
        Args:
            task_type: "folding", "inverse_folding", or "motif_scaffolding"
            evaluator: Task-specific evaluator (e.g., StructureEvaluator, SequenceEvaluator)
            max_depth: Maximum search depth
            exploration_constant: UCB exploration parameter
            num_children_select: Number of children to keep per expansion
            k_rollouts_per_expert: Number of rollouts per expert
            use_plddt_masking: Whether to use pLDDT-based masking
            use_ph_uct: Whether to use PH-UCT enhancements
            device: CUDA device
        """
        self.task_type = task_type
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.num_children_select = num_children_select
        self.k_rollouts_per_expert = k_rollouts_per_expert
        self.use_plddt_masking = use_plddt_masking
        self.use_ph_uct = use_ph_uct
        
        # Initialize DPLM-2 integration
        self.dplm2_integration = DPLM2Integration(device=device)
        
        # Cache for avoiding re-evaluation
        self.evaluation_cache = SequenceCache()
        
        # Expert models for multi-expert rollouts
        self.expert_ids = [0, 1, 2]  # 650M, 150M, 3B
        
        logger.info(f"🧬 UnifiedMCTS initialized:")
        logger.info(f"   Task type: {task_type}")
        logger.info(f"   Max depth: {max_depth}")
        logger.info(f"   pLDDT masking: {use_plddt_masking}")
        logger.info(f"   PH-UCT: {use_ph_uct}")
    
    def search(self, 
               input_sequence: Optional[str] = None,
               input_structure_tokens: Optional[str] = None,
               num_iterations: int = 30) -> UnifiedMCTSNode:
        """
        Run MCTS search to optimize for the given task.
        
        Args:
            input_sequence: Input amino acid sequence (for folding)
            input_structure_tokens: Input structure tokens (for inverse folding)
            num_iterations: Number of MCTS iterations
            
        Returns:
            Best node found by MCTS
        """
        # Create root node with baseline generation
        root = self._create_root_node(input_sequence, input_structure_tokens)
        
        logger.info(f"🌳 Starting MCTS search with {num_iterations} iterations")
        logger.info(f"   Root evaluation: {root.average_value:.4f}")
        
        for iteration in range(num_iterations):
            # MCTS cycle: Selection -> Expansion -> Simulation -> Backpropagation
            leaf = self._select_leaf(root)
            
            if leaf.depth < self.max_depth:
                # Expand if not at max depth
                children = self._expand_node(leaf)
                if children:
                    # Select one child for simulation
                    selected_child = random.choice(children)
                    reward = self._simulate(selected_child)
                    self._backpropagate(selected_child, reward)
                else:
                    # No children created, simulate current leaf
                    reward = self._simulate(leaf)
                    self._backpropagate(leaf, reward)
            else:
                # At max depth, just simulate
                reward = self._simulate(leaf)
                self._backpropagate(leaf, reward)
            
            if (iteration + 1) % 10 == 0:
                best_node = self._get_best_child(root)
                logger.info(f"   Iteration {iteration + 1}: Best reward = {best_node.average_value:.4f}")
        
        # Return best child
        best_node = self._get_best_child(root)
        logger.info(f"🎯 MCTS search complete. Best reward: {best_node.average_value:.4f}")
        
        return best_node
    
    def _create_root_node(self, input_sequence: Optional[str], 
                         input_structure_tokens: Optional[str]) -> UnifiedMCTSNode:
        """Create root node with baseline generation."""
        if self.task_type == "folding":
            # For folding: sequence -> structure
            if not input_sequence:
                raise ValueError("Folding task requires input_sequence")
            
            # Generate baseline structure using DPLM-2
            baseline_structure = self._generate_baseline_structure(input_sequence)
            
            root = UnifiedMCTSNode(
                sequence=input_sequence,
                structure_tokens=baseline_structure,
                task_type=self.task_type
            )
            
        elif self.task_type == "inverse_folding":
            # For inverse folding: structure -> sequence
            if not input_structure_tokens:
                raise ValueError("Inverse folding task requires input_structure_tokens")
            
            # Generate baseline sequence using DPLM-2
            baseline_sequence = self._generate_baseline_sequence(input_structure_tokens)
            
            root = UnifiedMCTSNode(
                sequence=baseline_sequence,
                structure_tokens=input_structure_tokens,
                task_type=self.task_type
            )
            
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # Evaluate root node
        reward = self._evaluate_node(root)
        root.visit_count = 1
        root.value_sum = reward
        
        return root
    
    def _generate_baseline_structure(self, sequence: str) -> str:
        """Generate baseline structure tokens for folding task."""
        # Create fully masked structure input
        masked_structure = ",".join(["<mask>"] * len(sequence))
        
        # Generate using DPLM-2
        result = self.dplm2_integration.generate_from_masked_input(
            aa_sequence=sequence,
            struct_tokens=masked_structure,
            task_type="folding",
            expert_id=1  # Use 150M model for baseline
        )
        
        return result or masked_structure
    
    def _generate_baseline_sequence(self, structure_tokens: str) -> str:
        """Generate baseline sequence for inverse folding task."""
        # Estimate sequence length from structure tokens
        structure_list = structure_tokens.split(',')
        seq_length = len(structure_list)
        
        # Create fully masked sequence input
        masked_sequence = "X" * seq_length
        
        # Generate using DPLM-2
        result = self.dplm2_integration.generate_from_masked_input(
            aa_sequence=masked_sequence,
            struct_tokens=structure_tokens,
            task_type="inverse_folding",
            expert_id=1  # Use 150M model for baseline
        )
        
        return result or masked_sequence.replace('X', 'A')
    
    def _select_leaf(self, root: UnifiedMCTSNode) -> UnifiedMCTSNode:
        """Select leaf node using UCB1 with PH-UCT."""
        current = root
        
        while current.children:
            # Select child with highest UCB score
            best_child = max(
                current.children,
                key=lambda child: child.ucb_score(
                    self.exploration_constant, 
                    current.visit_count
                )
            )
            current = best_child
        
        return current
    
    def _expand_node(self, node: UnifiedMCTSNode) -> List[UnifiedMCTSNode]:
        """Expand node by creating children with different masking strategies."""
        children = []
        
        # Generate children using different expert models and masking strategies
        for expert_id in self.expert_ids[:self.num_children_select]:
            child = self._create_child_node(node, expert_id)
            if child:
                children.append(child)
        
        # Add children to node
        node.children.extend(children)
        
        return children
    
    def _create_child_node(self, parent: UnifiedMCTSNode, expert_id: int) -> Optional[UnifiedMCTSNode]:
        """Create a child node using specific expert and masking strategy."""
        try:
            if self.task_type == "folding":
                # For folding: mask structure tokens and regenerate
                masked_structure = self._apply_masking_strategy(
                    parent.structure_tokens, "structure"
                )
                
                new_structure = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=parent.sequence,
                    struct_tokens=masked_structure,
                    task_type="folding",
                    expert_id=expert_id
                )
                
                if new_structure and new_structure != parent.structure_tokens:
                    child = UnifiedMCTSNode(
                        sequence=parent.sequence,
                        structure_tokens=new_structure,
                        task_type=self.task_type,
                        depth=parent.depth + 1,
                        parent=parent
                    )
                    return child
                    
            elif self.task_type == "inverse_folding":
                # For inverse folding: mask sequence and regenerate
                masked_sequence = self._apply_masking_strategy(
                    parent.sequence, "sequence"
                )
                
                new_sequence = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_sequence,
                    struct_tokens=parent.structure_tokens,
                    task_type="inverse_folding",
                    expert_id=expert_id
                )
                
                if new_sequence and new_sequence != parent.sequence:
                    child = UnifiedMCTSNode(
                        sequence=new_sequence,
                        structure_tokens=parent.structure_tokens,
                        task_type=self.task_type,
                        depth=parent.depth + 1,
                        parent=parent
                    )
                    return child
            
        except Exception as e:
            logger.warning(f"Failed to create child with expert {expert_id}: {e}")
        
        return None
    
    def _apply_masking_strategy(self, input_data: str, data_type: str) -> str:
        """Apply intelligent masking strategy based on pLDDT or random."""
        if not input_data:
            return input_data
        
        if data_type == "sequence":
            # Mask amino acid sequence
            if self.use_plddt_masking:
                # TODO: Implement pLDDT-based masking for sequences
                # For now, use random masking
                return self._apply_random_masking(input_data, "X", mask_ratio=0.3)
            else:
                return self._apply_random_masking(input_data, "X", mask_ratio=0.3)
        
        elif data_type == "structure":
            # Mask structure tokens
            if self.use_plddt_masking:
                # TODO: Implement pLDDT-based masking for structure tokens
                # For now, use random masking
                return self._apply_random_masking(input_data, "<mask>", mask_ratio=0.3)
            else:
                return self._apply_random_masking(input_data, "<mask>", mask_ratio=0.3)
        
        return input_data
    
    def _apply_random_masking(self, input_data: str, mask_token: str, mask_ratio: float = 0.3) -> str:
        """Apply random masking to input data."""
        if not input_data:
            return input_data
        
        if "," in input_data:
            # Structure tokens (comma-separated)
            tokens = input_data.split(",")
            num_to_mask = max(1, int(len(tokens) * mask_ratio))
            positions = random.sample(range(len(tokens)), num_to_mask)
            
            for pos in positions:
                tokens[pos] = mask_token
            
            return ",".join(tokens)
        else:
            # Amino acid sequence
            chars = list(input_data)
            num_to_mask = max(1, int(len(chars) * mask_ratio))
            positions = random.sample(range(len(chars)), num_to_mask)
            
            for pos in positions:
                chars[pos] = mask_token
            
            return "".join(chars)
    
    def _simulate(self, node: UnifiedMCTSNode) -> float:
        """Simulate (evaluate) a node using the task-specific evaluator."""
        return self._evaluate_node(node)
    
    def _evaluate_node(self, node: UnifiedMCTSNode) -> float:
        """Evaluate node using task-specific evaluator."""
        # Create cache key
        cache_key = f"{node.task_type}_{compute_sequence_hash(node.get_primary_output())}"
        
        # Check cache
        cached_result = self.evaluation_cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Evaluate using task-specific evaluator
        try:
            if self.task_type == "folding":
                # For folding: evaluate structure quality
                reward = self.evaluator.evaluate_structure(
                    node.sequence, node.structure_tokens
                )
            elif self.task_type == "inverse_folding":
                # For inverse folding: evaluate sequence quality
                reward = self.evaluator.evaluate_sequence(
                    node.sequence, node.structure_tokens
                )
            else:
                reward = 0.0
            
            # Cache result
            self.evaluation_cache.put(cache_key, reward)
            
            return reward
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return 0.0
    
    def _backpropagate(self, node: UnifiedMCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        
        while current is not None:
            current.visit_count += 1
            current.value_sum += reward
            current = current.parent
    
    def _get_best_child(self, node: UnifiedMCTSNode) -> UnifiedMCTSNode:
        """Get child with highest average value."""
        if not node.children:
            return node
        
        return max(node.children, key=lambda child: child.average_value)


# Legacy compatibility - alias for the old GeneralMCTS
GeneralMCTS = UnifiedMCTS





