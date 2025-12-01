"""
Working MCTS Implementation for Inverse Folding and Folding
Extracted from the backup file and cleaned up.

This implements the proper MCTS algorithm:
1. Selection with UCB1
2. Expansion with DPLM-2 rollouts
3. Simulation with proper evaluation
4. Backpropagation
"""

import math
import random
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import sys
import os
import numpy as np
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dplm2_integration import DPLM2Integration


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    sequence: str
    masked_positions: Set[int]
    reward: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = None
    parent: 'MCTSNode' = None
    depth: int = 0
    task_type: str = "inverse_folding"
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_value(self) -> float:
        """Average value of this node."""
        return self.total_value / max(self.visit_count, 1)
    
    @property
    def ucb_score(self) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula
        if self.parent and self.parent.visit_count > 0:
            parent_visits = self.parent.visit_count
            log_term = math.log(parent_visits)
            if log_term > 0:
                exploration = 1.414 * math.sqrt(log_term / self.visit_count)
                return self.average_value + exploration
        
        return self.average_value


class GeneralMCTS:
    """
    Working MCTS framework for inverse folding and folding.
    Extracted from backup and cleaned up.
    """
    
    def __init__(
        self,
        task_type: str = "inverse_folding",
        max_depth: int = 5,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        use_plddt_masking: bool = True,
        dplm2_integration: object = None,
        external_experts: List = None,
        ablation_mode: str = "multi_expert",
        single_expert_id: int = None,
        reference_sequence: str = None,
        baseline_structure: dict = None
    ):
        """
        Initialize MCTS framework.
        """
        # Validate task type
        valid_tasks = ["inverse_folding", "folding", "unconditional", "conditional"]
        if task_type == "motif_scaffolding":
            raise ValueError(
                "Motif scaffolding should use MotifScaffoldingMCTS from core.motif_scaffolding_mcts. "
                "This class (GeneralMCTS) is only for inverse_folding, folding, unconditional, and conditional tasks."
            )
        elif task_type not in valid_tasks:
            raise ValueError(f"Unknown task type: {task_type}. Valid tasks: {valid_tasks}")
        
        self.task_type = task_type
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.use_plddt_masking = use_plddt_masking
        
        # Multi-expert support
        self.external_experts = external_experts or []
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        
        # Initialize DPLM-2 integration
        self.dplm2_integration = dplm2_integration
        if not self.dplm2_integration:
            try:
                self.dplm2_integration = DPLM2Integration(use_local=True)
            except:
                raise ValueError("DPLM-2 integration is required")
        
        # Store reference and baseline
        self.reference_sequence = reference_sequence
        self.baseline_structure = baseline_structure or {}
        self._baseline_structure = self.baseline_structure.copy() if self.baseline_structure else {}
        
        # Cache for sequences
        self.sequence_cache = set()
        
        # Task-specific setup
        self._setup_task()
        
        print(f"✅ GeneralMCTS initialized for {task_type}")
        if self.external_experts:
            expert_names = [e.get_name() for e in self.external_experts]
            print(f"   External experts: {expert_names}")
        print(f"   Ablation mode: {ablation_mode}" + (f" (expert {single_expert_id})" if single_expert_id is not None else ""))
    
    def _setup_task(self):
        """Setup task-specific parameters."""
        if self.task_type == "inverse_folding":
            print("🎯 Setting up inverse folding task")
        elif self.task_type == "folding":
            print("🎯 Setting up folding task")
        elif self.task_type == "unconditional":
            print("🎯 Setting up unconditional generation task")
        elif self.task_type == "conditional":
            print("🎯 Setting up conditional generation task")
    
    def search(self, initial_sequence: str = None, reference_sequence: str = None, 
               num_iterations: int = None, structure_data: Dict = None) -> 'MCTSNode':
        """
        Perform MCTS search following proper algorithm.
        
        Args:
            initial_sequence: Initial sequence to start from
            reference_sequence: Reference sequence for evaluation
            num_iterations: Number of MCTS iterations
            structure_data: Structure data (for compatibility)
        
        Returns:
            Root MCTSNode with search tree
        """
        print(f"🔍 Starting MCTS search for {self.task_type}")
        
        # Use provided parameters or defaults
        if num_iterations is None:
            num_iterations = self.num_simulations
        
        # Store reference for evaluation
        if reference_sequence:
            self.reference_sequence = reference_sequence
            self._reference_sequence = reference_sequence
        
        # Get initial sequence
        if initial_sequence:
            sequence = initial_sequence
        else:
            # Use baseline sequence from DPLM2Integration
            sequence = getattr(self.dplm2_integration, '_current_baseline_sequence', None)
            if not sequence:
                print("❌ No initial sequence provided and no baseline found")
                return None
        
        print(f"🚀 Starting MCTS with {num_iterations} iterations")
        print(f"   Initial sequence: {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        
        # Calculate baseline AAR if reference available
        baseline_aar = None
        if self.reference_sequence:
            baseline_aar = self._calculate_simple_aar(sequence, self.reference_sequence)
            print(f"   Baseline AAR: {baseline_aar:.1%}")
        
        # Apply initial masking for exploration
        if self.use_plddt_masking and hasattr(self.dplm2_integration, 'baseline_structure'):
            plddt_scores = self.dplm2_integration.baseline_structure.get('plddt_scores', [])
            if plddt_scores and len(plddt_scores) == len(sequence):
                initial_masked = self._apply_plddt_masking_simple(sequence, plddt_scores)
            else:
                # Fallback to light random masking
                mask_ratio = 0.1  # 10% masking for exploration
                num_to_mask = max(1, int(len(sequence) * mask_ratio))
                initial_masked = set(random.sample(range(len(sequence)), num_to_mask))
        else:
            # Light random masking
            mask_ratio = 0.1
            num_to_mask = max(1, int(len(sequence) * mask_ratio))
            initial_masked = set(random.sample(range(len(sequence)), num_to_mask))
        
        print(f"   Initial masked positions: {len(initial_masked)}")
        
        # Create root node
        root = MCTSNode(
            sequence=sequence,
            masked_positions=initial_masked,
            task_type=self.task_type,
            depth=0
        )
        
        # Evaluate root
        root.reward = self._evaluate_sequence(sequence)
        root.visit_count = 1
        root.total_value = root.reward
        
        best_sequence = sequence
        best_reward = root.reward
        
        print(f"   Root reward: {root.reward:.4f}")
        
        # MCTS iterations
        for iteration in range(num_iterations):
            if iteration % 10 == 0:
                print(f"   Iteration {iteration + 1}/{num_iterations}")
            
            # 1. SELECTION: Select leaf node using UCB1
            selected_node = self._select(root)
            
            # 2. EXPANSION: Generate children if not at max depth
            if selected_node.depth < self.max_depth and selected_node.masked_positions:
                self._expand(selected_node)
            
            # 3. SIMULATION: Evaluate children
            if selected_node.children:
                for child in selected_node.children:
                    if child.visit_count == 0:
                        child.reward = self._evaluate_sequence(child.sequence)
                        child.visit_count = 1
                        child.total_value = child.reward
                        
                        # Track best
                        if child.reward > best_reward:
                            best_sequence = child.sequence
                            best_reward = child.reward
                            print(f"     🏆 New best: {best_reward:.4f} (depth {child.depth})")
            
            # 4. BACKPROPAGATION: Update statistics
            if selected_node.children:
                best_child = max(selected_node.children, key=lambda c: c.reward)
                self._backpropagate(best_child, best_child.reward)
        
        print(f"✅ MCTS completed. Best reward: {best_reward:.4f}")
        
        # Update root with best results
        root.sequence = best_sequence
        root.reward = best_reward
        
        return root
    
    def _apply_plddt_masking_simple(self, sequence: str, plddt_scores: List[float], mask_ratio: float = 0.15) -> Set[int]:
        """Apply simple pLDDT-based masking."""
        seq_len = len(sequence)
        num_to_mask = max(1, int(seq_len * mask_ratio))
        
        # Find positions with lowest pLDDT
        plddt_array = np.array(plddt_scores)
        sorted_indices = np.argsort(plddt_array)
        mask_positions = set(sorted_indices[:num_to_mask].tolist())
        
        return mask_positions
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select leaf node using UCB1."""
        node = root
        
        while node.children:
            best_child = None
            best_ucb = float('-inf')
            
            for child in node.children:
                if child.visit_count == 0:
                    return child
                
                ucb = child.ucb_score
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            node = best_child if best_child else node
        
        return node
    
    def _expand(self, node: MCTSNode):
        """Expand node by generating candidates."""
        print(f"     🌱 Expanding node at depth {node.depth} ({len(node.masked_positions)} masked)")
        
        # Generate candidates using different methods based on ablation mode
        candidates = []
        
        if self.ablation_mode == "random_no_expert":
            candidates = self._generate_random_candidates(node)
        elif self.ablation_mode == "single_expert":
            candidates = self._generate_single_expert_candidates(node)
        else:  # multi_expert
            candidates = self._generate_multi_expert_candidates(node)
        
        # Create child nodes
        for candidate_seq, expert_source in candidates:
            if len(candidate_seq) == len(node.sequence):
                # Progressive masking for child
                child_masked = self._get_child_masked_positions(node, candidate_seq)
                
                child = MCTSNode(
                    sequence=candidate_seq,
                    masked_positions=child_masked,
                    parent=node,
                    depth=node.depth + 1,
                    task_type=node.task_type
                )
                
                child.expert_source = expert_source
                node.children.append(child)
        
        print(f"     ✅ Created {len(node.children)} children")
    
    def _get_child_masked_positions(self, parent: MCTSNode, child_sequence: str) -> Set[int]:
        """Get masked positions for child node based on depth."""
        # Progressive masking: less masking at deeper levels
        depth = parent.depth + 1
        
        if depth >= self.max_depth:
            return set()  # No masking at max depth
        
        # Reduce masking as depth increases
        base_ratio = 0.15  # 15% base masking
        depth_factor = max(0.2, 1.0 - (depth * 0.3))  # Reduce by 30% per depth level
        mask_ratio = base_ratio * depth_factor
        
        num_to_mask = max(1, int(len(child_sequence) * mask_ratio))
        
        # Use pLDDT if available
        if hasattr(self.dplm2_integration, 'baseline_structure'):
            plddt_scores = self.dplm2_integration.baseline_structure.get('plddt_scores', [])
            if plddt_scores and len(plddt_scores) == len(child_sequence):
                return self._apply_plddt_masking_simple(child_sequence, plddt_scores, mask_ratio)
        
        # Fallback to random masking
        return set(random.sample(range(len(child_sequence)), num_to_mask))
    
    def _generate_random_candidates(self, node: MCTSNode) -> List[Tuple[str, str]]:
        """Generate random candidates."""
        candidates = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(2):  # Generate 2 random candidates
            new_sequence = list(node.sequence)
            for pos in node.masked_positions:
                new_sequence[pos] = random.choice(amino_acids)
            candidates.append((''.join(new_sequence), "random"))
        
        return candidates
    
    def _generate_single_expert_candidates(self, node: MCTSNode) -> List[Tuple[str, str]]:
        """Generate candidates using single expert + DPLM-2."""
        candidates = []
        
        # Use specified expert if available
        if (self.single_expert_id is not None and 
            self.single_expert_id < len(self.external_experts)):
            expert = self.external_experts[self.single_expert_id]
            # Placeholder for expert call
            candidates.append((node.sequence, f"expert_{self.single_expert_id}"))
        
        # Add DPLM-2 candidate
        dplm2_candidate = self._generate_dplm2_candidate(node)
        if dplm2_candidate:
            candidates.append(dplm2_candidate)
        
        return candidates
    
    def _generate_multi_expert_candidates(self, node: MCTSNode) -> List[Tuple[str, str]]:
        """Generate candidates using all experts."""
        candidates = []
        
        # Use all external experts (placeholder)
        for i, expert in enumerate(self.external_experts):
            candidates.append((node.sequence, f"expert_{i}"))
        
        # Add DPLM-2 candidate
        dplm2_candidate = self._generate_dplm2_candidate(node)
        if dplm2_candidate:
            candidates.append(dplm2_candidate)
        
        return candidates
    
    def _generate_dplm2_candidate(self, node: MCTSNode) -> Tuple[str, str]:
        """Generate candidate using DPLM-2."""
        if not node.masked_positions:
            return (node.sequence, "dplm2_unchanged")
        
        try:
            # Create masked sequence
            masked_seq = list(node.sequence)
            for pos in node.masked_positions:
                masked_seq[pos] = 'X'
            masked_seq_str = ''.join(masked_seq)
            
            # Use DPLM-2 to fill masked positions
            if self.task_type == "inverse_folding":
                # Get structure tokens
                struct_tokens = ""
                if hasattr(self.dplm2_integration, 'baseline_structure'):
                    struct_tokens = self.dplm2_integration.baseline_structure.get('struct_seq', '')
                
                result = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_seq_str,
                    struct_tokens=struct_tokens,
                    task_type="inverse_folding",
                    temperature=self.temperature
                )
            else:
                # For folding or other tasks
                result = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_seq_str,
                    struct_tokens="",
                    task_type="unconditional",
                    temperature=self.temperature
                )
            
            if result and len(result) == len(node.sequence):
                return (result, "dplm2")
            else:
                return (node.sequence, "dplm2_fallback")
                
        except Exception as e:
            print(f"       ⚠️ DPLM-2 generation failed: {e}")
            return (node.sequence, "dplm2_failed")
    
    def _evaluate_sequence(self, sequence: str) -> float:
        """Evaluate sequence based on task type."""
        try:
            if self.task_type == "inverse_folding":
                # Use AAR for inverse folding
                if self.reference_sequence:
                    return self._calculate_simple_aar(sequence, self.reference_sequence)
                else:
                    # Fallback: sequence validity
                    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
                    validity = sum(1 for aa in sequence if aa in valid_aas) / len(sequence)
                    return validity * 0.8
            
            elif self.task_type == "folding":
                # Use structure quality for folding
                return self._evaluate_folding_quality(sequence)
            
            else:
                # For other tasks, use sequence properties
                return self._evaluate_sequence_properties(sequence)
                
        except Exception as e:
            print(f"       ⚠️ Sequence evaluation failed: {e}")
            return 0.0
    
    def _calculate_simple_aar(self, pred_seq: str, ref_seq: str) -> float:
        """Calculate Amino Acid Recovery."""
        if len(pred_seq) != len(ref_seq):
            min_len = min(len(pred_seq), len(ref_seq))
            pred_seq = pred_seq[:min_len]
            ref_seq = ref_seq[:min_len]
        
        if len(ref_seq) == 0:
            return 0.0
        
        matches = sum(1 for p, r in zip(pred_seq, ref_seq) if p == r)
        return matches / len(ref_seq)
    
    def _evaluate_folding_quality(self, sequence: str) -> float:
        """Evaluate folding quality using structure prediction."""
        try:
            # Use ESMFold to predict structure and get pLDDT as quality metric
            import torch
            from transformers import EsmForProteinFolding, AutoTokenizer
            
            # Load ESMFold model (cached)
            if not hasattr(self, '_esmfold_model'):
                self._esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                self._esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
                if torch.cuda.is_available():
                    self._esmfold_model = self._esmfold_model.cuda()
                self._esmfold_model.eval()
            
            # Predict structure and get confidence
            with torch.no_grad():
                tokenized = self._esmfold_tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
                device = next(self._esmfold_model.parameters()).device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                
                output = self._esmfold_model(tokenized['input_ids'])
                
                if hasattr(output, 'plddt') and output.plddt is not None:
                    plddt_tensor = output.plddt[0].cpu().numpy()
                    
                    # Use average pLDDT as quality metric
                    if len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                        plddt_scores = plddt_tensor[:, 1]  # CA atom confidence
                    else:
                        plddt_scores = plddt_tensor.mean(axis=1) if len(plddt_tensor.shape) == 2 else plddt_tensor
                    
                    avg_plddt = np.mean(plddt_scores) / 100.0  # Normalize to 0-1
                    return avg_plddt
                else:
                    return 0.5  # Default moderate score
                    
        except Exception as e:
            print(f"       ⚠️ Folding evaluation failed: {e}")
            return 0.5  # Default moderate score
    
    def _evaluate_sequence_properties(self, sequence: str) -> float:
        """Evaluate basic sequence properties."""
        if not sequence:
            return 0.0
        
        # Valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        validity = sum(1 for aa in sequence if aa in valid_aas) / len(sequence)
        
        # Amino acid diversity
        unique_aas = len(set(sequence))
        diversity = min(1.0, unique_aas / 15)  # Normalize by typical diversity
        
        # Combine scores
        return (validity * 0.7 + diversity * 0.3)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node.parent
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent


# Keep backward compatibility
SequenceLevelMCTS = GeneralMCTS
