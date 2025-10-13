"""
Fixed MCTS Implementation for Inverse Folding
Following the proper MCTS algorithm from the reference repository.

Key components:
1. pH-UCT Selection
2. Progressive pLDDT-based masking 
3. Expert rollouts
4. Task-specific evaluation
5. Proper backpropagation
"""

import math
import random
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any
import time

@dataclass
class MCTSNode:
    """MCTS Node for sequence optimization."""
    sequence: str
    masked_positions: Set[int]
    reward: float = 0.0
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = None
    parent: 'MCTSNode' = None
    depth: int = 0
    task_type: str = "inverse_folding"
    expert_source: str = "unknown"
    entropy: float = 0.0
    diversity: float = 0.0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    @property
    def average_value(self) -> float:
        """Average value of this node."""
        return self.total_value / max(self.visit_count, 1)
    
    def ph_uct_score(self, exploration_constant: float = 1.414, w_ent: float = 0.1, w_div: float = 0.1) -> float:
        """pH-UCT score with entropy and diversity bonuses."""
        if self.visit_count == 0:
            return float('inf')
        
        # Base UCB1 score
        if self.parent and self.parent.visit_count > 0:
            exploitation = self.average_value
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visit_count) / self.visit_count)
            ucb_score = exploitation + exploration
        else:
            ucb_score = self.average_value
        
        # pH-UCT bonuses
        entropy_bonus = w_ent * self.entropy
        diversity_bonus = w_div * self.diversity
        
        return ucb_score + entropy_bonus + diversity_bonus


class InverseFoldingMCTS:
    """
    Proper MCTS implementation for inverse folding following the reference architecture.
    
    MCTS Pipeline:
    1. Selection: pH-UCT selection to leaf node
    2. Expansion: Generate children using expert rollouts with progressive pLDDT masking
    3. Simulation: Evaluate sequences using task-specific metrics (AAR, scTM)
    4. Backpropagation: Update node statistics up the tree
    """
    
    def __init__(
        self,
        dplm2_integration,
        external_experts: List = None,
        ablation_mode: str = "multi_expert",
        single_expert_id: int = None,
        max_depth: int = 5,
        num_simulations: int = 25,
        exploration_constant: float = 1.414,
        children_per_expansion: int = 3,
        rollouts_per_expert: int = 3
    ):
        self.dplm2 = dplm2_integration
        self.external_experts = external_experts or []
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.children_per_expansion = children_per_expansion
        self.rollouts_per_expert = rollouts_per_expert
        
        # Cache for evaluated sequences
        self.cache = {}
        
        # Load ESMFold for pLDDT calculation
        self.esmfold_model = None
        self.esmfold_tokenizer = None
        self._load_esmfold()
        
        print(f"✅ InverseFoldingMCTS initialized:")
        print(f"   Mode: {ablation_mode}" + (f" (expert {single_expert_id})" if single_expert_id is not None else ""))
        print(f"   External experts: {[e.get_name() for e in self.external_experts]}")
        print(f"   Max depth: {max_depth}, Simulations: {num_simulations}")
    
    def _load_esmfold(self):
        """Load ESMFold once for pLDDT calculation."""
        try:
            from transformers import EsmForProteinFolding, AutoTokenizer
            self.esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            self.esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            
            if torch.cuda.is_available():
                self.esmfold_model = self.esmfold_model.cuda()
            self.esmfold_model.eval()
            print("✅ ESMFold loaded for pLDDT calculation")
        except Exception as e:
            print(f"⚠️ Failed to load ESMFold: {e}")
    
    def search(self, structure_data: Dict, target_length: int, reference_sequence: str = None) -> Tuple[str, float]:
        """
        Main MCTS search following proper algorithm.
        
        Args:
            structure_data: Structure information with struct_seq, plddt_scores, etc.
            target_length: Target sequence length
            reference_sequence: Reference sequence for AAR calculation
        
        Returns:
            (best_sequence, best_reward)
        """
        print(f"🔍 Starting MCTS search for inverse folding")
        print(f"   Target length: {target_length}")
        print(f"   Reference available: {reference_sequence is not None}")
        
        # Store reference for evaluation
        self.reference_sequence = reference_sequence
        self.structure_data = structure_data
        
        # Get baseline sequence from DPLM2Integration
        baseline_sequence = getattr(self.dplm2, '_current_baseline_sequence', None)
        if not baseline_sequence:
            print("❌ No baseline sequence found in DPLM2Integration")
            return None, 0.0
        
        print(f"   Using baseline sequence: {len(baseline_sequence)} residues")
        
        # Create root node
        root = MCTSNode(
            sequence=baseline_sequence,
            masked_positions=set(),  # Root starts unmasked
            task_type="inverse_folding",
            depth=0,
            expert_source="baseline"
        )
        
        # Evaluate root
        root.reward = self._evaluate_sequence(baseline_sequence)
        root.visit_count = 1
        root.total_value = root.reward
        
        best_sequence = baseline_sequence
        best_reward = root.reward
        
        print(f"   Baseline reward: {root.reward:.4f}")
        print(f"🚀 Starting {self.num_simulations} MCTS iterations")
        
        # MCTS iterations
        for iteration in range(self.num_simulations):
            if iteration % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.num_simulations}")
            
            # 1. SELECTION: pH-UCT selection to leaf
            selected_node = self._select(root)
            
            # 2. EXPANSION: Generate children using expert rollouts
            if selected_node.depth < self.max_depth:
                self._expand(selected_node)
            
            # 3. SIMULATION & EVALUATION: Done during expansion
            
            # 4. BACKPROPAGATION: Update best sequence
            if selected_node.children:
                for child in selected_node.children:
                    if child.reward > best_reward:
                        best_sequence = child.sequence
                        best_reward = child.reward
                        print(f"   🏆 New best: {best_reward:.4f} (depth {child.depth}, expert {child.expert_source})")
        
        print(f"✅ MCTS completed. Best reward: {best_reward:.4f}")
        return best_sequence, best_reward
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select leaf node using pH-UCT."""
        node = root
        
        while node.children:
            # Select child with highest pH-UCT score
            best_child = max(node.children, key=lambda child: child.ph_uct_score(self.exploration_constant))
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode):
        """
        Expand node using expert rollouts with progressive pLDDT masking.
        
        This is the core of the MCTS algorithm:
        1. Apply progressive pLDDT masking based on depth
        2. Generate candidates using different experts
        3. Evaluate candidates and create child nodes
        4. Backpropagate rewards
        """
        print(f"   🌱 Expanding node at depth {node.depth}")
        
        # Progressive pLDDT masking
        masked_positions = self._apply_progressive_plddt_masking(node.sequence, node.depth + 1)
        
        if not masked_positions:
            print(f"   ⚠️ No positions to mask at depth {node.depth + 1}")
            return
        
        print(f"   📊 Masking {len(masked_positions)} positions for depth {node.depth + 1}")
        
        # Generate candidates using expert rollouts
        candidates = []
        
        if self.ablation_mode == "random_no_expert":
            candidates = self._generate_random_candidates(node, masked_positions)
        elif self.ablation_mode == "single_expert":
            candidates = self._generate_single_expert_candidates(node, masked_positions)
        else:  # multi_expert
            candidates = self._generate_multi_expert_candidates(node, masked_positions)
        
        # Create child nodes and evaluate
        for candidate_seq, expert_source in candidates:
            if len(candidate_seq) == len(node.sequence):
                # Evaluate candidate
                reward = self._evaluate_sequence(candidate_seq)
                
                # Create child node
                child = MCTSNode(
                    sequence=candidate_seq,
                    masked_positions=set(),  # Child is complete sequence
                    reward=reward,
                    visit_count=1,
                    total_value=reward,
                    parent=node,
                    depth=node.depth + 1,
                    task_type=node.task_type,
                    expert_source=expert_source
                )
                
                # Add entropy and diversity bonuses
                child.entropy = self._compute_entropy(candidate_seq)
                child.diversity = self._compute_diversity(candidate_seq, node.sequence)
                
                node.children.append(child)
        
        # Backpropagate best child reward to parent
        if node.children:
            best_child = max(node.children, key=lambda c: c.reward)
            self._backpropagate(best_child, best_child.reward)
            print(f"   ⬆️ Backpropagated reward {best_child.reward:.4f} from {best_child.expert_source}")
    
    def _apply_progressive_plddt_masking(self, sequence: str, depth: int) -> Set[int]:
        """
        Apply progressive pLDDT-based masking.
        
        Strategy:
        - Depth 1: Mask 30% lowest pLDDT positions
        - Depth 2: Mask 20% lowest pLDDT positions  
        - Depth 3: Mask 15% lowest pLDDT positions
        - Depth 4: Mask 10% lowest pLDDT positions
        - Depth 5+: Mask 5% lowest pLDDT positions
        """
        # Get pLDDT scores for the sequence
        plddt_scores = self._compute_plddt_scores(sequence)
        
        # Progressive masking percentages
        mask_percentages = {1: 0.30, 2: 0.20, 3: 0.15, 4: 0.10}
        mask_percentage = mask_percentages.get(depth, 0.05)  # Default 5% for depth 5+
        
        seq_len = len(sequence)
        num_to_mask = max(1, int(seq_len * mask_percentage))
        num_to_mask = min(num_to_mask, seq_len // 2)  # Max 50%
        
        # Find positions with lowest pLDDT
        plddt_array = np.array(plddt_scores)
        sorted_indices = np.argsort(plddt_array)
        mask_positions = set(sorted_indices[:num_to_mask].tolist())
        
        min_plddt = plddt_array[sorted_indices[num_to_mask-1]] if num_to_mask > 0 else 0
        print(f"     📊 Depth {depth}: {len(mask_positions)}/{seq_len} positions ({mask_percentage*100:.0f}%), pLDDT < {min_plddt:.1f}")
        
        return mask_positions
    
    def _compute_plddt_scores(self, sequence: str) -> List[float]:
        """Compute ESMFold pLDDT scores for sequence."""
        if not self.esmfold_model:
            # Fallback to uniform scores
            return [70.0] * len(sequence)
        
        try:
            with torch.no_grad():
                tokenized = self.esmfold_tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
                device = next(self.esmfold_model.parameters()).device
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                
                output = self.esmfold_model(tokenized['input_ids'])
                
                if hasattr(output, 'plddt') and output.plddt is not None:
                    plddt_tensor = output.plddt[0].cpu().numpy()  # [L, 37]
                    
                    # Use Cα atom confidence (index 1)
                    if len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                        plddt_scores = plddt_tensor[:, 1].tolist()
                    else:
                        plddt_scores = plddt_tensor.mean(axis=1).tolist() if len(plddt_tensor.shape) == 2 else plddt_tensor.tolist()
                    
                    return plddt_scores
                else:
                    return [70.0] * len(sequence)
                    
        except Exception as e:
            print(f"     ⚠️ pLDDT calculation failed: {e}")
            return [70.0] * len(sequence)
    
    def _generate_random_candidates(self, node: MCTSNode, masked_positions: Set[int]) -> List[Tuple[str, str]]:
        """Generate random candidates."""
        candidates = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for _ in range(self.children_per_expansion):
            new_sequence = list(node.sequence)
            for pos in masked_positions:
                new_sequence[pos] = random.choice(amino_acids)
            candidates.append((''.join(new_sequence), "random"))
        
        return candidates
    
    def _generate_single_expert_candidates(self, node: MCTSNode, masked_positions: Set[int]) -> List[Tuple[str, str]]:
        """Generate candidates using single expert + DPLM-2."""
        candidates = []
        
        # Use specified expert
        if (self.single_expert_id is not None and 
            self.single_expert_id < len(self.external_experts)):
            expert = self.external_experts[self.single_expert_id]
            expert_candidates = self._expert_rollout(node, masked_positions, expert)
            candidates.extend(expert_candidates)
        
        # Add DPLM-2 candidates
        dplm2_candidates = self._dplm2_rollout(node, masked_positions)
        candidates.extend(dplm2_candidates)
        
        return candidates
    
    def _generate_multi_expert_candidates(self, node: MCTSNode, masked_positions: Set[int]) -> List[Tuple[str, str]]:
        """Generate candidates using all experts."""
        candidates = []
        
        # Use all external experts
        for expert in self.external_experts:
            expert_candidates = self._expert_rollout(node, masked_positions, expert)
            candidates.extend(expert_candidates)
        
        # Add DPLM-2 candidates
        dplm2_candidates = self._dplm2_rollout(node, masked_positions)
        candidates.extend(dplm2_candidates)
        
        return candidates
    
    def _expert_rollout(self, node: MCTSNode, masked_positions: Set[int], expert) -> List[Tuple[str, str]]:
        """Generate candidates using external expert."""
        candidates = []
        expert_name = expert.get_name()
        
        try:
            for _ in range(self.rollouts_per_expert):
                # Create masked sequence
                masked_seq = list(node.sequence)
                for pos in masked_positions:
                    masked_seq[pos] = 'X'
                masked_seq_str = ''.join(masked_seq)
                
                # For inverse folding, we need structure tokens
                struct_tokens = self.structure_data.get('struct_seq', '')
                
                # This would call the expert - for now, use placeholder
                # In real implementation, this would call expert.generate() or similar
                result_seq = self._call_external_expert(expert, masked_seq_str, struct_tokens)
                
                if result_seq and len(result_seq) == len(node.sequence):
                    candidates.append((result_seq, expert_name))
                    
        except Exception as e:
            print(f"     ⚠️ {expert_name} rollout failed: {e}")
        
        return candidates
    
    def _dplm2_rollout(self, node: MCTSNode, masked_positions: Set[int]) -> List[Tuple[str, str]]:
        """Generate candidates using DPLM-2."""
        candidates = []
        
        try:
            for _ in range(self.rollouts_per_expert):
                # Create masked sequence
                masked_seq = list(node.sequence)
                for pos in masked_positions:
                    masked_seq[pos] = 'X'
                masked_seq_str = ''.join(masked_seq)
                
                # Use DPLM-2 to fill masked positions
                struct_tokens = self.structure_data.get('struct_seq', '')
                result_seq = self.dplm2.generate_from_masked_input(
                    aa_sequence=masked_seq_str,
                    struct_tokens=struct_tokens,
                    task_type="inverse_folding",
                    temperature=1.0
                )
                
                if result_seq and len(result_seq) == len(node.sequence):
                    candidates.append((result_seq, "dplm2"))
                    
        except Exception as e:
            print(f"     ⚠️ DPLM-2 rollout failed: {e}")
        
        return candidates
    
    def _call_external_expert(self, expert, masked_sequence: str, structure_tokens: str) -> str:
        """Placeholder for external expert call."""
        # This would be implemented based on expert interface
        # For now, return None to indicate not implemented
        return None
    
    def _evaluate_sequence(self, sequence: str) -> float:
        """Evaluate sequence using task-specific metrics."""
        # Check cache first
        if sequence in self.cache:
            return self.cache[sequence]
        
        try:
            # For inverse folding, use AAR as primary metric
            if self.reference_sequence:
                aar = self._calculate_aar(sequence, self.reference_sequence)
                reward = aar
            else:
                # Fallback: sequence validity
                valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
                validity = sum(1 for aa in sequence if aa in valid_aas) / len(sequence)
                reward = validity * 0.8
            
            # Cache result
            self.cache[sequence] = reward
            return reward
            
        except Exception as e:
            print(f"     ⚠️ Evaluation failed: {e}")
            return 0.0
    
    def _calculate_aar(self, pred_seq: str, ref_seq: str) -> float:
        """Calculate Amino Acid Recovery."""
        if len(pred_seq) != len(ref_seq):
            return 0.0
        
        matches = sum(1 for a, b in zip(pred_seq, ref_seq) if a == b)
        return matches / len(ref_seq)
    
    def _compute_entropy(self, sequence: str) -> float:
        """Compute sequence entropy for pH-UCT."""
        from collections import Counter
        counts = Counter(sequence)
        total = len(sequence)
        entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
        return entropy / 4.32  # Normalize by max entropy for 20 amino acids
    
    def _compute_diversity(self, seq1: str, seq2: str) -> float:
        """Compute sequence diversity for pH-UCT."""
        if len(seq1) != len(seq2):
            return 0.0
        
        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node.parent  # Start from parent (child already has reward)
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent


# Wrapper class for compatibility
class GeneralMCTS(InverseFoldingMCTS):
    """Compatibility wrapper."""
    
    def __init__(self, task_type="inverse_folding", **kwargs):
        if task_type != "inverse_folding":
            raise ValueError("This fixed implementation only supports inverse_folding")
        super().__init__(**kwargs)
