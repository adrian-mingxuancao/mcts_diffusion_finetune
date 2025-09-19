"""
Clean MCTS implementation with complete sequences only.
Single-source design that enforces one invariant: nodes store complete sequences, never masked.
"""

import math
import random
import hashlib
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import numpy as np

from .mcts_utils import (
    apply_patch, compute_mask_schedule, compute_sequence_hash, 
    compute_hamming_distance, compute_fast_aar, ph_uct_score, SequenceCache
)
from .dplm2_integration import DPLM2Integration

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class MCTSNode:
    """
    MCTS Node that stores ONLY complete sequences (no X tokens ever).
    
    Design invariant: sequence field never contains 'X' tokens.
    Masking happens only during expansion as temporary variables.
    """
    sequence: str
    depth: int = 0
    parent: Optional['MCTSNode'] = None
    children: Optional[List['MCTSNode']] = None
    visit_count: int = 0
    value_sum: float = 0.0
    
    # PH-UCT bonuses (cached during expansion)
    entropy_proposals: float = 0.0
    novelty_vs_parent: float = 0.0
    
    def __post_init__(self):
        """Enforce complete sequence invariant."""
        if 'X' in self.sequence:
            raise ValueError(f"MCTSNode must store complete sequences only. Found X tokens in: {self.sequence[:50]}...")
        if self.children is None:
            self.children = []
    
    @property
    def average_value(self) -> float:
        """Average reward value."""
        return self.value_sum / max(1, self.visit_count)
    
    def ucb_score(self, parent_visits: int, exploration_constant: float = 1.414) -> float:
        """PH-UCT score for selection."""
        if self.visit_count == 0:
            return float('inf')
        return ph_uct_score(
            self.average_value,
            self.visit_count,
            parent_visits,
            c=exploration_constant,
            w_ent=0.1,
            w_div=0.1,
            entropy_proposals=self.entropy_proposals,
            novelty_vs_parent=self.novelty_vs_parent
        )
    
    @property
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0


class GeneralMCTS:
    """
    General MCTS implementation for protein sequence optimization.
    
    Uses complete sequences only (no masked sequences in nodes).
    Masking happens only during expansion as temporary variables.
    """
    
    def __init__(self, dplm2_integration, baseline_structure: Dict, reference_sequence: str, 
                 max_depth: int = 3, exploration_constant: float = 1.414,
                 num_children_select: int = 2, k_rollouts_per_expert: int = 2,
                 ablation_mode: str = "multi_expert", single_expert_id: int = 0,
                 backup_rule: str = "sum", task_type: str = "inverse_folding",
                 reference_coords: np.ndarray = None, device: str = "cuda"):
        """
        Initialize MCTS for sequence-level optimization.
        
        Args:
            dplm2_integration: DPLM-2 model integration
            reference_sequence: Target sequence for AAR calculation (inverse folding) or initial sequence (folding)
            baseline_structure: Structure data with coordinates/pLDDT
            max_depth: Maximum search depth
            exploration_constant: UCB exploration parameter
            num_children_select: Number of top children to keep per expansion
            k_rollouts_per_expert: Number of rollouts per expert
            ablation_mode: "random_no_expert", "single_expert", or "multi_expert"
            single_expert_id: Expert ID for single expert mode
            backup_rule: "max" for max backup (MCTS), "sum" for sum backup (Monte Carlo)
            task_type: "inverse_folding" or "folding" - determines reward calculation
            device: CUDA device for DPLM-2 integration
        """
        # Initialize DPLM-2 integration if not provided
        if dplm2_integration is None:
            self.dplm2_integration = DPLM2Integration(device=device)
        else:
            self.dplm2_integration = dplm2_integration
        self.reference_sequence = reference_sequence
        self._baseline_structure = baseline_structure
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.num_children_select = num_children_select
        self.k_rollouts_per_expert = k_rollouts_per_expert
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        self.backup_rule = backup_rule
        self.task_type = task_type
        self.reference_coords = reference_coords  # For folding task RMSD/TM-score calculation

        # Derived parameters
        self.num_candidates_per_expansion = 6  # For random mode
        
        # Cache for sequence evaluation
        self.sequence_cache = SequenceCache()
        
        # Track seen sequences to avoid duplicates
        self.seen_sequences = set()
        
        print(f"🌳 GeneralMCTS initialized:")
        print(f"   Task type: {task_type}")
        print(f"   Max depth: {max_depth}")
        print(f"   Candidates per expansion: {self.num_candidates_per_expansion}")
        print(f"   Ablation mode: {ablation_mode}")
        print(f"   Backup rule: {backup_rule}")
        if ablation_mode == "single_expert":
            print(f"   Single expert ID: {single_expert_id}")
        print(f"   Reference sequence length: {len(reference_sequence) if reference_sequence else 'None'}")
        
    def search(self, initial_sequence: str, num_iterations: int = 100, max_depth: int = None,
               exploration_constant: float = 1.414, structure: Dict = None) -> 'MCTSNode':
        """
        Perform MCTS search starting from complete initial sequence.
        
        Args:
            initial_sequence: Complete starting sequence (no X tokens)
            num_iterations: Number of MCTS iterations
            max_depth: Maximum tree depth (uses instance default if None)
            exploration_constant: UCB exploration parameter
            structure: Optional structure data to merge with baseline
            
        Returns:
            Root node of the search tree
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        # Merge structure data if provided
        if structure:
            self._baseline_structure.update(structure)
        elif not hasattr(self, '_baseline_structure') or not self._baseline_structure:
            self._baseline_structure = {}
        
        # Root always stores complete initial sequence
        root = MCTSNode(sequence=initial_sequence, depth=0)
        
        print(f"🎯 MCTS search starting from complete sequence (length: {len(initial_sequence)})")
        
        for iteration in range(num_iterations):
            # Selection: find leaf node using UCB
            leaf = self._select(root, exploration_constant)
            
            # Terminal check: depth or budget based only
            if self._is_terminal(leaf, max_depth):
                # Evaluate terminal node
                reward = self._simulate(leaf)
                self._backpropagate(leaf, reward)
                print(f"[iter {iteration+1}] selected depth={leaf.depth}, terminal node")
                continue
            
            # Expansion: create children using experts (masking happens internally)
            children = self._expand(leaf)
            
            if not children:
                # No children created, evaluate current leaf
                reward = self._simulate(leaf)
                self._backpropagate(leaf, reward)
                print(f"[iter {iteration+1}] selected depth={leaf.depth}, expansion failed")
                continue
            
            # Add children to leaf
            leaf.children.extend(children)
            
            # Log AFTER expansion to show tree growth
            print(f"[iter {iteration+1}] selected depth={leaf.depth}, expanded to {len(leaf.children)} children")
            
            # Simulation: evaluate each new child
            for child in children:
                reward = self._simulate(child)
                self._backpropagate(child, reward)
            
            if iteration % 10 == 0:
                best_node = self._get_best_node(root)
                print(f"   Best so far: depth={best_node.depth}, reward={best_node.average_value:.3f}")
        
        # Return root node (contains full search tree)
        return root
    
    def _select(self, node: MCTSNode, exploration_constant: float) -> MCTSNode:
        """Select leaf node using PH-UCT traversal with proper UCB calculation."""
        while not node.is_leaf:
            # Find unvisited children first
            unvisited = [child for child in node.children if child.visit_count == 0]
            if unvisited:
                return random.choice(unvisited)
            
            # All children visited, use PH-UCT selection
            best_child = None
            best_score = float('-inf')
            
            for child in node.children:
                # Use the centralized UCB score calculation
                score = child.ucb_score(node.visit_count, exploration_constant)
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node by generating children using DPLM-2 experts.
        For folding: mask structure tokens, keep AA sequence constant
        For inverse folding: mask AA sequence, keep structure tokens constant
        """
        seq = node.sequence
        
        if self.task_type == "folding":
            return self._expand_folding(node)
        elif self.task_type == "motif_scaffolding":
            return self._expand_motif_scaffolding(node)
        else:
            return self._expand_inverse_folding(node)
    
    def _expand_folding(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node for folding task: mask structure tokens, generate new structures.
        """
        seq = node.sequence
        
        # Get current structure tokens from ESMFold baseline or previous generation
        if hasattr(node, 'structure_tokens') and node.structure_tokens:
            current_struct_tokens = node.structure_tokens
        else:
            # Generate structure tokens from current sequence using ESMFold
            coords = self._generate_esmfold_baseline(seq)
            current_struct_tokens = self._coords_to_structure_tokens(coords)
        
        print(f"🔍 Current structure tokens: {len(current_struct_tokens)} chars")
        
        # Use pLDDT scores from detokenization or fallback
        if hasattr(self, 'last_plddt_scores') and self.last_plddt_scores is not None:
            plddt_scores = self.last_plddt_scores
            print(f"✅ Using detokenized pLDDT: mean={np.mean(plddt_scores):.3f}")
        else:
            # Fallback to baseline pLDDT
            baseline_plddt = self._baseline_structure.get('plddt_scores', [])
            if baseline_plddt is not None and len(baseline_plddt) == len(seq):
                plddt_scores = baseline_plddt
                print(f"✅ Using baseline pLDDT: mean={np.mean(plddt_scores):.3f}")
            else:
                # Generate realistic confidence scores with some variation
                import random
                plddt_scores = [random.uniform(40.0, 90.0) for _ in range(len(seq))]  # Realistic range
                print(f"⚠️ Using random pLDDT fallback: mean={np.mean(plddt_scores):.3f}")
        
        # Compute mask positions based on low confidence structure regions
        mask = compute_mask_schedule(seq, plddt_scores, node.depth, self.max_depth)
        print(f"🎯 Structure masking strategy: {len(mask)} positions ({len(mask)/len(seq)*100:.1f}%) at depth {node.depth}")
        
        # Create masked structure tokens (not AA sequence!)
        masked_struct_tokens = self._mask_structure_tokens(current_struct_tokens, mask)
        
        # 🔍 DEBUG: Show structure token masking
        print(f"🔍 STRUCTURE TOKEN MASKING DEBUG:")
        print(f"   Original struct: {current_struct_tokens[:100]}...")
        print(f"   Masked struct:   {masked_struct_tokens[:100]}...")
        print(f"   Mask positions: {sorted(list(mask))[:10]}...")
        print(f"   AA sequence (unchanged): {seq[:50]}...")
        
        return self._generate_folding_candidates(seq, masked_struct_tokens, mask)
    
    def _expand_inverse_folding(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node for inverse folding task: mask AA sequence, generate new sequences.
        """
        seq = node.sequence
        
        # Compute dynamic pLDDT for this sequence
        print(f"🔍 Computing pLDDT for sequence length {len(seq)}")
        current_plddt = self._compute_sequence_plddt(seq)
        if current_plddt and len(current_plddt) == len(seq):
            print(f"✅ Using dynamic pLDDT: mean={np.mean(current_plddt):.3f}")
            mask = compute_mask_schedule(seq, current_plddt, node.depth, self.max_depth)
        else:
            # Fallback to baseline pLDDT
            print(f"⚠️ Dynamic pLDDT failed, using baseline pLDDT")
            baseline_plddt = self._baseline_structure.get('plddt_scores', [])
            if baseline_plddt is not None and len(baseline_plddt) == len(seq):
                # Convert numpy array to list if needed
                if hasattr(baseline_plddt, 'tolist'):
                    baseline_plddt = baseline_plddt.tolist()
                elif not isinstance(baseline_plddt, list):
                    baseline_plddt = list(baseline_plddt)
                print(f"✅ Using baseline pLDDT: mean={np.mean(baseline_plddt):.3f}")
                mask = compute_mask_schedule(seq, baseline_plddt, node.depth, self.max_depth)
            else:
                # Final fallback: small random mask
                print(f"⚠️ No pLDDT available, using random masking")
                num_mask = max(2, int(len(seq) * 0.05))
                mask = set(random.sample(range(len(seq)), min(num_mask, len(seq))))
        
        print(f"🎯 AA masking strategy: {len(mask)} positions ({len(mask)/len(seq)*100:.1f}%) at depth {node.depth}")
        
        # Create masked sequence for expert generation (temporary variable)
        masked = list(seq)
        for i in mask:
            masked[i] = 'X'
        masked_seq = ''.join(masked)
        
        # 🔍 DEBUG: Show partial masking for verification
        print(f"🔍 PARTIAL MASKING DEBUG:")
        print(f"   Original:  {seq[:50]}...")
        print(f"   Masked:    {masked_seq[:50]}...")
        print(f"   Mask positions: {sorted(list(mask))[:10]}...")
        print(f"   X count: {masked_seq.count('X')}/{len(masked_seq)} ({masked_seq.count('X')/len(masked_seq)*100:.1f}%)")
        
        return self._generate_inverse_folding_candidates(seq, masked_seq, mask)
    
    def _mask_structure_tokens(self, structure_tokens: str, mask_positions: set) -> str:
        """
        Mask structure tokens at specified positions.
        Structure tokens are comma-separated, so we need to handle them properly.
        """
        try:
            # Parse structure tokens
            if ',' in structure_tokens:
                # Comma-separated token IDs
                token_list = structure_tokens.split(',')
            else:
                # Space-separated or other format - try to parse
                token_list = structure_tokens.strip().split()
            
            # Mask tokens at specified positions
            masked_tokens = []
            for i, token in enumerate(token_list):
                if i in mask_positions:
                    masked_tokens.append('<mask_struct>')  # Use structure mask token
                else:
                    masked_tokens.append(token.strip())
            
            # Rejoin with commas
            masked_struct_tokens = ','.join(masked_tokens)
            print(f"   🎯 Masked {len(mask_positions)} structure tokens out of {len(token_list)}")
            return masked_struct_tokens
            
        except Exception as e:
            print(f"   ❌ Structure token masking failed: {e}")
            # Fallback: return original tokens
            return structure_tokens
    
    def _generate_folding_candidates(self, sequence: str, masked_struct_tokens: str, mask_positions: set) -> List[MCTSNode]:
        """
        Generate candidate structures for folding task using DPLM experts.
        """
        candidates: List[str] = []

        # =========================
        # ABLATION BRANCHES FOR FOLDING
        # =========================

        # (A) RANDOM ABLATION
        if self.ablation_mode == "random_no_expert":
            print(f"🎲 Folding Ablation: RANDOM (no expert). Rollouts={self.k_rollouts_per_expert}")
            candidates = []
            
            # For folding, generate random structure tokens
            for _ in range(self.k_rollouts_per_expert):
                structure_tokens = self._generate_random_structure_tokens(len(sequence))
                if structure_tokens:
                    rmsd, tm_score = self._compute_folding_metrics(structure_tokens)
                    reward = self._compute_structure_reward(rmsd, tm_score)
                    candidates.append((reward, sequence, structure_tokens, rmsd, tm_score))
            
            if candidates:
                candidates.sort(reverse=True, key=lambda x: x[0])
                top_candidates = candidates[:3]  # Keep full candidate info
            else:
                top_candidates = []

        # (B) SINGLE EXPERT — one expert only, spawn children directly
        elif self.ablation_mode == "single_expert":
            expert_id = int(self.single_expert_id) % 3  # 0,1,2
            print(f"🧪 Folding Ablation: SINGLE EXPERT id={expert_id}. Rollouts={self.k_rollouts_per_expert}")
            expert_candidates = []
            
            for r in range(self.k_rollouts_per_expert):
                try:
                    # Generate new structure tokens using DPLM expert
                    new_struct_tokens = self.dplm2_integration.generate_structure_tokens(
                        sequence=sequence,
                        masked_struct_tokens=masked_struct_tokens,
                        expert_id=expert_id,
                        temperature=0.9
                    )
                    
                    if new_struct_tokens:
                        # Compute folding metrics
                        rmsd, tm_score = self._compute_folding_metrics(new_struct_tokens)
                        reward = self._compute_structure_reward(rmsd, tm_score)
                        expert_candidates.append((reward, sequence, new_struct_tokens, rmsd, tm_score))
                        
                except Exception as e:
                    print(f"   Expert {expert_id} rollout {r} failed: {e}")

            if not expert_candidates:
                print("   ⚠️ No valid candidates from single expert")
                return []

            # Rank candidates by their scores
            expert_candidates.sort(reverse=True, key=lambda x: x[0])
            top_candidates = expert_candidates[:3]  # Keep full candidate info

        # (C) MULTI-EXPERT — use all 3 experts
        else:  # multi_expert
            expert_ids = [0, 1, 2]
            k_rollouts = self.k_rollouts_per_expert
            print(f"🎯 Folding Multi-expert expansion: {len(expert_ids)} experts × {k_rollouts} rollouts")
            
            all_candidates = []
            for expert_id in expert_ids:
                for r in range(k_rollouts):
                    try:
                        # Generate new structure tokens using DPLM expert
                        new_struct_tokens = self.dplm2_integration.generate_structure_tokens(
                            sequence=sequence,
                            masked_struct_tokens=masked_struct_tokens,
                            expert_id=expert_id,
                            temperature=0.9
                        )
                        
                        if new_struct_tokens:
                            # Compute folding metrics
                            rmsd, tm_score = self._compute_folding_metrics(new_struct_tokens)
                            reward = self._compute_structure_reward(rmsd, tm_score)
                            all_candidates.append((reward, sequence, new_struct_tokens, rmsd, tm_score))
                            
                    except Exception as e:
                        print(f"   Expert {expert_id} rollout {r} failed: {e}")
            
            if not all_candidates:
                print("   ⚠️ No valid candidates from multi-expert")
                return []
            
            # Select top candidates across all experts
            all_candidates.sort(reverse=True, key=lambda x: x[0])
            top_candidates = all_candidates[:self.num_children_select]  # Keep full candidate info

        # Convert to MCTSNode objects WITH structure evaluation results
        children = []
        for i, (reward, seq, structure_tokens, rmsd, tm_score) in enumerate(top_candidates):
            child = MCTSNode(sequence=seq, parent=None)
            child.depth = 0  # Will be set properly by caller
            
            # 🔥 CRITICAL FIX: Store structure evaluation results on child node
            child.structure_tokens = structure_tokens
            child.rmsd = rmsd
            child.tm_score = tm_score
            child.reward = reward  # Store the computed reward
            
            print(f"   🏆 Child {i+1}: RMSD={rmsd:.3f}Å, TM-score={tm_score:.3f}, reward={reward:.3f}")
            children.append(child)
        
        return children
    
    def _generate_inverse_folding_candidates(self, sequence: str, masked_seq: str, mask_positions: set) -> List[MCTSNode]:
        """
        Generate candidate sequences for inverse folding task using DPLM experts.
        """
        candidates: List[str] = []

        # =========================
        # ABLATION BRANCHES FOR INVERSE FOLDING
        # =========================

        # (A) RANDOM ABLATION
        if self.ablation_mode == "random_no_expert":
            print(f"🎲 Inverse Folding Ablation: RANDOM (no expert). Rollouts={self.k_rollouts_per_expert}")
            
            # Original inverse folding random logic
            for _ in range(self.k_rollouts_per_expert):
                proposal = list(sequence)
                for i in mask_positions:
                    proposal[i] = random.choice(AMINO_ACIDS)
                complete_seq = ''.join(proposal)
                candidates.append(complete_seq)
            
            # score & keep top-N (use same multi-critic reward)
            scored = [(self._compute_reward(c), c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [c for _, c in scored[:self.num_children_select]]

        # (B) SINGLE EXPERT — one expert only, spawn children directly
        elif self.ablation_mode == "single_expert":
            expert_id = int(self.single_expert_id) % 3  # 0,1,2
            print(f"🧪 Inverse Folding Ablation: SINGLE EXPERT id={expert_id}. Rollouts={self.k_rollouts_per_expert}")
            expert_candidates = []
            
            for r in range(self.k_rollouts_per_expert):
                try:
                    # Inverse folding: mask AA tokens, keep structure tokens
                    raw_seq = self.dplm2_integration.generate_inverse_folding_batch(
                        structure_data=self._baseline_structure,
                        masked_sequences=[masked_seq],
                        expert_id=expert_id,
                        temperature=1.0
                    )
                    
                    if raw_seq and len(raw_seq) > 0:
                        complete_seq = apply_patch(sequence, raw_seq[0], mask_positions)  # Take first result from batch
                        aar = compute_fast_aar(complete_seq, self.reference_sequence)
                        expert_candidates.append((aar, complete_seq))
                        
                except Exception as e:
                    print(f"   Expert {expert_id} rollout {r} failed: {e}")

            if not expert_candidates:
                print("   ⚠️ No valid candidates from single expert")
                return []

            # Rank candidates by their scores
            expert_candidates.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [c[1] for c in expert_candidates[:3]]  # Return sequences

        # (C) MULTI-EXPERT — use all 3 experts
        else:  # multi_expert
            expert_ids = [0, 1, 2]
            k_rollouts = self.k_rollouts_per_expert
            print(f"🎯 Inverse Folding Multi-expert expansion: {len(expert_ids)} experts × {k_rollouts} rollouts")
            
            all_candidates = []
            for expert_id in expert_ids:
                for r in range(k_rollouts):
                    try:
                        # Inverse folding: mask AA tokens, keep structure tokens
                        raw_seq = self.dplm2_integration.generate_inverse_folding_batch(
                            structure_data=self._baseline_structure,
                            masked_sequences=[masked_seq],
                            expert_id=expert_id,
                            temperature=1.0
                        )
                        
                        if raw_seq and len(raw_seq) > 0:
                            complete_seq = apply_patch(sequence, raw_seq[0], mask_positions)
                            aar = compute_fast_aar(complete_seq, self.reference_sequence)
                            all_candidates.append((aar, complete_seq))
                            
                    except Exception as e:
                        print(f"   Expert {expert_id} rollout {r} failed: {e}")
            
            if not all_candidates:
                print("   ⚠️ No valid candidates from multi-expert")
                return []
            
            # Select top candidates across all experts
            all_candidates.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [c[1] for c in all_candidates[:self.num_children_select]]

        # Convert to MCTSNode objects
        children = []
        for i, seq in enumerate(top_candidates):
            child = MCTSNode(sequence=seq, parent=None)
            child.depth = 0  # Will be set properly by caller
            children.append(child)
        
        return children
    
    def _expand_motif_scaffolding(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node for motif scaffolding task: mask scaffold region, generate new sequences/structures.
        Following DPLM2 paper approach: motif is fixed, scaffold is optimized.
        """
        seq = node.sequence
        
        # Get motif length from baseline structure
        motif_length = self._baseline_structure.get('motif_length', len(seq) // 2)
        
        print(f"🔍 Motif scaffolding expansion: motif_length={motif_length}, total_length={len(seq)}")
        
        # Only mask scaffold region (keep motif fixed)
        scaffold_positions = set(range(motif_length, len(seq)))
        
        # Use confidence-based masking for scaffold region only
        plddt_scores = self._baseline_structure.get('plddt_scores', [70.0] * len(seq))
        
        # Compute mask schedule for scaffold region only
        mask = compute_mask_schedule(seq, plddt_scores, node.depth, self.max_depth)
        # Filter mask to only include scaffold positions
        mask = mask.intersection(scaffold_positions)
        
        print(f"🎯 Scaffold masking: {len(mask)} positions ({len(mask)/(len(seq)-motif_length)*100:.1f}% of scaffold) at depth {node.depth}")
        
        # Generate candidates using multi-expert approach
        if self.ablation_mode == "multi_expert":
            expert_ids = [0, 1, 2]
            k_rollouts = self.k_rollouts_per_expert
            
            all_candidates = []
            for expert_id in expert_ids:
                for r in range(k_rollouts):
                    try:
                        # For motif scaffolding, we generate both sequence and structure
                        # Use the same approach as folding but only for scaffold region
                        
                        # Create masked sequence for generation
                        masked = list(seq)
                        for i in mask:
                            masked[i] = 'X'
                        masked_seq = ''.join(masked)
                        
                        # Generate new sequence for scaffold region
                        raw = self.dplm2_integration.generate_with_expert(
                            expert_id, self._baseline_structure, len(seq), masked_seq, temperature=1.0
                        )
                        
                        if raw:
                            complete_seq = apply_patch(seq, raw, mask)
                            
                            # Calculate motif scaffolding reward
                            reward = self._compute_motif_scaffolding_reward(complete_seq)
                            all_candidates.append((reward, complete_seq))
                            
                    except Exception as e:
                        print(f"Expert {expert_id} motif scaffolding rollout {r} failed: {e}")
                        continue
            
            # Select top candidates
            if all_candidates:
                scored = [(self._compute_reward(c), c) for _, c in all_candidates]
                scored.sort(reverse=True, key=lambda x: x[0])
                top_candidates = [c for _, c in scored[:self.num_children_select]]
                print(f"🏆 Selected {len(top_candidates)} motif scaffolding children from {len(all_candidates)} total rollouts")
            else:
                top_candidates = []
        else:
            # Single expert or random mode
            top_candidates = [seq]  # Fallback
        
        # Convert to MCTSNode objects
        children = []
        for i, seq in enumerate(top_candidates):
            child = MCTSNode(sequence=seq, parent=None)
            child.depth = 0  # Will be set properly by caller
            children.append(child)
        
        return children
    
    def _compute_motif_scaffolding_reward(self, sequence: str) -> float:
        """
        Compute reward for motif scaffolding task.
        Based on DPLM2 paper: motif-RMSD < 1Å and scTM > 0.8
        """
        try:
            # For now, use a placeholder reward
            # In full implementation, this would:
            # 1. Generate structure from sequence
            # 2. Calculate motif-RMSD for motif region
            # 3. Calculate scTM for overall structure
            # 4. Combine into reward score
            
            motif_length = self._baseline_structure.get('motif_length', len(sequence) // 2)
            
            # Placeholder: penalize changes in motif region
            motif_seq = sequence[:motif_length]
            original_motif = self._baseline_structure['sequence'][:motif_length]
            motif_conservation = sum(1 for a, b in zip(motif_seq, original_motif) if a == b) / len(motif_seq)
            
            # Reward should be high when motif is conserved and scaffold is reasonable
            reward = motif_conservation * 0.8 + 0.2  # Base reward for scaffold
            
            print(f"🎯 Motif scaffolding reward: {reward:.3f} (motif_conservation: {motif_conservation:.3f})")
            return reward
            
        except Exception as e:
            print(f"❌ Motif scaffolding reward calculation failed: {e}")
            return 0.0
    
    def _compute_empirical_entropy(self, candidates: List[str], masked_positions: List[int]) -> float:
        """
        Compute empirical entropy for random fill-in ablation studies.
        
        Args:
            candidates: List of candidate sequences
            masked_positions: Positions that were masked during generation
            
        Returns:
            Average empirical entropy across masked positions
        """
        if not candidates or not masked_positions:
            return 0.0
        
        total_entropy = 0.0
        
        for pos in masked_positions:
            # Count amino acid frequencies at this position
            aa_counts = {}
            for seq in candidates:
                if pos < len(seq):
                    aa = seq[pos]
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Compute empirical entropy: H = -sum(p * log(p))
            total_candidates = len(candidates)
            entropy = 0.0
            for count in aa_counts.values():
                p = count / total_candidates
                if p > 0:
                    entropy -= p * math.log(p)
            
            total_entropy += entropy
        
        # Average entropy across positions
        return total_entropy / len(masked_positions)

    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate (evaluate) a complete sequence.
        Pure reward evaluation, no rollouts needed.
        """
        # For folding tasks, use stored structure evaluation results if available
        if self.task_type == "folding" and hasattr(node, 'rmsd') and hasattr(node, 'tm_score'):
            print(f"   🎯 Using stored structure results: RMSD={node.rmsd:.3f}Å, TM-score={node.tm_score:.3f}")
            reward = self._compute_structure_reward(node.rmsd, node.tm_score)
        else:
            # For other tasks or nodes without stored results, compute fresh
            reward = self._compute_reward(node.sequence)
        
        # Store reward on node for tracking
        node.reward = reward
        return reward
    
    def _compute_reward(self, sequence: str, rmsd: float = None, tm_score: float = None) -> float:
        """Compute task-specific reward based on task_type."""
        if self.task_type == "folding":
            return self._compute_structure_reward(rmsd, tm_score)
        elif self.task_type == "motif_scaffolding":
            return self._compute_motif_scaffolding_reward(sequence)
        else:
            return self._compute_inverse_folding_reward(sequence)
    
    def _compute_folding_reward(self, sequence: str) -> float:
        """
{{ ... }}
        Compute folding reward based on structure quality metrics.
        
        Args:
            sequence: Generated amino acid sequence
            
        Returns:
            Reward score (0.0 to 1.0)
        """
        try:
            # For folding, we evaluate structure quality
            # This is a simplified reward - in practice you'd use structure prediction
            
            # Basic sequence quality checks
            valid_aa_ratio = sum(1 for aa in sequence if aa in "ACDEFGHIKLMNPQRSTVWY") / len(sequence)
            
            # Simple biophysical penalties
            charge_penalty = self._compute_charge_penalty(sequence)
            hydro_penalty = self._compute_hydrophobicity_penalty(sequence)
            
            # Combine metrics (simplified)
            biophysical_score = 1.0 - (charge_penalty + hydro_penalty) / 2.0
            
            # Weight components
            final_reward = (
                0.7 * valid_aa_ratio +
                0.3 * biophysical_score
            )
            
            return max(0.0, min(1.0, final_reward))
            
        except Exception as e:
            print(f"⚠️ Folding reward calculation failed: {e}")
            return 0.5  # Neutral reward on failure
    
    def _compute_charge_penalty(self, sequence: str) -> float:
        """Compute charge penalty based on charged residue distribution."""
        if not sequence:
            return 0.0
        
        # Count charged residues
        positive = sum(1 for aa in sequence if aa in "KRH")
        negative = sum(1 for aa in sequence if aa in "DE")
        total = len(sequence)
        
        # Penalty for extreme charge imbalance
        charge_ratio = abs(positive - negative) / total
        return min(1.0, charge_ratio * 2.0)  # Scale penalty
    
    def _compute_hydrophobicity_penalty(self, sequence: str) -> float:
        """Compute hydrophobicity penalty based on hydrophobic residue distribution."""
        if not sequence:
            return 0.0
        
        # Count hydrophobic residues
        hydrophobic = sum(1 for aa in sequence if aa in "AILMFPWV")
        total = len(sequence)
        
        # Penalty for extreme hydrophobicity (too high or too low)
        hydro_ratio = hydrophobic / total
        optimal_ratio = 0.4  # Typical hydrophobic content
        penalty = abs(hydro_ratio - optimal_ratio) / optimal_ratio
        return min(1.0, penalty)
    
    def _compute_inverse_folding_reward(self, sequence: str) -> float:
        """Compute comprehensive reward for inverse folding: AAR + scTM + biophysical scores."""
        # Check cache first
        seq_hash = compute_sequence_hash(sequence)
        if seq_hash in self.sequence_cache.rewards:
            return self.sequence_cache.rewards[seq_hash]
        
        # 1. Compute AAR (primary metric)
        aar = compute_fast_aar(sequence, self.reference_sequence)
        
        # DEBUG: Print actual sequences to understand low AAR
        if aar < 0.3:  # Only debug when AAR is suspiciously low
            print(f"   🔍 DEBUG LOW AAR ({aar:.3f}):")
            print(f"   Generated: {sequence[:50]}...")
            print(f"   Reference: {self.reference_sequence[:50]}...")
            print(f"   Lengths: gen={len(sequence)}, ref={len(self.reference_sequence)}")
        
        # 2. Calculate scTM score using ESMFold and TMalign
        sctm_score = 0.0
        try:
            from utils.sctm_calculation import calculate_sctm_score
            
            print(f"🧬 Calculating scTM score using ESMFold...")
            
            # Load reference coordinates from baseline structure or .pkl file
            if hasattr(self, '_baseline_structure') and self._baseline_structure:
                baseline = self._baseline_structure
                reference_coords = None
                
                # First try to get coordinates directly from baseline structure
                if 'backbone_coords' in baseline and baseline['backbone_coords'] is not None:
                    coords = baseline['backbone_coords']
                    if len(coords.shape) == 3 and coords.shape[1] == 3:
                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                    else:
                        reference_coords = coords
                    print(f"🧬 Using backbone_coords from baseline: {reference_coords.shape}")
                
                elif 'coordinates' in baseline and baseline['coordinates'] is not None:
                    reference_coords = baseline['coordinates']
                    print(f"🧬 Using coordinates from baseline: {reference_coords.shape}")
                
                elif 'atom_positions' in baseline and baseline['atom_positions'] is not None:
                    coords = baseline['atom_positions']
                    if len(coords.shape) == 3 and coords.shape[1] >= 2:
                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                    else:
                        reference_coords = coords
                    print(f"🧬 Using atom_positions from baseline: {reference_coords.shape}")
                
                # If no coordinates in baseline, try to load from .pkl file using structure_idx
                if reference_coords is None and 'structure_idx' in baseline:
                    structure_idx = baseline['structure_idx']
                    print(f"🧬 No coordinates in baseline, loading from .pkl file (idx: {structure_idx})")
                    
                    # Try CAMEO data loader first (since we're using CAMEO data)
                    try:
                        from utils.cameo_data_loader import CAMEODataLoader
                        cameo_loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
                        cameo_structure = cameo_loader.get_structure_by_index(structure_idx)
                        if cameo_structure:
                            # Extract coordinates from CAMEO structure
                            if 'backbone_coords' in cameo_structure and cameo_structure['backbone_coords'] is not None:
                                coords = cameo_structure['backbone_coords']
                                if len(coords.shape) == 3 and coords.shape[1] == 3:
                                    reference_coords = coords[:, 1, :]  # CA atoms at index 1
                                else:
                                    reference_coords = coords
                                print(f"🧬 Using backbone_coords from CAMEO .pkl: {reference_coords.shape}")
                            elif 'coordinates' in cameo_structure and cameo_structure['coordinates'] is not None:
                                reference_coords = cameo_structure['coordinates']
                                print(f"🧬 Using coordinates from CAMEO .pkl: {reference_coords.shape}")
                            elif 'atom_positions' in cameo_structure and cameo_structure['atom_positions'] is not None:
                                coords = cameo_structure['atom_positions']
                                if len(coords.shape) == 3 and coords.shape[1] >= 2:
                                    reference_coords = coords[:, 1, :]  # CA atoms at index 1
                                else:
                                    reference_coords = coords
                                print(f"🧬 Using atom_positions from CAMEO .pkl: {reference_coords.shape}")
                    except Exception as cameo_e:
                        print(f"⚠️ CAMEO loader failed: {cameo_e}")
                    
                    # If CAMEO failed, try PDB data loader as fallback
                    if reference_coords is None:
                        try:
                            from utils.pdb_data_loader import PDBDataLoader
                            pdb_loader = PDBDataLoader(data_path="/home/caom/AID3/dplm/data-bin/PDB_date")
                            pdb_structure = pdb_loader.get_structure_by_index(structure_idx)
                            if pdb_structure:
                                # Extract coordinates from PDB structure
                                if 'backbone_coords' in pdb_structure and pdb_structure['backbone_coords'] is not None:
                                    coords = pdb_structure['backbone_coords']
                                    if len(coords.shape) == 3 and coords.shape[1] == 3:
                                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                                    else:
                                        reference_coords = coords
                                    print(f"🧬 Using backbone_coords from PDB .pkl: {reference_coords.shape}")
                                elif 'coordinates' in pdb_structure and pdb_structure['coordinates'] is not None:
                                    reference_coords = pdb_structure['coordinates']
                                    print(f"🧬 Using coordinates from PDB .pkl: {reference_coords.shape}")
                                elif 'atom_positions' in pdb_structure and pdb_structure['atom_positions'] is not None:
                                    coords = pdb_structure['atom_positions']
                                    if len(coords.shape) == 3 and coords.shape[1] >= 2:
                                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                                    else:
                                        reference_coords = coords
                                    print(f"Using atom_positions from PDB .pkl: {reference_coords.shape}")
                        except Exception as pdb_e:
                            print(f"PDB loader failed: {pdb_e}")
                
                if reference_coords is not None and hasattr(reference_coords, 'shape'):
                    print(f"Using reference coordinates for scTM calculation: {reference_coords.shape}")
                    sctm_score = calculate_sctm_score(sequence, reference_coords)
                    if sctm_score is not None and sctm_score > 0:
                        print(f"scTM score: {sctm_score:.3f}")
                    else:
                        print(f"scTM calculation returned invalid score: {sctm_score}")
                        sctm_score = 0.0
                else:
                    print(f"No valid reference coordinates found")
                    sctm_score = 0.0
            else:
                print(f"No baseline structure available for scTM calculation")
                sctm_score = 0.0
            
        except Exception as e:
            print(f"scTM calculation failed: {e}")
            sctm_score = 0.0
        
        # 3. Compute biophysical scores (basic penalties)
        biophys_score = 0.0
        try:
            # Basic biophysical penalties
            # Penalize extreme compositions
            aa_counts = {aa: sequence.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
            total_len = len(sequence)
            
            # Penalty for too many charged residues
            charged = (aa_counts.get('D', 0) + aa_counts.get('E', 0) + 
                      aa_counts.get('K', 0) + aa_counts.get('R', 0)) / total_len
            charge_penalty = max(0, charged - 0.3) * 0.5  # Penalty if >30% charged
            
            # Penalty for too many hydrophobic residues
            hydrophobic = (aa_counts.get('A', 0) + aa_counts.get('I', 0) + 
                          aa_counts.get('L', 0) + aa_counts.get('V', 0) + 
                          aa_counts.get('F', 0)) / total_len
            hydro_penalty = max(0, hydrophobic - 0.4) * 0.3  # Penalty if >40% hydrophobic
            
            biophys_score = max(0.0, 1.0 - charge_penalty - hydro_penalty)
            print(f"   Biophys: {biophys_score:.3f} (charge_penalty={charge_penalty:.3f}, hydro_penalty={hydro_penalty:.3f})")
            
        except Exception as e:
            print(f"⚠️ Biophysical calculation failed: {e}")
            biophys_score = 1.0  # Neutral if calculation fails
        
        # 3. Calculate biophysical quality (from backup)
        biophysical_quality = 0.8  # Default fallback
        try:
            from mcts_diffusion_finetune.utils.reward_computation import LengthAwareRewardComputation
            biophysical_calc = LengthAwareRewardComputation(use_real_structure_eval=False)
            biophysical_quality = biophysical_calc.compute_biophysical_properties(sequence)
        except Exception as e:
            print(f"⚠️ Biophysical calculation failed: {e}, using fallback")
            biophysical_quality = 0.8
        
        # 4. Combine rewards based on task type
        if self.task_type == "inverse_folding":
            # Inverse folding: AAR + scTM + biophysical
            aar_weight = 0.60
            sctm_weight = 0.35
            biophysical_weight = 0.05
            
            reward = (aar * aar_weight) + (sctm_score * sctm_weight) + (biophysical_quality * biophysical_weight)
            
            print(f"🎯 INVERSE FOLDING REWARD:")
            print(f"   AAR: {aar:.3f} (weight: {aar_weight:.1%})")
            print(f"   scTM: {sctm_score:.3f} (weight: {sctm_weight:.1%})")
            print(f"   Biophysical: {biophysical_quality:.3f} (weight: {biophysical_weight:.1%})")
            print(f"   Final reward: {reward:.3f}")
            
        elif self.task_type == "folding":
            # Folding task: This method should NOT be called for folding
            # Folding rewards are computed in _compute_structure_reward() using RMSD/TM-score
            print("⚠️ _compute_reward() called for folding task - this should use _compute_structure_reward()")
            print("   Returning minimal reward as fallback")
            reward = biophysical_quality * 0.05  # Only biophysical component
            
            print(f"🎯 FOLDING REWARD (FALLBACK):")
            print(f"   Biophysical: {biophysical_quality:.3f} (weight: 100%)")
            print(f"   Final reward: {reward:.3f}")
            print("   Note: Proper folding rewards computed via _compute_structure_reward()")
            
        elif self.task_type == "motif_scaffolding":
            # For motif scaffolding, use similar metrics but focus on scaffold quality
            return self._compute_motif_scaffolding_reward(sequence)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        # Cache result
        self.sequence_cache.rewards[seq_hash] = reward
        return reward
    
    def _compute_structure_reward(self, rmsd: float, tm_score: float) -> float:
        """
        Compute reward for folding tasks based on RMSD and TM-score.
        This is the correct reward function for folding (not _compute_reward).
        """
        try:
            # Convert RMSD to reward (lower RMSD = higher reward)
            # Cap RMSD at 50Å for numerical stability
            rmsd_capped = min(rmsd, 50.0)
            rmsd_reward = max(0.0, 1.0 - (rmsd_capped / 50.0))  # 0.0 to 1.0
            
            # TM-score is already 0.0 to 1.0 (higher = better)
            tm_reward = max(0.0, min(1.0, tm_score))
            
            # Compute biophysical score (placeholder - could be enhanced)
            biophysical_reward = 0.9  # Default good biophysical score
            
            # Weighted combination for folding
            rmsd_weight = 0.40
            tm_weight = 0.55
            biophysical_weight = 0.05
            
            reward = (rmsd_reward * rmsd_weight) + (tm_reward * tm_weight) + (biophysical_reward * biophysical_weight)
            
            print(f"🎯 STRUCTURE REWARD:")
            print(f"   RMSD: {rmsd:.3f}Å → reward: {rmsd_reward:.3f} (weight: {rmsd_weight:.1%})")
            print(f"   TM-score: {tm_score:.3f} → reward: {tm_reward:.3f} (weight: {tm_weight:.1%})")
            print(f"   Biophysical: {biophysical_reward:.3f} (weight: {biophysical_weight:.1%})")
            print(f"   Final reward: {reward:.3f}")
            
            return reward
            
        except Exception as e:
            print(f"⚠️ Structure reward calculation failed: {e}")
            return 0.0
    
    def _predict_structure_coords(self, sequence: str) -> Optional[np.ndarray]:
        """Predict structure coordinates for a sequence using ESMFold"""
        try:
            import torch
            import numpy as np
            from transformers import EsmForProteinFolding, AutoTokenizer
            
            # Load ESMFold model (cached)
            if not hasattr(self, '_esmfold_model'):
                print("🔄 Loading ESMFold for folding metrics...")
                self._esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
                self._esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                self._esmfold_model.eval()
                if torch.cuda.is_available():
                    self._esmfold_model = self._esmfold_model.cuda()
            
            # Tokenize and predict
            tokenized = self._esmfold_tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            with torch.no_grad():
                output = self._esmfold_model(tokenized["input_ids"])
            
            # Extract CA coordinates
            positions = output["positions"]  # Shape: [1, L, 37, 3]
            ca_positions = positions[0, :, 1, :].cpu().numpy()  # CA is atom index 1
            
            return ca_positions
            
        except Exception as e:
            print(f"❌ Structure prediction failed: {e}")
            return None
    
    def _calculate_rmsd_and_tmscore(self, pred_coords: np.ndarray, ref_coords: np.ndarray) -> Tuple[float, float]:
        """Calculate RMSD and TM-score between predicted and reference coordinates"""
        try:
            import numpy as np
            
            print(f"🧮 Computing RMSD/TM-score: pred={pred_coords.shape}, ref={ref_coords.shape}")
            
            # Ensure same length
            min_len = min(len(pred_coords), len(ref_coords))
            pred_coords = pred_coords[:min_len]
            ref_coords = ref_coords[:min_len]
            
            if len(pred_coords) == 0:
                print("⚠️ Empty coordinates for RMSD/TM-score calculation")
                return float('inf'), 0.0
            
            # Align coordinates (remove center of mass)
            pred_centered = pred_coords - np.mean(pred_coords, axis=0)
            ref_centered = ref_coords - np.mean(ref_coords, axis=0)
            
            # Calculate RMSD after alignment
            rmsd = np.sqrt(np.mean(np.sum((pred_centered - ref_centered) ** 2, axis=1)))
            
            # Calculate TM-score (normalized by target length)
            L_target = len(ref_coords)
            d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
            
            distances = np.sqrt(np.sum((pred_centered - ref_centered) ** 2, axis=1))
            tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
            
            print(f"📊 Calculated: RMSD={rmsd:.3f}Å, TM-score={tm_score:.3f}")
            return rmsd, tm_score
            
        except Exception as e:
            print(f"⚠️ RMSD/TM-score calculation failed: {e}")
            return float('inf'), 0.0
    
    def _compute_folding_metrics_simple(self, structure_tokens: str) -> Tuple[float, float]:
        """Simplified folding metrics - avoid coordinate conversion."""
        try:
            # Simple heuristic based on structure token diversity
            if structure_tokens:
                tokens = structure_tokens.split()
                unique_tokens = len(set(tokens))
                total_tokens = len(tokens)
                
                # Higher diversity = better structure
                diversity = unique_tokens / max(1, total_tokens)
                
                # Convert to RMSD-like and TM-score-like metrics
                pseudo_rmsd = 10.0 * (1.0 - diversity)  # Lower is better
                pseudo_tm = diversity  # Higher is better
                
                return pseudo_rmsd, pseudo_tm
            
            return 10.0, 0.0  # Poor scores for empty tokens
            
        except Exception as e:
            print(f"   ⚠️ Simple folding metrics failed: {e}")
            return 10.0, 0.0

    def _coords_to_structure_tokens(self, coords: np.ndarray) -> str:
        """
        Convert 3D coordinates to DPLM structure tokens.
        
        For folding tasks: Convert ESMFold coordinates to REAL structure tokens
        For inverse folding: Try coordinate tokenization, fallback to mask tokens
        """
        try:
            print(f"🔄 Converting coordinates to structure tokens: {coords.shape}")
            
            from byprot.models.utils import get_struct_tokenizer
            import torch
            
            # Load structure tokenizer
            try:
                struct_tokenizer = get_struct_tokenizer()
                device = next(self.dplm2_integration.model.parameters()).device
                struct_tokenizer = struct_tokenizer.to(device)
                print(f"✅ Loaded struct tokenizer on {device}")
            except Exception as e:
                print(f"❌ Failed to load struct tokenizer: {e}")
                # Fallback to mask tokens if tokenizer fails
                seq_len = coords.shape[0]
                struct_tokens = '<cls_struct> ' + '<mask_struct> ' * seq_len + '<eos_struct>'
                print(f"🔧 Fallback to mask tokens: {seq_len} positions")
                return struct_tokens
            
            seq_len = coords.shape[0]
            
            # Create full atom positions with CA coordinates
            full_coords = torch.zeros((1, seq_len, 37, 3), dtype=torch.float32, device=device)
            
            # Handle coordinate tensor conversion properly
            coords_tensor = torch.from_numpy(coords.astype(np.float32)).to(device)
            if len(coords_tensor.shape) == 2 and coords_tensor.shape[0] == seq_len:
                # Shape: [seq_len, 3] - correct format
                full_coords[0, :, 1, :] = coords_tensor  # CA at index 1
            else:
                print(f"   ⚠️ Unexpected coordinate shape: {coords_tensor.shape}, expected [{seq_len}, 3]")
                # Try to reshape or handle the mismatch
                if coords_tensor.numel() == seq_len * 3:
                    coords_tensor = coords_tensor.reshape(seq_len, 3)
                    full_coords[0, :, 1, :] = coords_tensor
                else:
                    raise ValueError(f"Cannot reshape coordinates: {coords_tensor.shape} to [{seq_len}, 3]")
            
            # Create residue mask (all positions valid)
            res_mask = torch.ones((1, seq_len), dtype=torch.bool, device=device)
            seq_length = torch.tensor([seq_len], device=device)
            
            # Tokenize coordinates to get REAL structure tokens
            struct_ids = struct_tokenizer.tokenize(full_coords, res_mask, seq_length)
            struct_seq = struct_tokenizer.struct_ids_to_seq(struct_ids.cpu().tolist()[0])
            
            print(f"✅ Generated REAL structure tokens: {len(struct_seq)} chars")
            return struct_seq
            
        except Exception as e:
            print(f"❌ Structure token conversion failed: {e}")
            # Ultimate fallback
            try:
                seq_len = coords.shape[0] if coords is not None else 100
                struct_tokens = '<cls_struct> ' + '<mask_struct> ' * seq_len + '<eos_struct>'
                print(f"🔧 Using ultimate fallback structure mask tokens: {seq_len} positions")
                return struct_tokens
            except:
                return None
    
    def _generate_esmfold_baseline(self, sequence: str) -> np.ndarray:
        """Generate baseline structure using ESMFold."""
        try:
            print(f"🔄 Generating ESMFold baseline for sequence length {len(sequence)}")
            
            # Import ESMFold
            import torch
            from transformers import EsmForProteinFolding, AutoTokenizer
            
            # Load ESMFold model
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            
            device = next(self.dplm2_integration.model.parameters()).device
            model = model.to(device)
            model.eval()
            
            # Tokenize and predict
            with torch.no_grad():
                tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
                tokenized = {k: v.to(device) for k, v in tokenized.items()}
                
                output = model(tokenized['input_ids'])
                positions = output.positions  # Shape: [batch, length, atoms, 3]
                
                print(f"   🔍 ESMFold output shape: {positions.shape}")
                
                # Handle different ESMFold output shapes
                if len(positions.shape) == 5:
                    # Shape: [batch, 1, length, atoms, 3] 
                    ca_coords = positions[0, 0, :, 1, :].cpu().numpy()  # [length, 3]
                elif len(positions.shape) == 4:
                    # Shape: [batch, length, atoms, 3]
                    ca_coords = positions[0, :, 1, :].cpu().numpy()  # [length, 3]
                else:
                    raise ValueError(f"Unexpected ESMFold output shape: {positions.shape}")
                
                # Ensure we have the right sequence length
                expected_len = len(sequence)
                if ca_coords.shape[0] != expected_len:
                    print(f"   ⚠️ Length mismatch: got {ca_coords.shape[0]}, expected {expected_len}")
                    # Pad or truncate to match sequence length
                    if ca_coords.shape[0] < expected_len:
                        # Pad with last coordinate
                        padding = np.tile(ca_coords[-1:], (expected_len - ca_coords.shape[0], 1))
                        ca_coords = np.vstack([ca_coords, padding])
                    else:
                        # Truncate to sequence length
                        ca_coords = ca_coords[:expected_len]
                
            print(f"✅ ESMFold baseline generated: {ca_coords.shape}")
            return ca_coords
            
        except Exception as e:
            print(f"❌ ESMFold baseline generation failed: {e}")
            # Fallback to random coordinates
            coords = np.random.rand(len(sequence), 3) * 10
            print(f"🔧 Using random fallback coordinates: {coords.shape}")
            return coords
    
    def _structure_tokens_to_coords(self, structure_tokens: str) -> Optional[np.ndarray]:
        """
        Convert structure tokens back to coordinates using official DPLM evaluator approach.
        For folding tasks with mask tokens, use fallback coordinates.
        """
        try:
            print(f"🔍 Converting structure tokens to coordinates: {len(structure_tokens)} chars")
            print(f"   Sample: {structure_tokens[:100]}...")
            
            # Check if these are mask tokens (used in folding tasks)
            if '<mask_struct>' in structure_tokens or 'mask_struct' in structure_tokens:
                print(f"   🔧 Detected mask tokens - using fallback coordinates for folding task")
                # For folding tasks, we can't convert mask tokens to real coordinates
                # Use the baseline ESMFold coordinates as fallback
                if hasattr(self, 'baseline_coords') and self.baseline_coords is not None:
                    print(f"   ✅ Using cached baseline coordinates: {self.baseline_coords.shape}")
                    return self.baseline_coords
                else:
                    # Generate dummy coordinates based on sequence length
                    mask_count = structure_tokens.count('<mask_struct>')
                    if mask_count > 0:
                        seq_len = mask_count
                    else:
                        seq_len = 284  # Default fallback
                    
                    print(f"   🔧 Generating dummy coordinates for {seq_len} residues")
                    dummy_coords = np.random.rand(seq_len, 3) * 10  # Random coordinates
                    return dummy_coords
            
            # For real structure tokens, follow the official evaluator_dplm2.py approach
            print(f"   🔄 Processing real structure tokens (not mask tokens)")
            
            # Load struct tokenizer first (evaluator_dplm2.py approach)
            from byprot.models.utils import get_struct_tokenizer
            
            # Try to get structure tokenizer (with fallback for network issues)
            try:
                struct_tokenizer = get_struct_tokenizer()
                # Move tokenizer to the same device as the model
                device = next(self.dplm2_integration.model.parameters()).device
                struct_tokenizer = struct_tokenizer.to(device)
                print(f"   ✅ Loaded struct tokenizer successfully on {device}")
            except Exception as e1:
                print(f"   ❌ Failed to load struct tokenizer: {e1}")
                print(f"   🔧 Using coordinate fallback due to tokenizer loading failure")
                # Return dummy coordinates as fallback
                seq_len = len(structure_tokens.split(',')) if ',' in structure_tokens else 284
                dummy_coords = np.random.rand(seq_len, 3) * 10  # Random coordinates
                return dummy_coords
            
            # Check if structure_tokens is a sequence string or comma-separated IDs
            if ',' in structure_tokens and structure_tokens.replace(',', '').replace(' ', '').isdigit():
                # Already tokenized as comma-separated IDs
                print(f"   🔍 Detected comma-separated token IDs")
                token_ids = [int(x.strip()) for x in structure_tokens.split(',') if x.strip()]
            else:
                # Check if this is a very long string of digits (raw token IDs concatenated)
                if structure_tokens.isdigit() and len(structure_tokens) > 100:
                    print(f"   🔍 Detected concatenated token IDs, splitting into reasonable chunks")
                    # Split into reasonable token ID chunks (assume 3-4 digits per token)
                    chunk_size = 4
                    token_ids = []
                    for i in range(0, len(structure_tokens), chunk_size):
                        chunk = structure_tokens[i:i+chunk_size]
                        if chunk:
                            token_id = int(chunk) % 8192  # Clamp to reasonable vocab size
                            token_ids.append(token_id)
                    print(f"   🔧 Split into {len(token_ids)} token chunks")
                else:
                    # Structure sequence string - convert to IDs using official method
                    print(f"   🔍 Converting structure sequence to token IDs")
                    try:
                        # This should be a structure sequence like the evaluator expects
                        token_ids = struct_tokenizer.struct_seq_to_ids(structure_tokens)
                        print(f"   ✅ Converted structure sequence successfully")
                    except Exception as e:
                        print(f"   ❌ Failed to convert structure sequence: {e}")
                        # Fallback: create dummy tokens based on length
                        token_ids = [0] * min(len(structure_tokens), 284)
            
            print(f"   🔍 Got {len(token_ids)} structure token IDs")
            print(f"   🔍 Sample token IDs: {token_ids[:5]}")
            
            # Ensure all token IDs are valid integers
            valid_token_ids = []
            for i, tid in enumerate(token_ids):
                if isinstance(tid, (int, np.integer)):
                    # Clamp to reasonable range
                    clamped_id = int(tid) % 8192  # Typical structure vocab size
                    valid_token_ids.append(clamped_id)
                else:
                    print(f"   ⚠️ Invalid token at {i}: {tid}, using 0")
                    valid_token_ids.append(0)
            
            # Convert to tensor following evaluator_dplm2.py line 204-206
            structok = torch.LongTensor(valid_token_ids).unsqueeze(0).to(device)
            res_mask = torch.ones_like(structok, dtype=torch.float).to(device)
            
            print(f"   🔍 structok shape: {structok.shape}")
            print(f"   🔍 res_mask shape: {res_mask.shape}")
            
            # Use official detokenization method (evaluator_dplm2.py line 290-292)
            decoder_out = struct_tokenizer.detokenize(structok, res_mask)
            print(f"   ✅ Detokenization successful: {list(decoder_out.keys())}")
            
            # Extract coordinates and pLDDT from decoder output (following evaluator_dplm2.py)
            if 'atom37_positions' in decoder_out:
                # Extract CA coordinates (atom index 1 for CA) following evaluator approach
                coords = decoder_out['atom37_positions'][0, :, 1, :].cpu().numpy()  # CA atoms
                
                # Extract pLDDT scores for progressive masking
                plddt = None
                if 'plddt' in decoder_out:
                    plddt = decoder_out['plddt'][0].cpu().numpy()  # [seq_len]
                    print(f"   ✅ Extracted pLDDT scores: {plddt.shape}, mean={plddt.mean():.3f}")
                
                print(f"   ✅ Extracted coordinates: {coords.shape}")
                
                # Store pLDDT for progressive masking (if this is for folding)
                if hasattr(self, 'task_type') and self.task_type == 'folding' and plddt is not None:
                    self.last_plddt_scores = plddt
                
                return coords
            elif 'all_atom_positions' in decoder_out:
                coords = decoder_out['all_atom_positions'][0, :, 1, :].cpu().numpy()  # CA atoms
                print(f"   ✅ Extracted coordinates: {coords.shape}")
                return coords
            else:
                print(f"   ⚠️ No atom37_positions or all_atom_positions in decoder output")
                print(f"   🔍 Available keys: {list(decoder_out.keys())}")
                return None
                
        except Exception as e:
            print(f"❌ Structure token to coordinates conversion failed: {e}")
            return None
    
    def _compute_folding_metrics(self, structure_tokens: str) -> Tuple[float, float]:
        """
        Compute RMSD and TM-score for folding task against GROUND TRUTH coordinates.
        Following official evaluator_dplm2.py approach: compare against all_atom_positions_gt
        
        Args:
            structure_tokens: Generated structure tokens
            
        Returns:
            Tuple of (RMSD, TM-score) compared to ground truth
        """
        try:
            # Convert structure tokens to coordinates
            coords = self._structure_tokens_to_coords(structure_tokens)
            if coords is None:
                return float('inf'), 0.0
            
            # Use GROUND TRUTH coordinates as reference (following official evaluator)
            if hasattr(self, 'reference_coords') and self.reference_coords is not None:
                ref_coords = self.reference_coords  # This should be ground truth coordinates
                print(f"   🎯 Using ground truth reference coordinates: {ref_coords.shape}")
            else:
                print("   ⚠️ No ground truth reference coordinates available - this is incorrect for folding evaluation!")
                # This should not happen in proper folding evaluation
                return float('inf'), 0.0
            
            # Calculate RMSD (backbone RMSD to ground truth)
            min_len = min(len(coords), len(ref_coords))
            coords_aligned = coords[:min_len]
            ref_aligned = ref_coords[:min_len]
            
            rmsd = np.sqrt(np.mean(np.sum((coords_aligned - ref_aligned) ** 2, axis=1)))
            
            # Calculate TM-score (following official evaluator approach)
            L_target = len(ref_aligned)
            d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
            
            distances = np.sqrt(np.sum((coords_aligned - ref_aligned) ** 2, axis=1))
            tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
            
            print(f"📊 Folding metrics vs GT: RMSD={rmsd:.3f}Å, TM-score={tm_score:.3f}")
            return rmsd, tm_score
            
        except Exception as e:
            print(f"❌ Folding metrics calculation failed: {e}")
            return float('inf'), 0.0
    
    def _compute_structure_reward(self, rmsd: float, tm_score: float) -> float:
        """Compute structure quality reward from RMSD and TM-score."""
        # Handle None values
        if rmsd is None or tm_score is None:
            print(f"⚠️ Structure reward called with None values: rmsd={rmsd}, tm_score={tm_score}")
            return 0.0
        
        # Convert RMSD to reward (lower is better, so invert)
        rmsd_reward = max(0.0, 1.0 - min(rmsd / 10.0, 1.0))  # Normalize RMSD
        
        # Combine metrics
        rmsd_weight = 0.40
        tm_weight = 0.60
        
        reward = (rmsd_reward * rmsd_weight) + (tm_score * tm_weight)
        return reward
    
    def _compute_reward_folding(self, sequence: str, structure_tokens: str) -> float:
        """Compute reward for folding task based on structural quality."""
        try:
            rmsd, tm_score = self._compute_folding_metrics(structure_tokens)
            return self._compute_structure_reward(rmsd, tm_score)
        except Exception as e:
            print(f"⚠️ Folding reward computation failed: {e}")
            return 0.0
    
    def _compute_sequence_plddt(self, sequence: str) -> Optional[List[float]]:
        """
        Simplified pLDDT calculation - use baseline structure confidence or fallback to uniform scores.
        No unnecessary structure token generation for inverse folding.
        """
        try:
            # For inverse folding: use baseline structure confidence if available
            if self.task_type == "inverse_folding" and hasattr(self, '_baseline_structure'):
                baseline_plddt = self._baseline_structure.get('plddt_scores', [])
                if baseline_plddt is not None and len(baseline_plddt) == len(sequence):
                    # Convert numpy array to list if needed
                    if hasattr(baseline_plddt, 'tolist'):
                        baseline_plddt_list = baseline_plddt.tolist()
                    else:
                        baseline_plddt_list = list(baseline_plddt)
                    print(f"✅ Using baseline pLDDT: mean={np.mean(baseline_plddt_list):.3f}, length={len(baseline_plddt_list)}")
                    return baseline_plddt_list
            
            # For folding: generate simple confidence based on sequence properties
            if self.task_type == "folding":
                # Simple heuristic: hydrophobic regions tend to be more structured
                confidence_scores = []
                for aa in sequence:
                    if aa in 'AILVFWY':  # Hydrophobic
                        confidence_scores.append(0.7)
                    elif aa in 'DEKR':   # Charged
                        confidence_scores.append(0.5)
                    else:                 # Other
                        confidence_scores.append(0.6)
                
                print(f"✅ Simple folding confidence: mean={np.mean(confidence_scores):.3f}, length={len(confidence_scores)}")
                return confidence_scores
            
            # Fallback: uniform confidence
            uniform_confidence = [0.6] * len(sequence)
            print(f"✅ Uniform confidence fallback: mean=0.6, length={len(uniform_confidence)}")
            return uniform_confidence
            
        except Exception as e:
            print(f"⚠️ pLDDT calculation failed: {e}, using uniform fallback")
            return [0.6] * len(sequence)
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree using configurable backup rule."""
        while node is not None:
            node.visit_count += 1
            if self.backup_rule == "max":
                # Max backup: W ← max(W, v), Q = W/N
                node.value_sum = max(node.value_sum, reward * node.visit_count)
            else:
                # Sum backup (Monte Carlo average): value_sum += reward
                node.value_sum += reward
            node = node.parent
    
    def _is_terminal(self, node: MCTSNode, max_depth: int) -> bool:
        """Terminal condition based on depth only."""
        return node.depth >= max_depth
    
    def _get_best_node(self, root: MCTSNode) -> MCTSNode:
        """Get best node from tree by average value."""
        best_node = root
        stack = [root]
        
        while stack:
            node = stack.pop()
            if node.average_value > best_node.average_value:
                best_node = node
            stack.extend(node.children)
        
        return best_node
    
    def _generate_random_structure_tokens(self, sequence_length: int) -> str:
        """Generate random structure tokens for folding ablation."""
        try:
            # Import here to avoid circular imports
            import random
            
            # Get structure tokenizer
            from byprot.models.utils import get_struct_tokenizer
            struct_tokenizer = get_struct_tokenizer()
            
            # Generate random structure token IDs
            # Use a reasonable range based on DPLM structure vocabulary
            vocab_size = 8192  # Approximate structure vocab size
            random_token_ids = [random.randint(1, vocab_size-1) for _ in range(sequence_length)]
            
            # Convert to string format like other structure tokens
            random_tokens = ','.join(map(str, random_token_ids))
            
            # Add structure token markers
            structure_tokens = f"<cls_struct>{random_tokens}<eos_struct>"
            
            print(f"🎲 Generated random structure tokens: {len(structure_tokens)} chars")
            return structure_tokens
            
        except Exception as e:
            print(f"❌ Random structure token generation failed: {e}")
            # Fallback: generate mask tokens
            mask_tokens = "<mask_struct>," * sequence_length
            return f"<cls_struct>{mask_tokens.rstrip(',')}<eos_struct>"


# Backward compatibility alias
SequenceLevelMCTS = GeneralMCTS
