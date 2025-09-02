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

from .mcts_utils import (
    apply_patch, compute_mask_schedule, compute_sequence_hash, 
    compute_hamming_distance, compute_fast_aar, ph_uct_score, SequenceCache
)
from .dplm2_integration_fixed import DPLM2Integration

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
    
    @property
    def ucb_score(self) -> float:
        """PH-UCT score for selection."""
        if self.visit_count == 0:
            return float('inf')
        parent_visits = max(1, self.parent.visit_count) if self.parent else 1
        return ph_uct_score(
            self.average_value,
            self.visit_count,
            parent_visits,
            c=1.4,
            w_ent=0.1,
            w_div=0.05,
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
    
    def __init__(self, dplm2_integration: DPLM2Integration, initial_sequence: str = None,
                 reference_sequence: str = None, baseline_structure: Dict = None, 
                 max_depth: int = 4, num_candidates_per_expansion: int = 6,
                 ablation_mode: str = "multi_expert",   # {"multi_expert","random_no_expert","single_expert"}
                 single_expert_id: int = 0,             # used when ablation_mode == "single_expert"
                 k_rollouts_per_expert: int = 3,        # per-expert rollouts
                 num_children_select: int = 2):         # children kept after scoring
        """
        Initialize MCTS with DPLM-2 integration.
        
        Args:
            dplm2_integration: DPLM-2 model integration
            initial_sequence: Starting complete sequence (no X tokens)
            reference_sequence: Target sequence for AAR calculation
            baseline_structure: Structure data with coordinates and metadata
            max_depth: Maximum tree depth
            num_candidates_per_expansion: Number of candidates to generate per expansion
            ablation_mode: Ablation study mode
            single_expert_id: Expert ID for single expert mode
            k_rollouts_per_expert: Rollouts per expert
            num_children_select: Number of children to select
        """
        self.dplm2_integration = dplm2_integration
        self._baseline_structure = baseline_structure or {}
        self.reference_sequence = reference_sequence
        self.max_depth = max_depth
        self.num_candidates_per_expansion = num_candidates_per_expansion

        # Store ablation knobs
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        self.k_rollouts_per_expert = k_rollouts_per_expert
        self.num_children_select = num_children_select

        # Sequence cache for reward computation
        self.sequence_cache = SequenceCache()
        
        # Track seen sequences to avoid duplicates
        self.seen_sequences = set()
        
        # Store initial sequence
        self.initial_sequence = initial_sequence
        
        # Fix scTM inputs by setting baseline sequence
        if self._baseline_structure and initial_sequence:
            self._baseline_structure['sequence'] = initial_sequence
        
        print(f"🌳 GeneralMCTS initialized:")
        print(f"   Max depth: {max_depth}")
        print(f"   Candidates per expansion: {num_candidates_per_expansion}")
        print(f"   Ablation mode: {ablation_mode}")
        if ablation_mode == "single_expert":
            print(f"   Single expert ID: {single_expert_id}")
        print(f"   Initial sequence length: {len(initial_sequence) if initial_sequence else 'None'}")
        print(f"   Reference sequence length: {len(reference_sequence) if reference_sequence else 'None'}")
        
    def search(self, num_iterations: int = 100, max_depth: int = None,
               exploration_constant: float = 1.414, structure: Dict = None) -> 'MCTSNode':
        """
        Perform MCTS search starting from complete initial sequence.
        
        Args:
            num_iterations: Number of MCTS iterations
            max_depth: Maximum tree depth (uses instance default if None)
            exploration_constant: UCB exploration constant
            structure: Structure data for masking and rewards
            
        Returns:
            Root MCTSNode with search tree
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        # Update baseline structure if new structure provided, but preserve existing one
        if structure:
            self._baseline_structure.update(structure)
        elif not hasattr(self, '_baseline_structure') or not self._baseline_structure:
            self._baseline_structure = {}
        
        # Root always stores complete initial sequence
        root = MCTSNode(sequence=self.initial_sequence, depth=0)
        
        print(f"🎯 MCTS search starting from complete sequence (length: {len(self.initial_sequence)})")
        
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
        
        # Return root node with complete search tree
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
                # Standard UCB1 score
                exploitation = child.average_value
                exploration = exploration_constant * math.sqrt(math.log(node.visit_count) / child.visit_count)
                
                # PH-UCT bonuses (cached during expansion)
                entropy_bonus = getattr(child, 'entropy_proposals', 0.0) * 0.1
                novelty_bonus = getattr(child, 'novelty_vs_parent', 0.0) * 0.1
                
                # Combined PH-UCT score
                score = exploitation + exploration + entropy_bonus + novelty_bonus
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode) -> List[MCTSNode]:
        """
        Expand node by generating children using DPLM-2 experts.
        Masking happens only internally, children store complete sequences.
        """
        seq = node.sequence
        
        # Compute dynamic pLDDT for this sequence
        current_plddt = self._compute_sequence_plddt(seq)
        if current_plddt and len(current_plddt) == len(seq):
            mask = compute_mask_schedule(seq, current_plddt, node.depth, self.max_depth)
        else:
            # Fallback to baseline pLDDT
            baseline_plddt = self._baseline_structure.get('plddt_scores', [])
            if baseline_plddt and len(baseline_plddt) == len(seq):
                mask = compute_mask_schedule(seq, baseline_plddt, node.depth, self.max_depth)
            else:
                # Final fallback: small random mask
                num_mask = max(2, int(len(seq) * 0.05))
                mask = set(random.sample(range(len(seq)), min(num_mask, len(seq))))
        
        # Create masked sequence for expert generation (temporary variable)
        masked = list(seq)
        for i in mask:
            masked[i] = 'X'
        masked_seq = ''.join(masked)
        
        candidates: List[str] = []

        # =========================
        # ABLATION BRANCHES
        # =========================

        # (A) RANDOM FILL — no experts at all
        if self.ablation_mode == "random_no_expert":
            print("🎲 Ablation: RANDOM (no experts).")
            # generate num_candidates_per_expansion random fills
            for _ in range(self.num_candidates_per_expansion):
                proposal = list(seq)
                for i in mask:
                    proposal[i] = random.choice(AMINO_ACIDS)
                complete_seq = ''.join(proposal)
                candidates.append(complete_seq)

            # score & keep top-N (use same multi-critic reward)
            scored = [(self._compute_reward(c), c) for c in candidates]
            scored.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [c for _, c in scored[:self.num_children_select]]

        # (B) SINGLE EXPERT — one expert only, spawn 3 children directly
        elif self.ablation_mode == "single_expert":
            expert_id = int(self.single_expert_id) % 3  # 0,1,2
            print(f"🧪 Ablation: SINGLE EXPERT id={expert_id}. Rollouts={self.k_rollouts_per_expert}")
            expert_candidates = []
            for r in range(self.k_rollouts_per_expert):
                try:
                    base_struct = self._baseline_structure.copy()
                    if 'struct_ids' not in base_struct and 'struct_seq' not in base_struct:
                        print("   ⚠️ No struct tokens; using seq-only fill as fallback")
                        with self.dplm2_integration._with_model(
                            self.dplm2_integration.expert_models[expert_id]
                        ):
                            raw = self.dplm2_integration.fill_masked_positions_seq_only(masked_seq, temperature=0.9)
                    else:
                        raw = self.dplm2_integration.generate_with_expert(
                            expert_id=expert_id,
                            structure=base_struct,
                            target_length=len(seq),
                            masked_sequence=masked_seq,
                            temperature=1.0
                        )
                    complete_seq = apply_patch(seq, raw, mask)
                    aar = compute_fast_aar(complete_seq, self.reference_sequence)
                    expert_candidates.append((aar, complete_seq))
                except Exception as e:
                    print(f"   Expert {expert_id} rollout {r} failed: {e}")

            if not expert_candidates:
                print("   ⚠️ No valid candidates from single expert")
                return []

            # Rank by proxy AAR then full reward, and KEEP 3 children directly
            expert_candidates.sort(reverse=True, key=lambda x: x[0])
            # refine top pool with full reward
            refined = [(self._compute_reward(c), c) for _, c in expert_candidates[:max(3, self.num_children_select)]]
            refined.sort(key=lambda x: x[0], reverse=True)
            top_candidates = [c for _, c in refined[:3]]  # force 3 as per your request
            print(f"🏆 SINGLE EXPERT: spawned {len(top_candidates)} children")

        # (C) DEFAULT MULTI-EXPERT (original behavior)
        else:
            expert_ids = [0, 1, 2]
            k_rollouts = self.k_rollouts_per_expert
            print(f"🎯 Multi-expert expansion: {len(expert_ids)} experts × {k_rollouts} rollouts")
            if len(seq) > 0:
                print(f"   Mask size: {len(mask)} positions ({len(mask)/len(seq)*100:.1f}%)")

            per_expert_bests = []
            for expert_id in expert_ids:
                expert_pool = []
                for r in range(k_rollouts):
                    try:
                        base_struct = self._baseline_structure.copy()
                        if 'struct_ids' not in base_struct and 'struct_seq' not in base_struct:
                            print(f"   ⚠️ No struct tokens; seq-only fallback for expert {expert_id}")
                            with self.dplm2_integration._with_model(
                                self.dplm2_integration.expert_models[expert_id]
                            ):
                                raw = self.dplm2_integration.fill_masked_positions_seq_only(masked_seq, temperature=0.9)
                        else:
                            raw = self.dplm2_integration.generate_with_expert(
                                expert_id=expert_id,
                                structure=base_struct,
                                target_length=len(seq),
                                masked_sequence=masked_seq,
                                temperature=1.0
                            )
                        complete_seq = apply_patch(seq, raw, mask)
                        aar = compute_fast_aar(complete_seq, self.reference_sequence)
                        expert_pool.append((aar, complete_seq))
                    except Exception as e:
                        print(f"Expert {expert_id} rollout {r} failed: {e}")
                if expert_pool:
                    expert_pool.sort(reverse=True, key=lambda x: x[0])
                    per_expert_bests.append(expert_pool[0][1])

            if not per_expert_bests:
                print("⚠️ No valid candidates generated")
                return []

            scored = [(self._compute_reward(c), c) for c in per_expert_bests]
            scored.sort(reverse=True, key=lambda x: x[0])
            top_candidates = [c for _, c in scored[:self.num_children_select]]
            print(f"🏆 Selected {len(top_candidates)} children from {len(per_expert_bests)} expert candidates")
        
        # Create child nodes (all store complete sequences)
        children = []
        for cand in top_candidates:
            child = MCTSNode(sequence=cand, parent=node, depth=node.depth + 1)
            
            # Cache PH-UCT bonuses
            if self.ablation_mode == "single_expert":
                child.entropy_proposals = self.k_rollouts_per_expert / 10.0
            else:
                child.entropy_proposals = len(candidates) / 10.0
            child.novelty_vs_parent = sum(1 for a, b in zip(cand, seq) if a != b) / len(seq)
            
            children.append(child)
        
        return children
    
    def _simulate(self, node: MCTSNode) -> float:
        """
        Simulate (evaluate) a complete sequence.
        Pure reward evaluation, no rollouts needed.
        """
        reward = self._compute_reward(node.sequence)
        
        # Store reward on node for tracking
        node.reward = reward
        return reward
    
    def _compute_reward(self, sequence: str) -> float:
        """Compute comprehensive reward: AAR + scTM + biophysical scores."""
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
        
        # 2. Calculate scTM score using ESMFold and TMalign (from backup)
        sctm_score = 0.0
        try:
            from utils.sctm_calculation import calculate_sctm_with_cameo_data
            
            print(f"🧬 Calculating scTM score using ESMFold...")
            
            # Create CAMEO-format structure data from _baseline_structure (from backup)
            structure_data = None
            if hasattr(self, '_baseline_structure') and self._baseline_structure:
                baseline = self._baseline_structure
                
                # Create CAMEO-format dict from _baseline_structure
                structure_data = {
                    'bb_positions': baseline.get('backbone_coords', baseline.get('coordinates')),
                    'sequence': baseline.get('sequence', ''),
                    'bb_mask': [True] * baseline.get('length', 0) if baseline.get('length') else None
                }
                
                # Convert sequence to aatype if needed
                if structure_data['sequence']:
                    AA_TO_IDX = {
                        'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
                        'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
                    }
                    import numpy as np
                    aatype = np.array([AA_TO_IDX.get(aa, 20) for aa in structure_data['sequence']])
                    structure_data['aatype'] = aatype
                
                if structure_data is not None:
                    sctm_score = calculate_sctm_with_cameo_data(sequence, structure_data)
                    if sctm_score is not None and sctm_score > 0:
                        print(f"✅ scTM score: {sctm_score:.3f}")
                    else:
                        print(f"⚠️ scTM calculation failed, using fallback")
                        sctm_score = 0.0
                else:
                    print(f"⚠️ No reference structure data available for scTM calculation")
                    sctm_score = 0.0
            
        except Exception as e:
            print(f"⚠️ scTM calculation failed: {e}")
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
        
        # 4. Combine rewards (85% AAR + 10% scTM + 5% Biophysical) - from backup
        aar_weight = 0.85
        sctm_weight = 0.10
        biophysical_weight = 0.05
        
        reward = (aar * aar_weight) + (sctm_score * sctm_weight) + (biophysical_quality * biophysical_weight)
        
        print(f"🎯 MULTI-CRITIC REWARD:")
        print(f"   AAR: {aar:.3f} (weight: {aar_weight:.1%})")
        print(f"   scTM: {sctm_score:.3f} (weight: {sctm_weight:.1%})")
        print(f"   Biophysical: {biophysical_quality:.3f} (weight: {biophysical_weight:.1%})")
        print(f"   Final reward: {reward:.3f}")
        
        # Cache result
        self.sequence_cache.rewards[seq_hash] = reward
        return reward
    
    def _compute_sequence_plddt(self, sequence: str) -> Optional[List[float]]:
        """
        Compute true dynamic pLDDT scores using ESMFold structure prediction.
        Uses real_plddt_computation.py for actual structure-based confidence.
        """
        try:
            # Import and use the real ESMFold pLDDT calculation
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
            
            from real_plddt_computation import compute_esmfold_plddt
            
            print(f"🎯 Computing real ESMFold pLDDT for sequence length {len(sequence)}")
            
            # Use actual ESMFold structure prediction to get pLDDT
            per_residue_plddt, mean_plddt = compute_esmfold_plddt(sequence)
            
            print(f"✅ ESMFold structure-based pLDDT: mean={mean_plddt:.3f}, length={len(per_residue_plddt)}")
            
            return per_residue_plddt.tolist()
            
        except ImportError as e:
            print(f"⚠️ Could not import real_plddt_computation: {e}")
            return None
        except Exception as e:
            print(f"❌ ESMFold structure prediction failed: {e}")
            return None
    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visit_count += 1
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


# Backward compatibility alias
SequenceLevelMCTS = GeneralMCTS
