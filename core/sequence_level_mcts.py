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
    
    def __init__(self, dplm2_integration, reference_sequence: str, baseline_structure: Dict, 
                 max_depth: int = 3, exploration_constant: float = 1.414, 
                 num_children_select: int = 2, k_rollouts_per_expert: int = 2,
                 ablation_mode: str = "multi_expert", single_expert_id: int = 0,
                 backup_rule: str = "max"):
        """
        Initialize MCTS for sequence-level optimization.
        
        Args:
            dplm2_integration: DPLM-2 model integration
            reference_sequence: Target sequence for AAR calculation
            baseline_structure: Structure data with coordinates/pLDDT
            max_depth: Maximum search depth
            exploration_constant: UCB exploration parameter
            num_children_select: Number of top children to keep per expansion
            k_rollouts_per_expert: Number of rollouts per expert
            ablation_mode: "random_no_expert", "single_expert", or "multi_expert"
            single_expert_id: Expert ID for single expert mode
            backup_rule: "max" for max backup (MCTS), "sum" for sum backup (Monte Carlo)
        """
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
        
        # Derived parameters
        self.num_candidates_per_expansion = 6  # For random mode
        
        # Cache for sequence evaluation
        self.sequence_cache = SequenceCache()
        
        # Track seen sequences to avoid duplicates
        self.seen_sequences = set()
        
        print(f"🌳 GeneralMCTS initialized:")
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

            # Extract predictive uncertainty BEFORE generation for each expert
            expert_uncertainties = {}
            base_struct = self._baseline_structure.copy()
            for expert_id in expert_ids:
                try:
                    uncertainty = self.dplm2_integration.compute_predictive_entropy(
                        base_struct, masked_seq, expert_id
                    )
                    expert_uncertainties[expert_id] = uncertainty
                    print(f"   Expert {expert_id} predictive uncertainty: {uncertainty:.3f}")
                except Exception as e:
                    print(f"   Warning: Failed to compute uncertainty for expert {expert_id}: {e}")
                    expert_uncertainties[expert_id] = 1.0  # Default uncertainty

            # Build single all_candidates list with every rollout across experts
            all_candidates = []
            for expert_id in expert_ids:
                for r in range(k_rollouts):
                    try:
                        raw = self.dplm2_integration.generate_with_expert(expert_id, base_struct, len(seq), masked_seq, temperature=1.0)
                        if raw:
                            complete_seq = apply_patch(seq, raw, mask)
                            # Use cache-style de-dupe
                            seq_hash = hash(complete_seq)
                            if seq_hash not in self.sequence_cache.cache:
                                aar = compute_fast_aar(complete_seq, self.reference_sequence)
                                all_candidates.append((aar, complete_seq))
                                self.sequence_cache.cache[seq_hash] = {'aar': aar}
                    except Exception as e:
                        print(f"Expert {expert_id} rollout {r} failed: {e}")
                        continue

            # TopK from all candidates (not per-expert best)
            if all_candidates:
                # Score all candidates once and select TopK
                scored = [(self._compute_reward(c), c) for _, c in all_candidates]
                scored.sort(reverse=True, key=lambda x: x[0])
                top_candidates = [c for _, c in scored[:self.num_children_select]]
                print(f"🏆 Selected {len(top_candidates)} children from {len(all_candidates)} total rollouts")
            else:
                top_candidates = []
        
        # Create child nodes with proper entropy calculation
        children = []
        for cand in top_candidates:
            child = MCTSNode(sequence=cand, parent=node, depth=node.depth + 1)
            
            # Use pre-computed predictive uncertainty (computed BEFORE generation)
            try:
                if self.ablation_mode == "random_no_expert":
                    # Empirical entropy for random fill-in ablation
                    child.entropy_proposals = self._compute_empirical_entropy(top_candidates, list(mask))
                elif self.ablation_mode == "single_expert":
                    # Use pre-computed predictive entropy for single expert
                    expert_id = int(self.single_expert_id) % 3
                    child.entropy_proposals = self.dplm2_integration.compute_predictive_entropy(
                        base_struct, masked_seq, expert_id
                    )
                else:
                    # For multi-expert: compute child-specific ensemble surprisal
                    try:
                        child.entropy_proposals = self.dplm2_integration.compute_ensemble_surprisal(
                            base_struct, cand, list(mask)
                        )
                        print(f"   ✅ Computed ensemble surprisal: {child.entropy_proposals:.3f} for child sequence")
                    except Exception as surprisal_error:
                        print(f"   Warning: Ensemble surprisal failed: {surprisal_error}")
                        # Fallback: use average predictive uncertainty across experts
                        if 'expert_uncertainties' in locals():
                            child.entropy_proposals = sum(expert_uncertainties.values()) / len(expert_uncertainties)
                            print(f"   ⚠️ Using fallback average uncertainty: {child.entropy_proposals:.3f}")
                        else:
                            child.entropy_proposals = 1.0
                            print(f"   ⚠️ Using default entropy: 1.0")
                
                # Novelty vs parent (Hamming distance normalized)
                child.novelty_vs_parent = sum(1 for a, b in zip(cand, seq) if a != b) / len(seq)
                
                print(f"   Child entropy: {child.entropy_proposals:.3f}, novelty: {child.novelty_vs_parent:.3f}")
                
            except Exception as e:
                print(f"   Warning: Failed to compute entropy/novelty for child: {e}")
                # Fallback to simple counts
                child.entropy_proposals = len(top_candidates) / 10.0 if self.ablation_mode != "single_expert" else self.k_rollouts_per_expert / 10.0
                child.novelty_vs_parent = sum(1 for a, b in zip(cand, seq) if a != b) / len(seq)
            
            children.append(child)
        
        return children
    
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
        
        # 2. Calculate scTM score using ESMFold and TMalign
        sctm_score = 0.0
        try:
            from utils.sctm_calculation import calculate_sctm_score
            from utils.cameo_data_loader import CAMEODataLoader
            
            print(f"🧬 Calculating scTM score using ESMFold...")
            
            # Load reference coordinates directly from .pkl file (same as ablation script)
            if hasattr(self, '_baseline_structure') and self._baseline_structure:
                baseline = self._baseline_structure
                structure_idx = baseline.get('structure_idx')
                
                if structure_idx is not None:
                    # Load coordinates from .pkl file
                    scTM_loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
                    cameo_structure = scTM_loader.get_structure_by_index(structure_idx)
                    
                    if cameo_structure:
                        # Extract reference coordinates (CA atoms)
                        reference_coords = None
                        
                        if 'backbone_coords' in cameo_structure and cameo_structure['backbone_coords'] is not None:
                            coords = cameo_structure['backbone_coords']
                            if len(coords.shape) == 3 and coords.shape[1] == 3:
                                reference_coords = coords[:, 1, :]  # CA atoms at index 1
                            else:
                                reference_coords = coords
                        elif 'coordinates' in cameo_structure and cameo_structure['coordinates'] is not None:
                            reference_coords = cameo_structure['coordinates']
                        elif 'atom_positions' in cameo_structure and cameo_structure['atom_positions'] is not None:
                            coords = cameo_structure['atom_positions']
                            if len(coords.shape) == 3 and coords.shape[1] >= 2:
                                reference_coords = coords[:, 1, :]  # CA atoms at index 1
                            else:
                                reference_coords = coords
                        
                        if reference_coords is not None and hasattr(reference_coords, 'shape'):
                            sctm_score = calculate_sctm_score(sequence, reference_coords)
                            if sctm_score is not None and sctm_score > 0:
                                print(f"✅ scTM score: {sctm_score:.3f}")
                            else:
                                print(f"⚠️ scTM calculation returned invalid score: {sctm_score}")
                                sctm_score = 0.0
                        else:
                            print(f"⚠️ No valid reference coordinates found in .pkl file")
                            sctm_score = 0.0
                    else:
                        print(f"⚠️ Could not load .pkl file for scTM calculation")
                        sctm_score = 0.0
                else:
                    print(f"⚠️ No structure_idx available for scTM calculation")
                    sctm_score = 0.0
            else:
                print(f"⚠️ No baseline structure available for scTM calculation")
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
        
        # 4. Combine rewards (more balanced AAR vs scTM)
        # Previously: 85% AAR, 10% scTM, 5% Biophysical
        # Updated:    60% AAR, 35% scTM, 5% Biophysical
        aar_weight = 0.60
        sctm_weight = 0.35
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


# Backward compatibility alias
SequenceLevelMCTS = GeneralMCTS
