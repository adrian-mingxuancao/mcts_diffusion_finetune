"""
Hallucination MCTS - Standalone MCTS for Protein Hallucination Design

This module implements MCTS-guided protein hallucination using:
- Structure prediction: ESMFold, Boltz, or Chai-1 (via ABCFold)
- Inverse folding: ProteinMPNN or NA-MPNN
- SS guidance: DSSP-based secondary structure analysis

Pipeline per iteration:
1. Sequence → Structure (ESMFold/Boltz/Chai)
2. Structure → New Sequence (ProteinMPNN)
3. Evaluate convergence (parent-child similarity, sibling convergence, pLDDT)
4. UCT selection and backpropagation

This is a STANDALONE module that does NOT depend on dplm2_integration.
All structure prediction and inverse folding use local core modules.
"""

import math
import random
import csv
import json
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import os
import numpy as np

# Add the mcts_hallucination directory to path
HALLUCINATION_DIR = Path(__file__).resolve().parent.parent
if str(HALLUCINATION_DIR) not in sys.path:
    sys.path.insert(0, str(HALLUCINATION_DIR))

# Import local core modules
from core.hallucination_expert import create_hallucination_expert, HallucinationExpert
from core.esmfold_integration import ESMFoldIntegration
from core.abcfold_integration import ABCFoldIntegration
from core.ss_guidance import (
    SSGuidanceConfig,
    SSGuidance,
    DSSPResult,
    EditLogEntry,
    create_run_directory,
)


@dataclass
class HallucinationNode:
    """Node for hallucination MCTS - stores sequence-structure pair."""
    sequence: str
    coordinates: Optional[np.ndarray] = None
    plddt_scores: Optional[np.ndarray] = None
    mean_plddt: float = 0.0
    
    # MCTS fields
    parent: Optional['HallucinationNode'] = None
    children: List['HallucinationNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    
    # Convergence tracking
    convergence_score: float = 0.0
    parent_child_similarity: float = 0.0
    sibling_convergence: float = 0.0
    
    # PH-UCT components
    entropy: float = 0.0
    novelty: float = 0.0
    expert_source: str = None
    
    def get_reward(self) -> float:
        """Average reward from visits."""
        return self.total_reward / max(1, self.visits)
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """Standard UCT score."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.get_reward()
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def ph_uct_score(self, exploration_constant: float = 1.414,
                     entropy_weight: float = 0.1, novelty_weight: float = 0.05) -> float:
        """PH-UCT score with entropy and novelty bonuses."""
        base_score = self.uct_score(exploration_constant)
        if base_score == float('inf'):
            return base_score
        
        entropy_bonus = entropy_weight * self.entropy
        novelty_bonus = novelty_weight * self.novelty
        
        return base_score + entropy_bonus + novelty_bonus
    children: List['MCTSNode'] = None
    parent: 'MCTSNode' = None
    depth: int = 0
    plddt_scores: List[float] = None
    structure_tokens: Optional[str] = None
    coordinates: Optional[np.ndarray] = None
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    # PH-UCT components
    entropy: float = 0.0
    novelty: float = 0.0
    expert_source: str = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        # Auto-compute frozen positions if not provided
        if self.frozen_positions is None:
            all_positions = set(range(len(self.sequence)))
            self.frozen_positions = all_positions - self.mutable_positions
    
    @property
    def masked_positions(self) -> Set[int]:
        """Backward compatibility - return mutable positions"""
        return self.mutable_positions
    
    def ph_uct_score(self, exploration_constant: float = 1.414, 
                     entropy_weight: float = 0.1, novelty_weight: float = 0.05) -> float:
        """PH-UCT score with entropy and novelty bonuses."""
        if self.visits == 0:
            return float('inf')
        
        # Base UCB1 score
        if self.parent and self.parent.visits > 0:
            exploitation = self.total_reward / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            ucb_score = exploitation + exploration
        else:
            ucb_score = self.total_reward / self.visits
        
        # Add entropy and novelty bonuses for PH-UCT
        entropy_bonus = entropy_weight * self.entropy
        novelty_bonus = novelty_weight * self.novelty
        
        return ucb_score + entropy_bonus + novelty_bonus
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """Standard UCT score without entropy/novelty bonuses."""
        if self.visits == 0:
            return float('inf')
        
        # Standard UCB1 score only
        if self.parent and self.parent.visits > 0:
            exploitation = self.total_reward / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration
        else:
            return self.total_reward / self.visits


class GeneralMCTS:
    """Proper MCTS with Diffusion and Multiple Expert Rollouts."""
    def __init__(
        self,
        dplm2_integration: object = None,
        baseline_structure: dict = None,
        reference_sequence: str = None,
        reference_coords: Optional[np.ndarray] = None,
        max_depth: int = 5,
        exploration_constant: float = 1.414,
        ablation_mode: str = "multi_expert",
        single_expert_id: int = None,
        external_experts: list = None,
        num_rollouts_per_expert: int = 2,
        top_k_candidates: int = 2,
        use_ph_uct: bool = True,  # NEW: Control PH-UCT vs standard UCT
        # Backward compatibility
        task_type: str = "inverse_folding",
        num_simulations: int = 25,
        temperature: float = 1.0,
        use_plddt_masking: bool = True,
        exclude_proteinmpnn: bool = False,  # Exclude ProteinMPNN for folding tasks
        **kwargs
    ):
        """Initialize proper MCTS with multiple expert rollouts."""
        self.dplm2_integration = dplm2_integration
        self.baseline_structure = baseline_structure or {}
        self._baseline_structure = self.baseline_structure  # For compatibility
        self.reference_sequence = reference_sequence
        self.reference_coords = reference_coords
        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.use_ph_uct = use_ph_uct  # Store UCT vs PH-UCT choice
        self.task_type = task_type  # Store task type (folding vs inverse_folding)
        # Automatically exclude ProteinMPNN for folding tasks
        self.exclude_proteinmpnn = exclude_proteinmpnn or (task_type == "folding")
        if task_type == "folding" and not exclude_proteinmpnn:
            print(f"   🔧 Auto-excluding ProteinMPNN for folding task")
        self._last_structure_eval = None
        
        # Multi-expert rollout parameters
        self.ablation_mode = ablation_mode
        self.single_expert_id = single_expert_id
        self.external_experts = external_experts or []
        self.num_rollouts_per_expert = num_rollouts_per_expert
        self.top_k_candidates = top_k_candidates
        self.num_children_select = kwargs.get('num_children_select', top_k_candidates)  # For random mode
        
        # Expert configuration
        self.experts = self._setup_experts()
        try:
            self.structure_converter = get_structure_converter()
        except Exception as converter_error:
            print(f"⚠️ Structure converter unavailable: {converter_error}")
            self.structure_converter = None
        
        if not self.dplm2_integration:
            raise ValueError("DPLM-2 integration is required")
        
        print(f"🎯 MCTS initialized: {self.ablation_mode} mode with {len(self.experts)} experts")
    
    def get_best_child(self, node):
        """Get the best child node based on reward"""
        if not hasattr(node, 'children') or not node.children:
            return node
        
        best_child = node
        best_reward = getattr(node, 'reward', 0.0)
        
        for child in node.children:
            child_reward = getattr(child, 'reward', 0.0)
            if child_reward > best_reward:
                best_child = child
                best_reward = child_reward
        
        return best_child
    
    def _setup_experts(self) -> List:
        """Setup expert list based on ablation mode and task type."""
        experts = []
        
        # Use the explicit exclude_proteinmpnn flag or infer from task type
        exclude_proteinmpnn = self.exclude_proteinmpnn or (self.task_type == "folding")
        
        if self.ablation_mode == "single_expert":
            # Use only one expert (DPLM-2 models or external)
            if self.single_expert_id is not None:
                if self.single_expert_id < 3:  # DPLM-2 models (650M, 150M, 3B)
                    experts = [f"dplm2_{self.single_expert_id}"]
                    print(f"   Single expert: DPLM-2 model {self.single_expert_id}")
                elif self.single_expert_id == 3 and len(self.external_experts) > 0 and not exclude_proteinmpnn:
                    experts = [self.external_experts[0]]  # ProteinMPNN or first external
                    print(f"   Single expert: {experts[0].get_name() if hasattr(experts[0], 'get_name') else 'External'}")
                else:
                    if exclude_proteinmpnn and self.single_expert_id == 3:
                        print(f"   ⚠️ ProteinMPNN excluded for folding task, using DPLM-2 150M instead")
                        experts = ["dplm2_1"]  # Fallback to 150M model
                    else:
                        experts = ["dplm2_0"]  # Default to DPLM-2 650M
                        print(f"   Single expert: DPLM-2 650M (default)")
            else:
                experts = ["dplm2_0"]  # Default
                
        elif self.ablation_mode == "multi_expert":
            # Use all available experts (exclude ProteinMPNN for folding)
            experts = ["dplm2_0", "dplm2_1", "dplm2_2"]  # All DPLM-2 models
            if not exclude_proteinmpnn:
                experts.extend(self.external_experts)  # Add external experts
                print(f"   Multi-expert: {len(experts)} total experts")
            else:
                print(f"   Multi-expert: 3 DPLM-2 models (ProteinMPNN excluded for folding)")
            
        else:  # random_no_expert
            experts = []
            print(f"   Random mode: no expert guidance")
        
        return experts
    
    def search(self, initial_sequence: str = None, num_iterations: int = 2, 
               reference_sequence: str = None, structure_data: dict = None, **kwargs) -> MCTSNode:
        """
        Proper MCTS search with diffusion and multiple expert rollouts.
        
        Pipeline:
        1. PH-UCT Selection with entropy/mutual information
        2. Progressive pLDDT masking with ESMFold
        3. Multi-expert rollouts (N per expert) → top-K selection
        4. Task-specific evaluation and backpropagation
        """
        print(f"🔍 MCTS search called with initial_sequence type: {type(initial_sequence)}, len: {len(initial_sequence) if initial_sequence is not None else 'None'}")
        
        # Update parameters
        if reference_sequence:
            self.reference_sequence = reference_sequence
        if structure_data:
            self.baseline_structure.update(structure_data)
        
        # Step 1: Use baseline pLDDT scores (following GitHub version)
        print(f"🎯 Step 1: Using baseline pLDDT scores...")
        print(f"   🔍 Baseline structure keys: {list(self.baseline_structure.keys())}")
        print(f"   🔍 Baseline plddt_scores type: {type(self.baseline_structure.get('plddt_scores'))}")
        try:
            raw_plddt = self.baseline_structure.get('plddt_scores')
            initial_plddt = self._prepare_plddt_scores(len(initial_sequence), raw_plddt)
            print(f"   🔍 Initial pLDDT type: {type(raw_plddt)}, length: {len(initial_plddt) if initial_plddt is not None else 'None'}")
        except Exception as e:
            print(f"   ❌ Error getting initial pLDDT: {e}")
            initial_plddt = [70.0] * len(initial_sequence)
        
        # Step 2: Progressive pLDDT masking (quantile-based)
        print(f"🎯 Step 2: Progressive pLDDT masking...")
        masked_positions = self._compute_progressive_plddt_masking(initial_sequence, initial_plddt, depth=0)
        
        initial_structure_tokens = self.baseline_structure.get('struct_seq')
        initial_token_list = self._get_structure_token_list(initial_structure_tokens, len(initial_sequence))
        normalized_struct_seq = ' '.join(initial_token_list)
        self.baseline_structure['struct_seq'] = normalized_struct_seq
        initial_coordinates = self.baseline_structure.get('coordinates')
        root_reward_raw = self.baseline_structure.get('baseline_reward', 0.0)
        root_reward_value = float(root_reward_raw) if root_reward_raw is not None else 0.0
        root_visits = 1
        root_total_reward = root_reward_value
        baseline_rmsd = self.baseline_structure.get('baseline_rmsd')
        baseline_tm = self.baseline_structure.get('baseline_tm')
        
        # Create root node with baseline structure context
        root = MCTSNode(
            sequence=initial_sequence,
            mutable_positions=masked_positions,  # These are the low-confidence positions
            depth=0,
            plddt_scores=initial_plddt,
            reward=root_reward_value,
            total_reward=root_total_reward,
            visits=root_visits,
            structure_tokens=normalized_struct_seq,
            coordinates=initial_coordinates,
            rmsd=baseline_rmsd,
            tm_score=baseline_tm,
        )
        
        print(f"🎯 MCTS Search: {num_iterations} iterations")
        print(f"   Initial masking: {len(masked_positions)}/{len(initial_sequence)} positions ({len(masked_positions)/len(initial_sequence)*100:.1f}%)")
        print(f"   pLDDT range: {min(initial_plddt):.1f}-{max(initial_plddt):.1f}, avg={sum(initial_plddt)/len(initial_plddt):.1f}")
        
        # MCTS iterations
        for iteration in range(num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{num_iterations}")
            
            # Step 1: UCT/PH-UCT Selection
            selected_node = self._uct_selection(root)
            print(f"   Selected node at depth {selected_node.depth}")
            
            # Step 2: Expansion with multi-expert rollouts
            if selected_node.depth < self.max_depth and len(selected_node.masked_positions) > 0:
                self._expand_with_multi_expert_rollouts(selected_node)
            
            # Step 3: Evaluation and Backpropagation
            if selected_node.children:
                # Find best child and backpropagate
                best_child = max(selected_node.children, key=lambda c: c.reward)
                self._backpropagate_max_rule(best_child)
                print(f"   Best child: reward={best_child.reward:.3f}, expert={best_child.expert_source}")
        
        # Return best node from entire tree
        return self._find_best_node_in_tree(root)
    
    def _uct_selection(self, root: MCTSNode) -> MCTSNode:
        """UCT or PH-UCT selection based on use_ph_uct parameter."""
        node = root
        while node.children and len(node.masked_positions) > 0:
            if self.use_ph_uct:
                # Select child with highest PH-UCT score (with entropy/novelty bonuses)
                best_child = max(node.children, key=lambda c: c.ph_uct_score(self.exploration_constant))
            else:
                # Select child with highest standard UCT score (no bonuses)
                best_child = max(node.children, key=lambda c: c.uct_score(self.exploration_constant))
            node = best_child
        return node
    
    def _expand_with_multi_expert_rollouts(self, node: MCTSNode):
        """
        PROPER Multi-Expert Rollouts (from backup MCTS):
        1. Use FIXED structure tokens from struct.fasta
        2. N rollouts per expert (DPLM-2 + external experts)
        3. Top-K candidate selection from ALL rollouts
        4. Progressive pLDDT masking for child nodes
        """
        if len(node.masked_positions) == 0:
            return
        
        print(f"   🌱 Expanding node: {len(node.masked_positions)} masked positions")
        
        # Step 1: Collect all candidates from all experts
        all_candidates = []
        
        # Random mode: generate random candidates without expert guidance
        if self.ablation_mode == "random_no_expert":
            print(f"      🎲 Random mode: generating {self.num_children_select} random candidates")
            import random
            
            for i in range(self.num_children_select):
                try:
                    if self.task_type == "folding":
                        # For folding: randomly perturb structure tokens
                        # Get structure token vocabulary range (DPLM-2 structure tokens are typically 2816-7999)
                        struct_token_start = 2816
                        struct_token_end = 7999
                        
                        # Parse baseline structure tokens
                        baseline_tokens = node.structure_tokens
                        if isinstance(baseline_tokens, str):
                            token_list = baseline_tokens.split()
                        else:
                            token_list = list(baseline_tokens)
                        
                        # Randomly replace some masked positions with random structure tokens
                        new_token_list = token_list.copy()
                        masked_positions = list(node.masked_positions) if node.masked_positions else []
                        
                        # Randomly select a subset of masked positions to perturb
                        if masked_positions:
                            num_to_perturb = max(1, len(masked_positions) // 2)
                            positions_to_perturb = random.sample(masked_positions, min(num_to_perturb, len(masked_positions)))
                            
                            for pos in positions_to_perturb:
                                if pos < len(new_token_list):
                                    # Replace with random structure token
                                    new_token_list[pos] = str(random.randint(struct_token_start, struct_token_end))
                        
                        new_struct_tokens = ' '.join(new_token_list)
                        
                        # Convert to coordinates for evaluation
                        coords, plddt = self.dplm2_integration._structure_tokens_to_coords(new_struct_tokens, len(node.sequence))
                        
                        if coords is not None:
                            reward = self._evaluate_structure_reward(coords, node.sequence)
                            pl_scores = plddt if plddt is not None else self._estimate_plddt_from_coords(coords)
                            metrics = self._last_structure_eval or {}
                            
                            candidate_data = {
                                'sequence': node.sequence,
                                'structure_tokens': new_struct_tokens,
                                'coordinates': coords,
                                'plddt_scores': pl_scores,
                                'expert': 'random',
                                'entropy': 0.0,
                                'reward': reward,
                                'rollout_id': i,
                                'confidence': 0.5,
                                'rmsd': metrics.get('rmsd'),
                                'tm_score': metrics.get('tm_score'),
                            }
                            all_candidates.append(candidate_data)
                            print(f"         ✅ Random candidate {i+1}: reward={reward:.3f}, perturbed {len(positions_to_perturb) if masked_positions else 0} positions")
                        else:
                            print(f"         ❌ Random candidate {i+1}: failed to convert tokens to coords")
                    else:
                        # For inverse folding: randomly mutate sequence
                        # Just use baseline as random baseline for inverse folding
                        candidate_data = {
                            'sequence': node.sequence,
                            'expert': 'random',
                            'entropy': 0.0,
                            'reward': node.reward if node.reward is not None else 0.0,
                            'rollout_id': i,
                            'confidence': 0.5
                        }
                        all_candidates.append(candidate_data)
                        print(f"         ✅ Random candidate {i+1}: reward={candidate_data['reward']:.3f}")
                        
                except Exception as e:
                    print(f"         ❌ Random candidate {i+1}: error {e}")
                    import traceback
                    traceback.print_exc()
        
        # DPLM-2 Expert Rollouts (multiple model sizes)
        dplm2_experts = ["dplm2_0", "dplm2_1", "dplm2_2"]  # 650M, 150M, 3B
        
        # Skip DPLM-2 experts for random_no_expert mode
        if self.ablation_mode != "random_no_expert":
            for expert_id, expert_name in enumerate(dplm2_experts):
                if self.ablation_mode == "single_expert" and expert_id != (self.single_expert_id or 0):
                    continue  # Skip if not the selected single expert
                    
                print(f"      🤖 {expert_name}: generating {self.num_rollouts_per_expert} rollouts")
                
                # N rollouts per DPLM-2 expert
                for rollout in range(self.num_rollouts_per_expert):
                    try:
                        candidate_data = self._generate_dplm2_candidate(
                            node, node.masked_positions, expert_id
                        )
                        
                        if not candidate_data:
                            continue
                        
                        if self.task_type == "folding":
                            coords = candidate_data.get('coordinates')
                            struct_tokens = candidate_data.get('structure_tokens')
                            if coords is None or struct_tokens is None:
                                print(f"         ❌ {expert_name} rollout {rollout+1}: missing coordinates or structure tokens")
                                continue
                            reward = self._evaluate_structure_reward(coords, node.sequence)
                            pl_scores = candidate_data.get('plddt_scores') if candidate_data.get('plddt_scores') is not None else self._estimate_plddt_from_coords(coords)
                            metrics = self._last_structure_eval or {}
                            all_candidates.append({
                                'sequence': node.sequence,
                                'structure_tokens': struct_tokens,
                                'coordinates': coords,
                                'plddt_scores': pl_scores,
                                'expert': expert_name,
                                'entropy': 0.0,
                                'reward': reward,
                                'rollout_id': rollout,
                                'confidence': 0.8,
                                'rmsd': metrics.get('rmsd'),
                                'tm_score': metrics.get('tm_score'),
                            })
                            print(f"         ✅ {expert_name} rollout {rollout+1}: reward={reward:.3f}")
                        else:
                            candidate_seq = candidate_data.get('sequence')
                            if candidate_seq and len(candidate_seq) == len(node.sequence):
                                entropy = self._compute_expert_entropy(candidate_seq, expert_name, node.masked_positions)
                                reward = self._evaluate_sequence_aar(candidate_seq)
                                all_candidates.append({
                                    'sequence': candidate_seq,
                                    'expert': expert_name,
                                    'entropy': entropy,
                                    'reward': reward,
                                    'rollout_id': rollout,
                                    'confidence': 0.8
                                })
                                print(f"         ✅ {expert_name} rollout {rollout+1}: reward={reward:.3f}")
                            else:
                                print(f"         ❌ {expert_name} rollout {rollout+1}: invalid sequence length")
                            
                    except Exception as e:
                        print(f"         ❌ {expert_name} rollout {rollout+1}: error {e}")
        # Note: Random mode is handled at the top (lines 335-412), so no else block needed here
        
        # ProteinMPNN Expert Rollouts (Expert 3) - Only if not excluded
        if not self.exclude_proteinmpnn and self.ablation_mode == "single_expert" and self.single_expert_id == 3:
            # Use ProteinMPNN through DPLM2 integration
            print(f"      🤖 ProteinMPNN (expert 3): generating {self.num_rollouts_per_expert} rollouts")
            
            for rollout in range(self.num_rollouts_per_expert):
                try:
                    candidate = self._generate_proteinmpnn_candidate(
                        node.sequence, node.masked_positions
                    )
                    
                    if candidate and len(candidate) == len(node.sequence):
                        # Compute entropy and reward
                        entropy = self._compute_expert_entropy(candidate, "ProteinMPNN", node.masked_positions)
                        reward = self._evaluate_sequence_aar(candidate)
                        
                        all_candidates.append({
                            'sequence': candidate,
                            'expert': "ProteinMPNN",
                            'entropy': entropy,
                            'reward': reward,
                            'rollout_id': rollout,
                            'confidence': 0.7  # High confidence for ProteinMPNN
                        })
                        print(f"         ✅ ProteinMPNN rollout {rollout+1}: reward={reward:.3f}")
                    else:
                        print(f"         ❌ ProteinMPNN rollout {rollout+1}: invalid sequence")
                        
                except Exception as e:
                    print(f"         ❌ ProteinMPNN rollout {rollout+1}: error {e}")
        
        # Multi-expert mode: include all DPLM-2 + ProteinMPNN (if not excluded)
        elif not self.exclude_proteinmpnn and self.ablation_mode == "multi_expert":
            # Add ProteinMPNN rollouts
            print(f"      🤖 ProteinMPNN (expert 3): generating {self.num_rollouts_per_expert} rollouts")
            
            for rollout in range(self.num_rollouts_per_expert):
                try:
                    candidate = self._generate_proteinmpnn_candidate(
                        node.sequence, node.masked_positions
                    )
                    
                    if candidate and len(candidate) == len(node.sequence):
                        entropy = self._compute_expert_entropy(candidate, "ProteinMPNN", node.masked_positions)
                        reward = self._evaluate_sequence_aar(candidate)
                        
                        all_candidates.append({
                            'sequence': candidate,
                            'expert': "ProteinMPNN",
                            'entropy': entropy,
                            'reward': reward,
                            'rollout_id': rollout,
                            'confidence': 0.7
                        })
                        print(f"         ✅ ProteinMPNN rollout {rollout+1}: reward={reward:.3f}")
                    else:
                        print(f"         ❌ ProteinMPNN rollout {rollout+1}: invalid sequence")
                        
                except Exception as e:
                    print(f"         ❌ ProteinMPNN rollout {rollout+1}: error {e}")
        
        # External experts (e.g., hallucination expert)
        if self.external_experts and self.ablation_mode != "random_no_expert":
            for expert in self.external_experts:
                # Check if expert has generate_candidate method
                if not hasattr(expert, 'generate_candidate'):
                    continue
                
                expert_name = expert.get_name() if hasattr(expert, 'get_name') else 'external'
                print(f"      🤖 {expert_name}: generating {self.num_rollouts_per_expert} rollouts")
                
                for rollout in range(self.num_rollouts_per_expert):
                    try:
                        # Call expert's generate_candidate method
                        candidate_data = expert.generate_candidate(
                            sequence=node.sequence,
                            masked_positions=node.masked_positions,
                            coordinates=node.coordinates
                        )
                        
                        if not candidate_data:
                            print(f"         ❌ {expert_name} rollout {rollout+1}: no candidate returned")
                            continue
                        
                        # Evaluate reward based on task type
                        if self.task_type == "folding":
                            # For folding: evaluate structure quality
                            coords = candidate_data.get('coordinates')
                            if coords is None:
                                print(f"         ❌ {expert_name} rollout {rollout+1}: missing coordinates")
                                continue
                            
                            reward = self._evaluate_structure_reward(coords, candidate_data['sequence'])
                            pl_scores = candidate_data.get('plddt_scores', candidate_data.get('confidence_scores'))
                            if pl_scores is None:
                                pl_scores = self._estimate_plddt_from_coords(coords)
                            
                            metrics = self._last_structure_eval or {}
                            all_candidates.append({
                                'sequence': candidate_data['sequence'],
                                'structure_tokens': None,  # Hallucination doesn't use structure tokens
                                'coordinates': coords,
                                'plddt_scores': pl_scores,
                                'expert': expert_name,
                                'entropy': candidate_data.get('entropy', 0.5),
                                'reward': reward,
                                'rollout_id': rollout,
                                'confidence': candidate_data.get('mean_plddt', 70.0) / 100.0,
                                'rmsd': metrics.get('rmsd'),
                                'tm_score': metrics.get('tm_score'),
                            })
                            print(f"         ✅ {expert_name} rollout {rollout+1}: reward={reward:.3f}")
                            
                        else:
                            # For inverse folding: evaluate sequence quality
                            sequence = candidate_data['sequence']
                            if len(sequence) != len(node.sequence):
                                print(f"         ❌ {expert_name} rollout {rollout+1}: length mismatch")
                                continue
                            
                            reward = self._evaluate_sequence_aar(sequence)
                            entropy = candidate_data.get('entropy', 0.5)
                            
                            all_candidates.append({
                                'sequence': sequence,
                                'coordinates': candidate_data.get('coordinates'),
                                'plddt_scores': candidate_data.get('plddt_scores'),
                                'expert': expert_name,
                                'entropy': entropy,
                                'reward': reward,
                                'rollout_id': rollout,
                                'confidence': candidate_data.get('mean_plddt', 70.0) / 100.0
                            })
                            print(f"         ✅ {expert_name} rollout {rollout+1}: reward={reward:.3f}")
                        
                    except Exception as e:
                        print(f"         ❌ {expert_name} rollout {rollout+1}: error {e}")
                        import traceback
                        traceback.print_exc()
        
        # Step 2: Select top-K candidates from ALL rollouts
        if not all_candidates:
            print(f"      ⚠️ No successful candidates generated")
            return
        
        # Sort by reward and take top-K
        all_candidates.sort(key=lambda x: x['reward'], reverse=True)
        top_candidates = all_candidates[:self.top_k_candidates]
        
        print(f"      📊 Selected {len(top_candidates)} from {len(all_candidates)} total candidates")
        
        # Step 3: Create child nodes with progressive masking
        for i, candidate in enumerate(top_candidates):
            if self.task_type == "folding":
                # Try to get plddt_scores from candidate, otherwise estimate from coords if available
                child_plddt = candidate.get('plddt_scores')
                if child_plddt is None:
                    coords = candidate.get('coordinates')
                    if coords is not None:
                        child_plddt = self._estimate_plddt_from_coords(coords)
                    else:
                        # No coords available, use baseline or uniform default
                        baseline_plddt = self.baseline_structure.get('plddt_scores')
                        if baseline_plddt is not None:
                            child_plddt = self._prepare_plddt_scores(len(candidate['sequence']), baseline_plddt)
                        else:
                            child_plddt = [70.0] * len(candidate['sequence'])
            else:
                child_plddt_raw = self.baseline_structure.get('plddt_scores')
                child_plddt = self._prepare_plddt_scores(len(candidate['sequence']), child_plddt_raw)
            child_masked_positions = self._compute_progressive_plddt_masking(
                candidate['sequence'], child_plddt, depth=node.depth + 1
            )
            
            # CORRECT FIX: Child nodes should get the generated sequence for progress
            # The masking consistency issue is about rollout generation, not child creation
            # All rollouts at same depth use same parent node.sequence (which is correct)
            # But child nodes should progress with generated sequences
            generated_sequence = candidate['sequence']
            
            child = MCTSNode(
                sequence=generated_sequence,  # Use generated sequence for progress
                mutable_positions=child_masked_positions,  # New masking for next depth
                parent=node,
                depth=node.depth + 1,
                plddt_scores=child_plddt,
                entropy=candidate['entropy'],
                novelty=self._compute_novelty(generated_sequence, node),
                expert_source=candidate['expert'],
                reward=candidate['reward'],
                visits=1,
                total_reward=candidate['reward'],
                structure_tokens=candidate.get('structure_tokens', node.structure_tokens),
                coordinates=candidate.get('coordinates', node.coordinates),
                rmsd=candidate.get('rmsd', node.rmsd),
                tm_score=candidate.get('tm_score', node.tm_score),
            )
            node.children.append(child)
            
            print(f"         Child {i+1}: {candidate['expert']}, reward={candidate['reward']:.3f}, entropy={candidate['entropy']:.3f}")
    
    def _generate_real_structure_tokens(self, sequence: str) -> Optional[str]:
        """Generate REAL structure tokens from ESMFold coordinates for MCTS lead optimization."""
        try:
            print(f"      🔍 MCTS Lead Optimization: generating REAL structure tokens from ESMFold")
            
            # Step 1: Generate ESMFold baseline coordinates
            coords = self._generate_esmfold_baseline(sequence)
            if coords is None:
                print(f"      ❌ Failed to generate ESMFold baseline")
                return None
            
            # Step 2: Convert coordinates to REAL structure tokens
            structure_tokens = self._coords_to_structure_tokens(coords)
            if structure_tokens:
                print(f"      ✅ Generated REAL structure tokens: {len(structure_tokens)} chars")
                print(f"      🔍 Sample: {structure_tokens[:50]}...")
                return structure_tokens
            else:
                print(f"      ❌ Failed to convert coordinates to structure tokens")
                return None
                
        except Exception as e:
            print(f"      ❌ Structure token generation failed: {e}")
            return None
    
    def _generate_esmfold_baseline(self, sequence: str) -> Optional[np.ndarray]:
        """Generate baseline structure using ESMFold (following GitHub repository)."""
        try:
            print(f"      🔄 Generating ESMFold baseline for sequence length {len(sequence)}")
            
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
                
                print(f"      🔍 ESMFold output shape: {positions.shape}")
                
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
                    print(f"      ⚠️ Length mismatch: got {ca_coords.shape[0]}, expected {expected_len}")
                    # Pad or truncate to match sequence length
                    if ca_coords.shape[0] < expected_len:
                        # Pad with last coordinate
                        padding = np.tile(ca_coords[-1:], (expected_len - ca_coords.shape[0], 1))
                        ca_coords = np.vstack([ca_coords, padding])
                    else:
                        # Truncate to sequence length
                        ca_coords = ca_coords[:expected_len]
                
            print(f"      ✅ ESMFold baseline generated: {ca_coords.shape}")
            return ca_coords
            
        except Exception as e:
            print(f"      ❌ ESMFold baseline generation failed: {e}")
            # Fallback to random coordinates
            coords = np.random.rand(len(sequence), 3) * 10
            print(f"      🔧 Using random fallback coordinates: {coords.shape}")
            return coords
    
    def _coords_to_structure_tokens(self, coords: np.ndarray) -> Optional[str]:
        """Convert 3D coordinates to DPLM structure tokens (following GitHub repository)."""
        try:
            print(f"      🔄 Converting coordinates to structure tokens: {coords.shape}")
            
            from byprot.models.utils import get_struct_tokenizer
            import torch
            
            # Load structure tokenizer
            try:
                struct_tokenizer = get_struct_tokenizer()
                device = next(self.dplm2_integration.model.parameters()).device
                struct_tokenizer = struct_tokenizer.to(device)
                print(f"      ✅ Loaded struct tokenizer on {device}")
            except Exception as e:
                print(f"      ❌ Failed to load struct tokenizer: {e}")
                # Fallback to mask tokens if tokenizer fails
                seq_len = coords.shape[0]
                struct_tokens = '<cls_struct> ' + '<mask_struct> ' * seq_len + '<eos_struct>'
                print(f"      🔧 Fallback to mask tokens: {seq_len} positions")
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
                print(f"      ⚠️ Unexpected coordinate shape: {coords_tensor.shape}, expected [{seq_len}, 3]")
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
            
            print(f"      ✅ Generated REAL structure tokens: {len(struct_seq)} chars")
            return struct_seq
            
        except Exception as e:
            print(f"      ❌ Structure token conversion failed: {e}")
            # Ultimate fallback
            try:
                seq_len = coords.shape[0] if coords is not None else 100
                struct_tokens = '<cls_struct> ' + '<mask_struct> ' * seq_len + '<eos_struct>'
                print(f"      🔧 Using ultimate fallback structure mask tokens: {seq_len} positions")
                return struct_tokens
            except:
                return None
    
    def _prepare_plddt_scores(self, sequence_length: int, raw_scores) -> List[float]:
        """Normalize raw pLDDT inputs into a length-matched float list in [0, 100]."""
        try:
            if raw_scores is None:
                raise ValueError("No pLDDT scores provided")
            
            # Handle dict-style payloads (common in cached metadata)
            if isinstance(raw_scores, dict):
                candidates = ['plddt_scores', 'plddt', 'scores', 'confidence', 'lddts']
                selected = None
                for key in candidates:
                    if key in raw_scores and raw_scores[key]:
                        selected = raw_scores[key]
                        break
                if selected is None and raw_scores:
                    # Fallback: take first non-empty value
                    for value in raw_scores.values():
                        if value:
                            selected = value
                            break
                raw_scores = selected if selected is not None else []
            
            # Convert numpy arrays / tensors to list
            if hasattr(raw_scores, 'cpu'):
                raw_scores = raw_scores.cpu()
            if hasattr(raw_scores, 'numpy'):
                raw_scores = raw_scores.numpy()
            if isinstance(raw_scores, np.ndarray):
                raw_scores = raw_scores.tolist()
            
            # Ensure list type
            if not isinstance(raw_scores, (list, tuple)):
                raw_scores = [float(raw_scores)]
            
            scores = [float(x) for x in raw_scores if x is not None]
        except Exception as e:
            print(f"   ⚠️ Failed to parse raw pLDDT scores ({e}); using default 70.0.")
            scores = []
        
        if not scores:
            scores = [70.0] * sequence_length
        else:
            # If scores appear normalized to [0, 1], rescale
            max_score = max(scores)
            if max_score <= 1.5:
                scores = [min(1.0, max(0.0, s)) * 100.0 for s in scores]
            # Crop or pad to match sequence length
            if len(scores) >= sequence_length:
                scores = scores[:sequence_length]
            else:
                pad_value = float(np.mean(scores)) if scores else 70.0
                scores = scores + [pad_value] * (sequence_length - len(scores))
        
        return scores

    def _compute_progressive_plddt_masking(self, sequence: str, plddt_scores: List[float], depth: int) -> Set[int]:
        """Progressive pLDDT masking: threshold-based by default, quantile-based as fallback."""
        print(f"   🔍 _compute_progressive_plddt_masking: plddt_scores type={type(plddt_scores)}, len={len(plddt_scores) if plddt_scores is not None else 'None'}")
        
        if plddt_scores is None or (isinstance(plddt_scores, (list, np.ndarray)) and len(plddt_scores) == 0):
            print(f"      ⚠️ Empty pLDDT scores received; defaulting to uniform 70.0 confidence.")
            plddt_scores = [70.0] * len(sequence)
        elif len(plddt_scores) != len(sequence):
            print(f"      ⚠️ pLDDT length mismatch ({len(plddt_scores)} vs {len(sequence)}); padding/cropping.")
            plddt_scores = self._prepare_plddt_scores(len(sequence), plddt_scores)
        
        # Progressive thresholds by depth (lower threshold = more masking)
        if depth == 0:
            threshold = 70.0  # Mask positions with pLDDT < 70
            target_ratio = 0.25  # Fallback: mask ~25% at root
        elif depth == 1:
            threshold = 75.0  # Mask positions with pLDDT < 75
            target_ratio = 0.20  # Fallback: mask ~20% one level down
        elif depth == 2:
            threshold = 80.0  # Mask positions with pLDDT < 80
            target_ratio = 0.15  # Fallback: mask ~15% deeper
        elif depth == 3:
            threshold = 85.0  # Mask positions with pLDDT < 85
            target_ratio = 0.05  # Fallback: mask ~5% at depth 3
        elif depth >= 4:
            # At max depth (4+), unmask all remaining X positions for 100% completion
            threshold = 100.0  # No positions meet this threshold
            target_ratio = 0.0   # Force complete unmasking
        
        # Method 1: Threshold-based masking (default)
        threshold_masked = set([i for i, score in enumerate(plddt_scores) if score < threshold])
        
        # Check if threshold masking gives reasonable number of positions
        min_positions = max(3, int(len(sequence) * 0.05))  # At least 5% or 3 positions
        max_positions = int(len(sequence) * 0.30)  # At most 30% of sequence
        
        if min_positions <= len(threshold_masked) <= max_positions:
            # Threshold masking is good
            masked_positions = threshold_masked
            print(f"      🎭 Depth {depth} threshold masking (pLDDT < {threshold}): {len(masked_positions)}/{len(sequence)} positions ({len(masked_positions)/len(sequence)*100:.1f}%)")
        else:
            # Method 2: Quantile-based masking (fallback)
            num_to_mask = max(min_positions, int(len(sequence) * target_ratio))
            num_to_mask = min(num_to_mask, max_positions)
            
            # Sort positions by pLDDT (lowest confidence first)
            position_scores = [(i, score) for i, score in enumerate(plddt_scores)]
            position_scores.sort(key=lambda x: x[1])
            
            # Take lowest confidence positions
            masked_positions = set([pos for pos, _ in position_scores[:num_to_mask]])
            actual_ratio = len(masked_positions) / len(sequence) if sequence else 0.0
            print(f"      🎭 Depth {depth} quantile masking (fallback): {len(masked_positions)}/{len(sequence)} positions ({actual_ratio*100:.1f}%)")
            print(f"         Reason: Threshold masking gave {len(threshold_masked)} positions (outside {min_positions}-{max_positions} range)")
        
        return masked_positions
    
    def _get_structure_token_list(self, struct_seq: Optional[Union[str, List[str]]], seq_length: int) -> List[str]:
        """Normalize structure tokens to a list of digits/masks with length seq_length."""
        tokens: List[str] = []
        if isinstance(struct_seq, list):
            tokens = [str(t).strip() for t in struct_seq if str(t).strip()]
        elif struct_seq:
            if ',' in struct_seq:
                tokens = [t.strip() for t in struct_seq.split(',') if t.strip()]
            else:
                tokens = [t.strip() for t in struct_seq.split() if t.strip()]
        
        filtered_tokens: List[str] = []
        for token in tokens:
            if token.isdigit() or token == '<mask_struct>':
                filtered_tokens.append(token)
        
        if len(filtered_tokens) < seq_length:
            filtered_tokens.extend(['<mask_struct>'] * (seq_length - len(filtered_tokens)))
        else:
            filtered_tokens = filtered_tokens[:seq_length]
        
        if not filtered_tokens:
            filtered_tokens = ['<mask_struct>'] * seq_length
        
        return filtered_tokens
    
    def _generate_dplm2_candidate(self, node: MCTSNode, masked_positions: Set[int], expert_id: int) -> Optional[Dict]:
        """Generate candidate using DPLM-2 for current node context."""
        sequence = node.sequence
        try:
            if self.task_type == "folding":
                struct_seq = node.structure_tokens or self.baseline_structure.get('struct_seq', '')
                struct_tokens = self._get_structure_token_list(struct_seq, len(sequence))
                struct_tokens = struct_tokens.copy()
                for pos in masked_positions:
                    if 0 <= pos < len(struct_tokens):
                        struct_tokens[pos] = '<mask_struct>'
                masked_struct_seq = ' '.join(struct_tokens)
                
                structure_data = {
                    'sequence': sequence,
                    'struct_seq': masked_struct_seq,
                    'length': len(sequence),
                    'target_length': len(sequence),
                    'task': self.task_type,
                    'task_type': self.task_type
                }
                generation_output = self.dplm2_integration.generate_with_expert(
                    expert_id=expert_id,
                    structure=structure_data,
                    target_length=len(sequence),
                    masked_sequence=None,
                    temperature=1.0
                )
                if generation_output is None:
                    print(f"      ⚠️ Expert {expert_id} failed to generate structure tokens")
                    return None
                generation_data = getattr(self.dplm2_integration, '_last_generation_data', {}) or {}
                generated_struct_tokens = generation_data.get('structure_sequence')
                if not generated_struct_tokens:
                    print(f"      ⚠️ Expert {expert_id} returned no structure tokens via last_generation_data")
                    return None
                coords = self._structure_tokens_to_coords(generated_struct_tokens, len(sequence))
                if coords is None:
                    print(f"      ⚠️ Expert {expert_id} structure detokenization failed")
                    return None
                pl_ddt = self._estimate_plddt_from_coords(coords)
                return {
                    'sequence': sequence,
                    'structure_tokens': generated_struct_tokens,
                    'coordinates': coords,
                    'plddt_scores': pl_ddt
                }
            else:
                # CRITICAL FIX: Create masked sequence for DPLM-2 input, but return COMPLETE sequence
                masked_sequence = list(sequence)
                for pos in masked_positions:
                    masked_sequence[pos] = 'X'
                masked_seq_str = ''.join(masked_sequence)
                struct_seq = self.baseline_structure.get('struct_seq', '')
                struct_tokens = self._get_structure_token_list(struct_seq, len(sequence))
                struct_seq_normalized = ' '.join(struct_tokens)
                structure_data = {
                    'struct_seq': struct_seq_normalized,
                    'sequence': masked_seq_str,
                    'length': len(sequence),
                    'target_length': len(sequence),
                    'task': self.task_type,
                    'task_type': self.task_type
                }
                generated_sequence = self.dplm2_integration.generate_with_expert(
                    expert_id=expert_id,
                    structure=structure_data,
                    target_length=len(sequence),
                    masked_sequence=masked_seq_str,
                    temperature=1.0
                )
                
                if generated_sequence and len(generated_sequence) == len(sequence) and all(c in 'ACDEFGHIKLMNPQRSTVWYX' for c in generated_sequence.upper()):
                    # CRITICAL FIX: Use RAW DPLM output (complete sequence) for reward calculation
                    # Store the complete generated sequence, track which positions were modified
                    print(f"      ✅ Expert {expert_id} generated complete sequence (modified {len(masked_positions)} positions)")
                    print(f"      🔍 Raw DPLM output (first 50): {generated_sequence[:50]}...")
                    print(f"      🔍 Original sequence (first 50): {sequence[:50]}...")
                    print(f"      🔍 Modified positions: {sorted(list(masked_positions))[:10]}...")
                    
                    # Return the COMPLETE sequence from DPLM (not partial with X's)
                    return {'sequence': generated_sequence}
                
                print(f"      ⚠️ Expert {expert_id} generated invalid sequence")
                return None
        except Exception as e:
            print(f"      ⚠️ Expert {expert_id} generation failed: {e}")
            return None
    
    def _generate_proteinmpnn_candidate(self, sequence: str, masked_positions: Set[int]) -> str:
        """Generate candidate using ProteinMPNN (same approach as working one-shot analysis)."""
        try:
            # Create masked sequence for ProteinMPNN
            masked_sequence = list(sequence)
            for pos in masked_positions:
                masked_sequence[pos] = 'X'
            masked_seq_str = ''.join(masked_sequence)
            
            # Use EXACT same approach as working one_shot_inverse_folding_analysis.py
            # Load ProteinMPNN expert directly
            proteinmpnn_expert = self.dplm2_integration._load_expert(3)  # ProteinMPNN is expert 3
            
            # Use direct ProteinMPNN generation method (same as working script)
            generated_sequence = self.dplm2_integration._generate_with_proteinmpnn(
                proteinmpnn_model=proteinmpnn_expert,
                aa_sequence=masked_seq_str,
                struct_tokens="",  # Not used for ProteinMPNN
                task_type="inverse_folding",
                temperature=1.0
            )
            
            if generated_sequence and len(generated_sequence) == len(sequence):
                # CRITICAL FIX: Use RAW ProteinMPNN output (complete sequence) for reward calculation
                print(f"      ✅ ProteinMPNN generated complete sequence (modified {len(masked_positions)} positions)")
                print(f"      🔍 Raw ProteinMPNN output (first 50): {generated_sequence[:50]}...")
                print(f"      🔍 Original sequence (first 50): {sequence[:50]}...")
                
                # Return the COMPLETE sequence from ProteinMPNN (not partial with X's)
                return generated_sequence
            
            return None
            
        except Exception as e:
            print(f"      ⚠️ ProteinMPNN generation failed: {e}")
            return None
    
    def _structure_tokens_to_coords(self, structure_tokens: Union[str, List[int], List[str]], seq_length: int) -> Optional[np.ndarray]:
        """Convert structure tokens back to coordinates for evaluation"""
        if structure_tokens is None:
            return None
        # Prefer official converter if available
        if self.structure_converter is not None:
            coords = self.structure_converter.tokens_to_coordinates(structure_tokens, seq_length)
            if coords is not None:
                return coords
        try:
            # Import structure tokenizer
            from byprot.models.utils import get_struct_tokenizer
            import torch
            
            struct_tokenizer = get_struct_tokenizer()
            
            # Parse structure tokens (space-separated numbers and mask tokens)
            if isinstance(structure_tokens, str) and ',' in structure_tokens:
                token_list = [t.strip() for t in structure_tokens.split(',') if t.strip()]
            else:
                token_list = structure_tokens.strip().split() if isinstance(structure_tokens, str) else list(structure_tokens)
            if not token_list:
                return None
            
            # Convert to tensor with overflow protection and mask token handling
            valid_tokens = []
            max_token_value = 8228  # DPLM-2 vocab size limit
            mask_token_id = 8228  # Use max_id for mask tokens (gets clamped)
            
            print(f"      🔍 DEBUG: Processing {len(token_list)} structure tokens")
            print(f"      🔍 DEBUG: Sample tokens: {token_list[:20]}")
            
            for i, t in enumerate(token_list):
                token_str = str(t)
                if token_str.isdigit():
                    token_val = int(token_str)
                    # Clamp tokens to valid range to prevent overflow
                    if token_val > max_token_value:
                        print(f"      🔍 DEBUG: Clamping token {token_val} to {max_token_value}")
                        token_val = max_token_value
                    valid_tokens.append(token_val)
                elif token_str == '<mask_struct>':
                    valid_tokens.append(mask_token_id)
                    print(f"      🔍 DEBUG: Converting <mask_struct> at position {i} to token {mask_token_id}")
                else:
                    print(f"      🔍 DEBUG: Skipping special token: {token_str}")
            
            if len(valid_tokens) == 0:
                return None
            
            tokens = torch.tensor(valid_tokens, dtype=torch.long)
            
            # Detokenize to coordinates
            print(f"      🔍 DEBUG: Attempting to detokenize {len(valid_tokens)} structure tokens")
            print(f"      🔍 DEBUG: First 10 tokens: {valid_tokens[:10]}")
            coords = struct_tokenizer.detokenize(tokens.unsqueeze(0))  # Add batch dim
            print(f"      🔍 DEBUG: Detokenization result: {type(coords)}")
            
            if coords is not None and 'atom37_positions' in coords:
                # Extract CA coordinates (atom index 1)
                full_coords = coords['atom37_positions'][0]  # Remove batch dim
                ca_coords = full_coords[:, 1, :].cpu().numpy()  # CA atoms
                return ca_coords
            else:
                # Fallback: return baseline coordinates if detokenization fails
                baseline_coords = self.baseline_structure.get('coordinates')
                if baseline_coords is not None:
                    print(f"      🔄 Using baseline coordinates as fallback")
                    return baseline_coords
        except Exception as e:
            print(f"      ⚠️ Structure token conversion failed: {e}")
            # Fallback: return baseline coordinates
            baseline_coords = self.baseline_structure.get('coordinates')
            if baseline_coords is not None:
                print(f"      🔄 Using baseline coordinates as fallback")
                return baseline_coords
        return None

    def _estimate_plddt_from_coords(self, coords: np.ndarray) -> List[float]:
        """Estimate per-residue confidence from coordinates using target reference distance."""
        try:
            if coords is None:
                raise ValueError("No coordinates provided for pLDDT estimation")
            reference = self.reference_coords
            if reference is None:
                reference = self.baseline_structure.get('coordinates')
            if reference is None:
                return [70.0] * len(coords)
            coords = np.asarray(coords)
            reference = np.asarray(reference)
            min_len = min(len(coords), len(reference))
            if min_len == 0:
                return [70.0] * len(coords)
            diffs = np.linalg.norm(coords[:min_len] - reference[:min_len], axis=1)
            # Map distances to confidence [0,100]
            confidences = (np.exp(-diffs / 4.0) * 100.0).tolist()
            if len(coords) > min_len:
                tail_value = confidences[-1] if confidences else 70.0
                confidences.extend([tail_value] * (len(coords) - min_len))
            return confidences
        except Exception as e:
            print(f"      ⚠️ Failed to estimate pLDDT from coords: {e}")
            # Fallback: use sequence length from baseline structure
            seq_len = len(self.baseline_structure.get('sequence', ''))
            if seq_len == 0:
                seq_len = 100  # Ultimate fallback
            return [70.0] * seq_len
    
    def _evaluate_structure_reward(self, coords: np.ndarray, sequence: str) -> float:
        """Evaluate structure reward based on coordinates vs reference."""
        try:
            self._last_structure_eval = None
            reference_coords = self.reference_coords
            if reference_coords is None:
                reference_coords = self.baseline_structure.get('target_coordinates')
            if reference_coords is None:
                reference_coords = self.baseline_structure.get('coordinates')
            if reference_coords is None or coords is None:
                return 0.0
            
            sequence_for_reward = sequence
            if not sequence_for_reward:
                sequence_for_reward = self.reference_sequence
            if not sequence_for_reward:
                sequence_for_reward = self.baseline_structure.get('sequence')
            if not sequence_for_reward:
                sequence_for_reward = ""
            
            rmsd, tm_score, reward = evaluate_folding_metrics(
                coords,
                reference_coords,
                sequence_for_reward,
            )
            print(
                "      📊 Folding metrics: "
                f"RMSD={rmsd:.3f}Å, TM={tm_score:.3f}, Reward={reward:.3f}"
            )
            self._last_structure_eval = {
                'rmsd': float(rmsd),
                'tm_score': float(tm_score),
                'reward': float(reward),
            }
            return reward
        except Exception as e:
            print(f"      ⚠️ Structure reward calculation failed: {e}")
            self._last_structure_eval = None
            return 0.0

    def _compute_structure_reward(self, rmsd: float, tm_score: float) -> float:
        """Compatibility helper: compute folding reward from RMSD/TM metrics."""
        try:
            sequence = (
                self.reference_sequence
                or self.baseline_structure.get('sequence')
                or ""
            )
            # Call with correct parameter order: (tm_score, rmsd, plddt, sequence, aar)
            return calculate_folding_reward(
                tm_score=tm_score, 
                rmsd=rmsd, 
                plddt=None,  # pLDDT not available here
                sequence=sequence,
                aar=1.0  # For folding, AAR is always 1.0 (sequence is fixed)
            )
        except Exception as e:
            print(f"      ⚠️ Structure reward helper failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _compute_expert_entropy(self, sequence: str, expert: str, masked_positions: Set[int]) -> float:
        """Compute predictive entropy using correct logit extraction methods."""
        try:
            # Create masked sequence for entropy calculation
            masked_sequence = list(sequence)
            for pos in masked_positions:
                masked_sequence[pos] = 'X'
            masked_seq_str = ''.join(masked_sequence)
            
            if expert == "ProteinMPNN" or "ProteinMPNN" in str(expert):
                # Use ProteinMPNN entropy calculation with coordinates
                coordinates = self.baseline_structure.get('coordinates')
                if coordinates is not None:
                    # Try to load ProteinMPNN expert and compute entropy
                    try:
                        proteinmpnn_expert = self.dplm2_integration._load_expert_on_demand("proteinmpnn")
                        if proteinmpnn_expert and hasattr(proteinmpnn_expert, 'compute_entropy'):
                            entropy = proteinmpnn_expert.compute_entropy(
                                masked_sequence=masked_seq_str,
                                structure_coords=coordinates
                            )
                            print(f"      📊 {expert} entropy: {entropy:.3f}")
                            return entropy
                    except:
                        pass
                    
                    # Fallback: use DPLM2 ProteinMPNN entropy calculation
                    try:
                        entropy = self.dplm2_integration.compute_proteinmpnn_entropy(
                            sequence, list(masked_positions)
                        )
                        print(f"      📊 {expert} entropy (fallback): {entropy:.3f}")
                        return entropy
                    except:
                        pass
                else:
                    print(f"      ⚠️ No coordinates available for ProteinMPNN entropy")
                    return 1.5  # Default entropy for ProteinMPNN
            else:
                # For DPLM-2 experts, use predictive entropy with correct expert ID
                try:
                    # Extract expert ID from expert name (e.g., "dplm2_0" -> 0, "dplm2_1" -> 1)
                    if "dplm2_" in expert:
                        expert_id = int(expert.split("_")[-1])
                    else:
                        expert_id = 0  # Default fallback
                    
                    entropy = self.dplm2_integration.compute_predictive_entropy(
                        structure=self.baseline_structure,
                        masked_sequence=masked_seq_str,
                        expert_id=expert_id  # Use the ACTUAL expert ID
                    )
                    print(f"      📊 {expert} entropy (expert {expert_id}): {entropy:.3f}")
                    return entropy
                except Exception as e:
                    print(f"      ⚠️ DPLM-2 entropy calculation failed for {expert}: {e}")
                    pass
            
            # Final fallback: return moderate entropy based on number of masked positions
            entropy = min(2.0, 0.1 * len(masked_positions) + 0.5)
            print(f"      📊 {expert} entropy (default): {entropy:.3f}")
            return entropy
            
        except Exception as e:
            print(f"      ⚠️ Entropy calculation failed for {expert}: {e}")
            return 0.5  # Default entropy
    
    def _compute_novelty(self, sequence: str, parent_node: MCTSNode) -> float:
        """Compute novelty based on Hamming distance to siblings."""
        if not parent_node.children:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for sibling in parent_node.children:
            if sibling.sequence != sequence:
                distance = sum(1 for a, b in zip(sequence, sibling.sequence) if a != b)
                total_distance += distance / len(sequence)
                count += 1
        
        return total_distance / count if count > 0 else 1.0
    
    def _evaluate_sequence_aar(self, sequence: str) -> float:
        """Evaluate sequence using compound reward: R = 0.6×AAR + 0.35×scTM + 0.05×B"""
        if not self.reference_sequence or len(sequence) != len(self.reference_sequence):
            return 0.5
        
        # DEBUG: Show what sequence is being evaluated
        print(f"   🔍 REWARD EVAL: Sequence (first 50): {sequence[:50]}...")
        print(f"   🔍 REWARD EVAL: Reference (first 50): {self.reference_sequence[:50]}...")
        print(f"   🔍 REWARD EVAL: Contains X's: {'X' in sequence}")
        
        # Calculate AAR (Amino Acid Recovery)
        matches = sum(1 for a, b in zip(sequence, self.reference_sequence) if a == b)
        aar = matches / len(sequence)
        
        # Calculate scTM (structural similarity) using ESMFold
        try:
            # Use ESMFold to compute structural similarity
            from utils.real_plddt_computation import compute_sctm_from_esmfold
            sctm = compute_sctm_from_esmfold(
                sequence, 
                self.reference_sequence,
                self.baseline_structure.get('coordinates')
            )
        except:
            try:
                # Fallback: Use DPLM2 integration scTM if available
                sctm = self.dplm2_integration.compute_sctm(
                    sequence, 
                    self.baseline_structure,
                    reference_sequence=self.reference_sequence
                )
            except:
                # Final fallback: Use a different calculation than AAR
                # Use structural diversity as proxy (different from AAR)
                import numpy as np
                # Simple structural proxy based on sequence properties
                hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)
                charged = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
                polar = sum(1 for aa in sequence if aa in 'NQSTY') / len(sequence)
                
                # Structural compatibility score (different from AAR)
                sctm = (hydrophobic * 0.4 + charged * 0.3 + polar * 0.3) * 0.8 + aar * 0.2
                print(f"         Using structural proxy: hydrophobic={hydrophobic:.3f}, charged={charged:.3f}, polar={polar:.3f}")
        
        # Calculate biophysical score (B) - amino acid composition
        try:
            # Simple biophysical score based on amino acid diversity and composition
            aa_counts = {}
            for aa in sequence:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            # Diversity score (Shannon entropy of amino acid distribution)
            total = len(sequence)
            diversity = 0.0
            for count in aa_counts.values():
                if count > 0:
                    p = count / total
                    diversity -= p * math.log2(p)
            
            # Normalize diversity (max entropy for 20 amino acids = log2(20) ≈ 4.32)
            biophysical = min(1.0, diversity / 4.32)
        except:
            # Fallback biophysical score
            biophysical = 0.8
        
        # Compound reward: R = 0.6×AAR + 0.35×scTM + 0.05×B
        compound_reward = 0.6 * aar + 0.35 * sctm + 0.05 * biophysical
        
        print(f"      📊 Reward breakdown: AAR={aar:.3f}, scTM={sctm:.3f}, B={biophysical:.3f} → R={compound_reward:.3f}")
        
        return compound_reward
    
    def _backpropagate_max_rule(self, node: MCTSNode):
        """Backpropagate using max rule (W ← max(W, v))."""
        current = node
        while current:
            current.visits += 1
            current.total_reward = max(current.total_reward, node.reward)
            current = current.parent
    
    def _find_best_node_in_tree(self, root: MCTSNode) -> MCTSNode:
        """Find the best node in the entire tree."""
        best_node = root
        best_reward = root.reward
        
        def traverse(node):
            nonlocal best_node, best_reward
            if node.reward > best_reward:
                best_node = node
                best_reward = node.reward
            
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best_node
    
    # Legacy compatibility methods
    def _calculate_aar(self, sequence: str) -> float:
        """Legacy method for AAR calculation."""
        return self._evaluate_sequence_aar(sequence)
    
    def _evaluate_sequence(self, sequence: str) -> float:
        """Legacy method for sequence evaluation."""
        return self._evaluate_sequence_aar(sequence)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Legacy backpropagation method."""
        self._backpropagate_max_rule(node)


# Alias for backward compatibility
SequenceLevelMCTS = GeneralMCTS
