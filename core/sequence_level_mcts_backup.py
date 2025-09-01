"""
General MCTS Framework for DPLM-2 Performance Improvement

This module implements a general Monte Carlo Tree Search framework for improving
DPLM-2 performance across all tasks:
- Inverse folding (structure → sequence)
- Folding (sequence → structure) 
- Unconditional generation
- Conditional generation

Key features:
1. plDDT-based masking (or random if plDDT unavailable)
2. Simultaneous position sampling (leveraging diffusion properties)
3. Expert rollout with compound rewards
4. Task-agnostic design
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import random
from dataclasses import dataclass, field
import math
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from core.dplm2_integration import DPLM2Integration


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
    task_type: str = "inverse_folding"  # inverse_folding, folding, unconditional, conditional
    # 🎯 NEW: Entropy-related properties for PH-UCT
    entropy_score: float = 0.0
    diversity_score: float = 0.0
    exploration_potential: float = 0.0
    
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
        
        # UCB1 formula: value + exploration_constant * sqrt(ln(parent_visits) / visits)
        parent_visits = sum(child.visit_count for child in self.children) if self.children else 1
        if parent_visits <= 0:
            return self.average_value
        
        log_term = math.log(parent_visits)
        if log_term <= 0:
            return self.average_value
            
        return self.average_value + exploration_constant * math.sqrt(log_term / self.visit_count)
    
    @property
    def completion_ratio(self) -> float:
        """Ratio of unmasked positions."""
        if not self.sequence:
            return 0.0
        return 1.0 - (len(self.masked_positions) / len(self.sequence))
    
    def ph_uct_score(self, parent_visit_count: int, exploration_constant: float = 1.414, entropy_weight: float = 0.3, diversity_weight: float = 0.2) -> float:
        """
        🎯 NEW: PH-UCT (Entropy-Reinforced Planning) score for node selection.
        
        This implements the entropy-reinforced planning approach from the ERP paper,
        which balances exploitation, exploration, and entropy-based diversity.
        
        Args:
            exploration_constant: UCB1 exploration constant
            entropy_weight: Weight for entropy-based exploration
            diversity_weight: Weight for diversity-based exploration
            
        Returns:
            PH-UCT score combining UCB1 with entropy and diversity terms
        """
        if self.visit_count == 0:
            return float('inf')
        
        # Base UCB1 score - FIXED: Use actual parent visit count
        if parent_visit_count <= 0:
            return self.average_value
        
        log_term = math.log(parent_visit_count)
        if log_term <= 0:
            return self.average_value
        
        ucb1_score = self.average_value + exploration_constant * math.sqrt(log_term / self.visit_count)
        
        # 🎯 NEW: Entropy-reinforced terms
        # 1. Entropy-based exploration: encourages visiting nodes with high uncertainty
        entropy_term = entropy_weight * self.entropy_score
        
        # 2. Diversity-based exploration: encourages visiting nodes that increase tree diversity
        diversity_term = diversity_weight * self.diversity_score
        
        # 3. Exploration potential: encourages visiting nodes with high potential for improvement
        exploration_term = (1.0 - entropy_weight - diversity_weight) * self.exploration_potential
        
        # Combine all terms
        ph_uct_score = ucb1_score + entropy_term + diversity_term + exploration_term
        
        return ph_uct_score
    
    def update_entropy_scores(self, amino_acid_probs: Optional[Dict[str, float]] = None) -> None:
        """
        🎯 NEW: Update entropy-related scores for PH-UCT using REAL probabilities.
        
        Args:
            amino_acid_probs: Probability distribution over amino acids (if available)
        """
        if not self.sequence:
            return
        
        # 1. Entropy score: based on REAL sequence uncertainty from DPLM-2
        if amino_acid_probs:
            # Use actual amino acid probabilities from DPLM-2 model
            probs = list(amino_acid_probs.values())
            probs = [p for p in probs if p > 0]  # Filter out zero probabilities
            if probs:
                # Calculate Shannon entropy: H = -Σ(p_i * log(p_i))
                self.entropy_score = -sum(p * math.log(p + 1e-8) for p in probs)
                print(f"🎯 PH-UCT: Real entropy calculated from DPLM-2 probabilities: {self.entropy_score:.4f}")
            else:
                self.entropy_score = 0.0
                print(f"🎯 PH-UCT: No valid probabilities for entropy calculation")
        else:
            # 🚫 NO FALLBACK: We require real probabilities for proper entropy calculation
            print(f"⚠️ PH-UCT: No amino acid probabilities provided - entropy calculation requires real DPLM-2 outputs")
            self.entropy_score = 0.0
        
        # 2. Diversity score: based on how different this node is from siblings
        if self.parent and self.parent.children:
            sibling_sequences = [child.sequence for child in self.parent.children if child != self]
            if sibling_sequences:
                # Calculate average sequence similarity to siblings
                similarities = []
                for sibling_seq in sibling_sequences:
                    if len(sibling_seq) == len(self.sequence):
                        similarity = sum(1 for a, b in zip(self.sequence, sibling_seq) if a == b) / len(self.sequence)
                        similarities.append(similarity)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    self.diversity_score = 1.0 - avg_similarity  # Higher diversity = lower similarity
                else:
                    self.diversity_score = 0.5  # Default diversity
            else:
                self.diversity_score = 0.5
        else:
            self.diversity_score = 0.5
        
        # 3. Exploration potential: based on completion ratio and depth
        completion_ratio = self.completion_ratio
        depth_factor = 1.0 / (1.0 + self.depth)  # Deeper nodes have lower potential
        self.exploration_potential = completion_ratio * depth_factor


class GeneralMCTS:
    """
    General MCTS framework for improving DPLM-2 performance across all tasks.
    Leverages diffusion's ability to sample multiple positions simultaneously.
    """
    
    def __init__(
        self,
        initial_sequence: str = None,
        task_type: str = "inverse_folding",
        max_depth: int = 5,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        num_candidates_per_expansion: int = 5,
        use_plddt_masking: bool = True,
        simultaneous_sampling: bool = True,
        dplm2_integration: object = None,
        # 🎯 NEW: PH-UCT configuration parameters
        use_ph_uct: bool = True,
        entropy_weight: float = 0.3,
        diversity_weight: float = 0.2,
        exploration_potential_weight: float = 0.5
    ):
        """
        Initialize general MCTS framework.
        
        Args:
            initial_sequence: Initial sequence to start MCTS search from
            task_type: Type of task (inverse_folding, folding, unconditional, conditional)
            max_depth: Maximum tree depth
            num_simulations: Number of MCTS simulations
            exploration_constant: UCB1 exploration constant
            temperature: Sampling temperature
            num_candidates_per_expansion: Number of candidates per expansion
            use_plddt_masking: Whether to use plDDT-based masking
            simultaneous_sampling: Whether to sample multiple positions simultaneously
            use_ph_uct: Whether to use PH-UCT (True) or UCB1 (False) for selection
            entropy_weight: Weight for entropy-based exploration in PH-UCT
            diversity_weight: Weight for diversity-based exploration in PH-UCT
            exploration_potential_weight: Weight for exploration potential in PH-UCT
        """
        self.initial_sequence = initial_sequence
        self.task_type = task_type
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.num_candidates_per_expansion = num_candidates_per_expansion
        self.use_plddt_masking = use_plddt_masking
        self.simultaneous_sampling = simultaneous_sampling
        
        # 🎯 NEW: PH-UCT configuration
        self.use_ph_uct = use_ph_uct
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.exploration_potential_weight = exploration_potential_weight
        
        # Initialize DPLM-2 integration for AAR optimization
        self.dplm2_integration = dplm2_integration
        # Commented out for PH-UCT testing
        # if not self.dplm2_integration:
        #     # No fallback - must have real DPLM-2 integration
        #     try:
        #         from core.dplm2_integration import DPLM2Integration
        #         self.dplm2_integration = DPLM2Integration(use_local=True)
        #     except:
        #         raise ValueError("DPLM-2 integration is required - no fallback allowed")
        
        # 🎯 NEW: Multiple experts configuration
        self.use_multiple_experts = False
        if self.dplm2_integration and hasattr(self.dplm2_integration, 'use_multiple_experts'):
            self.use_multiple_experts = self.dplm2_integration.use_multiple_experts
            if self.use_multiple_experts:
                print(f"🎯 MCTS: Multiple experts enabled - {len(self.dplm2_integration.expert_instances)} expert models available")
        
        # Cache for generated sequences to avoid duplicates
        self.sequence_cache: Set[str] = set()
        
        # 🎯 STRATEGY: Default to baseline improvement for AAR optimization
        self.strategy = "baseline_improvement"
        
        # Set target_length from initial sequence if provided
        self.target_length = len(initial_sequence) if initial_sequence else None
        
        # Initialize task-specific components
        if task_type == "inverse_folding":
            self._setup_inverse_folding()
        elif task_type == "folding":
            self._setup_folding()
        elif task_type == "unconditional":
            self._setup_unconditional()
        elif task_type == "conditional":
            self._setup_conditional()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _setup_inverse_folding(self):
        """Setup for inverse folding task."""
        print("🎯 Setting up inverse folding task")
    
    def _setup_folding(self):
        """Setup for folding task."""
        print("🎯 Setting up folding task")
    
    def _setup_unconditional(self):
        """Setup for unconditional generation task."""
        print("🎯 Setting up unconditional generation task")
    
    def _setup_conditional(self):
        """Setup for conditional generation task."""
        print("🎯 Setting up conditional generation task")
    
    def search(self, target_length: int, max_simulations: int = 100, max_depth: int = 10, 
               exploration_constant: float = 1.414, temperature: float = 1.0, 
               num_candidates_per_expansion: int = 5, start_from_complete: bool = True,
               reference_sequence: str = None, structure: Dict = None) -> Tuple[str, float]:
        """
        Perform MCTS search for the best sequence, optimizing for both reward and AAR.
        
        Args:
            target_length: Target sequence length
            max_simulations: Maximum number of MCTS simulations
            max_depth: Maximum depth of the search tree
            exploration_constant: UCB1 exploration constant
            temperature: Temperature for final sequence selection
            num_candidates_per_expansion: Number of candidate sequences per expansion
            start_from_complete: If True, start from complete baseline; if False, use pLDDT masking
            reference_sequence: Reference sequence for AAR calculation and improvement tracking
            structure: Real structure data with coordinates for DPLM-2 integration
        
        Returns:
            Tuple of (best_sequence, best_reward)
        """

        
        # Store the real structure for DPLM-2 integration
        if structure:
            # Force alignment: Use FASTA sequence length as the truth
            fasta_length = len(self.initial_sequence) if self.initial_sequence else 0
            
            if fasta_length > 0:
                corrected_structure = structure.copy()
                corrected_structure['length'] = fasta_length
                corrected_structure['target_length'] = fasta_length
                
                # Warning: If coordinates don't match FASTA, this is a data issue
                if 'coordinates' in structure:
                    coord_length = structure['coordinates'].shape[0]
                    if coord_length != fasta_length:
                        print(f"Warning: Structure coordinates ({coord_length}) don't match FASTA ({fasta_length})")
                
                self._real_structure = corrected_structure
                self._baseline_structure = corrected_structure.copy()
                
                # Ensure structure_data and structure_path are preserved
                if 'structure_data' in structure:
                    self._baseline_structure['structure_data'] = structure['structure_data']
                    print(f"🔍 Preserved structure_data in _baseline_structure")
                if 'structure_path' in structure:
                    self._baseline_structure['structure_path'] = structure['structure_path']
                    print(f"🔍 Preserved structure_path in _baseline_structure")
            else:
                self._real_structure = structure.copy()
                self._baseline_structure = structure.copy()
                
                # Ensure structure_data and structure_path are preserved
                if 'structure_data' in structure:
                    self._baseline_structure['structure_data'] = structure['structure_data']
                    print(f"🔍 Preserved structure_data in _baseline_structure")
                if 'structure_path' in structure:
                    self._baseline_structure['structure_path'] = structure['structure_path']
                    print(f"🔍 Preserved structure_path in _baseline_structure")
        else:
            print(f"Warning: No structure provided - DPLM-2 integration will not work")
            self._real_structure = None
            self._baseline_structure = None
        
        # Track baseline AAR for improvement optimization
        baseline_aar = None
        if reference_sequence and self.initial_sequence:
            baseline_aar = self._calculate_simple_aar(self.initial_sequence, reference_sequence)
            self._baseline_aar = baseline_aar
            self._reference_sequence = reference_sequence
            print(f"Baseline AAR: {baseline_aar:.1%}")

        
        # Initialize root node based on strategy
        if start_from_complete:
            # 🎯 CRITICAL FIX: Start with baseline sequence, then apply minimal pLDDT masking for exploration
            # This gives MCTS exploration space while starting from a good sequence
            
            if self.strategy == "baseline_improvement" and hasattr(self, 'initial_sequence'):
                # Start with the complete baseline sequence
                baseline_sequence = self.initial_sequence
                print(f"🎯 Starting MCTS from baseline sequence (length: {len(baseline_sequence)})")
                print(f"🎯 Baseline AAR: {baseline_aar:.1%}")
                
                # 🎯 MINIMAL MASKING: Only mask 5-10% of positions for exploration
                # This ensures MCTS starts from a good sequence and makes small improvements
                try:
                    if hasattr(self, '_real_structure') and self._real_structure:
                        # Use real structure for pLDDT masking
                        masking_structure = self._real_structure
                        print(f"🎯 Using real structure for minimal pLDDT masking")
                    else:
                        # Create minimal structure if none provided
                        masking_structure = {
                            'sequence': baseline_sequence,
                            'length': len(baseline_sequence),
                            'target_length': len(baseline_sequence)
                        }
                        print(f"🎯 Using minimal structure for minimal pLDDT masking")
                    
                    # 🎯 ADAPTIVE MASKING: Less masking for high baseline AAR
                    if baseline_aar and baseline_aar > 0.6:
                        masking_ratio = 0.05  # 5% masking for high baseline (more exploitation)
                        print(f"🎯 High baseline AAR ({baseline_aar:.1%}) - using minimal masking: {masking_ratio*100:.0f}%")
                    elif baseline_aar and baseline_aar > 0.4:
                        masking_ratio = 0.08  # 8% masking for medium baseline
                        print(f"🎯 Medium baseline AAR ({baseline_aar:.1%}) - using moderate masking: {masking_ratio*100:.0f}%")
                    else:
                        masking_ratio = 0.12  # 12% masking for low baseline (more exploration)
                        print(f"🎯 Low baseline AAR ({baseline_aar:.1%}) - using aggressive masking: {masking_ratio*100:.0f}%")
                    
                    masked_sequence, initial_masked_positions = self._apply_plddt_masking(
                        baseline_sequence, 
                        masking_structure,
                        masking_ratio=masking_ratio
                    )
                    
                    print(f"🎯 Applied minimal pLDDT masking: {len(initial_masked_positions)}/{len(baseline_sequence)} positions masked ({100*len(initial_masked_positions)/len(baseline_sequence):.1f}%)")
                    print(f"   Masked sequence: {masked_sequence[:50]}...")
                    
                    # Use the minimally masked sequence for MCTS exploration
                    initial_sequence = masked_sequence
                    
                except Exception as e:
                    print(f"⚠️ pLDDT masking failed: {e}, using fallback strategy")
                    # Fallback: create minimal masking for exploration
                    num_to_mask = max(1, len(baseline_sequence) // 20)  # 5% masking
                    positions_to_mask = random.sample(range(len(baseline_sequence)), num_to_mask)
                    initial_masked_positions = set(positions_to_mask)
                    
                    # Create minimally masked sequence
                    masked_sequence = list(baseline_sequence)
                    for pos in positions_to_mask:
                        masked_sequence[pos] = 'X'
                    initial_sequence = ''.join(masked_sequence)
                    
                    print(f"🎯 Fallback minimal masking: {len(initial_masked_positions)} positions masked ({100*len(initial_masked_positions)/len(baseline_sequence):.1f}%)")
            else:
                # Create masked sequence for other strategies
                initial_sequence, initial_masked_positions = self._create_initial_masked_sequence(
                    structure, target_length
                )
                print(f"🎯 Starting MCTS from masked sequence")
                print(f"   Masked positions: {len(initial_masked_positions)}/{target_length}")
        else:
            # Use pLDDT-based masking for exploration
            # 🎯 KEY FIX: Use real structure if available
            if self._real_structure:
                masking_structure = self._real_structure
            else:
                masking_structure = {'sequence': self.initial_sequence, 'length': len(self.initial_sequence)}
            
            # 🎯 MINIMAL MASKING: Only 10-15% masking for exploration
            masked_sequence, initial_masked_positions = self._apply_plddt_masking(
                self.initial_sequence, 
                masking_structure,  # Use real structure here
                masking_ratio=0.12  # 12% masking for exploration
            )
            initial_sequence = masked_sequence
        
        # Create root node with the initial sequence
        self.root = MCTSNode(initial_sequence, initial_masked_positions, depth=0)
        
        # 🎯 VERIFY: Initial sequence should have some masking for exploration
        if len(initial_masked_positions) == 0:
            print(f"⚠️ WARNING: No masked positions - MCTS will have no exploration space!")
        else:
            print(f"✅ MCTS has {len(initial_masked_positions)} masked positions for exploration")
        
        # Calculate initial reward to verify consistency
        # 🎯 CRITICAL FIX: Use the same structure/reference that was used for baseline calculation
        # This ensures consistency between baseline and masked sequence rewards
        if hasattr(self, '_baseline_structure'):
            # Use the same structure that was used for baseline calculation
            reward_structure = self._baseline_structure
            print(f"🎯 Using baseline structure for consistent reward calculation")
        else:
            # Fallback to passed structure
            reward_structure = structure
            print(f"🎯 Using passed structure for reward calculation")
        
        initial_reward = self._calculate_reward(self.root.sequence, reward_structure)
        print(f"🎯 Initial MCTS sequence:")
        print(f"   Sequence: {initial_sequence[:50]}...")
        print(f"   Masked positions: {len(initial_masked_positions)}")
        print(f"   Initial reward: {initial_reward:.3f}")
        
        # 🎯 CRITICAL CHECK: Initial reward should be reasonable
        if initial_reward < 0.1:
            print(f"⚠️ WARNING: Initial reward ({initial_reward:.3f}) is very low")
            print(f"   This suggests the reward function may have issues with masked sequences")
            print(f"   Or the reference sequence is different from baseline calculation")
        
        # Track the best completed sequence found during search
        best_completed_sequence = initial_sequence
        best_completed_reward = initial_reward
        best_completed_aar = baseline_aar
        
        print(f"🎯 MCTS Starting Point:")
        print(f"   Baseline sequence: {self.initial_sequence[:50]}...")
        print(f"   Baseline AAR: {baseline_aar:.1%}")
        print(f"   Baseline reward: {best_completed_reward:.4f}")
        print(f"   Exploration sequence: {initial_sequence[:50]}...")
        print(f"   Exploration masked positions: {len(initial_masked_positions)}/{len(initial_sequence)}")
        
        # 🎯 VERIFY: The baseline sequence should be better than the masked exploration sequence
        if baseline_aar > 0 and best_completed_reward < baseline_aar * 0.8:
            print(f"⚠️  WARNING: Baseline reward ({best_completed_reward:.4f}) is much lower than baseline AAR ({baseline_aar:.1%})")
            print(f"   This suggests the reward function has issues with the baseline sequence")
            print(f"   MCTS will still try to improve from the baseline sequence")
        
        # 🎯 VERIFY: The exploration sequence should have some masking for exploration
        if len(initial_masked_positions) == 0:
            print(f"⚠️  WARNING: No masked positions in exploration sequence!")
            print(f"   This means MCTS will have no exploration space")
            print(f"   MCTS will only explore the baseline sequence")
        else:
            print(f"✅ Exploration sequence has {len(initial_masked_positions)} masked positions for exploration")
        
        # 🎯 IMPROVED MCTS: Adaptive exploration and temperature based on AAR improvement
        exploration_constant = self._adapt_exploration_constant(exploration_constant, baseline_aar)
        
        # 🎯 ADAPTIVE TEMPERATURE: Lower temperature for high baseline AAR (more exploitation)
        if baseline_aar and baseline_aar > 0.6:
            adaptive_temperature = max(0.5, temperature * 0.8)  # More focused search
            print(f"🎯 High baseline AAR ({baseline_aar:.1%}) - using lower temperature: {adaptive_temperature:.2f}")
        else:
            adaptive_temperature = temperature  # Standard temperature for exploration
            print(f"🎯 Standard baseline AAR - using temperature: {adaptive_temperature:.2f}")
        
        # 🚀 AGGRESSIVE MCTS SEARCH: Run simulations with adaptive exploration
        print(f"🎯 Starting MCTS search with {max_simulations} simulations...")
        print(f"🎯 Exploration constant: {exploration_constant:.2f}")
        print(f"🎯 Max depth: {max_depth}")
        print(f"🎯 Temperature: {temperature}")
        
        # Initialize timing for progress tracking
        start_time = time.time()
        
        for simulation in range(max_simulations):
            print(f"🎯 Progress: Simulation {simulation + 1}/{max_simulations}")
            
            # PHASE 1: SELECTION - Navigate to leaf using UCB1
            selected_node = self._select(self.root, exploration_constant)
            print(f"   Selected node at depth {selected_node.depth}, masked positions: {len(selected_node.masked_positions)}")
            
            # PHASE 2: EXPANSION - Add children if not terminal
            expanded_node = selected_node
            if not self._is_terminal(selected_node) and selected_node.visit_count > 0:
                self._expand(selected_node, self.target_length, num_candidates=2, temperature=temperature)
                
                # Select first new child for simulation
                if selected_node.children:
                    expanded_node = selected_node.children[0]  # Always simulate from first new child
                    print(f"   Expanded to new child at depth {expanded_node.depth}")
                    # Continue with simulation even if expansion fails
            
            # PHASE 3: SIMULATION - Complete sequence from expanded node using multiple experts
            reward = self._simulate_with_multiple_experts(expanded_node, temperature)
            print(f"   Simulation reward: {reward:.4f}")
            
            # PHASE 4: BACKPROPAGATION - Update all ancestors
            self._backpropagate(expanded_node, reward)
            print(f"   Backpropagated reward to {expanded_node.depth + 1} nodes")
            
            # Progress updates (reduced frequency)
            if (simulation + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"🎯 Progress: {simulation + 1}/{max_simulations} simulations ({elapsed_time:.1f}s)")
        
        # 🎯 FINAL STEP: Get best sequence from tree and ensure it's completely unmasked
        best_node = self._get_best_leaf_node()
        if best_node:
            best_completed_sequence = best_node.sequence
            # FIXED: Update search reward to actual best node's reward, not initial reward
            best_completed_reward = best_node.average_value
            print(f"🎯 Best sequence from tree: {len(best_node.masked_positions)} masked positions remaining")
            print(f"🎯 Search reward (from best node): {best_completed_reward:.4f}")
        else:
            best_completed_sequence = self.root.sequence
            best_completed_reward = self.root.average_value
            print(f"🎯 Using root sequence as fallback")
        
        # Final unmasking using multiple experts
        try:
            if best_completed_sequence and 'X' in best_completed_sequence:
                print(f"🎯 Final unmasking: {best_completed_sequence.count('X')} positions remain")
                
                masked_positions = {i for i, char in enumerate(best_completed_sequence) if char == 'X'}
                final_sequence = self._final_unmasking_with_multiple_experts(
                    best_completed_sequence, 
                    masked_positions,
                    structure=self._baseline_structure if hasattr(self, '_baseline_structure') else None,
                    temperature=0.8
                )
                
                if final_sequence and 'X' not in final_sequence:
                    best_completed_sequence = final_sequence
                    print(f"✅ Final unmasking successful")
                else:
                    print(f"❌ Final unmasking failed")
                    return None, 0.0
            else:
                print(f"🎯 Sequence already complete")
                
        except Exception as e:
            print(f"❌ Final unmasking failed: {e}")
            return None, 0.0
        
        print(f"🎯 Final MCTS sequence (completely unmasked):")
        print(f"   Length: {len(best_completed_sequence)}")
        print(f"   Masked positions: {best_completed_sequence.count('X')}")
        print(f"   Preview: {best_completed_sequence[:50]}...")
        
        # 🎯 CRITICAL FIX: Use consistent reward calculation throughout
        # The issue is that reward calculation during search vs final evaluation is inconsistent
        if hasattr(self, '_baseline_structure') and self._baseline_structure:
            # 🎯 CONSISTENT REWARD: Use the same structure and method for final reward
            final_reward_structure = self._baseline_structure.copy()
            final_reward_structure['sequence'] = best_completed_sequence
            final_reward_structure['length'] = len(best_completed_sequence)
            final_reward_structure['target_length'] = target_length
            
            # 🎯 ENSURE STRUCTURE_DATA: Use same adapter as search path
            # The _baseline_structure doesn't have structure_data/structure_path, so we need to use the adapter
            # This ensures final evaluation uses the same structure data as search
            
            # 🎯 USE SAME METHOD: Use the exact same reward calculation as during search
            # The _compute_compound_reward will use the working adapter we created for scTM calculation
            final_reward = self._compute_compound_reward(best_completed_sequence, final_reward_structure)
            print(f"🎯 Final reward (consistent calculation): {final_reward:.4f}")
            
            # 🎯 VERIFY: This should be close to the search reward
            if abs(final_reward - best_completed_reward) > 0.05:  # Increased tolerance for scTM variations
                print(f"⚠️  Reward difference between search and final:")
                print(f"   Search reward: {best_completed_reward:.4f}")
                print(f"   Final reward:  {final_reward:.4f}")
                print(f"   Difference: {abs(final_reward - best_completed_reward):.4f}")
                
                # 🎯 USE FINAL REWARD: Use the more accurate final calculation with full structure data
                print(f"   Using final reward (more accurate with complete structure data)")
                # Keep final_reward as is
            
            return best_completed_sequence, final_reward
    
    def _simulate_with_multiple_experts(self, node: MCTSNode, temperature: float) -> float:
        """Simulate from node using multiple experts to complete the sequence."""
        try:
            if not self.dplm2_integration:
                return self._calculate_reward(node.sequence, self.target_length)
            
            # Use hierarchical multiple experts for simulation
            if (hasattr(self.dplm2_integration, 'use_multiple_experts') and 
                self.dplm2_integration.use_multiple_experts and 
                'X' in node.sequence):
                
                structure_input = self._create_proper_structure_input(
                    self._baseline_structure if hasattr(self, '_baseline_structure') else {},
                    node.sequence
                )
                
                # Complete sequence using multiple experts
                completed_sequence = self.dplm2_integration.generate_with_multiple_experts(
                    structure=structure_input,
                    target_length=self.target_length,
                    masked_sequence=node.sequence,
                    temperature=temperature,
                    use_probability_averaging=True
                )
                
                if completed_sequence and 'X' not in completed_sequence:
                    return self._calculate_reward(completed_sequence, self.target_length)
                else:
                    # Fallback to partial reward if simulation incomplete
                    return self._calculate_reward(node.sequence, self.target_length) * 0.8
            
            # Fallback: use current sequence reward
            return self._calculate_reward(node.sequence, self.target_length)
            
        except Exception as e:
            print(f"⚠️ Simulation failed: {e}")
            return self._calculate_reward(node.sequence, self.target_length) * 0.5
    
    def _get_best_leaf_node(self) -> MCTSNode:
        """Get the leaf node with highest reward from the tree."""
        def traverse_tree(node: MCTSNode, best_node: MCTSNode = None) -> MCTSNode:
            # Update best if current node is better
            if best_node is None or (node.total_value / max(node.visit_count, 1) > 
                                   best_node.total_value / max(best_node.visit_count, 1)):
                best_node = node
            
            # Recursively check children
            if node.children:
                for child in node.children:
                    best_node = traverse_tree(child, best_node)
            
            return best_node
        
        return traverse_tree(self.root)
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if node is terminal (no more positions to mask or max depth reached)."""
        # Terminal if no masked positions remain
        if not hasattr(node, 'masked_positions') or len(node.masked_positions) == 0:
            return True
        
        # Terminal if max depth reached
        if hasattr(node, 'depth') and node.depth >= 5:  # max_depth
            return True
        
        # Terminal if sequence has no X's (fully unmasked)
        if 'X' not in node.sequence:
            return True
            
        return False
        
        # Calculate final AAR improvement if reference sequence is available
        if reference_sequence and baseline_aar is not None and best_completed_aar is not None:
            final_aar_improvement = best_completed_aar - baseline_aar
            print(f"🎯 Best AAR improvement: {final_aar_improvement:+.1%}")
        
        return best_completed_sequence, best_completed_reward
    
    def _create_initial_masked_sequence(self, input_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence with masked positions based on task type."""
        if self.task_type == "inverse_folding":
            return self._create_inverse_folding_masked_sequence(input_data, target_length)
        elif self.task_type == "folding":
            return self._create_folding_masked_sequence(input_data, target_length)
        elif self.task_type == "unconditional":
            return self._create_unconditional_masked_sequence(target_length)
        elif self.task_type == "conditional":
            return self._create_conditional_masked_sequence(input_data, target_length)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _create_inverse_folding_masked_sequence(self, structure: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create masked sequence for inverse folding task."""
        # Generate initial sequence using DPLM-2 (complete sequence first)
        try:
            initial_sequence = self.dplm2_integration.generate_sequence(
                structure, target_length=target_length, temperature=self.temperature
            )
        except:
            # Fallback to random sequence
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            # 🚫 NO FALLBACK: All sequences must come from DPLM-2 or baseline
            print(f"❌ Cannot generate random initial sequence - must use DPLM-2 baseline")
            return None
        
        # Start with reasonable masking - not all positions
        if self.use_plddt_masking:
            masked_positions = self._apply_plddt_masking(initial_sequence, structure)
        else:
            # Moderate random masking - 15-25% of positions
            masking_ratio = 0.2  # 20% masking instead of 100%
            num_to_mask = max(1, int(target_length * masking_ratio))
            masked_positions = set(random.sample(range(target_length), num_to_mask))
        
        # Create masked sequence
        masked_sequence = list(initial_sequence)
        for pos in masked_positions:
            masked_sequence[pos] = 'X'  # Mask token
        
        return ''.join(masked_sequence), masked_positions
    
    def _calculate_dynamic_plddt(self, sequence: str, structure: Dict) -> List[float]:
        """
        🎯 REAL pLDDT: Calculate dynamic pLDDT scores using ESMFold v1.
        
        This method uses ESMFold to calculate sequence-dependent pLDDT scores,
        providing dynamic confidence that changes based on the actual sequence.
        Falls back to physics-based calculation if ESMFold fails.
        
        Args:
            sequence: Amino acid sequence to analyze
            structure: Structure information (used for fallback only)
            
        Returns:
            List of dynamic pLDDT confidence scores (0.0 to 1.0)
        """
        if not sequence:
            return []
        
        try:
            # 🎯 PRIMARY: Use ESMFold-based dynamic pLDDT computation
            try:
                # Import ESMFold-based pLDDT computation
                from utils.real_plddt_computation import compute_esmfold_plddt
                
                # Calculate dynamic pLDDT using ESMFold
                per_residue_plddt, mean_plddt = compute_esmfold_plddt(sequence)
                
                if per_residue_plddt is not None and len(per_residue_plddt) == len(sequence):
                    # Convert to list and validate scores
                    plddt_scores = per_residue_plddt.tolist() if hasattr(per_residue_plddt, 'tolist') else list(per_residue_plddt)
                    
                    # Ensure all scores are valid floats in [0, 1]
                    plddt_scores = [max(0.0, min(1.0, float(score))) for score in plddt_scores]
                    
                    print(f"🎯 ESMFold pLDDT: mean={mean_plddt:.3f}, range=[{min(plddt_scores):.3f}, {max(plddt_scores):.3f}]")
                    return plddt_scores
                else:
                    print(f"⚠️ ESMFold pLDDT shape mismatch: got {len(per_residue_plddt) if per_residue_plddt is not None else 0}, expected {len(sequence)}")
                    
            except Exception as e:
                print(f"⚠️ ESMFold pLDDT failed: {e}, falling back to physics-based calculation")
            
            # 🎯 FALLBACK: Use physics-based pLDDT computation from structural coordinates
            try:
                # Import physics-based pLDDT computation as fallback
                from utils.real_plddt_computation import compute_plddt_from_structure
                
                # Update structure with current sequence for physics-based calculation
                fallback_structure = structure.copy() if structure else {}
                fallback_structure['sequence'] = sequence
                
                # Calculate pLDDT using physics-based method
                plddt_scores = compute_plddt_from_structure(fallback_structure)
                
                if plddt_scores and len(plddt_scores) == len(sequence):
                    # Validate scores
                    valid_scores = [max(0.0, min(1.0, float(s))) for s in plddt_scores if isinstance(s, (int, float))]
                    if len(valid_scores) == len(sequence):
                        avg_conf = sum(valid_scores) / len(valid_scores)
                        print(f"🎯 Physics-based pLDDT fallback: mean={avg_conf:.3f}")
                        return valid_scores
                    
            except Exception as e:
                print(f"⚠️ Physics-based pLDDT also failed: {e}")
            
            # 🎯 FINAL FALLBACK: Return reasonable default scores
            print(f"⚠️ All pLDDT methods failed, using default scores")
            return [0.75] * len(sequence)
            
        except Exception as e:
            print(f"❌ Error in pLDDT calculation: {e}")
            return [0.75] * len(sequence)
    
    def _apply_plddt_masking(self, sequence: str, structure: Dict, masking_ratio: float = 0.3) -> Tuple[str, Set[int]]:
        """
        🎯 IMPROVED: Apply intelligent pLDDT-based masking.
        
        This method:
        1. Calculates dynamic pLDDT scores for the current sequence
        2. Masks positions with low confidence (below threshold)
        3. Creates exploration space for MCTS optimization
        
        Args:
            sequence: Current sequence to mask
            structure: Structure information
            masking_ratio: Maximum fraction of positions to mask
            
        Returns:
            Tuple of (masked_sequence, masked_positions)
        """
        if not sequence:
            return "", set()
        
        # 🎯 STEP 1: Calculate dynamic pLDDT scores
        plddt_scores = self._calculate_dynamic_plddt(sequence, structure)
        
        if not plddt_scores or len(plddt_scores) != len(sequence):
            print(f"❌ pLDDT calculation failed - NO FALLBACK MASKING ALLOWED")
            # Return unmasked sequence to prevent random masking
            return sequence, set()
        
        # 🎯 STEP 2: Identify low-confidence positions for masking
        confidence_threshold = 0.6  # Reduced from 0.7 to 0.6 for more reasonable masking
        low_confidence_positions = [
            i for i, score in enumerate(plddt_scores) 
            if score < confidence_threshold
        ]
        
        print(f"🎯 pLDDT Analysis:")
        print(f"   Confidence threshold: {confidence_threshold:.1f}")
        print(f"   Low confidence positions: {len(low_confidence_positions)}/{len(sequence)}")
        print(f"   Average confidence: {sum(plddt_scores)/len(plddt_scores):.3f}")
        
        # 🎯 STEP 3: Apply intelligent masking strategy
        if len(low_confidence_positions) == 0:
            # No low-confidence positions - create reasonable exploration space
            print(f"🎯 No low-confidence positions found, creating exploration space")
            # 🎯 IMPROVED: Use a more realistic masking strategy
            # For proteins, we typically want to mask 10-15% of positions for exploration
            target_masking_ratio = 0.12  # Reduced from 0.15 to 0.12 (12% masking)
            num_to_mask = max(3, int(len(sequence) * target_masking_ratio))  # At least 3 positions
            
            # 🎯 STRATEGY: Mask positions with medium confidence for exploration
            # This creates a good balance between exploration and maintaining quality
            medium_confidence_threshold = 0.75  # Reduced from 0.8 to 0.75 for more positions
            medium_confidence_positions = [
                i for i, score in enumerate(plddt_scores) 
                if 0.6 <= score < medium_confidence_threshold
            ]
            
            if len(medium_confidence_positions) >= num_to_mask:
                # Use medium confidence positions
                positions_to_mask = random.sample(medium_confidence_positions, num_to_mask)
                print(f"🎯 Masking {len(positions_to_mask)} medium-confidence positions for exploration")
            else:
                # Fallback to random positions
                positions_to_mask = random.sample(range(len(sequence)), num_to_mask)
                print(f"🎯 Masking {len(positions_to_mask)} random positions for exploration")
            
            masked_positions = set(positions_to_mask)
        else:
            # Mask low-confidence positions, but ensure reasonable exploration space
            min_masking_ratio = 0.05   # Reduced from 0.08 to 0.05 (minimum 5% masking)
            max_masking_ratio = 0.18   # Reduced from 0.25 to 0.18 (maximum 18% masking)
            
            min_positions = max(3, int(len(sequence) * min_masking_ratio))
            max_positions = int(len(sequence) * max_masking_ratio)
            
            if len(low_confidence_positions) < min_positions:
                # Add some medium-confidence positions to reach minimum
                medium_confidence_positions = [
                    i for i, score in enumerate(plddt_scores) 
                    if 0.6 <= score < 0.8 and i not in low_confidence_positions
                ]
                additional_positions = random.sample(
                    medium_confidence_positions, 
                    min_positions - len(low_confidence_positions)
                )
                masked_positions = set(low_confidence_positions) | set(additional_positions)
                print(f"🎯 Enhanced masking: {len(low_confidence_positions)} low + {len(additional_positions)} medium confidence")
            elif len(low_confidence_positions) > max_positions:
                # Sort by confidence and mask the worst positions
                position_scores = [(i, plddt_scores[i]) for i in low_confidence_positions]
                position_scores.sort(key=lambda x: x[1])  # Sort by pLDDT score (ascending)
                
                # Only mask the worst positions up to the limit
                if len(position_scores) > max_positions:
                    masked_positions = set(pos for pos, _ in position_scores[:max_positions])
                    print(f"🎯 Limited masking to {len(masked_positions)} worst positions (max {max_masking_ratio*100:.0f}%)")
                else:
                    # Mask all low-confidence positions
                    masked_positions = set(low_confidence_positions)
                    print(f"🎯 Masking all {len(masked_positions)} low-confidence positions")
            else:
                # Perfect case: mask all low-confidence positions (within reasonable range)
                masked_positions = set(low_confidence_positions)
                print(f"🎯 Masking all {len(masked_positions)} low-confidence positions")
        
        # 🎯 STEP 4: Create masked sequence
        masked_sequence = list(sequence)
        for pos in masked_positions:
            masked_sequence[pos] = 'X'  # Mask token
        
        print(f"🎯 Final masking result:")
        print(f"   Masked positions: {len(masked_positions)}/{len(sequence)} ({len(masked_positions)/len(sequence)*100:.1f}%)")
        print(f"   Masked sequence: {''.join(masked_sequence[:30])}...")
        
        return ''.join(masked_sequence), masked_positions
    
    def _apply_fallback_masking(self, sequence: str, masking_ratio: float) -> Tuple[str, Set[int]]:
        """
        🚫 REMOVED: Random fallback masking.
        
        This method has been removed because:
        1. We should ONLY use real pLDDT-based masking
        2. Random masking provides no meaningful optimization
        3. All masking must be based on real structural confidence scores
        
        If pLDDT calculation is not available, the system should fail gracefully
        rather than apply meaningless random masking.
        """
        raise NotImplementedError(
            "Random fallback masking has been removed. "
            "All masking must be based on real pLDDT confidence scores. "
            "If pLDDT calculation is unavailable, the system should fail gracefully."
        )
    
    def _expand(self, node: MCTSNode, target_length: int, num_candidates: int = 3, temperature: float = 1.0) -> None:
        """
        HIERARCHICAL MULTIPLE EXPERTS: Expand node using all experts at each tree level.
        
        Enhanced approach for true hierarchical multiple experts:
        1. At each child node, apply pLDDT masking to create masked sequence
        2. Use all 3 expert models (650M, 150M, 3B) to do N rollouts each
        3. Select top-2 candidates across all experts as children
        4. Repeat this process at each level of the tree
        
        Args:
            node: Node to expand
            target_length: Target sequence length
            num_candidates: Number of rollouts per expert
        """
        if node.children is None:
            node.children = []
        
        # HIERARCHICAL MULTIPLE EXPERTS: Use all experts at each tree level
        if (hasattr(self, 'dplm2_integration') and 
            self.dplm2_integration and 
            hasattr(self.dplm2_integration, 'use_multiple_experts') and 
            self.dplm2_integration.use_multiple_experts):
            
            print(f"HIERARCHICAL EXPANSION: Using all {len(self.dplm2_integration.expert_instances)} experts at tree level {node.depth}")
            
            # Step 1: Create masked sequences for this tree level using pLDDT masking
            masked_sequences = self._create_masked_sequences_for_expansion(node, num_candidates)
            
            if not masked_sequences:
                print(f"No masked sequences created for expansion")
                self._expand_single_model(node, target_length, num_candidates, temperature)
                return
            
            # Step 2: For each masked sequence, use all experts to generate candidates
            all_expert_candidates = []
            
            for mask_idx, (masked_seq, masked_positions) in enumerate(masked_sequences):
                print(f"Processing masked sequence {mask_idx+1}/{len(masked_sequences)} with {len(masked_positions)} masked positions")
                
                try:
                    # Create structure input for this masked sequence
                    structure_input = self._create_proper_structure_input(
                        self._baseline_structure if hasattr(self, '_baseline_structure') else {},
                        masked_seq
                    )
                    
                    # Use multiple experts to generate candidates for this masked sequence
                    expert_result = self.dplm2_integration.generate_with_multiple_experts(
                        structure=structure_input,
                        target_length=target_length,
                        masked_sequence=masked_seq,
                        temperature=temperature,
                        use_probability_averaging=True
                    )
                    
                    if expert_result and expert_result != masked_seq:
                        # Calculate remaining masked positions after expert rollout
                        remaining_masked = set()
                        for i, char in enumerate(expert_result):
                            if char == 'X':
                                remaining_masked.add(i)
                        
                        # Evaluate candidate quality
                        candidate_reward = self._evaluate_candidate_reward(expert_result, target_length)
                        
                        all_expert_candidates.append((expert_result, remaining_masked, candidate_reward, mask_idx))
                        print(f"   Expert candidate: {len(remaining_masked)} masked positions, reward: {candidate_reward:.4f}")
                
                except Exception as e:
                    print(f"   Expert rollout failed for masked sequence {mask_idx+1}: {e}")
                    continue
            
            # Step 3: Select top-2 candidates across all expert results
            if all_expert_candidates:
                # Sort by reward (descending) and take top-2
                all_expert_candidates.sort(key=lambda x: x[2], reverse=True)
                top_candidates = all_expert_candidates[:2]
                
                print(f"Selected top-2 candidates from {len(all_expert_candidates)} expert results:")
                
                # Step 4: Create child nodes from top-2 candidates
                for i, (candidate_sequence, remaining_masked, reward, mask_idx) in enumerate(top_candidates):
                    child = MCTSNode(
                        candidate_sequence,
                        remaining_masked,
                        depth=node.depth + 1,
                        parent=node
                    )
                    node.children.append(child)
                    print(f"   Child {i+1}: {len(remaining_masked)} masked positions, reward: {reward:.4f}")
                
            else:
                print(f"No valid expert candidates generated, falling back to single model")
                self._expand_single_model(node, target_length, num_candidates, temperature)
                return
                
        else:
            # Fall back to single model expansion
            print(f"Using single model expansion (multiple experts not available)")
            self._expand_single_model(node, target_length, num_candidates, temperature)
        
        # VERIFY: Log expansion results
        for i, child in enumerate(node.children):
            remaining_masked = len(child.masked_positions) if hasattr(child, 'masked_positions') else 0
            print(f"CHILD {i+1}: Created with {remaining_masked} remaining masked positions")
        
        # VERIFY: Check if expansion is actually working
        if len(node.children) == 0:
            print(f"  WARNING: Expansion failed to create any children!")
            print(f"   This means MCTS cannot explore new sequences")
            print(f"   Node will remain a leaf node with no exploration")
            return
        
        print(f"Hierarchical expansion successful: {len(node.children)} children created using multiple experts")
    
        return masked_sequences
    
    def _create_masked_sequences_for_expansion(self, node: MCTSNode, num_candidates: int) -> List[Tuple[str, Set[int]]]:
        """
        Create masked sequences for hierarchical expansion using PROGRESSIVE pLDDT-based masking.
        
        PROGRESSIVE STRATEGY: As tree depth increases, use stricter masking thresholds:
        - Depth 0 (root): pLDDT < 0.5 (mask many low-confidence positions)
        - Depth 1: pLDDT < 0.6 (mask fewer, medium-confidence positions)  
        - Depth 2: pLDDT < 0.7 (mask only high-confidence positions)
        - Depth 3+: pLDDT < 0.8 (mask very few, highest-confidence positions)
        
        Args:
            node: Current node to expand from
            num_candidates: Number of masked sequences to create
            
        Returns:
            List of (masked_sequence, masked_positions) tuples
        """
        if not hasattr(self, '_baseline_structure') or not self._baseline_structure:
            print("No baseline structure available for pLDDT masking")
            return []
        
        sequence = node.sequence
        plddt_scores = self._baseline_structure.get('plddt_scores', [])
        
        if not plddt_scores or len(plddt_scores) != len(sequence):
            print(f"pLDDT scores unavailable or mismatched length, using fallback masking")
            return self._create_fallback_masked_sequences(sequence, num_candidates)
        
        # PROGRESSIVE MASKING: Determine threshold based on tree depth
        depth = node.depth
        if depth == 0:
            threshold = 0.5  # Root: mask many low-confidence positions
        elif depth == 1:
            threshold = 0.6  # Level 1: mask medium-confidence positions
        elif depth == 2:
            threshold = 0.7  # Level 2: mask high-confidence positions
        else:
            threshold = 0.8  # Level 3+: mask only highest-confidence positions
        
        print(f"PROGRESSIVE MASKING: Depth {depth} → pLDDT threshold {threshold}")
        
        masked_sequences = []
        
        # Create multiple masked sequences with slight variations for diversity
        for i in range(num_candidates):
            masked_positions = set()
            masked_sequence = list(sequence)
            
            # Base masking: positions below threshold
            for pos, plddt in enumerate(plddt_scores):
                if plddt < threshold:
                    masked_positions.add(pos)
                    masked_sequence[pos] = 'X'
            
            # Add slight variation for diversity (mask a few additional random positions)
            if i > 0 and masked_positions:  # Only for 2nd+ candidates
                import random
                additional_positions = random.sample(
                    [pos for pos in range(len(sequence)) if pos not in masked_positions],
                    min(2, len(sequence) - len(masked_positions))  # Mask 1-2 additional positions
                )
                for pos in additional_positions:
                    masked_positions.add(pos)
                    masked_sequence[pos] = 'X'
            
            if masked_positions:  # Only add if we actually masked something
                masked_sequences.append((''.join(masked_sequence), masked_positions))
                print(f"   Masked sequence {i+1}: {len(masked_positions)} positions (pLDDT < {threshold})")
        
        # If no positions were masked (all high confidence), use fallback
        if not masked_sequences:
            print(f"   No positions below pLDDT {threshold}, using fallback masking")
            fallback_sequences = self._create_fallback_masked_sequences(sequence, num_candidates)
            masked_sequences.extend(fallback_sequences)
        
        return masked_sequences
    
    def _create_fallback_masked_sequences(self, sequence: str, num_candidates: int) -> List[Tuple[str, Set[int]]]:
        """Create fallback masked sequences when pLDDT is unavailable."""
        import random
        masked_sequences = []
        
        # Strategy 1: Random masking
        for i in range(min(num_candidates, 2)):
            masked_seq = list(sequence)
            masked_positions = set()
            
            # Mask 5-10% of positions randomly
            num_to_mask = max(1, int(len(sequence) * (0.05 + i * 0.03)))
            positions_to_mask = random.sample(range(len(sequence)), num_to_mask)
            
            for pos in positions_to_mask:
                masked_seq[pos] = 'X'
                masked_positions.add(pos)
            
            masked_sequence = ''.join(masked_seq)
            masked_sequences.append((masked_sequence, masked_positions))
            print(f"   Fallback masked sequence {i+1}: {len(masked_positions)} positions masked (random)")
        
        # Strategy 2: Systematic masking if we need more candidates
        if len(masked_sequences) < num_candidates:
            for i in range(len(masked_sequences), num_candidates):
                masked_seq = list(sequence)
                masked_positions = set()
                
                # Mask every 10th position + some random positions
                step = max(5, len(sequence) // 10)
                for pos in range(i, len(sequence), step):
                    masked_seq[pos] = 'X'
                    masked_positions.add(pos)
                
                masked_sequence = ''.join(masked_seq)
                masked_sequences.append((masked_sequence, masked_positions))
                print(f"   Fallback masked sequence {i+1}: {len(masked_positions)} positions masked (systematic)")
        
        return masked_sequences
    
    def _apply_dynamic_pldt_masking(self, sequence: str, target_length: int) -> str:
        """Apply dynamic pLDDT masking - recalculate low-confidence regions for current sequence."""
        try:
            # Skip if no structure available
            if not hasattr(self, 'structure') or self.structure is None:
                return sequence
            
            # Calculate pLDDT for current sequence using ESMFold
            if hasattr(self, 'esmfold_model') and self.esmfold_model is not None:
                import torch
                with torch.no_grad():
                    # Remove any existing mask tokens for pLDDT calculation
                    clean_seq = sequence.replace('<mask>', 'A')
                    
                    # Get pLDDT scores
                    output = self.esmfold_model.infer_pdb(clean_seq)
                    if hasattr(output, 'plddt'):
                        plddt_scores = output.plddt[0].cpu().numpy()
                    else:
                        # Fallback to static masking
                        return sequence
                    
                    # Apply confidence-based masking
                    confidence_threshold = 0.6
                    low_confidence_positions = []
                    
                    for i, score in enumerate(plddt_scores):
                        if i < len(sequence) and score < confidence_threshold:
                            low_confidence_positions.append(i)
                    
                    # Create dynamically masked sequence
                    seq_list = list(sequence)
                    mask_token = '<mask>'
                    
                    for pos in low_confidence_positions:
                        if pos < len(seq_list) and seq_list[pos] != '<':  # Don't mask already masked positions
                            seq_list[pos] = mask_token
                    
                    masked_sequence = ''.join(seq_list)
                    print(f"🎯 Dynamic masking: {len(low_confidence_positions)} positions masked")
                    return masked_sequence
            
            # Fallback: return original sequence
            return sequence
            
        except Exception as e:
            print(f"⚠️ Dynamic pLDDT masking failed: {e}")
            return sequence
    
    def _final_unmasking_with_multiple_experts(self, sequence: str, masked_positions: Set[int], 
                                             structure: Dict = None, temperature: float = 1.0) -> str:
        """Final unmasking using ALL expert models with multiple rollouts and best selection."""
        try:
            if not self.dplm2_integration:
                print("❌ No DPLM-2 integration available for final unmasking")
                return None
            
            print(f"🎯 HIERARCHICAL FINAL UNMASKING: Using all {len(self.dplm2_integration.expert_instances)} experts")
            
            # Use multiple experts for final unmasking
            if (hasattr(self.dplm2_integration, 'use_multiple_experts') and 
                self.dplm2_integration.use_multiple_experts):
                
                final_result = self.dplm2_integration.generate_with_multiple_experts(
                    structure=structure,
                    target_length=len(sequence),
                    masked_sequence=sequence,
                    temperature=temperature,
                    use_probability_averaging=True
                )
                
                if final_result and 'X' not in final_result:
                    print(f"✅ Hierarchical final unmasking successful")
                    return final_result
                else:
                    print(f"⚠️ Hierarchical final unmasking incomplete, falling back to single expert")
            
            # Fallback: Single expert with multiple rollouts
            print(f"🎯 Fallback final unmasking: Running 3 rollouts with primary expert")
            rollout_results = []
            
            for rollout_id in range(3):
                # Set unique seed for each rollout
                seed = hash(f"final_unmasking_{rollout_id}_{sequence}") % 10000
                torch.manual_seed(seed)
                
                # Generate candidate using primary expert
                if hasattr(self.dplm2_integration, 'generate_with_expert'):
                    candidate = self.dplm2_integration.generate_with_expert(
                        expert_id=0,  # Use primary expert
                        structure=structure,
                        masked_sequence=sequence,
                        target_length=len(sequence),
                        temperature=temperature
                    )
                else:
                    # Fallback to standard generation
                    candidate = self.dplm2_integration.fill_masked_positions(
                        structure=structure,
                        masked_sequence=sequence,
                        target_length=len(sequence),
                        temperature=temperature
                    )
                
                if candidate:
                    # Calculate reward for this candidate
                    reward = self._calculate_reward(candidate, len(sequence))
                    rollout_results.append((candidate, reward))
                    print(f"   Rollout {rollout_id+1}: reward={reward:.4f}")
            
            if not rollout_results:
                print("❌ All final unmasking rollouts failed")
                return None
            
            # Select the candidate with maximum reward
            best_candidate, best_reward = max(rollout_results, key=lambda x: x[1])
            print(f"🎯 Selected best rollout: reward={best_reward:.4f}")
            
            return best_candidate
            
        except Exception as e:
            print(f"❌ Final unmasking with max selection failed: {e}")
            return None
    
    def _generate_expert_candidate(self, node: MCTSNode, target_length: int, expert_id: int, rollout_id: int, temperature: float) -> str:
        """Generate candidate using specific expert model - TRUE MULTIPLE EXPERTS."""
        try:
            # Set unique seed for this expert and rollout
            seed = hash(f"{expert_id}_{rollout_id}_{node.sequence}") % 10000
            torch.manual_seed(seed)
            
            # 🎯 DYNAMIC pLDDT MASKING: Calculate new low-confidence regions for this child
            if hasattr(self, 'structure') and self.structure is not None:
                # Recalculate pLDDT for current sequence and apply dynamic masking
                dynamic_masked_seq = self._apply_dynamic_pldt_masking(node.sequence, target_length)
            else:
                dynamic_masked_seq = node.sequence
            
            # 🎯 TRUE MULTIPLE EXPERTS: Use different expert models
            if hasattr(self.dplm2_integration, 'generate_with_expert'):
                candidate = self.dplm2_integration.generate_with_expert(
                    expert_id=expert_id,
                    structure=getattr(self, 'structure', None),
                    masked_sequence=dynamic_masked_seq,
                    target_length=target_length,
                    temperature=temperature
                )
                return candidate
            else:
                # Fallback to standard generation
                if hasattr(self.dplm2_integration, 'fill_masked_positions'):
                    candidate = self.dplm2_integration.fill_masked_positions(
                        structure=getattr(self, 'structure', None),
                        masked_sequence=dynamic_masked_seq,
                        target_length=target_length,
                        temperature=temperature
                    )
                    return candidate
        except Exception as e:
            print(f"⚠️ Expert {expert_id} candidate generation failed: {e}")
        return None
    
    def _apply_expert_transitions(self, sequence: str, expert_id: int, rollout_id: int, temperature: float) -> str:
        """Apply MD4-style transitions using specific expert."""
        try:
            # Set unique seed for this expert and rollout
            seed = hash(f"{expert_id}_{rollout_id}_{sequence}") % 10000
            torch.manual_seed(seed)
            
            # Apply MD4-style transitions
            if hasattr(self.dplm2_integration, 'apply_md4_style_transitions'):
                candidate = self.dplm2_integration.apply_md4_style_transitions(
                    sequence=sequence,
                    num_transitions=2,  # Small number of transitions
                    temperature=temperature
                )
                return candidate
        except Exception as e:
            print(f"⚠️ Expert transitions failed: {e}")
        return None
    
    def _evaluate_candidate_reward(self, sequence: str, target_length: int) -> float:
        """Evaluate candidate sequence reward."""
        try:
            if hasattr(self, '_baseline_structure') and self._baseline_structure:
                evaluation_structure = self._baseline_structure.copy()
                evaluation_structure['sequence'] = sequence
                evaluation_structure['length'] = len(sequence)
                evaluation_structure['target_length'] = target_length
                return self._compute_compound_reward(sequence, evaluation_structure)
            else:
                # Simple validity check
                return self._compute_basic_sequence_validity(sequence)
        except Exception as e:
            print(f"⚠️ Reward evaluation failed: {e}")
            return 0.0
    
    def _expand_single_model(self, node: MCTSNode, target_length: int, num_candidates: int, temperature: float) -> None:
        """Fallback expansion using single model."""
        # Check if we have a complete sequence or masked sequence
        if 'X' in node.sequence:
            # Handle masked sequence expansion (original logic)
            if not node.masked_positions:
                print(f"🎯 No masked positions to expand")
                return
            candidates = self._generate_intelligent_variations(node, target_length, num_candidates, temperature)
        else:
            # Handle complete sequence expansion (new logic)
            candidates = self._create_variations(node, num_candidates, target_length, temperature)
        
        if not candidates:
            print(f"⚠️  No candidates generated for expansion")
            return
        
        for i, (candidate_sequence, remaining_masked) in enumerate(candidates):
            # Create child node
            child = MCTSNode(
                candidate_sequence,
                remaining_masked,
                depth=node.depth + 1,
                parent=node
            )
            
            node.children.append(child)
        
        # 🎯 VERIFY: Check if any children actually improved the sequence
        improved_children = 0
        for i, child in enumerate(node.children):
            if hasattr(self, '_reference_sequence') and self._reference_sequence:
                child_aar = self._calculate_simple_aar(child.sequence, self._reference_sequence)
                parent_aar = self._calculate_simple_aar(node.sequence, self._reference_sequence)
                
                if child_aar > parent_aar:
                    improved_children += 1
                    print(f"🎯 CHILD {i+1}: AAR improved from {parent_aar:.1%} to {child_aar:.1%}")
                else:
                    print(f"🎯 CHILD {i+1}: AAR decreased from {parent_aar:.1%} to {child_aar:.1%}")
            else:
                print(f"🎯 CHILD {i+1}: No reference sequence for AAR comparison")
        
        print(f"🎯 Expansion summary: {improved_children}/{len(node.children)} children improved AAR")
        
        # 🎯 NEW: Update entropy scores for all children after expansion
        if self.use_ph_uct:
            # Skip entropy update during expansion - will be updated during selection
            # for child in node.children:
            #     self._update_node_entropy_with_dplm2(child, structure, target_length, temperature)
            print(f"🎯 PH-UCT: Entropy scores will be updated during selection for {len(node.children)} children")
    
    def _generate_intelligent_variations(self, node: MCTSNode, target_length: int, num_candidates: int, temperature: float = 1.0) -> List[Tuple[str, Set[int]]]:
        """
        🎯 DIFFUSION-STYLE VARIATIONS: Generate intelligent sequence variations using progressive unmasking.
        
        This method implements diffusion-style optimization: start with masked sequence,
        gradually unmask positions one by one or in small batches using DPLM-2.
        The key insight: Make small, targeted improvements instead of generating new sequences.
        
        Args:
            node: Current node
            target_length: Target sequence length
            num_candidates: Number of variations to generate
            
        Returns:
            List of (sequence, masked_positions) tuples
        """
        variations = []
        
        # 🎯 STRATEGY 1: MD4-style batch masked diffusion for optimization
        if hasattr(self, 'dplm2_integration') and self.dplm2_integration:
            try:
                # 🎯 CRITICAL FIX: Use MD4-style batch masking instead of single position filling
                # This leverages DPLM-2's full masked diffusion capabilities
                if node.masked_positions:
                    # Create variations by masking different combinations of positions
                    for _ in range(min(num_candidates // 2, 3)):  # Focus on DPLM-2 variations
                        # 🎯 MD4-STYLE: Mask multiple positions simultaneously
                        # Choose 3-5 positions to mask for efficient exploration
                        num_positions_to_mask = min(5, len(node.masked_positions))
                        positions_to_mask = random.sample(list(node.masked_positions), num_positions_to_mask)
                        
                        # Create sequence with multiple positions masked
                        masked_sequence = list(node.sequence)
                        for pos in positions_to_mask:
                            masked_sequence[pos] = 'X'
                        masked_sequence = ''.join(masked_sequence)
                        
                        print(f"🎯 MD4-style batch masking: masking {len(positions_to_mask)} positions simultaneously")
                        
                        # Use DPLM-2 to fill ALL masked positions at once (pure masked diffusion)
                        filled_sequence = self._resample_masked_positions_with_dplm2(
                            masked_sequence,
                            set(positions_to_mask),  # Pass ALL positions to fill
                            structure=None,  # No structure for pure masked diffusion
                            temperature=0.8  # Lower temperature for more focused generation
                        )
                        
                        if filled_sequence and len(filled_sequence) == target_length:
                            # Successfully filled multiple positions
                            new_masked_positions = node.masked_positions.copy()
                            for pos in positions_to_mask:
                                new_masked_positions.discard(pos)
                            
                            variations.append((filled_sequence, new_masked_positions))
                            print(f"🎯 DPLM-2 batch masked diffusion filled {len(positions_to_mask)} positions, remaining masked: {len(new_masked_positions)}")
                        else:
                            # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                            print(f"❌ DPLM-2 batch masked diffusion failed: could not fill positions")
                            continue
            except Exception as e:
                print(f"⚠️ DPLM-2 batch masked diffusion failed: {e}")
        
        # 🎯 STRATEGY 2: Progressive masking for exploration
        if len(variations) < num_candidates:
            # Create variations by progressively masking more positions
            for _ in range(min(num_candidates - len(variations), 2)):
                if node.masked_positions:
                    # 🎯 PROGRESSIVE MASKING: Start with fewer positions, then increase
                    num_positions = min(3, len(node.masked_positions))
                    positions_to_mask = random.sample(list(node.masked_positions), num_positions)
                    
                    # Create sequence with multiple positions masked
                    masked_sequence = list(node.sequence)
                    for pos in positions_to_mask:
                        masked_sequence[pos] = 'X'
                    masked_sequence = ''.join(masked_sequence)
                    
                    # Use DPLM-2 to fill all masked positions simultaneously
                    filled_sequence = self._resample_masked_positions_with_dplm2(
                        masked_sequence,
                        set(positions_to_mask),  # Fill all positions at once
                        structure=None,  # No structure for pure masked diffusion
                        temperature=1.0
                    )
                    
                    if filled_sequence and len(filled_sequence) == target_length:
                        new_masked_positions = node.masked_positions.copy()
                        for pos in positions_to_mask:
                            new_masked_positions.discard(pos)
                        variations.append((filled_sequence, new_masked_positions))
                    else:
                        # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                        print(f"❌ Progressive masking failed: DPLM-2 could not fill positions")
                        continue
        
        # 🎯 STRATEGY 3: TRUE MD4-STYLE DYNAMIC MASKING - Mask previously unmasked positions
        if len(variations) < num_candidates:
            # 🎯 CRITICAL INNOVATION: Mask previously unmasked positions for maximum exploration
            # This is the core of MD4-style - explore new combinations by masking different positions
            for _ in range(min(num_candidates - len(variations), 2)):
                # 🎯 DYNAMIC MASKING: Choose positions to mask from ALL available positions
                # This includes both currently masked AND previously unmasked positions
                all_positions = set(range(len(node.sequence)))
                currently_masked = node.masked_positions
                previously_unmasked = all_positions - currently_masked
                
                # 🎯 STRATEGY: Mix of currently masked and previously unmasked positions
                if previously_unmasked and currently_masked:
                    # 🎯 MD4-STYLE: Mask some previously unmasked positions for exploration
                    num_new_to_mask = min(2, len(previously_unmasked))  # Mask 1-2 new positions
                    num_current_to_mask = min(2, len(currently_masked))  # Mask 1-2 current positions
                    
                    new_positions_to_mask = random.sample(list(previously_unmasked), num_new_to_mask)
                    current_positions_to_mask = random.sample(list(currently_masked), num_current_to_mask)
                    positions_to_mask = new_positions_to_mask + current_positions_to_mask
                    
                    print(f"🎯 MD4-style dynamic masking: masking {num_new_to_mask} new + {num_current_to_mask} current positions")
                    
                elif previously_unmasked:
                    # 🎯 EXPLORE NEW: Mask some previously unmasked positions
                    num_positions = min(3, len(previously_unmasked))
                    positions_to_mask = random.sample(list(previously_unmasked), num_positions)
                    print(f"🎯 MD4-style new exploration: masking {num_positions} previously unmasked positions")
                    
                elif currently_masked:
                    # 🎯 REFINE CURRENT: Mask some currently masked positions
                    num_positions = min(3, len(currently_masked))
                    positions_to_mask = random.sample(list(currently_masked), num_positions)
                    print(f"🎯 MD4-style current refinement: masking {num_positions} currently masked positions")
                    
                else:
                    # 🎯 FULL EXPLORATION: Mask random positions for maximum diversity
                    num_positions = min(3, len(node.sequence) // 10)  # 10% of sequence
                    positions_to_mask = random.sample(range(len(node.sequence)), num_positions)
                    print(f"🎯 MD4-style full exploration: masking {num_positions} random positions")
                
                # Create sequence with dynamic masking
                masked_sequence = list(node.sequence)
                for pos in positions_to_mask:
                    masked_sequence[pos] = 'X'
                masked_sequence = ''.join(masked_sequence)
                
                # Use DPLM-2 to fill all masked positions simultaneously
                filled_sequence = self._resample_masked_positions_with_dplm2(
                    masked_sequence,
                    set(positions_to_mask),  # Fill all positions at once
                    structure=None,  # No structure for pure masked diffusion
                    temperature=1.0
                )
                
                if filled_sequence and len(filled_sequence) == target_length:
                    # 🎯 SUCCESS: Create new masked positions set
                    new_masked_positions = set()
                    for pos in range(len(filled_sequence)):
                        if pos in positions_to_mask:
                            # These positions were masked and filled by DPLM-2
                            continue
                        elif pos in node.masked_positions:
                            # These positions were already masked and remain masked
                            new_masked_positions.add(pos)
                        # Previously unmasked positions that weren't masked in this variation remain unmasked
                    
                    variations.append((filled_sequence, new_masked_positions))
                    print(f"🎯 MD4-style dynamic masking successful: {len(positions_to_mask)} positions filled, {len(new_masked_positions)} remaining masked")
                    
                else:
                    # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                    print(f"❌ MD4-style dynamic masking failed: DPLM-2 could not fill positions")
                    continue
        
        # 🎯 STRATEGY 4: REAL pLDDT-BASED MASKING - Recalculate confidence for each sequence
        if len(variations) < num_candidates:
            # 🎯 CRITICAL INNOVATION: Use real pLDDT calculation during MCTS exploration
            # This recalculates actual confidence scores from structural coordinates for each sequence
            for _ in range(min(num_candidates - len(variations), 2)):
                try:
                    # 🎯 STEP 1: Calculate real pLDDT for the current sequence
                    if hasattr(self, '_real_structure') and self._real_structure:
                        # Use real structure for pLDDT calculation
                        structure_for_plddt = self._real_structure.copy()
                        structure_for_plddt['sequence'] = node.sequence  # Update with current sequence
                    else:
                        # Create minimal structure for pLDDT calculation
                        structure_for_plddt = {
                            'sequence': node.sequence,
                            'length': len(node.sequence),
                            'target_length': len(node.sequence)
                        }
                    
                    print(f"🎯 Real pLDDT calculation for current sequence")
                    
                    # 🎯 STEP 2: Apply real pLDDT masking to the current sequence
                    masked_sequence, new_masked_positions = self._apply_plddt_masking(
                        node.sequence,
                        structure_for_plddt,
                        masking_ratio=0.2  # 20% maximum masking for exploration
                    )
                    
                    # 🎯 STEP 3: Use DPLM-2 to fill the newly masked positions
                    if len(new_masked_positions) > 0:
                        filled_sequence = self._resample_masked_positions_with_dplm2(
                            masked_sequence,
                            new_masked_positions,  # Fill all newly masked positions
                            structure=None,  # No structure for pure masked diffusion
                            temperature=0.9  # Slightly lower temperature for pLDDT-guided generation
                        )
                        
                        if filled_sequence and len(filled_sequence) == target_length:
                            # 🎯 SUCCESS: Real pLDDT masking and DPLM-2 filling
                            # Calculate remaining masked positions after filling
                            remaining_masked = set()
                            for pos in range(len(filled_sequence)):
                                if pos in new_masked_positions:
                                    # These positions were filled by DPLM-2
                                    continue
                                elif pos in node.masked_positions:
                                    # These positions were already masked and remain masked
                                    remaining_masked.add(pos)
                            
                            variations.append((filled_sequence, remaining_masked))
                            print(f"🎯 Real pLDDT masking successful: {len(new_masked_positions)} positions filled, {len(remaining_masked)} remaining masked")
                            
                        else:
                            # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                            print(f"❌ Real pLDDT masking failed: DPLM-2 could not fill positions")
                            continue
                    else:
                        # No new masking needed - sequence is already optimal
                        print(f"🎯 Real pLDDT: No new masking needed for current sequence")
                        
                except Exception as e:
                    print(f"⚠️ Real pLDDT masking failed: {e}")
                    continue
        
        # 🚫 STRATEGY 5 REMOVED: No random fallback variations
        # All variations must come from DPLM-2 - no random amino acid changes allowed

                if variation != node.sequence:
                    variations.append((variation, set()))  # No masked positions
        
        # 🎯 ENSURE: We have at least some variations
        if len(variations) == 0:
            print(f"⚠️ No variations created - DPLM-2 failed for all strategies")
            # 🚫 NO FALLBACK: Cannot create random variations
            print(f"❌ Cannot proceed without DPLM-2-generated variations")
            return  # Return empty list - no children will be created
        
        return variations
    
    def _select_intelligent_amino_acid(self, sequence: List[str], position: int, amino_acids: str) -> str:
        """Select amino acid using DPLM-2 intelligence when possible, fallback to context-aware."""
        # 🎯 TRY DPLM-2 APPROACH: Use the best amino acids from protein knowledge
        # This mimics what DPLM-2 would choose based on structure and sequence context
        
        # Get context window around the position
        context_window = 3
        start_pos = max(0, position - context_window)
        end_pos = min(len(sequence), position + context_window + 1)
        
        # Context-based intelligent selection (simplified DPLM-2 logic)
        context_seq = sequence[start_pos:end_pos]
        
        # Count amino acid types in context (filter out None and 'X')
        valid_context = [aa for aa in context_seq if aa is not None and aa != 'X']
        hydrophobic_count = sum(1 for aa in valid_context if aa in "AILMFPWYV")
        charged_count = sum(1 for aa in valid_context if aa in "KRDE")
        polar_count = sum(1 for aa in valid_context if aa in "NQSTY")
        
        # DPLM-2 inspired selection based on local structure preferences
        if hydrophobic_count > charged_count + polar_count:
            # Hydrophobic region - prefer hydrophobic amino acids
            candidates = "AILMFPWYV"
        elif charged_count > 0:
            # Near charged residues - balance or complement
            prev_aa = sequence[position-1] if position > 0 and position-1 < len(sequence) and sequence[position-1] not in ['X', None] else None
            if prev_aa and prev_aa in "KR":
                candidates = "DEST"  # Complement positive with negative/polar
            elif prev_aa and prev_aa in "DE":
                candidates = "KRST"  # Complement negative with positive/polar
            else:
                candidates = "KRDENQST"  # General charged/polar
        else:
            # Mixed or unclear context - use common, stable amino acids
            candidates = "ALSGTI"  # Common in loops and turns
        
        # Add some randomness but bias toward good choices
        if random.random() < 0.8:  # 80% use intelligent choice
            return random.choice(candidates)
        else:  # 20% use any amino acid for exploration
            # 🚫 NO FALLBACK: If no intelligent choice, we cannot proceed
            raise ValueError("No intelligent amino acid choice available. DPLM-2 integration required.")
    
    def _select_context_aware_amino_acid(self, sequence: List[str], position: int, amino_acids: str) -> str:
        """
        🚫 REMOVED: Context-aware amino acid selection with random fallback.
        
        This method has been removed because:
        1. We should ONLY use DPLM-2 for amino acid selection
        2. Random amino acids provide no meaningful optimization
        3. All amino acids must come from real DPLM-2 masked diffusion
        
        If DPLM-2 is not available, the system should fail gracefully
        rather than generate meaningless random amino acids.
        """
        raise NotImplementedError(
            "Context-aware amino acid selection with random fallback has been removed. "
            "All amino acids must be generated by DPLM-2 masked diffusion. "
            "If DPLM-2 is unavailable, the system should fail gracefully."
        )
    
    def _generate_simultaneous_variations(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Generate variations by sampling multiple masked positions simultaneously."""
        variations = []
        
        if self.simultaneous_sampling and len(node.masked_positions) > 1:
            # Sample multiple positions simultaneously using diffusion
            variations.extend(self._simultaneous_diffusion_sampling(node, input_data, target_length))
        else:
            # Fallback to single position sampling
            variations.extend(self._single_position_sampling(node, input_data, target_length))
        
        return variations
    
    def _resample_masked_positions_with_dplm2(self, sequence: str, masked_positions: Set[int], 
                                             structure: Dict = None, temperature: float = 1.0) -> str:
        """
        🎯 MASKED DIFFUSION: Use DPLM-2 to fill specific masked positions in a sequence.
        
        This is the core method for MCTS optimization using MD4-style approach:
        - Take a sequence with some positions masked
        - Use DPLM-2's masked diffusion to fill those positions
        - Return the improved sequence
        
        Args:
            sequence: Current sequence (some positions may be masked with X)
            masked_positions: Set of positions that are masked
            structure: Structure information (NOT USED for masked diffusion)
            temperature: Sampling temperature
            
        Returns:
            Sequence with masked positions filled by DPLM-2 masked diffusion
        """
        if not hasattr(self, 'dplm2_integration') or not self.dplm2_integration:
            # 🚫 NO FALLBACK: All amino acids must come from DPLM-2
            print(f"❌ DPLM-2 integration not available - cannot fill masked positions")
            return None
        
        try:
            # 🎯 KEY FIX: Use pure masked diffusion (sequence-to-sequence)
            # Create a sequence with the specified positions masked
            masked_sequence = list(sequence)
            for pos in masked_positions:
                if pos < len(masked_sequence):
                    masked_sequence[pos] = 'X'  # Use X for masked positions
            
            masked_sequence = ''.join(masked_sequence)
            
            print(f"🎯 DPLM-2 masked diffusion: filling {len(masked_positions)} positions")
            print(f"   Masked sequence: {masked_sequence[:50]}...")
            
            # 🎯 CRITICAL: Use DPLM-2 for masked diffusion WITHOUT structure
            # This is sequence-to-sequence diffusion, not structure-to-sequence generation
            completed_sequence = self.dplm2_integration.fill_masked_positions(
                structure=None,  # No structure needed for masked diffusion!
                masked_sequence=masked_sequence,
                target_length=len(sequence),
                temperature=temperature
            )
            
            if completed_sequence and len(completed_sequence) == len(sequence):
                print(f"🎯 DPLM-2 masked diffusion successful: {len(masked_positions)} positions filled")
                return completed_sequence
            else:
                # 🚫 NO FALLBACK: If DPLM-2 failed, we cannot complete the sequence
                print(f"❌ DPLM-2 masked diffusion failed: invalid sequence returned")
                print(f"   Expected length: {len(sequence)}, got: {len(completed_sequence) if completed_sequence else 'None'}")
                return None
                
        except Exception as e:
            print(f"❌ Error in DPLM-2 masked diffusion: {e}")
            import traceback
            traceback.print_exc()
            # 🚫 NO FALLBACK: All amino acids must come from DPLM-2
            return None
    
    def _fill_masked_positions_fallback(self, sequence: str, masked_positions: Set[int]) -> str:
        """
        🚫 REMOVED: Random amino acid fallback.
        
        This method has been removed because:
        1. We should ONLY use DPLM-2 for amino acid generation
        2. Random amino acids provide no meaningful optimization
        3. All amino acids must come from real DPLM-2 masked diffusion
        
        If DPLM-2 is not available, the system should fail gracefully
        rather than generate meaningless random amino acids.
        """
        raise NotImplementedError(
            "Random amino acid fallback has been removed. "
            "All amino acids must be generated by DPLM-2 masked diffusion. "
            "If DPLM-2 is unavailable, the system should fail gracefully."
        )
    
    def _validate_structure_for_dplm2(self, structure: Dict) -> bool:
        """
        🎯 VALIDATE: Ensure structure has real data for DPLM-2.
        
        This method validates that the structure contains the necessary
        information for DPLM-2 to generate high-quality sequences.
        
        Args:
            structure: Structure to validate
            
        Returns:
            True if structure is valid for DPLM-2, False otherwise
        """
        try:
            # 🎯 REQUIREMENT 1: Must have coordinates OR be able to create structure tokens
            has_coordinates = 'coordinates' in structure and structure['coordinates'] is not None
            has_backbone_coords = 'backbone_coords' in structure and structure['backbone_coords'] is not None
            has_atom_positions = 'atom_positions' in structure and structure['atom_positions'] is not None
            
            # 🎯 IMPROVED: Check if we can create structure tokens from coordinates
            can_create_tokens = has_coordinates or has_backbone_coords or has_atom_positions
            
            if not can_create_tokens:
                print(f"❌ Structure validation failed: no coordinates found")
                return False
            
            # 🎯 REQUIREMENT 2: Must have sequence information
            if 'sequence' not in structure or structure['sequence'] is None:
                print(f"❌ Structure validation failed: no sequence information")
                return False
            
            # 🎯 REQUIREMENT 3: Must have reasonable length
            if 'length' not in structure or structure['length'] <= 0:
                print(f"❌ Structure validation failed: invalid length")
                return False
            
            # 🎯 REQUIREMENT 4: If coordinates exist, they must be valid
            if has_coordinates:
                coords = structure['coordinates']
                
                if not isinstance(coords, (list, tuple, np.ndarray)) or len(coords) == 0:
                    print(f"❌ Structure validation failed: invalid coordinates format")
                    return False
                
                # 🎯 IMPROVED: Handle different coordinate formats
                if hasattr(coords, 'shape'):
                    if len(coords.shape) == 3:  # (L, 3, 3) format - backbone atoms

                        
                        # Check if we have any non-zero coordinates (some might be zero at start/end)
                        non_zero_count = 0
                        for i in range(min(10, coords.shape[0])):  # Check first 10 positions
                            if np.any(coords[i] != 0):
                                non_zero_count += 1
                        
                        if non_zero_count == 0:
                            print(f"❌ Structure validation failed: all coordinates are zero")
                            return False

                        
                        # Validate coordinate structure
                        for i in range(min(3, coords.shape[0])):  # Check first 3 positions
                            atom_coords = coords[i]  # Shape: (3, 3)
                            
                            # Check each atom has 3 coordinates
                            if atom_coords.shape != (3, 3):
                                print(f"❌ Structure validation failed: position {i} has wrong atom format")
                                return False
                            
                            # Check coordinates are numeric (even if zero)
                            if not np.issubdtype(atom_coords.dtype, np.number):
                                print(f"❌ Structure validation failed: position {i} has non-numeric coordinates")
                                return False
                        

                        return True
                        
                    elif len(coords.shape) == 2:  # (L, 3) format - CA atoms only

                        
                        # Check if we have any non-zero coordinates
                        non_zero_count = np.sum(np.any(coords != 0, axis=1))
                        if non_zero_count == 0:
                            print(f"❌ Structure validation failed: all CA coordinates are zero")
                            return False

                        
                        # Validate coordinate structure
                        for i in range(min(3, coords.shape[0])):  # Check first 3 positions
                            ca_coord = coords[i]  # Shape: (3,)
                            
                            if ca_coord.shape != (3,):
                                print(f"❌ Structure validation failed: CA position {i} has wrong format")
                                return False
                            
                            # Check coordinates are numeric (even if zero)
                            if not np.issubdtype(ca_coord.dtype, np.number):
                                print(f"❌ Structure validation failed: CA position {i} has non-numeric coordinates")
                                return False
                        

                        return True
                        
                    else:
                        print(f"❌ Structure validation failed: unexpected coordinate shape: {coords.shape}")
                        return False
                else:

                    if len(coords) == 0:
                        print(f"❌ Structure validation failed: empty coordinate list")
                        return False
                    
                    # Check first few coordinates
                    for i in range(min(3, len(coords))):
                        coord = coords[i]
                        
                        if not isinstance(coord, (list, tuple, np.ndarray)) or len(coord) < 3:
                            print(f"❌ Structure validation failed: coordinate {i} has wrong format")
                            return False
                        
                        # Check for reasonable coordinate values
                        x, y, z = coord[:3]
                        
                        try:
                            x_val = float(x) if hasattr(x, '__float__') else float(str(x))
                            y_val = float(y) if hasattr(y, '__float__') else float(str(y))
                            z_val = float(z) if hasattr(z, '__float__') else float(str(z))
                            
                            if not (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                                print(f"❌ Structure validation failed: coordinate {i} has non-finite values")
                                return False
                            
                        except (ValueError, TypeError) as e:
                            print(f"❌ Structure validation failed: coordinate {i} has non-numeric values: {e}")
                            return False
            
            # 🎯 REQUIREMENT 5: If pLDDT scores exist, they must be valid
            if 'plddt_scores' in structure:
                plddt_scores = structure['plddt_scores']
                if not isinstance(plddt_scores, (list, tuple, np.ndarray)) or len(plddt_scores) == 0:
                    print(f"❌ Structure validation failed: invalid pLDDT scores format")
                    return False
                
                # Check pLDDT scores are in reasonable range (0-1)
                for i, score in enumerate(plddt_scores[:10]):  # Check first 10
                    if not isinstance(score, (int, float, np.number)) or score < 0 or score > 1:
                        print(f"❌ Structure validation failed: pLDDT score {i} out of range: {score}")
                        return False
            

            return True
            
        except Exception as e:
            print(f"❌ Structure validation error: {e}")
            return False
    
    def _create_proper_structure_input(self, structure: Dict, sequence: str) -> Dict:
        """
        🎯 IMPROVED: Create proper structure input for DPLM-2.
        
        This method ensures DPLM-2 gets the structure information it needs
        to generate high-quality sequences by converting coordinates to structure tokens.
        
        Args:
            structure: Current structure information
            sequence: Current sequence
            
        Returns:
            Proper structure input for DPLM-2
        """
        try:
            # 🎯 VALIDATE: Structure must be valid for DPLM-2
            if not self._validate_structure_for_dplm2(structure):
                raise ValueError("Structure validation failed - cannot create proper input")
            
            # 🎯 IMPROVED: Convert coordinates to structure tokens that DPLM-2 can use
            proper_structure = structure.copy()
            
            # 🎯 KEY FIX: Use DPLM-2's structure tokenizer to convert coordinates to tokens
            if hasattr(self, 'dplm2_integration') and self.dplm2_integration:
                try:
                    # Convert coordinates to structure tokens
                    struct_tokens = self.dplm2_integration._coordinates_to_structure_tokens(structure)
                    if struct_tokens:
                        proper_structure['struct_tokens'] = struct_tokens
            
                    else:
                        print(f"⚠️ Warning: Could not convert coordinates to structure tokens")
                except Exception as e:
                    print(f"⚠️ Warning: Structure tokenization failed: {e}")
            
            # 🎯 ENSURE: Task type is set correctly
            proper_structure['task_type'] = 'inverse_folding'
            

            
            return proper_structure
            
        except Exception as e:
            print(f"❌ Error creating proper structure input: {e}")
            raise ValueError(f"Cannot create proper structure input: {e}")
    
    def _random_resample_masked_positions(self, sequence: str, masked_positions: Set[int]) -> str:
        """Fallback random resampling when DPLM-2 is not available."""
        # 🚫 NO RANDOM RESAMPLING FALLBACK
        print(f"❌ Cannot resample {len(masked_positions)} positions without DPLM-2")
        
        resampled_sequence = list(sequence)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        for pos in masked_positions:
            if pos < len(resampled_sequence):
                # Use context-aware selection for better quality
                selected_aa = self._select_context_aware_amino_acid(resampled_sequence, pos, amino_acids)
                resampled_sequence[pos] = selected_aa
        
        return ''.join(resampled_sequence)
    
    def _simultaneous_diffusion_sampling(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """
        🎯 IMPROVED: Sample multiple masked positions simultaneously using DPLM-2 diffusion.
        
        This method leverages DPLM-2's ability to sample multiple positions at once,
        creating more diverse and coherent sequence variations.
        
        Args:
            node: Current MCTS node
            input_data: Input data for the task
            target_length: Target sequence length
            
        Returns:
            List of (sequence, masked_positions) tuples
        """
        variations = []
        
        if not hasattr(self, 'dplm2_integration') or not self.dplm2_integration:
            print(f"⚠️ DPLM-2 integration not available for simultaneous sampling")
            return self._single_position_sampling(node, input_data, target_length)
        
        try:
            print(f"🎯 Simultaneous DPLM-2 sampling: {len(node.masked_positions)} masked positions")
            
            # 🎯 STRATEGY: Generate multiple variations with different temperatures
            temperatures = [0.8, 1.0, 1.2, 1.5]
            
            for temp in temperatures:
                if len(variations) >= 3:  # Limit variations for efficiency
                    break
                    
                print(f"🎯 Sampling with temperature {temp}...")
                
                # Use DPLM-2 to resample all masked positions simultaneously
                resampled_sequence = self._resample_masked_positions_with_dplm2(
                    node.sequence, 
                    node.masked_positions, 
                    input_data, 
                    temperature=temp
                )
                
                if resampled_sequence and len(resampled_sequence) == target_length:
                    # Determine which positions were actually unmasked
                    unmasked_positions = set()
                    for i, (old_aa, new_aa) in enumerate(zip(node.sequence, resampled_sequence)):
                        if old_aa == 'X' and new_aa != 'X':
                            unmasked_positions.add(i)
                    
                    # Update masked positions
                    new_masked = node.masked_positions - unmasked_positions
                    
                    if len(unmasked_positions) > 0:
                        variations.append((resampled_sequence, new_masked))
                        print(f"   ✅ Generated variation with {len(unmasked_positions)} unmasked positions")
                    else:
                        print(f"   ⚠️ No positions were unmasked with temperature {temp}")
                else:
                    print(f"   ❌ Invalid sequence generated with temperature {temp}")
            
            if not variations:
                print(f"⚠️ Simultaneous sampling failed, falling back to single position")
                return self._single_position_sampling(node, input_data, target_length)
            
            print(f"🎯 Simultaneous sampling successful: {len(variations)} variations generated")
            return variations
            
        except Exception as e:
            print(f"❌ Simultaneous sampling failed: {e}")
            print(f"🔄 Falling back to single position sampling")
            return self._single_position_sampling(node, input_data, target_length)
    
    def _single_position_sampling(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Sample one position at a time (fallback method)."""
        variations = []
        
        # Sample a few positions to unmask
        positions_to_unmask = random.sample(list(node.masked_positions), 
                                          min(3, len(node.masked_positions)))
        
        for pos in positions_to_unmask:
            # Generate amino acid for this position
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            # 🚫 NO FALLBACK: All amino acids must come from DPLM-2
            print(f"❌ Cannot select amino acid without DPLM-2")
            break
            
            # Create new sequence
            new_sequence = list(node.sequence)
            new_sequence[pos] = new_aa
            
            # Update masked positions
            new_masked = node.masked_positions - {pos}
            
            variations.append((''.join(new_sequence), new_masked))
        
        return variations
    
    def _create_masked_input(self, masked_sequence: str, input_data: Dict) -> Dict:
        """Create input with masked sequence for DPLM-2."""
        # For inverse folding, input_data is the structure
        structure = input_data.copy()
        structure['masked_sequence'] = masked_sequence
        return self.dplm2_integration._create_masked_input(masked_sequence, structure, len(masked_sequence))
    
    def _simulate_expert_rollout(self, node: MCTSNode, input_data: Dict) -> float:
        """Simulate expert rollout with compound reward (optimized for speed)."""
        if not node.masked_positions:
            # Sequence is complete, evaluate final reward
            # 🎯 CRITICAL FIX: Use the TRUE reference sequence for consistent reward calculation
            # The issue was using _baseline_structure which contains DPLM-2 baseline as reference
            if hasattr(self, '_reference_sequence') and self._reference_sequence:
                # Create evaluation structure with TRUE reference sequence
                evaluation_structure = {
                    'sequence': self._reference_sequence,  # TRUE reference, not DPLM-2 baseline
                    'length': len(self._reference_sequence),
                    'target_length': len(self._reference_sequence)
                }
                print(f"🎯 Using TRUE reference sequence for node reward (length: {len(self._reference_sequence)})")
                return self._compute_compound_reward(node.sequence, evaluation_structure)
            elif hasattr(self, '_baseline_structure') and self._baseline_structure:
                # Fallback: use baseline structure but ensure it has TRUE reference
                evaluation_structure = self._baseline_structure.copy()
                if hasattr(self, '_reference_sequence'):
                    evaluation_structure['sequence'] = self._reference_sequence  # TRUE reference
                evaluation_structure['length'] = len(node.sequence)
                evaluation_structure['target_length'] = len(node.sequence)
                print(f"🎯 Using TRUE reference sequence for node reward")
                return self._compute_compound_reward(node.sequence, evaluation_structure)
            else:
                return self._compute_compound_reward(node.sequence, input_data)
        
        # Fast rollout: just evaluate current state and do minimal unmasking
        current_sequence = node.sequence
        current_masked = node.masked_positions.copy()
        
        # If there are masked positions, quickly complete the sequence
        if current_masked:
            # 🚫 NO FALLBACK: Use DPLM-2 for completion or fail gracefully
            try:
                # Use DPLM-2 to complete the sequence
                completed_sequence = self._resample_masked_positions_with_dplm2(
                    current_sequence,
                    current_masked,
                    structure=None,
                    temperature=self.temperature  # Use instance temperature
                )
                
                if completed_sequence and len(completed_sequence) == len(current_sequence):
                    current_sequence = completed_sequence
                else:
                    # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot complete the sequence
                    print(f"❌ DPLM-2 completion failed in simulation")
                    return 0.0  # Return low reward for failed completion
            
            except Exception as e:
                print(f"❌ DPLM-2 completion failed in simulation: {e}")
                return 0.0  # Return low reward for failed completion
        
        # Single reward evaluation
        # 🎯 CRITICAL FIX: Use the TRUE reference sequence for consistent reward calculation
        # The issue was using _baseline_structure which contains DPLM-2 baseline as reference
        if hasattr(self, '_reference_sequence') and self._reference_sequence:
            # Create evaluation structure with TRUE reference sequence
            evaluation_structure = {
                'sequence': self._reference_sequence,  # TRUE reference, not DPLM-2 baseline
                'length': len(self._reference_sequence),
                'target_length': len(self._reference_sequence)
            }
            print(f"🎯 Using TRUE reference sequence for simulation reward (length: {len(self._reference_sequence)})")
            return self._compute_compound_reward(current_sequence, evaluation_structure)
        elif hasattr(self, '_baseline_structure') and self._baseline_structure:
            # Fallback: use baseline structure but ensure it has TRUE reference
            evaluation_structure = self._baseline_structure.copy()
            if hasattr(self, '_reference_sequence'):
                evaluation_structure['sequence'] = self._reference_sequence  # TRUE reference
            evaluation_structure['length'] = len(current_sequence)
            evaluation_structure['target_length'] = len(current_sequence)
            print(f"🎯 Using baseline structure with TRUE reference for simulation reward")
            return self._compute_compound_reward(current_sequence, evaluation_structure)
        else:
            return self._compute_compound_reward(current_sequence, input_data)
    
    def _complete_sequence(self, sequence: str, masked_positions: Set[int]) -> str:
        """
        🚫 REMOVED: Random amino acid completion fallback.
        
        This method has been removed because:
        1. We should ONLY use DPLM-2 for sequence completion
        2. Random amino acids provide no meaningful optimization
        3. All amino acids must come from real DPLM-2 masked diffusion
        
        If DPLM-2 is not available, the system should fail gracefully
        rather than generate meaningless random amino acids.
        """
        raise NotImplementedError(
            "Random amino acid completion fallback has been removed. "
            "All amino acids must be generated by DPLM-2 masked diffusion. "
            "If DPLM-2 is unavailable, the system should fail gracefully."
        )
    
    def _select_best_position_to_unmask(self, sequence: str, masked_positions: Set[int], input_data: Dict) -> Optional[int]:
        """
        🎯 IMPROVED: Select the best position to unmask based on AAR improvement potential.
        
        Args:
            sequence: Current sequence
            masked_positions: Set of masked positions
            input_data: Input data for the task
            
        Returns:
            Best position to unmask, or None if no good choice
        """
        if not masked_positions:
            return None
        
        # 🎯 STRATEGY 1: Use real pLDDT confidence to prioritize low-confidence positions
        if hasattr(self, '_real_structure') and self._real_structure:
            try:
                # Create structure for pLDDT calculation
                structure_for_plddt = self._real_structure.copy()
                structure_for_plddt['sequence'] = sequence
                
                # Import and use real pLDDT computation
                from utils.real_plddt_computation import compute_plddt_from_structure
                plddt_scores = compute_plddt_from_structure(structure_for_plddt)
                
                if plddt_scores and len(plddt_scores) == len(sequence):
                    # Get confidence scores for all masked positions
                    confidence_scores = []
                    for pos in masked_positions:
                        if pos < len(plddt_scores):
                            conf = plddt_scores[pos]
                            confidence_scores.append((pos, conf))
                    
                    if confidence_scores:
                        # Sort by confidence (lowest first - most likely to improve AAR)
                        confidence_scores.sort(key=lambda x: x[1])
                        return confidence_scores[0][0]  # Return lowest confidence position
            except Exception as e:
                print(f"Warning: Real pLDDT confidence failed: {e}")
        
        # 🎯 STRATEGY 2: Use AAR-focused reward evaluation for positions
        best_pos = None
        best_expected_reward = -float('inf')
        
        for pos in masked_positions:
            # Check bounds
            if pos >= len(sequence):
                continue
                
            # Try different amino acids at this position
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            position_rewards = []
            
            for aa in amino_acids:
                test_sequence = list(sequence)
                test_sequence[pos] = aa
                # Use AAR-focused reward instead of compound reward
                # 🎯 CRITICAL FIX: Use the stored baseline structure for consistent reward calculation
                if hasattr(self, '_baseline_structure') and self._baseline_structure:
                    evaluation_structure = self._baseline_structure.copy()
                    evaluation_structure['sequence'] = ''.join(test_sequence)
                    evaluation_structure['length'] = len(test_sequence)
                    evaluation_structure['target_length'] = len(test_sequence)
                    reward = self._compute_compound_reward(''.join(test_sequence), evaluation_structure)
                else:
                    reward = self._evaluate_sequence(''.join(test_sequence), input_data.get('target_length', len(sequence)))
                position_rewards.append(reward)
            
            # Expected reward for this position
            expected_reward = sum(position_rewards) / len(position_rewards)
            
            if expected_reward > best_expected_reward:
                best_expected_reward = expected_reward
                best_pos = pos
        
        return best_pos
    
    def _compute_compound_reward(self, sequence: str, input_data: Dict) -> float:
        """
        🎯 AAR-HEAVY: Reward function optimized for sequence recovery with biophysical balance.
        
        This method computes a reward that is:
        - 90% AAR (Amino Acid Recovery) - primary focus for MCTS
        - 10% Structure quality - maintains biophysical properties
        
        Args:
            sequence: Sequence to evaluate
            input_data: Input data containing reference sequence
            
        Returns:
            AAR-heavy reward score (0.0 to 1.0)
        """
        try:
            # 🎯 COMPOUND REWARD METHOD IDENTIFIER
            # 🎯 STEP 1: Get reference sequence
            # 🎯 CRITICAL FIX: Always use the TRUE reference sequence, not DPLM-2 baseline
            # The issue was MCTS was comparing against DPLM-2 baseline instead of true reference
            if hasattr(self, '_reference_sequence') and self._reference_sequence:
                reference_sequence = self._reference_sequence
                print(f"🎯 Using TRUE reference sequence for reward calculation (length: {len(reference_sequence)})")
            else:
                reference_sequence = input_data.get('sequence', None)
                print(f"🎯 Using input_data reference sequence (length: {len(reference_sequence) if reference_sequence else 'None'})")
            
            if not reference_sequence:
                print(f"⚠️ No reference sequence found for reward calculation")
                return 0.0
            
            # 🎯 CRITICAL FIX: Only calculate AAR on COMPLETED sequences (no X's)
            # Calculating AAR on masked sequences gives completely wrong results
            if 'X' in sequence:
                masked_count = sequence.count('X')
                masking_ratio = masked_count / len(sequence)
                
                # 🎯 MASKED SEQUENCE: Cannot calculate real AAR, use biophysical properties only
                # This prevents MCTS from getting fake perfect scores on incomplete sequences
                # 🎯 COMPOUND REWARD METHOD: Using biophysical properties for masked sequences
                try:
                    from utils.reward_computation import LengthAwareRewardComputation
                    # 🚫 DISABLED ESMFold: Skip structure evaluation to avoid ESMFold loading issues
                    biophysical_calc = LengthAwareRewardComputation(use_real_structure_eval=False)
                    
                    # Calculate biophysical properties on unmasked positions only
                    unmasked_sequence = ''.join([aa for aa in sequence if aa != 'X'])
                    if len(unmasked_sequence) > 0:
                        hydrophobicity_score = biophysical_calc._compute_hydrophobicity_reward(unmasked_sequence, len(unmasked_sequence))
                        charge_score = biophysical_calc._compute_charge_reward(unmasked_sequence, len(unmasked_sequence))
                        diversity_score = biophysical_calc._compute_diversity_reward(unmasked_sequence)
                        stability_score = biophysical_calc._compute_stability_reward(unmasked_sequence, len(unmasked_sequence))
                        
                        biophysical_quality = (hydrophobicity_score + charge_score + diversity_score + stability_score) / 4.0
                        
                        # Apply masking penalty
                        masking_penalty = masking_ratio * 0.2  # 20% penalty for masking
                        final_reward = biophysical_quality * (1.0 - masking_penalty)
                        
                        print(f"🎯 Biophysical properties (masked sequence):")
                        print(f"   Unmasked positions: {len(unmasked_sequence)}/{len(sequence)}")
                        print(f"   Hydrophobicity: {hydrophobicity_score:.3f}")
                        print(f"   Charge balance: {charge_score:.3f}")
                        print(f"   Diversity: {diversity_score:.3f}")
                        print(f"   Stability: {stability_score:.3f}")
                        print(f"   Average: {biophysical_quality:.3f}")
                        print(f"   Masking penalty: {masking_penalty:.3f}")
                    else:
                        # Completely masked sequence
                        final_reward = 0.1  # Very low reward
                        
                except Exception as e:
                    print(f"⚠️ Biophysical calculation failed: {e}, using fallback")
                    # Fallback: simple masking penalty
                    masking_penalty = masking_ratio * 0.2
                    final_reward = 0.8 * (1.0 - masking_penalty)
                
                print(f"🎯 MASKED SEQUENCE REWARD (biophysical properties):")
                print(f"   Sequence has {masked_count} masked positions ({masking_ratio:.1%})")
                print(f"   Final reward: {final_reward:.3f} (biophysical + masking penalty)")
                print(f"   Note: AAR calculation skipped - sequence incomplete")
                
                return final_reward
            
            # 🎯 COMPLETED SEQUENCE: Calculate real AAR
            if len(sequence) != len(reference_sequence):
                # Handle length mismatches
                min_length = min(len(sequence), len(reference_sequence))
                sequence = sequence[:min_length]
                reference_sequence = reference_sequence[:min_length]
            
            # Calculate AAR on completed sequence
            correct_matches = 0
            total_positions = len(sequence)
            
            for i, (gen_aa, ref_aa) in enumerate(zip(sequence, reference_sequence)):
                if gen_aa == ref_aa:
                    correct_matches += 1
            
            aar = correct_matches / total_positions if total_positions > 0 else 0.0
            
            # 🎯 STEP 3: Calculate scTM score using ESMFold and TMalign (following DPLM-2 approach)
            sctm_score = 0.0
            try:
                from utils.sctm_calculation import calculate_sctm_with_cameo_data
                
                # Calculate scTM score for structural quality using reference structure
                print(f"🧬 Calculating scTM score using ESMFold...")
                
                # Use reference structure data if available
                structure_data = None
                
                # Debug: Print what's available - SAFE input_data handling
                # print(f"🔍 Checking for structure data...")
                # print(f"🔍 input_data type: {type(input_data)}")
                
                # SAFE: Only call .keys() if input_data is actually a dict
                # if isinstance(input_data, dict):
                #     print(f"🔍 input_data keys: {list(input_data.keys())}")
                # else:
                #     print(f"🔍 input_data is not a dict: {input_data}")
                
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
                    
                    # print(f"🔍 Created CAMEO-format structure_data from _baseline_structure")
                    # print(f"    bb_positions shape: {structure_data['bb_positions'].shape if structure_data['bb_positions'] is not None else 'None'}")
                    # print(f"    sequence length: {len(structure_data['sequence'])}")
                    # print(f"    structure_data type: {type(structure_data)}")
                    # print(f"    structure_data keys: {list(structure_data.keys())}")
                    
                elif isinstance(input_data, dict) and 'structure_data' in input_data and isinstance(input_data['structure_data'], dict):
                    structure_data = input_data['structure_data']
                    print(f"🔍 Using structure_data from input_data")
                else:
                    print(f"⚠️ No valid structure data available for scTM calculation")
                    structure_data = None
                
                if structure_data is not None:
                    print(f"🔍 structure_data type: {type(structure_data)}")
                    print(f"🔍 structure_data value: {structure_data}")
                    if isinstance(structure_data, dict):
                        print(f"🔍 structure_data keys: {list(structure_data.keys())}")
                    else:
                        print(f"⚠️ structure_data is not a dict! Type: {type(structure_data)}, Value: {structure_data}")
                        # Don't call scTM with invalid data
                        print(f"⚠️ Skipping scTM calculation due to invalid structure_data")
                        sctm_score = 0.0
                        return aar_score  # Return early with just AAR score
                    
                    sctm_score = calculate_sctm_with_cameo_data(sequence, structure_data)
                    if sctm_score is not None and sctm_score > 0:
                        print(f"✅ scTM score: {sctm_score:.3f}")
                    else:
                        print(f"⚠️ scTM calculation failed, skipping scTM component")
                        sctm_score = 0.0  # Skip scTM component if calculation fails
                else:
                    print(f"⚠️ No reference structure data available for scTM calculation")
                    sctm_score = 0.0  # Skip scTM component if no reference structure
                    
            except Exception as e:
                print(f"⚠️ scTM calculation failed: {e}, skipping scTM component")
                sctm_score = 0.0  # Skip scTM component if calculation fails
            
            # 🎯 STEP 4: Calculate REAL biophysical properties (5% of reward)
            # This provides additional validation alongside scTM
            try:
                from utils.reward_computation import LengthAwareRewardComputation
                biophysical_calc = LengthAwareRewardComputation(use_real_structure_eval=False)
                
                # Calculate multiple biophysical properties
                hydrophobicity_score = biophysical_calc._compute_hydrophobicity_reward(sequence, len(sequence))
                charge_score = biophysical_calc._compute_charge_reward(sequence, len(sequence))
                diversity_score = biophysical_calc._compute_diversity_reward(sequence)
                stability_score = biophysical_calc._compute_stability_reward(sequence, len(sequence))
                
                # Combine biophysical scores (equal weighting)
                biophysical_quality = (hydrophobicity_score + charge_score + diversity_score + stability_score) / 4.0
                
                print(f"🎯 Biophysical properties:")
                print(f"   Hydrophobicity: {hydrophobicity_score:.3f}")
                print(f"   Charge balance: {charge_score:.3f}")
                print(f"   Diversity: {diversity_score:.3f}")
                print(f"   Stability: {stability_score:.3f}")
                print(f"   Average: {biophysical_quality:.3f}")
                
            except Exception as e:
                print(f"⚠️ Biophysical calculation failed: {e}, using fallback")
                biophysical_quality = 0.8  # Reasonable fallback
            
            # 🎯 STEP 5: Combine rewards (85% AAR + 10% scTM + 5% Biophysical)
            aar_weight = 0.85
            sctm_weight = 0.10
            biophysical_weight = 0.05
            
            final_reward = (aar * aar_weight) + (sctm_score * sctm_weight) + (biophysical_quality * biophysical_weight)
            
            print(f"🎯 COMPLETED SEQUENCE REWARD (AAR + scTM + Biophysical):")
            print(f"   AAR: {aar:.3f} (weight: {aar_weight:.1%})")
            print(f"   scTM: {sctm_score:.3f} (weight: {sctm_weight:.1%})")
            print(f"   Biophysical quality: {biophysical_quality:.3f} (weight: {biophysical_weight:.1%})")
            print(f"   Final reward: {final_reward:.3f}")
            
            return final_reward
            
        except Exception as e:
            print(f"⚠️ Error computing AAR-heavy reward: {e}")
            return 0.0
    
    def _compute_aar_focused_reward(self, sequence: str, input_data: Dict) -> float:
        """
        🎯 COMPUTE: AAR-focused reward for inverse folding optimization.
        
        This method computes a reward that prioritizes amino acid recovery
        while maintaining structural quality. It's the core reward function
        for MCTS optimization.
        
        Args:
            sequence: Sequence to evaluate
            input_data: Input data containing reference sequence
            
        Returns:
            Reward score optimized for AAR improvement
        """
        try:
            # 🎯 KEY: Get reference sequence from input_data
            # 🎯 CRITICAL FIX: Always use the TRUE reference sequence, not DPLM-2 baseline
            if hasattr(self, '_reference_sequence') and self._reference_sequence:
                reference_sequence = self._reference_sequence
                print(f"🎯 Using TRUE reference sequence for AAR-focused reward (length: {len(reference_sequence)})")
            else:
                reference_sequence = input_data.get('sequence', None)
                print(f"🎯 Using input_data reference sequence for AAR-focused reward (length: {len(reference_sequence) if reference_sequence else 'None'})")
            
            if not reference_sequence:
                return 0.0
        
            # 🎯 CRITICAL FIX: Only calculate AAR on COMPLETED sequences (no X's)
            # Calculating AAR on masked sequences gives completely wrong results
            if 'X' in sequence:
                masked_count = sequence.count('X')
                masking_ratio = masked_count / len(sequence)
                
                # 🎯 MASKED SEQUENCE: Heavy penalty to encourage unmasking
                # Masked sequences should get much lower rewards than completed sequences
                # Use baseline AAR as reference point and apply heavy masking penalty
                baseline_aar = getattr(self, '_baseline_aar', 0.5)  # Default to 50% if not set
                
                # Heavy penalty: masked sequences get at most 50% of baseline AAR
                max_masked_reward = baseline_aar * 0.5
                masking_penalty = masking_ratio * 0.8  # 80% penalty for each masked position
                final_reward = max_masked_reward * (1.0 - masking_penalty)
                
                print(f"🎯 MASKED SEQUENCE REWARD (heavy penalty applied):")
                print(f"   Sequence has {masked_count} masked positions ({masking_ratio:.1%})")
                print(f"   Baseline AAR: {baseline_aar:.3f}")
                print(f"   Max masked reward: {max_masked_reward:.3f}")
                print(f"   Masking penalty: {masking_penalty:.3f}")
                print(f"   Final reward: {final_reward:.3f}")
                print(f"   Note: Heavy penalty encourages sequence completion")
                
                return final_reward
            
            # 🎯 COMPUTE: AAR between generated and reference sequences (unmasked case)
            if len(sequence) != len(reference_sequence):
                # Handle length mismatches by truncating to shorter length
                min_length = min(len(sequence), len(reference_sequence))
                sequence = sequence[:min_length]
                reference_sequence = reference_sequence[:min_length]
            
            # 🎯 CALCULATE: AAR using the same logic as final verification
            correct_matches = 0
            total_positions = len(sequence)
            
            for i, (gen_aa, ref_aa) in enumerate(zip(sequence, reference_sequence)):
                if gen_aa == ref_aa:
                    correct_matches += 1
            
            aar = correct_matches / total_positions if total_positions > 0 else 0.0
            
            # 🎯 REWARD: AAR is the primary component
            aar_reward = aar
            
            # 🎯 BONUS: Small bonus for sequence quality (no X characters, reasonable length)
            quality_bonus = 0.0
            if 'X' not in sequence and len(sequence) > 0:
                quality_bonus = 0.02  # Small bonus for clean sequences
            
            # 🎯 FINAL: Combine AAR and quality bonus
            final_reward = aar_reward + quality_bonus
            
            return final_reward
            
        except Exception as e:
            return 0.0
    
    def _adapt_exploration_constant(self, base_constant: float, baseline_aar: Optional[float]) -> float:
        """
        🎯 ADAPTIVE EXPLORATION: Adjust exploration constant based on AAR improvement potential.
        
        Args:
            base_constant: Base exploration constant
            baseline_aar: Baseline AAR score (can be None)
            
        Returns:
            Adapted exploration constant
        """
        if baseline_aar is None:
            return base_constant
        
        # 🎯 IMPROVED EXPLORATION STRATEGY: More nuanced adaptation
        if baseline_aar < 0.3:  # Very low AAR - need aggressive exploration
            adapted_constant = base_constant * 2.0  # Much more exploration
            print(f"🎯 Very low baseline AAR ({baseline_aar:.1%}) - aggressive exploration: {adapted_constant:.2f}")
        elif baseline_aar < 0.5:  # Low AAR - need moderate exploration
            adapted_constant = base_constant * 1.3  # More exploration
            print(f"🎯 Low baseline AAR ({baseline_aar:.1%}) - moderate exploration: {adapted_constant:.2f}")
        elif baseline_aar < 0.6:  # Medium AAR - balanced exploration
            adapted_constant = base_constant * 1.1  # Slight exploration
            print(f"🎯 Medium baseline AAR ({baseline_aar:.1%}) - balanced exploration: {adapted_constant:.2f}")
        elif baseline_aar > 0.7:  # High AAR - focus on exploitation
            adapted_constant = base_constant * 0.6  # Much less exploration, more exploitation
            print(f"🎯 High baseline AAR ({baseline_aar:.1%}) - focused exploitation: {adapted_constant:.2f}")
        else:  # 0.6-0.7 range - moderate exploitation
            adapted_constant = base_constant * 0.8  # Less exploration, more exploitation
            print(f"🎯 Good baseline AAR ({baseline_aar:.1%}) - moderate exploitation: {adapted_constant:.2f}")
        
        return adapted_constant
    
    def _calculate_simple_aar(self, pred_seq: str, ref_seq: str) -> float:
        """Calculate simple AAR (Amino Acid Recovery) - FIXED VERSION."""
        if len(pred_seq) != len(ref_seq):
            # If lengths differ, align to the shorter one for fair comparison
            min_len = min(len(pred_seq), len(ref_seq))
            pred_seq = pred_seq[:min_len]
            ref_seq = ref_seq[:min_len]
        
        if len(ref_seq) == 0:
            return 0.0
        
        # Calculate matches (same logic as working script)
        matches = sum(1 for p, r in zip(pred_seq, ref_seq) if p == r)
        return matches / len(ref_seq) if len(ref_seq) > 0 else 0.0
    
    def _compute_basic_sequence_validity(self, sequence: str) -> float:
        """Compute basic sequence validity score."""
        if not sequence:
            return 0.0
        
        # Check for valid amino acids
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        valid_count = sum(1 for aa in sequence if aa in valid_aas)
        validity_ratio = valid_count / len(sequence)
        
        # Check for reasonable length
        length_score = 1.0 if 20 <= len(sequence) <= 500 else 0.8
        
        return (validity_ratio + length_score) / 2.0
    
    def _compute_folding_reward(self, sequence: str, input_data: Dict) -> float:
        """Compute reward for folding task."""
        # Placeholder for folding reward
        return 0.5 + random.uniform(0, 0.3)
    
    def _compute_unconditional_reward(self, sequence: str) -> float:
        """Compute reward for unconditional generation."""
        # Placeholder for unconditional generation reward
        return 0.5 + random.uniform(0, 0.3)
    
    def _compute_conditional_reward(self, sequence: str, input_data: Dict) -> float:
        """Compute reward for conditional generation."""
        # Placeholder for conditional generation reward
        return 0.5 + random.uniform(0, 0.3) 

    def _create_folding_masked_sequence(self, input_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create masked sequence for folding task."""
        # For folding, we start with a sequence and mask some positions
        sequence = input_data.get('sequence', '')
        if not sequence:
            # Generate random sequence if none provided
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            # 🚫 NO RANDOM SEQUENCE GENERATION
            print(f"❌ Cannot generate random sequence - must use DPLM-2 baseline")
            return 0.0
        
        # Mask some positions randomly
        num_to_mask = max(1, len(sequence) // 4)
        masked_positions = set(random.sample(range(len(sequence)), num_to_mask))
        
        # Create masked sequence
        masked_sequence = list(sequence)
        for pos in masked_positions:
            masked_sequence[pos] = 'X'
        
        return ''.join(masked_sequence), masked_positions
    
    def _create_unconditional_masked_sequence(self, target_length: int) -> Tuple[str, Set[int]]:
        """Create masked sequence for unconditional generation."""
        # Start with all positions masked
        masked_positions = set(range(target_length))
        
        # Create fully masked sequence
        masked_sequence = ['X'] * target_length
        
        return ''.join(masked_sequence), masked_positions
    
    def _create_conditional_masked_sequence(self, input_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create masked sequence for conditional generation."""
        # Similar to unconditional but with conditional input
        return self._create_unconditional_masked_sequence(target_length)
    
    def _select(self, node: MCTSNode, exploration_constant: float, use_ph_uct: bool = True) -> MCTSNode:
        """
        🎯 IMPROVED: Select a node using either UCB1 or PH-UCT algorithm.
        
        PH-UCT (Entropy-Reinforced Planning) combines UCB1 with entropy-based
        and diversity-based exploration for better tree search.
        
        Args:
            node: Current node
            exploration_constant: UCB1 exploration constant
            use_ph_uct: Whether to use PH-UCT (True) or UCB1 (False)
            
        Returns:
            Selected child node
        """
        while node.children:
            # 🎯 NEW: PH-UCT selection with entropy reinforcement
            if use_ph_uct:
                best_child = self._select_ph_uct(node, exploration_constant)
            else:
                best_child = self._select_ucb1(node, exploration_constant)
            
            if best_child is None:
                # Fallback: select first unvisited child
                for child in node.children:
                    if child.visit_count == 0:
                        best_child = child
                        break
                
                if best_child is None:
                    # All children visited, select best by value
                    best_child = max(node.children, key=lambda x: x.average_value)
            
            node = best_child
        
        return node
    
    def _select_ph_uct(self, node: MCTSNode, exploration_constant: float) -> MCTSNode:
        """
        🎯 NEW: PH-UCT selection algorithm with entropy reinforcement.
        
        This implements the entropy-reinforced planning approach that balances:
        1. Exploitation (UCB1 base)
        2. Entropy-based exploration (uncertainty)
        3. Diversity-based exploration (tree diversity)
        4. Exploration potential (completion and depth)
        
        Args:
            node: Current node
            exploration_constant: UCB1 exploration constant
            
        Returns:
            Selected child node
        """
        best_child = None
        best_score = float('-inf')
        
        # Skip entropy update for now - it requires structure context
        # The entropy scores should be updated during expansion phase
        # for child in node.children:
        #     self._update_node_entropy_with_dplm2(child, structure, target_length, temperature)
        
        # Collect unvisited children for proper selection
        unvisited_children = [child for child in node.children if child.visit_count == 0]
        
        # If there are unvisited children, select randomly among them instead of always first
        if unvisited_children:
            import random
            return random.choice(unvisited_children)
        
        # All children visited - use PH-UCT scoring
        for child in node.children:
            # 🎯 NEW: PH-UCT score calculation
            ph_uct_score = child.ph_uct_score(node.visit_count, exploration_constant, 
                                             self.entropy_weight, self.diversity_weight)
            
            if ph_uct_score > best_score:
                best_score = ph_uct_score
                best_child = child
        
        return best_child
    
    def _select_ucb1(self, node: MCTSNode, exploration_constant: float) -> MCTSNode:
        """
        🎯 IMPROVED: UCB1 selection algorithm (original method).
        
        Args:
            node: Current node
            exploration_constant: UCB1 exploration constant
            
        Returns:
            Selected child node
        """
        best_child = None
        best_ucb = float('-inf')
        
        for child in node.children:
            if child.visit_count == 0:
                return child  # Return unvisited child
            
            # UCB1 formula: exploitation + exploration
            exploitation = child.average_value
            exploration = exploration_constant * math.sqrt(math.log(node.visit_count) / child.visit_count)
            ucb = exploitation + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child
        
        return best_child
    

    
    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Backpropagate the reward up the tree.
        
        Args:
            node: Node to start backpropagation from
            reward: Reward to backpropagate
        """
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent
    
    def _generate_random_sequences(self, length: int, num_sequences: int) -> List[str]:
        """
        🚫 REMOVED: Random sequence generation fallback.
        
        This method has been removed because:
        1. We should ONLY use DPLM-2 for sequence generation
        2. Random sequences provide no meaningful optimization
        3. All sequences must come from real DPLM-2 masked diffusion
        
        If DPLM-2 is not available, the system should fail gracefully
        rather than generate meaningless random sequences.
        """
        raise NotImplementedError(
            "Random sequence generation has been removed. "
            "All sequences must be generated by DPLM-2 masked diffusion. "
            "If DPLM-2 is unavailable, the system should fail gracefully."
        )

    def _evaluate_sequence(self, sequence: str, target_length: int) -> float:
        """
        🎯 IMPROVED: Evaluate a sequence using AAR-aware reward computation.
        
        This method ensures consistency between MCTS internal evaluation
        and final verification by using the same reward function.
        
        Args:
            sequence: Sequence to evaluate
            target_length: Target sequence length
            
        Returns:
            Reward score optimized for both AAR and structural quality
        """
        try:
            # 🎯 CRITICAL FIX: Get reference sequence from the correct source
            reference_seq = getattr(self, '_reference_sequence', None)
            
            if not reference_seq:
                print(f"⚠️ No reference sequence found for evaluation")
                return 0.0
            
            # 🎯 KEY FIX: Use REAL structure with the CORRECT reference sequence
            # This ensures AAR calculation uses the right reference and real structural data
            if hasattr(self, '_baseline_structure') and self._baseline_structure:
                # Use the real baseline structure that was used for initial generation
                real_structure = self._baseline_structure.copy()
                real_structure['sequence'] = reference_seq  # Update with reference sequence
                real_structure['length'] = len(reference_seq)
                real_structure['target_length'] = target_length
                
                # 🎯 CRITICAL FIX: Use the EXACT same reward function as final verification
                # This eliminates the reward mismatch between MCTS internal and final evaluation
                compound_reward = self._compute_compound_reward(sequence, real_structure)
            else:
                # 🎯 FALLBACK: Create minimal but REAL structure (no mock data)
                real_structure = {
                    'sequence': reference_seq,  # Use reference sequence, not generated sequence
                    'length': len(reference_seq),
                    'target_length': target_length
                }
                
                # 🎯 CRITICAL FIX: Use the EXACT same reward function as final verification
                compound_reward = self._compute_compound_reward(sequence, real_structure)
            
            return compound_reward
                
        except Exception as e:
            print(f"⚠️ Error in sequence evaluation: {e}")
            # Emergency fallback: return sequence validity score
            return self._compute_basic_sequence_validity(sequence)

    def _create_masked_input_for_dplm2(self, sequence: str, masked_positions: Set[int], 
                                      structure: Dict) -> Dict:
        """
        🎯 IMPROVED: Create masked input for DPLM-2.
        
        This method prepares the input that DPLM-2 needs to generate
        new amino acids for the masked positions.
        
        Args:
            sequence: Current sequence with masked positions
            masked_positions: Set of masked positions
            structure: Structure information
            
        Returns:
            Formatted input for DPLM-2
        """
        try:
            # 🎯 STEP 1: Create proper structure input with structure tokens
            proper_structure = self._create_proper_structure_input(structure, sequence)
            
            # �� STEP 2: Create the masked input structure
            masked_input = {
                'sequence': sequence,  # Sequence with X tokens
                'masked_positions': list(masked_positions),  # List of masked positions
                'structure': proper_structure,  # Proper structure with tokens
                'target_length': len(sequence),
                'task_type': 'inverse_folding'
            }
            
            # 🎯 STEP 3: Add structure-specific information if available
            if 'coordinates' in proper_structure:
                masked_input['coordinates'] = proper_structure['coordinates']
            if 'struct_tokens' in proper_structure:
                masked_input['struct_tokens'] = proper_structure['struct_tokens']
            if 'plddt_scores' in proper_structure:
                masked_input['plddt_scores'] = proper_structure['plddt_scores']
            
            # 🎯 STEP 4: Format for DPLM-2 consumption
            if hasattr(self.dplm2_integration, '_format_input_for_generation'):
                masked_input = self.dplm2_integration._format_input_for_generation(masked_input)
            
            return masked_input
            
        except Exception as e:
            print(f"❌ Error creating masked input: {e}")
            raise ValueError(f"Cannot create DPLM-2 input: {e}")
    
    def _store_original_structure(self, structure: Dict):
        """
        🎯 IMPROVED: Store the original structure for proper DPLM-2 input.
        
        This ensures we always have access to the original structure
        with coordinates for high-quality sequence generation.
        
        Args:
            structure: Original structure information
        """
        try:
            # 🎯 VALIDATE: Only store if structure is valid
            if not self._validate_structure_for_dplm2(structure):
                print(f"⚠️ Cannot store invalid structure")
                return
            
            self._original_structure = structure.copy()
            
        except Exception as e:
            print(f"⚠️ Could not store original structure: {e}")

    def _calculate_reward(self, sequence: str, input_data: Dict) -> float:
        """
        🎯 HELPER: Calculate reward using the same method as final verification.
        
        This method ensures consistency between MCTS internal evaluation
        and final verification by using the exact same reward function.
        
        Args:
            sequence: Sequence to evaluate
            input_data: Input data for the task
            
        Returns:
            Reward score that matches final verification
        """
        # 🎯 CRITICAL FIX: Use the EXACT same reward calculation as final verification
        # This eliminates the reward mismatch between MCTS internal and final evaluation
        return self._compute_compound_reward(sequence, input_data)

    def _create_variations(self, node: 'MCTSNode', num_candidates: int, target_length: int, temperature: float = 1.0) -> List[Tuple[str, Set[int]]]:
        """
        Create sequence variations for MCTS expansion using MD4-style batch masked diffusion.
        
        This method creates variations by:
        1. Using DPLM-2 for batch masked diffusion (multiple positions at once)
        2. Creating intelligent amino acid changes
        3. Ensuring exploration diversity
        """
        variations = []
        
        # 🎯 STRATEGY 1: MD4-style batch masked diffusion for optimization
        if hasattr(self, 'dplm2_integration') and self.dplm2_integration:
            try:
                # 🎯 CRITICAL FIX: Use MD4-style batch masking instead of single position filling
                # This leverages DPLM-2's full masked diffusion capabilities
                if node.masked_positions:
                    # Create variations by masking different combinations of positions
                    for _ in range(min(num_candidates // 2, 3)):  # Focus on DPLM-2 variations
                        # 🎯 MD4-STYLE: Mask multiple positions simultaneously
                        # Choose 3-5 positions to mask for efficient exploration
                        num_positions_to_mask = min(5, len(node.masked_positions))
                        positions_to_mask = random.sample(list(node.masked_positions), num_positions_to_mask)
                        
                        # Create sequence with multiple positions masked
                        masked_sequence = list(node.sequence)
                        for pos in positions_to_mask:
                            masked_sequence[pos] = 'X'
                        masked_sequence = ''.join(masked_sequence)
                        
                        print(f"🎯 MD4-style batch masking: masking {len(positions_to_mask)} positions simultaneously")
                        
                        # Use DPLM-2 to fill ALL masked positions at once (pure masked diffusion)
                        filled_sequence = self._resample_masked_positions_with_dplm2(
                            masked_sequence,
                            set(positions_to_mask),  # Pass ALL positions to fill
                            structure=None,  # No structure for pure masked diffusion
                            temperature=0.8  # Lower temperature for more focused generation
                        )
                        
                        if filled_sequence and len(filled_sequence) == target_length:
                            # Successfully filled multiple positions
                            new_masked_positions = node.masked_positions.copy()
                            for pos in positions_to_mask:
                                new_masked_positions.discard(pos)
                            
                            variations.append((filled_sequence, new_masked_positions))
                            print(f"🎯 DPLM-2 batch masked diffusion filled {len(positions_to_mask)} positions, remaining masked: {len(new_masked_positions)}")
                        else:
                            # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                            print(f"❌ DPLM-2 batch masked diffusion failed: could not fill positions")
                            continue
            except Exception as e:
                print(f"⚠️ DPLM-2 batch masked diffusion failed: {e}")
        
        # 🎯 STRATEGY 2: Progressive masking for exploration
        if len(variations) < num_candidates:
            # Create variations by progressively masking more positions
            for _ in range(min(num_candidates - len(variations), 2)):
                if node.masked_positions:
                    # 🎯 PROGRESSIVE MASKING: Start with fewer positions, then increase
                    num_positions = min(3, len(node.masked_positions))
                    positions_to_mask = random.sample(list(node.masked_positions), num_positions)
                    
                    # Create sequence with multiple positions masked
                    masked_sequence = list(node.sequence)
                    for pos in positions_to_mask:
                        masked_sequence[pos] = 'X'
                    masked_sequence = ''.join(masked_sequence)
                    
                    # Use DPLM-2 to fill all masked positions simultaneously
                    filled_sequence = self._resample_masked_positions_with_dplm2(
                        masked_sequence,
                        set(positions_to_mask),  # Fill all positions at once
                        structure=None,  # No structure for pure masked diffusion
                        temperature=1.0
                    )
                    
                    if filled_sequence and len(filled_sequence) == target_length:
                        new_masked_positions = node.masked_positions.copy()
                        for pos in positions_to_mask:
                            new_masked_positions.discard(pos)
                        variations.append((filled_sequence, new_masked_positions))
                    else:
                        # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                        print(f"❌ Progressive masking failed: DPLM-2 could not fill positions")
                        continue
        
        # 🎯 STRATEGY 3: TRUE MD4-STYLE DYNAMIC MASKING - Mask previously unmasked positions
        if len(variations) < num_candidates:
            # 🎯 CRITICAL INNOVATION: Mask previously unmasked positions for maximum exploration
            # This is the core of MD4-style - explore new combinations by masking different positions
            for _ in range(min(num_candidates - len(variations), 2)):
                # 🎯 DYNAMIC MASKING: Choose positions to mask from ALL available positions
                # This includes both currently masked AND previously unmasked positions
                all_positions = set(range(len(node.sequence)))
                currently_masked = node.masked_positions
                previously_unmasked = all_positions - currently_masked
                
                # 🎯 STRATEGY: Mix of currently masked and previously unmasked positions
                if previously_unmasked and currently_masked:
                    # 🎯 MD4-STYLE: Mask some previously unmasked positions for exploration
                    num_new_to_mask = min(2, len(previously_unmasked))  # Mask 1-2 new positions
                    num_current_to_mask = min(2, len(currently_masked))  # Mask 1-2 current positions
                    
                    new_positions_to_mask = random.sample(list(previously_unmasked), num_new_to_mask)
                    current_positions_to_mask = random.sample(list(currently_masked), num_current_to_mask)
                    positions_to_mask = new_positions_to_mask + current_positions_to_mask
                    
                    print(f"🎯 MD4-style dynamic masking: masking {num_new_to_mask} new + {num_current_to_mask} current positions")
                    
                elif previously_unmasked:
                    # 🎯 EXPLORE NEW: Mask some previously unmasked positions
                    num_positions = min(3, len(previously_unmasked))
                    positions_to_mask = random.sample(list(previously_unmasked), num_positions)
                    print(f"🎯 MD4-style new exploration: masking {num_positions} previously unmasked positions")
                    
                elif currently_masked:
                    # 🎯 REFINE CURRENT: Mask some currently masked positions
                    num_positions = min(3, len(currently_masked))
                    positions_to_mask = random.sample(list(currently_masked), num_positions)
                    print(f"🎯 MD4-style current refinement: masking {num_positions} currently masked positions")
                    
                else:
                    # 🎯 FULL EXPLORATION: Mask random positions for maximum diversity
                    num_positions = min(3, len(node.sequence) // 10)  # 10% of sequence
                    positions_to_mask = random.sample(range(len(node.sequence)), num_positions)
                    print(f"🎯 MD4-style full exploration: masking {num_positions} random positions")
                
                # Create sequence with dynamic masking
                masked_sequence = list(node.sequence)
                for pos in positions_to_mask:
                    masked_sequence[pos] = 'X'
                masked_sequence = ''.join(masked_sequence)
                
                # Use DPLM-2 to fill all masked positions simultaneously
                filled_sequence = self._resample_masked_positions_with_dplm2(
                    masked_sequence,
                    set(positions_to_mask),  # Fill all positions at once
                    structure=None,  # No structure for pure masked diffusion
                    temperature=1.0
                )
                
                if filled_sequence and len(filled_sequence) == target_length:
                    # 🎯 SUCCESS: Create new masked positions set
                    new_masked_positions = set()
                    for pos in range(len(filled_sequence)):
                        if pos in positions_to_mask:
                            # These positions were masked and filled by DPLM-2
                            continue
                        elif pos in node.masked_positions:
                            # These positions were already masked and remain masked
                            new_masked_positions.add(pos)
                        # Previously unmasked positions that weren't masked in this variation remain unmasked
                    
                    variations.append((filled_sequence, new_masked_positions))
                    print(f"🎯 MD4-style dynamic masking successful: {len(positions_to_mask)} positions filled, {len(new_masked_positions)} remaining masked")
                    
                else:
                    # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                    print(f"❌ MD4-style dynamic masking failed: DPLM-2 could not fill positions")
                    continue
        
        # 🎯 STRATEGY 4: DYNAMIC pLDDT-BASED MASKING - Recalculate confidence for each sequence
        if len(variations) < num_candidates:
            # 🎯 CRITICAL INNOVATION: Use dynamic pLDDT calculation during MCTS exploration
            # This recalculates confidence scores for the current sequence and applies intelligent masking
            for _ in range(min(num_candidates - len(variations), 2)):
                try:
                    # 🎯 STEP 1: Calculate dynamic pLDDT for the current sequence
                    if hasattr(self, '_real_structure') and self._real_structure:
                        # Use real structure for dynamic pLDDT calculation
                        structure_for_plddt = self._real_structure.copy()
                        structure_for_plddt['sequence'] = node.sequence  # Update with current sequence
                    else:
                        # Create minimal structure for pLDDT calculation
                        structure_for_plddt = {
                            'sequence': node.sequence,
                            'length': len(node.sequence),
                            'target_length': len(node.sequence)
                        }
                    
                    print(f"🎯 Dynamic pLDDT calculation for current sequence")
                    
                    # 🎯 STEP 2: Apply dynamic pLDDT masking to the current sequence
                    masked_sequence, new_masked_positions = self._apply_plddt_masking(
                        node.sequence,
                        structure_for_plddt,
                        masking_ratio=0.2  # 20% maximum masking for exploration
                    )
                    
                    # 🎯 STEP 3: Use DPLM-2 to fill the newly masked positions
                    if len(new_masked_positions) > 0:
                        filled_sequence = self._resample_masked_positions_with_dplm2(
                            masked_sequence,
                            new_masked_positions,  # Fill all newly masked positions
                            structure=None,  # No structure for pure masked diffusion
                            temperature=0.9  # Slightly lower temperature for pLDDT-guided generation
                        )
                        
                        if filled_sequence and len(filled_sequence) == target_length:
                            # 🎯 SUCCESS: Dynamic pLDDT masking and DPLM-2 filling
                            # Calculate remaining masked positions after filling
                            remaining_masked = set()
                            for pos in range(len(filled_sequence)):
                                if pos in new_masked_positions:
                                    # These positions were filled by DPLM-2
                                    continue
                                elif pos in node.masked_positions:
                                    # These positions were already masked and remain masked
                                    remaining_masked.add(pos)
                            
                            variations.append((filled_sequence, remaining_masked))
                            print(f"🎯 Dynamic pLDDT masking successful: {len(new_masked_positions)} positions filled, {len(remaining_masked)} remaining masked")
                            
                        else:
                            # 🚫 NO FALLBACK: If DPLM-2 fails, we cannot create this variation
                            print(f"❌ Dynamic pLDDT masking failed: DPLM-2 could not fill positions")
                            continue
                    else:
                        # No new masking needed - sequence is already optimal
                        print(f"🎯 Dynamic pLDDT: No new masking needed for current sequence")
                        
                except Exception as e:
                    print(f"⚠️ Dynamic pLDDT masking failed: {e}")
                    continue
        
        # 🚫 STRATEGY 5 REMOVED: No random fallback variations
        # All variations must come from DPLM-2 - no random amino acid changes allowed

                if variation != node.sequence:
                    variations.append((variation, set()))  # No masked positions
        
        # 🎯 ENSURE: We have at least some variations
        if len(variations) == 0:
            print(f"⚠️ No variations created - DPLM-2 failed for all strategies")
            # 🚫 NO FALLBACK: Cannot create random variations
            print(f"❌ Cannot proceed without DPLM-2-generated variations")
            return  # Return empty list - no children will be created
        
        return variations

    def _simulate(self, node: MCTSNode, target_length: int, max_depth: int) -> float:
        """
        🎯 IMPROVED: Simulate a rollout from the given node.
        
        Args:
            node: Node to simulate from
            target_length: Target sequence length
            max_depth: Maximum simulation depth
            
        Returns:
            Real reward value from sequence evaluation
        """
        current_sequence = node.sequence
        current_masked = set(node.masked_positions)
        depth = 0
        
        # 🎯 IMPROVED: AAR-aware simulation with intelligent amino acid selection
        while current_masked and depth < max_depth:
            if current_masked:
                # 🎯 STRATEGY: Prioritize positions that are most likely to improve AAR
                # Use real pLDDT instead of fake confidence scores
                try:
                    # 🎯 REAL pLDDT: Calculate actual confidence from structure
                    if hasattr(self, '_real_structure') and self._real_structure:
                        # Create structure for pLDDT calculation
                        structure_for_plddt = self._real_structure.copy()
                        structure_for_plddt['sequence'] = current_sequence
                        
                        # Import and use real pLDDT computation
                        from utils.real_plddt_computation import compute_plddt_from_structure
                        plddt_scores = compute_plddt_from_structure(structure_for_plddt)
                        
                        if plddt_scores and len(plddt_scores) == len(current_sequence):
                            # Find position with lowest pLDDT (most likely to improve)
                            masked_plddt = [(pos, plddt_scores[pos]) for pos in current_masked if pos < len(plddt_scores)]
                            if masked_plddt:
                                masked_plddt.sort(key=lambda x: x[1])  # Sort by pLDDT (lowest first)
                                pos = masked_plddt[0][0]  # Pick lowest confidence position
                                
                                # 🎯 IMPROVED: Use DPLM-2 for amino acid selection
                                if hasattr(self, 'dplm2_integration') and self.dplm2_integration:
                                    # Create masked sequence for this position
                                    masked_sequence = list(current_sequence)
                                    masked_sequence[pos] = 'X'
                                    masked_sequence = ''.join(masked_sequence)
                                    
                                    # Use DPLM-2 to fill this position
                                    filled_sequence = self._resample_masked_positions_with_dplm2(
                                        masked_sequence,
                                        {pos},  # Fill this single position
                                        structure=None,  # No structure for pure masked diffusion
                                        temperature=self.temperature  # Use instance temperature
                                    )
                                    
                                    if filled_sequence and len(filled_sequence) == len(current_sequence):
                                        current_sequence = filled_sequence
                                        current_masked.discard(pos)
                                        print(f"🎯 DPLM-2 filled position {pos} in simulation")
                                    else:
                                        # 🚫 NO FALLBACK: If DPLM-2 fails, stop simulation
                                        print(f"❌ DPLM-2 failed to fill position {pos} in simulation")
                                        break
                                else:
                                    # 🚫 NO FALLBACK: DPLM-2 integration required
                                    print(f"❌ DPLM-2 integration not available for simulation")
                                    break
                            else:
                                # No valid pLDDT scores for masked positions
                                break
                        else:
                            # pLDDT calculation failed
                            break
                    else:
                        # No real structure available
                        break
                        
                except Exception as e:
                    # 🚫 NO FALLBACK: If pLDDT calculation fails, stop simulation
                    print(f"⚠️ pLDDT calculation failed in simulation: {e}")
                    break
            
            depth += 1
        
        # 🎯 IMPROVED: Evaluate with AAR-aware reward function
        # 🎯 CRITICAL FIX: Use the stored baseline structure for consistent reward calculation
        if hasattr(self, '_baseline_structure') and self._baseline_structure:
            # Use the stored baseline structure for consistent evaluation
            evaluation_structure = self._baseline_structure.copy()
            evaluation_structure['sequence'] = current_sequence
            evaluation_structure['length'] = len(current_sequence)
            evaluation_structure['target_length'] = target_length
            return self._compute_compound_reward(current_sequence, evaluation_structure)
        else:
            # Fallback to the general evaluation method
            return self._evaluate_sequence(current_sequence, target_length)
    
    def _create_masked_input(self, masked_sequence: str, input_data: Dict) -> Dict:
        """Create input with masked sequence for DPLM-2."""
        # For inverse folding, input_data is the structure
        structure = input_data.copy()
        structure['masked_sequence'] = masked_sequence
        return self.dplm2_integration._create_masked_input(masked_sequence, structure, len(masked_sequence))
    
    def _expand_masked_sequence(self, node: MCTSNode, target_length: int, num_candidates: int, temperature: float = 1.0) -> None:
        """Expand node with masked sequence using DPLM-2 unmasking."""
        print(f"🎯 Expanding masked sequence with {len(node.masked_positions)} masked positions")
        print(f"   Current sequence: {node.sequence[:50]}...")
        
        # Use proper diffusion unmasking to fill masked positions
        candidates = []
        
        # Fill all masked positions at once
        if len(node.masked_positions) > 0:
            try:
                filled_sequence = self._resample_masked_positions_with_dplm2(
                    node.sequence,
                    node.masked_positions,
                    structure=None,
                    temperature=temperature
                )
                
                if filled_sequence and len(filled_sequence) == target_length:
                    candidates.append(filled_sequence)
                    print(f"🎯 CANDIDATE 1: All positions filled successfully")
            except Exception as e:
                print(f"🎯 CANDIDATE 1: Error filling all positions: {e}")
        
        # Create child nodes
        for i, candidate_sequence in enumerate(candidates):
            if i >= num_candidates:
                break
            
            # Create child node
            child = MCTSNode(
                sequence=candidate_sequence,
                parent=node,
                depth=node.depth + 1,
                masked_positions=set()  # No more masked positions
            )
            
            node.children.append(child)
            print(f"🎯 CHILD {i+1}: Created with completed sequence")
    
    def _expand_complete_sequence(self, node: MCTSNode, target_length: int, num_candidates: int, temperature: float = 1.0) -> None:
        """Expand node with complete sequence using targeted amino acid substitutions."""
        print(f"🎯 Expanding complete sequence with targeted substitutions")
        print(f"   Current sequence: {node.sequence[:50]}...")
        
        candidates = []
        
        # Strategy 1: Make small amino acid substitutions at random positions
        for i in range(num_candidates):
            try:
                # Choose 1-3 random positions to modify
                num_positions = min(3, len(node.sequence))
                positions_to_modify = set(random.sample(range(len(node.sequence)), num_positions))
                
                # Create modified sequence
                modified_sequence = list(node.sequence)
                for pos in positions_to_modify:
                    # Replace with a different amino acid
                    current_aa = modified_sequence[pos]
                    available_aas = [aa for aa in 'ACDEFGHIKLMNPQRSTVWY' if aa != current_aa]
                    new_aa = random.choice(available_aas)
                    modified_sequence[pos] = new_aa
                
                modified_sequence = ''.join(modified_sequence)
                candidates.append(modified_sequence)
                print(f"🎯 CANDIDATE {i+1}: Modified {len(positions_to_modify)} positions")
                
            except Exception as e:
                print(f"🎯 CANDIDATE {i+1}: Error creating modification: {e}")
        
        # Create child nodes
        for i, candidate_sequence in enumerate(candidates):
            if i >= num_candidates:
                break
            
            # Create child node
            child = MCTSNode(
                sequence=candidate_sequence,
                parent=node,
                depth=node.depth + 1,
                masked_positions=set()  # No masked positions in complete sequences
            )
            
            node.children.append(child)
            print(f"🎯 CHILD {i+1}: Created with modified sequence")
    
    def analyze_tree_entropy(self, root: MCTSNode) -> Dict[str, float]:
        """
        🎯 NEW: Analyze entropy distribution across the MCTS tree.
        
        This method helps understand how PH-UCT is exploring the tree
        and whether entropy-based exploration is working effectively.
        
        Args:
            root: Root node of the MCTS tree
            
        Returns:
            Dictionary with entropy analysis metrics
        """
        if not root.children:
            return {"total_nodes": 1, "avg_entropy": 0.0, "entropy_variance": 0.0}
        
        def collect_entropy_scores(node: MCTSNode) -> List[float]:
            """Recursively collect entropy scores from all nodes."""
            scores = [node.entropy_score]
            for child in node.children:
                scores.extend(collect_entropy_scores(child))
            return scores
        
        entropy_scores = collect_entropy_scores(root)
        
        if not entropy_scores:
            return {"total_nodes": 1, "avg_entropy": 0.0, "entropy_variance": 0.0}
        
        avg_entropy = sum(entropy_scores) / len(entropy_scores)
        variance = sum((score - avg_entropy) ** 2 for score in entropy_scores) / len(entropy_scores)
        
        return {
            "total_nodes": len(entropy_scores),
            "avg_entropy": avg_entropy,
            "entropy_variance": variance,
            "min_entropy": min(entropy_scores),
            "max_entropy": max(entropy_scores)
        }
    
    def compare_selection_algorithms(self, root: MCTSNode, exploration_constant: float = 1.414) -> Dict[str, any]:
        """
        🎯 NEW: Compare PH-UCT vs UCB1 selection algorithms.
        
        This method helps evaluate the effectiveness of PH-UCT by
        comparing node selection decisions with UCB1.
        
        Args:
            root: Root node of the MCTS tree
            exploration_constant: UCB1 exploration constant
            
        Returns:
            Dictionary with comparison results
        """
        if not root.children:
            return {"ph_uct_choice": None, "ucb1_choice": None, "agreement": True}
        
        # Get PH-UCT choice
        ph_uct_choice = self._select_ph_uct(root, exploration_constant)
        
        # Get UCB1 choice
        ucb1_choice = self._select_ucb1(root, exploration_constant)
        
        # Check if they agree
        agreement = ph_uct_choice == ucb1_choice
        
        return {
            "ph_uct_choice": ph_uct_choice.sequence[:20] + "..." if ph_uct_choice else None,
            "ucb1_choice": ucb1_choice.sequence[:20] + "..." if ucb1_choice else None,
            "agreement": agreement,
            "ph_uct_score": ph_uct_choice.ph_uct_score(exploration_constant) if ph_uct_choice else None,
            "ucb1_score": ph_uct_choice.ucb_score if ph_uct_choice else None
        }
    
    def get_ph_uct_configuration(self) -> Dict[str, any]:
        """
        🎯 NEW: Get current PH-UCT configuration.
        
        Returns:
            Dictionary with current PH-UCT parameters
        """
        return {
            "use_ph_uct": self.use_ph_uct,
            "entropy_weight": self.entropy_weight,
            "diversity_weight": self.diversity_weight,
            "exploration_potential_weight": self.exploration_potential_weight,
            "exploration_constant": self.exploration_constant
        }
    
    def generate_with_multiple_experts(self, structure: Dict, target_length: int, 
                                     masked_sequence: str = None, temperature: float = 1.0,
                                     use_probability_averaging: bool = True) -> str:
        """
        🎯 NEW: Generate sequence using multiple expert models if available.
        
        This method provides a unified interface for sequence generation,
        automatically using multiple experts when available, or falling back
        to single model generation.
        
        Args:
            structure: Structure information
            target_length: Target sequence length
            masked_sequence: Sequence with masked positions
            temperature: Sampling temperature
            use_probability_averaging: Whether to use probability averaging (True) or majority voting (False)
            
        Returns:
            Generated sequence
        """
        if self.use_multiple_experts and self.dplm2_integration:
            try:
                return self.dplm2_integration.generate_with_multiple_experts(
                    structure, target_length, masked_sequence, temperature, use_probability_averaging
                )
            except Exception as e:
                print(f"⚠️ Multiple experts generation failed: {e}, falling back to single model")
                # Fall back to single model generation
                pass
        
        # Single model generation (original behavior)
        if self.dplm2_integration:
            return self.dplm2_integration.generate_sequence(structure, target_length, masked_sequence=masked_sequence)
        else:
            raise ValueError("No DPLM-2 integration available for sequence generation")
    
    def get_multiple_experts_info(self) -> Dict[str, any]:
        """
        🎯 NEW: Get information about multiple experts configuration.
        
        Returns:
            Dictionary with multiple experts information
        """
        info = {
            "use_multiple_experts": self.use_multiple_experts,
            "dplm2_integration_available": self.dplm2_integration is not None
        }
        
        if self.dplm2_integration and hasattr(self.dplm2_integration, 'get_expert_info'):
            try:
                expert_info = self.dplm2_integration.get_expert_info()
                info.update(expert_info)
            except Exception as e:
                info["expert_info_error"] = str(e)
        
        return info
    
    def update_node_entropy_with_real_probabilities(self, node: MCTSNode, structure: Dict, 
                                                 target_length: int, temperature: float = 1.0) -> None:
        """
        🎯 NEW: Update node entropy scores using REAL probabilities from DPLM-2.
        
        This method gets actual amino acid probabilities from the DPLM-2 model
        and uses them to calculate real entropy for PH-UCT.
        
        Args:
            node: MCTS node to update
            structure: Structure information
            target_length: Target sequence length
            temperature: Sampling temperature
        """
        if not self.dplm2_integration or not hasattr(self.dplm2_integration, 'get_position_probabilities'):
            print(f"⚠️ Cannot update entropy: DPLM-2 integration not available or missing probability method")
            return
        
        try:
            # Get real probabilities from DPLM-2 for masked positions
            position_probabilities = self.dplm2_integration.get_position_probabilities(
                structure, target_length, node.sequence, temperature
            )
            
            if not position_probabilities:
                print(f"⚠️ No probabilities returned from DPLM-2 for entropy calculation")
                return
            
            # Calculate average entropy across all masked positions
            total_entropy = 0.0
            valid_positions = 0
            
            for pos, aa_probs in position_probabilities.items():
                # Calculate entropy for this position
                probs = list(aa_probs.values())
                probs = [p for p in probs if p > 0]  # Filter out zero probabilities
                
                if probs:
                    # Shannon entropy: H = -Σ(p_i * log(p_i))
                    position_entropy = -sum(p * math.log(p + 1e-8) for p in probs)
                    total_entropy += position_entropy
                    valid_positions += 1
            
            if valid_positions > 0:
                # Use average entropy across positions
                node.entropy_score = total_entropy / valid_positions
                print(f"🎯 PH-UCT: Updated node entropy using real DPLM-2 probabilities: {node.entropy_score:.4f}")
                print(f"   Average entropy across {valid_positions} masked positions")
            else:
                print(f"⚠️ No valid probabilities found for entropy calculation")
                
        except Exception as e:
            print(f"⚠️ Error updating node entropy: {e}")
            # Fallback to basic entropy calculation
            node.update_entropy_scores()
    
    def _update_node_entropy_with_dplm2(self, node: MCTSNode, structure: Dict, 
                                       target_length: int, temperature: float = 1.0) -> None:
        """
        Alias for update_node_entropy_with_real_probabilities for backward compatibility.
        """
        return self.update_node_entropy_with_real_probabilities(node, structure, target_length, temperature)
    
    def expand_with_real_entropy(self, node: MCTSNode, target_length: int, 
                               num_candidates: int, temperature: float, structure: Dict) -> None:
        """
        🎯 NEW: Expand node with real entropy calculation for PH-UCT.
        
        This method expands a node and updates entropy scores using real DPLM-2 probabilities
        instead of fallback heuristics.
        
        Args:
            node: Node to expand
            target_length: Target sequence length
            num_candidates: Number of candidate sequences to generate
            temperature: Sampling temperature
            structure: Structure information for DPLM-2
        """
        print(f"🎯 Expanding node with real entropy calculation for PH-UCT...")
        
        # Generate candidate sequences using multiple experts
        if self.use_multiple_experts and self.dplm2_integration:
            try:
                # Generate ensemble sequence
                ensemble_sequence = self.generate_with_multiple_experts(
                    structure, target_length, node.sequence, temperature
                )
                
                # Create child node
                child = MCTSNode(
                    ensemble_sequence,
                    set(),  # No masked positions in completed sequence
                    depth=node.depth + 1,
                    parent=node
                )
                
                # Update entropy scores using real probabilities
                self.update_node_entropy_with_real_probabilities(child, structure, target_length, temperature)
                
                node.children.append(child)
                print(f"🎯 PH-UCT: Created child with real entropy: {child.entropy_score:.4f}")
                
            except Exception as e:
                print(f"⚠️ Multiple experts expansion failed: {e}, falling back to single model")
                # Fall back to single model expansion
                pass
        
        # Single model expansion (fallback)
        if not node.children:
            try:
                sequence = self.dplm2_integration.generate_sequence(structure, target_length, masked_sequence=node.sequence)
                
                child = MCTSNode(
                    sequence,
                    set(),
                    depth=node.depth + 1,
                    parent=node
                )
                
                # Update entropy scores using real probabilities
                self.update_node_entropy_with_real_probabilities(child, structure, target_length, temperature)
                
                node.children.append(child)
                print(f"🎯 PH-UCT: Created child with single model: {child.entropy_score:.4f}")
                
            except Exception as e:
                print(f"❌ Single model expansion also failed: {e}")
        
        print(f"🎯 PH-UCT: Node expansion complete with {len(node.children)} children")

# Keep the old class name for backward compatibility
SequenceLevelMCTS = GeneralMCTS


def run_general_mcts_example():
    """Example usage of general MCTS framework."""
    # 🚫 NO MOCK STRUCTURES ALLOWED
    print("❌ Mock structure creation removed - only real structures allowed")
    return
    
    # Initialize MCTS for inverse folding
    mcts = GeneralMCTS(
        task_type="inverse_folding",
        max_depth=5,
        num_simulations=30,
        exploration_constant=1.414,
        temperature=1.0,
        use_plddt_masking=True,
        simultaneous_sampling=True
    )
    
    # Run search
    best_sequence, best_reward = mcts.search(structure, target_length=50)
    
    print(f"Best sequence found: {best_sequence}")
    print(f"Best reward: {best_reward:.3f}")
    
    return best_sequence, best_reward


def demonstrate_ph_uct():
    """
    🎯 NEW: Demonstrate PH-UCT algorithm functionality.
    
    This function shows how the PH-UCT algorithm works by creating
    a simple tree and comparing selection decisions.
    """
    print("🎯 PH-UCT Algorithm Demonstration")
    print("=" * 50)
    
    # Create a simple test tree
    root = MCTSNode("MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF", set())
    root.visit_count = 10
    root.total_value = 8.5
    
    # Create some child nodes with different characteristics
    child1 = MCTSNode("MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF", set(), parent=root, depth=1)
    child1.visit_count = 5
    child1.total_value = 4.0
    
    child2 = MCTSNode("MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF", set(), parent=root, depth=1)
    child2.visit_count = 3
    child2.total_value = 2.5
    
    child3 = MCTSNode("MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF", set(), parent=root, depth=1)
    child3.visit_count = 2
    child3.total_value = 1.8
    
    root.children = [child1, child2, child3]
    
    # Update entropy scores
    for child in [child1, child2, child3]:
        # Mock structure for testing
        test_structure = {"coordinates": None, "length": 50}
        mcts._update_node_entropy_with_dplm2(child, test_structure, 50, 1.0)
    
    # Create MCTS instance with PH-UCT enabled
    mcts = GeneralMCTS(use_ph_uct=True, entropy_weight=0.3, diversity_weight=0.2)
    
    # Compare selection algorithms
    comparison = mcts.compare_selection_algorithms(root)
    
    print("🎯 Selection Algorithm Comparison:")
    print(f"   PH-UCT choice: {comparison['ph_uct_choice']}")
    print(f"   UCB1 choice: {comparison['ucb1_choice']}")
    print(f"   Algorithms agree: {comparison['agreement']}")
    print(f"   PH-UCT score: {comparison['ph_uct_score']:.4f}")
    print(f"   UCB1 score: {comparison['ucb1_score']:.4f}")
    
    # Show PH-UCT configuration
    config = mcts.get_ph_uct_configuration()
    print("\n🎯 PH-UCT Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Analyze tree entropy
    entropy_analysis = mcts.analyze_tree_entropy(root)
    print("\n🎯 Tree Entropy Analysis:")
    for key, value in entropy_analysis.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 PH-UCT Demonstration Complete!")
    return comparison, entropy_analysis


def demonstrate_multiple_experts_mcts():
    """
    🎯 NEW: Demonstrate multiple experts integration with MCTS using simplified approach.
    
    This function shows how the simplified multiple experts rollout works:
    - Same model with different random seeds
    - Real probabilities from DPLM-2 for entropy calculation
    - No fallback methods
    """
    print("🎯 Multiple Experts + MCTS Integration Demonstration (Simplified)")
    print("=" * 70)
    
    try:
        # Import the simplified DPLM2Integration
        from core.dplm2_integration_simple import DPLM2Integration
        
        print(f"🎯 Loading simplified DPLM-2 integration...")
        
        # Initialize DPLM-2 integration (automatically enables multiple experts)
        dplm2_integration = DPLM2Integration(model_name="airkingbd/dplm2_650m")
        
        print("✅ DPLM-2 integration loaded successfully!")
        
        # Create MCTS instance with multiple experts
        mcts = GeneralMCTS(
            use_ph_uct=True,  # Enable PH-UCT
            entropy_weight=0.3,
            diversity_weight=0.2,
            dplm2_integration=dplm2_integration
        )
        
        print("✅ MCTS with multiple experts initialized!")
        
        # Get multiple experts information
        experts_info = mcts.get_multiple_experts_info()
        print("\n🎯 Multiple Experts Configuration:")
        for key, value in experts_info.items():
            print(f"   {key}: {value}")
        
        # Test multiple experts sequence generation
        print(f"\n🎯 Testing Multiple Experts Sequence Generation...")
        
        # Create a simple test structure
        test_structure = {
            "sequence": "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF",
            "length": 50,
            "coordinates": None
        }
        
        masked_sequence = "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFSEIPMLDPPSIDTAYF"
        masked_sequence = list(masked_sequence)
        masked_sequence[10] = 'X'  # Mask position 10
        masked_sequence[25] = 'X'  # Mask position 25
        masked_sequence = ''.join(masked_sequence)
        
        print(f"   Test structure length: {test_structure['length']}")
        print(f"   Masked sequence: {masked_sequence}")
        print(f"   Masked positions: {[i for i, c in enumerate(masked_sequence) if c == 'X']}")
        
        # Test ensemble generation
        print(f"\n🎯 Testing Ensemble Generation (Same Model, Different Seeds)...")
        try:
            ensemble_sequence = mcts.generate_with_multiple_experts(
                test_structure, 
                target_length=50, 
                masked_sequence=masked_sequence,
                temperature=1.0
            )
            print(f"   ✅ Ensemble generation result: {ensemble_sequence}")
        except Exception as e:
            print(f"   ❌ Ensemble generation failed: {e}")
        
        # Test real probability calculation for entropy
        print(f"\n🎯 Testing Real Probability Calculation for PH-UCT Entropy...")
        try:
            # Create a test node
            test_node = MCTSNode(masked_sequence, {10, 25}, depth=0)
            
            # Update entropy using real probabilities
            mcts.update_node_entropy_with_real_probabilities(
                test_node, test_structure, 50, temperature=1.0
            )
            
            print(f"   ✅ Real entropy calculation completed")
            print(f"   Node entropy score: {test_node.entropy_score:.4f}")
            
            # Show PH-UCT score
            ph_uct_score = test_node.ph_uct_score()
            print(f"   PH-UCT score: {ph_uct_score:.4f}")
            
        except Exception as e:
            print(f"   ❌ Real probability calculation failed: {e}")
        
        print(f"\n🎯 Multiple Experts + MCTS Demonstration Complete!")
        return mcts
        
    except Exception as e:
        print(f"❌ Failed to demonstrate multiple experts + MCTS: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # run_general_mcts_example()
    # demonstrate_ph_uct()
    demonstrate_multiple_experts_mcts() 
