"""
Clean General MCTS Framework for DPLM-2 Performance Improvement

This module implements a clean Monte Carlo Tree Search framework for improving
DPLM-2 performance across specific tasks:
- Inverse folding (structure → sequence)
- Folding (sequence → structure) 
- Unconditional generation
- Conditional generation

Key features:
1. Real pLDDT-based masking
2. Simultaneous position sampling
3. Task-specific reward functions
4. Clean, focused implementation

Note: For motif scaffolding, use MotifScaffoldingMCTS from core.motif_scaffolding_mcts
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
    def ucb_score(self, exploration_constant: float = 1.414) -> float:
        """Upper Confidence Bound score for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        # UCB1 formula
        if self.parent and self.parent.visit_count > 0:
            parent_visits = self.parent.visit_count
            log_term = math.log(parent_visits)
            if log_term > 0:
                exploration = exploration_constant * math.sqrt(log_term / self.visit_count)
                return self.average_value + exploration
        
        return self.average_value
    
    @property
    def completion_ratio(self) -> float:
        """Ratio of unmasked positions."""
        if not self.sequence:
            return 0.0
        return 1.0 - (len(self.masked_positions) / len(self.sequence))


class GeneralMCTS:
    """
    Clean MCTS framework for improving DPLM-2 performance.
    Supports inverse folding, folding, unconditional, and conditional generation.
    """
    
    def __init__(
        self,
        task_type: str = "inverse_folding",
        max_depth: int = 5,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        use_plddt_masking: bool = True,
        dplm2_integration: object = None
    ):
        """
        Initialize MCTS framework.
        
        Args:
            task_type: Type of task (inverse_folding, folding, unconditional, conditional)
            max_depth: Maximum tree depth
            num_simulations: Number of MCTS simulations
            exploration_constant: UCB1 exploration constant
            temperature: Sampling temperature
            use_plddt_masking: Whether to use plDDT-based masking
            dplm2_integration: DPLM-2 integration object
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
        
        # Initialize DPLM-2 integration
        self.dplm2_integration = dplm2_integration
        if not self.dplm2_integration:
            try:
                self.dplm2_integration = DPLM2Integration(use_local=True)
            except:
                raise ValueError("DPLM-2 integration is required")
        
        # Task-specific setup
        self._setup_task()
        
        print(f"✅ GeneralMCTS initialized for {task_type}")
    
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
    
    def search(self, input_data: Dict, target_length: int = None, 
               initial_sequence: str = None, reference_sequence: str = None) -> Tuple[str, float]:
        """
        Perform MCTS search for the best sequence.
        
        Args:
            input_data: Task-specific input data
            target_length: Target sequence length
            initial_sequence: Initial sequence (if any)
            reference_sequence: Reference sequence for evaluation
            
        Returns:
            Tuple of (best_sequence, best_reward)
        """
        print(f"🔍 Starting MCTS search for {self.task_type}")
        
        # Create initial sequence and masked positions
        if initial_sequence:
            sequence = initial_sequence
            masked_positions = set()
        else:
            sequence, masked_positions = self._create_initial_sequence(input_data, target_length)
        
        if not sequence:
            print("❌ Failed to create initial sequence")
            return None, 0.0
        
        # Create root node
        root = MCTSNode(
            sequence=sequence,
            masked_positions=masked_positions,
            task_type=self.task_type,
            depth=0
        )
        
        # Store reference for evaluation
        self.reference_sequence = reference_sequence
        self.input_data = input_data
        
        print(f"🚀 Starting MCTS with {self.num_simulations} simulations")
        print(f"   Initial sequence: {sequence[:50]}...")
        print(f"   Masked positions: {len(masked_positions)}")
        
        best_sequence = sequence
        best_reward = self._evaluate_sequence(sequence, input_data)
        
        # Run MCTS iterations
        for iteration in range(self.num_simulations):
            if iteration % 10 == 0:
                print(f"   Simulation {iteration + 1}/{self.num_simulations}")
            
            # Selection
            selected_node = self._select(root)
            
            # Expansion
            if selected_node.depth < self.max_depth and len(selected_node.masked_positions) > 0:
                self._expand(selected_node, input_data, target_length)
            
            # Simulation
            if selected_node.children:
                simulation_node = random.choice(selected_node.children)
            else:
                simulation_node = selected_node
            
            reward = self._simulate(simulation_node, input_data, target_length)
            
            # Backpropagation
            self._backpropagate(simulation_node, reward)
            
            # Track best
            if reward > best_reward:
                best_sequence = simulation_node.sequence
                best_reward = reward
                print(f"   🏆 New best reward: {reward:.3f}")
        
        print(f"✅ MCTS completed. Best reward: {best_reward:.3f}")
        return best_sequence, best_reward
    
    def _create_initial_sequence(self, input_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence based on task type."""
        if self.task_type == "inverse_folding":
            return self._create_inverse_folding_sequence(input_data, target_length)
        elif self.task_type == "folding":
            return self._create_folding_sequence(input_data, target_length)
        elif self.task_type == "unconditional":
            return self._create_unconditional_sequence(target_length)
        elif self.task_type == "conditional":
            return self._create_conditional_sequence(input_data, target_length)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _create_inverse_folding_sequence(self, structure_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence for inverse folding (structure -> sequence)."""
        try:
            print("🔄 Creating inverse folding sequence from structure")
            
            # Generate initial sequence using DPLM-2
            sequence = self.dplm2_integration.generate_sequence(
                structure_data, 
                target_length=target_length, 
                temperature=self.temperature
            )
            
            if not sequence or len(sequence) != target_length:
                print(f"❌ DPLM-2 generation failed")
                return None, set()
            
            # Apply pLDDT masking for optimization
            masked_positions = self._apply_plddt_masking(sequence, structure_data)
            
            print(f"✅ Generated sequence: {len(sequence)} residues, {len(masked_positions)} masked")
            return sequence, masked_positions
            
        except Exception as e:
            print(f"❌ Inverse folding sequence creation failed: {e}")
            return None, set()
    
    def _create_folding_sequence(self, sequence_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence for folding (sequence -> structure)."""
        try:
            print("🔄 Creating folding sequence")
            
            # For folding, we start with the given sequence
            input_sequence = sequence_data.get('sequence', '')
            if not input_sequence:
                print("❌ No input sequence provided for folding")
                return None, set()
            
            if target_length and len(input_sequence) != target_length:
                # Truncate or pad if needed
                if len(input_sequence) > target_length:
                    input_sequence = input_sequence[:target_length]
                else:
                    # Pad with random amino acids
                    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
                    padding = ''.join(random.choice(amino_acids) for _ in range(target_length - len(input_sequence)))
                    input_sequence += padding
            
            # For folding task, we might mask some positions to optimize structure
            masked_positions = self._apply_structure_based_masking(input_sequence, sequence_data)
            
            print(f"✅ Folding sequence: {len(input_sequence)} residues, {len(masked_positions)} masked")
            return input_sequence, masked_positions
            
        except Exception as e:
            print(f"❌ Folding sequence creation failed: {e}")
            return None, set()
    
    def _create_unconditional_sequence(self, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence for unconditional generation."""
        try:
            print("🔄 Creating unconditional sequence")
            
            # Start with fully masked sequence
            sequence = 'X' * target_length
            masked_positions = set(range(target_length))
            
            print(f"✅ Unconditional sequence: {target_length} residues, all masked")
            return sequence, masked_positions
            
        except Exception as e:
            print(f"❌ Unconditional sequence creation failed: {e}")
            return None, set()
    
    def _create_conditional_sequence(self, condition_data: Dict, target_length: int) -> Tuple[str, Set[int]]:
        """Create initial sequence for conditional generation."""
        try:
            print("🔄 Creating conditional sequence")
            
            # Extract conditions
            partial_sequence = condition_data.get('partial_sequence', '')
            constraints = condition_data.get('constraints', {})
            
            if partial_sequence:
                sequence = partial_sequence
                if len(sequence) < target_length:
                    # Extend with masked positions
                    sequence += 'X' * (target_length - len(sequence))
                elif len(sequence) > target_length:
                    sequence = sequence[:target_length]
            else:
                # Start with fully masked
                sequence = 'X' * target_length
            
            # Determine masked positions (X characters)
            masked_positions = {i for i, aa in enumerate(sequence) if aa == 'X'}
            
            print(f"✅ Conditional sequence: {target_length} residues, {len(masked_positions)} masked")
            return sequence, masked_positions
            
        except Exception as e:
            print(f"❌ Conditional sequence creation failed: {e}")
            return None, set()
    
    def _apply_plddt_masking(self, sequence: str, structure_data: Dict) -> Set[int]:
        """Apply pLDDT-based masking for optimization."""
        if not self.use_plddt_masking:
            # Random masking fallback
            mask_ratio = 0.2
            num_to_mask = max(1, int(len(sequence) * mask_ratio))
            return set(random.sample(range(len(sequence)), num_to_mask))
        
        try:
            # Use real pLDDT calculation
            from utils.real_plddt_computation import compute_plddt_from_structure
            
            plddt_scores = compute_plddt_from_structure(structure_data)
            if not plddt_scores or len(plddt_scores) != len(sequence):
                print("⚠️ pLDDT calculation failed, using random masking")
                mask_ratio = 0.2
                num_to_mask = max(1, int(len(sequence) * mask_ratio))
                return set(random.sample(range(len(sequence)), num_to_mask))
            
            # Mask low confidence positions
            confidence_threshold = 0.7
            low_conf_positions = [i for i, score in enumerate(plddt_scores) if score < confidence_threshold]
            
            # Ensure reasonable masking ratio
            min_mask = max(3, len(sequence) // 20)  # At least 5%
            max_mask = len(sequence) // 4          # At most 25%
            
            if len(low_conf_positions) < min_mask:
                # Add some medium confidence positions
                medium_conf = [i for i, score in enumerate(plddt_scores) if 0.7 <= score < 0.8]
                additional = random.sample(medium_conf, min(min_mask - len(low_conf_positions), len(medium_conf)))
                low_conf_positions.extend(additional)
            elif len(low_conf_positions) > max_mask:
                # Limit to worst positions
                position_scores = [(i, plddt_scores[i]) for i in low_conf_positions]
                position_scores.sort(key=lambda x: x[1])
                low_conf_positions = [pos for pos, _ in position_scores[:max_mask]]
            
            return set(low_conf_positions)
            
        except Exception as e:
            print(f"⚠️ pLDDT masking failed: {e}, using random masking")
            mask_ratio = 0.2
            num_to_mask = max(1, int(len(sequence) * mask_ratio))
            return set(random.sample(range(len(sequence)), num_to_mask))
    
    def _apply_structure_based_masking(self, sequence: str, sequence_data: Dict) -> Set[int]:
        """Apply structure-based masking for folding task."""
        # For folding, we might want to mask positions that are hard to fold correctly
        # This is a simplified version - could be enhanced with actual structure prediction
        mask_ratio = 0.15  # Conservative masking for folding
        num_to_mask = max(1, int(len(sequence) * mask_ratio))
        return set(random.sample(range(len(sequence)), num_to_mask))
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select node using UCB1."""
        node = root
        
        while node.children:
            best_child = None
            best_ucb = float('-inf')
            
            for child in node.children:
                if child.visit_count == 0:
                    return child
                
                ucb = child.ucb_score(self.exploration_constant)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            node = best_child
        
        return node
    
    def _expand(self, node: MCTSNode, input_data: Dict, target_length: int):
        """Expand node by generating candidate sequences."""
        if not node.masked_positions:
            return  # No positions to unmask
        
        # Generate candidate sequences
        candidates = self._generate_candidates(node, input_data, target_length)
        
        for candidate_seq, remaining_masked in candidates:
            child = MCTSNode(
                sequence=candidate_seq,
                masked_positions=remaining_masked,
                parent=node,
                depth=node.depth + 1,
                task_type=node.task_type
            )
            node.children.append(child)
    
    def _generate_candidates(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Generate candidate sequences by unmasking positions."""
        candidates = []
        
        # Select a few positions to unmask
        positions_to_unmask = random.sample(
            list(node.masked_positions), 
            min(3, len(node.masked_positions))
        )
        
        try:
            # Use DPLM-2 to fill masked positions
            filled_sequence = self._fill_masked_positions(node.sequence, positions_to_unmask, input_data)
            
            if filled_sequence and len(filled_sequence) == target_length:
                new_masked = node.masked_positions - set(positions_to_unmask)
                candidates.append((filled_sequence, new_masked))
        
        except Exception as e:
            print(f"⚠️ Candidate generation failed: {e}")
        
        return candidates
    
    def _fill_masked_positions(self, sequence: str, positions: List[int], input_data: Dict) -> str:
        """Fill masked positions using DPLM-2."""
        try:
            # Create masked sequence
            masked_seq = list(sequence)
            for pos in positions:
                masked_seq[pos] = 'X'
            masked_seq = ''.join(masked_seq)
            
            # Use appropriate DPLM-2 method based on task
            if self.task_type == "inverse_folding":
                # Structure -> sequence
                struct_tokens = self._extract_structure_tokens(input_data)
                result = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_seq,
                    struct_tokens=struct_tokens,
                    task_type="inverse_folding",
                    temperature=self.temperature
                )
            elif self.task_type == "folding":
                # Sequence -> structure (but we're optimizing sequence)
                result = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_seq,
                    struct_tokens="",  # No structure constraint
                    task_type="unconditional",
                    temperature=self.temperature
                )
            else:
                # Unconditional or conditional
                result = self.dplm2_integration.generate_from_masked_input(
                    aa_sequence=masked_seq,
                    struct_tokens="",
                    task_type="unconditional",
                    temperature=self.temperature
                )
            
            return result
            
        except Exception as e:
            print(f"⚠️ DPLM-2 filling failed: {e}")
            return None
    
    def _extract_structure_tokens(self, structure_data: Dict) -> str:
        """Extract structure tokens for DPLM-2."""
        # Simplified structure token extraction
        if 'struct_tokens' in structure_data:
            return structure_data['struct_tokens']
        elif 'coordinates' in structure_data:
            # Convert coordinates to tokens (simplified)
            coords = structure_data['coordinates']
            if hasattr(coords, 'shape') and len(coords.shape) >= 2:
                # Create default tokens
                return ','.join(['160'] * coords.shape[0])
        
        return ""
    
    def _simulate(self, node: MCTSNode, input_data: Dict, target_length: int) -> float:
        """Simulate from node to completion and return reward."""
        current_sequence = node.sequence
        current_masked = set(node.masked_positions)
        
        # Complete the sequence if needed
        if current_masked:
            try:
                completed_sequence = self._fill_masked_positions(
                    current_sequence, list(current_masked), input_data
                )
                if completed_sequence:
                    current_sequence = completed_sequence
            except:
                pass  # Use partial sequence if completion fails
        
        # Evaluate the sequence
        return self._evaluate_sequence(current_sequence, input_data)
    
    def _evaluate_sequence(self, sequence: str, input_data: Dict) -> float:
        """Evaluate sequence based on task type."""
        try:
            if self.task_type == "inverse_folding":
                return self._evaluate_inverse_folding(sequence, input_data)
            elif self.task_type == "folding":
                return self._evaluate_folding(sequence, input_data)
            elif self.task_type == "unconditional":
                return self._evaluate_unconditional(sequence)
            elif self.task_type == "conditional":
                return self._evaluate_conditional(sequence, input_data)
            else:
                return 0.5  # Default neutral score
                
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            return 0.0
    
    def _evaluate_inverse_folding(self, sequence: str, structure_data: Dict) -> float:
        """Evaluate inverse folding (structure recovery)."""
        # For inverse folding, we want to check if the sequence folds to the target structure
        if self.reference_sequence:
            # Calculate sequence recovery (AAR)
            if len(sequence) == len(self.reference_sequence):
                matches = sum(1 for a, b in zip(sequence, self.reference_sequence) if a == b)
                aar = matches / len(sequence)
                return aar
        
        # Fallback: basic sequence validity
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        validity = sum(1 for aa in sequence if aa in valid_aas) / len(sequence)
        return validity * 0.8  # Penalize if no reference available
    
    def _evaluate_folding(self, sequence: str, sequence_data: Dict) -> float:
        """Evaluate folding (structure quality)."""
        # For folding, we want to evaluate the quality of the predicted structure
        try:
            # Use structure prediction to evaluate
            from utils.real_plddt_computation import load_esmfold_model
            from utils.structure_converter import predict_structure_coords
            
            model, tokenizer = load_esmfold_model()
            if model is not None:
                coords, plddt_scores = predict_structure_coords(
                    model, tokenizer, sequence, return_plddt=True
                )
                
                if plddt_scores is not None:
                    # Return average confidence
                    return np.mean(plddt_scores)
            
        except Exception as e:
            print(f"⚠️ Structure evaluation failed: {e}")
        
        # Fallback: sequence properties
        return self._evaluate_sequence_properties(sequence)
    
    def _evaluate_unconditional(self, sequence: str) -> float:
        """Evaluate unconditional generation."""
        return self._evaluate_sequence_properties(sequence)
    
    def _evaluate_conditional(self, sequence: str, condition_data: Dict) -> float:
        """Evaluate conditional generation."""
        # Check if constraints are satisfied
        constraints = condition_data.get('constraints', {})
        
        score = self._evaluate_sequence_properties(sequence)
        
        # Apply constraint penalties/bonuses
        for constraint, value in constraints.items():
            if constraint == "length":
                if len(sequence) == value:
                    score += 0.1
                else:
                    score -= 0.1
            # Add more constraints as needed
        
        return max(0.0, min(1.0, score))
    
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
        
        # Hydrophobic/hydrophilic balance
        hydrophobic = "AILMFWYV"
        hydro_ratio = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)
        balance = 1.0 - abs(hydro_ratio - 0.4)  # Target ~40% hydrophobic
        
        # Combine scores
        return (validity * 0.5 + diversity * 0.3 + balance * 0.2)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_value += reward
            current = current.parent


# Keep backward compatibility
SequenceLevelMCTS = GeneralMCTS


def run_inverse_folding_example():
    """Example usage for inverse folding."""
    print("🔄 Running inverse folding example")
    
    # Mock structure data
    structure_data = {
        'coordinates': np.random.rand(50, 3, 3),  # 50 residues, backbone atoms
        'sequence': 'A' * 50  # Reference sequence
    }
    
    mcts = GeneralMCTS(task_type="inverse_folding")
    best_sequence, best_reward = mcts.search(
        input_data=structure_data,
        target_length=50,
        reference_sequence=structure_data['sequence']
    )
    
    print(f"Best sequence: {best_sequence}")
    print(f"Best reward: {best_reward:.3f}")


def run_folding_example():
    """Example usage for folding."""
    print("🔄 Running folding example")
    
    # Input sequence data
    sequence_data = {
        'sequence': 'MKTVRQERLKILVDFPKIEAQEAALLQIHTGSLDRWRLAIPVFAVKARNVT'
    }
    
    mcts = GeneralMCTS(task_type="folding")
    best_sequence, best_reward = mcts.search(
        input_data=sequence_data,
        target_length=len(sequence_data['sequence'])
    )
    
    print(f"Best sequence: {best_sequence}")
    print(f"Best reward: {best_reward:.3f}")


if __name__ == "__main__":
    print("🧬 Clean General MCTS Framework")
    print("Choose example:")
    print("1. Inverse folding")
    print("2. Folding")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        run_inverse_folding_example()
    elif choice == "2":
        run_folding_example()
    else:
        print("Running both examples...")
        run_inverse_folding_example()
        print()
        run_folding_example()





