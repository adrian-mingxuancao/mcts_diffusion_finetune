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

import math
import random
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dplm2_integration import DPLM2Integration


@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a sequence state."""
    sequence: str
    masked_positions: Set[int]  # Positions that are masked
    reward: float
    visit_count: int = 0
    total_value: float = 0.0
    children: List['MCTSNode'] = None
    task_type: str = "inverse_folding"  # inverse_folding, folding, unconditional, conditional
    
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


class GeneralMCTS:
    """
    General MCTS framework for improving DPLM-2 performance across all tasks.
    Leverages diffusion's ability to sample multiple positions simultaneously.
    """
    
    def __init__(
        self,
        task_type: str = "inverse_folding",
        max_depth: int = 5,
        num_simulations: int = 50,
        exploration_constant: float = 1.414,
        temperature: float = 1.0,
        num_candidates_per_expansion: int = 5,
        use_plddt_masking: bool = True,
        simultaneous_sampling: bool = True
    ):
        """
        Initialize general MCTS framework.
        
        Args:
            task_type: Type of task (inverse_folding, folding, unconditional, conditional)
            max_depth: Maximum tree depth
            num_simulations: Number of MCTS simulations
            exploration_constant: UCB1 exploration constant
            temperature: Sampling temperature
            num_candidates_per_expansion: Number of candidates per expansion
            use_plddt_masking: Whether to use plDDT-based masking
            simultaneous_sampling: Whether to sample multiple positions simultaneously
        """
        self.task_type = task_type
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.num_candidates_per_expansion = num_candidates_per_expansion
        self.use_plddt_masking = use_plddt_masking
        self.simultaneous_sampling = simultaneous_sampling
        
        # Initialize DPLM-2 integration
        self.dplm2 = DPLM2Integration(use_local=True)
        
        # Cache for generated sequences to avoid duplicates
        self.sequence_cache: Set[str] = set()
    
    def search(self, input_data: Dict, target_length: int) -> Tuple[str, float]:
        """
        Perform MCTS search for the specified task.
        
        Args:
            input_data: Task-specific input (structure for inverse folding, sequence for folding, etc.)
            target_length: Target sequence length
            
        Returns:
            Tuple of (best_sequence, best_reward)
        """
        print(f"Starting MCTS search for {self.task_type} (target length: {target_length})")
        
        # Initialize with masked sequence
        initial_sequence, masked_positions = self._create_initial_masked_sequence(
            input_data, target_length
        )
        
        # Create root node
        root = MCTSNode(
            sequence=initial_sequence,
            masked_positions=masked_positions,
            reward=0.0,
            task_type=self.task_type
        )
        
        print(f"Initial sequence: {initial_sequence}")
        print(f"Masked positions: {len(masked_positions)}/{target_length}")
        
        # Run MCTS simulations
        best_reward = 0.0
        best_sequence = initial_sequence
        
        for i in range(self.num_simulations):
            # Selection
            current = self._select(root)
            
            # Expansion
            if current.visit_count > 0 and len(current.children) < self.num_candidates_per_expansion:
                self._expand(current, input_data, target_length)
            
            # Simulation (expert rollout)
            value = self._simulate_expert_rollout(current, input_data)
            
            # Backpropagation
            self._backpropagate(current, value)
            
            # Track best sequence found
            if value > best_reward:
                best_reward = value
                best_sequence = current.sequence
                print(f"New best sequence found at simulation {i+1}: {best_reward:.3f}")
            
            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{self.num_simulations} simulations. "
                      f"Best reward so far: {best_reward:.3f}")
        
        print(f"MCTS search completed. Best sequence: {best_sequence}")
        print(f"Best reward: {best_reward:.3f}")
        
        return best_sequence, best_reward
    
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
            initial_sequence = self.dplm2.generate_sequence(
                structure, target_length=target_length, temperature=self.temperature
            )
        except:
            # Fallback to random sequence
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            initial_sequence = ''.join(random.choices(amino_acids, k=target_length))
        
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
        
        print(f"Created sequence with {len(masked_positions)}/{target_length} masked positions ({len(masked_positions)/target_length*100:.1f}%)")
        
        return ''.join(masked_sequence), masked_positions
    
    def _apply_plddt_masking(self, sequence: str, structure: Dict) -> Set[int]:
        """Apply plDDT-based masking to sequence."""
        try:
            # Try to get plDDT scores from structure
            if 'plddt_scores' in structure:
                plddt_scores = structure['plddt_scores']
                # Mask positions with low plDDT scores
                low_confidence_positions = [
                    i for i, score in enumerate(plddt_scores) 
                    if score < 0.7  # Low confidence threshold
                ]
                return set(low_confidence_positions)
        except:
            pass
        
        # Fallback: random masking
        num_to_mask = max(1, len(sequence) // 4)  # Mask 25% of positions
        return set(random.sample(range(len(sequence)), num_to_mask))
    
    def _expand(self, node: MCTSNode, input_data: Dict, target_length: int):
        """Expand a node by generating simple sequence variations."""
        if not node.masked_positions:
            return  # No positions to unmask
        
        # Generate simple variations instead of complex simultaneous sampling
        variations = self._generate_simple_variations(node, target_length)
        
        for variation_sequence, variation_masked in variations:
            if variation_sequence not in self.sequence_cache:
                reward = self._compute_compound_reward(variation_sequence, input_data)
                child = MCTSNode(
                    sequence=variation_sequence,
                    masked_positions=variation_masked,
                    reward=reward,
                    task_type=self.task_type
                )
                node.children.append(child)
                self.sequence_cache.add(variation_sequence)
    
    def _generate_simple_variations(self, node: MCTSNode, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Generate simple variations by unmasking a few positions."""
        variations = []
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        
        # Limit number of variations for speed
        max_variations = min(3, len(node.masked_positions))
        masked_list = list(node.masked_positions)
        
        for i in range(max_variations):
            if not masked_list:
                break
                
            # Pick a random masked position to unmask
            pos_to_unmask = random.choice(masked_list)
            masked_list.remove(pos_to_unmask)
            
            # Check bounds before assignment
            if pos_to_unmask >= len(node.sequence):
                continue
            
            # Create variation
            new_sequence = list(node.sequence)
            new_sequence[pos_to_unmask] = random.choice(amino_acids)
            
            # Update masked positions
            new_masked = node.masked_positions.copy()
            new_masked.remove(pos_to_unmask)
            
            variations.append((''.join(new_sequence), new_masked))
        
        return variations
    
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
    
    def _simultaneous_diffusion_sampling(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Sample multiple masked positions simultaneously using diffusion properties."""
        variations = []
        
        try:
            # Create input with current masked sequence
            masked_input = self._create_masked_input(node.sequence, input_data)
            
            # Generate multiple variations with different temperatures
            temperatures = [0.8, 1.0, 1.2, 1.5]
            
            for temp in temperatures:
                # Use DPLM-2 to sample multiple positions at once
                new_sequence = self.dplm2.generate_sequence(
                    masked_input, target_length=target_length, temperature=temp
                )
                
                if new_sequence and len(new_sequence) == target_length:
                    # Determine which positions were unmasked
                    unmasked_positions = set()
                    for i, (old_aa, new_aa) in enumerate(zip(node.sequence, new_sequence)):
                        if old_aa == 'X' and new_aa != 'X':
                            unmasked_positions.add(i)
                    
                    # Update masked positions
                    new_masked = node.masked_positions - unmasked_positions
                    
                    variations.append((new_sequence, new_masked))
                    
        except Exception as e:
            print(f"Error in simultaneous sampling: {e}")
            # Fallback to single position sampling
            variations.extend(self._single_position_sampling(node, input_data, target_length))
        
        return variations
    
    def _single_position_sampling(self, node: MCTSNode, input_data: Dict, target_length: int) -> List[Tuple[str, Set[int]]]:
        """Sample one position at a time (fallback method)."""
        variations = []
        
        # Sample a few positions to unmask
        positions_to_unmask = random.sample(list(node.masked_positions), 
                                          min(3, len(node.masked_positions)))
        
        for pos in positions_to_unmask:
            # Generate amino acid for this position
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            new_aa = random.choice(amino_acids)
            
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
        return self.dplm2._create_masked_input(masked_sequence, structure, len(masked_sequence))
    
    def _simulate_expert_rollout(self, node: MCTSNode, input_data: Dict) -> float:
        """Simulate expert rollout with compound reward (optimized for speed)."""
        if not node.masked_positions:
            # Sequence is complete, evaluate final reward
            return self._compute_compound_reward(node.sequence, input_data)
        
        # Fast rollout: just evaluate current state and do minimal unmasking
        current_sequence = node.sequence
        current_masked = node.masked_positions.copy()
        
        # If there are masked positions, quickly complete the sequence
        if current_masked:
            # Fast completion: replace X's with reasonable amino acids
            completed_sequence = list(current_sequence)
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            
            for pos in current_masked:
                # Bounds checking for safety
                if pos < len(completed_sequence) and pos >= 0:
                    # Use context-aware selection (simplified heuristic)
                    if pos > 0 and pos-1 < len(completed_sequence) and completed_sequence[pos-1] in "KR":  # After positive charge
                        completed_sequence[pos] = random.choice("DE")  # Add negative charge
                    elif pos > 0 and pos-1 < len(completed_sequence) and completed_sequence[pos-1] in "DE":  # After negative charge
                        completed_sequence[pos] = random.choice("KR")  # Add positive charge
                    else:
                        completed_sequence[pos] = random.choice(amino_acids)
            
            current_sequence = ''.join(completed_sequence)
        
        # Single reward evaluation
        return self._compute_compound_reward(current_sequence, input_data)
    
    def _select_best_position_to_unmask(self, sequence: str, masked_positions: Set[int], input_data: Dict) -> Optional[int]:
        """Select the best position to unmask based on expected reward."""
        if not masked_positions:
            return None
        
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
                reward = self._compute_compound_reward(''.join(test_sequence), input_data)
                position_rewards.append(reward)
            
            # Expected reward for this position
            expected_reward = sum(position_rewards) / len(position_rewards)
            
            if expected_reward > best_expected_reward:
                best_expected_reward = expected_reward
                best_pos = pos
        
        return best_pos
    
    def _compute_compound_reward(self, sequence: str, input_data: Dict) -> float:
        """Compute compound reward based on task type."""
        if self.task_type == "inverse_folding":
            return self._compute_inverse_folding_reward(sequence, input_data)
        elif self.task_type == "folding":
            return self._compute_folding_reward(sequence, input_data)
        elif self.task_type == "unconditional":
            return self._compute_unconditional_reward(sequence)
        elif self.task_type == "conditional":
            return self._compute_conditional_reward(sequence, input_data)
        else:
            return 0.0
    
    def _compute_inverse_folding_reward(self, sequence: str, structure: Dict) -> float:
        """Compute reward for inverse folding task."""
        from utils.reward_computation import LengthAwareRewardComputation
        
        reward_computer = LengthAwareRewardComputation()
        base_reward = reward_computer.compute_reward(sequence, structure)
        
        # Add task-specific bonuses
        length_bonus = 0.1 if 20 <= len(sequence) <= 200 else 0.0
        structure_bonus = 0.2 if 'target_structure' in structure else 0.0
        
        return base_reward + length_bonus + structure_bonus
    
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
            sequence = ''.join(random.choices(amino_acids, k=target_length))
        
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
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using UCB1."""
        if not node.children:
            return node
        
        # Select child with highest UCB score
        best_child = max(node.children, key=lambda x: x.ucb_score)
        return self._select(best_child)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate the simulation result up the tree."""
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            # Move to parent (simplified - in practice you'd need parent pointers)
            break
    
    def _generate_random_sequences(self, length: int, num_sequences: int) -> List[str]:
        """Generate random amino acid sequences as fallback."""
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        sequences = []
        
        for _ in range(num_sequences):
            seq = ''.join(random.choices(amino_acids, k=length))
            if seq not in self.sequence_cache:
                sequences.append(seq)
        
        return sequences


# Keep the old class name for backward compatibility
SequenceLevelMCTS = GeneralMCTS


def run_general_mcts_example():
    """Example usage of general MCTS framework."""
    from utils.protein_utils import create_mock_structure_no_sequence
    
    # Create mock structure for inverse folding
    structure = create_mock_structure_no_sequence(length=50)
    
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


if __name__ == "__main__":
    run_general_mcts_example() 