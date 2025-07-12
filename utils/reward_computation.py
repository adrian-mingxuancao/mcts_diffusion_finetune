"""
Enhanced reward computation for MCTS-guided inverse folding.

This module provides sophisticated, length-aware reward functions for proteins
of different sizes, taking into account biophysical properties and structural
constraints that vary with protein length.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import math
from collections import Counter


class LengthAwareRewardComputation:
    """
    Sophisticated reward computation that adapts to protein length.
    
    Different protein sizes have different optimization priorities:
    - Small proteins (<100): Focus on local structure and stability
    - Medium proteins (100-300): Balance local and global properties  
    - Large proteins (>300): Emphasize global structure compatibility
    """
    
    def __init__(self):
        # Amino acid properties
        self.hydrophobicity_scores = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        self.charge_scores = {
            'R': +1, 'K': +1, 'H': +0.5,  # Positive
            'D': -1, 'E': -1,              # Negative
        }
        
        # Default amino acid frequencies in natural proteins
        self.natural_frequencies = {
            'A': 0.082, 'R': 0.055, 'N': 0.041, 'D': 0.054, 'C': 0.013,
            'E': 0.062, 'Q': 0.039, 'G': 0.072, 'H': 0.022, 'I': 0.053,
            'L': 0.096, 'K': 0.058, 'M': 0.024, 'F': 0.039, 'P': 0.050,
            'S': 0.068, 'T': 0.058, 'W': 0.013, 'Y': 0.032, 'V': 0.066
        }
    
    def compute_reward(self, sequence: str, structure: Dict, 
                      detailed: bool = False) -> float:
        """
        Compute length-aware reward for a protein sequence.
        
        Args:
            sequence: Amino acid sequence
            structure: Structure dictionary with backbone coordinates
            detailed: Whether to return detailed breakdown
            
        Returns:
            Total reward score (or detailed dictionary)
        """
        length = len(sequence)
        
        # Get length-specific weights
        weights = self._get_length_specific_weights(length)
        
        # Compute individual reward components
        structure_score = self._compute_structure_compatibility(sequence, structure)
        hydrophobicity_score = self._compute_hydrophobicity_reward(sequence, length)
        charge_score = self._compute_charge_reward(sequence, length)
        diversity_score = self._compute_diversity_reward(sequence)
        stability_score = self._compute_stability_reward(sequence, length)
        
        # Weighted combination
        total_reward = (
            weights['structure'] * structure_score +
            weights['hydrophobicity'] * hydrophobicity_score +
            weights['charge'] * charge_score +
            weights['diversity'] * diversity_score +
            weights['stability'] * stability_score
        )
        
        if detailed:
            return {
                'total_reward': total_reward,
                'structure_compatibility': structure_score,
                'hydrophobicity_balance': hydrophobicity_score,
                'charge_balance': charge_score,
                'sequence_diversity': diversity_score,
                'stability_score': stability_score,
                'length': length,
                'weights': weights
            }
        
        return total_reward
    
    def _get_length_specific_weights(self, length: int) -> Dict[str, float]:
        """Get reward weights based on protein length."""
        if length < 100:
            # Small proteins: Focus on local properties
            return {
                'structure': 0.30,
                'hydrophobicity': 0.30,
                'charge': 0.25,
                'diversity': 0.10,
                'stability': 0.05
            }
        elif length < 300:
            # Medium proteins: Balanced approach
            return {
                'structure': 0.40,
                'hydrophobicity': 0.25,
                'charge': 0.20,
                'diversity': 0.10,
                'stability': 0.05
            }
        else:
            # Large proteins: Emphasize global structure
            return {
                'structure': 0.50,
                'hydrophobicity': 0.20,
                'charge': 0.15,
                'diversity': 0.10,
                'stability': 0.05
            }
    
    def _compute_structure_compatibility(self, sequence: str, structure: Dict) -> float:
        """
        Compute structure-sequence compatibility score.
        
        This is a placeholder for actual structure compatibility metrics
        like TM-score, GDT-TS, or learned compatibility functions.
        """
        length = len(sequence)
        
        # Mock compatibility based on sequence properties
        # In practice, this would use actual structure prediction/comparison
        
        # Factor 1: Length consistency
        target_length = structure.get('length', length)
        length_penalty = abs(length - target_length) / max(length, target_length)
        
        # Factor 2: Mock local structure compatibility
        # This could be replaced with actual secondary structure prediction
        hydrophobic_residues = sum(1 for aa in sequence if self.hydrophobicity_scores.get(aa, 0) > 2.0)
        hydrophobic_fraction = hydrophobic_residues / length
        
        # Factor 3: Mock global structure score
        # This could be replaced with actual fold recognition or energy functions
        global_score = 0.7 + 0.3 * np.random.random()  # Placeholder
        
        # Combine factors
        compatibility = (
            (1.0 - length_penalty) * 0.3 +
            (1.0 - abs(hydrophobic_fraction - 0.3)) * 0.3 +
            global_score * 0.4
        )
        
        return max(0.0, min(1.0, compatibility))
    
    def _compute_hydrophobicity_reward(self, sequence: str, length: int) -> float:
        """Compute hydrophobicity balance reward with length scaling."""
        scores = [self.hydrophobicity_scores.get(aa, 0) for aa in sequence]
        avg_hydrophobicity = np.mean(scores)
        
        # Optimal hydrophobicity depends on protein size
        if length < 100:
            # Small proteins can be more hydrophobic
            optimal_hydrophobicity = 0.5
            tolerance = 1.5
        elif length < 300:
            # Medium proteins need balance
            optimal_hydrophobicity = 0.0
            tolerance = 1.0
        else:
            # Large proteins need careful hydrophobicity distribution
            optimal_hydrophobicity = -0.2
            tolerance = 0.8
        
        # Compute deviation from optimal
        deviation = abs(avg_hydrophobicity - optimal_hydrophobicity)
        reward = max(0.0, 1.0 - deviation / tolerance)
        
        # Length scaling: longer proteins need more careful balance
        length_factor = 1.0 / (1.0 + math.log(length / 50.0))
        
        return reward * length_factor
    
    def _compute_charge_reward(self, sequence: str, length: int) -> float:
        """Compute charge balance reward with length considerations."""
        total_charge = sum(self.charge_scores.get(aa, 0) for aa in sequence)
        
        # Normalize by length
        charge_density = total_charge / length
        
        # Length-specific charge tolerance
        if length < 100:
            # Small proteins can tolerate more charge imbalance
            tolerance = 0.15
        elif length < 300:
            # Medium proteins need moderate balance
            tolerance = 0.10
        else:
            # Large proteins need careful charge distribution
            tolerance = 0.05
        
        # Reward for being close to neutral
        charge_reward = max(0.0, 1.0 - abs(charge_density) / tolerance)
        
        return charge_reward
    
    def _compute_diversity_reward(self, sequence: str) -> float:
        """Compute amino acid diversity reward."""
        # Count amino acid frequencies
        aa_counts = Counter(sequence)
        length = len(sequence)
        
        # Compute Shannon entropy
        entropy = 0.0
        for aa, count in aa_counts.items():
            freq = count / length
            if freq > 0:
                entropy -= freq * math.log2(freq)
        
        # Normalize by maximum possible entropy
        max_entropy = math.log2(20)  # 20 amino acids
        diversity_score = entropy / max_entropy
        
        # Bonus for natural-like amino acid distribution
        natural_score = 0.0
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            observed_freq = aa_counts.get(aa, 0) / length
            natural_freq = self.natural_frequencies[aa]
            natural_score += 1.0 - abs(observed_freq - natural_freq)
        natural_score /= 20.0
        
        # Combine diversity and naturalness
        return 0.7 * diversity_score + 0.3 * natural_score
    
    def _compute_stability_reward(self, sequence: str, length: int) -> float:
        """Compute sequence stability reward based on biophysical properties."""
        # Factor 1: Avoid problematic residue patterns
        problematic_patterns = [
            'PP',  # Proline-proline (rigid)
            'CC',  # Cysteine-cysteine (potential disulfide issues)
            'GG',  # Glycine-glycine (too flexible)
        ]
        
        pattern_penalty = 0.0
        for pattern in problematic_patterns:
            pattern_penalty += sequence.count(pattern) / length
        
        # Factor 2: Moderate secondary structure propensity
        # This is a simplified version - real implementation would use
        # secondary structure prediction
        helix_forming = sum(1 for aa in sequence if aa in 'AELM')
        sheet_forming = sum(1 for aa in sequence if aa in 'IVFY')
        
        helix_fraction = helix_forming / length
        sheet_fraction = sheet_forming / length
        
        # Balanced secondary structure is generally good
        structure_balance = 1.0 - abs(helix_fraction - sheet_fraction)
        
        # Factor 3: Length-specific stability considerations
        if length < 100:
            # Small proteins benefit from compact structures
            stability_bonus = 0.1 if helix_fraction > 0.3 else 0.0
        else:
            # Larger proteins need balanced structures
            stability_bonus = 0.1 if 0.2 < helix_fraction < 0.4 else 0.0
        
        stability_score = (
            (1.0 - pattern_penalty) * 0.4 +
            structure_balance * 0.5 +
            stability_bonus * 0.1
        )
        
        return max(0.0, min(1.0, stability_score))


def compute_detailed_reward_analysis(sequence: str, structure: Dict) -> Dict:
    """
    Compute detailed reward analysis for a sequence.
    
    Args:
        sequence: Amino acid sequence
        structure: Structure dictionary
        
    Returns:
        Detailed analysis dictionary
    """
    reward_computer = LengthAwareRewardComputation()
    
    # Get detailed reward breakdown
    detailed_reward = reward_computer.compute_reward(sequence, structure, detailed=True)
    
    # Add additional analysis
    length = len(sequence)
    aa_composition = Counter(sequence)
    
    # Compute amino acid statistics
    hydrophobic_count = sum(1 for aa in sequence if reward_computer.hydrophobicity_scores.get(aa, 0) > 1.0)
    charged_count = sum(1 for aa in sequence if aa in 'RKHED')
    
    detailed_reward.update({
        'amino_acid_composition': dict(aa_composition),
        'hydrophobic_residues': hydrophobic_count,
        'hydrophobic_fraction': hydrophobic_count / length,
        'charged_residues': charged_count,
        'charged_fraction': charged_count / length,
        'length_category': 'small' if length < 100 else 'medium' if length < 300 else 'large'
    })
    
    return detailed_reward


# Convenience functions for different protein sizes
def compute_small_protein_reward(sequence: str, structure: Dict) -> float:
    """Optimized reward for small proteins (<100 residues)."""
    reward_computer = LengthAwareRewardComputation()
    return reward_computer.compute_reward(sequence, structure)


def compute_medium_protein_reward(sequence: str, structure: Dict) -> float:
    """Optimized reward for medium proteins (100-300 residues)."""
    reward_computer = LengthAwareRewardComputation()
    return reward_computer.compute_reward(sequence, structure)


def compute_large_protein_reward(sequence: str, structure: Dict) -> float:
    """Optimized reward for large proteins (>300 residues)."""
    reward_computer = LengthAwareRewardComputation()
    return reward_computer.compute_reward(sequence, structure) 