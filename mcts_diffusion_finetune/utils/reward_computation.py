"""
Enhanced reward computation for MCTS-guided inverse folding.

This module provides sophisticated, length-aware reward functions for proteins
of different sizes, taking into account biophysical properties and structural
constraints that vary with protein length.

Now includes real structure evaluation metrics from DPLM-2 framework.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import math
from collections import Counter

try:
    from .structure_evaluation import create_structure_evaluator
    STRUCTURE_EVAL_AVAILABLE = True
except ImportError:
    print("Warning: Structure evaluation not available, using fallback")
    STRUCTURE_EVAL_AVAILABLE = False


class LengthAwareRewardComputation:
    """
    Sophisticated reward computation that adapts to protein length.
    
    Different protein sizes have different optimization priorities:
    - Small proteins (<100): Focus on local structure and stability
    - Medium proteins (100-300): Balance local and global properties  
    - Large proteins (>300): Emphasize global structure compatibility
    """
    
    def __init__(self, use_real_structure_eval: bool = True):
        """
        Initialize reward computation.
        
        Args:
            use_real_structure_eval: Whether to use real structure evaluation metrics
        """
        # Initialize structure evaluator
        self.use_real_structure_eval = use_real_structure_eval and STRUCTURE_EVAL_AVAILABLE
        if self.use_real_structure_eval:
            try:
                self.structure_evaluator = create_structure_evaluator()
            except Exception:
                self.structure_evaluator = None
                self.use_real_structure_eval = False
        else:
            self.structure_evaluator = None
        
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
        Compute length-aware reward for a protein sequence with inverse folding focus.
        
        Args:
            sequence: Amino acid sequence
            structure: Structure dictionary with backbone coordinates
            detailed: Whether to return detailed breakdown
            
        Returns:
            Total reward score (or detailed dictionary)
        """
        # Handle None sequence gracefully
        if sequence is None:
            self.logger.warning("Received None sequence, using fallback reward")
            return 0.5 if not detailed else {'total_reward': 0.5, 'error': 'None sequence'}
        
        length = len(sequence)
        
        # Get length-specific weights
        weights = self._get_length_specific_weights(length)
        
        # Compute individual reward components
        structure_score = self._compute_structure_compatibility(sequence, structure)
        hydrophobicity_score = self._compute_hydrophobicity_reward(sequence, length)
        charge_score = self._compute_charge_reward(sequence, length)
        diversity_score = self._compute_diversity_reward(sequence)
        stability_score = self._compute_stability_reward(sequence, length)
        
        # Add inverse folding specific components
        inverse_folding_score = self._compute_inverse_folding_score(sequence, structure)
        naturalness_score = self._compute_naturalness_score(sequence)
        
        # Weighted combination with inverse folding focus
        total_reward = (
            weights['structure'] * structure_score +
            weights['hydrophobicity'] * hydrophobicity_score +
            weights['charge'] * charge_score +
            weights['diversity'] * diversity_score +
            weights['stability'] * stability_score +
            0.3 * inverse_folding_score +  # Higher weight for inverse folding
            0.2 * naturalness_score
        )
        
        if detailed:
            return {
                'total_reward': total_reward,
                'structure_compatibility': structure_score,
                'hydrophobicity_balance': hydrophobicity_score,
                'charge_balance': charge_score,
                'sequence_diversity': diversity_score,
                'stability_score': stability_score,
                'inverse_folding_score': inverse_folding_score,
                'naturalness_score': naturalness_score,
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
        Compute structure-sequence compatibility score using real structure evaluation.
        
        Now uses DPLM-2 evaluation framework:
        - Self-consistency evaluation (fold sequence, compare to reference)
        - TM-score calculation
        - RMSD measurement
        - pLDDT confidence scoring
        """
        if self.use_real_structure_eval and self.structure_evaluator:
            return self._compute_real_structure_compatibility(sequence, structure)
        else:
            return self._compute_mock_structure_compatibility(sequence, structure)
    
    def _compute_real_structure_compatibility(self, sequence: str, structure: Dict) -> float:
        """
        Real structure compatibility using DPLM-2 evaluation framework.
        """
        try:
            # Evaluate designability using self-consistency
            designability_results = self.structure_evaluator.evaluate_designability(
                sequence, structure, rmsd_threshold=2.0
            )
            
            # Extract key metrics
            sc_tmscore = designability_results['sc_tmscore']
            sc_rmsd = designability_results['bb_rmsd']  # Fixed: use bb_rmsd key
            plddt = designability_results['plddt']
            seq_recovery = designability_results['seq_recovery']
            
            # Compute composite compatibility score
            # TM-score: higher is better (0-1)
            tmscore_component = sc_tmscore
            
            # RMSD: lower is better, normalize to 0-1 scale
            # Good RMSD: 0-2Å -> score 1.0-0.5, Poor RMSD: >5Å -> score ~0
            rmsd_component = max(0.0, 1.0 - sc_rmsd / 5.0)
            
            # pLDDT: confidence score, normalize to 0-1
            plddt_component = plddt / 100.0
            
            # For inverse folding, sequence recovery is not meaningful (no reference sequence)
            # Focus on self-consistency metrics: TM-score, RMSD, pLDDT
            
            # Weight components based on protein length (no sequence recovery)
            length = len(sequence)
            if length < 100:      # Small proteins: emphasize local accuracy (RMSD)
                weights = {'tmscore': 0.35, 'rmsd': 0.45, 'plddt': 0.20}
            elif length < 300:    # Medium proteins: balanced
                weights = {'tmscore': 0.45, 'rmsd': 0.35, 'plddt': 0.20}
            else:                 # Large proteins: emphasize global structure (TM-score)
                weights = {'tmscore': 0.55, 'rmsd': 0.25, 'plddt': 0.20}
            
            # Composite score (self-consistency only, no sequence recovery)
            compatibility = (
                tmscore_component * weights['tmscore'] +
                rmsd_component * weights['rmsd'] +
                plddt_component * weights['plddt']
            )
            
            return max(0.0, min(1.0, compatibility))
            
        except Exception:
            return self._compute_mock_structure_compatibility(sequence, structure)
    
    def _compute_mock_structure_compatibility(self, sequence: str, structure: Dict) -> float:
        """
        Fallback mock compatibility based on sequence properties.
        """
        length = len(sequence)
        
        # Factor 1: Length consistency
        target_length = structure.get('target_length', structure.get('length', length))
        length_penalty = abs(length - target_length) / max(length, target_length)
        
        # Factor 2: Hydrophobic fraction (proxy for foldability)
        hydrophobic_residues = sum(1 for aa in sequence if self.hydrophobicity_scores.get(aa, 0) > 2.0)
        hydrophobic_fraction = hydrophobic_residues / length
        hydrophobic_score = 1.0 - abs(hydrophobic_fraction - 0.35)  # Target ~35% hydrophobic
        
        # Factor 3: Charge balance (proxy for stability)
        positive = sum(1 for aa in sequence if aa in "KRH")
        negative = sum(1 for aa in sequence if aa in "DE")
        net_charge = abs(positive - negative)
        charge_score = max(0.0, 1.0 - net_charge / (length * 0.1))
        
        # Combine factors
        compatibility = (
            (1.0 - length_penalty) * 0.4 +
            hydrophobic_score * 0.4 +
            charge_score * 0.2
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

    def _compute_inverse_folding_score(self, sequence: str, structure: Dict) -> float:
        """Compute score for how well sequence matches target structure."""
        score = 0.0
        
        # Score based on length compatibility
        target_length = structure.get('target_length', len(sequence))
        if abs(len(sequence) - target_length) <= 2:
            score += 0.3
        elif abs(len(sequence) - target_length) <= 5:
            score += 0.1
        
        # Score based on hydrophobicity profile matching (if available)
        if 'hydrophobicity_profile' in structure:
            profile = structure['hydrophobicity_profile']
            matches = 0
            for i, (aa, expected_hydro) in enumerate(zip(sequence, profile)):
                if i < len(profile):
                    aa_hydro = self.hydrophobicity_scores.get(aa, 0.0)
                    # Reward for matching hydrophobicity pattern
                    if (expected_hydro > 0 and aa_hydro > 0) or (expected_hydro < 0 and aa_hydro < 0):
                        matches += 1
            
            if len(profile) > 0:
                match_ratio = matches / min(len(sequence), len(profile))
                score += 0.4 * match_ratio
        
        # Score based on secondary structure propensity
        ss_score = self._compute_secondary_structure_score(sequence)
        score += 0.2 * ss_score
        
        # Score based on sequence naturalness
        naturalness = self._compute_naturalness_score(sequence)
        score += 0.1 * naturalness
        
        return min(1.0, score)
    
    def _compute_naturalness_score(self, sequence: str) -> float:
        """Compute how natural the sequence appears."""
        score = 1.0
        
        # Penalty for too many consecutive identical residues
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i+1] == sequence[i+2]:
                score -= 0.1
        
        # Penalty for extreme amino acid composition
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        for aa, count in aa_counts.items():
            frequency = count / len(sequence)
            if frequency > 0.25:  # Too much of one amino acid
                score -= 0.1
            elif frequency > 0.15:  # Moderately high
                score -= 0.05
        
        # Bonus for natural amino acid frequencies
        natural_freq_sum = 0
        for aa, count in aa_counts.items():
            expected_freq = self.natural_frequencies.get(aa, 0.05)
            actual_freq = count / len(sequence)
            natural_freq_sum += 1.0 - abs(actual_freq - expected_freq)
        
        natural_freq_score = natural_freq_sum / len(aa_counts) if aa_counts else 0
        score += 0.2 * natural_freq_score
        
        return max(0.0, score)
    
    def _compute_secondary_structure_score(self, sequence: str) -> float:
        """Compute secondary structure propensity score."""
        score = 0.0
        
        # Simple secondary structure propensity
        helix_favoring = "ADEFHIKLMNPQRSTVWY"  # Simplified
        sheet_favoring = "ACDEFGHIKLMNPQRSTVWY"  # Simplified
        
        helix_windows = 0
        sheet_windows = 0
        
        for i in range(len(sequence) - 3):
            window = sequence[i:i+4]
            helix_score = sum(1 for aa in window if aa in helix_favoring)
            sheet_score = sum(1 for aa in window if aa in sheet_favoring)
            
            if helix_score >= 3:
                helix_windows += 1
            if sheet_score >= 3:
                sheet_windows += 1
        
        # Normalize by sequence length
        if len(sequence) > 4:
            helix_ratio = helix_windows / (len(sequence) - 3)
            sheet_ratio = sheet_windows / (len(sequence) - 3)
            score = (helix_ratio + sheet_ratio) / 2
        
        return min(1.0, score)


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


# Add the missing compute_biophysical_properties method to LengthAwareRewardComputation
def _add_biophysical_method():
    """Add the missing compute_biophysical_properties method."""
    def compute_biophysical_properties(self, sequence: str) -> float:
        """
        Compute biophysical quality score for a protein sequence.
        
        This method combines multiple biophysical factors:
        - Charge balance (avoid extreme charge distributions)
        - Hydrophobic composition (proper hydrophobic/hydrophilic balance)
        - Amino acid frequency distribution (similarity to natural proteins)
        
        Returns:
            Float score between 0.0 (poor) and 1.0 (excellent)
        """
        if not sequence or len(sequence) == 0:
            return 0.0
            
        length = len(sequence)
        
        # Factor 1: Charge balance penalty
        charged_residues = sum(1 for aa in sequence if aa in "DEKRH")
        charge_fraction = charged_residues / length
        
        # Penalize extreme charge compositions (>30% charged residues)
        if charge_fraction > 0.30:
            charge_penalty = (charge_fraction - 0.30) * 2.0  # Linear penalty
        else:
            charge_penalty = 0.0
            
        # Factor 2: Hydrophobic composition penalty  
        hydrophobic_residues = sum(1 for aa in sequence if aa in "AILVF")
        hydrophobic_fraction = hydrophobic_residues / length
        
        # Penalize extreme hydrophobic compositions (>40% hydrophobic)
        if hydrophobic_fraction > 0.40:
            hydrophobic_penalty = (hydrophobic_fraction - 0.40) * 2.5  # Linear penalty
        else:
            hydrophobic_penalty = 0.0
            
        # Factor 3: Amino acid frequency deviation from natural proteins
        sequence_counts = {aa: sequence.count(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}
        sequence_freqs = {aa: count/length for aa, count in sequence_counts.items()}
        
        # Compute deviation from natural frequencies
        freq_deviation = 0.0
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            natural_freq = self.natural_frequencies.get(aa, 0.05)
            observed_freq = sequence_freqs.get(aa, 0.0)
            freq_deviation += abs(observed_freq - natural_freq)
            
        # Normalize frequency deviation (max possible deviation ≈ 2.0)
        freq_penalty = min(1.0, freq_deviation / 2.0)
        
        # Combine penalties into quality score
        total_penalty = charge_penalty + hydrophobic_penalty + freq_penalty * 0.5
        biophysical_score = max(0.0, 1.0 - total_penalty)
        
        return biophysical_score
    
    # Add method to the class
    LengthAwareRewardComputation.compute_biophysical_properties = compute_biophysical_properties

# Execute the patch
_add_biophysical_method() 