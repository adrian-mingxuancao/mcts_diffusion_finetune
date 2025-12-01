"""
Hallucination Expert for MCTS

This expert integrates AF3 (via ABCFold) + ProteinMPNN as a two-step process:
1. AF3 hallucinates structure from masked sequence
2. ProteinMPNN designs sequence from hallucinated structure

This plugs into the existing GeneralMCTS as a new expert type.
"""

import numpy as np
from typing import Dict, Set, Optional, List
import sys
import os

# Import the integrations
from core.abcfold_integration import ABCFoldIntegration

# Simple ProteinMPNN wrapper for mock mode
class ProteinMPNNIntegration:
    """Simple ProteinMPNN wrapper for hallucination pipeline."""
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        print(f"🔧 ProteinMPNN Integration initialized (mock mode)")
    
    def design_sequence(self, coordinates: np.ndarray) -> str:
        """Design sequence from coordinates (mock mode)."""
        # Mock: generate random sequence of same length
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        n = len(coordinates)
        sequence = ''.join(np.random.choice(list(amino_acids), size=n))
        return sequence


class HallucinationExpert:
    """
    Expert that uses AF3 hallucination + ProteinMPNN inverse folding.
    
    This can be added to GeneralMCTS.external_experts list.
    """
    
    def __init__(self, model_params: str = None, use_mock: bool = True):
        """
        Initialize hallucination expert.
        
        Args:
            model_params: Path to AF3 model parameters (for real mode)
            use_mock: Use mock mode for testing (default: True)
        """
        self.abcfold = ABCFoldIntegration(model_params=model_params, use_mock=use_mock)
        self.proteinmpnn = ProteinMPNNIntegration()
        self.name = "hallucination"
        
        print(f"🔧 HallucinationExpert initialized")
    
    def get_name(self) -> str:
        """Return expert name."""
        return self.name
    
    def generate_candidate(
        self, 
        sequence: str, 
        masked_positions: Set[int],
        coordinates: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict:
        """
        Generate candidate using AF3 + ProteinMPNN pipeline.
        
        Process:
        1. Create masked sequence (X at masked positions)
        2. AF3 hallucinates structure from masked sequence
        3. ProteinMPNN designs sequence from hallucinated structure
        4. Return candidate with sequence, coordinates, and confidence
        
        Args:
            sequence: Current sequence
            masked_positions: Positions to re-design
            coordinates: Optional current coordinates (not used for hallucination)
            
        Returns:
            Dictionary with:
                - sequence: New designed sequence
                - coordinates: Hallucinated structure coordinates
                - confidence_scores: AF3 confidence scores (pLDDT-like)
                - expert: Expert name
                - entropy: Model uncertainty
        """
        try:
            # Step 1: Create masked sequence
            masked_seq = self._create_masked_sequence(sequence, masked_positions)
            print(f"      🎭 Hallucination: {len(masked_positions)} positions masked")
            
            # Step 2: AF3 hallucination
            print(f"      🔮 AF3: Predicting structure...")
            af3_result = self.abcfold.predict_structure(masked_seq)
            hallucinated_coords = af3_result['coordinates']
            confidence_scores = af3_result['confidence']
            mean_plddt = np.mean(confidence_scores)
            print(f"      ✅ AF3: Structure predicted (mean pLDDT: {mean_plddt:.1f})")
            
            # Step 3: ProteinMPNN inverse folding
            print(f"      🧬 ProteinMPNN: Designing sequence...")
            designed_sequence = self.proteinmpnn.design_sequence(hallucinated_coords)
            print(f"      ✅ ProteinMPNN: Sequence designed")
            
            # Step 4: Compute entropy from confidence variance
            entropy = self._compute_entropy(confidence_scores)
            
            return {
                'sequence': designed_sequence,
                'coordinates': hallucinated_coords,
                'confidence_scores': confidence_scores.tolist() if isinstance(confidence_scores, np.ndarray) else confidence_scores,
                'plddt_scores': confidence_scores.tolist() if isinstance(confidence_scores, np.ndarray) else confidence_scores,
                'expert': self.name,
                'entropy': entropy,
                'mean_plddt': mean_plddt,
                'pae_mean': af3_result.get('pae_mean', 0.0)
            }
            
        except Exception as e:
            print(f"      ❌ Hallucination expert failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_masked_sequence(self, sequence: str, masked_positions: Set[int]) -> str:
        """Create masked sequence with X at masked positions."""
        masked = list(sequence)
        for pos in masked_positions:
            if pos < len(masked):
                masked[pos] = 'X'
        return ''.join(masked)
    
    def _compute_entropy(self, confidence_scores: np.ndarray) -> float:
        """
        Compute entropy from confidence score variance.
        
        Higher variance = higher uncertainty = higher entropy
        """
        if confidence_scores is None or len(confidence_scores) == 0:
            return 0.5
        
        # Convert to numpy if needed
        if not isinstance(confidence_scores, np.ndarray):
            confidence_scores = np.array(confidence_scores)
        
        # Compute variance and normalize to [0, 1]
        variance = np.var(confidence_scores)
        # Typical pLDDT variance is 0-400, normalize
        normalized_entropy = min(1.0, variance / 400.0)
        
        return normalized_entropy
    
    def compute_entropy(
        self, 
        sequence: str, 
        masked_positions: Set[int],
        **kwargs
    ) -> float:
        """
        Compute predictive entropy for PH-UCT.
        
        For hallucination, we can't easily compute entropy without running AF3,
        so we return a default value. The actual entropy is computed during generation.
        """
        # Default entropy for hallucination (medium uncertainty)
        return 0.5


def create_hallucination_expert(model_params: str = None, use_mock: bool = True) -> HallucinationExpert:
    """
    Factory function to create hallucination expert.
    
    Args:
        model_params: Path to AF3 model parameters (for real mode)
        use_mock: Use mock mode for testing (default: True)
    
    Usage:
        # Mock mode (for testing)
        hallucination_expert = create_hallucination_expert()
        
        # Real mode (with AF3)
        hallucination_expert = create_hallucination_expert(
            model_params="/path/to/af3/params",
            use_mock=False
        )
        
        # Add to MCTS
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            external_experts=[hallucination_expert],
            ablation_mode="single_expert",
            single_expert_id=3
        )
    """
    return HallucinationExpert(model_params=model_params, use_mock=use_mock)
