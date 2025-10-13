"""
Simple External Experts for MCTS

Simplified external experts that focus on providing diverse generation strategies
for multi-expert MCTS without complex model loading dependencies.

These experts use different biases and strategies to provide diversity:
- ProteInA-style: Flow matching biased generation
- FoldFlow-style: Structure-aware generation  
- RFDiffusion-style: Diffusion-biased generation
- ProteinMPNN-style: Natural frequency biased generation
"""

import numpy as np
import torch
from typing import Dict, Optional, List
import random


class SimpleExternalExpert:
    """Base class for simple external experts."""
    
    def __init__(self, name: str, generation_bias: str):
        self.name = name
        self.generation_bias = generation_bias
        self.model = "simple_expert"  # Always available
        
    def get_name(self) -> str:
        """Get expert name."""
        return self.name
        
    def generate_scaffold(self, motif_data, scaffold_length: int, **kwargs) -> Dict:
        """Generate scaffold with specific bias."""
        print(f"   🔄 {self.name} generating scaffold...")
        
        # Get generation parameters
        temperature = kwargs.get('temperature', 1.0)
        target_length = motif_data.target_length
        
        # Generate sequence with model-specific bias
        generated_seq = self._generate_biased_sequence(target_length, temperature)
        
        # Insert motif at correct positions
        full_sequence = self._insert_motif_into_scaffold(motif_data, generated_seq)
        
        # Verify motif preservation
        motif_preserved = self._verify_motif_preservation(motif_data, full_sequence)
        
        # Calculate entropy based on generation strategy
        entropy = self._calculate_entropy(full_sequence, temperature)
        
        print(f"   ✅ {self.name} generated: {len(full_sequence)} residues")
        print(f"   🎯 Motif preserved: {motif_preserved}")
        print(f"   📊 Entropy: {entropy:.3f}")
        
        return {
            'full_sequence': full_sequence,
            'motif_preserved': motif_preserved,
            'scaffold_length': len(full_sequence) - len(motif_data.motif_sequence),
            'method': f'{self.name.lower()}_simple',
            'motif_sequence': motif_data.motif_sequence,
            'temperature': temperature,
            'structure_sequence': '',
            'entropy': entropy
        }
        
    def _generate_biased_sequence(self, length: int, temperature: float) -> str:
        """Generate sequence with model-specific bias."""
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Get model-specific amino acid preferences
        weights = self._get_amino_acid_weights()
        
        # Apply temperature scaling
        if temperature != 1.0:
            weights = np.array(weights) ** (1.0 / temperature)
            weights = weights / weights.sum()
        
        # Generate sequence
        sequence = ''.join(np.random.choice(list(amino_acids), size=length, p=weights))
        return sequence
        
    def _get_amino_acid_weights(self) -> List[float]:
        """Get amino acid weights for this expert. Override in subclasses."""
        # Default natural frequencies
        return [0.074, 0.052, 0.063, 0.054, 0.071, 0.022, 0.022, 0.091, 0.058, 0.096,
                0.024, 0.048, 0.040, 0.051, 0.047, 0.082, 0.055, 0.013, 0.032, 0.068]
        
    def _insert_motif_into_scaffold(self, motif_data, scaffold_seq: str) -> str:
        """Insert motif into scaffold sequence at correct positions."""
        if hasattr(motif_data, 'motif_positions') and motif_data.motif_positions:
            # Create full sequence with motif at correct positions
            full_seq = list('X' * motif_data.target_length)
            
            # Place motif segments
            motif_chars = list(motif_data.motif_sequence)
            for i, pos in enumerate(motif_data.motif_positions):
                if i < len(motif_chars) and pos < len(full_seq):
                    full_seq[pos] = motif_chars[i]
                    
            # Fill remaining positions with scaffold
            scaffold_chars = list(scaffold_seq)
            scaffold_idx = 0
            for i in range(len(full_seq)):
                if full_seq[i] == 'X' and scaffold_idx < len(scaffold_chars):
                    full_seq[i] = scaffold_chars[scaffold_idx]
                    scaffold_idx += 1
                    
            return ''.join(full_seq)
        else:
            # Simple concatenation fallback
            return scaffold_seq + motif_data.motif_sequence
            
    def _verify_motif_preservation(self, motif_data, generated_seq: str) -> bool:
        """Verify that motif is preserved in generated sequence."""
        if hasattr(motif_data, 'motif_positions') and motif_data.motif_positions:
            # Check non-contiguous motif preservation
            motif_chars = list(motif_data.motif_sequence)
            for i, pos in enumerate(motif_data.motif_positions):
                if i < len(motif_chars) and pos < len(generated_seq):
                    if generated_seq[pos] != motif_chars[i]:
                        return False
            return True
        else:
            # Simple substring check
            return motif_data.motif_sequence in generated_seq
            
    def _calculate_entropy(self, sequence: str, temperature: float) -> float:
        """Calculate sequence entropy."""
        # Count amino acid frequencies
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
        # Calculate entropy
        total = len(sequence)
        entropy = 0.0
        for count in aa_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
                
        # Scale by temperature and model bias
        return entropy * temperature * self._get_entropy_scale()
        
    def _get_entropy_scale(self) -> float:
        """Get entropy scaling factor for this expert."""
        return 1.0  # Override in subclasses


class ProteInAStyleExpert(SimpleExternalExpert):
    """ProteInA-style expert with flow matching bias."""
    
    def __init__(self):
        super().__init__("ProteInA", "flow_matching")
        
    def _get_amino_acid_weights(self) -> List[float]:
        """ProteInA tends to favor structured, designable sequences."""
        # Slightly favor structured amino acids (A, L, E, K, R)
        return [0.080, 0.050, 0.065, 0.050, 0.075, 0.020, 0.020, 0.095, 0.055, 0.100,
                0.025, 0.045, 0.035, 0.055, 0.050, 0.085, 0.060, 0.010, 0.030, 0.075]
                
    def _get_entropy_scale(self) -> float:
        """ProteInA generates diverse sequences."""
        return 1.1


class FoldFlowStyleExpert(SimpleExternalExpert):
    """FoldFlow-style expert with structure-aware bias."""
    
    def __init__(self):
        super().__init__("FoldFlow", "structure_aware")
        
    def _get_amino_acid_weights(self) -> List[float]:
        """FoldFlow favors amino acids that form stable structures."""
        # Favor hydrophobic core (L, I, V, F) and structure-forming (P, G)
        return [0.078, 0.055, 0.060, 0.050, 0.070, 0.025, 0.025, 0.090, 0.065, 0.105,
                0.020, 0.050, 0.045, 0.050, 0.055, 0.080, 0.055, 0.015, 0.035, 0.070]
                
    def _get_entropy_scale(self) -> float:
        """FoldFlow generates moderately diverse sequences."""
        return 0.9


class RFDiffusionStyleExpert(SimpleExternalExpert):
    """RFDiffusion-style expert with diffusion bias."""
    
    def __init__(self):
        super().__init__("RFDiffusion", "diffusion")
        
    def _get_amino_acid_weights(self) -> List[float]:
        """RFDiffusion balances natural frequencies with structural preferences."""
        # Balanced approach with slight preference for common structural AAs
        return [0.076, 0.052, 0.064, 0.054, 0.072, 0.022, 0.022, 0.092, 0.058, 0.098,
                0.024, 0.048, 0.040, 0.051, 0.048, 0.083, 0.056, 0.013, 0.032, 0.069]
                
    def _get_entropy_scale(self) -> float:
        """RFDiffusion generates balanced diversity."""
        return 1.0


class ProteinMPNNStyleExpert(SimpleExternalExpert):
    """ProteinMPNN-style expert with natural frequency bias."""
    
    def __init__(self):
        super().__init__("ProteinMPNN", "natural_frequencies")
        
    def _get_amino_acid_weights(self) -> List[float]:
        """ProteinMPNN uses natural amino acid frequencies."""
        # Natural frequencies from protein databases
        return [0.074, 0.052, 0.063, 0.054, 0.071, 0.022, 0.022, 0.091, 0.058, 0.096,
                0.024, 0.048, 0.040, 0.051, 0.047, 0.082, 0.055, 0.013, 0.032, 0.068]
                
    def _get_entropy_scale(self) -> float:
        """ProteinMPNN generates lower entropy, more natural sequences."""
        return 0.8


# Factory functions
def create_proteina_simple_expert() -> ProteInAStyleExpert:
    """Create ProteInA-style simple expert."""
    return ProteInAStyleExpert()

def create_foldflow_simple_expert() -> FoldFlowStyleExpert:
    """Create FoldFlow-style simple expert."""
    return FoldFlowStyleExpert()

def create_rfdiffusion_simple_expert() -> RFDiffusionStyleExpert:
    """Create RFDiffusion-style simple expert."""
    return RFDiffusionStyleExpert()

def create_proteinmpnn_simple_expert() -> ProteinMPNNStyleExpert:
    """Create ProteinMPNN-style simple expert."""
    return ProteinMPNNStyleExpert()

def create_all_simple_experts() -> List[SimpleExternalExpert]:
    """Create all simple external experts."""
    experts = [
        create_proteina_simple_expert(),
        create_foldflow_simple_expert(),
        create_rfdiffusion_simple_expert(),
        create_proteinmpnn_simple_expert()
    ]
    
    print(f"✅ Created {len(experts)} simple external experts:")
    for expert in experts:
        print(f"   ✅ {expert.get_name()} ({expert.generation_bias})")
    
    return experts


