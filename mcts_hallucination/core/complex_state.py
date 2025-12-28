"""
Complex State for MCTS

Immutable state representation for multi-chain complexes in MCTS hallucination.
Used as the state in ComplexNode for tree search.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
import numpy as np
import copy

from .complex_input import ComplexInput, ProteinChain, DNAChain, RNAChain, Ligand


@dataclass
class ChainCoordinates:
    """Coordinates and confidence for a single chain."""
    chain_id: str
    coordinates: np.ndarray  # (N, 3) CA/C1' coordinates
    plddt_scores: np.ndarray  # (N,) per-residue confidence
    
    @property
    def mean_plddt(self) -> float:
        return float(np.mean(self.plddt_scores))
    
    @property
    def length(self) -> int:
        return len(self.coordinates)


@dataclass
class ComplexState:
    """
    Immutable state of a multi-chain complex for MCTS.
    
    This wraps ComplexInput and adds structural information.
    """
    
    # Input specification
    complex_input: ComplexInput
    
    # Structure (optional - may be None before prediction)
    chain_coords: Optional[Dict[str, ChainCoordinates]] = None
    pae_matrix: Optional[np.ndarray] = None  # (total_len, total_len) PAE
    
    def get_sequence(self, chain_id: str) -> str:
        """Get sequence of a specific chain."""
        chain = self.complex_input.chains.get(chain_id)
        if chain and hasattr(chain, 'sequence'):
            return chain.sequence
        return ""
    
    def get_all_sequences(self) -> Dict[str, str]:
        """Get all sequences as dict."""
        return {
            chain_id: chain.sequence
            for chain_id, chain in self.complex_input.chains.items()
            if hasattr(chain, 'sequence')
        }
    
    def get_designable_sequences(self) -> Dict[str, str]:
        """Get sequences of designable chains."""
        return self.complex_input.get_designable_sequences()
    
    def get_fixed_residues_str(self) -> str:
        """Get NA-MPNN --fixed_residues argument."""
        return self.complex_input.get_fixed_residues_str()
    
    def get_chains_to_design(self) -> List[str]:
        """Get list of designable chain IDs."""
        return self.complex_input.get_designable_chains()
    
    @property
    def mean_plddt(self) -> float:
        """Overall mean pLDDT across all chains."""
        if not self.chain_coords:
            return 0.0
        
        all_scores = []
        for coords in self.chain_coords.values():
            all_scores.extend(coords.plddt_scores.tolist())
        
        return float(np.mean(all_scores)) if all_scores else 0.0
    
    @property
    def total_length(self) -> int:
        """Total number of residues across all chains."""
        return self.complex_input.get_total_length()
    
    def update_sequence(self, chain_id: str, new_sequence: str) -> "ComplexState":
        """
        Return a new ComplexState with updated sequence.
        
        Structure information is cleared since sequence changed.
        """
        new_complex_input = self.complex_input.update_sequence(chain_id, new_sequence)
        return ComplexState(
            complex_input=new_complex_input,
            chain_coords=None,  # Clear structure - sequence changed
            pae_matrix=None,
        )
    
    def update_sequences(self, new_sequences: Dict[str, str]) -> "ComplexState":
        """Update multiple sequences at once."""
        new_complex_input = self.complex_input.copy()
        for chain_id, new_seq in new_sequences.items():
            new_complex_input = new_complex_input.update_sequence(chain_id, new_seq)
        
        return ComplexState(
            complex_input=new_complex_input,
            chain_coords=None,
            pae_matrix=None,
        )
    
    def with_structure(
        self,
        chain_coords: Dict[str, ChainCoordinates],
        pae_matrix: np.ndarray = None,
    ) -> "ComplexState":
        """Return new state with structure information added."""
        return ComplexState(
            complex_input=self.complex_input,  # Same input
            chain_coords=chain_coords,
            pae_matrix=pae_matrix,
        )
    
    def copy(self) -> "ComplexState":
        """Create a deep copy."""
        return ComplexState(
            complex_input=self.complex_input.copy(),
            chain_coords=copy.deepcopy(self.chain_coords) if self.chain_coords else None,
            pae_matrix=self.pae_matrix.copy() if self.pae_matrix is not None else None,
        )
    
    def to_dict(self) -> Dict:
        """Convert to AF3-compatible JSON format."""
        return self.complex_input.to_dict()
    
    def __repr__(self) -> str:
        chain_info = ", ".join(
            f"{cid}:{c.molecule_type}"
            for cid, c in self.complex_input.chains.items()
        )
        has_struct = "with structure" if self.chain_coords else "no structure"
        return f"ComplexState([{chain_info}], {has_struct})"


# =============================================================================
# MCTS Node for Complex Design
# =============================================================================

@dataclass
class ComplexNode:
    """
    MCTS node for multi-chain complex design.
    
    Each node represents a state in the design space.
    """
    
    state: ComplexState
    
    # MCTS fields
    parent: Optional['ComplexNode'] = None
    children: List['ComplexNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    
    # Metrics
    convergence_score: float = 0.0
    
    @property
    def mean_plddt(self) -> float:
        return self.state.mean_plddt
    
    def get_reward(self) -> float:
        """Average reward from visits."""
        return self.total_reward / max(1, self.visits)
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """UCT score for selection."""
        import math
        
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.get_reward()
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0
        
        return exploitation + exploration
    
    def add_child(self, child: 'ComplexNode') -> None:
        """Add a child node."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def __repr__(self) -> str:
        return f"ComplexNode(depth={self.depth}, visits={self.visits}, reward={self.get_reward():.3f})"


# =============================================================================
# Factory functions
# =============================================================================

def create_complex_state(
    complex_input: ComplexInput,
) -> ComplexState:
    """Create a ComplexState from ComplexInput."""
    return ComplexState(complex_input=complex_input)


def create_root_node(
    complex_input: ComplexInput,
) -> ComplexNode:
    """Create a root node for MCTS from ComplexInput."""
    state = create_complex_state(complex_input)
    return ComplexNode(state=state, depth=0)


def create_protein_dna_state(
    protein_sequence: str,
    dna_sequence: str,
    protein_designable: bool = True,
    dna_designable: bool = False,
    name: str = "protein_dna_complex",
) -> ComplexState:
    """Convenience function to create a protein-DNA complex state."""
    from .complex_input import create_protein_dna_complex
    
    complex_input = create_protein_dna_complex(
        protein_sequence=protein_sequence,
        dna_sequence=dna_sequence,
        protein_designable=protein_designable,
        dna_designable=dna_designable,
        name=name,
    )
    return ComplexState(complex_input=complex_input)
