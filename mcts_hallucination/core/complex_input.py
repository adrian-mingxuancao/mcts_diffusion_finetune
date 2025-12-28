"""
Complex Input Builder

Dataclasses for constructing multi-chain complex input (AF3-compatible JSON format).
Supports proteins, DNA, RNA, ligands, modifications, templates, and bonds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import json
import copy


# =============================================================================
# Chain Specifications
# =============================================================================

@dataclass
class ProteinChain:
    """Specification for a protein chain."""
    chain_id: str
    sequence: str
    fixed_sequence_positions: Optional[List[int]] = None  # 1-indexed positions to keep fixed during inverse folding
    modifications: Optional[List[Dict]] = None   # [{"ptmType": "HY3", "ptmPosition": 1}]
    template_mmcif: Optional[str] = None         # mmCIF string for template
    template_indices: Optional[List[int]] = None # Mapping for template
    
    @property
    def molecule_type(self) -> str:
        return "protein"
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def is_designable(self) -> bool:
        """A chain is designable if it has any positions that are not fixed."""
        if self.fixed_sequence_positions is None:
            return True  # All positions designable
        return len(self.fixed_sequence_positions) < len(self.sequence)
    
    def fix_all(self) -> 'ProteinChain':
        """Return a copy with all positions fixed (not designable)."""
        return ProteinChain(
            chain_id=self.chain_id,
            sequence=self.sequence,
            fixed_sequence_positions=list(range(1, len(self.sequence) + 1)),
            modifications=self.modifications,
            template_mmcif=self.template_mmcif,
            template_indices=self.template_indices,
        )


@dataclass
class DNAChain:
    """Specification for a DNA chain."""
    chain_id: str
    sequence: str  # ACGT
    fixed_sequence_positions: Optional[List[int]] = None
    modifications: Optional[List[Dict]] = None  # [{"modificationType": "6OG", "basePosition": 1}]
    
    @property
    def molecule_type(self) -> str:
        return "dna"
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def is_designable(self) -> bool:
        """A chain is designable if it has any positions that are not fixed."""
        if self.fixed_sequence_positions is None:
            return True
        return len(self.fixed_sequence_positions) < len(self.sequence)
    
    def fix_all(self) -> 'DNAChain':
        """Return a copy with all positions fixed (not designable)."""
        return DNAChain(
            chain_id=self.chain_id,
            sequence=self.sequence,
            fixed_sequence_positions=list(range(1, len(self.sequence) + 1)),
            modifications=self.modifications,
        )


@dataclass
class RNAChain:
    """Specification for an RNA chain."""
    chain_id: str
    sequence: str  # ACGU
    fixed_sequence_positions: Optional[List[int]] = None
    modifications: Optional[List[Dict]] = None  # [{"modificationType": "2MG", "basePosition": 1}]
    
    @property
    def molecule_type(self) -> str:
        return "rna"
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def is_designable(self) -> bool:
        """A chain is designable if it has any positions that are not fixed."""
        if self.fixed_sequence_positions is None:
            return True
        return len(self.fixed_sequence_positions) < len(self.sequence)
    
    def fix_all(self) -> 'RNAChain':
        """Return a copy with all positions fixed (not designable)."""
        return RNAChain(
            chain_id=self.chain_id,
            sequence=self.sequence,
            fixed_sequence_positions=list(range(1, len(self.sequence) + 1)),
            modifications=self.modifications,
        )


@dataclass
class Ligand:
    """Specification for a ligand."""
    chain_id: str
    ccd_code: Optional[str] = None   # e.g., "ATP", "MG"
    smiles: Optional[str] = None     # SMILES string
    
    @property
    def molecule_type(self) -> str:
        return "ligand"

    def __post_init__(self):
        if not self.ccd_code and not self.smiles:
            raise ValueError("Ligand must have either ccd_code or smiles")
        if self.ccd_code and self.smiles:
            raise ValueError("Ligand cannot have both ccd_code and smiles")


@dataclass
class BondedAtomPair:
    """Covalent bond between two atoms."""
    chain1: str
    residue1: int  # 1-indexed
    atom1: str
    chain2: str
    residue2: int  # 1-indexed
    atom2: str
    
    def to_list(self) -> List:
        return [
            [self.chain1, self.residue1, self.atom1],
            [self.chain2, self.residue2, self.atom2],
        ]


# =============================================================================
# Complex Input Builder
# =============================================================================

ChainType = Union[ProteinChain, DNAChain, RNAChain, Ligand]


class ComplexInput:
    """
    Builder for multi-chain complex input (AF3-compatible JSON format).
    
    Example:
        complex_input = ComplexInput("my_complex")
        complex_input.add_protein("A", "ACDEFGHIKLMNPQRSTVWY")  # Fully designable
        complex_input.add_dna("B", "AGCTGGATCC", fixed_sequence_positions=[1,2,3,4,5,6,7,8,9,10])  # Fixed
        complex_input.add_ligand("C", ccd_code="ATP")
        
        # Generate JSON
        json_dict = complex_input.to_dict()
        complex_input.to_json("/path/to/input.json")
    """
    
    def __init__(self, name: str = "complex", seeds: List[int] = None):
        """
        Initialize ComplexInput builder.
        
        Args:
            name: Name of the job
            seeds: List of random seeds (default: [1])
        """
        self.name = name
        self.seeds = seeds or [1]
        self.chains: Dict[str, ChainType] = {}
        self.bonds: List[BondedAtomPair] = []
        self._chain_order: List[str] = []  # Preserve insertion order
    
    # -------------------------------------------------------------------------
    # Chain addition methods
    # -------------------------------------------------------------------------
    
    def add_protein(
        self,
        chain_id: str,
        sequence: str,
        fixed_sequence_positions: List[int] = None,
        modifications: List[Dict] = None,
        template_mmcif: str = None,
        template_indices: List[int] = None,
    ) -> "ComplexInput":
        """Add a protein chain."""
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        self.chains[chain_id] = ProteinChain(
            chain_id=chain_id,
            sequence=sequence,
            fixed_sequence_positions=fixed_sequence_positions,
            modifications=modifications,
            template_mmcif=template_mmcif,
            template_indices=template_indices,
        )
        self._chain_order.append(chain_id)
        return self
    
    def add_dna(
        self,
        chain_id: str,
        sequence: str,
        fixed_sequence_positions: List[int] = None,
        modifications: List[Dict] = None,
    ) -> "ComplexInput":
        """Add a DNA chain."""
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        # Validate DNA sequence
        valid_bases = set("ACGT")
        if not all(b in valid_bases for b in sequence.upper()):
            raise ValueError(f"Invalid DNA sequence: must contain only ACGT")
        
        self.chains[chain_id] = DNAChain(
            chain_id=chain_id,
            sequence=sequence.upper(),
            fixed_sequence_positions=fixed_sequence_positions,
            modifications=modifications,
        )
        self._chain_order.append(chain_id)
        return self
    
    def add_rna(
        self,
        chain_id: str,
        sequence: str,
        fixed_sequence_positions: List[int] = None,
        modifications: List[Dict] = None,
    ) -> "ComplexInput":
        """Add an RNA chain."""
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        # Validate RNA sequence
        valid_bases = set("ACGU")
        if not all(b in valid_bases for b in sequence.upper()):
            raise ValueError(f"Invalid RNA sequence: must contain only ACGU")
        
        self.chains[chain_id] = RNAChain(
            chain_id=chain_id,
            sequence=sequence.upper(),
            fixed_sequence_positions=fixed_sequence_positions,
            modifications=modifications,
        )
        self._chain_order.append(chain_id)
        return self
    
    def add_ligand(
        self,
        chain_id: str,
        ccd_code: str = None,
        smiles: str = None,
    ) -> "ComplexInput":
        """Add a ligand."""
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        self.chains[chain_id] = Ligand(
            chain_id=chain_id,
            ccd_code=ccd_code,
            smiles=smiles,
        )
        self._chain_order.append(chain_id)
        return self
    
    def add_bond(
        self,
        chain1: str,
        residue1: int,
        atom1: str,
        chain2: str,
        residue2: int,
        atom2: str,
    ) -> "ComplexInput":
        """Add a covalent bond between two atoms."""
        self.bonds.append(BondedAtomPair(
            chain1=chain1, residue1=residue1, atom1=atom1,
            chain2=chain2, residue2=residue2, atom2=atom2,
        ))
        return self
    
    # -------------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------------
    
    def get_designable_chains(self) -> List[str]:
        """Get list of designable chain IDs (chains with at least one designable position)."""
        return [
            chain_id for chain_id, chain in self.chains.items()
            if hasattr(chain, 'is_designable') and chain.is_designable
        ]
    
    def get_designable_sequences(self) -> Dict[str, str]:
        """Get sequences of designable chains."""
        return {
            chain_id: chain.sequence
            for chain_id, chain in self.chains.items()
            if hasattr(chain, 'sequence') and hasattr(chain, 'is_designable') and chain.is_designable
        }
    
    def get_fixed_residues_str(self) -> str:
        """
        Generate NA-MPNN --fixed_residues argument string.
        
        Returns:
            String like "A1 A2 A3 B10 B11" for fixed positions
        """
        parts = []
        for chain_id, chain in self.chains.items():
            if hasattr(chain, 'fixed_sequence_positions') and chain.fixed_sequence_positions:
                for pos in chain.fixed_sequence_positions:
                    parts.append(f"{chain_id}{pos}")
        return " ".join(parts)
    
    def get_chains_to_design_str(self) -> str:
        """
        Generate NA-MPNN --chains_to_design argument string.
        
        Returns:
            Comma-separated chain IDs like "A,B,C"
        """
        return ",".join(self.get_designable_chains())
    
    def get_total_length(self) -> int:
        """Get total number of residues across all chains."""
        return sum(
            chain.length for chain in self.chains.values()
            if hasattr(chain, 'length')
        )
    
    # -------------------------------------------------------------------------
    # Mutation methods
    # -------------------------------------------------------------------------
    
    def update_sequence(self, chain_id: str, new_sequence: str) -> "ComplexInput":
        """
        Return a new ComplexInput with updated sequence for a chain.
        
        Args:
            chain_id: Chain to update
            new_sequence: New sequence
            
        Returns:
            New ComplexInput instance (original unchanged)
        """
        new_input = self.copy()
        chain = new_input.chains[chain_id]
        
        if isinstance(chain, ProteinChain):
            new_input.chains[chain_id] = ProteinChain(
                chain_id=chain_id,
                sequence=new_sequence,
                fixed_sequence_positions=chain.fixed_sequence_positions,
                modifications=chain.modifications,
                template_mmcif=chain.template_mmcif,
                template_indices=chain.template_indices,
            )
        elif isinstance(chain, DNAChain):
            new_input.chains[chain_id] = DNAChain(
                chain_id=chain_id,
                sequence=new_sequence,
                fixed_sequence_positions=chain.fixed_sequence_positions,
                modifications=chain.modifications,
            )
        elif isinstance(chain, RNAChain):
            new_input.chains[chain_id] = RNAChain(
                chain_id=chain_id,
                sequence=new_sequence,
                fixed_sequence_positions=chain.fixed_sequence_positions,
                modifications=chain.modifications,
            )
        else:
            raise ValueError(f"Cannot update sequence for ligand chain {chain_id}")
        
        return new_input
    
    def copy(self) -> "ComplexInput":
        """Create a deep copy of this ComplexInput."""
        new_input = ComplexInput(name=self.name, seeds=list(self.seeds))
        new_input.chains = copy.deepcopy(self.chains)
        new_input.bonds = copy.deepcopy(self.bonds)
        new_input._chain_order = list(self._chain_order)
        return new_input
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict:
        """
        Convert to AF3 JSON format dictionary.
        
        Returns:
            Dict compatible with AlphaFold3 input format
        """
        sequences = []
        
        for chain_id in self._chain_order:
            chain = self.chains[chain_id]
            
            if isinstance(chain, ProteinChain):
                protein_dict = {
                    "id": chain.chain_id,
                    "sequence": chain.sequence,
                }
                if chain.modifications:
                    protein_dict["modifications"] = chain.modifications
                
                # Add template if specified (uses fixed_sequence_positions for structure conditioning)
                if chain.template_mmcif and chain.fixed_sequence_positions:
                    # Convert 1-indexed positions to 0-indexed queryIndices
                    query_indices = [p - 1 for p in chain.fixed_sequence_positions]
                    template_indices = chain.template_indices or list(range(len(query_indices)))
                    protein_dict["templates"] = [{
                        "mmcif": chain.template_mmcif,
                        "queryIndices": query_indices,
                        "templateIndices": template_indices,
                    }]
                
                sequences.append({"protein": protein_dict})
                
            elif isinstance(chain, DNAChain):
                dna_dict = {
                    "id": chain.chain_id,
                    "sequence": chain.sequence,
                }
                if chain.modifications:
                    dna_dict["modifications"] = chain.modifications
                sequences.append({"dna": dna_dict})
                
            elif isinstance(chain, RNAChain):
                rna_dict = {
                    "id": chain.chain_id,
                    "sequence": chain.sequence,
                }
                if chain.modifications:
                    rna_dict["modifications"] = chain.modifications
                sequences.append({"rna": rna_dict})
                
            elif isinstance(chain, Ligand):
                ligand_dict = {"id": chain.chain_id}
                if chain.ccd_code:
                    ligand_dict["ccdCodes"] = [chain.ccd_code]
                elif chain.smiles:
                    ligand_dict["smiles"] = chain.smiles
                sequences.append({"ligand": ligand_dict})
        
        result = {
            "name": self.name,
            "modelSeeds": self.seeds,
            "sequences": sequences,
            "dialect": "alphafold3",
            "version": 1,
        }
        
        if self.bonds:
            result["bondedAtomPairs"] = [bond.to_list() for bond in self.bonds]
        
        return result
    
    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """
        Write AF3 input to JSON file.
        
        Args:
            path: Output file path
            indent: JSON indentation
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def __repr__(self) -> str:
        chain_summary = ", ".join(
            f"{c.chain_id}:{c.molecule_type}" for c in self.chains.values()
        )
        return f"ComplexInput(name={self.name!r}, chains=[{chain_summary}])"


# =============================================================================
# Convenience functions
# =============================================================================

def create_protein_only(
    sequence: str,
    chain_id: str = "A",
    name: str = "hallucination",
) -> ComplexInput:
    """Create ComplexInput for a single protein chain."""
    return ComplexInput(name=name).add_protein(chain_id, sequence)


def create_protein_dna_complex(
    protein_sequence: str,
    dna_sequence: str,
    protein_fixed_positions: List[int] = None,
    dna_fixed_positions: List[int] = None,
    name: str = "protein_dna_complex",
) -> ComplexInput:
    """
    Create ComplexInput for a protein-DNA complex.
    
    Args:
        protein_sequence: Protein sequence
        dna_sequence: DNA sequence
        protein_fixed_positions: Positions to keep fixed in protein (None = all designable)
        dna_fixed_positions: Positions to keep fixed in DNA (None = all designable, 
                            or use list(range(1, len+1)) to fix entire chain)
        name: Job name
    """
    return (
        ComplexInput(name=name)
        .add_protein("A", protein_sequence, fixed_sequence_positions=protein_fixed_positions)
        .add_dna("B", dna_sequence, fixed_sequence_positions=dna_fixed_positions)
    )


def fix_all_positions(sequence_length: int) -> List[int]:
    """Helper to generate a list that fixes all positions in a chain."""
    return list(range(1, sequence_length + 1))


