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
class StructureTemplate:
    """
    Structural template configuration for guiding structure prediction.
    
    Used by AF3 and Boltz to provide structural priors during prediction.
    
    Content (mutually exclusive):
        mmcif: mmCIF content as string (AF3)
        path: Path to CIF/PDB file (AF3/Boltz)
    
    Residue mapping (0-indexed, for partial templates):
        query_indices: Which query residues to apply the template to
        template_indices: Corresponding residue indices in the template
    
    Boltz-specific options:
        chain_id: Chain ID in the template file to use
        force: If True, applies a potential to enforce the template during diffusion
        threshold: Max allowed deviation from template in Å (required if force=True)
    """
    # Template content (mutually exclusive)
    mmcif: Optional[str] = None        # mmCIF content as string (AF3)
    path: Optional[str] = None         # Path to CIF/PDB file (AF3/Boltz)
    
    # Residue mapping (0-indexed, for partial template)
    query_indices: Optional[List[int]] = None     # Query residues using template
    template_indices: Optional[List[int]] = None  # Corresponding template residues
    
    # Boltz-specific options
    chain_id: Optional[str] = None     # Chain ID in template file
    force: bool = False                # Enforce template during prediction
    threshold: Optional[float] = None  # Max deviation from template (Å)
    
    def __post_init__(self):
        # Validate mutual exclusivity
        if self.mmcif and self.path:
            raise ValueError("Cannot specify both mmcif and path - they are mutually exclusive")
        
        # Validate force requires threshold
        if self.force and self.threshold is None:
            raise ValueError("threshold is required when force=True")
        
        # Validate index lists have same length
        if (self.query_indices is None) != (self.template_indices is None):
            raise ValueError("query_indices and template_indices must both be specified or both be None")
        if self.query_indices is not None and self.template_indices is not None:
            if len(self.query_indices) != len(self.template_indices):
                raise ValueError(
                    f"query_indices ({len(self.query_indices)}) and template_indices "
                    f"({len(self.template_indices)}) must have the same length"
                )
    
    @property
    def has_content(self) -> bool:
        """Check if template has actual content (mmcif or path)."""
        return self.mmcif is not None or self.path is not None


@dataclass
class ProteinChain:
    """
    Specification for a protein chain.
    
    Attributes:
        chain_id: Unique identifier for the chain
        sequence: Amino acid sequence
        fixed_sequence_positions: 1-indexed positions to keep fixed during inverse folding
        modifications: List of modifications [{"ptmType": "HY3", "ptmPosition": 1}]
        template: Optional structural template configuration
    
    For partial templates (scaffolding):
        Use template.query_indices to specify which residues USE the template
        Residues NOT in query_indices will be predicted de novo
    """
    chain_id: str
    sequence: str
    fixed_sequence_positions: Optional[List[int]] = None
    modifications: Optional[List[Dict]] = None
    template: Optional[StructureTemplate] = None
    
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
    
    @property
    def has_template(self) -> bool:
        """Check if this chain has a template configured."""
        return self.template is not None and self.template.has_content
    
    def fix_all(self) -> 'ProteinChain':
        """Return a copy with all positions fixed (not designable)."""
        return ProteinChain(
            chain_id=self.chain_id,
            sequence=self.sequence,
            fixed_sequence_positions=list(range(1, len(self.sequence) + 1)),
            modifications=self.modifications,
            template=self.template,
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
        template: StructureTemplate = None,
    ) -> "ComplexInput":
        """Add a protein chain."""
        if chain_id in self.chains:
            raise ValueError(f"Chain {chain_id} already exists")
        
        self.chains[chain_id] = ProteinChain(
            chain_id=chain_id,
            sequence=sequence,
            fixed_sequence_positions=fixed_sequence_positions,
            modifications=modifications,
            template=template,
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
                template=chain.template,
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
                
                # Add template if specified
                if chain.has_template:
                    template_entry = {}
                    tpl = chain.template
                    
                    # mmCIF content or path
                    if tpl.mmcif:
                        template_entry["mmcif"] = tpl.mmcif
                    elif tpl.path:
                        template_entry["mmcifPath"] = tpl.path
                    
                    # Residue mapping (query_indices/template_indices)
                    if tpl.query_indices is not None and tpl.template_indices is not None:
                        template_entry["queryIndices"] = tpl.query_indices
                        template_entry["templateIndices"] = tpl.template_indices
                    elif chain.fixed_sequence_positions:
                        # Fallback: use fixed_sequence_positions for full coverage
                        template_entry["queryIndices"] = [p - 1 for p in chain.fixed_sequence_positions]
                        template_entry["templateIndices"] = list(range(len(chain.fixed_sequence_positions)))
                    
                    protein_dict["templates"] = [template_entry]
                
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
    
    # -------------------------------------------------------------------------
    # Boltz Format Export
    # -------------------------------------------------------------------------
    
    def to_boltz_fasta(self, path: Union[str, Path] = None) -> str:
        """
        Convert to Boltz FASTA format.
        
        Format: >chain_id|entity_type
        
        Args:
            path: Optional output file path
            
        Returns:
            FASTA string
        """
        lines = []
        
        for chain_id in self._chain_order:
            chain = self.chains[chain_id]
            entity_type = chain.molecule_type
            
            if isinstance(chain, Ligand):
                # Ligands use SMILES or CCD code
                if chain.smiles:
                    lines.append(f">{chain_id}|ligand")
                    lines.append(chain.smiles)
                elif chain.ccd_code:
                    lines.append(f">{chain_id}|ligand")
                    lines.append(chain.ccd_code)
            else:
                # Protein/DNA/RNA use sequence
                lines.append(f">{chain_id}|{entity_type}")
                lines.append(chain.sequence)
        
        fasta_content = "\n".join(lines) + "\n"
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fasta_content)
        
        return fasta_content
    
    def to_boltz_yaml_dict(self) -> Dict:
        """
        Convert to Boltz YAML format dictionary.
        
        Returns:
            Dict for Boltz YAML input
        """
        sequences = []
        
        for chain_id in self._chain_order:
            chain = self.chains[chain_id]
            
            if isinstance(chain, ProteinChain):
                entry = {
                    "protein": {
                        "id": chain.chain_id,
                        "sequence": chain.sequence,
                    }
                }
                if chain.modifications:
                    entry["protein"]["modifications"] = [
                        {"position": m["ptmPosition"], "ccd": m["ptmType"]}
                        for m in chain.modifications
                    ]
                sequences.append(entry)
                
            elif isinstance(chain, DNAChain):
                entry = {
                    "dna": {
                        "id": chain.chain_id,
                        "sequence": chain.sequence,
                    }
                }
                if chain.modifications:
                    entry["dna"]["modifications"] = [
                        {"position": m["basePosition"], "ccd": m["modificationType"]}
                        for m in chain.modifications
                    ]
                sequences.append(entry)
                
            elif isinstance(chain, RNAChain):
                entry = {
                    "rna": {
                        "id": chain.chain_id,
                        "sequence": chain.sequence,
                    }
                }
                if chain.modifications:
                    entry["rna"]["modifications"] = [
                        {"position": m["basePosition"], "ccd": m["modificationType"]}
                        for m in chain.modifications
                    ]
                sequences.append(entry)
                
            elif isinstance(chain, Ligand):
                entry = {"ligand": {"id": chain.chain_id}}
                if chain.smiles:
                    entry["ligand"]["smiles"] = chain.smiles
                elif chain.ccd_code:
                    entry["ligand"]["ccd"] = chain.ccd_code
                sequences.append(entry)
        
        result = {"sequences": sequences}
        
        # Add constraints (bonds) if present
        if self.bonds:
            constraints = []
            for bond in self.bonds:
                constraints.append({
                    "bond": {
                        "atom1": [bond.chain1, bond.residue1, bond.atom1],
                        "atom2": [bond.chain2, bond.residue2, bond.atom2],
                    }
                })
            result["constraints"] = constraints
        
        # Add templates section (Boltz format)
        templates = []
        for chain_id in self._chain_order:
            chain = self.chains[chain_id]
            if isinstance(chain, ProteinChain) and chain.has_template:
                template_entry = {}
                tpl = chain.template
                
                # File path (cif or pdb)
                if tpl.path:
                    if tpl.path.lower().endswith('.pdb'):
                        template_entry["pdb"] = tpl.path
                    else:
                        template_entry["cif"] = tpl.path
                
                # Chain mapping
                template_entry["chain_id"] = chain.chain_id
                if tpl.chain_id:
                    template_entry["template_id"] = tpl.chain_id
                
                # Force constraint (Boltz-specific)
                if tpl.force:
                    template_entry["force"] = True
                    if tpl.threshold:
                        template_entry["threshold"] = tpl.threshold
                
                templates.append(template_entry)
        
        if templates:
            result["templates"] = templates
        
        return result
    
    def to_boltz_yaml(self, path: Union[str, Path] = None) -> str:
        """
        Convert to Boltz YAML format string.
        
        Args:
            path: Optional output file path
            
        Returns:
            YAML string
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML export. Install with: pip install pyyaml")
        
        yaml_dict = self.to_boltz_yaml_dict()
        yaml_content = yaml.dump(yaml_dict, default_flow_style=False, sort_keys=False)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml_content)
        
        return yaml_content
    
    # -------------------------------------------------------------------------
    # Chai-1 Format Export
    # -------------------------------------------------------------------------
    
    def to_chai_fasta(self, path: Union[str, Path] = None) -> str:
        """
        Convert to Chai-1 FASTA format.
        
        Format: >entity_type|name=chain_id
        
        Args:
            path: Optional output file path
            
        Returns:
            FASTA string
        """
        lines = []
        
        for chain_id in self._chain_order:
            chain = self.chains[chain_id]
            entity_type = chain.molecule_type
            
            if isinstance(chain, Ligand):
                # Ligands use SMILES
                if chain.smiles:
                    lines.append(f">ligand|name={chain_id}")
                    lines.append(chain.smiles)
                elif chain.ccd_code:
                    # Chai-1 prefers SMILES, but can use CCD as name
                    lines.append(f">ligand|name={chain_id}")
                    lines.append(chain.ccd_code)
            else:
                # Protein/DNA/RNA use sequence
                lines.append(f">{entity_type}|name={chain_id}")
                lines.append(chain.sequence)
        
        fasta_content = "\n".join(lines) + "\n"
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(fasta_content)
        
        return fasta_content
    
    # -------------------------------------------------------------------------
    # AF3 JSON (alias for consistency)
    # -------------------------------------------------------------------------
    
    def to_af3_json(self, path: Union[str, Path] = None, indent: int = 2) -> str:
        """
        Convert to AF3 JSON format string.
        
        Args:
            path: Optional output file path
            indent: JSON indentation
            
        Returns:
            JSON string
        """
        json_content = json.dumps(self.to_dict(), indent=indent)
        
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json_content)
        
        return json_content
    
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


# =============================================================================
# Template Helper Functions
# =============================================================================

def set_partial_template(
    complex_input: ComplexInput,
    chain_id: str,
    query_indices: List[int],
    template_indices: List[int],
    template_path: str = None,
    template_mmcif: str = None,
    force: bool = False,
    threshold: float = None,
    template_chain_id: str = None,
) -> ComplexInput:
    """
    Set up a partial template with explicit AF3-style residue indices.
    
    Args:
        complex_input: The ComplexInput to modify
        chain_id: Chain ID to set template for
        query_indices: 0-indexed query residue indices that use the template
        template_indices: 0-indexed indices of corresponding residues in template
        template_path: Path to CIF/PDB template file
        template_mmcif: mmCIF content as string (mutually exclusive with template_path)
        force: Boltz force constraint (enforce template during prediction)
        threshold: Max deviation from template in Å (required if force=True)
        template_chain_id: Chain ID in template file (optional)
        
    Returns:
        The modified ComplexInput (modified in place, also returned for chaining)
        
    Example:
        # Template residues 0-29 and 60-99, skip 30-59 for de novo design
        set_partial_template(
            complex_input,
            chain_id="A",
            query_indices=list(range(30)) + list(range(60, 100)),
            template_indices=list(range(70)),  # 70 residues in template
            template_path="scaffold.cif",
        )
    """
    chain = complex_input.chains.get(chain_id)
    if not isinstance(chain, ProteinChain):
        raise ValueError(f"Chain {chain_id} is not a ProteinChain")
    
    # Create and assign StructureTemplate (validation happens in dataclass)
    chain.template = StructureTemplate(
        mmcif=template_mmcif,
        path=template_path,
        query_indices=query_indices,
        template_indices=template_indices,
        chain_id=template_chain_id,
        force=force,
        threshold=threshold,
    )
    
    return complex_input


def set_full_template(
    complex_input: ComplexInput,
    chain_id: str,
    template_path: str = None,
    template_mmcif: str = None,
    force: bool = False,
    threshold: float = None,
    template_chain_id: str = None,
) -> ComplexInput:
    """
    Set up a full template (all residues use template).
    
    Args:
        complex_input: The ComplexInput to modify
        chain_id: Chain ID to set template for
        template_path: Path to CIF/PDB template file
        template_mmcif: mmCIF content as string (mutually exclusive with template_path)
        force: Boltz force constraint
        threshold: Max deviation from template in Å (required if force=True)
        template_chain_id: Chain ID in template file (optional)
        
    Returns:
        The modified ComplexInput
    """
    chain = complex_input.chains.get(chain_id)
    if not isinstance(chain, ProteinChain):
        raise ValueError(f"Chain {chain_id} is not a ProteinChain")
    
    if not template_path and not template_mmcif:
        raise ValueError("Either template_path or template_mmcif must be provided")
    
    # Full coverage: all residues
    n = len(chain.sequence)
    query_indices = list(range(n))
    template_indices = list(range(n))
    
    # Create and assign StructureTemplate
    chain.template = StructureTemplate(
        mmcif=template_mmcif,
        path=template_path,
        query_indices=query_indices,
        template_indices=template_indices,
        chain_id=template_chain_id,
        force=force,
        threshold=threshold,
    )
    
    return complex_input
