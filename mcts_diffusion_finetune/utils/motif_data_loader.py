#!/usr/bin/env python3
"""
Motif Scaffolding Data Loader

Loads and processes motif scaffolding data from PDB files and FASTA sequences.
Handles the DPLM-2 motif scaffolding dataset format.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle

# Bio imports
try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser, PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Chain import Chain
    from Bio.PDB.Residue import Residue
    BIO_AVAILABLE = True
except ImportError:
    print("⚠️ BioPython not available - PDB parsing will be limited")
    BIO_AVAILABLE = False

# Project imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

@dataclass
class MotifData:
    """Data structure for motif scaffolding problems."""
    name: str
    motif_sequence: str
    motif_positions: List[int]  # Positions in full sequence where motif occurs
    full_sequence: str  # Full target sequence (for evaluation)
    motif_structure: Optional[str] = None  # Structure tokens for motif
    motif_coordinates: Optional[np.ndarray] = None  # 3D coordinates for motif
    pdb_file: Optional[str] = None  # Path to motif PDB file
    reference_pdb: Optional[str] = None  # Path to reference (full) PDB file
    scaffold_length_range: Optional[Tuple[int, int]] = None  # Min/max scaffold lengths
    
    def __post_init__(self):
        """Validate motif data after initialization."""
        if not self.motif_sequence:
            raise ValueError("Motif sequence cannot be empty")
        if len(self.motif_positions) != len(self.motif_sequence):
            # If positions not provided, assume motif is at the beginning
            if not self.motif_positions:
                self.motif_positions = list(range(len(self.motif_sequence)))
            else:
                raise ValueError(f"Motif positions ({len(self.motif_positions)}) must match sequence length ({len(self.motif_sequence)})")

class MotifScaffoldingDataLoader:
    """Loader for motif scaffolding datasets following DPLM-2 format."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.motifs = []
        self.aa_sequences = {}
        self.struct_sequences = {}
        
        # Standard amino acid mapping
        self.aa_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        print(f"🔄 Initializing motif scaffolding data loader from {data_dir}")
        self._load_data()
    
    def _load_data(self):
        """Load motif scaffolding data from directory."""
        try:
            # First try to load from FASTA files (simple format)
            self._load_from_fasta_files()
            
            # If no simple data found, try complex PDB format
            if not self.motifs:
                self._load_from_pdb_files()
            
            print(f"✅ Loaded {len(self.motifs)} motif scaffolding problems")
            
        except Exception as e:
            print(f"❌ Failed to load motif data: {e}")
            self.motifs = []
    
    def _load_from_fasta_files(self):
        """Load from simple FASTA format (seq.fasta and struct.fasta)."""
        seq_file = self.data_dir / "seq.fasta"
        struct_file = self.data_dir / "struct.fasta"
        
        if not seq_file.exists():
            print(f"⚠️ Simple sequence file not found: {seq_file}")
            return
        
        # Load motif sequences
        motif_sequences = {}
        try:
            for record in SeqIO.parse(str(seq_file), "fasta"):
                motif_sequences[record.id] = str(record.seq).strip()
            print(f"✅ Loaded {len(motif_sequences)} motif sequences")
        except Exception as e:
            print(f"⚠️ Error loading sequences: {e}")
            return
        
        # Load motif structure tokens (optional)
        motif_structures = {}
        if struct_file.exists():
            try:
                for record in SeqIO.parse(str(struct_file), "fasta"):
                    motif_structures[record.id] = str(record.seq).strip()
                print(f"✅ Loaded {len(motif_structures)} motif structure sequences")
            except Exception as e:
                print(f"⚠️ Error loading structure sequences: {e}")
        
        # Create MotifData objects
        for motif_id, motif_seq in motif_sequences.items():
            if motif_seq and len(motif_seq) > 0:
                motif_struct = motif_structures.get(motif_id, "")
                
                motif_data = MotifData(
                    name=motif_id,
                    motif_sequence=motif_seq,
                    motif_positions=list(range(len(motif_seq))),
                    full_sequence=motif_seq,  # For simple format, motif is the full sequence
                    motif_structure=motif_struct,
                    pdb_file=None,
                    scaffold_length_range=(20, 100)  # Default scaffold length range
                )
                self.motifs.append(motif_data)
                print(f"✅ Added simple motif {motif_id}: {len(motif_seq)} residues")
    
    def _load_from_pdb_files(self):
        """Load from complex PDB format with full structures."""
        aa_seq_file = self.data_dir / "aa_seq.fasta"
        struct_seq_file = self.data_dir / "struct_seq.fasta"
        
        if not aa_seq_file.exists():
            print(f"⚠️ Complex sequence file not found: {aa_seq_file}")
            return
        
        # Load full sequences
        try:
            for record in SeqIO.parse(str(aa_seq_file), "fasta"):
                self.aa_sequences[record.id] = str(record.seq).replace(" ", "").upper()
            print(f"✅ Loaded {len(self.aa_sequences)} full sequences")
        except Exception as e:
            print(f"⚠️ Error loading full sequences: {e}")
            return
        
        # Load structure sequences
        if struct_seq_file.exists():
            try:
                for record in SeqIO.parse(str(struct_seq_file), "fasta"):
                    self.struct_sequences[record.id] = str(record.seq)
                print(f"✅ Loaded {len(self.struct_sequences)} structure sequences")
            except Exception as e:
                print(f"⚠️ Error loading structure sequences: {e}")
        
        # Process PDB files to extract motifs
        self._extract_motifs_from_pdbs()
    
    def _extract_motifs_from_pdbs(self):
        """Extract motifs from PDB files in the data directory."""
        if not BIO_AVAILABLE:
            print("⚠️ BioPython not available - cannot extract motifs from PDB files")
            return
        
        # Find all PDB files
        pdb_files = list(self.data_dir.glob("*_motif.pdb"))
        
        for pdb_file in pdb_files:
            try:
                # Extract PDB ID from filename
                pdb_id = pdb_file.stem.replace("_motif", "")
                
                if pdb_id not in self.aa_sequences:
                    print(f"⚠️ No full sequence found for {pdb_id}")
                    continue
                
                # Extract motif from PDB file
                motif_seq, motif_coords = self._extract_motif_from_pdb(pdb_file)
                
                if not motif_seq:
                    print(f"⚠️ Could not extract motif from {pdb_file}")
                    continue
                
                # Find motif positions in full sequence
                full_seq = self.aa_sequences[pdb_id]
                motif_positions = self._find_motif_positions(motif_seq, full_seq)
                
                if not motif_positions:
                    print(f"⚠️ Motif {motif_seq} not found in full sequence {pdb_id}")
                    continue
                
                # Get structure tokens for motif
                motif_struct_tokens = ""
                if pdb_id in self.struct_sequences:
                    struct_seq = self.struct_sequences[pdb_id]
                    motif_struct_tokens = self._extract_motif_structure_tokens(
                        struct_seq, motif_positions
                    )
                
                # Find reference PDB file
                ref_pdb = self.data_dir / f"{pdb_id}_reference.pdb"
                if not ref_pdb.exists():
                    ref_pdb = self.data_dir / f"{pdb_id}_clean.pdb"
                    if not ref_pdb.exists():
                        ref_pdb = None
                
                # Determine scaffold length range
                scaffold_length_range = self._determine_scaffold_length_range(
                    len(motif_seq), len(full_seq)
                )
                
                motif_data = MotifData(
                    name=pdb_id,
                    motif_sequence=motif_seq,
                    motif_positions=motif_positions,
                    full_sequence=full_seq,
                    motif_structure=motif_struct_tokens,
                    motif_coordinates=motif_coords,
                    pdb_file=str(pdb_file),
                    reference_pdb=str(ref_pdb) if ref_pdb else None,
                    scaffold_length_range=scaffold_length_range
                )
                
                self.motifs.append(motif_data)
                print(f"✅ Added complex motif {pdb_id}: {len(motif_seq)} residues, full length {len(full_seq)}")
                
            except Exception as e:
                print(f"⚠️ Error processing {pdb_file}: {e}")
                continue
    
    def _extract_motif_from_pdb(self, pdb_file: Path) -> Tuple[str, Optional[np.ndarray]]:
        """Extract amino acid sequence and coordinates from motif PDB file."""
        if not BIO_AVAILABLE:
            return "", None
        
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("motif", str(pdb_file))
            
            # Get first chain
            chain = next(structure.get_chains())
            
            sequence = ""
            coordinates = []
            
            for residue in chain:
                if residue.get_resname() in self.aa_map:
                    # Add amino acid to sequence
                    sequence += self.aa_map[residue.get_resname()]
                    
                    # Extract CA coordinates
                    if 'CA' in residue:
                        ca_coord = residue['CA'].get_coord()
                        coordinates.append(ca_coord)
                    else:
                        # If no CA, use centroid of heavy atoms
                        coords = [atom.get_coord() for atom in residue.get_atoms() if atom.element != 'H']
                        if coords:
                            centroid = np.mean(coords, axis=0)
                            coordinates.append(centroid)
                        else:
                            coordinates.append([0.0, 0.0, 0.0])  # Fallback
            
            coords_array = np.array(coordinates) if coordinates else None
            
            return sequence, coords_array
            
        except Exception as e:
            print(f"⚠️ Error extracting motif from {pdb_file}: {e}")
            return "", None
    
    def _find_motif_positions(self, motif_seq: str, full_seq: str) -> List[int]:
        """Find positions where motif occurs in full sequence."""
        positions = []
        start = 0
        
        # Find all occurrences of motif in full sequence
        while True:
            pos = full_seq.find(motif_seq, start)
            if pos == -1:
                break
            positions.extend(range(pos, pos + len(motif_seq)))
            start = pos + 1
        
        # Remove duplicates and sort
        positions = sorted(list(set(positions)))
        
        # If multiple occurrences, take the first one
        if len(positions) > len(motif_seq):
            positions = positions[:len(motif_seq)]
        
        return positions
    
    def _extract_motif_structure_tokens(self, struct_seq: str, motif_positions: List[int]) -> str:
        """Extract structure tokens corresponding to motif positions."""
        try:
            if not struct_seq or not motif_positions:
                return ""
            
            # Structure sequence is comma-separated
            struct_tokens = struct_seq.split(',')
            
            # Extract tokens at motif positions
            motif_tokens = []
            for pos in motif_positions:
                if pos < len(struct_tokens):
                    motif_tokens.append(struct_tokens[pos].strip())
            
            return ','.join(motif_tokens)
            
        except Exception as e:
            print(f"⚠️ Error extracting motif structure tokens: {e}")
            return ""
    
    def _determine_scaffold_length_range(self, motif_length: int, full_length: int) -> Tuple[int, int]:
        """Determine reasonable scaffold length range for motif."""
        scaffold_length = full_length - motif_length
        
        if scaffold_length <= 0:
            # If motif is same length as full sequence, use reasonable range
            return (20, 100)
        
        # Allow 50% variation around the actual scaffold length
        min_length = max(10, int(scaffold_length * 0.5))
        max_length = min(200, int(scaffold_length * 1.5))
        
        return (min_length, max_length)
    
    def get_motifs(self) -> List[MotifData]:
        """Get all loaded motifs."""
        return self.motifs
    
    def get_motif_by_name(self, name: str) -> Optional[MotifData]:
        """Get specific motif by name."""
        for motif in self.motifs:
            if motif.name == name:
                return motif
        return None
    
    def filter_motifs(self, min_motif_length: int = 5, max_motif_length: int = 50,
                     min_full_length: int = 30, max_full_length: int = 300) -> List[MotifData]:
        """Filter motifs by length criteria."""
        filtered = []
        
        for motif in self.motifs:
            if (min_motif_length <= len(motif.motif_sequence) <= max_motif_length and
                min_full_length <= len(motif.full_sequence) <= max_full_length):
                filtered.append(motif)
        
        print(f"🔍 Filtered {len(filtered)}/{len(self.motifs)} motifs by length criteria")
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.motifs:
            return {}
        
        motif_lengths = [len(m.motif_sequence) for m in self.motifs]
        full_lengths = [len(m.full_sequence) for m in self.motifs]
        
        stats = {
            'total_motifs': len(self.motifs),
            'motif_length_range': (min(motif_lengths), max(motif_lengths)),
            'motif_length_mean': np.mean(motif_lengths),
            'full_length_range': (min(full_lengths), max(full_lengths)),
            'full_length_mean': np.mean(full_lengths),
            'has_structure_tokens': sum(1 for m in self.motifs if m.motif_structure),
            'has_coordinates': sum(1 for m in self.motifs if m.motif_coordinates is not None),
            'has_pdb_files': sum(1 for m in self.motifs if m.pdb_file)
        }
        
        return stats














