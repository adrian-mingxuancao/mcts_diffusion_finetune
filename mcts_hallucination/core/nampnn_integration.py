"""
NA-MPNN Integration for Nucleic Acid Inverse Folding

Wrapper around NA-MPNN (https://github.com/baker-laboratory/NA-MPNN) for
inverse folding of protein-DNA/RNA complexes.
"""

import numpy as np
from typing import Dict, Optional, List, Union
import tempfile
import subprocess
import shutil
from pathlib import Path


class NAMPNNIntegration:
    """Integration with NA-MPNN for nucleic acid sequence design."""
    
    # Residue type mappings (from NA-MPNN)
    RESTYPE_3TO1 = {
        # Protein
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'UNK': 'X',
        # DNA
        'DA': 'a', 'DC': 'c', 'DG': 'g', 'DT': 't', 'DX': 'x',
        # RNA
        'A': 'b', 'C': 'd', 'G': 'h', 'U': 'u', 'RX': 'y',
    }
    
    RESTYPE_1TO3 = {v: k for k, v in RESTYPE_3TO1.items()}
    
    # Amino acids for random protein sequence generation
    AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
    # DNA bases (single letter)
    DNA_BASES = "acgt"
    # RNA bases (single letter)
    RNA_BASES = "bdhu"
    
    def __init__(
        self,
        nampnn_path: str = None,
        checkpoint_path: str = None,
        mode: str = "design",
        use_mock: bool = False,
        device: str = "cuda",
        temperature: float = 0.1,
        batch_size: int = 1,
        num_samples: int = 1,
        na_shared_tokens: bool = True,
    ):
        """
        Initialize NA-MPNN integration.
        
        Args:
            nampnn_path: Path to NA-MPNN repository (auto-detects from cwd if None)
            checkpoint_path: Path to model checkpoint (auto-detects based on mode)
            mode: "design" or "specificity"
            use_mock: Use mock mode for testing
            device: CUDA device for inference
            temperature: Sampling temperature (default 0.1 for design, 0.6 for specificity)
            batch_size: Number of sequences per batch
            num_samples: Number of samples to generate
            na_shared_tokens: Use shared tokens for DNA/RNA (recommended)
        """
        self.use_mock = use_mock
        self.device = device
        self.mode = mode
        self.temperature = temperature
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.na_shared_tokens = na_shared_tokens
        
        # Find NA-MPNN path
        if nampnn_path:
            self.nampnn_path = Path(nampnn_path)
        else:
            # Try to find NA-MPNN relative to this file
            self.nampnn_path = self._find_nampnn_path()
        
        # Set checkpoint path
        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = self._get_default_checkpoint()
        
        if not use_mock:
            self._check_nampnn_available()
        
        mode_str = "MOCK MODE" if use_mock else "REAL MODE"
        print(f"🔧 NA-MPNN Integration initialized ({mode_str})")
        if not use_mock:
            print(f"   NA-MPNN path: {self.nampnn_path}")
            print(f"   Checkpoint: {self.checkpoint_path}")
            print(f"   Mode: {mode}, Temperature: {temperature}")
    
    def _find_nampnn_path(self) -> Path:
        """Find NA-MPNN repository path."""
        # Look in common locations relative to this file
        current_file = Path(__file__).resolve()
        
        # Check in extra/ directory
        extra_path = current_file.parents[1] / "extra" / "NA-MPNN"
        if extra_path.exists():
            return extra_path
        
        # Check in parent directories
        for parent in current_file.parents:
            nampnn = parent / "NA-MPNN"
            if nampnn.exists():
                return nampnn
        
        # Return a placeholder for mock mode
        return Path("NA-MPNN")
    
    def _get_default_checkpoint(self) -> Path:
        """Get default checkpoint based on mode."""
        if self.mode == "design":
            return self.nampnn_path / "models" / "design_model" / "s_19137.pt"
        elif self.mode == "specificity":
            return self.nampnn_path / "models" / "specificity_model" / "s_70114.pt"
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose 'design' or 'specificity'")
    
    def _check_nampnn_available(self):
        """Check if NA-MPNN is properly set up."""
        if not self.nampnn_path.exists():
            raise RuntimeError(
                f"NA-MPNN not found at {self.nampnn_path}. "
                "Please set nampnn_path to the NA-MPNN repository directory."
            )
        
        run_script = self.nampnn_path / "inference" / "run.py"
        if not run_script.exists():
            raise RuntimeError(
                f"NA-MPNN run.py not found at {run_script}. "
                "Please ensure the NA-MPNN repository is properly cloned."
            )
        
        if not self.checkpoint_path.exists():
            raise RuntimeError(
                f"NA-MPNN checkpoint not found at {self.checkpoint_path}. "
                "Please download the model weights."
            )
    
    def design_sequence(
        self,
        coordinates: Optional[np.ndarray] = None,
        masked_sequence: Optional[str] = None,
        pdb_path: Optional[str] = None,
        chains_to_design: Optional[List[str]] = None,
        design_na_only: bool = False,
        fixed_residues: Optional[str] = None,
    ) -> Union[str, Dict]:
        """
        Design sequences from structure using NA-MPNN.
        
        This method provides interface compatibility with ProteinMPNNIntegration.
        
        Args:
            coordinates: (N, 3) atom coordinates (for interface compatibility)
            masked_sequence: Masked sequence string (for interface compatibility)
            pdb_path: Path to input PDB file
            chains_to_design: List of chain IDs to design (None = all chains)
            design_na_only: Only design nucleic acid chains (keep protein fixed)
            fixed_residues: Space-separated list of fixed residues (e.g., "A12 A13 B2")
            
        Returns:
            If called with just coordinates/masked_sequence: returns sequence string
            If called with pdb_path: returns Dict with sequences, confidence, etc.
        """
        # Interface compatibility mode: just coordinates and masked_sequence
        if coordinates is not None and pdb_path is None:
            if self.use_mock:
                result = self._mock_design(coordinates)
                return result["sequences"][0] if result["sequences"] else ""
            else:
                # Need to create temp PDB for real mode
                seq_len = len(masked_sequence) if masked_sequence else len(coordinates)
                return self.design_from_coords(coordinates, seq_len, "protein")
        
        # Full API mode with pdb_path
        if self.use_mock:
            return self._mock_design(coordinates)
        else:
            return self._real_design(
                pdb_path,
                chains_to_design,
                design_na_only,
                fixed_residues,
            )
    
    def _real_design(
        self,
        pdb_path: str,
        chains_to_design: Optional[List[str]],
        design_na_only: bool,
        fixed_residues: Optional[str],
    ) -> Dict:
        """Real NA-MPNN design using CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            
            # Build command
            cmd = [
                "python", str(self.nampnn_path / "inference" / "run.py"),
                "--model_type", "na_mpnn",
                "--mode", self.mode,
                "--pdb_path", str(pdb_path),
                "--out_folder", str(output_dir),
                "--checkpoint_na_mpnn", str(self.checkpoint_path),
                "--temperature", str(self.temperature),
                "--batch_size", str(self.batch_size),
                "--number_of_batches", str(self.num_samples),
                "--na_shared_tokens", "1" if self.na_shared_tokens else "0",
                "--output_sequences", "1",
                "--output_pdbs", "0",
            ]
            
            if chains_to_design:
                cmd.extend(["--chains_to_design", ",".join(chains_to_design)])
            
            if design_na_only:
                cmd.extend(["--design_na_only", "1"])
            
            if fixed_residues:
                cmd.extend(["--fixed_residues", fixed_residues])
            
            print(f"      Running NA-MPNN design...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.nampnn_path / "inference"),
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"NA-MPNN failed: {result.stderr.strip()}")
            
            # Parse output
            return self._parse_design_output(output_dir, pdb_path)
    
    def _parse_design_output(self, output_dir: Path, pdb_path: str) -> Dict:
        """Parse NA-MPNN design output."""
        # Find FASTA output
        fasta_files = list((output_dir / "seqs").glob("*.fa*"))
        if not fasta_files:
            raise RuntimeError("NA-MPNN produced no FASTA output")
        
        sequences = []
        confidences = []
        native_sequence = None
        
        with open(fasta_files[0]) as f:
            current_seq = ""
            for line in f:
                if line.startswith(">"):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                    
                    # Parse header for confidence
                    if "overall_confidence=" in line:
                        try:
                            conf = float(line.split("overall_confidence=")[1].split()[0])
                            confidences.append(conf)
                        except (ValueError, IndexError):
                            confidences.append(0.5)
                    
                    # First entry is native sequence
                    if "native" in line.lower() or len(sequences) == 0:
                        # Mark to capture as native
                        pass
                else:
                    current_seq += line.strip()
            
            if current_seq:
                sequences.append(current_seq)
        
        # First sequence is typically the native
        if len(sequences) > 1:
            native_sequence = sequences[0]
            sequences = sequences[1:]  # Designed sequences only
        elif sequences:
            native_sequence = sequences[0]
        
        return {
            "sequences": sequences,
            "native_sequence": native_sequence,
            "confidence": confidences[0] if confidences else 0.5,
            "log_probs": None,  # Would need to parse stats file for this
        }
    
    def _mock_design(self, coordinates: Optional[np.ndarray] = None) -> Dict:
        """Mock sequence design for testing."""
        # Determine sequence length
        if coordinates is not None:
            n = len(coordinates)
        else:
            n = 50  # Default mock length
        
        # Generate random sequences (protein-like)
        sequences = []
        for _ in range(max(1, self.num_samples)):
            # Generate a mix of protein and DNA (simplified mock)
            # In real usage, this would be determined by the input structure
            seq = ''.join(np.random.choice(list(self.AMINO_ACIDS), size=n))
            sequences.append(seq)
        
        return {
            "sequences": sequences,
            "native_sequence": ''.join(np.random.choice(list(self.AMINO_ACIDS), size=n)),
            "confidence": np.random.uniform(0.5, 0.9),
            "log_probs": None,
        }
    
    def design_from_coords(
        self,
        coordinates: np.ndarray,
        sequence_length: int,
        molecule_type: str = "protein",
    ) -> str:
        """
        Design sequence from coordinates directly (creates temp PDB).
        
        Args:
            coordinates: (N, 3) coordinates
            sequence_length: Length of sequence
            molecule_type: "protein", "dna", or "rna"
            
        Returns:
            Designed sequence string
        """
        if self.use_mock:
            result = self._mock_design(coordinates)
            return result["sequences"][0] if result["sequences"] else ""
        
        # For real mode, we need to create a temporary PDB
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            pdb_path = f.name
            self._write_temp_pdb(coordinates, sequence_length, molecule_type, pdb_path)
        
        try:
            result = self._real_design(pdb_path, None, False, None)
            return result["sequences"][0] if result["sequences"] else ""
        finally:
            Path(pdb_path).unlink(missing_ok=True)
    
    def _write_temp_pdb(
        self,
        coordinates: np.ndarray,
        sequence_length: int,
        molecule_type: str,
        output_path: str,
    ):
        """Write coordinates to a temporary PDB file.
        
        For nucleic acids, NA-MPNN requires multiple backbone atoms per residue:
        P, OP1, OP2, O5', C5', C4', O4', C3', O3', C2', O2' (RNA only), C1'
        
        We generate these atoms with approximate B-DNA/A-RNA geometry offsets
        from the provided C1' coordinates.
        """
        with open(output_path, 'w') as f:
            atom_num = 1
            
            for i in range(min(len(coordinates), sequence_length)):
                x, y, z = coordinates[i]  # This is the C1' position
                
                if molecule_type == "protein":
                    resname = "ALA"
                    # For protein, write CA atom
                    f.write(
                        f"ATOM  {atom_num:5d}  CA  {resname:3s} A{i+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 70.00           C\n"
                    )
                    atom_num += 1
                else:
                    # DNA or RNA - need full backbone atoms
                    if molecule_type == "dna":
                        # Cycle through DNA bases for variety
                        bases = ["DA", "DC", "DG", "DT"]
                        resname = bases[i % 4]
                    else:  # rna
                        bases = ["A", "C", "G", "U"]
                        resname = bases[i % 4]
                    
                    # Generate backbone atoms with approximate offsets from C1'
                    # These offsets are approximate B-DNA geometry
                    backbone_atoms = [
                        ("P",    x - 1.5, y + 6.0, z - 0.5),
                        ("OP1",  x - 2.5, y + 6.5, z - 0.5),
                        ("OP2",  x - 0.5, y + 6.5, z - 0.5),
                        ("O5'",  x - 1.5, y + 4.5, z + 0.0),
                        ("C5'",  x - 1.0, y + 3.5, z + 0.5),
                        ("C4'",  x - 0.5, y + 2.5, z + 1.0),
                        ("O4'",  x + 0.5, y + 2.0, z + 0.5),
                        ("C3'",  x - 1.5, y + 2.0, z + 1.5),
                        ("O3'",  x - 2.0, y + 1.0, z + 2.0),
                        ("C2'",  x - 0.5, y + 1.5, z + 1.5),
                        ("C1'",  x, y, z),
                    ]
                    
                    # Add O2' for RNA only
                    if molecule_type == "rna":
                        backbone_atoms.insert(-1, ("O2'", x - 0.5, y + 0.5, z + 2.0))
                    
                    for atom_name, ax, ay, az in backbone_atoms:
                        # Format atom name for PDB (left-justified in 4 chars)
                        if len(atom_name) < 4:
                            atom_name_fmt = f" {atom_name:<3s}"
                        else:
                            atom_name_fmt = atom_name[:4]
                        
                        # Get element from first character
                        element = atom_name[0]
                        
                        f.write(
                            f"ATOM  {atom_num:5d} {atom_name_fmt} {resname:3s} A{i+1:4d}    "
                            f"{ax:8.3f}{ay:8.3f}{az:8.3f}  1.00 70.00           {element:>2s}\n"
                        )
                        atom_num += 1
            
            f.write("END\n")
    
    @staticmethod
    def sequence_to_one_letter(sequence_3: List[str]) -> str:
        """Convert 3-letter codes to 1-letter codes."""
        return ''.join(
            NAMPNNIntegration.RESTYPE_3TO1.get(res.upper(), 'X')
            for res in sequence_3
        )
    
    @staticmethod
    def one_letter_to_three(sequence_1: str) -> List[str]:
        """Convert 1-letter codes to 3-letter codes."""
        return [
            NAMPNNIntegration.RESTYPE_1TO3.get(res, 'UNK')
            for res in sequence_1
        ]
    
    # -------------------------------------------------------------------------
    # Complex Design API
    # -------------------------------------------------------------------------
    
    def design_complex(
        self,
        pdb_path: str,
        chains_to_design: List[str] = None,
        fixed_residues: str = None,
        design_na_only: bool = False,
    ) -> Dict[str, str]:
        """
        Design sequences for a multi-chain complex.
        
        This is the unified interface for complex design, compatible with
        ComplexInput and ComplexState.
        
        Args:
            pdb_path: Path to input PDB file containing the complex
            chains_to_design: List of chain IDs to redesign (None = all)
            fixed_residues: NA-MPNN format string: "A1 A2 A3 B10 B11"
            design_na_only: If True, only redesign nucleic acid chains
            
        Returns:
            Dict mapping chain_id to designed sequence
        """
        if self.use_mock:
            return self._mock_complex_design(chains_to_design)
        
        result = self._real_design(
            pdb_path=pdb_path,
            chains_to_design=chains_to_design,
            design_na_only=design_na_only,
            fixed_residues=fixed_residues,
        )
        
        # Parse sequences by chain (from full sequence output)
        # NA-MPNN outputs concatenated sequence - need to split by chain
        # For now, return the first designed sequence
        if result.get("sequences"):
            return {"designed": result["sequences"][0]}
        return {}
    
    def _mock_complex_design(self, chains_to_design: List[str] = None) -> Dict[str, str]:
        """Mock complex design for testing."""
        if chains_to_design is None:
            chains_to_design = ["A"]
        
        result = {}
        for chain_id in chains_to_design:
            # Generate random protein sequence
            seq_len = np.random.randint(30, 60)
            seq = ''.join(np.random.choice(list(self.AMINO_ACIDS), size=seq_len))
            result[chain_id] = seq
        
        return result
    
    def design_from_complex_input(
        self,
        pdb_path: str,
        complex_input: "ComplexInput",
    ) -> Dict[str, str]:
        """
        Design sequences using ComplexInput specification.
        
        Uses designable chains and fixed_sequence_positions from ComplexInput.
        
        Args:
            pdb_path: Path to predicted structure PDB
            complex_input: ComplexInput with chains and fixed_sequence_positions
            
        Returns:
            Dict mapping chain_id to designed sequence
        """
        chains_to_design = complex_input.get_designable_chains()
        fixed_residues = complex_input.get_fixed_residues_str()
        
        return self.design_complex(
            pdb_path=pdb_path,
            chains_to_design=chains_to_design if chains_to_design else None,
            fixed_residues=fixed_residues if fixed_residues else None,
        )

