"""
ABCFold Integration for AlphaFold3 Structure Hallucination

Wrapper around ABCFold (https://github.com/rigdenlab/ABCFold) for structure prediction
from masked sequences in the hallucination pipeline.

Supports:
- AF3 via ABCFold CLI
- Boltz via boltz CLI
- Chai-1 via direct Python API (chai_lab.chai1)
"""

import numpy as np
from typing import Dict, Optional, List
import json
import tempfile
import subprocess
import os
from pathlib import Path


class ABCFoldIntegration:
    """Integration with ABCFold for AF3/Boltz/Chai-1 structure hallucination."""
    
    ENGINE_FLAGS = {
        "af3": "-a",
        "boltz": "-b",
        "chai1": "-c",
    }
    
    def __init__(
        self, 
        model_params: str = None,
        database_dir: str = None,
        use_mmseqs: bool = True,
        use_mock: bool = False,
        engine: str = "af3",
        allow_fallback: bool = False,
        molecule_type: str = "protein",
    ):
        """
        Initialize ABCFold integration.
        
        Args:
            model_params: Path to AF3 model parameters
            database_dir: Path to AF3 databases (not needed if use_mmseqs=True)
            use_mmseqs: Use MMseqs2 for faster MSA (recommended)
            use_mock: Use mock mode for testing (set False for real AF3)
            engine: Which ABCFold backend to call (af3, boltz, chai1)
            allow_fallback: Fall back to mock on errors
            molecule_type: "protein", "dna", or "rna" - determines FASTA/JSON format
        """
        self.model_params = model_params
        self.database_dir = database_dir
        self.use_mmseqs = use_mmseqs
        self.use_mock = use_mock
        self.engine = engine.lower()
        self.allow_fallback = allow_fallback
        self.molecule_type = molecule_type.lower()
        
        if self.engine not in self.ENGINE_FLAGS:
            raise ValueError(f"Unsupported ABCFold engine '{engine}'. Choose from {list(self.ENGINE_FLAGS)}.")
        
        if not use_mock and self.engine == "af3" and not self.model_params:
            raise ValueError("Real AF3 mode requires --model_params pointing to the AlphaFold3 weights directory.")
        
        if use_mock:
            print(f"🔧 ABCFold Integration initialized (MOCK MODE)")
        else:
            print(f"🔧 ABCFold Integration initialized (REAL MODE - {self.engine.upper()})")
            print(f"   Molecule type: {self.molecule_type}")
            if self.engine == "af3":
                print(f"   Model params: {model_params}")
                print(f"   Use MMseqs2: {use_mmseqs}")
    
    def predict_structure(self, sequence: str, num_recycles: int = 3) -> Dict:
        """
        Predict structure from sequence using ABCFold/AF3.
        
        Args:
            sequence: Amino acid sequence (can contain 'X' for masked positions)
            num_recycles: Number of recycling iterations
            
        Returns:
            Dictionary with:
                - coordinates: (N, 3) CA coordinates
                - confidence: (N,) pLDDT scores
                - pae_mean: Mean predicted aligned error
        """
        # Replace X with A for structure prediction (AF3 doesn't handle X)
        clean_sequence = sequence.replace('X', 'A')
        
        if self.use_mock:
            return self._mock_predict(clean_sequence)
        else:
            return self._real_predict(clean_sequence, num_recycles)
    
    def _real_predict(self, sequence: str, num_recycles: int) -> Dict:
        """
        Real prediction using the appropriate backend.
        
        Routes to:
        - Boltz: Direct CLI
        - Chai-1: Direct Python API
        - AF3: ABCFold CLI
        """
        if self.engine == "boltz":
            return self._predict_boltz(sequence, num_recycles)
        elif self.engine == "chai1":
            return self._predict_chai(sequence, num_recycles)
        else:
            return self._predict_abcfold(sequence, num_recycles)
    
    def _predict_boltz(self, sequence: str, num_recycles: int) -> Dict:
        """Run Boltz prediction directly via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create FASTA file in Boltz format: >CHAIN_ID|ENTITY_TYPE
            # Entity type must match molecule_type (protein, dna, rna)
            entity_type = self.molecule_type  # "protein", "dna", or "rna"
            fasta_path = tmpdir / "input.fasta"
            with open(fasta_path, 'w') as f:
                f.write(f">A|{entity_type}\n{sequence}\n")
            
            output_dir = tmpdir / "output"
            
            # Set cache directory to avoid home quota issues
            env = os.environ.copy()
            env['XDG_CACHE_HOME'] = '/net/scratch/caom/.cache'
            env['HF_HOME'] = '/net/scratch/caom/.cache/huggingface'
            env['TORCH_HOME'] = '/net/scratch/caom/.cache/torch'
            
            cmd = [
                "boltz", "predict",
                str(fasta_path),
                "--out_dir", str(output_dir),
                "--use_msa_server",
                "--override",
                "--recycling_steps", str(num_recycles),
            ]
            
            print(f"      Running Boltz: {' '.join(cmd[:6])}...")
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            
            if result.returncode != 0:
                msg = f"Boltz failed: {result.stderr.strip()[-500:]}"
                if self.allow_fallback:
                    print(f"      {msg}\n      Falling back to mock output")
                    return self._mock_predict(sequence)
                raise RuntimeError(msg)
            
            # Find output files
            cif_files = list(output_dir.glob("**/predictions/**/*.cif"))
            plddt_files = list(output_dir.glob("**/predictions/**/plddt*.npz"))
            
            if not cif_files:
                msg = "Boltz produced no CIF output."
                if self.allow_fallback:
                    print(f"      {msg}\n      Falling back to mock output")
                    return self._mock_predict(sequence)
                raise RuntimeError(msg)
            
            # Parse CIF for coordinates
            coords, _, pae_mean = self._parse_cif(cif_files[0], len(sequence))
            
            # Get pLDDT from npz file if available
            if plddt_files:
                plddt_data = np.load(plddt_files[0])
                confidence = plddt_data['plddt']
                # Boltz returns pLDDT in 0-1 range, scale to 0-100
                if confidence.max() <= 1.0:
                    confidence = confidence * 100.0
            else:
                confidence = np.ones(len(sequence)) * 70.0
            
            print(f"      ✅ Boltz prediction complete (mean pLDDT: {np.mean(confidence):.1f})")
            
            # Convert CIF to PDB for downstream tools like NA-MPNN
            pdb_path = self._cif_to_pdb(cif_files[0])
            
            return {
                'coordinates': coords,
                'confidence': confidence,
                'pae_mean': pae_mean,
                'structure_path': pdb_path,  # Full structure file for NA-MPNN
            }
    
    def _predict_chai(self, sequence: str, num_recycles: int) -> Dict:
        """Run Chai-1 prediction via Python API."""
        try:
            from chai_lab.chai1 import run_inference
        except ImportError as e:
            msg = f"Chai-1 not available: {e}"
            if self.allow_fallback:
                print(f"      {msg}\n      Falling back to mock output")
                return self._mock_predict(sequence)
            raise ImportError(msg) from e
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create FASTA file in Chai format: >entity_type|name=...
            # Entity type must match molecule_type (protein, dna, rna)
            entity_type = self.molecule_type  # "protein", "dna", or "rna"
            fasta_path = tmpdir / "input.fasta"
            with open(fasta_path, 'w') as f:
                f.write(f">{entity_type}|name=hallucination\n{sequence}\n")
            
            output_dir = tmpdir / "output"
            
            print(f"      Running Chai-1 inference...")
            
            try:
                candidates = run_inference(
                    fasta_file=fasta_path,
                    output_dir=output_dir,
                    num_trunk_recycles=num_recycles,
                    num_diffn_timesteps=50,  # Faster inference
                    seed=42,
                    device='cuda:0' if self._cuda_available() else 'cpu',
                    use_esm_embeddings=False,
                )
                
                # Parse the best structure
                if candidates.cif_paths:
                    coords, confidence, pae_mean = self._parse_cif(
                        candidates.cif_paths[0], len(sequence)
                    )
                    print(f"      ✅ Chai-1 prediction complete (mean pLDDT: {np.mean(confidence):.1f})")
                    return {
                        'coordinates': coords,
                        'confidence': confidence,
                        'pae_mean': pae_mean
                    }
                else:
                    raise RuntimeError("Chai-1 produced no structures")
                    
            except Exception as e:
                msg = f"Chai-1 inference failed: {e}"
                if self.allow_fallback:
                    print(f"      {msg}\n      Falling back to mock output")
                    return self._mock_predict(sequence)
                raise RuntimeError(msg) from e
    
    def _predict_abcfold(self, sequence: str, num_recycles: int) -> Dict:
        """Run AF3 prediction via ABCFold CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create AF3 input JSON
            input_json = tmpdir / "input.json"
            self._create_af3_json(sequence, input_json)
            
            # Run ABCFold
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            
            cmd = [
                "abcfold",
                str(input_json),
                str(output_dir),
                self.ENGINE_FLAGS[self.engine],
            ]
            
            if self.use_mmseqs:
                cmd.append("--mmseqs2")
            
            if self.model_params:
                cmd.extend(["--model_params", self.model_params])
            
            if self.database_dir and not self.use_mmseqs:
                cmd.extend(["--database", self.database_dir])
            
            cmd.extend([
                "--num_recycles", str(num_recycles),
                "--override", "--no_visuals"
            ])
            
            print(f"      Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                msg = f"ABCFold command failed (engine={self.engine}): {result.stderr.strip()}"
                if self.allow_fallback:
                    print(f"      {msg}\n      Falling back to mock output")
                    return self._mock_predict(sequence)
                raise RuntimeError(msg)
            
            # Parse output CIF file
            cif_files = list(output_dir.glob("**/*.cif"))
            if not cif_files:
                msg = "ABCFold produced no CIF output."
                if self.allow_fallback:
                    print(f"      {msg}\n      Falling back to mock output")
                    return self._mock_predict(sequence)
                raise RuntimeError(msg)
            
            # Parse the best model (usually model_0)
            coords, confidence, pae_mean = self._parse_cif(cif_files[0], len(sequence))
            
            return {
                'coordinates': coords,
                'confidence': confidence,
                'pae_mean': pae_mean
            }
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _create_af3_json(self, sequence: str, output_path: Path):
        """Create AF3 input JSON file with correct molecule type."""
        # Build sequence entry based on molecule_type
        # AF3 JSON uses keys: "protein", "dna", "rna"
        mol_type = self.molecule_type  # "protein", "dna", or "rna"
        
        sequence_entry = {
            mol_type: {
                "id": ["A"],
                "sequence": sequence
            }
        }
        
        af3_input = {
            "name": "hallucination",
            "sequences": [sequence_entry],
            "modelSeeds": [1],
            "dialect": "alphafold3",
            "version": 1
        }
        
        with open(output_path, 'w') as f:
            json.dump(af3_input, f, indent=2)
    
    def _parse_cif(self, cif_path: Path, sequence_length: int):
        """
        Parse CIF file to extract coordinates and confidence.
        
        Returns:
            coords: (N, 3) representative atom coordinates (CA for protein, C1' for NA)
            confidence: (N,) pLDDT scores
            pae_mean: Mean PAE
        """
        try:
            from Bio.PDB.MMCIFParser import MMCIFParser
            
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("hallucination", cif_path)
            
            # Extract representative atom coordinates based on molecule type
            # For protein: CA atom
            # For nucleic acids: C1' atom
            coords = []
            confidence = []
            
            # Determine which atom to use based on molecule_type
            if self.molecule_type == "protein":
                target_atoms = ["CA"]
            else:
                # For DNA/RNA, use C1' as representative atom
                target_atoms = ["C1'", "C1*"]  # C1* is alternative naming
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Try each target atom
                        atom_found = None
                        for atom_name in target_atoms:
                            if atom_name in residue:
                                atom_found = residue[atom_name]
                                break
                        
                        if atom_found is not None:
                            coords.append(atom_found.get_coord())
                            # pLDDT is stored in B-factor column
                            confidence.append(atom_found.get_bfactor())
            
            coords = np.array(coords)
            confidence = np.array(confidence)
            
            # PAE is typically in a separate JSON file
            pae_json = cif_path.parent / f"{cif_path.stem}_confidence.json"
            pae_mean = 8.0  # Default
            if pae_json.exists():
                with open(pae_json) as f:
                    conf_data = json.load(f)
                    if 'pae' in conf_data:
                        pae_mean = np.mean(conf_data['pae'])
            
            return coords, confidence, pae_mean
            
        except Exception as e:
            msg = f"CIF parsing failed: {e}"
            if self.allow_fallback:
                print(f"      {msg}\n      Falling back to mock output")
                mock = self._mock_predict("A" * sequence_length)
                return mock['coordinates'], mock['confidence'], mock['pae_mean']
            raise RuntimeError(msg) from e
    
    def _cif_to_pdb(self, cif_path: Path) -> str:
        """Convert CIF file to PDB format for downstream tools like NA-MPNN.
        
        Returns:
            Path to the converted PDB file (persistent temp file)
        """
        try:
            from Bio.PDB.MMCIFParser import MMCIFParser
            from Bio.PDB import PDBIO
            
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("structure", cif_path)
            
            # Save to a persistent temp file (not in the temp directory that will be deleted)
            pdb_file = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False)
            pdb_path = pdb_file.name
            pdb_file.close()
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(pdb_path)
            
            return pdb_path
        except Exception as e:
            print(f"      Warning: CIF to PDB conversion failed: {e}")
            return None
    
    def _mock_predict(self, sequence: str) -> Dict:
        """Mock structure prediction for testing."""
        n = len(sequence)
        
        # Generate random but reasonable structure
        coords = np.cumsum(np.random.randn(n, 3) * 1.5, axis=0)
        coords += np.random.randn(n, 3) * 2.0
        
        # Generate confidence scores
        confidence = 70 + 20 * np.exp(-((np.arange(n) - n/2)**2) / (n/4)**2)
        confidence += np.random.randn(n) * 5
        confidence = np.clip(confidence, 40, 95)
        
        # Mock PAE
        pae_mean = 8.0 + np.random.randn() * 2.0
        
        return {
            'coordinates': coords,
            'confidence': confidence,
            'pae_mean': max(0, pae_mean)
        }


# Convenience functions for direct backend access
def predict_with_boltz(sequence: str, num_recycles: int = 3, allow_fallback: bool = True) -> Dict:
    """Predict structure using Boltz directly."""
    integration = ABCFoldIntegration(engine="boltz", allow_fallback=allow_fallback)
    return integration.predict_structure(sequence, num_recycles)


def predict_with_chai(sequence: str, num_recycles: int = 1, allow_fallback: bool = True) -> Dict:
    """Predict structure using Chai-1 directly."""
    integration = ABCFoldIntegration(engine="chai1", allow_fallback=allow_fallback)
    return integration.predict_structure(sequence, num_recycles)
