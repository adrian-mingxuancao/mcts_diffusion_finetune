"""
ABCFold Integration for AlphaFold3 Structure Hallucination

Wrapper around ABCFold (https://github.com/rigdenlab/ABCFold) for structure prediction
from masked sequences in the hallucination pipeline.
"""

import numpy as np
from typing import Dict, Optional, List
import json
import tempfile
import subprocess
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
    ):
        """
        Initialize ABCFold integration.
        
        Args:
            model_params: Path to AF3 model parameters
            database_dir: Path to AF3 databases (not needed if use_mmseqs=True)
            use_mmseqs: Use MMseqs2 for faster MSA (recommended)
            use_mock: Use mock mode for testing (set False for real AF3)
            engine: Which ABCFold backend to call (af3, boltz, chai1)
        """
        self.model_params = model_params
        self.database_dir = database_dir
        self.use_mmseqs = use_mmseqs
        self.use_mock = use_mock
        self.engine = engine.lower()
        self.allow_fallback = allow_fallback
        
        if self.engine not in self.ENGINE_FLAGS:
            raise ValueError(f"Unsupported ABCFold engine '{engine}'. Choose from {list(self.ENGINE_FLAGS)}.")
        
        if not use_mock and self.engine == "af3" and not self.model_params:
            raise ValueError("Real AF3 mode requires --model_params pointing to the AlphaFold3 weights directory.")
        
        if use_mock:
            print(f"🔧 ABCFold Integration initialized (MOCK MODE)")
        else:
            print(f"🔧 ABCFold Integration initialized (REAL MODE - {self.engine.upper()})")
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
        Real ABCFold prediction using command-line interface.
        
        This runs ABCFold via subprocess and parses the output CIF file.
        """
        # Create temporary directory for ABCFold run
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
            
            # Only AF3 requires model/database paths and recycle controls
            if self.engine == "af3":
                if self.model_params:
                    cmd.extend(["--model_params", self.model_params])
                
                if self.database_dir and not self.use_mmseqs:
                    cmd.extend(["--database", self.database_dir])
                
                cmd.extend([
                    "--num_recycles", str(num_recycles),
                ])
            
            cmd.extend(["--override", "--no_visuals"])  # Always overwrite temp dir; skip HTML output
            
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
    
    def _create_af3_json(self, sequence: str, output_path: Path):
        """Create AF3 input JSON file."""
        af3_input = {
            "name": "hallucination",
            "sequences": [{
                "protein": {
                    "id": ["A"],
                    "sequence": sequence
                }
            }],
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
            coords: (N, 3) CA coordinates
            confidence: (N,) pLDDT scores
            pae_mean: Mean PAE
        """
        # This is a simplified parser - you may want to use a proper CIF parser
        # like gemmi or BioPython
        try:
            from Bio.PDB.MMCIFParser import MMCIFParser
            from Bio.PDB import PDBIO
            
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("hallucination", cif_path)
            
            # Extract CA coordinates
            coords = []
            confidence = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            ca = residue['CA']
                            coords.append(ca.get_coord())
                            # pLDDT is stored in B-factor column
                            confidence.append(ca.get_bfactor())
            
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
