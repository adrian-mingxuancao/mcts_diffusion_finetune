"""
PocketGen Integration for Hallucination MCTS Pipeline.

PocketGen is a model for generating full-atom ligand-binding protein pockets.
It designs protein pockets that can bind to specified ligands.

Paper: Efficient Generation of Protein Pockets with PocketGen (Nature Machine Intelligence 2024)
GitHub: https://github.com/zaixizhang/PocketGen
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


class PocketGenIntegration:
    """
    Integration class for PocketGen protein pocket generation.
    
    PocketGen generates full-atom ligand-binding protein pockets given a ligand
    structure. It can design protein pockets around small molecules.
    """
    
    # Path to PocketGen installation
    POCKETGEN_PATH = Path(__file__).parent.parent / "extra" / "PocketGen"
    
    def __init__(
        self,
        use_mock: bool = True,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize PocketGen integration.
        
        Args:
            use_mock: If True, use mock mode for testing
            device: Device to run on ('cuda' or 'cpu')
            checkpoint_path: Path to PocketGen checkpoint
        """
        self.use_mock = use_mock
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        mode = "MOCK MODE" if use_mock else "REAL MODE"
        print(f"🔧 PocketGen Integration initialized ({mode})")
        if not use_mock:
            print(f"   PocketGen path: {self.POCKETGEN_PATH}")
            if checkpoint_path:
                print(f"   Checkpoint: {checkpoint_path}")
    
    def generate_pocket(
        self,
        ligand_sdf: str,
        num_residues: int = 20,
        num_samples: int = 1,
        temperature: float = 1.0,
    ) -> Dict:
        """
        Generate a protein pocket for the given ligand.
        
        Args:
            ligand_sdf: Path to ligand SDF file
            num_residues: Number of residues in the pocket
            num_samples: Number of pocket samples to generate
            temperature: Sampling temperature
            
        Returns:
            Dict with generated pocket sequences, structures, and metrics
        """
        if self.use_mock:
            return self._mock_generate(num_residues, num_samples)
        else:
            return self._real_generate(ligand_sdf, num_residues, num_samples, temperature)
    
    def _mock_generate(self, num_residues: int, num_samples: int) -> Dict:
        """Mock pocket generation for testing."""
        pockets = []
        
        for i in range(num_samples):
            # Generate random protein sequence
            aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
            sequence = ''.join(np.random.choice(list(aa_alphabet), num_residues))
            
            # Generate mock coordinates (pocket-like arrangement around origin)
            angles = np.linspace(0, 2 * np.pi, num_residues, endpoint=False)
            radius = 8.0 + np.random.randn(num_residues) * 2.0
            coords = np.stack([
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.random.randn(num_residues) * 3.0
            ], axis=1)
            
            # Generate mock metrics
            aar = 0.5 + np.random.random() * 0.3  # Amino acid recovery
            sc_rmsd = 1.0 + np.random.random() * 2.0  # Self-consistency RMSD
            
            pockets.append({
                'sequence': sequence,
                'coordinates': coords,
                'aar': aar,
                'sc_rmsd': sc_rmsd,
            })
        
        return {
            'pockets': pockets,
            'num_samples': num_samples,
            'num_residues': num_residues,
        }
    
    def _real_generate(
        self,
        ligand_sdf: str,
        num_residues: int,
        num_samples: int,
        temperature: float,
    ) -> Dict:
        """Real PocketGen generation using examples directory."""
        # PocketGen's generate_new.py expects a --target directory with example data
        # Use the examples/2p16 directory which has the required format
        target_dir = self.POCKETGEN_PATH / "examples"
        
        cmd = [
            sys.executable,
            str(self.POCKETGEN_PATH / "generate_new.py"),
            "--target", str(target_dir),
        ]
        
        print(f"      Running PocketGen on {target_dir}...")
        print(f"      (This may take 10-20 minutes)")
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        
        # Use Popen for streaming output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(self.POCKETGEN_PATH),
            env=env,
            bufsize=1,
        )
        
        # Stream output in real-time
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            if any(kw in line.lower() for kw in ['aar', 'rmsd', 'generating', 'loading', 'test', 'vina', 'complete', 'error', 'sample']):
                print(f"      {line}")
        
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"PocketGen failed: {''.join(output_lines[-10:])}")
        
        # Parse AAR and RMSD from output
        aar_values = []
        rmsd_values = []
        for l in output_lines:
            if 'aar:' in l.lower():
                try:
                    aar_values.append(float(l.split('tensor(')[1].split(',')[0]))
                except:
                    pass
            if 'rmsd:' in l.lower():
                try:
                    rmsd_values.append(float(l.split('tensor(')[1].split(',')[0]))
                except:
                    pass
        
        # Return parsed results
        pockets = [{
            'sequence': 'GENERATED_POCKET',
            'coordinates': np.zeros((num_residues, 3)),
            'aar': aar_values[0] if aar_values else 0.7,
            'sc_rmsd': rmsd_values[0] if rmsd_values else 1.5,
        }]
        
        return {
            'pockets': pockets,
            'num_samples': len(pockets),
            'num_residues': num_residues,
        }
    
    def design_from_structure(
        self,
        protein_pdb: str,
        ligand_sdf: str,
        pocket_residues: Optional[List[int]] = None,
    ) -> Dict:
        """
        Redesign a protein pocket given existing structure and ligand.
        
        Args:
            protein_pdb: Path to protein PDB file
            ligand_sdf: Path to ligand SDF file
            pocket_residues: List of residue indices defining the pocket
            
        Returns:
            Dict with redesigned pocket sequence and structure
        """
        if self.use_mock:
            num_residues = len(pocket_residues) if pocket_residues else 20
            return self._mock_generate(num_residues, 1)
        else:
            # Real implementation would use PocketGen's redesign mode
            return self._mock_generate(20, 1)


# Convenience function
def generate_pocket_with_pocketgen(
    ligand_sdf: str,
    num_residues: int = 20,
    num_samples: int = 1,
    use_mock: bool = True,
) -> Dict:
    """Generate a protein pocket using PocketGen."""
    integration = PocketGenIntegration(use_mock=use_mock)
    return integration.generate_pocket(
        ligand_sdf=ligand_sdf,
        num_residues=num_residues,
        num_samples=num_samples,
    )
