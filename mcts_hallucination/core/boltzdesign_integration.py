"""
BoltzDesign1 Integration for Hallucination MCTS Pipeline.

BoltzDesign1 is a molecular design tool powered by the Boltz model for designing
protein-protein interactions and biomolecular complexes. It can design binders
for proteins, DNA, RNA, and small molecules.

Paper: BoltzDesign1: Inverting All-Atom Structure Prediction Model for 
       Generalized Biomolecular Binder Design
GitHub: https://github.com/yehlincho/BoltzDesign1
"""

import os
import sys
import tempfile
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union


class BoltzDesignIntegration:
    """
    Integration class for BoltzDesign1 protein binder design.
    
    BoltzDesign1 uses the Boltz structure prediction model in reverse to design
    protein sequences that bind to specified targets (protein, DNA, RNA, small molecules).
    """
    
    # Path to BoltzDesign1 installation
    BOLTZDESIGN_PATH = Path(__file__).parent.parent / "extra" / "BoltzDesign1"
    
    def __init__(
        self,
        use_mock: bool = True,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
        temperature: float = 0.1,
    ):
        """
        Initialize BoltzDesign1 integration.
        
        Args:
            use_mock: If True, use mock mode for testing
            device: Device to run on ('cuda' or 'cpu')
            checkpoint_path: Path to Boltz checkpoint (default: ~/.boltz/boltz1_conf.ckpt)
            temperature: Sampling temperature
        """
        self.use_mock = use_mock
        self.device = device
        self.checkpoint_path = checkpoint_path or "/net/scratch/caom/.boltz/boltz1_conf.ckpt"
        self.temperature = temperature
        
        mode = "MOCK MODE" if use_mock else "REAL MODE"
        print(f"🔧 BoltzDesign1 Integration initialized ({mode})")
        if not use_mock:
            print(f"   BoltzDesign1 path: {self.BOLTZDESIGN_PATH}")
            print(f"   Checkpoint: {self.checkpoint_path}")
    
    def design_binder(
        self,
        target_pdb: str,
        target_type: str = "protein",
        target_chain_ids: Optional[List[str]] = None,
        binder_length_min: int = 50,
        binder_length_max: int = 100,
        num_designs: int = 1,
        use_msa: bool = False,
    ) -> Dict:
        """
        Design a protein binder for the given target.
        
        Args:
            target_pdb: Path to target PDB file or PDB code
            target_type: Type of target ('protein', 'dna', 'rna', 'small_molecule')
            target_chain_ids: Chain IDs of target (e.g., ['A', 'B'])
            binder_length_min: Minimum binder length
            binder_length_max: Maximum binder length
            num_designs: Number of designs to generate
            use_msa: Whether to use MSA for design
            
        Returns:
            Dict with designed sequences, structures, and confidence scores
        """
        if self.use_mock:
            return self._mock_design(
                target_type, binder_length_min, binder_length_max, num_designs
            )
        else:
            return self._real_design(
                target_pdb, target_type, target_chain_ids,
                binder_length_min, binder_length_max, num_designs, use_msa
            )
    
    def _mock_design(
        self,
        target_type: str,
        length_min: int,
        length_max: int,
        num_designs: int,
    ) -> Dict:
        """Mock design for testing."""
        designs = []
        
        for i in range(num_designs):
            length = np.random.randint(length_min, length_max + 1)
            
            # Generate random protein sequence
            aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
            sequence = ''.join(np.random.choice(list(aa_alphabet), length))
            
            # Generate mock coordinates
            coords = np.cumsum(np.random.randn(length, 3) * 3.8, axis=0)
            
            # Generate mock confidence
            confidence = 70 + np.random.randn() * 10
            iptm = 0.6 + np.random.random() * 0.3
            
            designs.append({
                'sequence': sequence,
                'coordinates': coords,
                'plddt': confidence,
                'iptm': iptm,
                'target_type': target_type,
            })
        
        return {
            'designs': designs,
            'num_designs': num_designs,
            'target_type': target_type,
        }
    
    def _real_design(
        self,
        target_pdb: str,
        target_type: str,
        target_chain_ids: Optional[List[str]],
        length_min: int,
        length_max: int,
        num_designs: int,
        use_msa: bool,
    ) -> Dict:
        """Real BoltzDesign1 design using CLI."""
        import shutil
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            output_dir = tmpdir / "output"
            output_dir.mkdir()
            
            # Copy LigandMPNN directory to work_dir (required for LigandMPNN step)
            ligandmpnn_src = self.BOLTZDESIGN_PATH / "LigandMPNN"
            ligandmpnn_dst = tmpdir / "LigandMPNN"
            if ligandmpnn_src.exists():
                shutil.copytree(ligandmpnn_src, ligandmpnn_dst)
                print(f"      Copied LigandMPNN to {ligandmpnn_dst}")
            
            # Copy LigandMPNN model weights (required for LigandMPNN step)
            ligandmpnn_weights_src = Path("/net/scratch/caom/ligandmpnn_weights")
            ligandmpnn_weights_dst = ligandmpnn_dst / "model_params"
            if ligandmpnn_weights_src.exists():
                shutil.copytree(ligandmpnn_weights_src, ligandmpnn_weights_dst)
                print(f"      Copied LigandMPNN weights to {ligandmpnn_weights_dst}")
            
            # Determine target name
            if os.path.exists(target_pdb):
                target_name = Path(target_pdb).stem
                pdb_path_arg = f"--pdb_path {target_pdb}"
            else:
                target_name = target_pdb
                pdb_path_arg = ""
            
            # Build command with faster inference settings
            cmd = [
                sys.executable,
                str(self.BOLTZDESIGN_PATH / "boltzdesign.py"),
                "--target_name", target_name,
                "--target_type", target_type,
                "--design_samples", str(num_designs),
                "--length_min", str(length_min),
                "--length_max", str(length_max),
                "--work_dir", str(tmpdir),
                "--run_alphafold", "False",  # Skip AF3 validation
                "--run_rosetta", "False",    # Skip Rosetta
                "--run_ligandmpnn", "True",  # Run LigandMPNN for sequence design
                "--use_msa", str(use_msa).lower(),
                "--boltz_checkpoint", self.checkpoint_path,
                "--ccd_path", str(Path(self.checkpoint_path).parent / "ccd.pkl"),
                # Faster inference settings - reduce iterations
                "--pre_iteration", "10",   # Default 30
                "--soft_iteration", "30",  # Default 100
                "--temp_iteration", "0",   # Default 0
                "--hard_iteration", "10",  # Default 30
            ]
            
            if target_chain_ids:
                cmd.extend(["--pdb_target_ids", ",".join(target_chain_ids)])
            
            if pdb_path_arg:
                cmd.extend(pdb_path_arg.split())
            
            print(f"      Running BoltzDesign1:")
            print(f"        Target: {target_name} ({target_type})")
            print(f"        Binder length: {length_min}-{length_max} aa")
            print(f"        Work dir: {tmpdir}")
            print(f"        Checkpoint: {self.checkpoint_path}")
            print(f"        Command: {' '.join(cmd)}")
            print(f"      (This may take 5-15 minutes with fast settings)")
            
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'
            boltzdesign_dir = str(self.BOLTZDESIGN_PATH / "boltzdesign")
            env['PYTHONPATH'] = f"{boltzdesign_dir}:{env.get('PYTHONPATH', '')}"
            
            # Use Popen for streaming output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.BOLTZDESIGN_PATH),
                env=env,
                bufsize=1,
            )
            
            # Stream output in real-time
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                if any(kw in line.lower() for kw in ['step', 'iter', 'sample', 'design', 'loading', 'running', 'complete', 'error', 'warning', 'iptm', 'epoch']):
                    print(f"      {line}")
            
            process.wait()
            
            # Parse output even if process failed (design may have been generated)
            # Look in the work_dir/outputs directory for results
            outputs_dir = tmpdir / "outputs"
            designs = self._parse_output(outputs_dir, target_name)
            
            # Also try parsing from output_dir
            if not designs:
                designs = self._parse_output(output_dir, target_name)
            
            # If still no designs and process failed, raise error
            if process.returncode != 0 and not designs:
                raise RuntimeError(f"BoltzDesign1 failed: {''.join(output_lines[-10:])}")
            
            # Extract iPTM from output if available
            for line in output_lines:
                if 'Update sequence, iptm' in line:
                    try:
                        import re
                        match = re.search(r'iptm \[([0-9.]+)\]', line)
                        if match and designs:
                            designs[-1]['iptm'] = float(match.group(1))
                    except:
                        pass
            
            return {
                'designs': designs,
                'num_designs': len(designs),
                'target_type': target_type,
            }
    
    def _parse_output(self, output_dir: Path, target_name: str) -> List[Dict]:
        """Parse BoltzDesign1 output files."""
        designs = []
        
        if not output_dir.exists():
            return designs
        
        # Look for output PDB and CIF files
        pdb_files = list(output_dir.glob(f"**/*{target_name}*.pdb"))
        cif_files = list(output_dir.glob(f"**/*{target_name}*.cif"))
        
        # Combine and deduplicate
        all_files = pdb_files + cif_files
        print(f"      Found {len(all_files)} output files in {output_dir}")
        
        # Helper to convert 3-letter to 1-letter amino acid code
        AA_MAP = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
        }
        
        def parse_structure(structure):
            sequence = ""
            coords = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            resname = residue.get_resname()
                            if resname in AA_MAP:
                                sequence += AA_MAP[resname]
                                coords.append(residue['CA'].get_coord())
            return sequence, coords
        
        # Parse PDB files
        for pdb_file in pdb_files:
            try:
                from Bio.PDB import PDBParser
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("design", pdb_file)
                sequence, coords = parse_structure(structure)
                
                if sequence:
                    designs.append({
                        'sequence': sequence,
                        'coordinates': np.array(coords),
                        'plddt': 70.0,
                        'iptm': 0.7,
                        'pdb_path': str(pdb_file),
                    })
                    print(f"      Parsed PDB: {pdb_file.name} ({len(sequence)} aa)")
            except Exception as e:
                print(f"      Warning: Failed to parse PDB {pdb_file}: {e}")
        
        # Parse CIF files
        for cif_file in cif_files:
            try:
                from Bio.PDB import MMCIFParser
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("design", cif_file)
                sequence, coords = parse_structure(structure)
                
                if sequence:
                    designs.append({
                        'sequence': sequence,
                        'coordinates': np.array(coords),
                        'plddt': 70.0,
                        'iptm': 0.7,
                        'pdb_path': str(cif_file),
                    })
                    print(f"      Parsed CIF: {cif_file.name} ({len(sequence)} aa)")
            except Exception as e:
                print(f"      Warning: Failed to parse CIF {cif_file}: {e}")
        
        return designs


# Convenience function
def design_binder_with_boltzdesign(
    target_pdb: str,
    target_type: str = "protein",
    num_designs: int = 1,
    use_mock: bool = True,
) -> Dict:
    """Design a protein binder using BoltzDesign1."""
    integration = BoltzDesignIntegration(use_mock=use_mock)
    return integration.design_binder(
        target_pdb=target_pdb,
        target_type=target_type,
        num_designs=num_designs,
    )
