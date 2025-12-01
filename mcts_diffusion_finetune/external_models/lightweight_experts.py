"""
Lightweight External Model Integration for MCTS
Uses existing environment and minimal dependencies
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LightweightExpertManager:
    """Manages external protein design experts with minimal resource usage"""
    
    def __init__(self, project_root: str = "/home/caom/AID3/dplm"):
        self.project_root = Path(project_root)
        self.denovo_server_root = self.project_root / "denovo-protein-server"
        self.models_dir = self.denovo_server_root / "models"
        self.third_party_dir = self.denovo_server_root / "third_party"
        
        # Available experts
        self.available_experts = []
        self._check_available_experts()
    
    def _check_available_experts(self):
        """Check which experts are available based on installed components"""
        experts = []
        
        # Check ProteinMPNN
        proteinmpnn_path = self.third_party_dir / "proteinmpnn"
        if proteinmpnn_path.exists():
            experts.append("proteinmpnn")
            logger.info("✅ ProteinMPNN available")
        else:
            logger.warning("⚠️ ProteinMPNN not found")
        
        # Check ProteInA weights
        proteina_weights = self.models_dir / "proteina" / "proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt"
        if proteina_weights.exists():
            experts.append("proteina")
            logger.info("✅ ProteInA weights available")
        else:
            logger.warning("⚠️ ProteInA weights not found")
        
        # Check FoldFlow
        foldflow_path = self.third_party_dir / "foldflow"
        if foldflow_path.exists():
            experts.append("foldflow")
            logger.info("✅ FoldFlow repository available")
        else:
            logger.warning("⚠️ FoldFlow not found")
        
        # Check RFDiffusion
        rfdiffusion_path = self.third_party_dir / "rfdiffusion"
        if rfdiffusion_path.exists():
            experts.append("rfdiffusion")
            logger.info("✅ RFDiffusion repository available")
        else:
            logger.warning("⚠️ RFDiffusion not found")
        
        self.available_experts = experts
        logger.info(f"🤖 Available experts: {len(experts)} ({experts})")
    
    def get_available_experts(self) -> List[str]:
        """Return list of available expert names"""
        return self.available_experts.copy()


class ProteinMPNNExpert:
    """Lightweight ProteinMPNN integration"""
    
    def __init__(self, proteinmpnn_path: Path):
        self.proteinmpnn_path = proteinmpnn_path
        self.model_path = proteinmpnn_path / "vanilla_model_weights" / "v_48_020.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"ProteinMPNN weights not found: {self.model_path}")
    
    def generate_sequence(self, pdb_content: str, chain_id: str = "A", 
                         temperature: float = 0.1, num_samples: int = 1) -> List[str]:
        """Generate sequences using ProteinMPNN"""
        try:
            # Create temporary PDB file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_content)
                temp_pdb = f.name
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Run ProteinMPNN
                cmd = [
                    sys.executable,
                    str(self.proteinmpnn_path / "protein_mpnn_run.py"),
                    "--pdb_path", temp_pdb,
                    "--pdb_path_chains", chain_id,
                    "--out_folder", str(output_dir),
                    "--num_seq_per_target", str(num_samples),
                    "--sampling_temp", str(temperature),
                    "--seed", "42",
                    "--batch_size", "1"
                ]
                
                # Run command
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                      cwd=self.proteinmpnn_path)
                
                if result.returncode != 0:
                    logger.error(f"ProteinMPNN failed: {result.stderr}")
                    return []
                
                # Parse output sequences
                sequences = []
                for fasta_file in output_dir.glob("*.fa"):
                    with open(fasta_file, 'r') as f:
                        lines = f.readlines()
                        for i in range(1, len(lines), 2):  # Skip header lines
                            if i < len(lines):
                                sequences.append(lines[i].strip())
                
                # Cleanup
                os.unlink(temp_pdb)
                
                return sequences[:num_samples]
                
        except Exception as e:
            logger.error(f"ProteinMPNN generation failed: {e}")
            return []


class ProteInAExpert:
    """Lightweight ProteInA integration"""
    
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"ProteInA weights not found: {self.weights_path}")
    
    def generate_structure(self, sequence: str, num_samples: int = 1) -> List[str]:
        """Generate structures using ProteInA (placeholder - needs proper integration)"""
        logger.warning("ProteInA integration not yet implemented - returning mock structure")
        
        # Mock PDB structure for now
        mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.685  16.849  14.897  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.638  18.076  14.897  1.00 20.00           O  
END
"""
        return [mock_pdb] * num_samples


class FoldFlowExpert:
    """Lightweight FoldFlow integration"""
    
    def __init__(self, foldflow_path: Path):
        self.foldflow_path = foldflow_path
    
    def generate_structure(self, sequence: str, num_samples: int = 1) -> List[str]:
        """Generate structures using FoldFlow (placeholder - needs proper integration)"""
        logger.warning("FoldFlow integration not yet implemented - returning mock structure")
        
        # Mock PDB structure for now
        mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
END
"""
        return [mock_pdb] * num_samples


class RFDiffusionExpert:
    """Lightweight RFDiffusion integration"""
    
    def __init__(self, rfdiffusion_path: Path):
        self.rfdiffusion_path = rfdiffusion_path
    
    def generate_structure(self, motif_pdb: str, scaffold_length: int, 
                          num_samples: int = 1) -> List[str]:
        """Generate scaffolds using RFDiffusion (placeholder - needs proper integration)"""
        logger.warning("RFDiffusion integration not yet implemented - returning mock structure")
        
        # Mock PDB structure for now
        mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
END
"""
        return [mock_pdb] * num_samples


class LightweightExpertIntegration:
    """Main integration class for lightweight external experts"""
    
    def __init__(self, project_root: str = "/home/caom/AID3/dplm"):
        self.manager = LightweightExpertManager(project_root)
        self.experts = {}
        self._initialize_experts()
    
    def _initialize_experts(self):
        """Initialize available experts"""
        available = self.manager.get_available_experts()
        
        # Initialize ProteinMPNN
        if "proteinmpnn" in available:
            try:
                proteinmpnn_path = self.manager.third_party_dir / "proteinmpnn"
                self.experts["proteinmpnn"] = ProteinMPNNExpert(proteinmpnn_path)
                logger.info("✅ ProteinMPNN expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ProteinMPNN: {e}")
        
        # Initialize ProteInA
        if "proteina" in available:
            try:
                weights_path = self.manager.models_dir / "proteina" / "proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt"
                self.experts["proteina"] = ProteInAExpert(weights_path)
                logger.info("✅ ProteInA expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ProteInA: {e}")
        
        # Initialize FoldFlow
        if "foldflow" in available:
            try:
                foldflow_path = self.manager.third_party_dir / "foldflow"
                self.experts["foldflow"] = FoldFlowExpert(foldflow_path)
                logger.info("✅ FoldFlow expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FoldFlow: {e}")
        
        # Initialize RFDiffusion
        if "rfdiffusion" in available:
            try:
                rfdiffusion_path = self.manager.third_party_dir / "rfdiffusion"
                self.experts["rfdiffusion"] = RFDiffusionExpert(rfdiffusion_path)
                logger.info("✅ RFDiffusion expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RFDiffusion: {e}")
    
    def get_available_expert_names(self) -> List[str]:
        """Get list of initialized expert names"""
        return list(self.experts.keys())
    
    def expert_rollout(self, expert_name: str, **kwargs) -> List[str]:
        """Perform rollout with specified expert"""
        if expert_name not in self.experts:
            raise ValueError(f"Expert '{expert_name}' not available. Available: {list(self.experts.keys())}")
        
        expert = self.experts[expert_name]
        
        # Route to appropriate method based on expert type
        if expert_name == "proteinmpnn":
            pdb_content = kwargs.get("pdb_content", "")
            return expert.generate_sequence(pdb_content)
        
        elif expert_name in ["proteina", "foldflow"]:
            sequence = kwargs.get("sequence", "")
            return expert.generate_structure(sequence)
        
        elif expert_name == "rfdiffusion":
            motif_pdb = kwargs.get("motif_pdb", "")
            scaffold_length = kwargs.get("scaffold_length", 100)
            return expert.generate_structure(motif_pdb, scaffold_length)
        
        else:
            raise ValueError(f"Unknown expert type: {expert_name}")


# Test function
def test_lightweight_integration():
    """Test the lightweight integration"""
    try:
        integration = LightweightExpertIntegration()
        available = integration.get_available_expert_names()
        
        print(f"🤖 Initialized experts: {available}")
        
        # Test each available expert
        for expert_name in available:
            print(f"\n🧪 Testing {expert_name}...")
            try:
                if expert_name == "proteinmpnn":
                    # Mock PDB for testing
                    mock_pdb = "ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N\nEND\n"
                    results = integration.expert_rollout(expert_name, pdb_content=mock_pdb)
                    print(f"   ✅ Generated {len(results)} sequences")
                
                else:
                    results = integration.expert_rollout(expert_name, sequence="ACDEFGHIKLMNPQRSTVWY")
                    print(f"   ✅ Generated {len(results)} structures")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_lightweight_integration()

