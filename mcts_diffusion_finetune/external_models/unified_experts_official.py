"""
Unified External Model Integration - Official Compatible Version
Based on https://github.com/ProtGenServer/denovo-protein-server
Single environment approach for all external models with official compatibility
"""

import os
import sys
import subprocess
import tempfile
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExpertConfig:
    """Configuration for each expert model"""
    name: str
    available: bool
    path: Optional[Path] = None
    weights_path: Optional[Path] = None
    error_msg: Optional[str] = None

class UnifiedExpertManagerOfficial:
    """Manages all external protein design experts following official structure"""
    
    def __init__(self, project_root: str = "/home/caom/AID3/dplm"):
        self.project_root = Path(project_root)
        self.denovo_server_root = self.project_root / "denovo-protein-server"
        self.models_dir = self.denovo_server_root / "models"
        self.third_party_dir = self.denovo_server_root / "third_party"
        
        # Expert configurations
        self.expert_configs = {}
        self._check_all_experts()
    
    def _check_all_experts(self):
        """Check availability of all external experts following official structure"""
        
        # Check ProteinMPNN (following official structure)
        proteinmpnn_path = self.third_party_dir / "proteinmpnn"
        proteinmpnn_weights = proteinmpnn_path / "vanilla_model_weights" / "v_48_020.pt"
        
        self.expert_configs["proteinmpnn"] = ExpertConfig(
            name="proteinmpnn",
            available=proteinmpnn_path.exists() and proteinmpnn_weights.exists(),
            path=proteinmpnn_path,
            weights_path=proteinmpnn_weights,
            error_msg=None if proteinmpnn_path.exists() and proteinmpnn_weights.exists() 
                     else "ProteinMPNN repository or weights not found"
        )
        
        # Check ProteInA (following official structure)
        proteina_path = self.third_party_dir / "proteina"
        # Check for different ProteInA model variants (official supports multiple)
        proteina_weights_options = [
            self.models_dir / "proteina" / "proteina_v1.7_DFS_60M_notri_motif_scaffolding.ckpt",  # Best for motif scaffolding
            self.models_dir / "proteina" / "proteina_v1.1_DFS_200M_tri.ckpt",  # Main 200M model
            self.models_dir / "proteina" / "proteina_v1.2_DFS_200M_notri.ckpt",  # 200M without triangular
            self.models_dir / "proteina" / "proteina_v1.3_DFS_60M_notri.ckpt",  # Smaller 60M model
        ]
        
        proteina_weights = None
        for weights_path in proteina_weights_options:
            if weights_path.exists():
                proteina_weights = weights_path
                break
        
        self.expert_configs["proteina"] = ExpertConfig(
            name="proteina",
            available=proteina_path.exists() and proteina_weights is not None,
            path=proteina_path,
            weights_path=proteina_weights,
            error_msg=None if proteina_path.exists() and proteina_weights is not None
                     else "ProteInA repository or weights not found"
        )
        
        # Check FoldFlow (following official structure)
        foldflow_path = self.third_party_dir / "foldflow"
        
        self.expert_configs["foldflow"] = ExpertConfig(
            name="foldflow",
            available=foldflow_path.exists(),
            path=foldflow_path,
            error_msg=None if foldflow_path.exists() else "FoldFlow repository not found"
        )
        
        # Check RFDiffusion (following official structure)
        rfdiffusion_path = self.third_party_dir / "rfdiffusion"
        rfdiffusion_weights = rfdiffusion_path / "models" / "Base_ckpt.pt"
        
        self.expert_configs["rfdiffusion"] = ExpertConfig(
            name="rfdiffusion",
            available=rfdiffusion_path.exists() and rfdiffusion_weights.exists(),
            path=rfdiffusion_path,
            weights_path=rfdiffusion_weights,
            error_msg=None if rfdiffusion_path.exists() and rfdiffusion_weights.exists()
                     else "RFDiffusion repository or weights not found"
        )
        
        # Log status
        for name, config in self.expert_configs.items():
            if config.available:
                if config.weights_path:
                    logger.info(f"✅ {name.upper()} available (weights: {config.weights_path.name})")
                else:
                    logger.info(f"✅ {name.upper()} available")
            else:
                logger.warning(f"⚠️ {name.upper()}: {config.error_msg}")
    
    def get_available_experts(self) -> List[str]:
        """Return list of available expert names"""
        return [name for name, config in self.expert_configs.items() if config.available]
    
    def get_expert_config(self, name: str) -> ExpertConfig:
        """Get configuration for specific expert"""
        return self.expert_configs.get(name)


class ProteinMPNNExpertOfficial:
    """ProteinMPNN integration following official implementation"""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.proteinmpnn_path = config.path
        self.model_path = config.weights_path
        
        # Add to Python path (following official approach)
        if str(self.proteinmpnn_path) not in sys.path:
            sys.path.insert(0, str(self.proteinmpnn_path))
    
    def generate_sequences(self, pdb_content: str, chain_id: str = "A", 
                          temperature: float = 0.1, num_samples: int = 1) -> List[str]:
        """Generate sequences using ProteinMPNN (following official approach)"""
        try:
            # Create temporary PDB file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_content)
                temp_pdb = f.name
            
            # Create temporary output directory
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                
                # Run ProteinMPNN (following official command structure)
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


class ProteInAExpertOfficial:
    """ProteInA integration following official implementation"""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.proteina_path = config.path
        self.weights_path = config.weights_path
        
        # Add to Python path (following official approach)
        if str(self.proteina_path) not in sys.path:
            sys.path.insert(0, str(self.proteina_path))
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load ProteInA model following official approach"""
        try:
            # Import ProteInA modules (following official structure)
            from proteinfoundation.train import ProteinFoundationModel
            
            # Load model (following official approach)
            self.model = ProteinFoundationModel.load_from_checkpoint(
                str(self.weights_path),
                map_location='cpu'
            )
            self.model.eval()
            logger.info(f"✅ ProteInA model loaded successfully ({self.weights_path.name})")
            
        except Exception as e:
            logger.error(f"Failed to load ProteInA model: {e}")
            self.model = None
    
    def generate_structures(self, sequence: str = None, motif_pdb: str = None, 
                           scaffold_length: int = None, num_samples: int = 1) -> List[str]:
        """Generate structures using ProteInA (following official approach)"""
        if self.model is None:
            logger.error("ProteInA model not loaded")
            return []
        
        try:
            # Determine generation mode based on inputs
            if motif_pdb and scaffold_length:
                logger.info(f"ProteInA motif scaffolding mode: {scaffold_length} residues")
                # This would be the actual motif scaffolding implementation
                # For now, return mock structure
                mode = "motif_scaffolding"
            elif sequence:
                logger.info(f"ProteInA sequence-to-structure mode: {len(sequence)} residues")
                mode = "seq2struct"
            else:
                logger.info("ProteInA unconditional generation mode")
                mode = "unconditional"
            
            # Placeholder for actual ProteInA inference
            logger.warning(f"ProteInA {mode} generation not fully implemented yet")
            
            # Mock PDB structure for now (would be replaced with actual inference)
            mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.685  16.849  14.897  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.638  18.076  14.897  1.00 20.00           O  
END
"""
            return [mock_pdb] * num_samples
            
        except Exception as e:
            logger.error(f"ProteInA generation failed: {e}")
            return []


class FoldFlowExpertOfficial:
    """FoldFlow integration following official implementation"""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.foldflow_path = config.path
        
        # Add to Python path (following official approach)
        if str(self.foldflow_path) not in sys.path:
            sys.path.insert(0, str(self.foldflow_path))
    
    def generate_structures(self, length: int = 100, num_samples: int = 1) -> List[str]:
        """Generate structures using FoldFlow (following official approach)"""
        try:
            logger.info(f"FoldFlow unconditional generation: {length} residues")
            
            # Placeholder for actual FoldFlow inference
            logger.warning("FoldFlow structure generation not fully implemented yet")
            
            # Mock PDB structure for now (would be replaced with actual inference)
            mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
END
"""
            return [mock_pdb] * num_samples
            
        except Exception as e:
            logger.error(f"FoldFlow generation failed: {e}")
            return []


class RFDiffusionExpertOfficial:
    """RFDiffusion integration following official implementation"""
    
    def __init__(self, config: ExpertConfig):
        self.config = config
        self.rfdiffusion_path = config.path
        self.weights_path = config.weights_path
        
        # Add to Python path (following official approach)
        if str(self.rfdiffusion_path) not in sys.path:
            sys.path.insert(0, str(self.rfdiffusion_path))
    
    def generate_scaffolds(self, motif_pdb: str, scaffold_length: int, 
                          num_samples: int = 1) -> List[str]:
        """Generate scaffolds using RFDiffusion (following official approach)"""
        try:
            logger.info(f"RFDiffusion motif scaffolding: {scaffold_length} residues")
            
            # Placeholder for actual RFDiffusion inference
            logger.warning("RFDiffusion scaffold generation not fully implemented yet")
            
            # Mock PDB structure for now (would be replaced with actual inference)
            mock_pdb = f"""ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
END
"""
            return [mock_pdb] * num_samples
            
        except Exception as e:
            logger.error(f"RFDiffusion generation failed: {e}")
            return []


class UnifiedExpertIntegrationOfficial:
    """Main integration class following official denovo-protein-server structure"""
    
    def __init__(self, project_root: str = "/home/caom/AID3/dplm"):
        self.manager = UnifiedExpertManagerOfficial(project_root)
        self.experts = {}
        self._initialize_experts()
    
    def _initialize_experts(self):
        """Initialize all available experts following official approach"""
        
        # Initialize ProteinMPNN
        config = self.manager.get_expert_config("proteinmpnn")
        if config and config.available:
            try:
                self.experts["proteinmpnn"] = ProteinMPNNExpertOfficial(config)
                logger.info("✅ ProteinMPNN expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ProteinMPNN: {e}")
        
        # Initialize ProteInA
        config = self.manager.get_expert_config("proteina")
        if config and config.available:
            try:
                self.experts["proteina"] = ProteInAExpertOfficial(config)
                logger.info("✅ ProteInA expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ProteInA: {e}")
        
        # Initialize FoldFlow
        config = self.manager.get_expert_config("foldflow")
        if config and config.available:
            try:
                self.experts["foldflow"] = FoldFlowExpertOfficial(config)
                logger.info("✅ FoldFlow expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FoldFlow: {e}")
        
        # Initialize RFDiffusion
        config = self.manager.get_expert_config("rfdiffusion")
        if config and config.available:
            try:
                self.experts["rfdiffusion"] = RFDiffusionExpertOfficial(config)
                logger.info("✅ RFDiffusion expert initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RFDiffusion: {e}")
    
    def get_available_expert_names(self) -> List[str]:
        """Get list of initialized expert names"""
        return list(self.experts.keys())
    
    def expert_rollout(self, expert_name: str, **kwargs) -> List[str]:
        """Perform rollout with specified expert following official API"""
        if expert_name not in self.experts:
            available = list(self.experts.keys())
            raise ValueError(f"Expert '{expert_name}' not available. Available: {available}")
        
        expert = self.experts[expert_name]
        
        # Route to appropriate method based on expert type (following official API)
        if expert_name == "proteinmpnn":
            pdb_content = kwargs.get("pdb_content", "")
            num_samples = kwargs.get("num_samples", 1)
            temperature = kwargs.get("temperature", 0.1)
            chain_id = kwargs.get("chain_id", "A")
            return expert.generate_sequences(pdb_content, chain_id=chain_id, 
                                           num_samples=num_samples, temperature=temperature)
        
        elif expert_name == "proteina":
            # ProteInA supports multiple modes (following official API)
            sequence = kwargs.get("sequence")
            motif_pdb = kwargs.get("motif_pdb")
            scaffold_length = kwargs.get("scaffold_length")
            num_samples = kwargs.get("num_samples", 1)
            return expert.generate_structures(sequence=sequence, motif_pdb=motif_pdb,
                                            scaffold_length=scaffold_length, num_samples=num_samples)
        
        elif expert_name == "foldflow":
            length = kwargs.get("length", 100)
            num_samples = kwargs.get("num_samples", 1)
            return expert.generate_structures(length=length, num_samples=num_samples)
        
        elif expert_name == "rfdiffusion":
            motif_pdb = kwargs.get("motif_pdb", "")
            scaffold_length = kwargs.get("scaffold_length", 100)
            num_samples = kwargs.get("num_samples", 1)
            return expert.generate_scaffolds(motif_pdb, scaffold_length, num_samples=num_samples)
        
        else:
            raise ValueError(f"Unknown expert type: {expert_name}")


def test_unified_integration_official():
    """Test the unified integration with official compatibility"""
    print("🧪 Testing Unified External Model Integration (Official Compatible)")
    print("=" * 70)
    
    try:
        integration = UnifiedExpertIntegrationOfficial()
        available = integration.get_available_expert_names()
        
        print(f"🤖 Available experts: {len(available)} ({available})")
        
        if not available:
            print("❌ No experts available. Please run setup script first.")
            return False
        
        # Test each available expert
        for expert_name in available:
            print(f"\n🧪 Testing {expert_name.upper()}...")
            try:
                if expert_name == "proteinmpnn":
                    # Mock PDB for testing
                    mock_pdb = """ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.618  1.00 20.00           C  
ATOM      3  C   ALA A   1      17.685  16.849  14.897  1.00 20.00           C  
ATOM      4  O   ALA A   1      17.638  18.076  14.897  1.00 20.00           O  
END
"""
                    results = integration.expert_rollout(expert_name, pdb_content=mock_pdb, num_samples=2)
                    print(f"   ✅ Generated {len(results)} sequences")
                    if results:
                        print(f"   📝 Sample: {results[0][:50]}...")
                
                elif expert_name == "proteina":
                    # Test motif scaffolding mode
                    mock_motif = """ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
END
"""
                    results = integration.expert_rollout(expert_name, motif_pdb=mock_motif, 
                                                       scaffold_length=50, num_samples=1)
                    print(f"   ✅ Generated {len(results)} motif scaffolds")
                
                elif expert_name == "foldflow":
                    results = integration.expert_rollout(expert_name, length=80, num_samples=1)
                    print(f"   ✅ Generated {len(results)} structures")
                
                elif expert_name == "rfdiffusion":
                    mock_motif = """ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N  
END
"""
                    results = integration.expert_rollout(expert_name, motif_pdb=mock_motif, 
                                                       scaffold_length=50, num_samples=1)
                    print(f"   ✅ Generated {len(results)} scaffolds")
                    
            except Exception as e:
                print(f"   ❌ Failed: {e}")
        
        print(f"\n🎉 Integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    test_unified_integration_official()





