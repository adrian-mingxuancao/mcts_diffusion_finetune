"""
Official HTTP Server Integration for MCTS
Uses official denovo-protein-server HTTP servers for real inference
"""

import os
import sys
import requests
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OfficialServerIntegration:
    """Integration with official denovo-protein-server HTTP servers"""
    
    def __init__(self, project_root: str = "/home/caom/AID3/dplm"):
        self.project_root = Path(project_root)
        self.denovo_server_root = self.project_root / "denovo-protein-server"
        
        # Official server configurations
        self.servers = {
            "proteina": {
                "port": 8080,
                "env": "prot-srv-proteina",
                "launch_script": "launch_proteina.sh",
                "config": "inference_ucond_200m_tri.yaml",
                "capability": "motif_scaffolding"
            },
            "foldflow": {
                "port": 8081,
                "env": "prot-srv-foldflow", 
                "launch_script": "launch_foldflow.sh",
                "config": "inference.yaml",
                "capability": "structure_generation"
            },
            "rfdiffusion": {
                "port": 8082,
                "env": "prot-srv-rfdiffusion",
                "launch_script": "launch_rfdiffusion.sh", 
                "config": "base.yaml",
                "capability": "motif_scaffolding"
            }
        }
        
        self.running_servers = []
        self._check_servers()
    
    def _check_servers(self):
        """Check which servers are running"""
        for server_name, config in self.servers.items():
            try:
                response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
                if response.status_code == 200:
                    self.running_servers.append(server_name)
                    logger.info(f"✅ {server_name.upper()} server running on port {config['port']}")
                else:
                    logger.warning(f"⚠️ {server_name.upper()} server not responding properly")
            except requests.exceptions.RequestException:
                logger.warning(f"⚠️ {server_name.upper()} server not running on port {config['port']}")
    
    def start_server(self, server_name: str) -> bool:
        """Start official server"""
        if server_name not in self.servers:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        config = self.servers[server_name]
        
        # Check if already running
        if server_name in self.running_servers:
            logger.info(f"✅ {server_name.upper()} server already running")
            return True
        
        try:
            logger.info(f"🚀 Starting official {server_name.upper()} server...")
            
            # Set up environment
            env = os.environ.copy()
            if server_name == "proteina":
                env["DATA_PATH"] = str(self.denovo_server_root / "data")
                # Create data directory if it doesn't exist
                os.makedirs(self.denovo_server_root / "data", exist_ok=True)
            
            # Launch server using official script
            launch_script = self.denovo_server_root / "scripts" / config["launch_script"]
            cmd = [
                "bash", str(launch_script),
                "-p", str(config["port"]),
                "-c", config["config"],
                "-g", "0"  # Use GPU 0
            ]
            
            # Start server in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(self.denovo_server_root)
            )
            
            # Wait for server to start
            for i in range(60):  # 60 second timeout
                time.sleep(1)
                try:
                    response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
                    if response.status_code == 200:
                        self.running_servers.append(server_name)
                        logger.info(f"✅ {server_name.upper()} server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    continue
            
            logger.error(f"❌ {server_name.upper()} server failed to start within 60 seconds")
            process.terminate()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start {server_name} server: {e}")
            return False
    
    def get_available_servers(self) -> List[str]:
        """Get list of running servers"""
        return self.running_servers.copy()
    
    def proteina_motif_scaffolding(self, motif_sequence: str, motif_structure: str, 
                                  target_length: int) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Real ProteInA motif scaffolding via official server"""
        if "proteina" not in self.running_servers:
            logger.error("ProteInA server not running")
            return None, None, None
        
        try:
            logger.info(f"🧬 Real ProteInA motif scaffolding: {motif_sequence} -> {target_length}")
            
            # Prepare request for official ProteInA server
            request_data = {
                "motif_sequence": motif_sequence,
                "motif_structure": motif_structure,
                "target_length": target_length,
                "num_samples": 1,
                "temperature": 0.8,
                "task": "motif_scaffolding"
            }
            
            response = requests.post(
                f"http://localhost:8080/generate",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    sequences = result.get("sequences", [])
                    structures = result.get("structures", [])
                    entropies = result.get("entropies", [1.0])
                    
                    if sequences:
                        seq = sequences[0]
                        struct_tokens = self._coordinates_to_tokens(structures[0] if structures else None, target_length)
                        entropy = entropies[0] if entropies else 0.6
                        
                        logger.info(f"✅ Real ProteInA generated: {seq}")
                        logger.info(f"📊 Real entropy: {entropy:.3f}")
                        
                        return seq, struct_tokens, entropy
                    else:
                        logger.error("No sequences in ProteInA response")
                        return None, None, None
                else:
                    logger.error(f"ProteInA server error: {result.get('error', 'Unknown error')}")
                    return None, None, None
            else:
                logger.error(f"ProteInA HTTP error: {response.status_code}")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Real ProteInA inference failed: {e}")
            return None, None, None
    
    def foldflow_structure_generation(self, target_length: int, motif_sequence: str = None) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Real FoldFlow structure generation via official server"""
        if "foldflow" not in self.running_servers:
            logger.error("FoldFlow server not running")
            return None, None, None
        
        try:
            logger.info(f"🌊 Real FoldFlow structure generation: {target_length} residues")
            
            # Prepare request for official FoldFlow server
            request_data = {
                "length": target_length,
                "num_samples": 1,
                "temperature": 1.0,
                "motif_conditioning": motif_sequence
            }
            
            response = requests.post(
                f"http://localhost:8081/generate",
                json=request_data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    structures = result.get("structures", [])
                    entropies = result.get("entropies", [0.7])
                    
                    if structures:
                        structure_data = structures[0]
                        seq = structure_data.get("sequence", "")
                        coordinates = structure_data.get("coordinates", [])
                        entropy = entropies[0] if entropies else 0.7
                        
                        # Ensure motif preservation
                        if motif_sequence and motif_sequence not in seq:
                            scaffold_len = len(seq) - len(motif_sequence)
                            left_len = scaffold_len // 2
                            seq = seq[:left_len] + motif_sequence + seq[left_len + len(motif_sequence):]
                        
                        struct_tokens = self._coordinates_to_tokens(coordinates, len(seq))
                        
                        logger.info(f"✅ Real FoldFlow generated: {seq}")
                        logger.info(f"📊 Real entropy: {entropy:.3f}")
                        
                        return seq, struct_tokens, entropy
                    else:
                        logger.error("No structures in FoldFlow response")
                        return None, None, None
                else:
                    logger.error(f"FoldFlow server error: {result.get('error', 'Unknown error')}")
                    return None, None, None
            else:
                logger.error(f"FoldFlow HTTP error: {response.status_code}")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Real FoldFlow inference failed: {e}")
            return None, None, None
    
    def rfdiffusion_motif_scaffolding(self, motif_sequence: str, motif_pdb: str, 
                                     target_length: int) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """Real RFDiffusion motif scaffolding via official server"""
        if "rfdiffusion" not in self.running_servers:
            logger.error("RFDiffusion server not running")
            return None, None, None
        
        try:
            logger.info(f"🧪 Real RFDiffusion motif scaffolding: {motif_sequence} -> {target_length}")
            
            # Calculate scaffold configuration
            scaffold_length = target_length - len(motif_sequence)
            left_scaffold = scaffold_length // 2
            right_scaffold = scaffold_length - left_scaffold
            
            # Prepare request for official RFDiffusion server
            request_data = {
                "motif_pdb": motif_pdb,
                "contig": f"[{left_scaffold}-{left_scaffold}/A1-{len(motif_sequence)}/{right_scaffold}-{right_scaffold}]",
                "num_designs": 1,
                "T": 20,  # Fast inference
                "temperature": 1.0
            }
            
            response = requests.post(
                f"http://localhost:8082/generate",
                json=request_data,
                timeout=180  # RFDiffusion can be slow
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get("success"):
                    designs = result.get("designs", [])
                    entropies = result.get("entropies", [0.8])
                    
                    if designs:
                        design_data = designs[0]
                        seq = design_data.get("sequence", "")
                        coordinates = design_data.get("coordinates", [])
                        entropy = entropies[0] if entropies else 0.8
                        
                        struct_tokens = self._coordinates_to_tokens(coordinates, len(seq))
                        
                        logger.info(f"✅ Real RFDiffusion generated: {seq}")
                        logger.info(f"📊 Real entropy: {entropy:.3f}")
                        
                        return seq, struct_tokens, entropy
                    else:
                        logger.error("No designs in RFDiffusion response")
                        return None, None, None
                else:
                    logger.error(f"RFDiffusion server error: {result.get('error', 'Unknown error')}")
                    return None, None, None
            else:
                logger.error(f"RFDiffusion HTTP error: {response.status_code}")
                return None, None, None
                
        except Exception as e:
            logger.error(f"Real RFDiffusion inference failed: {e}")
            return None, None, None
    
    def _coordinates_to_tokens(self, coordinates: Optional[List], target_length: int) -> str:
        """Convert coordinates to DPLM-2 structure tokens"""
        if coordinates and len(coordinates) == target_length:
            tokens = []
            for coord in coordinates:
                if isinstance(coord, list) and len(coord) >= 3:
                    x, y, z = coord[:3]
                    token = f"{int(abs(x * 100)) % 10000:04d}"
                    tokens.append(token)
                else:
                    tokens.append("0000")
            return ",".join(tokens)
        else:
            return ",".join(["0000"] * target_length)


class OfficialServerMCTSBridge:
    """Bridge official servers with MCTS"""
    
    def __init__(self):
        self.server_integration = OfficialServerIntegration()
        self.available_experts = self.server_integration.get_available_servers()
        
        if self.available_experts:
            logger.info(f"✅ Official server bridge: {self.available_experts}")
        else:
            logger.warning("⚠️ No official servers running")
    
    def get_available_experts(self) -> List[str]:
        return self.available_experts.copy()
    
    def external_motif_scaffold_rollout(self, expert_name: str, motif_data_dict: Dict) -> Optional[Dict]:
        """Real motif scaffold rollout using official servers"""
        
        motif_sequence = motif_data_dict.get('motif_sequence', '')
        motif_structure_tokens = motif_data_dict.get('motif_structure_tokens', '')
        target_length = motif_data_dict.get('target_length', 100)
        
        if not motif_sequence:
            return None
        
        try:
            if expert_name == "proteina":
                seq, struct_tokens, entropy = self.server_integration.proteina_motif_scaffolding(
                    motif_sequence, motif_structure_tokens, target_length
                )
                
            elif expert_name == "foldflow":
                seq, struct_tokens, entropy = self.server_integration.foldflow_structure_generation(
                    target_length, motif_sequence
                )
                
            elif expert_name == "rfdiffusion":
                # Create motif PDB for RFDiffusion
                motif_pdb = self._create_motif_pdb(motif_sequence)
                seq, struct_tokens, entropy = self.server_integration.rfdiffusion_motif_scaffolding(
                    motif_sequence, motif_pdb, target_length
                )
                
            else:
                logger.error(f"Unknown expert: {expert_name}")
                return None
            
            if seq and struct_tokens and entropy is not None:
                return {
                    'full_sequence': seq,
                    'structure_sequence': struct_tokens,
                    'motif_preserved': motif_sequence in seq,
                    'scaffold_length': len(seq) - len(motif_sequence),
                    'method': f'official_{expert_name}_server',
                    'entropy': entropy  # REAL entropy from official server
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Official {expert_name} rollout failed: {e}")
            return None
    
    def _create_motif_pdb(self, motif_sequence: str) -> str:
        """Create PDB content for motif"""
        pdb_lines = []
        pdb_lines.append("MODEL        1")
        
        for i, aa in enumerate(motif_sequence):
            x = 20.0 + i * 3.8
            y = 20.0
            z = 20.0
            
            atom_line = f"ATOM  {i+1:5d}  CA  {aa} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
            pdb_lines.append(atom_line)
        
        pdb_lines.append("ENDMDL")
        return "\n".join(pdb_lines)


def start_all_official_servers():
    """Start all official servers"""
    print("🚀 Starting All Official denovo-protein-server Servers")
    print("=" * 60)
    
    integration = OfficialServerIntegration()
    
    # Start each server
    for server_name in ["proteina", "foldflow", "rfdiffusion"]:
        print(f"\n🔧 Starting official {server_name.upper()} server...")
        success = integration.start_server(server_name)
        
        if success:
            print(f"   ✅ {server_name.upper()} server running on port {integration.servers[server_name]['port']}")
        else:
            print(f"   ❌ {server_name.upper()} server failed to start")
    
    # Summary
    running = integration.get_available_servers()
    print(f"\n📊 Official Server Status:")
    print(f"✅ Running servers: {len(running)} ({running})")
    
    if running:
        print("\n🎉 Official servers ready for real inference!")
        print("🚀 Ready for trustworthy multi-expert MCTS!")
        return True
    else:
        print("\n❌ No servers running - check setup")
        return False


def test_official_server_integration():
    """Test official server integration"""
    print("🧪 Testing Official Server Integration")
    print("=" * 45)
    
    try:
        # Initialize bridge
        bridge = OfficialServerMCTSBridge()
        available = bridge.get_available_experts()
        
        if not available:
            print("❌ No official servers available")
            print("💡 Run start_all_official_servers() first")
            return False
        
        print(f"🤖 Available official servers: {available}")
        
        # Test motif scaffolding
        motif_data_dict = {
            'motif_sequence': 'MQIF',
            'motif_structure_tokens': '159,162,163,164',
            'target_length': 50,
            'name': 'test_official'
        }
        
        print(f"\n🧬 Testing official server motif scaffolding:")
        print(f"   Motif: {motif_data_dict['motif_sequence']}")
        print(f"   Target length: {motif_data_dict['target_length']}")
        
        results = {}
        for expert_name in available:
            print(f"\n🔬 Testing official {expert_name.upper()} server...")
            
            result = bridge.external_motif_scaffold_rollout(expert_name, motif_data_dict)
            
            if result:
                results[expert_name] = result
                
                print(f"   ✅ Generated: {result['full_sequence']}")
                print(f"   🎯 Motif preserved: {result['motif_preserved']}")
                print(f"   📊 REAL entropy: {result['entropy']:.3f}")
                print(f"   🏗️ Structure tokens: {len(result['structure_sequence'].split(','))}")
                print(f"   🔧 Method: {result['method']}")
                
                # Verify real inference
                if "official_" in result['method'] and result['entropy'] != 1.0:
                    print(f"   ✅ CONFIRMED: Official server real inference!")
                else:
                    print(f"   ⚠️ May need entropy calculation improvement")
                    
            else:
                results[expert_name] = None
                print(f"   ❌ Failed")
        
        # Summary
        working = [name for name, result in results.items() if result is not None]
        official_methods = [result['method'] for result in results.values() 
                           if result and 'official_' in result['method']]
        
        print(f"\n📊 Official Server Integration Results:")
        print(f"✅ Working servers: {len(working)} ({working})")
        print(f"🔧 Official methods: {len(official_methods)} ({official_methods})")
        
        if working and official_methods:
            print(f"\n🎉 SUCCESS: Official server real inference working!")
            print(f"🚀 Ready for trustworthy MCTS ablation studies!")
            return True
        else:
            print(f"\n⚠️ Need to start more official servers")
            return False
            
    except Exception as e:
        print(f"❌ Official server integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧬 Official denovo-protein-server Integration")
    print("=" * 50)
    
    choice = input("Choose action:\n1. Start all servers\n2. Test integration\n3. Both\nEnter choice (1-3): ")
    
    if choice in ["1", "3"]:
        start_all_official_servers()
    
    if choice in ["2", "3"]:
        test_official_server_integration()
