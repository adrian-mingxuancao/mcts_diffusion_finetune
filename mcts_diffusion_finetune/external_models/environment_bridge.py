"""
Environment Bridge for DPLM-2 MCTS + External Models Integration
Allows the existing dplm_env to use external models from unified-experts-simple
"""

import os
import sys
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentBridge:
    """Bridge between DPLM environment and external models environment"""
    
    def __init__(self):
        self.dplm_env_path = "/net/scratch/caom/dplm_env"
        self.external_env_name = "unified-experts-simple"
        self.project_root = Path("/home/caom/AID3/dplm")
        
        # Check environments
        self._check_environments()
    
    def _check_environments(self):
        """Check if both environments are available"""
        # Check DPLM environment
        dplm_python = Path(self.dplm_env_path) / "bin" / "python"
        if dplm_python.exists():
            logger.info(f"✅ DPLM environment found: {self.dplm_env_path}")
            self.dplm_available = True
        else:
            logger.warning(f"⚠️ DPLM environment not found: {self.dplm_env_path}")
            self.dplm_available = False
        
        # Check external models environment
        try:
            result = subprocess.run(
                ["conda", "env", "list"], 
                capture_output=True, text=True, timeout=10
            )
            if self.external_env_name in result.stdout:
                logger.info(f"✅ External models environment found: {self.external_env_name}")
                self.external_available = True
            else:
                logger.warning(f"⚠️ External models environment not found: {self.external_env_name}")
                self.external_available = False
        except Exception as e:
            logger.error(f"Failed to check conda environments: {e}")
            self.external_available = False
    
    def run_in_dplm_env(self, python_code: str) -> Dict[str, Any]:
        """Run Python code in DPLM environment"""
        if not self.dplm_available:
            raise RuntimeError("DPLM environment not available")
        
        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_script = f.name
        
        try:
            # Set up environment variables for DPLM
            env = os.environ.copy()
            env.update({
                "PATH": f"{self.dplm_env_path}/bin:" + env.get("PATH", ""),
                "PYTHONPATH": "/home/caom/.cache/torch_extensions/py39_cu121/attn_core_inplace_cuda:" + env.get("PYTHONPATH", ""),
                "HF_HOME": "/net/scratch/caom/.cache/huggingface",
                "TRANSFORMERS_CACHE": "/net/scratch/caom/.cache/huggingface/transformers",
                "TORCH_HOME": "/net/scratch/caom/.cache/torch"
            })
            
            # Run in DPLM environment
            result = subprocess.run(
                [f"{self.dplm_env_path}/bin/python", temp_script],
                capture_output=True, text=True, cwd=str(self.project_root),
                env=env, timeout=300
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        
        finally:
            # Cleanup
            os.unlink(temp_script)
    
    def run_in_external_env(self, python_code: str) -> Dict[str, Any]:
        """Run Python code in external models environment"""
        if not self.external_available:
            raise RuntimeError("External models environment not available")
        
        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_code)
            temp_script = f.name
        
        try:
            # Run in external environment using conda
            cmd = [
                "conda", "run", "-n", self.external_env_name,
                "python", temp_script
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                cwd=str(self.project_root), timeout=300
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        
        finally:
            # Cleanup
            os.unlink(temp_script)
    
    def expert_rollout_bridge(self, expert_name: str, **kwargs) -> List[str]:
        """Bridge function to run external expert from DPLM environment"""
        
        # Prepare the external model call
        python_code = f"""
import sys
import json
sys.path.insert(0, 'mcts_diffusion_finetune')

try:
    from external_models.unified_experts_official import UnifiedExpertIntegrationOfficial
    
    # Initialize integration
    integration = UnifiedExpertIntegrationOfficial()
    
    # Prepare kwargs
    kwargs = {json.dumps(kwargs)}
    
    # Run expert rollout
    results = integration.expert_rollout('{expert_name}', **kwargs)
    
    # Output results as JSON
    output = {{
        "success": True,
        "results": results,
        "error": None
    }}
    print("BRIDGE_OUTPUT_START")
    print(json.dumps(output))
    print("BRIDGE_OUTPUT_END")
    
except Exception as e:
    import traceback
    output = {{
        "success": False,
        "results": [],
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    print("BRIDGE_OUTPUT_START")
    print(json.dumps(output))
    print("BRIDGE_OUTPUT_END")
"""
        
        # Run in external environment
        result = self.run_in_external_env(python_code)
        
        if not result["success"]:
            logger.error(f"External environment failed: {result['stderr']}")
            return []
        
        # Parse output
        try:
            stdout = result["stdout"]
            start_marker = "BRIDGE_OUTPUT_START"
            end_marker = "BRIDGE_OUTPUT_END"
            
            start_idx = stdout.find(start_marker)
            end_idx = stdout.find(end_marker)
            
            if start_idx == -1 or end_idx == -1:
                logger.error("Could not find bridge output markers")
                return []
            
            json_str = stdout[start_idx + len(start_marker):end_idx].strip()
            output = json.loads(json_str)
            
            if output["success"]:
                return output["results"]
            else:
                logger.error(f"Expert rollout failed: {output['error']}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to parse bridge output: {e}")
            return []


class BridgedExternalExperts:
    """External experts that can be used from DPLM environment via bridge"""
    
    def __init__(self):
        self.bridge = EnvironmentBridge()
        self._check_available_experts()
    
    def _check_available_experts(self):
        """Check which external experts are available"""
        python_code = """
import sys
import json
sys.path.insert(0, 'mcts_diffusion_finetune')

try:
    from external_models.unified_experts_official import UnifiedExpertIntegrationOfficial
    integration = UnifiedExpertIntegrationOfficial()
    available = integration.get_available_expert_names()
    
    output = {
        "success": True,
        "available_experts": available
    }
    print("BRIDGE_OUTPUT_START")
    print(json.dumps(output))
    print("BRIDGE_OUTPUT_END")
    
except Exception as e:
    output = {
        "success": False,
        "available_experts": [],
        "error": str(e)
    }
    print("BRIDGE_OUTPUT_START")
    print(json.dumps(output))
    print("BRIDGE_OUTPUT_END")
"""
        
        result = self.bridge.run_in_external_env(python_code)
        
        if result["success"]:
            try:
                stdout = result["stdout"]
                start_idx = stdout.find("BRIDGE_OUTPUT_START")
                end_idx = stdout.find("BRIDGE_OUTPUT_END")
                json_str = stdout[start_idx + len("BRIDGE_OUTPUT_START"):end_idx].strip()
                output = json.loads(json_str)
                
                if output["success"]:
                    self.available_experts = output["available_experts"]
                    logger.info(f"✅ Available external experts: {self.available_experts}")
                else:
                    self.available_experts = []
                    logger.error(f"Failed to get experts: {output.get('error', 'Unknown error')}")
            except Exception as e:
                self.available_experts = []
                logger.error(f"Failed to parse expert list: {e}")
        else:
            self.available_experts = []
            logger.error("Failed to check available experts")
    
    def get_available_expert_names(self) -> List[str]:
        """Get list of available expert names"""
        return self.available_experts.copy()
    
    def expert_rollout(self, expert_name: str, **kwargs) -> List[str]:
        """Perform expert rollout via bridge"""
        if expert_name not in self.available_experts:
            raise ValueError(f"Expert '{expert_name}' not available. Available: {self.available_experts}")
        
        return self.bridge.expert_rollout_bridge(expert_name, **kwargs)


def test_bridge():
    """Test the environment bridge"""
    print("🧪 Testing Environment Bridge")
    print("=" * 40)
    
    try:
        # Test bridge initialization
        bridge = EnvironmentBridge()
        print(f"DPLM environment: {'✅' if bridge.dplm_available else '❌'}")
        print(f"External environment: {'✅' if bridge.external_available else '❌'}")
        
        if not (bridge.dplm_available and bridge.external_available):
            print("⚠️ Cannot test bridge - missing environments")
            return False
        
        # Test DPLM environment
        print("\\n🔧 Testing DPLM environment...")
        dplm_code = """
import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
except:
    print("PyTorch not available")

try:
    sys.path.insert(0, 'mcts_diffusion_finetune')
    from core.dplm2_integration import DPLM2Integration
    print("✅ DPLM-2 available")
except Exception as e:
    print(f"❌ DPLM-2 failed: {e}")
"""
        
        result = bridge.run_in_dplm_env(dplm_code)
        if result["success"]:
            print("✅ DPLM environment working")
            print(result["stdout"])
        else:
            print("❌ DPLM environment failed")
            print(result["stderr"])
        
        # Test external experts bridge
        print("\\n🔧 Testing external experts bridge...")
        experts = BridgedExternalExperts()
        available = experts.get_available_expert_names()
        print(f"Available experts: {available}")
        
        if available:
            # Test an expert
            expert_name = available[0]
            print(f"\\n🧪 Testing {expert_name}...")
            
            if expert_name == "foldflow":
                results = experts.expert_rollout(expert_name, length=30, num_samples=1)
            elif expert_name == "proteinmpnn":
                mock_pdb = "ATOM      1  N   ALA A   1      20.154  16.967  14.365  1.00 20.00           N\\nEND"
                results = experts.expert_rollout(expert_name, pdb_content=mock_pdb, num_samples=1)
            else:
                results = []
            
            print(f"✅ Generated {len(results)} outputs")
        
        print("\\n🎉 Bridge test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_bridge()





