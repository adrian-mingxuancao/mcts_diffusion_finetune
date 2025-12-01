"""
MCTS Integration Bridge for Fixed External Experts
Integrates fixed external models with existing motif_scaffolding_mcts.py pipeline
"""

import os
import sys
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTSIntegrationBridge:
    """Bridge fixed external models with existing MCTS pipeline"""
    
    def __init__(self):
        self.external_env_name = "unified-experts-simple"
        self.project_root = Path("/home/caom/AID3/dplm")
        self.available_experts = []
        self._check_external_experts()
    
    def _check_external_experts(self):
        """Check which fixed external experts are available"""
        try:
            # Test external environment
            cmd = [
                '/opt/conda/bin/conda', 'run', '-n', self.external_env_name,
                'python', '-c',
                '''
import sys
import json
sys.path.insert(0, "mcts_diffusion_finetune")

try:
    from external_models.fixed_motif_experts import FixedMotifExpertsIntegration
    integration = FixedMotifExpertsIntegration()
    available = integration.get_available_expert_names()
    
    print(json.dumps({"success": True, "experts": available}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
'''
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=str(self.project_root), timeout=60)
            
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    if output.get("success"):
                        self.available_experts = output.get("experts", [])
                        logger.info(f"✅ Available fixed external experts: {self.available_experts}")
                    else:
                        logger.error(f"External experts check failed: {output.get('error')}")
                except json.JSONDecodeError:
                    logger.error("Failed to parse external experts response")
            else:
                logger.error(f"External experts check failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to check external experts: {e}")
    
    def get_available_experts(self) -> List[str]:
        """Get list of available external expert names"""
        return self.available_experts.copy()
    
    def external_motif_scaffold_rollout(self, expert_name: str, motif_data_dict: Dict) -> Optional[Dict]:
        """
        Perform external expert rollout for MCTS in existing pipeline format
        
        Args:
            expert_name: Name of external expert (foldflow, rfdiffusion, proteina)
            motif_data_dict: Motif data in dict format from existing MCTS
            
        Returns:
            Dict with 'full_sequence', 'structure_sequence', etc. or None if failed
        """
        if expert_name not in self.available_experts:
            logger.error(f"Expert {expert_name} not available. Available: {self.available_experts}")
            return None
        
        try:
            # Extract required data from motif_data_dict
            motif_sequence = motif_data_dict.get('motif_sequence', '')
            motif_structure_tokens = motif_data_dict.get('motif_structure_tokens', '')
            target_length = motif_data_dict.get('target_length', 100)
            
            if not motif_sequence:
                logger.error("No motif sequence in motif_data_dict")
                return None
            
            # Prepare the rollout call
            rollout_code = f'''
import sys
import json
sys.path.insert(0, "mcts_diffusion_finetune")

try:
    from external_models.fixed_motif_experts import MCTSExternalExpertWrapper
    
    wrapper = MCTSExternalExpertWrapper("{expert_name}")
    
    motif_data_dict = {{
        "motif_sequence": "{motif_sequence}",
        "motif_structure_tokens": "{motif_structure_tokens}"
    }}
    
    scaffold_length = {target_length} - {len(motif_sequence)}
    
    result = wrapper.generate_scaffold(motif_data_dict, scaffold_length=scaffold_length)
    
    if result:
        output = {{
            "success": True,
            "result": result,
            "error": None
        }}
    else:
        output = {{
            "success": False,
            "result": None,
            "error": "Generation failed"
        }}
    
    print("BRIDGE_START")
    print(json.dumps(output))
    print("BRIDGE_END")
    
except Exception as e:
    import traceback
    output = {{
        "success": False,
        "result": None,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    
    print("BRIDGE_START")
    print(json.dumps(output))
    print("BRIDGE_END")
'''
            
            # Run in external environment
            cmd = [
                '/opt/conda/bin/conda', 'run', '-n', self.external_env_name,
                'python', '-c', rollout_code
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=str(self.project_root), timeout=120)
            
            if result.returncode == 0:
                # Parse bridge output
                stdout = result.stdout
                start_idx = stdout.find("BRIDGE_START")
                end_idx = stdout.find("BRIDGE_END")
                
                if start_idx != -1 and end_idx != -1:
                    json_str = stdout[start_idx + len("BRIDGE_START"):end_idx].strip()
                    try:
                        output = json.loads(json_str)
                        
                        if output.get("success"):
                            return output.get("result")
                        else:
                            logger.error(f"External rollout failed: {output.get('error')}")
                            return None
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse bridge output: {e}")
                        return None
                else:
                    logger.error("Bridge markers not found in output")
                    return None
            else:
                logger.error(f"External rollout command failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"External expert rollout failed: {e}")
            return None


def test_mcts_integration_bridge():
    """Test the MCTS integration bridge"""
    print("🧪 Testing MCTS Integration Bridge")
    print("=" * 45)
    
    try:
        bridge = MCTSIntegrationBridge()
        available = bridge.get_available_experts()
        
        if not available:
            print("❌ No external experts available")
            return False
        
        print(f"🤖 Available experts: {available}")
        
        # Test motif scaffolding with existing MCTS format
        motif_data_dict = {
            'motif_sequence': 'MQIF',
            'motif_structure_tokens': '159,162,163,164',
            'target_length': 50,
            'name': 'test_motif'
        }
        
        print(f"\n🧬 Testing motif scaffolding with existing MCTS format:")
        print(f"   Motif: {motif_data_dict['motif_sequence']}")
        print(f"   Structure tokens: {motif_data_dict['motif_structure_tokens']}")
        print(f"   Target length: {motif_data_dict['target_length']}")
        
        results = {}
        for expert_name in available:
            print(f"\n🔬 Testing {expert_name.upper()} via bridge...")
            
            result = bridge.external_motif_scaffold_rollout(expert_name, motif_data_dict)
            
            if result:
                results[expert_name] = result
                
                print(f"   ✅ Generated: {result['full_sequence']}")
                print(f"   🎯 Motif preserved: {result['motif_preserved']}")
                print(f"   📏 Scaffold length: {result['scaffold_length']}")
                print(f"   🏗️ Structure sequence: {len(result['structure_sequence'])} chars")
                print(f"   🔧 Method: {result['method']}")
                print(f"   📊 Entropy: {result.get('entropy', 'N/A')}")
                
                # Verify DPLM-2 compatibility
                structure_tokens = result['structure_sequence'].split(',')
                if len(structure_tokens) == motif_data_dict['target_length']:
                    print(f"   ✅ DPLM-2 compatible structure tokens")
                else:
                    print(f"   ⚠️ Structure token count mismatch: {len(structure_tokens)} vs {motif_data_dict['target_length']}")
            else:
                results[expert_name] = None
                print(f"   ❌ Failed")
        
        # Summary
        working = [name for name, result in results.items() if result is not None]
        print(f"\n📊 MCTS Integration Bridge Results:")
        print(f"✅ Working experts: {len(working)} ({working})")
        
        if working:
            print("🎉 MCTS integration bridge ready!")
            print("🚀 External experts can now be used in existing motif_scaffolding_mcts.py")
            
            # Show integration format
            print(f"\n📋 Integration format for existing MCTS:")
            for expert_name in working:
                result = results[expert_name]
                print(f"   {expert_name}: sequence={len(result['full_sequence'])}, structure={len(result['structure_sequence'])}, preserved={result['motif_preserved']}")
            
            return True
        else:
            print("❌ No external experts working via bridge")
            return False
            
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mcts_integration_bridge()





