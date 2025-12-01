"""
MCTS External Model Bridge
Bridges external models (ProteInA, FoldFlow, RFDiffusion) with DPLM-2 MCTS
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

class MCTSExternalBridge:
    """Bridge external models for MCTS rollouts"""
    
    def __init__(self):
        self.external_env_name = "unified-experts-simple"
        self.project_root = Path("/home/caom/AID3/dplm")
        self.available_experts = []
        self._check_external_experts()
    
    def _check_external_experts(self):
        """Check which external experts are available"""
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
    from external_models.real_motif_experts import RealMotifExpertsIntegration
    integration = RealMotifExpertsIntegration()
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
                        logger.info(f"✅ Available external experts: {self.available_experts}")
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
    
    def external_expert_rollout(self, expert_name: str, motif_sequence: str, 
                               motif_structure: str, target_length: int, **kwargs) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Perform external expert rollout for MCTS
        
        Args:
            expert_name: Name of external expert (proteina, foldflow, rfdiffusion)
            motif_sequence: Motif amino acid sequence
            motif_structure: Motif structure tokens
            target_length: Total target length
            
        Returns:
            (generated_sequence, structure_tokens) or (None, None) if failed
        """
        if expert_name not in self.available_experts:
            logger.error(f"Expert {expert_name} not available. Available: {self.available_experts}")
            return None, None
        
        try:
            # Prepare the rollout call
            rollout_code = f'''
import sys
import json
sys.path.insert(0, "mcts_diffusion_finetune")

try:
    from external_models.real_motif_experts import RealMotifExpertsIntegration
    
    integration = RealMotifExpertsIntegration()
    
    generated_seq, structure_tokens = integration.motif_scaffold_rollout(
        expert_name="{expert_name}",
        motif_sequence="{motif_sequence}",
        motif_structure="{motif_structure}",
        target_length={target_length}
    )
    
    result = {{
        "success": True,
        "sequence": generated_seq,
        "structure_tokens": structure_tokens,
        "error": None
    }}
    
    print("BRIDGE_START")
    print(json.dumps(result))
    print("BRIDGE_END")
    
except Exception as e:
    import traceback
    result = {{
        "success": False,
        "sequence": None,
        "structure_tokens": None,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    
    print("BRIDGE_START")
    print(json.dumps(result))
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
                            return output.get("sequence"), output.get("structure_tokens")
                        else:
                            logger.error(f"External rollout failed: {output.get('error')}")
                            return None, None
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse bridge output: {e}")
                        return None, None
                else:
                    logger.error("Bridge markers not found in output")
                    return None, None
            else:
                logger.error(f"External rollout command failed: {result.stderr}")
                return None, None
                
        except Exception as e:
            logger.error(f"External expert rollout failed: {e}")
            return None, None


def test_mcts_bridge():
    """Test the MCTS external bridge"""
    print("🧪 Testing MCTS External Model Bridge")
    print("=" * 50)
    
    try:
        bridge = MCTSExternalBridge()
        available = bridge.get_available_experts()
        
        if not available:
            print("❌ No external experts available")
            return False
        
        print(f"🤖 Available experts: {available}")
        
        # Test motif scaffolding with each expert
        test_motif = "MQIF"
        test_structure = "159,162,163,164"
        test_length = 50
        
        print(f"\n🧬 Testing motif scaffolding:")
        print(f"   Motif: {test_motif}")
        print(f"   Structure: {test_structure}")
        print(f"   Target length: {test_length}")
        
        results = {}
        for expert_name in available:
            print(f"\n🔬 Testing {expert_name.upper()} rollout...")
            
            seq, struct_tokens = bridge.external_expert_rollout(
                expert_name=expert_name,
                motif_sequence=test_motif,
                motif_structure=test_structure,
                target_length=test_length
            )
            
            if seq and struct_tokens:
                motif_preserved = test_motif in seq
                results[expert_name] = {
                    "sequence": seq,
                    "motif_preserved": motif_preserved,
                    "length": len(seq),
                    "success": True
                }
                
                print(f"   ✅ Generated: {seq}")
                print(f"   🎯 Motif preserved: {motif_preserved}")
                print(f"   📏 Length: {len(seq)}")
                print(f"   🏗️ Structure tokens: {len(struct_tokens)}")
            else:
                results[expert_name] = {"success": False}
                print(f"   ❌ Failed")
        
        # Summary
        working = [name for name, result in results.items() if result.get("success")]
        print(f"\n📊 Bridge Test Results:")
        print(f"✅ Working experts: {len(working)} ({working})")
        
        if working:
            print("🎉 MCTS bridge ready for external expert rollouts!")
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
    test_mcts_bridge()





