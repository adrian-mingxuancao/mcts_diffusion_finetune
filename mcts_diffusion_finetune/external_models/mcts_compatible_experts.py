"""
MCTS Compatible External Experts
Creates external expert objects that work seamlessly with existing motif_scaffolding_mcts.py
"""

import logging
from typing import Dict, List, Optional, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCTSCompatibleExternalExpert:
    """External expert that's compatible with existing MCTS pipeline"""
    
    def __init__(self, expert_name: str):
        self.expert_name = expert_name.lower()
        self.name = expert_name.upper()
        
        # Initialize bridge
        try:
            from external_models.mcts_integration_bridge import MCTSIntegrationBridge
            self.bridge = MCTSIntegrationBridge()
            
            if self.expert_name not in self.bridge.get_available_experts():
                raise ValueError(f"Expert {expert_name} not available via bridge")
            
            logger.info(f"✅ {self.name} expert initialized with bridge")
            
        except Exception as e:
            logger.error(f"Failed to initialize {expert_name} expert: {e}")
            raise
    
    def get_name(self) -> str:
        """Get expert name (required by existing MCTS)"""
        return self.name
    
    def generate_scaffold(self, motif_data, scaffold_length: int, **kwargs) -> Optional[Dict]:
        """
        Generate scaffold (required by existing MCTS)
        
        Args:
            motif_data: Motif data (dict format from existing MCTS)
            scaffold_length: Length of scaffold to generate
            
        Returns:
            Dict with generation results or None if failed
        """
        try:
            # Handle the data format from existing MCTS
            if hasattr(motif_data, 'motif_sequence'):
                # MotifScaffoldingData object
                motif_data_dict = {
                    'motif_sequence': motif_data.motif_sequence,
                    'motif_structure_tokens': motif_data.motif_structure_tokens,
                    'target_length': motif_data.target_length,
                    'name': motif_data.name
                }
            else:
                # Dict format (from existing MCTS expert rollout)
                motif_data_dict = {
                    'motif_sequence': motif_data.get('motif_sequence', ''),
                    'motif_structure_tokens': motif_data.get('motif_structure_tokens', ''),
                    'target_length': motif_data.get('target_length', len(motif_data.get('motif_sequence', '')) + scaffold_length),
                    'name': motif_data.get('name', 'unknown')
                }
            
            # Use bridge for generation
            result = self.bridge.external_motif_scaffold_rollout(self.expert_name, motif_data_dict)
            
            if result:
                logger.info(f"✅ {self.name} generated scaffold successfully")
                return result
            else:
                logger.error(f"❌ {self.name} scaffold generation failed")
                return None
                
        except Exception as e:
            logger.error(f"{self.name} scaffold generation error: {e}")
            return None


def create_external_experts_for_mcts() -> List[MCTSCompatibleExternalExpert]:
    """Create external expert objects for existing MCTS pipeline"""
    
    experts = []
    
    # Try to create each external expert
    for expert_name in ['foldflow', 'rfdiffusion', 'proteina']:
        try:
            expert = MCTSCompatibleExternalExpert(expert_name)
            experts.append(expert)
            print(f"✅ Created {expert_name.upper()} expert for MCTS")
        except Exception as e:
            print(f"⚠️ Failed to create {expert_name.upper()} expert: {e}")
    
    print(f"🤖 Created {len(experts)} external experts for MCTS")
    return experts


def test_mcts_compatible_experts():
    """Test MCTS compatible external experts"""
    print("🧪 Testing MCTS Compatible External Experts")
    print("=" * 50)
    
    try:
        # Create external experts
        external_experts = create_external_experts_for_mcts()
        
        if not external_experts:
            print("❌ No external experts created")
            return False
        
        print(f"🤖 Testing {len(external_experts)} external experts")
        
        # Test motif data in existing MCTS format
        motif_data_dict = {
            'motif_sequence': 'MQIF',
            'motif_structure_tokens': '159,162,163,164',
            'target_length': 50,
            'name': 'test_motif'
        }
        
        print(f"\n🧬 Test motif scaffolding:")
        print(f"   Motif: {motif_data_dict['motif_sequence']}")
        print(f"   Target length: {motif_data_dict['target_length']}")
        
        # Test each expert
        results = {}
        for expert in external_experts:
            expert_name = expert.get_name()
            print(f"\n🔬 Testing {expert_name}...")
            
            try:
                scaffold_length = motif_data_dict['target_length'] - len(motif_data_dict['motif_sequence'])
                result = expert.generate_scaffold(motif_data_dict, scaffold_length=scaffold_length)
                
                if result:
                    results[expert_name] = result
                    
                    print(f"   ✅ Generated: {result['full_sequence']}")
                    print(f"   🎯 Motif preserved: {result['motif_preserved']}")
                    print(f"   📏 Scaffold length: {result['scaffold_length']}")
                    print(f"   🏗️ Structure sequence: {len(result['structure_sequence'])} chars")
                    print(f"   🔧 Method: {result['method']}")
                    print(f"   📊 Entropy: {result.get('entropy', 'N/A')}")
                    
                else:
                    results[expert_name] = None
                    print(f"   ❌ Failed")
                    
            except Exception as e:
                results[expert_name] = None
                print(f"   ❌ Error: {e}")
        
        # Summary
        working = [name for name, result in results.items() if result is not None]
        print(f"\n📊 MCTS Compatible Expert Results:")
        print(f"✅ Working experts: {len(working)} ({working})")
        
        if working:
            print("🎉 MCTS compatible external experts ready!")
            print("🚀 Can be used directly in existing motif_scaffolding_mcts.py")
            
            # Show how to use in existing pipeline
            print(f"\n📋 Usage in existing MCTS:")
            print(f"   from external_models.mcts_compatible_experts import create_external_experts_for_mcts")
            print(f"   external_experts = create_external_experts_for_mcts()")
            print(f"   mcts = MotifScaffoldingMCTS(dplm2, external_experts)")
            
            return True
        else:
            print("❌ No external experts working")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_mcts_compatible_experts()





