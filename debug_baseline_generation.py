#!/usr/bin/env python3
"""
Debug script to test baseline generation and masking issues.
"""

import os, sys
# Add the mcts_diffusion_finetune directory to path to avoid parent core import
mcts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, mcts_dir)
sys.path.insert(0, os.path.join(os.path.dirname(mcts_dir), 'src'))

# Import directly from the file to avoid core module conflicts
from core.dplm2_integration_fixed_new import DPLM2IntegrationCorrected
from utils.cameo_data_loader import CAMEODataLoader

def test_baseline_generation():
    print("🧪 Testing baseline generation...")
    
    # Load structure
    loader = CAMEODataLoader()
    structure = loader.get_structure_by_index(0)
    print(f"Structure keys: {list(structure.keys())}")
    print(f"struct_ids type: {type(structure.get('struct_ids'))}")
    print(f"struct_ids length: {len(structure.get('struct_ids')) if structure.get('struct_ids') is not None else 'None'}")
    
    # Initialize DPLM2
    dplm2 = DPLM2IntegrationCorrected(model_name="airkingbd/dplm2_150m")
    
    # Test baseline generation
    try:
        print("\n🔄 Testing baseline generation...")
        baseline_seq = dplm2.generate_with_expert(expert_id=0, structure=structure, target_length=structure['length'])
        print(f"✅ Baseline generation successful: {len(baseline_seq)} chars")
        print(f"First 50 chars: {baseline_seq[:50]}")
        return baseline_seq
    except Exception as e:
        print(f"❌ Baseline generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_masking():
    print("\n🧪 Testing masking...")
    
    # Load structure
    loader = CAMEODataLoader()
    structure = loader.get_structure_by_index(0)
    
    # Initialize DPLM2
    dplm2 = DPLM2IntegrationCorrected(model_name="airkingbd/dplm2_150m")
    
    # Create a test sequence with some masked positions
    test_seq = "A" * 50  # 50 A's
    masked_seq = "A" * 20 + "X" * 10 + "A" * 20  # Mask middle 10 positions
    print(f"Original: {test_seq}")
    print(f"Masked:   {masked_seq}")
    
    try:
        print("\n🔄 Testing masked generation...")
        result = dplm2.generate_with_expert(
            expert_id=0, 
            structure=structure, 
            target_length=50,
            masked_sequence=masked_seq
        )
        print(f"✅ Masked generation successful: {len(result)} chars")
        print(f"Result: {result}")
        
        # Check if masked positions were actually changed
        original_masked = test_seq[20:30]
        result_masked = result[20:30]
        print(f"Original masked region: {original_masked}")
        print(f"Result masked region:   {result_masked}")
        print(f"Changed: {original_masked != result_masked}")
        
        return result
    except Exception as e:
        print(f"❌ Masked generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    baseline = test_baseline_generation()
    masked = test_masking()
    
    print(f"\n📊 Summary:")
    print(f"Baseline generation: {'✅' if baseline else '❌'}")
    print(f"Masked generation: {'✅' if masked else '❌'}")
