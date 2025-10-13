#!/usr/bin/env python3
"""
Debug script to test structure tokenizer behavior
"""

import torch
import numpy as np

def test_struct_tokenizer():
    try:
        from byprot.models.utils import get_struct_tokenizer
        
        print("🔧 Loading structure tokenizer...")
        struct_tokenizer = get_struct_tokenizer()
        print(f"✅ Loaded tokenizer: {type(struct_tokenizer)}")
        
        # Create test data (similar to working test code)
        seq_len = 10  # Small test
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy coordinates
        full_coords = torch.zeros((1, seq_len, 37, 3), dtype=torch.float32, device=device)
        # Add some dummy CA coordinates
        full_coords[0, :, 1, :] = torch.randn(seq_len, 3, device=device) * 10.0
        
        # Create residue mask
        res_mask = torch.ones((1, seq_len), dtype=torch.float32, device=device)
        
        # Create seq_length tensor
        seq_length = torch.tensor([seq_len], dtype=torch.long, device=device)
        
        print(f"🔍 Input shapes:")
        print(f"  full_coords: {full_coords.shape}")
        print(f"  res_mask: {res_mask.shape}")
        print(f"  seq_length: {seq_length}")
        
        # Test tokenization
        print(f"🔍 Calling tokenizer...")
        result = struct_tokenizer.tokenize(full_coords, res_mask, seq_length)
        
        print(f"🔍 Result:")
        print(f"  Type: {type(result)}")
        print(f"  Value: {result}")
        
        if hasattr(result, 'shape'):
            print(f"  Shape: {result.shape}")
        if hasattr(result, 'dtype'):
            print(f"  Dtype: {result.dtype}")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_struct_tokenizer()
