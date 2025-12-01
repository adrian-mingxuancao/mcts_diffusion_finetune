#!/usr/bin/env python3
"""
Simple standalone test for RMSD and TM-score calculations.
No heavy imports, just the metrics.
"""

import numpy as np
from typing import Tuple

def calculate_rmsd_and_tmscore(predicted_coords: np.ndarray, reference_coords: np.ndarray) -> Tuple[float, float]:
    """
    Calculate RMSD and TM-score between predicted and reference structures.
    
    Uses:
    - RMSD: OpenFold's superimpose (Kabsch optimal alignment); falls back to unaligned RMSD if that fails
    - TM-score: tmtools.tm_align for optimal alignment; falls back to length-scaled TM from coordinate distances
    """
    try:
        pred_coords = np.array(predicted_coords, dtype=np.float32)
        ref_coords = np.array(reference_coords, dtype=np.float32)
        
        # Handle length mismatches
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        if len(pred_coords) == 0:
            return float('inf'), 0.0
        
        # Calculate RMSD using OpenFold's superimpose
        rmsd = None
        try:
            from openfold.utils.superimposition import superimpose
            import torch
            
            # Convert to torch tensors
            pred_torch = torch.from_numpy(pred_coords).unsqueeze(0)  # [1, N, 3]
            ref_torch = torch.from_numpy(ref_coords).unsqueeze(0)    # [1, N, 3]
            
            # Superimpose returns aligned coordinates and RMSD
            aligned_pred, rmsd_tensor = superimpose(ref_torch, pred_torch)
            rmsd = rmsd_tensor.item()
            print(f"  ✅ OpenFold RMSD: {rmsd:.3f}Å")
            
        except Exception as e:
            print(f"  ⚠️ OpenFold superimpose failed: {e}, using fallback RMSD")
            # Fallback: unaligned RMSD (not ideal but better than nothing)
            rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords) ** 2, axis=1)))
            print(f"  ⚠️ Fallback RMSD: {rmsd:.3f}Å (unaligned)")
        
        # Calculate TM-score using tmtools
        tm_score = None
        try:
            import tmtools
            
            # tmtools expects coordinates as [N, 3] numpy arrays
            tm_results = tmtools.tm_align(pred_coords, ref_coords, ref_coords, ref_coords)
            tm_score = tm_results.tm_norm_chain1  # TM-score normalized by chain 1 length
            print(f"  ✅ tmtools TM-score: {tm_score:.3f}")
            
        except Exception as e:
            print(f"  ⚠️ tmtools tm_align failed: {e}, using fallback TM-score")
            # Fallback: length-scaled TM-score from coordinate distances
            L_target = len(ref_coords)
            d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
            
            distances = np.sqrt(np.sum((pred_coords - ref_coords) ** 2, axis=1))
            tm_score = np.mean(1.0 / (1.0 + (distances / d_0) ** 2))
            print(f"  ⚠️ Fallback TM-score: {tm_score:.3f}")
        
        return rmsd, tm_score
        
    except Exception as e:
        print(f"⚠️ RMSD/TM-score calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0

def test_metrics():
    """Test RMSD and TM-score calculation with simple cases"""
    
    print("="*80)
    print("TESTING RMSD AND TM-SCORE CALCULATIONS")
    print("="*80)
    
    # Test 1: Identical structures (should give RMSD=0, TM=1.0)
    print("\n📊 Test 1: Identical structures")
    coords1 = np.random.randn(50, 3).astype(np.float32)
    coords2 = coords1.copy()
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords2)
    print(f"   Result: RMSD={rmsd:.6f}, TM-score={tm:.6f}")
    print(f"   Expected: RMSD≈0.0, TM-score≈1.0")
    print(f"   ✅ PASS" if rmsd < 0.01 and tm > 0.99 else f"   ❌ FAIL")
    
    # Test 2: Translated structure (should still give RMSD≈0 after alignment)
    print("\n📊 Test 2: Translated structure (should align)")
    coords3 = coords1 + np.array([10.0, 5.0, -3.0])  # Translate
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords3)
    print(f"   Result: RMSD={rmsd:.6f}, TM-score={tm:.6f}")
    print(f"   Expected: RMSD≈0.0 (after alignment), TM-score≈1.0")
    print(f"   ✅ PASS" if rmsd < 0.01 and tm > 0.99 else f"   ❌ FAIL")
    
    # Test 3: Slightly perturbed structure
    print("\n📊 Test 3: Slightly perturbed structure")
    coords4 = coords1 + np.random.randn(50, 3) * 0.5  # Add small noise
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords4)
    print(f"   Result: RMSD={rmsd:.6f}, TM-score={tm:.6f}")
    print(f"   Expected: RMSD≈0.5-1.0, TM-score≈0.8-0.95")
    print(f"   ✅ PASS" if 0.3 < rmsd < 2.0 and 0.7 < tm < 1.0 else f"   ❌ FAIL")
    
    print("\n" + "="*80)
    print("✅ Metrics test completed!")
    print("="*80)

if __name__ == "__main__":
    test_metrics()
