#!/usr/bin/env python3
"""
Quick test to verify RMSD and TM-score calculations work correctly.
"""

import numpy as np
import sys
import os

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from test_mcts_folding_ablation import calculate_rmsd_and_tmscore

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
    print(f"   RMSD: {rmsd:.6f} (expected: ~0.0)")
    print(f"   TM-score: {tm:.6f} (expected: ~1.0)")
    print(f"   ✅ PASS" if rmsd < 0.01 and tm > 0.99 else f"   ❌ FAIL")
    
    # Test 2: Translated structure (should still give RMSD≈0 after alignment)
    print("\n📊 Test 2: Translated structure (should align)")
    coords3 = coords1 + np.array([10.0, 5.0, -3.0])  # Translate
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords3)
    print(f"   RMSD: {rmsd:.6f} (expected: ~0.0 after alignment)")
    print(f"   TM-score: {tm:.6f} (expected: ~1.0)")
    print(f"   ✅ PASS" if rmsd < 0.01 and tm > 0.99 else f"   ❌ FAIL")
    
    # Test 3: Slightly perturbed structure
    print("\n📊 Test 3: Slightly perturbed structure")
    coords4 = coords1 + np.random.randn(50, 3) * 0.5  # Add small noise
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords4)
    print(f"   RMSD: {rmsd:.6f} (expected: ~0.5-1.0)")
    print(f"   TM-score: {tm:.6f} (expected: ~0.8-0.95)")
    print(f"   ✅ PASS" if 0.3 < rmsd < 2.0 and 0.7 < tm < 1.0 else f"   ❌ FAIL")
    
    # Test 4: Very different structure
    print("\n📊 Test 4: Very different structure")
    coords5 = np.random.randn(50, 3).astype(np.float32) * 10  # Completely different
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords5)
    print(f"   RMSD: {rmsd:.6f} (expected: >5.0)")
    print(f"   TM-score: {tm:.6f} (expected: <0.5)")
    print(f"   ✅ PASS" if rmsd > 3.0 and tm < 0.6 else f"   ❌ FAIL")
    
    # Test 5: Length mismatch handling
    print("\n📊 Test 5: Length mismatch (should truncate)")
    coords6 = coords1[:30]  # Shorter structure
    
    rmsd, tm = calculate_rmsd_and_tmscore(coords1, coords6)
    print(f"   RMSD: {rmsd:.6f}")
    print(f"   TM-score: {tm:.6f}")
    print(f"   ✅ PASS (no crash)")
    
    print("\n" + "="*80)
    print("✅ All tests completed!")
    print("="*80)

if __name__ == "__main__":
    test_metrics()
