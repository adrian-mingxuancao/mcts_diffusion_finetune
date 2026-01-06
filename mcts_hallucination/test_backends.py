#!/usr/bin/env python
"""
Quick test script to verify ESMFold, Boltz, Chai, and ProteinMPNN integration.
"""

import sys
import os

# Add paths
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_diffusion_finetune')
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination')

# Set ProteinMPNN path
os.environ['PROTEINMPNN_PATH'] = '/home/caom/AID3/dplm/denovo-protein-server/third_party/proteinmpnn'

import numpy as np
import torch

def test_torch():
    """Test PyTorch and CUDA availability."""
    print("\n" + "="*60)
    print("TEST 1: PyTorch and CUDA")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return torch.cuda.is_available()


def test_esmfold(use_mock=False):
    """Test ESMFold integration."""
    print("\n" + "="*60)
    print(f"TEST 2: ESMFold (mock={use_mock})")
    print("="*60)
    
    from core.esmfold_integration import ESMFoldIntegration
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esmfold = ESMFoldIntegration(device=device, use_mock=use_mock)
    
    test_seq = "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLPLR"
    print(f"Test sequence: {test_seq[:30]}... ({len(test_seq)} aa)")
    
    result = esmfold.predict_structure(test_seq)
    
    print(f"Coordinates shape: {result['coordinates'].shape}")
    print(f"Confidence shape: {len(result['confidence'])}")
    print(f"Mean pLDDT: {np.mean(result['confidence']):.1f}")
    print("✅ ESMFold test PASSED")
    return True


def test_proteinmpnn(use_mock=False):
    """Test ProteinMPNN integration."""
    print("\n" + "="*60)
    print(f"TEST 3: ProteinMPNN (mock={use_mock})")
    print("="*60)
    
    from core.hallucination_expert import ProteinMPNNIntegration
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        mpnn = ProteinMPNNIntegration(use_real=not use_mock, device=device)
        
        # Generate mock coordinates for testing
        n_residues = 50
        coords = np.cumsum(np.random.randn(n_residues, 3) * 3.8, axis=0)
        masked_seq = 'X' * n_residues
        
        print(f"Input: {n_residues} residues, all masked")
        designed_seq = mpnn.design_sequence(coords, masked_sequence=masked_seq)
        print(f"Designed sequence: {designed_seq[:30]}... ({len(designed_seq)} aa)")
        print("✅ ProteinMPNN test PASSED")
        return True
    except Exception as e:
        print(f"❌ ProteinMPNN test FAILED: {e}")
        return False


def test_boltz(use_mock=True):
    """Test Boltz integration via ABCFold."""
    print("\n" + "="*60)
    print(f"TEST 4: Boltz (mock={use_mock})")
    print("="*60)
    
    from core.abcfold_integration import ABCFoldIntegration
    
    try:
        boltz = ABCFoldIntegration(
            engine="boltz",
            use_mock=use_mock,
            allow_fallback=True,
        )
        
        test_seq = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        print(f"Test sequence: {test_seq} ({len(test_seq)} aa)")
        
        result = boltz.predict_structure(test_seq)
        
        print(f"Coordinates shape: {result['coordinates'].shape}")
        print(f"Confidence shape: {len(result['confidence'])}")
        print(f"Mean pLDDT: {np.mean(result['confidence']):.1f}")
        print("✅ Boltz test PASSED")
        return True
    except Exception as e:
        print(f"❌ Boltz test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chai(use_mock=True):
    """Test Chai-1 integration via Python API."""
    print("\n" + "="*60)
    print(f"TEST 5: Chai-1 (mock={use_mock})")
    print("="*60)
    
    from core.abcfold_integration import ABCFoldIntegration
    
    try:
        chai = ABCFoldIntegration(
            engine="chai1",
            use_mock=use_mock,
            allow_fallback=True,
        )
        
        test_seq = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"
        print(f"Test sequence: {test_seq} ({len(test_seq)} aa)")
        
        result = chai.predict_structure(test_seq)
        
        print(f"Coordinates shape: {result['coordinates'].shape}")
        print(f"Confidence shape: {len(result['confidence'])}")
        print(f"Mean pLDDT: {np.mean(result['confidence']):.1f}")
        print("✅ Chai-1 test PASSED")
        return True
    except Exception as e:
        print(f"❌ Chai-1 test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hallucination_expert(backend="esmfold", use_mock=False):
    """Test full hallucination expert pipeline."""
    print("\n" + "="*60)
    print(f"TEST 6: HallucinationExpert (backend={backend}, mock={use_mock})")
    print("="*60)
    
    from core.hallucination_expert import create_hallucination_expert
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        expert = create_hallucination_expert(
            structure_backend=backend,
            esmfold_device=device,
            use_mock=use_mock,
            use_real_proteinmpnn=not use_mock,
        )
        
        test_seq = "MKFLILLFNILCLFPVLAADNHGVGPQGASGVDPITFDINSNQTGVQLTLPLR"
        masked_positions = set(range(10, 30))  # Mask positions 10-29
        
        print(f"Input sequence: {test_seq[:30]}... ({len(test_seq)} aa)")
        print(f"Masked positions: {len(masked_positions)}")
        
        result = expert.generate_candidate(test_seq, masked_positions)
        
        if result:
            print(f"Output sequence: {result['sequence'][:30]}... ({len(result['sequence'])} aa)")
            print(f"Mean pLDDT: {result.get('mean_plddt', 'N/A')}")
            print("✅ HallucinationExpert test PASSED")
            return True
        else:
            print("❌ HallucinationExpert returned None")
            return False
    except Exception as e:
        print(f"❌ HallucinationExpert test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test hallucination backends")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for all tests")
    parser.add_argument("--real-boltz", action="store_true", help="Test real Boltz (requires GPU)")
    parser.add_argument("--real-chai", action="store_true", help="Test real Chai-1 (requires GPU)")
    parser.add_argument("--backend", default="esmfold", choices=["esmfold", "boltz", "chai1"],
                       help="Structure prediction backend to test in hallucination expert")
    args = parser.parse_args()
    
    print("\n" + "#"*60)
    print("# HALLUCINATION BACKEND INTEGRATION TESTS")
    print("#"*60)
    
    results = {}
    
    # Test 1: PyTorch/CUDA
    results['torch'] = test_torch()
    
    # Test 2: ESMFold
    results['esmfold'] = test_esmfold(use_mock=args.mock)
    
    # Test 3: ProteinMPNN
    results['proteinmpnn'] = test_proteinmpnn(use_mock=args.mock)
    
    # Test 4: Boltz
    boltz_mock = not args.real_boltz
    results['boltz'] = test_boltz(use_mock=boltz_mock)
    
    # Test 5: Chai
    chai_mock = not args.real_chai
    results['chai'] = test_chai(use_mock=chai_mock)
    
    # Test 6: Full pipeline
    results['hallucination'] = test_hallucination_expert(
        backend=args.backend, 
        use_mock=args.mock
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 ALL TESTS PASSED!" if all_passed else "⚠️ SOME TESTS FAILED"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
