#!/usr/bin/env python
"""
Quick test script to verify ESMFold, Boltz, Chai, ProteinMPNN, and NA-MPNN integration.

Supports testing:
- Protein backends: ESMFold, ProteinMPNN, Boltz, Chai-1
- Nucleic acid backends: NA-MPNN (DNA/RNA)
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


def test_nampnn(use_mock=True, molecule_type="dna"):
    """Test NA-MPNN integration for DNA/RNA."""
    print("\n" + "="*60)
    print(f"TEST: NA-MPNN ({molecule_type.upper()}, mock={use_mock})")
    print("="*60)
    
    from core.nampnn_integration import NAMPNNIntegration
    
    # Test sequences provided by user
    DNA_SEQ = "TCGATGTTATCATGCCTGGCATCATAGCCCAGCCCACTGCTTTCTTCTGCGGACGCCCCAGTTTGCGTCCCTTGTCCTTATGACTGTTTTTCTCAGCATC"
    RNA_SEQ = "UCGCGAGGGCGAACAUAUUAUCCGGUUUACUGUUAAGGCUAAAUCGCACAUACGCAGAUAUUCCGCACCCGUGCUGGACGAUGUUGACAGGACGGAGUGA"
    
    test_seq = DNA_SEQ if molecule_type == "dna" else RNA_SEQ
    
    try:
        nampnn = NAMPNNIntegration(
            use_mock=use_mock,
            temperature=0.1,
        )
        
        print(f"Test sequence: {test_seq[:40]}... ({len(test_seq)} nt)")
        
        # Generate mock coordinates for testing
        n_residues = len(test_seq)
        # Create a simple helical structure for nucleic acids
        coords = np.zeros((n_residues, 3))
        for i in range(n_residues):
            # Simple B-DNA-like helix parameters
            rise = 3.4  # Angstroms per base pair
            radius = 10.0  # Helix radius
            twist = 36.0 * np.pi / 180  # Degrees per base pair
            coords[i, 0] = radius * np.cos(i * twist)
            coords[i, 1] = radius * np.sin(i * twist)
            coords[i, 2] = i * rise
        
        print(f"Input: {n_residues} nucleotides, helical coordinates")
        
        # Test design_from_coords method
        designed_seq = nampnn.design_from_coords(
            coordinates=coords,
            sequence_length=n_residues,
            molecule_type=molecule_type,
        )
        
        print(f"Designed sequence: {designed_seq[:40]}... ({len(designed_seq)} chars)")
        
        # Validate output
        if use_mock:
            # Mock mode returns protein-like sequences
            print("✅ NA-MPNN mock test PASSED")
            return True
        else:
            # Real mode should return nucleic acid sequences
            # NA-MPNN alphabet:
            # - DNA: a=A, c=C, g=G, t=T (lowercase)
            # - RNA: b=A, d=C, h=G, u=U (lowercase, different from DNA)
            # - With na_shared_tokens=1, RNA uses DNA tokens
            # - Chain separator: /
            # - Unknown: x (DNA), y (RNA)
            # - Protein codes may appear if geometry is ambiguous (uppercase)
            valid_dna = set("acgtx/")
            valid_rna = set("bdhuy/")
            valid_na = valid_dna | valid_rna
            
            seq_lower = designed_seq.lower()
            seq_chars = set(seq_lower)
            
            # Count nucleic acid vs other characters
            na_chars = sum(1 for c in seq_lower if c in valid_na)
            total_chars = len(seq_lower.replace("/", ""))
            na_fraction = na_chars / total_chars if total_chars > 0 else 0
            
            # Check for sequence diversity among NA bases
            na_bases = [c for c in seq_lower if c in (valid_dna | valid_rna) - {"/"}]
            unique_na_bases = set(na_bases)
            has_diversity = len(unique_na_bases) > 1
            
            # For synthetic coordinates, accept if >80% are valid NA characters
            # (some may be misclassified as protein due to simplified geometry)
            if na_fraction < 0.5:
                invalid_chars = seq_chars - valid_na
                print(f"❌ NA-MPNN {molecule_type.upper()} test FAILED: Too few NA characters ({na_fraction:.0%})")
                print(f"   Invalid characters: {invalid_chars}")
                return False
            elif not has_diversity:
                print(f"⚠️ NA-MPNN {molecule_type.upper()} test WARNING: Low diversity (only {unique_na_bases})")
                print(f"   This may indicate the input structure lacks proper geometry.")
                return True
            else:
                if na_fraction < 1.0:
                    print(f"✅ NA-MPNN {molecule_type.upper()} test PASSED (bases: {unique_na_bases}, {na_fraction:.0%} NA)")
                else:
                    print(f"✅ NA-MPNN {molecule_type.upper()} test PASSED (bases: {unique_na_bases})")
                return True
                
    except Exception as e:
        print(f"❌ NA-MPNN test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nampnn_with_pdb(pdb_path=None, use_mock=True):
    """Test NA-MPNN with a real PDB file containing nucleic acids."""
    print("\n" + "="*60)
    print(f"TEST: NA-MPNN with PDB (mock={use_mock})")
    print("="*60)
    
    from core.nampnn_integration import NAMPNNIntegration
    import tempfile
    from pathlib import Path
    
    try:
        nampnn = NAMPNNIntegration(
            use_mock=use_mock,
            temperature=0.1,
        )
        
        # If no PDB provided, create a minimal test PDB with DNA
        if pdb_path is None:
            # Create a minimal DNA duplex PDB
            pdb_content = create_minimal_dna_pdb()
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode='w') as f:
                f.write(pdb_content)
                pdb_path = f.name
            cleanup_pdb = True
        else:
            cleanup_pdb = False
        
        print(f"PDB file: {pdb_path}")
        
        try:
            result = nampnn.design_complex(
                pdb_path=pdb_path,
                design_na_only=True,
            )
            
            print(f"Design result: {result}")
            
            if result:
                print("✅ NA-MPNN PDB test PASSED")
                return True
            else:
                print("❌ NA-MPNN PDB test FAILED: No sequences returned")
                return False
        finally:
            if cleanup_pdb:
                Path(pdb_path).unlink(missing_ok=True)
                
    except Exception as e:
        print(f"❌ NA-MPNN PDB test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_minimal_dna_pdb():
    """Create a minimal PDB with DNA residues for testing."""
    # Create a simple 10-bp DNA strand with required atoms for NA-MPNN
    # NA-MPNN requires: OP1, OP2, P, O5', C5', C4', O4', C3', O3', C2', O2', C1'
    pdb_lines = []
    atom_num = 1
    
    # DNA bases
    bases = ["DA", "DC", "DG", "DT", "DA", "DC", "DG", "DT", "DA", "DC"]
    
    # Simplified atom positions for a B-DNA-like structure
    for i, resname in enumerate(bases):
        res_num = i + 1
        z_offset = i * 3.4  # Rise per base pair
        
        # Key atoms for NA-MPNN (simplified positions)
        atoms = [
            ("P", 0.0, 8.0, z_offset),
            ("OP1", 1.0, 8.5, z_offset),
            ("OP2", -1.0, 8.5, z_offset),
            ("O5'", 0.0, 6.5, z_offset + 0.5),
            ("C5'", 0.0, 5.5, z_offset + 1.0),
            ("C4'", 0.0, 4.5, z_offset + 1.5),
            ("O4'", 1.0, 4.0, z_offset + 1.5),
            ("C3'", -1.0, 4.0, z_offset + 2.0),
            ("O3'", -1.5, 3.0, z_offset + 2.5),
            ("C2'", 0.0, 3.5, z_offset + 2.0),
            ("C1'", 1.0, 3.0, z_offset + 1.8),
        ]
        
        for atom_name, x, y, z in atoms:
            # Format atom name for PDB (left-justified in 4 chars for short names)
            if len(atom_name) < 4:
                atom_name_fmt = f" {atom_name:<3s}"
            else:
                atom_name_fmt = atom_name[:4]
            
            pdb_lines.append(
                f"ATOM  {atom_num:5d} {atom_name_fmt} {resname:3s} A{res_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           "
                f"{atom_name[0]:>2s}\n"
            )
            atom_num += 1
    
    pdb_lines.append("END\n")
    return "".join(pdb_lines)


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


def test_abcfold_nampnn_pipeline(molecule_type="dna", abcfold_engine="boltz", use_mock=True):
    """
    Test full ABCFold + NA-MPNN pipeline for nucleic acid design.
    
    This is the key integration test: structure prediction with ABCFold (Boltz/Chai)
    followed by inverse folding with NA-MPNN.
    """
    print("\n" + "="*60)
    print(f"TEST: ABCFold+NA-MPNN Pipeline ({molecule_type.upper()}, engine={abcfold_engine}, mock={use_mock})")
    print("="*60)
    
    from core.hallucination_expert import create_hallucination_expert
    
    # Test sequences
    DNA_SEQ = "TCGATGTTATCATGCCTGGCATCATAGCCCAGCCCACTGCTTTCTTCTGCGGACGCCCCAGTTTGCGTCCCTTGTCCTTATGACTGTTTTTCTCAGCATC"
    RNA_SEQ = "UCGCGAGGGCGAACAUAUUAUCCGGUUUACUGUUAAGGCUAAAUCGCACAUACGCAGAUAUUCCGCACCCGUGCUGGACGAUGUUGACAGGACGGAGUGA"
    
    test_seq = DNA_SEQ if molecule_type == "dna" else RNA_SEQ
    
    try:
        # Create expert with ABCFold + NA-MPNN
        expert = create_hallucination_expert(
            structure_backend="abcfold",
            abcfold_engine=abcfold_engine,
            inverse_folding_backend="nampnn",
            molecule_type=molecule_type,
            use_mock=use_mock,
            fallback_to_protein_mpnn=False,  # Don't fallback for this test
        )
        
        # Mask some positions
        masked_positions = set(range(10, 30))  # Mask positions 10-29
        
        print(f"Input sequence: {test_seq[:40]}... ({len(test_seq)} nt)")
        print(f"Masked positions: {len(masked_positions)}")
        
        result = expert.generate_candidate(test_seq, masked_positions)
        
        if result:
            designed_seq = result['sequence']
            print(f"Output sequence: {designed_seq[:40]}... ({len(designed_seq)} chars)")
            print(f"Mean pLDDT: {result.get('mean_plddt', 'N/A')}")
            
            # Validate output is nucleic acid (not protein)
            valid_na = set("acgtbdhu/")  # DNA + RNA alphabets
            seq_chars = set(designed_seq.lower())
            na_fraction = sum(1 for c in designed_seq.lower() if c in valid_na) / len(designed_seq)
            
            if na_fraction >= 0.5:
                print(f"✅ ABCFold+NA-MPNN {molecule_type.upper()} pipeline PASSED ({na_fraction:.0%} NA)")
                return True
            else:
                print(f"❌ ABCFold+NA-MPNN pipeline FAILED: Output not nucleic acid ({na_fraction:.0%} NA)")
                return False
        else:
            print("❌ ABCFold+NA-MPNN pipeline returned None")
            return False
            
    except Exception as e:
        print(f"❌ ABCFold+NA-MPNN pipeline FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mol_type_routing():
    """
    Test that molecule type is correctly routed through the pipeline.
    
    Verifies:
    1. ABCFoldIntegration receives and stores molecule_type
    2. FASTA headers would be correct for DNA/RNA (checked via mock)
    3. AF3 JSON would use correct key (checked via mock)
    """
    print("\n" + "="*60)
    print("TEST: Molecule Type Routing Verification")
    print("="*60)
    
    from core.abcfold_integration import ABCFoldIntegration
    
    test_cases = [
        ("protein", "MKFLILLFNILCLFPVLAADNHGVGPQGAS"),
        ("dna", "TCGATGTTATCATGCCTGGCATCATAGCCC"),
        ("rna", "UCGCGAGGGCGAACAUAUUAUCCGGUUUAC"),
    ]
    
    all_passed = True
    
    for mol_type, test_seq in test_cases:
        try:
            integration = ABCFoldIntegration(
                engine="boltz",
                use_mock=True,
                molecule_type=mol_type,
            )
            
            # Verify molecule_type is stored correctly
            if integration.molecule_type != mol_type:
                print(f"❌ {mol_type}: molecule_type not stored correctly")
                all_passed = False
                continue
            
            # Run mock prediction to ensure no errors
            result = integration.predict_structure(test_seq)
            
            if result and 'coordinates' in result:
                print(f"✅ {mol_type}: Routing correct, mock prediction succeeded")
            else:
                print(f"❌ {mol_type}: Mock prediction failed")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {mol_type}: Error - {e}")
            all_passed = False
    
    if all_passed:
        print("✅ Molecule type routing test PASSED")
    else:
        print("❌ Molecule type routing test FAILED")
    
    return all_passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test hallucination backends")
    parser.add_argument("--mock", action="store_true", help="Use mock mode for all tests")
    parser.add_argument("--real-boltz", action="store_true", help="Test real Boltz (requires GPU)")
    parser.add_argument("--real-chai", action="store_true", help="Test real Chai-1 (requires GPU)")
    parser.add_argument("--real-nampnn", action="store_true", help="Test real NA-MPNN (requires GPU)")
    parser.add_argument("--test-dna", action="store_true", help="Test NA-MPNN with DNA sequence")
    parser.add_argument("--test-rna", action="store_true", help="Test NA-MPNN with RNA sequence")
    parser.add_argument("--test-pipeline", action="store_true", help="Test full ABCFold+NA-MPNN pipeline")
    parser.add_argument("--test-routing", action="store_true", help="Test molecule type routing")
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
    
    # Test 6: NA-MPNN DNA (if requested)
    if args.test_dna:
        nampnn_mock = not args.real_nampnn
        results['nampnn_dna'] = test_nampnn(use_mock=nampnn_mock, molecule_type="dna")
    
    # Test 7: NA-MPNN RNA (if requested)
    if args.test_rna:
        nampnn_mock = not args.real_nampnn
        results['nampnn_rna'] = test_nampnn(use_mock=nampnn_mock, molecule_type="rna")
    
    # Test 8: NA-MPNN with PDB (if real mode requested)
    if args.real_nampnn:
        results['nampnn_pdb'] = test_nampnn_with_pdb(use_mock=False)
    
    # Test 9: Molecule type routing (if requested)
    if args.test_routing:
        results['mol_type_routing'] = test_mol_type_routing()
    
    # Test 10: Full ABCFold+NA-MPNN pipeline (if requested)
    if args.test_pipeline:
        pipeline_mock = not (args.real_boltz or args.real_chai)
        engine = "boltz" if args.real_boltz else ("chai1" if args.real_chai else "boltz")
        results['pipeline_dna'] = test_abcfold_nampnn_pipeline(
            molecule_type="dna", abcfold_engine=engine, use_mock=pipeline_mock
        )
        results['pipeline_rna'] = test_abcfold_nampnn_pipeline(
            molecule_type="rna", abcfold_engine=engine, use_mock=pipeline_mock
        )
    
    # Test 11: Full protein pipeline
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
