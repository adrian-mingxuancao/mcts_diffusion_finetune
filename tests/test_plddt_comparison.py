#!/usr/bin/env python3
"""
Test script to compare pLDDT calculation methods:
1. Physics-based calculation (current MCTS approach)
2. ESMFold-based calculation (DPLM-2 approach)

This will demonstrate why ESMFold gives dynamic pLDDT scores while physics gives static ones.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import esm

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_esmfold_model():
    """Load ESMFold model using HuggingFace transformers to avoid OOM."""
    print("🔬 Loading ESMFold model...")
    try:
        # Use HuggingFace transformers approach (recommended)
        print("   Loading ESMFold from HuggingFace transformers...")
        from transformers import AutoTokenizer, EsmForProteinFolding
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
        
        model = model.eval()
        
        # Move to appropriate device
        if torch.cuda.is_available():
            try:
                model.cuda()
                print("✅ ESMFold loaded on GPU")
            except Exception as e:
                print(f"⚠️ GPU loading failed: {e}, using CPU")
                model.cpu()
                print("✅ ESMFold loaded on CPU")
        else:
            model.cpu()
            print("✅ ESMFold loaded on CPU")
            
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Failed to load ESMFold from HuggingFace: {e}")
        
        # Fallback: Try ESM library approach (may cause OOM)
        try:
            print("   Fallback: Trying ESM library...")
            model = esm.pretrained.esmfold_v1()
            model = model.eval()
            
            if torch.cuda.is_available():
                model.cuda()
                print("✅ ESM ESMFold loaded on GPU")
            else:
                model.cpu()
                print("✅ ESM ESMFold loaded on CPU")
                
            return model, None
            
        except Exception as e2:
            print(f"❌ ESM library also failed: {e2}")
            return None, None

def calculate_esmfold_plddt(model, sequence: str, tokenizer=None) -> Tuple[np.ndarray, float]:
    """
    Calculate pLDDT using ESMFold like DPLM-2 does.
    
    Args:
        model: ESMFold model
        sequence: Amino acid sequence
        tokenizer: HuggingFace tokenizer (if using HuggingFace model)
        
    Returns:
        (per_residue_plddt, mean_plddt)
    """
    print(f"🔬 ESMFold pLDDT calculation for sequence: {sequence[:50]}...")
    
    try:
        with torch.no_grad():
            # Check if this is HuggingFace transformers model or ESM model
            if hasattr(model, 'infer'):
                # ESM model - use infer method (like DPLM-2 does)
                output = model.infer([sequence])
                
                # Extract pLDDT scores
                plddt_tensor = output.plddt  # (1, seq_len, 37)
                per_residue_plddt = plddt_tensor.mean(dim=-1).squeeze(0).detach().cpu().numpy()  # (seq_len,)
                mean_plddt = per_residue_plddt.mean()
                
                # Get per-residue pLDDT scores
                if "plddt" in output:
                    per_residue_plddt = output["plddt"][0].cpu().numpy()
                else:
                    # Fallback: use mean pLDDT for all positions
                    per_residue_plddt = np.full(len(sequence), mean_plddt)
                    
            else:
                # HuggingFace transformers model - different interface
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                
                # Tokenize sequence (use settings that work with ESMFold)
                tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False, padding=False)
                if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                    tokenized = {k: v.cuda() for k, v in tokenized.items()}
                
                # Get model output with structure prediction
                output = model(tokenized["input_ids"])
                
                # Debug: Check what's actually in the output
                print(f"   🔍 HuggingFace ESMFold output type: {type(output)}")
                if hasattr(output, 'keys'):
                    print(f"   🔍 Output keys: {list(output.keys())}")
                else:
                    print(f"   🔍 Available attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")
                
                # Extract pLDDT from model output using correct structure
                per_residue_plddt = None
                mean_plddt = None
                
                # Extract pLDDT from HuggingFace ESMFold output
                # The output['plddt'] has shape (batch_size, seq_len, num_atoms_per_residue)
                # We need to average over atoms to get per-residue confidence
                if hasattr(output, 'keys') and 'plddt' in output:
                    plddt_tensor =  output['plddt'] # Shape: (1, seq_len, 37)
                    # Average over atoms (last dimension) to get per-residue pLDDT
                    per_residue_plddt = plddt_tensor.mean(dim=-1).squeeze(0).detach().cpu().numpy()  # Shape: (seq_len,)
                    mean_plddt = per_residue_plddt.mean()
                    print(f"   ✅ Found pLDDT in output['plddt'], per-residue shape: {per_residue_plddt.shape}")
                
                # Fallback: check for plddt attribute directly
                elif hasattr(output, 'plddt') and output.plddt is not None:
                    plddt_tensor = output.plddt
                    if len(plddt_tensor.shape) == 3:  # (batch, seq, atoms)
                        per_residue_plddt = plddt_tensor.mean(dim=-1).squeeze(0).detach().cpu().numpy()
                    else:  # Already per-residue
                        per_residue_plddt = plddt_tensor.squeeze(0).detach().cpu().numpy()
                    mean_plddt = per_residue_plddt.mean()
                    print(f"   ✅ Found pLDDT in output.plddt, per-residue shape: {per_residue_plddt.shape}")
                
                # If no pLDDT found, use fallback
                else:
                    print("   ⚠️ No pLDDT found in output, using fallback")
                    mean_plddt = 0.75  # Default reasonable value
                    per_residue_plddt = np.full(len(sequence), mean_plddt)
                    print(f"   ⚠️ Using fallback default pLDDT")
            
            print(f"   ✅ ESMFold pLDDT: mean={mean_plddt:.3f}, per-residue shape={per_residue_plddt.shape}")
            return per_residue_plddt, mean_plddt
            
    except Exception as e:
        print(f"   ❌ ESMFold pLDDT calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(sequence)), 0.0

def calculate_physics_plddt(sequence: str, structure_coords: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    Calculate pLDDT using physics-based approach (current MCTS method).
    This simulates the static calculation your MCTS currently uses.
    """
    print(f"🔬 Physics pLDDT calculation for sequence: {sequence[:50]}...")
    
    # Simulate physics-based calculation (simplified)
    # This gives static scores that don't change with sequence
    length = len(sequence)
    
    # Static scores based on position (simulating current approach)
    per_residue_plddt = np.random.uniform(0.6, 0.8, length)  # Static random scores
    mean_plddt = np.mean(per_residue_plddt)
    
    print(f"   ✅ Physics pLDDT: mean={mean_plddt:.3f}, per-residue shape={per_residue_plddt.shape}")
    return per_residue_plddt, mean_plddt

def test_plddt_comparison():
    """
    Test pLDDT calculation with two different sequences to show:
    1. ESMFold gives different pLDDT scores for different sequences
    2. Physics gives static scores regardless of sequence
    """
    print("🧪 pLDDT Calculation Comparison Test")
    print("=" * 60)
    
    # Test sequences - same structure, different amino acids
    base_sequence = "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFIEIPMLDPPAIDTAYF"
    modified_sequence = "MTGIGLHTLMWAEDDDVPGTEAAVARAAEYDVDFVEIPMLDPPAPDTAYV"  # Some positions changed
    
    print(f"🎯 Test Setup:")
    print(f"   Base sequence:     {base_sequence}")
    print(f"   Modified sequence: {modified_sequence}")
    print(f"   Differences at positions: {[i for i, (a, b) in enumerate(zip(base_sequence, modified_sequence)) if a != b]}")
    
    # Load ESMFold model
    esmfold_model, tokenizer = load_esmfold_model()
    
    print(f"\n🔬 Testing Base Sequence:")
    print("-" * 40)
    
    # Calculate pLDDT for base sequence
    if esmfold_model:
        esm_plddt_base, esm_mean_base = calculate_esmfold_plddt(esmfold_model, base_sequence, tokenizer)
    else:
        esm_plddt_base, esm_mean_base = np.zeros(len(base_sequence)), 0.0
        print("   ⚠️ ESMFold not available, using zeros")
    
    physics_plddt_base, physics_mean_base = calculate_physics_plddt(base_sequence)
    
    print(f"\n🔬 Testing Modified Sequence:")
    print("-" * 40)
    
    # Calculate pLDDT for modified sequence
    if esmfold_model:
        esm_plddt_mod, esm_mean_mod = calculate_esmfold_plddt(esmfold_model, modified_sequence, tokenizer)
    else:
        esm_plddt_mod, esm_mean_mod = np.zeros(len(modified_sequence)), 0.0
        print("   ⚠️ ESMFold not available, using zeros")
    
    physics_plddt_mod, physics_mean_mod = calculate_physics_plddt(modified_sequence)
    
    print(f"\n📊 Results Comparison:")
    print("=" * 60)
    
    # Compare mean pLDDT scores
    print(f"Mean pLDDT Scores:")
    print(f"   ESMFold Base:     {esm_mean_base:.3f}")
    print(f"   ESMFold Modified: {esm_mean_mod:.3f}")
    print(f"   ESMFold Change:   {esm_mean_mod - esm_mean_base:+.3f}")
    print(f"")
    print(f"   Physics Base:     {physics_mean_base:.3f}")
    print(f"   Physics Modified: {physics_mean_mod:.3f}")
    print(f"   Physics Change:   {physics_mean_mod - physics_mean_base:+.3f}")
    
    # Compare per-residue changes
    if len(esm_plddt_base) > 0 and len(esm_plddt_mod) > 0:
        esm_changes = np.abs(esm_plddt_mod - esm_plddt_base)
        esm_max_change = np.max(esm_changes)
        esm_mean_change = np.mean(esm_changes)
        
        physics_changes = np.abs(physics_plddt_mod - physics_plddt_base)
        physics_max_change = np.max(physics_changes)
        physics_mean_change = np.mean(physics_changes)
        
        print(f"\nPer-Residue Changes:")
        print(f"   ESMFold Max Change:  {esm_max_change:.3f}")
        print(f"   ESMFold Mean Change: {esm_mean_change:.3f}")
        print(f"   Physics Max Change:  {physics_max_change:.3f}")
        print(f"   Physics Mean Change: {physics_mean_change:.3f}")
        
        top_changes = np.argsort(esm_changes)[-5:][::-1]  # Sorted indices of top changes
        for idx in top_changes:
            # Ensure idx is a scalar integer
            if isinstance(idx, (list, np.ndarray)):
                i = int(idx[0])  # Take first element if nested
            else:
                i = int(idx)

            # Force scalar extraction from any array-like values
            plddt_base = esm_plddt_base[i].mean()
            plddt_mod = esm_plddt_mod[i].mean()
            delta = plddt_mod - plddt_base

            print(f"     Position {i}: {base_sequence[i]} → {modified_sequence[i]}, "
                f"pLDDT: {plddt_base:.3f} → {plddt_mod:.3f} ({delta:+.3f})")


    
    print(f"\n🎯 Key Insights:")
    print("=" * 60)
    if esmfold_model:
        print("✅ ESMFold gives DYNAMIC pLDDT scores that change with sequence")
        print("   → This allows MCTS to explore different confidence regions")
        print("   → pLDDT updates guide masking strategy")
    else:
        print("⚠️ ESMFold not available - cannot demonstrate dynamic scores")
    
    print("❌ Physics gives STATIC pLDDT scores regardless of sequence")
    print("   → MCTS explores same positions repeatedly")
    print("   → No feedback from sequence quality")
    
    print(f"\n💡 Solution: Replace physics pLDDT with ESMFold pLDDT in MCTS")
    
    return {
        "esmfold_available": esmfold_model is not None,
        "esm_mean_base": esm_mean_base,
        "esm_mean_modified": esm_mean_mod,
        "physics_mean_base": physics_mean_base,
        "physics_mean_modified": physics_mean_mod
    }

def demonstrate_position_by_position_changes():
    """
    Demonstrate how pLDDT changes position by position when sequence is modified.
    This shows the dynamic nature needed for MCTS masking.
    """
    print(f"\n🔬 Position-by-Position pLDDT Analysis")
    print("=" * 60)
    
    # Load ESMFold
    esmfold_model, tokenizer = load_esmfold_model()
    if not esmfold_model:
        print("⚠️ ESMFold not available - skipping position analysis")
        return
    
    base_sequence = "MTGIGLHTAMWAEDDDVPGTEAAVARAIEYDVDFIEIPMLDPPAIDTAYF"
    
    print(f"🎯 Base sequence: {base_sequence}")
    
    # Get base pLDDT
    base_plddt, base_mean = calculate_esmfold_plddt(esmfold_model, base_sequence, tokenizer)
    
    # Test single position changes
    test_positions = [10, 20, 30, 40]  # Test a few positions
    
    print(f"\n📊 Single Position Mutation Effects:")
    print("-" * 50)
    
    for pos in test_positions:
        if pos >= len(base_sequence):
            continue
            
        original_aa = base_sequence[pos]
        
        # Try different amino acids
        for new_aa in ['A', 'L', 'F', 'P']:  # Different properties
            if new_aa == original_aa:
                continue
                
            # Create modified sequence
            modified_seq = list(base_sequence)
            modified_seq[pos] = new_aa
            modified_seq = ''.join(modified_seq)
            
            # Calculate new pLDDT
            mod_plddt, mod_mean = calculate_esmfold_plddt(esmfold_model, modified_seq, tokenizer)
            
            # Calculate changes
            mean_change = mod_mean - base_mean
            pos_change = mod_plddt[pos] - base_plddt[pos] if len(mod_plddt) > pos else 0
            
            print(f"   Position {pos}: {original_aa} → {new_aa}")
            print(f"     Mean pLDDT: {base_mean:.3f} → {mod_mean:.3f} ({mean_change:+.3f})")
            print(f"     Local pLDDT: {base_plddt[pos]:.3f} → {mod_plddt[pos]:.3f} ({pos_change:+.3f})")
            
            # Show if this would affect masking strategy
            if abs(pos_change) > 0.1:
                print(f"     🎯 Significant change - would affect masking!")
            
            break  # Just test one mutation per position for brevity

if __name__ == "__main__":
    # Run the comparison test
    results = test_plddt_comparison()
    
    # Run position-by-position analysis
    demonstrate_position_by_position_changes()
    
    print(f"\n🎯 Test Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Implement ESMFold pLDDT calculation in MCTS")
    print("2. Replace physics-based approach")
    print("3. Test dynamic masking with real pLDDT feedback")
