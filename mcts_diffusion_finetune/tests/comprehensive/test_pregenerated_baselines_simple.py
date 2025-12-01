#!/usr/bin/env python3
"""
Simple test of pregenerated baselines for MCTS ablation studies
This test validates that we can load pregenerated baselines and use them for fair comparison
"""

import os, sys, json, time
from datetime import datetime

# Project path bootstrap
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bio import SeqIO
from utils.cameo_data_loader import CAMEODataLoader

def calculate_simple_aar(pred_seq, ref_seq):
    L = min(len(pred_seq), len(ref_seq))
    if L == 0:
        return 0.0
    return sum(p==r for p,r in zip(pred_seq[:L], ref_seq[:L]))/L

def load_correct_reference_sequences():
    reference_fasta = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    seqs = {}
    if os.path.exists(reference_fasta):
        for rec in SeqIO.parse(reference_fasta, "fasta"):
            seqs[rec.id] = str(rec.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(seqs)} reference sequences")
    else:
        print(f"⚠️ Reference FASTA not found: {reference_fasta}")
    return seqs

def load_pregenerated_baseline(structure_name):
    """Load pregenerated DPLM-2 150M baseline sequence"""
    print(f"  🎯 Loading FIXED pregenerated baseline for {structure_name}")
    
    # Try CAMEO pregenerated baselines
    fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_name}.fasta"
    
    try:
        def clean_aa(seq_str: str) -> str:
            valid = set("ACDEFGHIKLMNPQRSTVWY")
            s = "".join(c for c in str(seq_str).upper() if c in valid)
            return s
        
        if os.path.exists(fallback_path):
            for record in SeqIO.parse(fallback_path, "fasta"):
                baseline_seq = clean_aa(record.seq)
                print(f"  ✅ Loaded FIXED baseline from CAMEO pregenerated: {len(baseline_seq)} chars")
                return baseline_seq
        else:
            print(f"  ⚠️ CAMEO pregenerated file not found: {fallback_path}")
            
    except Exception as e:
        print(f"  ❌ Failed to load pregenerated baseline: {e}")
    
    return None

def test_pregenerated_baselines():
    """Test loading pregenerated baselines and computing AAR"""
    
    print("🧪 Testing Pregenerated Baseline Loading")
    print("=" * 50)
    
    # Load reference sequences
    refs = load_correct_reference_sequences()
    
    # Load CAMEO data
    loader = CAMEODataLoader()
    
    # Test first 5 structures
    results = []
    
    for idx in range(min(5, len(loader.structures))):
        structure_file = loader.structures[idx]
        base_name = structure_file.replace('.pkl', '')
        structure_name = f"CAMEO {base_name}"
        
        print(f"\n🧬 Testing {structure_name}")
        
        # Get reference sequence
        ref_id = base_name
        ref_seq = refs.get(ref_id)
        if not ref_seq:
            print(f"  ❌ No reference sequence for {ref_id}")
            continue
        
        # Load pregenerated baseline
        baseline_seq = load_pregenerated_baseline(ref_id)
        if not baseline_seq:
            print(f"  ❌ No pregenerated baseline for {ref_id}")
            continue
        
        # Compute AAR
        baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
        
        result = {
            "structure_name": structure_name,
            "ref_id": ref_id,
            "baseline_length": len(baseline_seq),
            "reference_length": len(ref_seq),
            "baseline_aar": baseline_aar,
            "baseline_sequence": baseline_seq[:50] + "..." if len(baseline_seq) > 50 else baseline_seq
        }
        
        results.append(result)
        
        print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")
        print(f"  📊 Lengths: baseline={len(baseline_seq)}, reference={len(ref_seq)}")
        
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 50)
    
    if results:
        avg_aar = sum(r['baseline_aar'] for r in results) / len(results)
        print(f"✅ Successfully tested {len(results)} pregenerated baselines")
        print(f"📈 Average baseline AAR: {avg_aar:.1%}")
        
        print(f"\n📋 Individual Results:")
        for r in results:
            print(f"  {r['ref_id']:<12} AAR: {r['baseline_aar']:.1%} ({r['baseline_length']} chars)")
        
        # Save results
        out_dir = "/tmp"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = os.path.join(out_dir, f"pregenerated_baseline_test_{ts}.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {out_json}")
        
        return True
    else:
        print("❌ No baselines could be loaded")
        return False

def main():
    success = test_pregenerated_baselines()
    
    if success:
        print(f"\n🎉 SUCCESS: Pregenerated baseline system is working!")
        print(f"🚀 Ready for fair MCTS ablation studies")
        print(f"")
        print(f"📝 Next steps:")
        print(f"  1. Use these pregenerated baselines as fixed starting points")
        print(f"  2. Run single expert mode (ProteinMPNN) with fixed baselines")
        print(f"  3. Run multi-expert mode with same fixed baselines")
        print(f"  4. Compare results for fair ablation study")
    else:
        print(f"\n❌ FAILED: Issues with pregenerated baseline loading")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
