#!/usr/bin/env python3
"""
MCTS Ablation Runner

Runs two ablations:
  1) random_no_expert  — MCTS expands with random fills only (no DPLM-2 rollouts)
  2) single_expert_k   — three separate studies with exactly one expert (k in {0,1,2}),
                         each spawning 3 children per expansion from that one expert.

Everything else (masking schedule, reward, logging, saving) mirrors the original script.
"""

import os, sys, json, time
from datetime import datetime

# --- project path bootstrap identical to your original ---
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# ---- your loader fallback kept verbatim ----
try:
    from utils.cameo_data_loader import CAMEODataLoader
except ImportError:
    class CAMEODataLoader:
        def __init__(self, *args, **kwargs):
            self.structures = []
        def get_test_structure(self, index=0):
            return {
                "name": f"test_structure_{index}",
                "struct_seq": "159,162,163,164,165",
                "sequence": "IKKSI",
                "length": 5
            }

from core.dplm2_integration_fixed_new import DPLM2IntegrationCorrected as DPLM2Integration
from core.sequence_level_mcts import GeneralMCTS
from Bio import SeqIO

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def generate_dplm2_baseline_sequence(structure, dplm2):
    # same as your original: we just rely on your integration's baseline method
    target_length = structure['length']
    baseline_structure = dict(structure)
    struct_seq = baseline_structure.get('struct_seq')
    struct_ids = baseline_structure.get('struct_ids')
    
    # Check if struct tokens are missing (handle numpy arrays properly)
    has_struct_seq = struct_seq is not None and (
        (isinstance(struct_seq, str) and len(struct_seq) > 0) or
        (hasattr(struct_seq, '__len__') and len(struct_seq) > 0)
    )
    has_struct_ids = struct_ids is not None and (
        (isinstance(struct_ids, str) and len(struct_ids) > 0) or
        (hasattr(struct_ids, '__len__') and len(struct_ids) > 0)
    )
    
    if not has_struct_seq and not has_struct_ids:
        struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
        pdb_id = structure.get('pdb_id', '')
        chain_id = structure.get('chain_id', '')
        structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
        from utils.struct_loader import load_struct_seq_from_fasta
        struct_seq = load_struct_seq_from_fasta(struct_fasta_path, structure_name)
        baseline_structure['struct_seq'] = struct_seq
    # Use fallback to pre-generated results directly (skip generation)
    pdb_id = structure.get('pdb_id', '')
    chain_id = structure.get('chain_id', '')
    structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
    
    fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_name}.fasta"
    try:
        from Bio import SeqIO
        
        def clean_aa(seq_str: str) -> str:
            valid = set("ACDEFGHIKLMNPQRSTVWY")
            s = "".join(c for c in str(seq_str).upper() if c in valid)  # drop spaces & non-AA
            return s
        
        for record in SeqIO.parse(fallback_path, "fasta"):
            baseline_seq = clean_aa(record.seq)
            print(f"  ✅ Loaded pregenerated 150M baseline: {len(baseline_seq)} chars")
            return baseline_seq, baseline_structure
            
    except Exception as fallback_e:
        print(f"  ❌ Pregenerated baseline failed: {fallback_e}")
        return None, None

def run_one_structure(structure, structure_name, dplm2, correct_reference_sequences, ablation_mode, single_expert_id=None):
    print(f"\n🧬 [{ablation_mode}{'' if single_expert_id is None else f'_{single_expert_id}'}] {structure_name}")
    ref_id = structure_name.replace('CAMEO ', '')
    ref_seq = correct_reference_sequences.get(ref_id)
    if not ref_seq:
        print(f"  ❌ no reference: {ref_id}")
        return None

    # baseline
    baseline_seq, baseline_struct = generate_dplm2_baseline_sequence(structure, dplm2)
    if not baseline_seq:
        print("  ❌ baseline generation failed")
        return None
    baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
    print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")

    # wire MCTS with ablation knobs
    kwargs = dict(
        dplm2_integration=dplm2,
        initial_sequence=baseline_seq,
        baseline_structure=baseline_struct,
        reference_sequence=ref_seq,
        max_depth=4,
    )

    if ablation_mode == "random_no_expert":
        kwargs.update(dict(ablation_mode="random_no_expert",
                           num_candidates_per_expansion=6,
                           num_children_select=3))  # a tad more breadth to be fair
    elif ablation_mode == "single_expert":
        kwargs.update(dict(ablation_mode="single_expert",
                           single_expert_id=int(single_expert_id or 0),
                           k_rollouts_per_expert=5,   # give the single expert a few shots
                           num_children_select=3))    # you asked for 3 children directly
    else:
        kwargs.update(dict(ablation_mode="multi_expert",
                           k_rollouts_per_expert=3,
                           num_children_select=2))

    mcts = GeneralMCTS(**kwargs)
    t0 = time.time()
    root = mcts.search(num_iterations=30)
    elapsed = time.time() - t0

    # find best
    def best(node):
        best_node, best_score = node, getattr(node, "reward", 0.0)
        for c in node.children:
            b = best(c)
            score = getattr(b, "reward", 0.0)
            if score > best_score:
                best_node, best_score = b, score
        return best_node
    best_node = best(root)
    best_seq = best_node.sequence
    best_aar = calculate_simple_aar(best_seq, ref_seq)

    out = {
        "structure_name": structure_name,
        "length": structure["length"],
        "mode": ablation_mode if single_expert_id is None else f"{ablation_mode}_{single_expert_id}",
        "baseline_aar": baseline_aar,
        "final_aar": best_aar,
        "aar_improvement": best_aar - baseline_aar,
        "final_reward": getattr(best_node, "reward", 0.0),
        "search_time": elapsed,
        "sequence": best_seq
    }
    print(f"  📊 {ablation_mode}{'' if single_expert_id is None else f'[{single_expert_id}]'}: "
          f"{baseline_aar:.1%} → {best_aar:.1%} (Δ {best_aar - baseline_aar:+.1%}) in {elapsed:.1f}s")
    return out

def main():
    print("🧬 MCTS Ablation Study - CAMEO 2022 Evaluation")
    print("=" * 60)
    
    # Check for structure range arguments
    import sys
    start_idx = 0
    end_idx = None
    
    if len(sys.argv) >= 3:
        start_idx = int(sys.argv[1])
        end_idx = int(sys.argv[2])
        print(f"🎯 Processing structures {start_idx}-{end_idx}")
    elif len(sys.argv) == 2:
        start_idx = int(sys.argv[1])
        print(f"🎯 Processing structures from {start_idx} onwards")
    else:
        print("🎯 Processing all structures")
    
    # Load data using same pattern as test_mcts_with_real_data.py
    loader = CAMEODataLoader()
    refs = load_correct_reference_sequences()
    
    # Initialize DPLM-2
    dplm2 = DPLM2Integration()
    
    results = []
    
    # Load structures using the working pattern
    all_structures = []
    for idx, structure_file in enumerate(loader.structures):
        structure = loader.get_structure_by_index(idx)
        if structure:
            all_structures.append((idx, structure))
    
    # Select structure range
    if end_idx is not None:
        test_structures = all_structures[start_idx:end_idx]
    else:
        test_structures = all_structures[start_idx:]
    
    print(f"📊 Selected {len(test_structures)} structures to process")
    
    for idx, structure in test_structures:
        name = f"CAMEO {structure.get('pdb_id','test')}_{structure.get('chain_id','A')}"
        
        # Check if this is for multiple experts (structures 11-17) or ablation study (0-10)
        if start_idx >= 11:
            # For structures 11-17: Run ONLY multiple experts mode
            print(f"🎯 Structure {idx}: Running MULTIPLE EXPERTS mode only")
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert")
            if r: results.append(r)
        else:
            # For structures 0-10: Run ablation study modes
            print(f"🎯 Structure {idx}: Running ABLATION STUDY modes")
            # 1) random no-expert
            r1 = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert")
            if r1: results.append(r1)

            # 2) single expert ids 0,1,2
            for eid in [0,1,2]:
                r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid)
                if r: results.append(r)

    # save
    out_dir = "/net/scratch/caom/cameo_evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"mcts_ablation_results_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved ablation results → {out_json}")

if __name__ == "__main__":
    main()
