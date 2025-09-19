#!/usr/bin/env python3
"""
MCTS Ablation Runner for PDB_date Dataset

Runs ablation studies on PDB_date dataset:
  1) random_no_expert  — MCTS expands with random fills only (no DPLM-2 rollouts)
  2) single_expert_k   — three separate studies with exactly one expert (k in {0,1,2}),
                         each spawning 3 children per expansion from that one expert.
  3) multi_expert      — uses all experts for comprehensive search

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

# ---- PDB loader import ----
try:
    from utils.pdb_data_loader import PDBDataLoader
except ImportError:
    class PDBDataLoader:
        def __init__(self, *args, **kwargs):
            self.structures = []
        def get_test_structure(self, index=0):
            return {
                "name": f"test_structure_{index}",
                "struct_seq": "159,162,163,164,165",
                "sequence": "IKKSI",
                "length": 5
            }

from core.dplm2_integration_memory_optimized import MemoryOptimizedDPLM2Integration as CleanDPLM2Integration
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
    reference_fasta = "/home/caom/AID3/dplm/data-bin/PDB_date/aatype.fasta"
    seqs = {}
    if os.path.exists(reference_fasta):
        for rec in SeqIO.parse(reference_fasta, "fasta"):
            seqs[rec.id] = str(rec.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(seqs)} reference sequences")
    else:
        print(f"⚠️ Reference FASTA not found: {reference_fasta}")
    return seqs

def generate_dplm2_baseline_sequence(structure, dplm2, structure_idx=None):
    # same as your original: we just rely on your integration's baseline method
    target_length = structure['length']
    baseline_structure = dict(structure)
    # Add structure_idx for scTM calculation
    if structure_idx is not None:
        baseline_structure['structure_idx'] = structure_idx
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
        struct_fasta_path = "/home/caom/AID3/dplm/data-bin/PDB_date/struct.fasta"
        pdb_id = structure.get('pdb_id', '')
        chain_id = structure.get('chain_id', '')
        structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('PDB ', '')
        from utils.struct_loader import load_struct_seq_from_fasta
        struct_seq = load_struct_seq_from_fasta(struct_fasta_path, structure_name)
        baseline_structure['struct_seq'] = struct_seq
    # Try inverse folding generation first, fallback to pregenerated if it fails
    pdb_id = structure.get('pdb_id', '')
    chain_id = structure.get('chain_id', '')
    structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('PDB ', '')
    
    # First attempt: Try direct inverse folding generation
    try:
        print(f"  🔄 Attempting direct inverse folding generation...")
        # Use the corrected DPLM2 integration - use 150M model (expert_id=1) for baseline
        baseline_seq = dplm2.generate_with_expert(expert_id=1, structure=baseline_structure, target_length=target_length)
        if baseline_seq and len(baseline_seq) > 0:
            print(f"  ✅ Generated baseline sequence: {len(baseline_seq)} chars")
            return baseline_seq, baseline_structure
    except Exception as gen_e:
        print(f"  ⚠️ Direct generation failed: {gen_e}")
    
    # Fallback: Use pregenerated sequences
    fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_name}.fasta"
    try:
        from Bio import SeqIO
        
        def clean_aa(seq_str: str) -> str:
            valid = set("ACDEFGHIKLMNPQRSTVWY")
            s = "".join(c for c in str(seq_str).upper() if c in valid)  # drop spaces & non-AA
            return s
        
        for record in SeqIO.parse(fallback_path, "fasta"):
            baseline_seq = clean_aa(record.seq)
            print(f"  ✅ Fallback: Loaded pregenerated 150M baseline: {len(baseline_seq)} chars")
            return baseline_seq, baseline_structure
            
    except Exception as fallback_e:
        print(f"  ❌ Both generation and pregenerated baseline failed: {fallback_e}")
        return None, None

def run_one_structure(structure, structure_name, dplm2, correct_reference_sequences, ablation_mode, single_expert_id=None, structure_idx=None, loader=None):
    print(f"\n🧬 [{ablation_mode}{'' if single_expert_id is None else f'_{single_expert_id}'}] {structure_name}")
    ref_id = structure_name.replace('PDB ', '')
    ref_seq = correct_reference_sequences.get(ref_id)
    if not ref_seq:
        print(f"  ❌ no reference: {ref_id}")
        return None

    # baseline
    baseline_seq, baseline_struct = generate_dplm2_baseline_sequence(structure, dplm2, structure_idx)
    if not baseline_seq:
        print("  ❌ baseline generation failed")
        return None
    baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
    print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")

    # wire MCTS with ablation knobs (tuned for faster runs)
    kwargs = dict(
        dplm2_integration=dplm2,
        baseline_structure=baseline_struct,
        reference_sequence=ref_seq,
        max_depth=4,
        backup_rule="max"  # Use max backup for pure MCTS
    )

    if ablation_mode == "random_no_expert":
        kwargs.update(dict(ablation_mode="random_no_expert",
                           num_children_select=4))  # a tad more breadth to be fair
    elif ablation_mode == "single_expert":
        kwargs.update(dict(ablation_mode="single_expert",
                           single_expert_id=int(single_expert_id or 0),
                           k_rollouts_per_expert=2,   # match 2-node growth
                           num_children_select=2))    # exactly two children per expansion
    else:
        kwargs.update(dict(ablation_mode="multi_expert",
                           k_rollouts_per_expert=2,   # faster
                           num_children_select=2))

    mcts = GeneralMCTS(**kwargs)

    # Baseline reward using pregenerated sequence and current reward config
    try:
        baseline_reward = mcts._compute_reward(baseline_seq)
        print(f"  📈 Baseline reward: {baseline_reward:.3f}")
    except Exception as e:
        print(f"  ⚠️ Baseline reward computation failed: {e}")
        baseline_reward = 0.0
    t0 = time.time()
    root = mcts.search(initial_sequence=baseline_seq, num_iterations=25)
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

    # Compute scTM scores using ESMFold prediction vs reference structure
    baseline_sctm, final_sctm = None, None
    try:
        # Use PDB-specific scTM calculation function (defined above)
        
        # Load reference coordinates from .pkl file for scTM calculation
        # Use the same loader instance that was passed in
        pdb_structure = loader.get_structure_by_index(structure_idx) if structure_idx is not None else None
        if pdb_structure:
            print(f"  🧬 Loaded PDB structure for scTM calculation")
            print(f"  🔍 Structure keys: {list(pdb_structure.keys())}")
            
            # Get reference coordinates (CA atoms)
            reference_coords = None
            
            # Check backbone_coords first (most reliable)
            if 'backbone_coords' in pdb_structure and pdb_structure['backbone_coords'] is not None:
                coords = pdb_structure['backbone_coords']
                if len(coords.shape) == 3 and coords.shape[1] == 3:
                    reference_coords = coords[:, 1, :]  # CA atoms at index 1
                else:
                    reference_coords = coords
                print(f"  🧬 Using backbone_coords: {reference_coords.shape}")
            
            # Check coordinates as fallback
            elif 'coordinates' in pdb_structure and pdb_structure['coordinates'] is not None:
                reference_coords = pdb_structure['coordinates']
                print(f"  🧬 Using coordinates: {reference_coords.shape}")
            
            # Check atom_positions as last resort
            elif 'atom_positions' in pdb_structure and pdb_structure['atom_positions'] is not None:
                coords = pdb_structure['atom_positions']
                if len(coords.shape) == 3 and coords.shape[1] >= 2:
                    reference_coords = coords[:, 1, :]  # CA atoms at index 1
                else:
                    reference_coords = coords
                print(f"  🧬 Using atom_positions: {reference_coords.shape}")
            
            if reference_coords is not None and hasattr(reference_coords, 'shape'):
                print(f"  🧬 Using reference coordinates for scTM calculation: {reference_coords.shape}")
                # Calculate scTM: ESMFold prediction vs reference using centralized function
                from utils.sctm_calculation import calculate_sctm_score
                baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
                final_sctm = calculate_sctm_score(best_seq, reference_coords)
            else:
                print(f"  ⚠️ No valid reference coordinates found in PDB .pkl file")
                print(f"  🔍 reference_coords type: {type(reference_coords)}")
                if reference_coords is not None:
                    print(f"  🔍 reference_coords has shape: {hasattr(reference_coords, 'shape')}")
        else:
            print(f"  ⚠️ Could not load PDB .pkl file for scTM calculation")
    except Exception as e:
        print(f"  ⚠️ scTM calculation failed: {e}")
        import traceback
        traceback.print_exc()
        baseline_sctm, final_sctm = None, None

    out = {
        "structure_name": structure_name,
        "length": structure["length"],
        "mode": ablation_mode if single_expert_id is None else f"{ablation_mode}_{single_expert_id}",
        "baseline_aar": baseline_aar,
        "final_aar": best_aar,
        "aar_improvement": best_aar - baseline_aar,
        "baseline_reward": baseline_reward,
        "final_reward": getattr(best_node, "reward", 0.0),
        "reward_improvement": getattr(best_node, "reward", 0.0) - baseline_reward,
        "baseline_sctm": baseline_sctm if baseline_sctm is not None else 0.0,
        "final_sctm": final_sctm if final_sctm is not None else 0.0,
        "sctm_improvement": (final_sctm - baseline_sctm) if (baseline_sctm is not None and final_sctm is not None) else 0.0,
        "baseline_sequence": baseline_seq,
        "final_sequence": best_seq,
        "search_time": elapsed,
        "mcts_success": True
    }
    # Small per-structure summary (AAR, scTM, Reward)
    print("  ─────────────────────────────────────────────────────────")
    print(f"  Summary [{ablation_mode}{'' if single_expert_id is None else f'/{single_expert_id}'}] {structure_name}")
    print(f"    AAR     : {baseline_aar:.1%} → {best_aar:.1%} (Δ {best_aar - baseline_aar:+.1%})")
    if baseline_sctm is not None and final_sctm is not None:
        print(f"    scTM    : {baseline_sctm:.3f} → {final_sctm:.3f} (Δ {final_sctm - baseline_sctm:+.3f})")
    else:
        print("    scTM    : N/A (no reference coords)")
    print(f"    Reward  : {baseline_reward:.3f} → {getattr(best_node, 'reward', 0.0):.3f} (Δ {getattr(best_node, 'reward', 0.0) - baseline_reward:+.3f})")
    print(f"    Time    : {elapsed:.1f}s")
    print("  ─────────────────────────────────────────────────────────")
    
    # Print incremental summary for real-time tracking (parseable format)
    print(f"  📈 Summary: {structure_name},{structure['length']},{baseline_aar:.3f},{best_aar:.3f},"
          f"{best_aar - baseline_aar:+.3f},{baseline_reward:.3f},{getattr(best_node, 'reward', 0.0):.3f},"
          f"{getattr(best_node, 'reward', 0.0) - baseline_reward:+.3f},"
          f"{baseline_sctm if baseline_sctm is not None else 0.0:.3f},"
          f"{final_sctm if final_sctm is not None else 0.0:.3f},"
          f"{(final_sctm - baseline_sctm) if (baseline_sctm is not None and final_sctm is not None) else 0.0:+.3f},"
          f"True,{elapsed:.1f}")
    
    return out

def main():
    print("🧬 MCTS Ablation Study - PDB_date Dataset Evaluation")
    print("=" * 60)
    
    # CLI: structure range + method switching
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs='?', type=int, default=0, help="start index (inclusive)")
    parser.add_argument("end", nargs='?', type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--mode", choices=["random_no_expert","single_expert","multi_expert","all"], default="all")
    parser.add_argument("--single_expert_id", type=int, default=None, help="single expert id (0/1/2)")
    args = parser.parse_args()

    start_idx = args.start
    end_idx = args.end
    print(f"🎯 Structure range: {start_idx}-{end_idx if end_idx is not None else 'end'} | Mode: {args.mode}")
    
    # Load data using PDB loader
    loader = PDBDataLoader()
    refs = load_correct_reference_sequences()
    
    # Initialize DPLM-2
    dplm2 = CleanDPLM2Integration(model_name="airkingbd/dplm2_150m")  # Use 150M for single expert testing
    
    results = []
    
    # Load structures using the PDB loader approach
    all_structures = []
    struct_fasta_path = "/home/caom/AID3/dplm/data-bin/PDB_date/struct.fasta"
    
    # Load structure sequences directly from struct.fasta
    struct_records = {}
    if os.path.exists(struct_fasta_path):
        from Bio import SeqIO
        for record in SeqIO.parse(struct_fasta_path, "fasta"):
            struct_records[record.id] = str(record.seq)
        print(f"✅ Loaded {len(struct_records)} structure sequences from struct.fasta")
    else:
        print(f"❌ struct.fasta not found: {struct_fasta_path}")
        return
    
    # Create structures using the PDB loader format - only include structures with both sequence and structure data
    valid_structures = []
    for idx, structure_file in enumerate(loader.structures):
        # Get the base name (e.g., "8A00" from "a0/8A00.pkl")
        base_name = os.path.splitext(os.path.basename(structure_file))[0]
        
        # Check if we have both struct sequence and reference sequence
        if base_name in struct_records and base_name in refs:
            # Create structure dict in the PDB format
            struct_seq = struct_records[base_name]
            structure = {
                'struct_seq': struct_seq,
                'length': len(struct_seq.split(',')),
                'pdb_id': base_name,
                'chain_id': 'A',  # Default chain ID for PDB
                'name': f"PDB {base_name}"
            }
            valid_structures.append((idx, structure))
            if len(valid_structures) <= 10:  # Only print first 10 for brevity
                print(f"✅ Created structure {len(valid_structures)-1}: {base_name} (length: {structure['length']})")
        else:
            if idx < 10:  # Only print first 10 for brevity
                missing = []
                if base_name not in struct_records:
                    missing.append("struct_seq")
                if base_name not in refs:
                    missing.append("ref_seq")
                print(f"⚠️ Missing {', '.join(missing)} for {base_name}")
    
    all_structures = valid_structures
    print(f"📊 Found {len(all_structures)} valid structures with both sequence and structure data")
    
    # Select structure range
    if end_idx is not None:
        test_structures = all_structures[start_idx:end_idx]
    else:
        test_structures = all_structures[start_idx:]
    
    print(f"📊 Selected {len(test_structures)} structures to process")
    
    # Method-by-method switching
    for idx, structure in test_structures:
        name = f"PDB {structure.get('pdb_id','test')}"

        if args.mode == "random_no_expert":
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert", structure_idx=idx, loader=loader)
            if r: results.append(r)
            continue

        if args.mode == "single_expert":
            if args.single_expert_id is None:
                ids = [0,1,2]
            else:
                ids = [int(args.single_expert_id)]
            for eid in ids:
                r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid, structure_idx=idx, loader=loader)
                if r: results.append(r)
            continue

        if args.mode == "multi_expert":
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert", structure_idx=idx, loader=loader)
            if r: results.append(r)
            continue

        # args.mode == "all": run all variants
        r = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert", structure_idx=idx, loader=loader)
        if r: results.append(r)
        for eid in [0,1,2]:
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid, structure_idx=idx, loader=loader)
            if r: results.append(r)
        r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert", structure_idx=idx, loader=loader)
        if r: results.append(r)

    # save grouped-by-mode summary for easier analysis
    out_dir = "/net/scratch/caom/pdb_evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"mcts_ablation_results_pdb_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved ablation results → {out_json}")

    # --- Build per-mode summary table similar to tree_search summary ---
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["mode"]].append(r)

    summary_path = os.path.join(out_dir, f"mcts_ablation_summary_pdb_{ts}.txt")
    try:
        with open(summary_path, "w") as sf:
            sf.write("MCTS Ablation Summary - PDB_date Dataset (grouped by mode)\n")
            sf.write("="*80 + "\n\n")
            for mode, rows in grouped.items():
                sf.write(f"Mode: {mode}\n")
                sf.write("-"*80 + "\n")
                sf.write(f"{'Structure':<20} {'Len':<5} {'Base AAR':<10} {'Final AAR':<10} {'ΔAAR':<8} {'Base R':<8} {'Final R':<8} {'ΔR':<8} {'Base scTM':<9} {'Final scTM':<10} {'ΔscTM':<8} {'Time(s)':<8}\n")
                for r in rows:
                    name = r['structure_name'].replace('PDB ', '')[:19]
                    sf.write(f"{name:<20} {r['length']:<5} {r['baseline_aar']:<10.3f} {r['final_aar']:<10.3f} "
                             f"{r['aar_improvement']:<8.3f} {r['baseline_reward']:<8.3f} {r['final_reward']:<8.3f} "
                             f"{r['reward_improvement']:<8.3f} {r['baseline_sctm']:<9.3f} {r['final_sctm']:<10.3f} "
                             f"{r['sctm_improvement']:<8.3f} {r['search_time']:<8.1f}\n")
                # stats
                if rows:
                    avg_base_aar = sum(x['baseline_aar'] for x in rows)/len(rows)
                    avg_final_aar = sum(x['final_aar'] for x in rows)/len(rows)
                    avg_base_reward = sum(x['baseline_reward'] for x in rows)/len(rows)
                    avg_final_reward = sum(x['final_reward'] for x in rows)/len(rows)
                    avg_base_sctm = sum(x['baseline_sctm'] for x in rows)/len(rows)
                    avg_final_sctm = sum(x['final_sctm'] for x in rows)/len(rows)
                    
                    sf.write(f"\nAvg Baseline AAR: {avg_base_aar:.3f}\n")
                    sf.write(f"Avg Final AAR:    {avg_final_aar:.3f}\n")
                    sf.write(f"Avg ΔAAR:         {(avg_final_aar-avg_base_aar):+.3f}\n")
                    sf.write(f"Avg Baseline Reward: {avg_base_reward:.3f}\n")
                    sf.write(f"Avg Final Reward:    {avg_final_reward:.3f}\n")
                    sf.write(f"Avg ΔReward:         {(avg_final_reward-avg_base_reward):+.3f}\n")
                    sf.write(f"Avg Baseline scTM: {avg_base_sctm:.3f}\n")
                    sf.write(f"Avg Final scTM:    {avg_final_sctm:.3f}\n")
                    sf.write(f"Avg ΔscTM:         {(avg_final_sctm-avg_base_sctm):+.3f}\n\n")
            sf.write("\nDone.\n")
        print(f"📊 Summary table saved → {summary_path}")
    except Exception as e:
        print(f"⚠️ Failed to write ablation summary: {e}")

if __name__ == "__main__":
    main()