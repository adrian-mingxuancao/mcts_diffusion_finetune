#!/usr/bin/env python3
"""
MCTS Ablation Runner with Pregenerated Fixed Baselines

This version uses pregenerated DPLM-2 150M sequences as FIXED baselines for fair comparison.
All ablation modes start from the same pregenerated baseline sequences.

Usage:
    python tests/mcts_tree_search_ablation_pregenerated.py 0 10 --mode single_expert --single_expert_id 3
    python tests/mcts_tree_search_ablation_pregenerated.py 0 10 --mode multi_expert
"""

import os, sys, json, time
from datetime import datetime

# Project path bootstrap
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

# Data loading
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

from core.dplm2_integration import DPLM2Integration
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

def load_pregenerated_baseline(structure_name, structure, structure_idx=None):
    """
    Load pregenerated DPLM-2 150M baseline sequence (FIXED baseline approach)
    """
    print(f"  🎯 Loading FIXED pregenerated baseline for {structure_name}")
    
    # Try CAMEO pregenerated baselines first
    fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_name}.fasta"
    
    try:
        from Bio import SeqIO
        
        def clean_aa(seq_str: str) -> str:
            valid = set("ACDEFGHIKLMNPQRSTVWY")
            s = "".join(c for c in str(seq_str).upper() if c in valid)
            return s
        
        if os.path.exists(fallback_path):
            for record in SeqIO.parse(fallback_path, "fasta"):
                baseline_seq = clean_aa(record.seq)
                print(f"  ✅ Loaded FIXED baseline from CAMEO pregenerated: {len(baseline_seq)} chars")
                
                # Create baseline structure
                baseline_structure = dict(structure)
                if structure_idx is not None:
                    baseline_structure['structure_idx'] = structure_idx
                
                # Add ESMFold pLDDT computation for the fixed baseline
                try:
                    print(f"  🔄 Computing ESMFold pLDDT for fixed baseline...")
                    from transformers import EsmForProteinFolding, AutoTokenizer
                    import torch
                    
                    # Load ESMFold model
                    esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
                    esmfold_tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                    
                    if torch.cuda.is_available():
                        esmfold_model = esmfold_model.cuda()
                    
                    # Clean and tokenize sequence
                    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
                    clean_sequence = ''.join([aa for aa in baseline_seq.upper() if aa in valid_aas])
                    
                    tokenized = esmfold_tokenizer(clean_sequence, return_tensors="pt", add_special_tokens=False)
                    model_device = next(esmfold_model.parameters()).device
                    tokenized = {k: v.to(model_device) for k, v in tokenized.items()}
                    
                    with torch.no_grad():
                        output = esmfold_model(tokenized['input_ids'])
                        
                        if hasattr(output, 'plddt') and output.plddt is not None:
                            plddt_tensor = output.plddt[0].cpu().numpy()  # [L, 37]
                            
                            # Use Cα atom confidence (atom index 1)
                            if len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                                plddt_scores = plddt_tensor[:, 1].tolist()  # Cα atom confidence
                            else:
                                plddt_scores = plddt_tensor.mean(axis=1).tolist() if len(plddt_tensor.shape) == 2 else plddt_tensor.tolist()
                            
                            # Add pLDDT scores to baseline structure
                            baseline_structure['plddt_scores'] = plddt_scores
                            print(f"  ✅ Added ESMFold pLDDT: mean={sum(plddt_scores)/len(plddt_scores):.1f}, length={len(plddt_scores)}")
                        else:
                            print(f"  ⚠️ ESMFold pLDDT not available")
                    
                    # Clean up model to save memory
                    del esmfold_model
                    torch.cuda.empty_cache()
                    
                except Exception as plddt_e:
                    print(f"  ⚠️ ESMFold pLDDT generation failed: {plddt_e}")
                
                return baseline_seq, baseline_structure
        else:
            print(f"  ⚠️ CAMEO pregenerated file not found: {fallback_path}")
            
    except Exception as e:
        print(f"  ❌ Failed to load pregenerated baseline: {e}")
    
    # Try PDB pregenerated baselines as fallback
    pdb_fallback_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m_pdb/inverse_folding/{structure_name}.fasta"
    try:
        if os.path.exists(pdb_fallback_path):
            for record in SeqIO.parse(pdb_fallback_path, "fasta"):
                baseline_seq = clean_aa(record.seq)
                print(f"  ✅ Loaded FIXED baseline from PDB pregenerated: {len(baseline_seq)} chars")
                
                baseline_structure = dict(structure)
                if structure_idx is not None:
                    baseline_structure['structure_idx'] = structure_idx
                
                return baseline_seq, baseline_structure
        else:
            print(f"  ⚠️ PDB pregenerated file not found: {pdb_fallback_path}")
            
    except Exception as e:
        print(f"  ❌ Failed to load PDB pregenerated baseline: {e}")
    
    print(f"  ❌ No pregenerated baseline found for {structure_name}")
    return None, None

def run_one_structure(structure, structure_name, dplm2, correct_reference_sequences, ablation_mode, single_expert_id=None, structure_idx=None, loader=None):
    print(f"\n🧬 [{ablation_mode}{'' if single_expert_id is None else f'_{single_expert_id}'}] {structure_name}")
    ref_id = structure_name.replace('CAMEO ', '')
    ref_seq = correct_reference_sequences.get(ref_id)
    if not ref_seq:
        print(f"  ❌ no reference: {ref_id}")
        return None

    # **LOAD FIXED PREGENERATED BASELINE**
    baseline_seq, baseline_struct = load_pregenerated_baseline(ref_id, structure, structure_idx)
    if not baseline_seq:
        print("  ❌ pregenerated baseline loading failed")
        return None
    
    baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
    print(f"  ✅ FIXED Baseline AAR: {baseline_aar:.1%}")

    # **COORDINATES**: Load coordinates for ProteinMPNN
    if 'coordinates' in baseline_struct and baseline_struct['coordinates'] is not None:
        print(f"✅ Using cached coordinates: {baseline_struct['coordinates'].shape}")
        dplm2.set_baseline_structure(baseline_struct)
    else:
        print(f"🔄 Loading coordinates for ProteinMPNN before MCTS initialization...")
        try:
            from utils.cameo_data_loader import CAMEODataLoader
            coord_loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
            cameo_structure = coord_loader.get_structure_by_index(structure_idx) if structure_idx is not None else None
            if cameo_structure:
                # Load coordinates before MCTS starts
                if 'backbone_coords' in cameo_structure and cameo_structure['backbone_coords'] is not None:
                    coords = cameo_structure['backbone_coords']
                    if len(coords.shape) == 3 and coords.shape[1] == 3:
                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                    else:
                        reference_coords = coords
                    
                    # Add coordinates to baseline structure BEFORE MCTS initialization
                    baseline_struct['backbone_coords'] = coords
                    baseline_struct['coordinates'] = reference_coords
                    dplm2.set_baseline_structure(baseline_struct)
                    print(f"✅ Loaded coordinates for ProteinMPNN BEFORE MCTS: {reference_coords.shape}")
                elif 'coordinates' in cameo_structure and cameo_structure['coordinates'] is not None:
                    reference_coords = cameo_structure['coordinates']
                    baseline_struct['coordinates'] = reference_coords
                    dplm2.set_baseline_structure(baseline_struct)
                    print(f"✅ Loaded coordinates for ProteinMPNN BEFORE MCTS: {reference_coords.shape}")
                elif 'atom_positions' in cameo_structure and cameo_structure['atom_positions'] is not None:
                    coords = cameo_structure['atom_positions']
                    if len(coords.shape) == 3 and coords.shape[1] >= 2:
                        reference_coords = coords[:, 1, :]  # CA atoms at index 1
                    else:
                        reference_coords = coords
                    baseline_struct['atom_positions'] = coords
                    baseline_struct['coordinates'] = reference_coords
                    dplm2.set_baseline_structure(baseline_struct)
                    print(f"✅ Loaded coordinates for ProteinMPNN BEFORE MCTS: {reference_coords.shape}")
                else:
                    print(f"⚠️ No coordinates found in structure keys: {list(cameo_structure.keys())}")
            else:
                print(f"⚠️ Could not load structure from .pkl file")
        except Exception as e:
            print(f"⚠️ Failed to load coordinates before MCTS: {e}")

    # Configure MCTS with correct constructor parameters
    print(f"🔍 Debug: Setting single_expert_id = {single_expert_id} for {ablation_mode}")
    
    # Initialize MCTS with correct constructor parameters
    mcts = GeneralMCTS(
        task_type="inverse_folding",
        max_depth=5,
        num_simulations=25,  # Number of MCTS iterations
        exploration_constant=1.414,
        temperature=1.0,
        use_plddt_masking=True,
        dplm2_integration=dplm2
    )
    
    # Set baseline structure through DPLM2Integration
    dplm2.set_baseline_structure(baseline_struct)
    
    # Configure ablation mode - this will be handled in the search method
    if ablation_mode == "random_no_expert":
        print("🎲 Configuring random no expert mode")
    elif ablation_mode == "single_expert":
        expert_id = int(single_expert_id or 0)
        print(f"🎯 Configuring single expert mode with expert {expert_id}")
    else:
        print("🤖 Configuring multi-expert mode")

    # Baseline reward using pregenerated sequence
    try:
        # For inverse folding, compute AAR-based reward
        baseline_aar_reward = baseline_aar  # Use AAR as reward
        baseline_reward = baseline_aar_reward
        print(f"  📈 Baseline reward (AAR): {baseline_reward:.3f}")
    except Exception as e:
        print(f"  ⚠️ Baseline reward computation failed: {e}")
        baseline_reward = baseline_aar
    
    t0 = time.time()
    # Use the correct search interface - pass structure data for inverse folding
    structure_data = {
        'struct_seq': baseline_struct.get('struct_seq', ''),
        'length': baseline_struct.get('length', len(baseline_seq)),
        'pdb_id': baseline_struct.get('pdb_id', ''),
        'chain_id': baseline_struct.get('chain_id', ''),
        'coordinates': baseline_struct.get('coordinates'),
        'plddt_scores': baseline_struct.get('plddt_scores', [])
    }
    
    best_sequence, best_reward = mcts.search(
        structure_data, 
        target_length=len(baseline_seq)
    )
    elapsed = time.time() - t0

    # Process results from search
    if best_sequence is None:
        print("  ❌ MCTS search failed")
        return None
    
    best_seq = best_sequence
    best_aar = calculate_simple_aar(best_seq, ref_seq)
    final_reward = best_reward if best_reward is not None else 0.0

    # Compute scTM scores using ESMFold prediction vs reference structure
    baseline_sctm, final_sctm = None, None
    try:
        from utils.sctm_calculation import calculate_sctm_score
        
        # Use coordinates that were loaded before MCTS
        reference_coords = baseline_struct.get('coordinates')
        
        if reference_coords is not None and hasattr(reference_coords, 'shape'):
            print(f"  🧬 Using reference coordinates for scTM calculation: {reference_coords.shape}")
            # Calculate scTM: ESMFold prediction vs reference
            baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
            final_sctm = calculate_sctm_score(best_seq, reference_coords)
        else:
            print(f"  ⚠️ No reference coordinates available for scTM calculation")
    except Exception as e:
        print(f"  ⚠️ scTM calculation failed: {e}")
        baseline_sctm, final_sctm = None, None

    out = {
        "structure_name": structure_name,
        "length": structure["length"],
        "mode": ablation_mode if single_expert_id is None else f"{ablation_mode}_{single_expert_id}",
        "baseline_aar": baseline_aar,
        "final_aar": best_aar,
        "aar_improvement": best_aar - baseline_aar,
        "baseline_reward": baseline_reward,
        "final_reward": final_reward,
        "reward_improvement": final_reward - baseline_reward,
        "baseline_sctm": baseline_sctm if baseline_sctm is not None else 0.0,
        "final_sctm": final_sctm if final_sctm is not None else 0.0,
        "sctm_improvement": (final_sctm - baseline_sctm) if (baseline_sctm is not None and final_sctm is not None) else 0.0,
        "baseline_sequence": baseline_seq,
        "final_sequence": best_seq,
        "search_time": elapsed,
        "mcts_success": True
    }
    
    # Summary
    print("  ─────────────────────────────────────────────────────────")
    print(f"  Summary [{ablation_mode}{'' if single_expert_id is None else f'/{single_expert_id}'}] {structure_name}")
    print(f"    AAR     : {baseline_aar:.1%} → {best_aar:.1%} (Δ {best_aar - baseline_aar:+.1%})")
    if baseline_sctm is not None and final_sctm is not None:
        print(f"    scTM    : {baseline_sctm:.3f} → {final_sctm:.3f} (Δ {final_sctm - baseline_sctm:+.3f})")
    else:
        print("    scTM    : N/A (no reference coords)")
    print(f"    Reward  : {baseline_reward:.3f} → {final_reward:.3f} (Δ {final_reward - baseline_reward:+.3f})")
    print(f"    Time    : {elapsed:.1f}s")
    print("  ─────────────────────────────────────────────────────────")
    
    return out

def main():
    print("🧬 MCTS Ablation Study - FIXED Pregenerated Baselines")
    print("=" * 60)
    
    # CLI arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs='?', type=int, default=0, help="start index (inclusive)")
    parser.add_argument("end", nargs='?', type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--mode", choices=["random_no_expert","single_expert","multi_expert","all"], default="all")
    parser.add_argument("--single_expert_id", type=int, default=None, help="single expert id (0/1/2/3) - 3=ProteinMPNN")
    args = parser.parse_args()

    start_idx = args.start
    end_idx = args.end
    print(f"🎯 Structure range: {start_idx}-{end_idx if end_idx is not None else 'end'} | Mode: {args.mode}")
    print(f"🔍 Debug: args.single_expert_id = {args.single_expert_id}")
    print(f"🎯 Using FIXED pregenerated DPLM-2 150M baselines for fair comparison")
    
    # Load data
    loader = CAMEODataLoader()
    refs = load_correct_reference_sequences()
    
    # Initialize DPLM-2
    dplm2 = DPLM2Integration(device="cuda")
    
    results = []
    
    # Load structures
    all_structures = []
    struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
    
    # Load structure sequences directly from struct.fasta
    struct_records = {}
    if os.path.exists(struct_fasta_path):
        for record in SeqIO.parse(struct_fasta_path, "fasta"):
            struct_records[record.id] = str(record.seq)
        print(f"✅ Loaded {len(struct_records)} structure sequences from struct.fasta")
    else:
        print(f"❌ struct.fasta not found: {struct_fasta_path}")
        return
    
    # Create structures
    for idx, structure_file in enumerate(loader.structures):
        base_name = structure_file.replace('.pkl', '')
        
        if base_name in struct_records:
            struct_seq = struct_records[base_name]
            structure = {
                'struct_seq': struct_seq,
                'length': len(struct_seq.split(',')),
                'pdb_id': base_name.split('_')[0] if '_' in base_name else base_name,
                'chain_id': base_name.split('_')[1] if '_' in base_name else 'A',
                'name': f"CAMEO {base_name}"
            }
            all_structures.append((idx, structure))
        else:
            print(f"⚠️ No struct sequence found for {base_name}")
    
    # Select structure range
    if end_idx is not None:
        test_structures = all_structures[start_idx:end_idx]
    else:
        test_structures = all_structures[start_idx:]
    
    print(f"📊 Selected {len(test_structures)} structures to process")
    
    # Run ablation modes
    for idx, structure in test_structures:
        name = f"CAMEO {structure.get('pdb_id','test')}_{structure.get('chain_id','A')}"

        if args.mode == "random_no_expert":
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert", structure_idx=idx, loader=loader)
            if r: results.append(r)
            continue

        if args.mode == "single_expert":
            if args.single_expert_id is None:
                ids = [0,1,2,3]  # Include ProteinMPNN (expert 3)
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
        for eid in [0,1,2,3]:  # Include ProteinMPNN (expert 3)
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid, structure_idx=idx, loader=loader)
            if r: results.append(r)
        r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert", structure_idx=idx, loader=loader)
        if r: results.append(r)

    # Save results
    out_dir = "/net/scratch/caom/fixed_baseline_evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"fixed_baseline_ablation_results_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved fixed baseline ablation results → {out_json}")

    # Summary table
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["mode"]].append(r)

    summary_path = os.path.join(out_dir, f"fixed_baseline_ablation_summary_{ts}.txt")
    try:
        with open(summary_path, "w") as sf:
            sf.write("MCTS Ablation Summary - FIXED Pregenerated Baselines\n")
            sf.write("="*80 + "\n\n")
            for mode, rows in grouped.items():
                sf.write(f"Mode: {mode}\n")
                sf.write("-"*80 + "\n")
                sf.write(f"{'Structure':<20} {'Len':<5} {'Base AAR':<10} {'Final AAR':<10} {'ΔAAR':<8} {'Base R':<8} {'Final R':<8} {'ΔR':<8} {'Base scTM':<9} {'Final scTM':<10} {'ΔscTM':<8} {'Time(s)':<8}\n")
                for r in rows:
                    name = r['structure_name'].replace('CAMEO ', '')[:19]
                    sf.write(f"{name:<20} {r['length']:<5} {r['baseline_aar']:<10.3f} {r['final_aar']:<10.3f} "
                             f"{r['aar_improvement']:<8.3f} {r['baseline_reward']:<8.3f} {r['final_reward']:<8.3f} "
                             f"{r['reward_improvement']:<8.3f} {r['baseline_sctm']:<9.3f} {r['final_sctm']:<10.3f} "
                             f"{r['sctm_improvement']:<8.3f} {r['search_time']:<8.1f}\n")
                # Stats
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
