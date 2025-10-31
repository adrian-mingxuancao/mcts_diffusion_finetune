#!/usr/bin/env python3
"""
UCT MCTS Experiments for Motif Scaffolding (Without Entropy)

Runs UCT-based MCTS experiments without entropy bonuses for motif scaffolding:
1. Single expert experiments: DPLM-2, Proteinea, FlowFlow, RFDiffusion
2. Multi-expert experiments: All experts combined
3. Pure UCB1 selection (no PH-UCT entropy bonuses)
4. Motif preservation and scaffold optimization

Usage:
    python uct_mcts_motif_scaffolding.py --mode single_expert --expert dplm2 --start 0 --end 3
    python uct_mcts_motif_scaffolding.py --mode multi_expert --start 0 --end 3
"""

import os, sys, argparse, json, time, numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from Bio import SeqIO, PDB

# Import MCTS components
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration

@dataclass
class MotifData:
    motif_sequence: str
    full_sequence: str
    scaffold_length: int
    motif_positions: List[int]
    pdb_id: str
    is_contiguous: bool

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_motif_data() -> Dict[str, MotifData]:
    """Load motif scaffolding data"""
    data_dir = "/home/caom/AID3/dplm/data-bin/scaffolding-pdbs"
    aa_seq_file = os.path.join(data_dir, "aa_seq.fasta")
    
    motif_data = {}
    full_sequences = {}
    
    # Load sequences
    if os.path.exists(aa_seq_file):
        for record in SeqIO.parse(aa_seq_file, "fasta"):
            full_sequences[record.id] = str(record.seq).upper()
        print(f"✅ Loaded {len(full_sequences)} sequences")
    
    # Extract motif info from PDB files
    parser = PDB.PDBParser(QUIET=True)
    aa_map = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
              'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
              'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
              'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    
    for pdb_id in full_sequences.keys():
        try:
            motif_pdb = os.path.join(data_dir, f"{pdb_id}_motif.pdb")
            if not os.path.exists(motif_pdb):
                continue
            
            structure = parser.get_structure(f"{pdb_id}_motif", motif_pdb)
            motif_residues = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.get_id()[0] == ' ':
                            res_num = residue.get_id()[1]
                            res_name = residue.get_resname()
                            if res_name in aa_map:
                                motif_residues.append((res_num, aa_map[res_name]))
            
            motif_residues.sort(key=lambda x: x[0])
            motif_sequence = ''.join([res[1] for res in motif_residues])
            motif_positions = [res[0] - 1 for res in motif_residues]
            
            is_contiguous = len(motif_positions) > 0 and (
                motif_positions == list(range(motif_positions[0], motif_positions[-1] + 1))
            )
            
            full_sequence = full_sequences[pdb_id]
            scaffold_length = len(full_sequence) - len(motif_sequence)
            
            motif_data[pdb_id] = MotifData(
                motif_sequence=motif_sequence,
                full_sequence=full_sequence,
                scaffold_length=scaffold_length,
                motif_positions=motif_positions,
                pdb_id=pdb_id,
                is_contiguous=is_contiguous
            )
            
            print(f"  ✅ {pdb_id}: motif='{motif_sequence}', scaffold={scaffold_length}, contiguous={is_contiguous}")
            
        except Exception as e:
            print(f"  ⚠️ Failed {pdb_id}: {e}")
    
    return motif_data

def calculate_motif_metrics(generated_seq: str, motif_data: MotifData) -> Dict[str, Any]:
    """Calculate motif preservation metrics"""
    motif_seq = motif_data.motif_sequence
    
    # Check exact preservation
    motif_preserved = motif_seq in generated_seq
    motif_position = generated_seq.find(motif_seq) if motif_preserved else -1
    
    # Coverage for non-contiguous
    if not motif_preserved and not motif_data.is_contiguous:
        motif_chars = set(motif_seq)
        generated_chars = set(generated_seq)
        motif_counts = {aa: motif_seq.count(aa) for aa in motif_chars}
        generated_counts = {aa: generated_seq.count(aa) for aa in generated_chars}
        
        coverage = sum(min(motif_counts[aa], generated_counts.get(aa, 0)) / motif_counts[aa] 
                      for aa in motif_chars) / len(motif_chars) if motif_chars else 0.0
        
        if coverage >= 0.8:
            motif_preserved = True
    else:
        coverage = 1.0 if motif_preserved else 0.0
    
    return {
        'motif_preserved': motif_preserved,
        'motif_coverage': coverage,
        'motif_position': motif_position,
        'scaffold_length': len(generated_seq) - len(motif_seq) if motif_preserved else len(generated_seq),
        'total_length': len(generated_seq),
        'length_accuracy': 1.0 - abs(len(generated_seq) - len(motif_data.full_sequence)) / len(motif_data.full_sequence)
    }

def run_uct_mcts_motif_experiment(
    motif_data: MotifData,
    mode: str,
    expert: Optional[str] = None,
    num_iterations: int = 25,
    max_depth: int = 5,
) -> Dict:
    """Run UCT MCTS experiment for motif scaffolding"""
    
    pdb_id = motif_data.pdb_id
    print(f"\n🧬 [{mode}{'_' + expert if expert else ''}] {pdb_id}")
    print(f"  Motif: '{motif_data.motif_sequence}' ({len(motif_data.motif_sequence)} res)")
    print(f"  Target: {len(motif_data.full_sequence)} res, Contiguous: {motif_data.is_contiguous}")
    
    try:
        # Initialize DPLM-2 integration
        dplm2 = DPLM2Integration(device="cuda")
        
        # Load external experts if needed
        external_experts = []
        if mode in ["single_expert", "multi_expert"]:
            try:
                from external_models.real_motif_experts import create_external_expert_for_mcts
                if expert in ["proteinea", "flowflow", "rfdiffusion"]:
                    ext_expert = create_external_expert_for_mcts(expert)
                    external_experts = [ext_expert]
                    print(f"  ✅ Loaded {expert} expert")
                elif mode == "multi_expert":
                    for exp_name in ["proteinea", "flowflow", "rfdiffusion"]:
                        ext_expert = create_external_expert_for_mcts(exp_name)
                        external_experts.append(ext_expert)
                    print(f"  ✅ Loaded {len(external_experts)} external experts")
            except Exception as e:
                print(f"  ⚠️ External experts failed: {e}")
        
        # Generate baseline using simple approach
        import random
        baseline_seq = motif_data.full_sequence  # Use reference as baseline
        baseline_metrics = calculate_motif_metrics(baseline_seq, motif_data)
        
        print(f"  ✅ Baseline: motif_preserved={baseline_metrics['motif_preserved']}, length={len(baseline_seq)}")
        
        # Prepare structure data for MCTS
        structure_data = {
            'motif_sequence': motif_data.motif_sequence,
            'motif_positions': motif_data.motif_positions,
            'full_sequence': motif_data.full_sequence,
            'scaffold_length': motif_data.scaffold_length,
            'is_contiguous': motif_data.is_contiguous,
            'pdb_id': pdb_id,
            'external_experts': external_experts
        }
        
        # Initialize UCT MCTS (NO ENTROPY - pure UCB1)
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            baseline_structure=structure_data,
            reference_sequence=motif_data.full_sequence,
            max_depth=max_depth,
            exploration_constant=1.414,
            ablation_mode=mode,
            single_expert_id=0 if expert == "dplm2" else None,
            external_experts=external_experts,
            num_rollouts_per_expert=2,
            top_k_candidates=2,
            use_ph_uct=False,  # Enforce pure UCT selection
            task_type="motif_scaffolding",
            num_simulations=num_iterations,
            temperature=1.0,
            use_plddt_masking=False,  # Not applicable for motif scaffolding
            use_entropy_bonus=False,  # CRITICAL: Pure UCT
            use_novelty_bonus=False   # CRITICAL: Pure UCT
        )
        
        # Run MCTS search
        print(f"  🔄 Running UCT MCTS search ({num_iterations} iterations, max_depth={max_depth})...")
        start_time = time.time()
        
        root_node = mcts.search(
            initial_sequence=baseline_seq,
            reference_sequence=motif_data.full_sequence,
            num_iterations=num_iterations,
            structure_data=structure_data
        )
        
        search_time = time.time() - start_time
        
        if root_node is None:
            print(f"  ❌ MCTS search failed")
            return None
        
        # Find best sequence
        def find_best_node(node):
            best_node, best_score = node, getattr(node, "reward", 0.0)
            for child in node.children:
                child_best = find_best_node(child)
                child_score = getattr(child_best, "reward", 0.0)
                if child_score > best_score:
                    best_node, best_score = child_best, child_score
            return best_node
        
        expected_length = len(motif_data.full_sequence)
        best_node = find_best_node(root_node)
        best_seq = getattr(best_node, "sequence", None) if best_node else None
        fallback_used = False
        if not best_seq or len(best_seq) != expected_length:
            fallback_used = True
            best_seq = baseline_seq
            final_metrics = baseline_metrics
        else:
            final_metrics = calculate_motif_metrics(best_seq, motif_data)
        
        # Compile results
        result = {
            'pdb_id': pdb_id,
            'mode': f"{mode}{'_' + expert if expert else ''}",
            'method': 'UCT_MCTS_Motif',
            'motif_sequence': motif_data.motif_sequence,
            'baseline_sequence': baseline_seq,
            'final_sequence': best_seq,
            'is_contiguous': motif_data.is_contiguous,
            'baseline_motif_preserved': baseline_metrics['motif_preserved'],
            'final_motif_preserved': final_metrics['motif_preserved'],
            'motif_preservation_improvement': final_metrics['motif_preserved'] - baseline_metrics['motif_preserved'],
            'baseline_coverage': baseline_metrics['motif_coverage'],
            'final_coverage': final_metrics['motif_coverage'],
            'coverage_improvement': final_metrics['motif_coverage'] - baseline_metrics['motif_coverage'],
            'baseline_length_accuracy': baseline_metrics['length_accuracy'],
            'final_length_accuracy': final_metrics['length_accuracy'],
            'length_accuracy_improvement': final_metrics['length_accuracy'] - baseline_metrics['length_accuracy'],
            'search_time': search_time,
            'entropy_bonus_used': False,
            'novelty_bonus_used': False,
            'num_iterations': num_iterations,
            'max_depth': max_depth,
            'fallback_to_baseline': fallback_used,
        }
        
        # Print summary
        print(f"  ─────────────────────────────────────────")
        print(f"  UCT MCTS Motif Results [{mode}{'/' + expert if expert else ''}]")
        print(f"    Motif: {baseline_metrics['motif_preserved']} → {final_metrics['motif_preserved']}")
        print(f"    Coverage: {baseline_metrics['motif_coverage']:.1%} → {final_metrics['motif_coverage']:.1%}")
        print(f"    Length: {len(baseline_seq)} → {len(best_seq)} (target: {len(motif_data.full_sequence)})")
        if fallback_used:
            print("    ⚠️ Fallback: returned baseline sequence (invalid candidate length)")
        print(f"    Time: {search_time:.1f}s")
        print(f"  ─────────────────────────────────────────")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Experiment failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="UCT MCTS experiments for motif scaffolding")
    parser.add_argument("--mode", choices=["single_expert", "multi_expert", "all"], default="all")
    parser.add_argument("--expert", choices=["dplm2", "proteinea", "flowflow", "rfdiffusion"], help="Single expert")
    parser.add_argument("--start", type=int, default=0, help="Start structure index")
    parser.add_argument("--end", type=int, default=3, help="End structure index")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/caom/AID3/dplm/mcts_diffusion_finetune/results/uct_mcts_motif_analysis",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=25,
        help="Number of UCT iterations per motif",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum search depth for motif UCT",
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🧬 UCT MCTS Experiments - Motif Scaffolding (No Entropy)")
    print("=" * 70)
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Structure range: {args.start}-{args.end}")
    print(f"🔁 Iterations per motif: {args.num_iterations}")
    print(f"🌳 Max depth: {args.max_depth}")
    
    # Load motif data
    motif_data_dict = load_motif_data()
    if not motif_data_dict:
        print("❌ No motif data loaded")
        return
    
    # Select structures
    structure_ids = list(motif_data_dict.keys())[args.start:args.end]
    print(f"🧬 Processing {len(structure_ids)} structures")
    
    # Run experiments
    results = []
    for structure_id in structure_ids:
        motif_data = motif_data_dict[structure_id]
        
        if args.mode == "all":
            modes = ["single_expert", "multi_expert"]
            experts = ["dplm2", "proteinea", "flowflow", "rfdiffusion"]
        elif args.mode == "single_expert":
            modes = ["single_expert"]
            experts = [args.expert] if args.expert else ["dplm2", "proteinea", "flowflow", "rfdiffusion"]
        else:
            modes = ["multi_expert"]
            experts = [None]
        
        for mode in modes:
            if mode == "single_expert":
                for expert in experts:
                    result = run_uct_mcts_motif_experiment(
                        motif_data,
                        mode,
                        expert,
                        num_iterations=args.num_iterations,
                        max_depth=args.max_depth,
                    )
                    if result:
                        results.append(result)
            else:
                result = run_uct_mcts_motif_experiment(
                    motif_data,
                    mode,
                    num_iterations=args.num_iterations,
                    max_depth=args.max_depth,
                )
                if result:
                    results.append(result)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"uct_mcts_motif_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n🎉 UCT MCTS motif experiments complete!")
    print(f"📊 Processed {len(results)} experiments")
    print(f"📁 Results: {results_file}")

if __name__ == "__main__":
    main()
