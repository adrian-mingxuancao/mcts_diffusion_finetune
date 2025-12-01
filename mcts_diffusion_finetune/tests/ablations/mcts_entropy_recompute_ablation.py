#!/usr/bin/env python3
"""
MCTS Entropy Recompute Ablation

This script tests MCTS with entropy recomputation at each node, as opposed to
caching entropy values. This addresses the reviewer's concern about sensitivity
to caching vs recomputing uncertainty (MI/entropy) during selection.

Key difference from cached entropy:
- Cached: Compute entropy once per candidate during expansion, store in node
- Recompute: Recompute entropy at each selection step based on current tree state

Parameters match mcts_tree_search_ablation.py for fair comparison.
"""

import os, sys, json, time
from datetime import datetime
import pickle
import glob

# Project path bootstrap
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np

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
from core.sequence_level_mcts import GeneralMCTS, MCTSNode
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

# Cached baselines are no longer used - we load pregenerated DPLM-2 baselines
# directly from FASTA files in the structure loading loop


class EntropyRecomputeMCTS(GeneralMCTS):
    """
    MCTS variant that recomputes entropy at each selection step.
    
    Instead of caching entropy values in nodes during expansion,
    this variant recomputes entropy dynamically during selection
    based on the current tree state and masked positions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_recompute_count = 0
        print("🔄 Entropy Recompute Mode: Entropy will be recomputed at each selection step")
    
    def _uct_selection(self, root: MCTSNode) -> MCTSNode:
        """
        UCT selection with entropy recomputation at each selection step.
        
        This overrides the parent's _uct_selection to recompute entropy
        dynamically during tree traversal rather than using cached values.
        """
        node = root
        while node.children and len(node.masked_positions) > 0:
            # Track selection calls
            self.selection_count += 1
            
            # Recompute entropy for each child before selection
            best_child = None
            best_score = float('-inf')
            
            for child in node.children:
                # RECOMPUTE entropy instead of using cached value
                # Use child's masked positions if available, otherwise use parent's
                masked_pos = child.masked_positions if hasattr(child, 'masked_positions') and child.masked_positions else node.masked_positions
                
                if child.sequence and masked_pos and len(masked_pos) > 0:
                    try:
                        recomputed_entropy = self._compute_expert_entropy(
                            child.sequence,
                            child.expert_source or "unknown",
                            masked_pos
                        )
                        self.entropy_recompute_count += 1
                        
                        # Temporarily update entropy for scoring
                        child.entropy = recomputed_entropy
                    except Exception as e:
                        print(f"      ⚠️ Entropy recomputation failed: {e}")
                        # Keep cached entropy
                
                # Compute PH-UCT score with recomputed entropy
                if self.use_ph_uct:
                    score = child.ph_uct_score(self.exploration_constant)
                else:
                    score = child.uct_score(self.exploration_constant)
                
                if score > best_score:
                    best_score = score
                    best_child = child
            
            node = best_child if best_child else node.children[0]
        
        return node
    
    def search(self, *args, **kwargs):
        """Run search and report entropy recomputation statistics"""
        self.entropy_recompute_count = 0
        self.selection_count = 0
        start_time = time.time()
        
        result = super().search(*args, **kwargs)
        
        elapsed = time.time() - start_time
        print(f"\n📊 Entropy Recomputation Statistics:")
        print(f"   Total selections: {self.selection_count}")
        print(f"   Total recomputations: {self.entropy_recompute_count}")
        if self.entropy_recompute_count > 0:
            print(f"   Recomputations per second: {self.entropy_recompute_count / elapsed:.1f}")
            print(f"   Average time per recomputation: {elapsed / self.entropy_recompute_count * 1000:.2f} ms")
        else:
            print(f"   ⚠️ No entropy recomputations occurred!")
            print(f"   This may indicate: (1) No children were created, or (2) All children had no masked positions")
        
        return result


def run_single_structure_entropy_recompute(
    structure, ref_seq, dplm2, structure_idx,
    num_iterations=50, max_depth=3,
    use_plddt_masking=True,
    ablation_mode="multi_expert",
    single_expert_id=None
):
    """
    Run MCTS with entropy recomputation on a single structure.
    
    Parameters match mcts_tree_search_ablation.py for fair comparison.
    """
    ref_id = structure.get('pdb_id', '') + '_' + structure.get('chain_id', '')
    if not ref_id or ref_id == '_':
        ref_id = structure.get('name', '').replace('CAMEO ', '')
    
    print(f"\n{'='*80}")
    print(f"Structure: {ref_id}")
    print(f"Mode: {ablation_mode} (Entropy Recompute)")
    if single_expert_id is not None:
        print(f"Single Expert ID: {single_expert_id}")
    print(f"{'='*80}\n")
    
    # Load pregenerated baseline
    pregenerated_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{ref_id}.fasta"
    baseline_seq = None
    
    if os.path.exists(pregenerated_path):
        try:
            from Bio import SeqIO
            for record in SeqIO.parse(pregenerated_path, "fasta"):
                baseline_seq = str(record.seq).replace(" ", "").upper()
                break
            if baseline_seq:
                print(f"✅ Loaded pregenerated baseline: {len(baseline_seq)} residues")
        except Exception as e:
            print(f"⚠️ Failed to load pregenerated baseline: {e}")
    
    if not baseline_seq:
        print("⚠️ No pregenerated baseline found, skipping structure")
        return None
    
    baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
    print(f"  📊 Baseline AAR: {baseline_aar:.1%}")
    
    # Prepare baseline structure
    baseline_struct = dict(structure)
    baseline_struct['structure_idx'] = structure_idx
    
    # Load external experts if needed
    # External experts are now integrated into DPLM2Integration
    # ProteinMPNN is expert ID 3 and will be loaded on-demand
    external_experts = []
    
    # No need to manually load external experts - they're handled by DPLM2Integration
    # The system will automatically load ProteinMPNN (expert 3) when needed
    print(f"🤖 Using built-in expert system (ProteinMPNN will be loaded on-demand)")
    
    if single_expert_id is not None:
        baseline_struct['single_expert_id'] = single_expert_id
    
    # Initialize MCTS with entropy recomputation
    mcts = EntropyRecomputeMCTS(
        dplm2_integration=dplm2,
        baseline_structure=baseline_struct,
        reference_sequence=ref_seq,
        max_depth=max_depth,
        exploration_constant=1.414,
        ablation_mode=ablation_mode,
        single_expert_id=single_expert_id,
        external_experts=external_experts,
        num_rollouts_per_expert=2,
        top_k_candidates=2,
        task_type="inverse_folding",
        num_simulations=num_iterations,
        temperature=1.0,
        use_plddt_masking=use_plddt_masking
    )
    
    # Set baseline in DPLM2Integration
    dplm2.set_baseline_structure(baseline_struct)
    dplm2.set_baseline_sequence(baseline_seq)
    
    # Configure ablation mode
    if ablation_mode == "random_no_expert":
        print("🎲 Configuring random no expert mode (with entropy recompute)")
    elif ablation_mode == "single_expert":
        expert_id = int(single_expert_id or 0)
        print(f"🎯 Configuring single expert mode with expert {expert_id} (with entropy recompute)")
    else:
        print("🤖 Configuring multi-expert mode (with entropy recompute)")
    
    # Calculate baseline reward
    def calculate_biophysical_score(sequence):
        if not sequence:
            return 0.0
        hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)
        charged = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
        charge_penalty = max(0, charged - 0.3) * 2
        hydrophobic_penalty = max(0, hydrophobic - 0.4) * 2
        base_score = 1.0 - charge_penalty - hydrophobic_penalty
        return max(0.0, min(1.0, base_score))
    
    baseline_sctm = None
    try:
        if baseline_struct.get('coordinates') is not None:
            from utils.sctm_calculation import calculate_sctm_score
            reference_coords = baseline_struct.get('coordinates')
            baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
        
        baseline_biophysical = calculate_biophysical_score(baseline_seq)
        
        if baseline_sctm is not None:
            baseline_reward = 0.4 * baseline_aar + 0.45 * baseline_sctm + 0.15 * baseline_biophysical
            print(f"  📈 Baseline reward: AAR={baseline_aar:.3f}, scTM={baseline_sctm:.3f}, B={baseline_biophysical:.3f} → R={baseline_reward:.3f}")
        else:
            baseline_reward = baseline_aar
            print(f"  📈 Baseline reward (AAR-only): {baseline_reward:.3f}")
    except Exception as e:
        print(f"  ⚠️ Baseline reward computation failed: {e}")
        baseline_reward = baseline_aar
    
    # Run MCTS search with entropy recomputation
    t0 = time.time()
    structure_data = {
        'struct_seq': baseline_struct.get('struct_seq', ''),
        'length': baseline_struct.get('length', len(baseline_seq)),
        'pdb_id': baseline_struct.get('pdb_id', ''),
        'chain_id': baseline_struct.get('chain_id', ''),
        'coordinates': baseline_struct.get('coordinates'),
        'plddt_scores': baseline_struct.get('plddt_scores', [])
    }
    
    root_node = mcts.search(
        initial_sequence=baseline_seq,
        reference_sequence=ref_seq,
        num_iterations=num_iterations,
        structure_data=structure_data
    )
    elapsed = time.time() - t0
    
    if root_node is None:
        print("  ❌ MCTS search failed")
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
    
    best_node = find_best_node(root_node)
    best_seq = best_node.sequence
    best_aar = calculate_simple_aar(best_seq, ref_seq)
    
    # Calculate final scTM
    final_sctm = None
    try:
        from utils.sctm_calculation import calculate_sctm_score
        reference_coords = baseline_struct.get('coordinates')
        
        if reference_coords is not None:
            print(f"  🧬 Calculating final scTM...")
            baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
            final_sctm = calculate_sctm_score(best_seq, reference_coords)
    except Exception as e:
        print(f"  ⚠️ scTM calculation failed: {e}")
        final_sctm = None
    
    # Calculate final reward
    try:
        final_biophysical = calculate_biophysical_score(best_seq)
        
        if final_sctm is not None:
            final_reward = 0.4 * best_aar + 0.45 * final_sctm + 0.15 * final_biophysical
            print(f"  🎯 Final reward: AAR={best_aar:.3f}, scTM={final_sctm:.3f}, B={final_biophysical:.3f} → R={final_reward:.3f}")
        else:
            final_reward = best_aar
            print(f"  🎯 Final reward (AAR-only): {final_reward:.3f}")
    except Exception as e:
        print(f"  ⚠️ Final reward computation failed: {e}")
        final_reward = best_aar
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY - {ref_id} (Entropy Recompute)")
    print(f"{'='*80}")
    print(f"Baseline AAR: {baseline_aar:.1%}")
    print(f"Final AAR: {best_aar:.1%} (Δ {(best_aar - baseline_aar):.1%})")
    if baseline_sctm is not None and final_sctm is not None:
        print(f"Baseline scTM: {baseline_sctm:.3f}")
        print(f"Final scTM: {final_sctm:.3f} (Δ {(final_sctm - baseline_sctm):+.3f})")
    print(f"Baseline Reward: {baseline_reward:.3f}")
    print(f"Final Reward: {final_reward:.3f} (Δ {(final_reward - baseline_reward):+.3f})")
    print(f"Time: {elapsed:.1f}s")
    print(f"Entropy Recomputations: {mcts.entropy_recompute_count}")
    print(f"{'='*80}\n")
    
    return {
        'structure_id': ref_id,
        'mode': f"{ablation_mode}_entropy_recompute",
        'baseline_sequence': baseline_seq,
        'final_sequence': best_seq,
        'baseline_aar': baseline_aar,
        'final_aar': best_aar,
        'baseline_sctm': baseline_sctm,
        'final_sctm': final_sctm,
        'baseline_reward': baseline_reward,
        'final_reward': final_reward,
        'time_seconds': elapsed,
        'entropy_recompute_count': mcts.entropy_recompute_count,
        'num_iterations': num_iterations,
        'max_depth': max_depth
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MCTS Entropy Recompute Ablation")
    parser.add_argument('--mode', type=str, default='multi_expert',
                       choices=['random_no_expert', 'single_expert', 'multi_expert'],
                       help='Ablation mode')
    parser.add_argument('--single_expert_id', type=int, default=None,
                       help='Expert ID for single_expert mode (0=650M, 1=150M, 2=3B, 3=ProteinMPNN)')
    parser.add_argument('--num_structures', type=int, default=10,
                       help='Number of structures to test')
    parser.add_argument('--start_index', type=int, default=0,
                       help='Start index for structures (for array/batch runs)')
    parser.add_argument('--num_iterations', type=int, default=50,
                       help='Number of MCTS iterations')
    parser.add_argument('--max_depth', type=int, default=3,
                       help='Maximum tree depth')
    parser.add_argument('--output_dir', type=str, 
                       default='/net/scratch/caom/mcts_entropy_recompute_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("\n" + "="*80)
    print("MCTS ENTROPY RECOMPUTE ABLATION")
    print("="*80)
    print(f"Mode: {args.mode}")
    if args.single_expert_id is not None:
        print(f"Single Expert ID: {args.single_expert_id}")
    print(f"Structures: {args.num_structures}")
    print(f"Start index: {args.start_index}")
    print(f"Iterations: {args.num_iterations}")
    print(f"Max Depth: {args.max_depth}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load reference sequences
    reference_seqs = load_correct_reference_sequences()
    
    # Initialize DPLM2
    print("🔄 Initializing DPLM2 integration...")
    dplm2 = DPLM2Integration(device='cuda')
    
    # Load CAMEO data
    data_dir = "/home/caom/AID3/dplm/data-bin/cameo2022"
    loader = CAMEODataLoader(data_dir)
    
    # Run experiments
    all_results = []
    start_idx = max(0, args.start_index)
    end_idx = min(start_idx + args.num_structures, len(loader.structures))
    for i in range(start_idx, end_idx):
        structure = loader.get_structure_by_index(i)
        if not structure:
            print(f"⚠️ Failed to load structure {i}, skipping")
            continue
        ref_id = structure.get('pdb_id', '') + '_' + structure.get('chain_id', '')
        
        if ref_id not in reference_seqs:
            print(f"⚠️ No reference sequence for {ref_id}, skipping")
            continue
        
        ref_seq = reference_seqs[ref_id]
        
        result = run_single_structure_entropy_recompute(
            structure=structure,
            ref_seq=ref_seq,
            dplm2=dplm2,
            structure_idx=i,
            num_iterations=args.num_iterations,
            max_depth=args.max_depth,
            ablation_mode=args.mode,
            single_expert_id=args.single_expert_id
        )
        
        if result:
            all_results.append(result)
            
            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(
                args.output_dir,
                f"{ref_id}_{args.mode}_entropy_recompute_{timestamp}.json"
            )
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    # Save summary
    if all_results:
        summary_file = os.path.join(
            args.output_dir,
            f"summary_{args.mode}_entropy_recompute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        summary = {
            'mode': f"{args.mode}_entropy_recompute",
            'num_structures': len(all_results),
            'num_iterations': args.num_iterations,
            'max_depth': args.max_depth,
            'mean_aar_improvement': np.mean([r['final_aar'] - r['baseline_aar'] for r in all_results]),
            'mean_time_seconds': np.mean([r['time_seconds'] for r in all_results]),
            'mean_entropy_recomputations': np.mean([r['entropy_recompute_count'] for r in all_results]),
            'results': all_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")
        print(f"   Mean AAR improvement: {summary['mean_aar_improvement']:.3%}")
        print(f"   Mean time: {summary['mean_time_seconds']:.1f}s")
        print(f"   Mean entropy recomputations: {summary['mean_entropy_recomputations']:.0f}")


if __name__ == '__main__':
    main()
