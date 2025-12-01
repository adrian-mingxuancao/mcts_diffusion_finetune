#!/usr/bin/env python3
"""
Run MCTS Experiments with Different Reward Weight Configurations

This script runs MCTS experiments with various reward weight configurations
to enable sensitivity analysis and Pareto front generation.

CRITICAL DESIGN DECISIONS:
1. **Fixed Baseline**: All reward weight configurations use the SAME pregenerated 
   baseline sequences from DPLM-2 150M. This ensures fair comparison across 
   different reward weights.
   
2. **Lead Optimization**: Starts from DPLM-2 generated sequences (AAR ~40-50%) 
   and uses MCTS to optimize them toward the native reference sequence.
   
3. **No Random Baselines**: Structures without pregenerated baselines are skipped
   to maintain experimental consistency.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration

# Import weight configurations from sensitivity analysis
from reward_weight_sensitivity_analysis import WEIGHT_CONFIGS, RewardWeightConfig
from utils.sctm_calculation import calculate_sctm_score


class WeightedMCTS(GeneralMCTS):
    """MCTS with configurable reward weights"""
    
    def __init__(self, reward_config: RewardWeightConfig, *args, **kwargs):
        """
        Initialize MCTS with specific reward weights.
        
        Args:
            reward_config: Reward weight configuration
            *args, **kwargs: Arguments for GeneralMCTS
        """
        super().__init__(*args, **kwargs)
        self.reward_config = reward_config
        self._sctm_cache: Dict[str, float] = {}
        print(f"🎯 Using reward weights: {reward_config.name}")
        print(f"   AAR: {reward_config.aar_weight:.2f}, "
              f"scTM: {reward_config.sctm_weight:.2f}, "
              f"Bio: {reward_config.biophysical_weight:.2f}")
    
    def _evaluate_sequence_aar(self, sequence: str) -> float:
        """
        Override reward calculation with configured weights.
        
        This replaces the hardcoded 0.6/0.35/0.05 weights with
        the configurable weights from reward_config.
        """
        if not self.reference_sequence or len(sequence) != len(self.reference_sequence):
            return 0.5
        
        # Reuse cached scTM when possible
        if sequence in self._sctm_cache:
            cached_sctm = self._sctm_cache[sequence]
        else:
            cached_sctm = None
        
        # Calculate AAR (Amino Acid Recovery)
        matches = sum(1 for a, b in zip(sequence, self.reference_sequence) if a == b)
        aar = matches / len(sequence)
        
        # Calculate scTM (structural similarity)
        if cached_sctm is not None:
            sctm = cached_sctm
        else:
            try:
                ref_coords = self.baseline_structure.get('coordinates')
                if ref_coords is not None:
                    sctm = float(calculate_sctm_score(sequence, np.asarray(ref_coords)))
                else:
                    raise ValueError("No reference coordinates for scTM")
            except Exception as e:
                try:
                    sctm = self.dplm2_integration.compute_sctm(  # type: ignore[attr-defined]
                        sequence, 
                        self.baseline_structure,
                        reference_sequence=self.reference_sequence
                    )
                except Exception:
                    # Structural proxy fallback
                    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)
                    charged = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
                    polar = sum(1 for aa in sequence if aa in 'NQSTY') / len(sequence)
                    sctm = (hydrophobic * 0.4 + charged * 0.3 + polar * 0.3) * 0.8 + aar * 0.2
            self._sctm_cache[sequence] = sctm
        
        # Calculate biophysical score
        try:
            import math
            aa_counts = {}
            for aa in sequence:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            total = len(sequence)
            diversity = 0.0
            for count in aa_counts.values():
                if count > 0:
                    p = count / total
                    diversity -= p * math.log2(p)
            
            biophysical = min(1.0, diversity / 4.32)
        except:
            biophysical = 0.8
        
        # USE CONFIGURED WEIGHTS instead of hardcoded values
        compound_reward = self.reward_config.compute_reward(aar, sctm, biophysical)
        
        print(f"      📊 [{self.reward_config.name}] AAR={aar:.3f}, scTM={sctm:.3f}, "
              f"B={biophysical:.3f} → R={compound_reward:.3f}")
        
        return compound_reward


def run_experiment_with_config(
    config: RewardWeightConfig,
    structure_data: Dict,
    output_dir: Path,
    mcts_params: Dict
) -> Dict:
    """
    Run single MCTS experiment with specific weight configuration.
    
    Args:
        config: Reward weight configuration
        structure_data: Structure data for experiment
        output_dir: Directory to save results
        mcts_params: MCTS parameters (depth, iterations, etc.)
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {config.name}")
    structure_name = f"{structure_data.get('pdb_id', 'unknown')}_{structure_data.get('chain_id', 'unknown')}"
    print(f"Structure: {structure_name}")
    print(f"{'='*80}\n")
    
    # Initialize DPLM2 integration
    dplm2_integration = DPLM2Integration(
        device='cuda'
    )
    
    # Set baseline structure in DPLM2Integration for inverse folding
    # This ensures real structure tokens are used, not mask tokens
    dplm2_integration.set_baseline_structure(structure_data['baseline_structure'])
    dplm2_integration.set_baseline_sequence(structure_data['baseline_sequence'])
    
    # CRITICAL: Compute baseline reward BEFORE MCTS initialization
    # This ensures the root node starts with the correct reward value
    baseline_sequence = structure_data['baseline_sequence']
    baseline_aar = sum(1 for a, b in zip(baseline_sequence, structure_data['reference_sequence']) 
                      if a == b) / len(baseline_sequence)
    
    # Compute baseline scTM
    baseline_sctm_precompute = None
    try:
        from utils.sctm_calculation import calculate_sctm_score
        reference_coords = structure_data['baseline_structure'].get('coordinates')
        if reference_coords is not None:
            baseline_sctm_precompute = calculate_sctm_score(baseline_sequence, reference_coords)
    except Exception as e:
        print(f"  ⚠️ Baseline scTM precomputation failed: {e}")
    
    # Compute baseline biophysical score
    from utils.folding_metrics import calculate_biophysical_score
    baseline_biophysical_precompute = calculate_biophysical_score(baseline_sequence)
    
    # Calculate baseline reward with config weights
    if baseline_sctm_precompute is not None:
        baseline_reward_precompute = (config.aar_weight * baseline_aar + 
                                     config.sctm_weight * baseline_sctm_precompute + 
                                     config.biophysical_weight * baseline_biophysical_precompute)
        print(f"  📊 Precomputed baseline: AAR={baseline_aar:.3f}, scTM={baseline_sctm_precompute:.3f}, Bio={baseline_biophysical_precompute:.3f}, Reward={baseline_reward_precompute:.3f}")
    else:
        baseline_reward_precompute = baseline_aar
        print(f"  📊 Precomputed baseline: AAR={baseline_aar:.3f}, Reward={baseline_reward_precompute:.3f} (scTM failed)")
    
    # Set baseline reward in structure data for MCTS root node
    structure_data['baseline_structure']['baseline_reward'] = baseline_reward_precompute
    
    # Initialize weighted MCTS
    mcts = WeightedMCTS(
        reward_config=config,
        dplm2_integration=dplm2_integration,
        baseline_structure=structure_data['baseline_structure'],
        reference_sequence=structure_data['reference_sequence'],
        max_depth=mcts_params.get('max_depth', 3),
        exploration_constant=1.414,
        ablation_mode='multi_expert',
        single_expert_id=None,
        external_experts=[],
        num_rollouts_per_expert=mcts_params.get('num_rollouts_per_expert', 2),
        top_k_candidates=mcts_params.get('top_k_candidates', 2),
        task_type='inverse_folding',
        num_simulations=mcts_params.get('num_iterations', 50),
        temperature=1.0,
        use_plddt_masking=True
    )
    
    # Save initial sequence as baseline BEFORE MCTS modifies it
    initial_sequence = structure_data['baseline_sequence']
    
    # Run MCTS search
    print(f"🌳 Starting MCTS search with {mcts_params.get('num_iterations', 50)} iterations...")
    root_node = mcts.search(
        initial_sequence=initial_sequence,
        reference_sequence=structure_data['reference_sequence'],
        num_iterations=mcts_params.get('num_iterations', 50)
    )
    
    # Extract best sequence from tree (search through all nodes)
    def find_best_node(node):
        best_node, best_score = node, getattr(node, "reward", 0.0)
        for child in node.children:
            child_best = find_best_node(child)
            child_score = getattr(child_best, "reward", 0.0)
            if child_score > best_score:
                best_node, best_score = child_best, child_score
        return best_node
    
    best_node = find_best_node(root_node)
    best_sequence = best_node.sequence
    
    # Use the initial sequence (before MCTS) as baseline for comparison
    baseline_sequence = initial_sequence
    
    # Evaluate final sequence
    final_aar = sum(1 for a, b in zip(best_sequence, structure_data['reference_sequence']) 
                    if a == b) / len(best_sequence)
    
    # Compute scTM using proper calculation method
    final_sctm = None
    baseline_sctm = None
    try:
        from utils.sctm_calculation import calculate_sctm_score
        reference_coords = structure_data['baseline_structure'].get('coordinates')
        
        if reference_coords is not None:
            print(f"  🧬 Calculating scTM scores...")
            baseline_sctm = calculate_sctm_score(baseline_sequence, reference_coords)
            final_sctm = calculate_sctm_score(best_sequence, reference_coords)
            print(f"  📊 Baseline scTM: {baseline_sctm:.3f}, Final scTM: {final_sctm:.3f}")
    except Exception as e:
        print(f"  ⚠️ scTM calculation failed: {e}")
        final_sctm = None
        baseline_sctm = None
    
    # Compute biophysical score
    from utils.folding_metrics import calculate_biophysical_score
    final_biophysical = calculate_biophysical_score(best_sequence)
    
    # Compute baseline metrics using root node sequence
    baseline_aar = sum(1 for a, b in zip(baseline_sequence, 
                                         structure_data['reference_sequence']) 
                      if a == b) / len(baseline_sequence)
    baseline_biophysical = calculate_biophysical_score(baseline_sequence)
    
    # Calculate rewards properly
    if final_sctm is not None and baseline_sctm is not None:
        baseline_reward = config.aar_weight * baseline_aar + config.sctm_weight * baseline_sctm + config.biophysical_weight * baseline_biophysical
        final_reward = config.aar_weight * final_aar + config.sctm_weight * final_sctm + config.biophysical_weight * final_biophysical
    else:
        # Fallback to AAR-only if scTM fails
        baseline_reward = baseline_aar
        final_reward = final_aar
    
    # Compile results
    results = {
        'config_name': config.name,
        'config_weights': {
            'aar': config.aar_weight,
            'sctm': config.sctm_weight,
            'biophysical': config.biophysical_weight
        },
        'structure_id': structure_name,
        'baseline_sequence': baseline_sequence,
        'final_sequence': best_sequence,
        'baseline_aar': baseline_aar,
        'final_aar': final_aar,
        'baseline_sctm': baseline_sctm if baseline_sctm is not None else 'N/A',
        'final_sctm': final_sctm if final_sctm is not None else 'N/A',
        'baseline_biophysical': baseline_biophysical,
        'final_biophysical': final_biophysical,
        'baseline_reward': baseline_reward,
        'final_reward': final_reward,
        'delta_aar': final_aar - baseline_aar,
        'delta_sctm': (final_sctm - baseline_sctm) if (final_sctm is not None and baseline_sctm is not None) else 'N/A',
        'delta_biophysical': final_biophysical - baseline_biophysical,
        'delta_reward': final_reward - baseline_reward,
        'mcts_params': mcts_params
    }
    
    # Save results
    output_file = output_dir / f"{structure_name}_{config.name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    if baseline_sctm is not None and final_sctm is not None:
        print(f"   Baseline: AAR={baseline_aar:.3f}, scTM={baseline_sctm:.3f}, Reward={baseline_reward:.3f}")
        print(f"   Final:    AAR={final_aar:.3f}, scTM={final_sctm:.3f}, Reward={final_reward:.3f}")
        print(f"   Delta:    AAR={results['delta_aar']:+.3f}, scTM={results['delta_sctm']:+.3f}, Reward={results['delta_reward']:+.3f}")
    else:
        print(f"   Baseline: AAR={baseline_aar:.3f}, Reward={baseline_reward:.3f}")
        print(f"   Final:    AAR={final_aar:.3f}, Reward={final_reward:.3f}")
        print(f"   Delta:    AAR={results['delta_aar']:+.3f}, Reward={results['delta_reward']:+.3f}")
    
    return results


def load_test_structures(
    data_dir: str,
    num_structures: int = 10,
    start_index: int = 0,
    dplm2: Optional[DPLM2Integration] = None
) -> List[Dict]:
    """Load test structures and generate DPLM-2 150M baselines deterministically."""
    from utils.cameo_data_loader import CAMEODataLoader
    from Bio import SeqIO
    import os
    import torch
    
    loader = CAMEODataLoader(data_dir)
    structures: List[Dict] = []
    
    print(f"📂 Found {len(loader.structures)} CAMEO structures")
    
    # Load reference sequences
    reference_fasta = os.path.join(data_dir, "aatype.fasta")
    reference_seqs = {}
    if os.path.exists(reference_fasta):
        for rec in SeqIO.parse(reference_fasta, "fasta"):
            reference_seqs[rec.id] = str(rec.seq).replace(" ", "").upper()
    
    # Use provided integration or create one for baseline generation
    dplm2 = dplm2 or DPLM2Integration(device='cuda')
    
    end_index = min(start_index + num_structures, len(loader.structures))
    for i in range(start_index, end_index):
        structure = loader.get_structure_by_index(i)
        if not structure:
            print(f"  ⚠️ Failed to load structure {i}")
            continue
        
        ref_id = f"{structure.get('pdb_id', '')}_{structure.get('chain_id', '')}"
        if ref_id not in reference_seqs:
            print(f"  ⚠️ No reference sequence for {ref_id}")
            continue
        
        structure['reference_sequence'] = reference_seqs[ref_id]
        
        # Deterministic baseline generation using DPLM-2 150M
        struct_tokens = structure.get('struct_seq', '')
        if isinstance(struct_tokens, (list, tuple)):
            struct_tokens = ','.join(map(str, struct_tokens))
        target_length = structure.get('length', len(structure['reference_sequence']))
        
        # Seed for reproducibility per structure
        seed_val = (hash(ref_id) % (2**31))
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val % (2**32 - 1))
        
        try:
            baseline_seq = dplm2.generate_baseline_sequence(struct_tokens, target_length, expert_id=1)
        except Exception as e:
            print(f"  ⚠️ Baseline generation failed for {ref_id}: {e}")
            continue
        
        if not baseline_seq or len(baseline_seq) == 0:
            print(f"  ⚠️ Empty baseline for {ref_id}, skipping")
            continue
        
        structure['baseline_sequence'] = baseline_seq
        structure['baseline_structure'] = structure  # Keep full structure dict
        
        baseline_aar = sum(1 for a, b in zip(baseline_seq, reference_seqs[ref_id]) if a == b) / len(baseline_seq)
        print(f"  ✅ Generated baseline for {ref_id} (AAR={baseline_aar:.1%}, seed={seed_val})")
        structures.append(structure)
        if len(structures) >= num_structures:
            break
    
    return structures


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Run MCTS experiments with different reward weight configurations"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/caom/AID3/dplm/data-bin/cameo2022',
        help='Directory containing CAMEO data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./weight_sensitivity_results',
        help='Directory to save experiment results'
    )
    parser.add_argument(
        '--num_structures',
        type=int,
        default=10,
        help='Number of structures to test'
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start index for structures (useful for batch/array runs)'
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=5,
        help='Maximum MCTS depth'
    )
    parser.add_argument(
        '--num_iterations',
        type=int,
        default=25,
        help='Number of MCTS iterations'
    )
    parser.add_argument(
        '--configs',
        type=str,
        nargs='+',
        default=None,
        help='Specific configurations to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("REWARD WEIGHT SENSITIVITY EXPERIMENTS")
    print("="*80 + "\n")
    
    # Load test structures
    print(f"📂 Loading test structures from: {args.data_dir} (start={args.start_index}, count={args.num_structures})")
    structures = load_test_structures(
        args.data_dir,
        args.num_structures,
        start_index=args.start_index
    )
    print(f"✅ Loaded {len(structures)} structures\n")
    
    # Select configurations to run
    if args.configs:
        configs_to_run = [c for c in WEIGHT_CONFIGS if c.name in args.configs]
    else:
        configs_to_run = WEIGHT_CONFIGS
    
    print(f"🎯 Running {len(configs_to_run)} weight configurations:")
    for config in configs_to_run:
        print(f"   - {config.name}: AAR={config.aar_weight:.2f}, "
              f"scTM={config.sctm_weight:.2f}, Bio={config.biophysical_weight:.2f}")
    print()
    
    # MCTS parameters
    mcts_params = {
        'max_depth': args.max_depth,
        'num_iterations': args.num_iterations,
        'num_rollouts_per_expert': 2,
        'top_k_candidates': 2,
        'use_entropy': True,
        'backup_rule': 'max',
        'model_size': '650m'
    }
    
    # Run experiments
    all_results = []
    total_experiments = len(structures) * len(configs_to_run)
    current_experiment = 0
    
    for structure in structures:
        for config in configs_to_run:
            current_experiment += 1
            print(f"\n{'='*80}")
            print(f"Experiment {current_experiment}/{total_experiments}")
            print(f"{'='*80}")
            
            try:
                results = run_experiment_with_config(
                    config=config,
                    structure_data=structure,
                    output_dir=output_dir,
                    mcts_params=mcts_params
                )
                all_results.append(results)
            except Exception as e:
                print(f"❌ Experiment failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Save summary
    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_experiments': total_experiments,
            'successful_experiments': len(all_results),
            'configurations': [c.name for c in configs_to_run],
            'structures': [f"{s['pdb_id']}_{s['chain_id']}" for s in structures],
            'mcts_params': mcts_params
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    print(f"\nNext step: Run sensitivity analysis:")
    print(f"  python analysis/reward_weight_sensitivity_analysis.py \\")
    print(f"    --results_dir {output_dir} \\")
    print(f"    --output_dir ./sensitivity_analysis_output")
    print()


if __name__ == '__main__':
    main()
