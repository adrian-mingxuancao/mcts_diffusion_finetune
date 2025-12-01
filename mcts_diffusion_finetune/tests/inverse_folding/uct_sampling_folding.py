#!/usr/bin/env python3
"""
UCT Sampling Folding (Depth=1) - Direct Rollout Optimization without Tree Growth

This script performs sampling-based structure optimization using DPLM-2 experts.
Instead of building an MCTS tree, it performs direct rollouts from the baseline
for comparison with full MCTS.

Key differences from full MCTS:
- max_depth=1: No tree growth, only direct sampling
- num_iterations controls number of rollout batches
- Still uses pLDDT-based masking and multi-expert rollouts
- Evaluates best candidate from all rollouts
"""

# Patch ESM before any other imports to prevent regression weights download
def patch_esm_regression_weights():
    """Patch ESM to skip regression weight downloads that cause 403 errors"""
    try:
        import esm.pretrained as _esm_pkg
        def skip_regression_weights(model_name):
            return False
        _esm_pkg._has_regression_weights = skip_regression_weights
        print("✓ ESM regression weights patched")
    except ImportError:
        print("⚠ ESM not available for patching")

# Apply patch immediately
patch_esm_regression_weights()

import sys
sys.path.append('/home/caom/AID3/dplm/src')

import os
import json
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from Bio import SeqIO

# Core imports
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration
from utils.folding_metrics import evaluate_folding_metrics, calculate_folding_reward

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_cameo_sequences() -> Dict[str, str]:
    """Load CAMEO sequences from aatype.fasta"""
    sequences = {}
    fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences[record.id] = str(record.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(sequences)} CAMEO sequences")
    except Exception as e:
        print(f"⚠️ CAMEO sequence loading failed: {e}")
        sequences = {}
    
    return sequences

def load_reference_coordinates() -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """Load reference coordinates and structure data from CAMEO pkl files"""
    coordinates = {}
    cameo_structure_data = {}
    
    try:
        from utils.cameo_data_loader import CAMEODataLoader
        loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
        
        for idx, struct_file in enumerate(loader.structures):
            structure_id = struct_file.replace('.pkl', '')
            
            try:
                structure_data = loader.get_structure_by_index(idx)
                if not structure_data:
                    continue
                
                # Extract coordinates (try multiple keys)
                coords = None
                for coord_key in ['backbone_coords', 'coordinates', 'atom_positions']:
                    if coord_key in structure_data and structure_data[coord_key] is not None:
                        coords = structure_data[coord_key]
                        break
                
                if coords is not None:
                    # Ensure we have CA coordinates
                    if len(coords.shape) == 3 and coords.shape[1] >= 2:
                        ca_coords = coords[:, 1, :]  # CA atoms at index 1
                    else:
                        ca_coords = coords
                    
                    coordinates[structure_id] = ca_coords
                    
                    # Store CAMEO structure data for baseline structure tokens
                    cameo_structure_data[structure_id] = {
                        'struct_ids': structure_data.get('struct_ids'),
                        'struct_seq': structure_data.get('struct_seq'),
                        'plddt_scores': structure_data.get('plddt_scores'),
                        'aatype': structure_data.get('aatype')
                    }
                
            except Exception as e:
                continue
        
        print(f"✅ Loaded coordinates for {len(coordinates)} structures")
        print(f"✅ Loaded structure data for {len(cameo_structure_data)} structures")
        
    except Exception as e:
        print(f"⚠️ Coordinate loading failed: {e}")
    
    return coordinates, cameo_structure_data

def generate_esmfold_baseline(sequence: str, timeout_seconds: int = 120) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Generate ESMFold baseline structure with timeout"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("ESMFold generation timed out")
    
    try:
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        print(f"    🔄 Loading ESMFold model (timeout: {timeout_seconds}s)...")
        # Load ESMFold
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        print(f"    ✅ ESMFold loaded on {device}")
        
        # Clean sequence
        clean_seq = ''.join([aa for aa in sequence.upper() if aa in 'ACDEFGHIKLMNPQRSTVWY'])
        print(f"    📝 Cleaned sequence length: {len(clean_seq)} (original: {len(sequence)})")
        
        # Tokenize and predict
        print(f"    🔄 Tokenizing sequence...")
        with torch.no_grad():
            tokenized = tokenizer(clean_seq, return_tensors="pt", add_special_tokens=False)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            
            print(f"    🔄 Running ESMFold inference...")
            output = model(tokenized['input_ids'])
            
            print(f"    📊 ESMFold output attributes: {list(output.keys())}")
            
            # Extract coordinates
            positions = output.positions
            print(f"    📐 Positions shape: {positions.shape}")
            
            # Handle different output shapes
            if len(positions.shape) == 5:
                # Shape: [batch, 1, length, atoms, 3]
                coordinates = positions[0, 0, :, 1, :].cpu().numpy()  # CA atoms
                print(f"    ✅ Extracted CA coordinates from 5D tensor: {coordinates.shape}")
            elif len(positions.shape) == 4:
                # Shape: [batch, length, atoms, 3]
                coordinates = positions[0, :, 1, :].cpu().numpy()  # CA atoms
                print(f"    ✅ Extracted CA coordinates from 4D tensor: {coordinates.shape}")
            else:
                raise ValueError(f"Unexpected positions shape: {positions.shape}")
            
            # Extract pLDDT scores
            plddt = output.plddt
            print(f"    📊 pLDDT shape: {plddt.shape}")
            
            if len(plddt.shape) == 3:
                # Shape: [batch, length, atoms] - take CA atom pLDDT
                plddt_scores = plddt[0, :, 1].cpu().numpy()  # CA pLDDT
                print(f"    ✅ Extracted CA pLDDT from 3D tensor: {plddt_scores.shape}")
            elif len(plddt.shape) == 2:
                # Shape: [batch, length]
                plddt_scores = plddt[0, :].cpu().numpy()
                print(f"    ✅ Extracted pLDDT from 2D tensor: {plddt_scores.shape}")
            else:
                print(f"    ⚠️ Unexpected pLDDT shape: {plddt.shape}, using fallback")
                plddt_scores = np.ones(len(clean_seq)) * 0.8
        
        # Convert ESMFold coordinates to REAL DPLM-2 structure tokens
        print(f"    🔄 Converting ESMFold coordinates to REAL DPLM-2 structure tokens...")
        try:
            # Import DPLM structure tokenizer
            from byprot.models.utils import get_struct_tokenizer
            import torch
            
            struct_tokenizer = get_struct_tokenizer()
            struct_device = next(struct_tokenizer.parameters()).device if hasattr(struct_tokenizer, 'parameters') else torch.device('cpu')
            
            # Convert CA coordinates to full atom37 format
            seq_len = len(clean_seq)
            full_coords = torch.zeros((1, seq_len, 37, 3), dtype=torch.float32, device=struct_device)
            coords_tensor = torch.from_numpy(coordinates).float().to(struct_device)
            
            # Place CA coordinates at atom index 1 (standard CA position)
            full_coords[0, :, 1, :] = coords_tensor
            
            # Create residue mask (all positions valid)
            res_mask = torch.ones((1, seq_len), dtype=torch.bool, device=struct_device)
            seq_length_tensor = torch.tensor([seq_len], dtype=torch.long, device=struct_device)
            
            # Tokenize coordinates to get REAL structure tokens
            print(f"    🔍 Tokenizing coordinates: shape={full_coords.shape}")
            struct_tokens = struct_tokenizer.tokenize(full_coords, res_mask, seq_length_tensor)
            
            if struct_tokens is not None and isinstance(struct_tokens, torch.Tensor):
                # Convert struct_ids to sequence format
                struct_ids = struct_tokens.squeeze(0).detach().cpu().tolist()
                structure_tokens = struct_tokenizer.struct_ids_to_seq(struct_ids)
                print(f"    ✅ Generated REAL structure tokens: {len(structure_tokens)} chars")
                print(f"    🔍 Sample: {structure_tokens[:50]}...")
            else:
                print(f"    ⚠️ Structure tokenizer returned None, using fallback")
                structure_tokens = f"<cls_struct>{'<mask_struct>' * seq_len}<eos_struct>"
                
        except Exception as e:
            print(f"    ⚠️ Structure tokenization failed: {e}, using fallback")
            # Fallback to mask tokens
            seq_len = len(clean_seq)
            structure_tokens = f"<cls_struct>{'<mask_struct>' * seq_len}<eos_struct>"
    
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        # Cancel timeout
        signal.alarm(0)
        
        print(f"    ✅ ESMFold baseline generation successful")
        return coordinates, plddt_scores, structure_tokens
        
    except TimeoutError:
        print(f"    ⏰ ESMFold generation timed out after {timeout_seconds}s")
        return None, None, None
    except Exception as e:
        print(f"    ❌ ESMFold generation failed: {e}")
        return None, None, None
    finally:
        # Ensure timeout is cancelled
        signal.alarm(0)

def run_uct_sampling_folding_experiment(
    structure_id: str,
    sequence: str,
    reference_coords: np.ndarray,
    mode: str,
    expert_id: Optional[int] = None,
    cameo_structure_data: Dict = None,
    num_iterations: int = 10,
    max_depth: int = 1,
) -> Dict:
    """Run UCT sampling folding experiment (depth=1, no tree growth)"""
    
    print(f"\n🧬 [{mode}_{expert_id if expert_id is not None else 'multi'}] {structure_id}")
    
    try:
        # Step 1: Generate ESMFold baseline
        print(f"  🔄 Generating ESMFold baseline...")
        baseline_coords, baseline_plddt, baseline_structure_tokens = generate_esmfold_baseline(sequence)
        
        if baseline_coords is None:
            return {
                'structure_id': structure_id,
                'mode': mode,
                'expert_id': expert_id,
                'status': 'failed',
                'error': 'ESMFold baseline generation failed'
            }
        
        # Ensure we have valid pLDDT scores
        if baseline_plddt is None:
            print(f"  ⚠️ No pLDDT from ESMFold, using fallback scores")
            baseline_plddt = np.ones(len(sequence)) * 0.8  # Fallback pLDDT scores
        
        # Ensure we have valid structure tokens
        if baseline_structure_tokens is None:
            print(f"  ⚠️ No structure tokens from ESMFold, using mask tokens")
            baseline_structure_tokens = f"<cls_struct>{'<mask_struct>' * len(sequence)}<eos_struct>"
        
        # Step 2: Evaluate baseline using shared folding metrics
        print(f"  📊 Evaluating baseline metrics...")
        if reference_coords is not None:
            try:
                baseline_rmsd, baseline_tm, baseline_reward = evaluate_folding_metrics(
                    baseline_coords,
                    reference_coords,
                    sequence,
                )
                print(f"  ✅ Baseline: RMSD={baseline_rmsd:.3f}Å, TM={baseline_tm:.3f}, Reward={baseline_reward:.3f}")
            except Exception as exc:
                print(f"  ⚠️ Baseline metric computation failed: {exc}")
                baseline_rmsd, baseline_tm, baseline_reward = float('nan'), float('nan'), 0.0
        else:
            print("  ⚠️ Reference coordinates unavailable; baseline reward set to 0")
            baseline_rmsd, baseline_tm, baseline_reward = float('nan'), float('nan'), 0.0
        
        # Step 3: Initialize DPLM-2 integration
        print(f"  🔄 Initializing DPLM-2 integration for forward folding...")
        dplm2 = DPLM2Integration()
        
        # Use CAMEO structure data if available for baseline structure tokens
        cameo_struct_ids = None
        cameo_struct_seq = None
        if cameo_structure_data:
            cameo_struct_ids = cameo_structure_data.get('struct_ids')
            cameo_struct_seq = cameo_structure_data.get('struct_seq')
            print(f"  🔍 CAMEO data available: struct_ids={cameo_struct_ids is not None}, struct_seq={cameo_struct_seq is not None}")
            
            if cameo_struct_ids is not None:
                print(f"  🔍 CAMEO struct_ids length: {len(cameo_struct_ids)}")
            if cameo_struct_seq is not None:
                print(f"  🔍 CAMEO struct_seq length: {len(cameo_struct_seq)}")

        baseline_structure = {
            'sequence': sequence,  # FIXED sequence from aatype.fasta
            'length': len(sequence),
            'coordinates': baseline_coords,  # ESMFold baseline coordinates
            'plddt_scores': baseline_plddt if baseline_plddt is not None else np.ones(len(sequence)) * 0.8,
            'struct_seq': baseline_structure_tokens if baseline_structure_tokens else ' '.join(['<mask_struct>'] * len(sequence)),
            'structure_type': 'folding',  # Forward folding task
            'pdb_id': structure_id,
            'chain_id': 'A',
            'task': 'forward_folding',  # Explicitly set task type
            'task_type': 'folding',  # Also set task_type for consistency
            # Add CAMEO structure data if available
            'struct_ids': cameo_struct_ids,  # Pre-tokenized structure IDs from CAMEO
            'baseline_rmsd': baseline_rmsd,
            'baseline_tm': baseline_tm,
        }
        
        # Add CAMEO struct_seq if available and better than ESMFold tokens
        if cameo_struct_seq and len(cameo_struct_seq) > 0:
            baseline_structure['struct_seq'] = cameo_struct_seq
            print(f"  ✅ Using CAMEO struct_seq as baseline: {len(cameo_struct_seq)} chars")

        print(f"  🔍 Baseline structure tokens: {baseline_structure['struct_seq'][:100]}...")
        print(f"  🔍 Final baseline struct_seq: {baseline_structure['struct_seq'][:100]}...")
        baseline_structure['baseline_reward'] = baseline_reward
        baseline_structure['target_coordinates'] = reference_coords

        # Step 4: Initialize UCT Sampling (DEPTH=1 - NO TREE GROWTH)
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            baseline_structure=baseline_structure,
            reference_sequence=sequence,  # Fixed sequence from aatype.fasta
            reference_coords=reference_coords,
            max_depth=max_depth,
            exploration_constant=1.414,
            ablation_mode=mode,
            single_expert_id=expert_id,
            external_experts=[],
            num_rollouts_per_expert=2,
            top_k_candidates=2,
            use_ph_uct=False,
            task_type='folding',
            num_simulations=num_iterations,
            temperature=1.0,
            use_plddt_masking=True,
            exclude_proteinmpnn=True
        )
        print("  ⚙️ UCT mode: standard UCB1 (entropy bonuses disabled)")

        # Step 5: Run UCT Sampling (Direct Rollouts)
        print(f"  🔄 Running UCT Sampling ({num_iterations} rollout batches, depth={max_depth})...")
        try:
            if mode == 'single_expert':
                external_experts = {}
                if expert_id == 0:
                    print("    🤖 Using DPLM-2 650M expert")
                elif expert_id == 1:
                    print("    🤖 Using DPLM-2 150M expert")
                elif expert_id == 2:
                    print("    🤖 Using DPLM-2 3B expert")
                best_candidate = mcts.search(
                    initial_sequence=sequence,
                    ablation_mode="single_expert",
                    single_expert_id=expert_id,
                    external_experts=external_experts,
                    num_iterations=num_iterations,
                )
            else:
                print("    🤖 Multi-expert: 3 DPLM-2 models (ProteinMPNN excluded for folding)")
                best_candidate = mcts.search(
                    initial_sequence=sequence,
                    ablation_mode="multi_expert",
                    external_experts={},
                    num_iterations=num_iterations,
                )

            if best_candidate:
                final_rmsd = getattr(best_candidate, 'rmsd', baseline_rmsd)
                final_tm = getattr(best_candidate, 'tm_score', baseline_tm)
                final_reward = getattr(best_candidate, 'reward', baseline_reward)
                
                candidate_coords = getattr(best_candidate, 'coordinates', None)
                if candidate_coords is not None and reference_coords is not None:
                    try:
                        final_rmsd, final_tm, final_reward = evaluate_folding_metrics(
                            candidate_coords,
                            reference_coords,
                            sequence,
                        )
                    except Exception as exc:
                        print(f"  ⚠️ Final metric computation failed: {exc}")
                        if not np.isnan(final_tm):
                            final_reward = calculate_folding_reward(final_tm, sequence)
                elif reference_coords is not None and not np.isnan(final_tm):
                    final_reward = calculate_folding_reward(final_tm, sequence)
                
                delta_rmsd = (baseline_rmsd - final_rmsd) if not np.isnan(baseline_rmsd) and not np.isnan(final_rmsd) else float('nan')
                delta_tm = (final_tm - baseline_tm) if not np.isnan(baseline_tm) and not np.isnan(final_tm) else float('nan')

                print(f"  ✅ Final: RMSD={final_rmsd:.3f}Å, TM={final_tm:.3f}, Reward={final_reward:.3f}")
                print(f"  📊 Improvement: ΔRMSD={delta_rmsd:+.3f}Å, ΔTM={delta_tm:+.3f}")

                return {
                    'structure_id': structure_id,
                    'mode': mode,
                    'expert_id': expert_id,
                    'status': 'success',
                    'baseline_rmsd': baseline_rmsd,
                    'baseline_tm_score': baseline_tm,
                    'baseline_reward': baseline_reward,
                    'final_rmsd': final_rmsd,
                    'final_tm_score': final_tm,
                    'final_reward': final_reward,
                    'rmsd_improvement': delta_rmsd,
                    'tm_improvement': delta_tm,
                    'reward_improvement': final_reward - baseline_reward,
                    'sequence_length': len(sequence),
                    'num_rollouts': num_iterations,
                    'sampling_method': 'uct_depth1'
                }
            else:
                return {
                    'structure_id': structure_id,
                    'mode': mode,
                    'expert_id': expert_id,
                    'status': 'failed',
                    'error': 'No valid candidates generated'
                }

        except Exception as e:
            print(f"  ❌ UCT Sampling failed: {e}")
            return {
                'structure_id': structure_id,
                'mode': mode,
                'expert_id': expert_id,
                'status': 'failed',
                'error': str(e)
            }

    except Exception as e:
        print(f"  ❌ UCT sampling experiment failed: {e}")
        return {
            'structure_id': structure_id,
            'mode': mode,
            'expert_id': expert_id,
            'status': 'failed',
            'error': str(e)
        }

def run_experiments(
    structure_ids: List[str],
    sequences: Dict[str, str],
    coordinates: Dict[str, np.ndarray],
    modes: List[str],
    expert_ids: List[Optional[int]],
    cameo_structure_data: Dict[str, Dict],
    num_iterations: int,
    max_depth: int,
) -> List[Dict]:
    """Run all UCT sampling folding experiments"""
    
    results = []
    total_experiments = len(structure_ids) * len(modes) * len(expert_ids)
    experiment_count = 0
    
    print(f"🚀 Starting UCT Sampling Folding Experiments (Depth=1)")
    print(f"📊 Total experiments: {total_experiments}")
    print(f"🧬 Structures: {len(structure_ids)}")
    print(f"🤖 Modes: {modes}")
    print(f"🔬 Expert IDs: {expert_ids}")
    print(f"🔁 Rollout batches per experiment: {num_iterations}")
    print(f"🌳 Max depth: {max_depth}")
    
    for structure_id in structure_ids:
        if structure_id not in sequences:
            print(f"⚠️ No sequence for {structure_id}")
            continue
            
        if structure_id not in coordinates:
            print(f"⚠️ No coordinates for {structure_id}")
            continue
            
        sequence = sequences[structure_id]
        reference_coords = coordinates[structure_id]
        
        for mode in modes:
            if mode == 'single_expert':
                for expert_id in expert_ids:
                    if expert_id is None:
                        continue
                    experiment_count += 1
                    print(f"\n{'='*60}")
                    print(f"🧪 Experiment {experiment_count}/{total_experiments}")
                    
                    result = run_uct_sampling_folding_experiment(
                        structure_id,
                        sequence,
                        reference_coords,
                        mode,
                        expert_id,
                        cameo_structure_data.get(structure_id),
                        num_iterations=num_iterations,
                        max_depth=max_depth,
                    )
                    results.append(result)
            else:
                experiment_count += 1
                print(f"\n{'='*60}")
                print(f"🧪 Experiment {experiment_count}/{total_experiments}")
                
                result = run_uct_sampling_folding_experiment(
                    structure_id,
                    sequence,
                    reference_coords,
                    mode,
                    None,
                    cameo_structure_data.get(structure_id),
                    num_iterations=num_iterations,
                    max_depth=max_depth,
                )
                results.append(result)
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='UCT Sampling Folding Experiments (Depth=1) with DPLM2Evaluator')
    parser.add_argument('--mode', choices=['single_expert', 'multi_expert', 'all'], 
                       default='all', help='Experiment mode')
    parser.add_argument('--expert_id', type=int, choices=[0, 1, 2], 
                       help='Expert ID for single expert mode (0=650M, 1=150M, 2=3B)')
    parser.add_argument('--start', type=int, default=0, help='Start structure index')
    parser.add_argument('--end', type=int, default=5, help='End structure index')
    parser.add_argument('--num_iterations', type=int, default=10,
                       help='Number of rollout batches per experiment')
    parser.add_argument('--max_depth', type=int, default=1,
                       help='Maximum depth for the sampling search')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load data
    print("📊 Loading CAMEO data...")
    sequences = load_cameo_sequences()
    coordinates, cameo_structure_data = load_reference_coordinates()
    
    if not sequences or not coordinates:
        print("❌ Failed to load CAMEO data")
        return
    
    # Get structure IDs
    structure_ids = list(sequences.keys())[args.start:args.end]
    print(f"📊 Processing {len(structure_ids)} structures: {structure_ids}")
    print(f"🔁 Rollout batches per structure: {args.num_iterations}")
    print(f"🌳 Max depth: {args.max_depth}")
    
    # Determine modes and expert IDs
    if args.mode == 'single_expert':
        modes = ['single_expert']
        expert_ids = [args.expert_id] if args.expert_id is not None else [0, 1, 2]
    elif args.mode == 'multi_expert':
        modes = ['multi_expert']
        expert_ids = [None]
    else:  # all
        modes = ['single_expert', 'multi_expert']
        expert_ids = [0, 1, 2, None]
    
    # Run experiments
    results = run_experiments(
        structure_ids,
        sequences,
        coordinates,
        modes,
        expert_ids,
        cameo_structure_data,
        num_iterations=args.num_iterations,
        max_depth=args.max_depth,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"uct_sampling_folding_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    successful_results = [r for r in results if r.get('status') == 'success']
    if successful_results:
        print(f"\n📊 Summary ({len(successful_results)} successful experiments):")
        
        for mode in set(r['mode'] for r in successful_results):
            mode_results = [r for r in successful_results if r['mode'] == mode]
            if not mode_results:
                continue
                
            avg_rmsd_improvement = np.mean([r.get('rmsd_improvement', 0) for r in mode_results])
            avg_tm_improvement = np.mean([r.get('tm_improvement', 0) for r in mode_results])
            avg_reward_improvement = np.mean([r.get('reward_improvement', 0) for r in mode_results])
            
            print(f"\n🎯 {mode.upper()}:")
            print(f"   📈 Avg RMSD improvement: {avg_rmsd_improvement:.3f}Å")
            print(f"   📈 Avg TM improvement: {avg_tm_improvement:.3f}")
            print(f"   📈 Avg Reward improvement: {avg_reward_improvement:.3f}")
            print(f"   ✅ Success rate: {len(mode_results)}/{len([r for r in results if r['mode'] == mode])}")
    
    print(f"\n🎉 UCT Sampling Folding Analysis Complete!")
    print(f"📊 Method: Direct rollout sampling (depth=1, no tree growth)")
    print(f"🔬 Total experiments: {len(results)}")
    print(f"✅ Successful: {len(successful_results)}")

if __name__ == "__main__":
    main()
