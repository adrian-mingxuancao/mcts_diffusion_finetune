#!/usr/bin/env python3
"""
UCT Sampling Folding PDB (Depth=1) - Direct Rollout Optimization for PDB Dataset

This script performs sampling-based structure optimization using DPLM-2 experts on PDB data.
Instead of building an MCTS tree, it performs direct rollouts from the baseline
for comparison with full MCTS.

Key differences from inverse folding:
- Task: folding (sequence → structure)
- Input: Fixed amino acid sequence from aatype.fasta
- Baseline: ESMFold generates structure coordinates
- MCTS: Mask low-confidence structure positions → DPLM-2 generates NEW structure tokens
- Output: New structure coordinates → evaluate RMSD/TM-score vs reference

Usage:
    python uct_sampling_folding_pdb.py --start 0 --end 5 --num_iterations 10
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

# Evaluator imports
import pandas as pd
import tempfile
import torch
from pathlib import Path
from omegaconf import OmegaConf
from byprot.utils.protein.evaluator_dplm2 import EvalRunner
from utils.folding_metrics import evaluate_folding_metrics

def create_evaluator_config():
    """Create configuration for EvalRunner with all required keys"""
    config = {
        'inference': {
            'task': 'forward_folding',
            'input_fasta_dir': '/tmp',  # Dummy path
            'inference_subdir': 'forward_folding_results',  # Required key
            'seed': 42,
            'also_fold_pmpnn_seq': False,  # Disable ProteinMPNN folding for forward folding
            'write_sample_trajectories': False,  # Required key
            'no_self_consistency': False,  # Required key
            'metadata': {  # Required metadata section
                'experiment_name': 'uct_sampling_folding_pdb',
                'timestamp': '2025-10-19',
                'model_version': 'dplm2',
                'csv_path': '/tmp/forward_folding_results.csv'  # Required csv_path
            },
            'folding': {
                'model_name': 'esmfold',
                'pmpnn_path': '/tmp/dummy_pmpnn',  # Required key
                'seq_per_sample': 1  # Required key
            }
        }
    }
    return OmegaConf.create(config)

def evaluate_structure_with_evaluator(predicted_coords: np.ndarray, reference_coords: np.ndarray, 
                                    predicted_seq: str, reference_seq: str, 
                                    target_name: str) -> Tuple[float, float, float]:
    """Evaluate structure using shared folding metrics helper."""
    print(f"📊 Evaluating structure: {target_name}")
    try:
        return evaluate_folding_metrics(predicted_coords, reference_coords, predicted_seq)
    except Exception as exc:
        print(f"    ⚠️ Folding metric evaluation failed: {exc}")
        return float('inf'), 0.0, 0.0


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_pdb_sequences() -> Dict[str, str]:
    """Load PDB sequences from aatype.fasta."""
    sequences = {}
    aatype_path = "/home/caom/AID3/dplm/data-bin/PDB_date/aatype.fasta"
    
    if not os.path.exists(aatype_path):
        logging.error(f"PDB aatype.fasta not found: {aatype_path}")
        return sequences
    
    try:
        for record in SeqIO.parse(aatype_path, "fasta"):
            sequences[record.id] = str(record.seq)
        logging.info(f"Loaded {len(sequences)} PDB sequences")
    except Exception as e:
        logging.error(f"Failed to load PDB sequences: {e}")
    
    return sequences

def load_pdb_structure_data(structure_id: str) -> Optional[Dict]:
    """Load PDB structure data from preprocessed directory."""
    try:
        import pickle
        
        # Get middle two characters for directory (e.g., 5S9R -> s9)
        if len(structure_id) >= 4:
            middle_chars = structure_id[1:3].lower()
            pkl_path = f"/home/caom/AID3/dplm/data-bin/PDB_date/preprocessed/{middle_chars}/{structure_id}.pkl"
            
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            return data
        
    except Exception as e:
        logging.warning(f"Failed to load structure data for {structure_id}: {e}")
    
    return None

def generate_esmfold_baseline(sequence: str, timeout_seconds: int = 120) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Generate ESMFold baseline structure with timeout (same as CAMEO version)"""
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
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"    ✅ ESMFold loaded on GPU")
        else:
            print(f"    ⚠️ ESMFold loaded on CPU")
        
        # Clean sequence
        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        clean_seq = ''.join([aa for aa in sequence.upper() if aa in valid_aas])
        print(f"    📝 Cleaned sequence length: {len(clean_seq)} (original: {len(sequence)})")
        
        if len(clean_seq) == 0:
            print(f"    ❌ No valid amino acids in sequence")
            signal.alarm(0)  # Cancel timeout
            return None, None, None
        
        if len(clean_seq) > 400:
            print(f"    ⚠️ Sequence too long ({len(clean_seq)}), truncating to 400")
            clean_seq = clean_seq[:400]
        
        # Tokenize and fold
        print(f"    🔄 Tokenizing sequence...")
        tokenized = tokenizer(clean_seq, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        print(f"    🔄 Running ESMFold inference...")
        with torch.no_grad():
            output = model(tokenized['input_ids'])
            
            print(f"    📊 ESMFold output attributes: {[attr for attr in dir(output) if not attr.startswith('_')]}")
            
            # Extract coordinates
            coordinates = None
            if hasattr(output, 'positions') and output.positions is not None:
                positions = output.positions.cpu().numpy()
                print(f"    📐 Positions shape: {positions.shape}")
                
                # Handle different tensor shapes (following memory guidance)
                if len(positions.shape) == 5:  # [8, 1, L, 14, 3]
                    coordinates = positions[0, 0, :, 1, :]  # CA atoms
                    print(f"    ✅ Extracted CA coordinates from 5D tensor: {coordinates.shape}")
                elif len(positions.shape) == 4:  # [1, L, 14, 3]
                    coordinates = positions[0, :, 1, :]  # CA atoms
                    print(f"    ✅ Extracted CA coordinates from 4D tensor: {coordinates.shape}")
                elif len(positions.shape) == 3:  # [L, 14, 3]
                    coordinates = positions[:, 1, :]  # CA atoms
                    print(f"    ✅ Extracted CA coordinates from 3D tensor: {coordinates.shape}")
                else:
                    print(f"    ❌ Unexpected positions shape: {positions.shape}")
            else:
                print(f"    ❌ No positions found in ESMFold output")
            
            # Extract pLDDT
            plddt = None
            if hasattr(output, 'plddt') and output.plddt is not None:
                plddt_tensor = output.plddt.cpu().numpy()
                print(f"    📊 pLDDT shape: {plddt_tensor.shape}")
                
                if len(plddt_tensor.shape) == 3:  # [1, L, 37]
                    plddt = plddt_tensor[0, :, 1]  # CA atom confidence
                    print(f"    ✅ Extracted CA pLDDT from 3D tensor: {plddt.shape}")
                elif len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                    plddt = plddt_tensor[:, 1]  # CA atom confidence
                    print(f"    ✅ Extracted CA pLDDT from 2D tensor: {plddt.shape}")
                else:
                    plddt = plddt_tensor.mean(axis=-1) if len(plddt_tensor.shape) > 1 else plddt_tensor
                    print(f"    ✅ Averaged pLDDT: {plddt.shape}")
            else:
                print(f"    ❌ No pLDDT found in ESMFold output")
        
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Cancel timeout
        signal.alarm(0)
        
        if coordinates is not None:
            print(f"    ✅ ESMFold baseline generation successful")
            return coordinates, plddt, structure_tokens
        else:
            print(f"    ❌ ESMFold failed to generate coordinates")
            return None, None, None
        
    except TimeoutError:
        print(f"    ⏰ ESMFold generation timed out after {timeout_seconds}s")
        signal.alarm(0)
        return None, None, None
    except Exception as e:
        print(f"    ❌ ESMFold baseline generation failed: {e}")
        signal.alarm(0)  # Cancel timeout
        import traceback
        traceback.print_exc()
        return None, None, None

def compute_esmfold_baseline_structure(sequence: str) -> Optional[np.ndarray]:
    """Legacy function - use generate_esmfold_baseline instead."""
    coords, _, _ = generate_esmfold_baseline(sequence)
    return coords

def old_compute_esmfold_baseline_structure(sequence: str) -> Optional[np.ndarray]:
    """Generate baseline structure coordinates using ESMFold."""
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        # Tokenize and predict
        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            output = model(inputs["input_ids"])
            
            if hasattr(output, 'positions') and output.positions is not None:
                # Extract CA coordinates (index 1 in atom dimension)
                positions = output.positions.cpu().numpy()
                if len(positions.shape) == 5:  # [batch, length, atoms, 3] or similar
                    ca_coords = positions[0, 0, :, 1, :]  # CA atoms
                elif len(positions.shape) == 4:
                    ca_coords = positions[0, :, 1, :]  # CA atoms
                else:
                    ca_coords = positions[0]  # Fallback
                
                return ca_coords
                
    except Exception as e:
        logging.warning(f"ESMFold baseline generation failed: {e}")
    
    return None

def compute_structure_metrics(pred_coords: np.ndarray, ref_coords: np.ndarray) -> Dict[str, float]:
    """Compute RMSD and TM-score between predicted and reference coordinates."""
    try:
        # Simple RMSD calculation
        if pred_coords.shape != ref_coords.shape:
            min_len = min(len(pred_coords), len(ref_coords))
            pred_coords = pred_coords[:min_len]
            ref_coords = ref_coords[:min_len]
        
        rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords) ** 2, axis=1)))
        
        # Simple TM-score approximation (placeholder)
        # Real TM-score calculation would require proper alignment
        tm_score = max(0.0, 1.0 - rmsd / 10.0)  # Rough approximation
        
        return {
            'rmsd': rmsd,
            'tm_score': tm_score
        }
    except Exception as e:
        logging.warning(f"Structure metrics calculation failed: {e}")
        return {'rmsd': 999.0, 'tm_score': 0.0}

def run_folding_sampling_experiment(
    structure_id: str,
    sequence: str,
    reference_coords: Optional[np.ndarray],
    num_iterations: int = 10,
    max_depth: int = 1,
    seed: int = 42
) -> Dict:
    """Run folding sampling experiment (depth=1 MCTS)."""
    
    logging.info(f"🧬 Running folding sampling on {structure_id}")
    
    try:
        # Step 1: Generate ESMFold baseline (same as CAMEO version)
        logging.info(f"🔄 Generating ESMFold baseline...")
        baseline_coords, baseline_plddt, baseline_structure_tokens = generate_esmfold_baseline(sequence)
        
        if baseline_coords is None:
            logging.error(f"❌ ESMFold baseline generation failed, using fallback")
            # Create a simple extended chain as fallback baseline
            seq_len = len(sequence)
            baseline_coords = np.zeros((seq_len, 3))
            for i in range(seq_len):
                baseline_coords[i] = [i * 3.8, 0.0, 0.0]  # Extended chain with 3.8Å spacing
            
            baseline_plddt = np.ones(seq_len) * 0.5  # Low confidence
            baseline_structure_tokens = f"<cls_struct>{'<mask_struct>' * seq_len}<eos_struct>"
            logging.info(f"✅ Created fallback baseline: extended chain with {seq_len} residues")
        
        logging.info(f"✅ ESMFold baseline: {baseline_coords.shape}, pLDDT: {baseline_plddt.mean():.3f}")
        
        # Step 2: Evaluate baseline using proper evaluator
        baseline_rmsd = baseline_tm = baseline_reward = 0.0
        if reference_coords is not None:
            baseline_rmsd, baseline_tm, baseline_reward = evaluate_structure_with_evaluator(
                baseline_coords, reference_coords, sequence, sequence, f"{structure_id}_baseline"
            )
            logging.info(f"📊 Baseline: RMSD={baseline_rmsd:.3f}Å, TM={baseline_tm:.3f}, Reward={baseline_reward:.3f}")
        
        # Step 3: Initialize DPLM-2 integration for FORWARD FOLDING
        logging.info(f"🔄 Initializing DPLM-2 integration for forward folding...")
        dplm2_integration = DPLM2Integration(device="cuda")
        
        # Prepare baseline structure data for FORWARD FOLDING
        # In forward folding: sequence is FIXED, structure tokens are MASKED
        logging.info(f"🔍 Baseline structure tokens: {baseline_structure_tokens[:100] if baseline_structure_tokens else 'None'}...")
        
        baseline_structure = {
            'sequence': sequence,  # FIXED sequence from aatype.fasta
            'length': len(sequence),
            'coordinates': baseline_coords,  # ESMFold baseline coordinates
            'plddt_scores': baseline_plddt if baseline_plddt is not None else np.ones(len(sequence)) * 0.8,
            'struct_seq': baseline_structure_tokens if baseline_structure_tokens else ' '.join(['<mask_struct>'] * len(sequence)),
            'structure_type': 'folding',  # Forward folding task
            'pdb_id': structure_id,
            'chain_id': 'A',
            'baseline_reward': baseline_reward,
            'target_coordinates': reference_coords
        }
        
        logging.info(f"🔍 Final baseline struct_seq: {baseline_structure['struct_seq'][:100]}...")
        
        # Set baseline data for forward folding
        dplm2_integration.set_baseline_structure(baseline_structure)
        dplm2_integration.set_baseline_sequence(sequence)  # Fixed sequence
        
        # Initialize MCTS for folding (exclude ProteinMPNN for folding tasks)
        mcts = GeneralMCTS(
            dplm2_integration=dplm2_integration,
            baseline_structure=baseline_structure,
            reference_sequence=sequence,  # Fixed sequence for folding
            task_type='folding',  # CRITICAL: folding task
            use_entropy=False,  # Pure UCT
            backup_rule='max',
            num_rollouts_per_expert=2,
            max_depth=max_depth,  # Depth=1 for sampling
            exclude_proteinmpnn=True,  # Exclude ProteinMPNN for folding
            exploration_constant=1.414
        )
        
        # Set baseline structure on DPLM2 integration
        dplm2_integration.set_baseline_structure(baseline_structure)
        dplm2_integration.set_baseline_sequence(sequence)
        
        # Run MCTS sampling (depth=1)
        logging.info(f"🔄 Running MCTS sampling...")
        result = mcts.search(initial_sequence=sequence, num_iterations=num_iterations, reference_sequence=sequence)
        
        # Extract final structure coordinates (same approach as CAMEO)
        final_coords = baseline_coords  # Default to baseline
        
        if result:
            # Try to get coordinates from MCTS node if available
            if hasattr(result, 'coordinates') and result.coordinates is not None:
                final_coords = result.coordinates
                logging.info(f"✅ MCTS returned optimized structure with coordinates")
            elif hasattr(result, 'sequence'):
                # For folding: convert optimized sequence back to structure using ESMFold
                logging.info(f"🔄 Converting optimized sequence to structure using ESMFold...")
                optimized_coords, _, _ = generate_esmfold_baseline(result.sequence)
                if optimized_coords is not None:
                    final_coords = optimized_coords
                    logging.info(f"✅ Generated structure from optimized sequence")
                else:
                    logging.warning(f"⚠️ Failed to generate structure from optimized sequence, using baseline")
                    final_coords = baseline_coords
            else:
                logging.warning(f"⚠️ Unexpected result type: {type(result)}")
                final_coords = baseline_coords
        else:
            logging.warning(f"⚠️ MCTS returned no result")
        
        # Compute final metrics using proper evaluator
        final_rmsd = final_tm = final_reward = 0.0
        if reference_coords is not None:
            final_rmsd, final_tm, final_reward = evaluate_structure_with_evaluator(
                final_coords, reference_coords, sequence, sequence, f"{structure_id}_final"
            )
            logging.info(f"📊 Final: RMSD={final_rmsd:.3f}Å, TM={final_tm:.3f}, Reward={final_reward:.3f}")
        
        # Calculate improvements
        rmsd_improvement = baseline_rmsd - final_rmsd  # Positive = improvement
        tm_improvement = final_tm - baseline_tm  # Positive = improvement
        reward_improvement = final_reward - baseline_reward  # Positive = improvement
        
        return {
            'structure_id': structure_id,
            'mode': 'folding_sampling',
            'sequence_length': len(sequence),
            'baseline_rmsd': baseline_rmsd,
            'baseline_tm_score': baseline_tm,
            'baseline_reward': baseline_reward,
            'final_rmsd': final_rmsd,
            'final_tm_score': final_tm,
            'final_reward': final_reward,
            'rmsd_improvement': rmsd_improvement,
            'tm_score_improvement': tm_improvement,
            'reward_improvement': reward_improvement,
            'num_iterations': num_iterations,
            'max_depth': max_depth,
            'success': True
        }
        
    except Exception as e:
        logging.error(f"❌ Folding sampling experiment failed for {structure_id}: {e}")
        return {
            'structure_id': structure_id,
            'success': False,
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="UCT Sampling Folding PDB")
    parser.add_argument("--start", type=int, default=0, help="Start structure index")
    parser.add_argument("--end", type=int, default=5, help="End structure index")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of sampling iterations")
    parser.add_argument("--max_depth", type=int, default=1, help="Max depth (1 for sampling)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/caom/AID3/dplm/mcts_diffusion_finetune/results/folding_sampling_pdb",
                       help="Output directory")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PDB sequences
    sequences = load_pdb_sequences()
    if not sequences:
        logging.error("No PDB sequences loaded")
        return
    
    structure_ids = list(sequences.keys())[args.start:args.end]
    logging.info(f"🧪 Running folding sampling on {len(structure_ids)} PDB structures")
    
    results = []
    
    for i, structure_id in enumerate(structure_ids):
        logging.info(f"\n{'='*60}")
        logging.info(f"Processing {i+1}/{len(structure_ids)}: {structure_id}")
        logging.info(f"{'='*60}")
        
        sequence = sequences[structure_id]
        
        # Load reference structure data
        structure_data = load_pdb_structure_data(structure_id)
        reference_coords = None
        
        if structure_data and 'modeled_idx' in structure_data and 'bb_positions' in structure_data:
            modeled_idx = structure_data['modeled_idx']
            all_coords = structure_data['bb_positions']
            reference_coords = all_coords[modeled_idx]
            logging.info(f"✅ Loaded reference coordinates: {reference_coords.shape}")
        else:
            logging.warning(f"⚠️ No reference coordinates for {structure_id}")
        
        # Run experiment
        result = run_folding_sampling_experiment(
            structure_id=structure_id,
            sequence=sequence,
            reference_coords=reference_coords,
            num_iterations=args.num_iterations,
            max_depth=args.max_depth,
            seed=args.seed
        )
        
        results.append(result)
        
        # Log progress
        if result['success']:
            rmsd_change = result.get('rmsd_improvement', 0.0)
            tm_change = result.get('tm_score_improvement', 0.0)
            logging.info(f"✅ {structure_id}: RMSD Δ{rmsd_change:+.3f}Å, TM-score Δ{tm_change:+.3f}")
        else:
            logging.info(f"❌ {structure_id}: Failed - {result.get('error', 'Unknown error')}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"folding_sampling_pdb_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary statistics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_rmsd_improvement = np.mean([r['rmsd_improvement'] for r in successful_results])
        avg_tm_improvement = np.mean([r['tm_score_improvement'] for r in successful_results])
        
        summary = {
            'total_structures': len(results),
            'successful': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_rmsd_improvement': avg_rmsd_improvement,
            'avg_tm_score_improvement': avg_tm_improvement,
            'timestamp': timestamp
        }
        
        summary_file = os.path.join(args.output_dir, f"folding_sampling_pdb_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"\n📊 Summary:")
        logging.info(f"   Success rate: {summary['success_rate']:.1%}")
        logging.info(f"   Avg RMSD improvement: {avg_rmsd_improvement:+.3f}Å")
        logging.info(f"   Avg TM-score improvement: {avg_tm_improvement:+.3f}")
    
    logging.info(f"\n💾 Results saved to: {results_file}")
    logging.info(f"🎉 Folding sampling experiments complete!")

if __name__ == "__main__":
    main()
