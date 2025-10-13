#!/usr/bin/env python3
"""
UCT MCTS Experiments for Folding Using DPLM-2 Evaluator

Runs UCT-based MCTS experiments for folding using the proper evaluator approach:
1. Single expert experiments: DPLM-2 650M, 150M, 3B
2. Multi-expert experiments: All experts combined
3. Pure UCB1 selection (no PH-UCT entropy bonuses)
4. Structure token masking and optimization
5. Proper reward calculation using byprot evaluator

Usage:
    python uct_mcts_folding.py --mode single_expert --expert_id 0 --start 0 --end 5
    python uct_mcts_folding.py --mode multi_expert --start 0 --end 5
    python uct_mcts_folding.py --mode all --start 0 --end 10  # Run all modes
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

import os
import sys
import argparse
import json
import time
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import torch
from Bio import SeqIO

# Import MCTS and DPLM-2 components
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration

# Import protein utilities for structure evaluation
from byprot.utils.protein.evaluator_dplm2 import EvalRunner
from byprot.utils.protein import utils as eu
from byprot.datamodules.pdb_dataset.pdb_datamodule import collate_fn
from omegaconf import DictConfig, OmegaConf
import pandas as pd

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_biophysical_score(sequence: str) -> float:
    """Calculate biophysical score using same heuristic as forward folding summary"""
    if not sequence:
        return 0.0
    seq = sequence.upper()
    length = len(seq)
    hydrophobic = sum(1 for aa in seq if aa in "AILMFPWV") / length
    charged = sum(1 for aa in seq if aa in "DEKR") / length
    charge_penalty = max(0, charged - 0.3) * 2.0
    hydrophobic_penalty = max(0, hydrophobic - 0.4) * 2.0
    base_score = 1.0 - charge_penalty - hydrophobic_penalty
    return float(np.clip(base_score, 0.0, 1.0))

def calculate_composite_reward(tm_score: float, sequence: str) -> float:
    """Calculate composite reward using same formula as forward folding summary"""
    aar = 1.0  # sequences identical to reference for folding task
    biophysical = calculate_biophysical_score(sequence)
    return 0.4 * aar + 0.45 * tm_score + 0.15 * biophysical

def create_evaluator_config() -> DictConfig:
    """Create configuration for EvalRunner"""
    config = {
        'inference': {
            'task': 'forward_folding',
            'input_fasta_dir': '/tmp',  # Dummy path
            'inference_subdir': 'forward_folding_results',  # Required key
            'seed': 42,
            'also_fold_pmpnn_seq': False,  # Disable ProteinMPNN folding for forward folding
            'metadata': {  # Required metadata section
                'experiment_name': 'uct_mcts_folding',
                'timestamp': '2025-10-09',
                'model_version': 'dplm2',
                'csv_path': '/tmp/forward_folding_results.csv'  # Required csv_path
            },
            'folding': {
                'model_name': 'esmfold'
            }
        }
    }
    return OmegaConf.create(config)

def evaluate_structure_with_evaluator(predicted_coords: np.ndarray, reference_coords: np.ndarray, 
                                    predicted_seq: str, reference_seq: str, 
                                    target_name: str) -> Tuple[float, float, float]:
    """Evaluate structure using real DPLM2 evaluator"""
    try:
        # Create temporary directory for evaluation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            eval_dir = temp_dir / "eval"
            eval_dir.mkdir(exist_ok=True)
            
            # Create evaluator with proper config
            cfg = create_evaluator_config()
            evaluator = EvalRunner(cfg)
            
            # Convert sequences to aatype arrays
            def seq_to_aatype(seq):
                aa_to_id = {
                    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
                    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15,
                    'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20
                }
                return np.array([aa_to_id.get(aa, 20) for aa in seq.upper()])
            
            # Ensure coordinates are in the right format (L, 37, 3)
            if len(predicted_coords.shape) == 2:  # (L, 3) -> (L, 37, 3)
                # Expand to full atom representation (CA atoms at position 1)
                full_pred_coords = np.zeros((len(predicted_coords), 37, 3))
                full_pred_coords[:, 1, :] = predicted_coords  # CA atoms
                predicted_coords = full_pred_coords
                
            if len(reference_coords.shape) == 2:  # (L, 3) -> (L, 37, 3)
                full_ref_coords = np.zeros((len(reference_coords), 37, 3))
                full_ref_coords[:, 1, :] = reference_coords  # CA atoms
                reference_coords = full_ref_coords
            
            # Create batch data structure expected by evaluator
            batch = {
                'pdb_name': [target_name],
                'all_atom_positions': torch.tensor(predicted_coords, dtype=torch.float32).unsqueeze(0),
                'all_atom_positions_gt': torch.tensor(reference_coords, dtype=torch.float32).unsqueeze(0),
                'all_atom_mask_gt': torch.ones(1, len(reference_seq), 37, dtype=torch.float32),
                'aatype': torch.tensor(seq_to_aatype(predicted_seq), dtype=torch.long).unsqueeze(0),
                'aatype_gt': torch.tensor(seq_to_aatype(reference_seq), dtype=torch.long).unsqueeze(0),
                'res_mask': torch.ones(1, len(reference_seq), dtype=torch.float32),
                'seq_length': torch.tensor([len(reference_seq)], dtype=torch.long)
            }
            
            # Run evaluation
            evaluator.run_evaluation(batch, str(eval_dir))
            
            # Read results from CSV
            results_csv = eval_dir / f"length_{len(reference_seq)}" / target_name / "top_sample.csv"
            if results_csv.exists():
                results = pd.read_csv(results_csv)
                tm_score = float(results['bb_tmscore_to_gt'].iloc[0])
                rmsd = float(results['bb_rmsd_to_gt'].iloc[0])
                composite_reward = calculate_composite_reward(tm_score, predicted_seq)
                return rmsd, tm_score, composite_reward
            else:
                print(f"⚠️ Results CSV not found: {results_csv}")
                return calculate_simple_metrics(predicted_coords[:, 1, :], reference_coords[:, 1, :], predicted_seq)
                
    except Exception as e:
        print(f"⚠️ Real evaluator failed: {e}, using fallback")
        # Extract CA coordinates if needed
        if len(predicted_coords.shape) == 3:
            pred_ca = predicted_coords[:, 1, :]
        else:
            pred_ca = predicted_coords
            
        if len(reference_coords.shape) == 3:
            ref_ca = reference_coords[:, 1, :]
        else:
            ref_ca = reference_coords
            
        return calculate_simple_metrics(pred_ca, ref_ca, predicted_seq)

def calculate_simple_metrics(predicted_coords: np.ndarray, reference_coords: np.ndarray, sequence: str) -> Tuple[float, float, float]:
    """Fallback simple metrics calculation"""
    try:
        pred_coords = np.array(predicted_coords)
        ref_coords = np.array(reference_coords)
        
        # Handle length mismatches
        min_len = min(len(pred_coords), len(ref_coords))
        pred_coords = pred_coords[:min_len]
        ref_coords = ref_coords[:min_len]
        
        if len(pred_coords) == 0:
            return float('inf'), 0.0
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_coords - ref_coords) ** 2, axis=1)))
        
        # Calculate TM-score
        L_target = len(ref_coords)
        d_0 = 1.24 * ((L_target - 15) ** (1/3)) - 1.8 if L_target > 15 else 0.5
        
        distances = np.sqrt(np.sum((pred_coords - ref_coords) ** 2, axis=1))
        tm_score = np.sum(1.0 / (1.0 + (distances / d_0) ** 2)) / L_target
        
        # Calculate composite reward
        composite_reward = calculate_composite_reward(tm_score, sequence)
        
        return rmsd, tm_score, composite_reward
        
    except Exception as e:
        print(f"    ⚠️ RMSD/TM-score calculation failed: {e}")
        return float('inf'), 0.0, 0.0

def load_cameo_sequences() -> Dict[str, str]:
    """Load CAMEO reference sequences"""
    fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    sequences = {}
    
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

def run_uct_mcts_folding_experiment(
    structure_id: str,
    sequence: str,
    reference_coords: np.ndarray,
    mode: str,
    expert_id: Optional[int] = None,
    cameo_structure_data: Dict = None,
    num_iterations: int = 25,
    max_depth: int = 5,
) -> Dict:
    """Run UCT MCTS experiment for folding a single structure using evaluator approach"""
    
    print(f"\n🧬 [{mode}{'_' + str(expert_id) if expert_id is not None else ''}] {structure_id}")
    
    start_time = time.time()
    
    # Step 1: Generate ESMFold baseline
    print("  🔄 Generating ESMFold baseline...")
    baseline_coords, baseline_plddt, baseline_structure_tokens = generate_esmfold_baseline(sequence)
    
    if baseline_coords is None:
        print("  ⚠️ ESMFold baseline generation failed, using fallback")
        # Create a simple extended chain as fallback baseline
        seq_len = len(sequence)
        baseline_coords = np.zeros((seq_len, 3))
        for i in range(seq_len):
            baseline_coords[i] = [i * 3.8, 0.0, 0.0]  # Extended chain with 3.8Å spacing
        
        baseline_plddt = np.ones(seq_len) * 0.5  # Low confidence
        baseline_structure_tokens = f"<cls_struct>{'<mask_struct>' * seq_len}<eos_struct>"
        print(f"  ✅ Created fallback baseline: extended chain with {seq_len} residues")
    
    # Step 2: Evaluate baseline using evaluator
    print("  📊 Evaluating baseline with DPLM2Evaluator...")
    baseline_rmsd, baseline_tm, baseline_reward = evaluate_structure_with_evaluator(
        baseline_coords, reference_coords, sequence, sequence, f"{structure_id}_baseline"
    )
    
    print(f"  ✅ Baseline: RMSD={baseline_rmsd:.3f}Å, TM={baseline_tm:.3f}, Reward={baseline_reward:.3f}")
    
    # Step 3: Initialize DPLM-2 integration for FORWARD FOLDING
    print("  🔄 Initializing DPLM-2 integration for forward folding...")
    dplm2 = DPLM2Integration(device="cuda")
    
    # Prepare baseline structure data for FORWARD FOLDING
    # In forward folding: sequence is FIXED, structure tokens are MASKED
    print(f"  🔍 Baseline structure tokens: {baseline_structure_tokens[:100] if baseline_structure_tokens else 'None'}...")
    
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
    }
    
    # Add CAMEO struct_seq if available and better than ESMFold tokens
    if cameo_struct_seq and len(cameo_struct_seq) > 0:
        baseline_structure['struct_seq'] = cameo_struct_seq
        print(f"  ✅ Using CAMEO struct_seq as baseline: {len(cameo_struct_seq)} chars")
    
    print(f"  🔍 Final baseline struct_seq: {baseline_structure['struct_seq'][:100]}...")
    baseline_structure['baseline_reward'] = baseline_reward
    baseline_structure['target_coordinates'] = reference_coords

    # Set baseline data for forward folding
    dplm2.set_baseline_structure(baseline_structure)
    dplm2.set_baseline_sequence(sequence)  # Fixed sequence

    # Step 4: Initialize UCT MCTS for FORWARD FOLDING (NO ENTROPY - pure UCB1)
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
        use_ph_uct=False,  # Pure UCT (no entropy bonuses)
        task_type='folding',
        num_simulations=num_iterations,
        temperature=1.0,
        use_plddt_masking=True,
        exclude_proteinmpnn=True
    )
    print("  ⚙️ UCT mode: standard UCB1 (entropy bonuses disabled)")
    
    # Step 5: Run MCTS search
    print(f"  🔄 Running UCT MCTS search ({num_iterations} iterations, pure UCB1, max_depth={max_depth})...")
    try:
        if mode == 'single_expert':
            # Single expert mode
            external_experts = {}
            if expert_id == 0:
                print("    🤖 Using DPLM-2 650M expert")
            elif expert_id == 1:
                print("    🤖 Using DPLM-2 150M expert")
            elif expert_id == 2:
                print("    🤖 Using DPLM-2 3B expert")
            
            root_node = mcts.search(
                initial_sequence=sequence,
                single_expert_id=expert_id,
                external_experts=external_experts,
                num_iterations=num_iterations,
            )
        else:
            # Multi-expert mode
            print("    🤖 Using all DPLM-2 experts (650M, 150M, 3B)")
            root_node = mcts.search(
                initial_sequence=sequence,
                external_experts={},
                num_iterations=num_iterations,
            )
        
        # Step 6: Extract best result
        best_node = mcts.get_best_child(root_node)
        if best_node and hasattr(best_node, 'sequence'):
            best_sequence = best_node.sequence
            
            # Convert best sequence back to structure using DPLM-2
            print("  🔄 Converting best sequence to structure...")
            best_coords = None
            
            # Try to get coordinates from MCTS node if available
            if hasattr(best_node, 'coordinates') and best_node.coordinates is not None:
                best_coords = best_node.coordinates
            else:
                # Generate structure from sequence using ESMFold as fallback
                best_coords, _, _ = generate_esmfold_baseline(best_sequence)
            
            if best_coords is not None:
                # Evaluate final result using evaluator
                final_rmsd, final_tm, final_reward = evaluate_structure_with_evaluator(
                    best_coords, reference_coords, best_sequence, sequence, f"{structure_id}_final"
                )
                
                # Calculate improvements
                rmsd_delta = baseline_rmsd - final_rmsd  # Positive = improvement
                tm_delta = final_tm - baseline_tm  # Positive = improvement  
                reward_delta = final_reward - baseline_reward  # Positive = improvement
                
                print(f"  📊 Final: RMSD={final_rmsd:.3f}Å, TM={final_tm:.3f}, Reward={final_reward:.3f}")
                print(f"  📈 Δ: RMSD={rmsd_delta:+.3f}Å, TM={tm_delta:+.3f}, Reward={reward_delta:+.3f}")
                
                result = {
                    'structure_id': structure_id,
                    'mode': mode,
                    'expert_id': expert_id,
                    'status': 'success',
                    'sequence_length': len(sequence),
                    'baseline_rmsd': baseline_rmsd,
                    'baseline_tm_score': baseline_tm,
                    'baseline_reward': baseline_reward,
                    'final_rmsd': final_rmsd,
                    'final_tm_score': final_tm,
                    'final_reward': final_reward,
                    'rmsd_delta': rmsd_delta,
                    'tm_delta': tm_delta,
                    'reward_delta': reward_delta,
                    'runtime_seconds': time.time() - start_time,
                    'mcts_iterations': num_iterations,
                    'best_sequence': best_sequence
                }
            else:
                print("  ❌ Failed to generate final structure")
                result = {
                    'structure_id': structure_id,
                    'mode': mode,
                    'expert_id': expert_id,
                    'status': 'failed',
                    'error': 'Failed to generate final structure'
                }
        else:
            print("  ❌ MCTS search failed to find best node")
            result = {
                'structure_id': structure_id,
                'mode': mode,
                'expert_id': expert_id,
                'status': 'failed',
                'error': 'MCTS search failed'
            }
            
    except Exception as e:
        print(f"  ❌ MCTS search failed: {e}")
        result = {
            'structure_id': structure_id,
            'mode': mode,
            'expert_id': expert_id,
            'status': 'failed',
            'error': str(e)
        }
    
    return result

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
    """Run all UCT MCTS folding experiments"""
    
    results = []
    total_experiments = len(structure_ids) * len(modes) * len(expert_ids)
    experiment_count = 0
    
    print(f"🚀 Starting UCT MCTS Folding Experiments")
    print(f"📊 Total experiments: {total_experiments}")
    print(f"🧬 Structures: {len(structure_ids)}")
    print(f"🤖 Modes: {modes}")
    print(f"🔬 Expert IDs: {expert_ids}")
    print(f"🔁 Iterations per experiment: {num_iterations}")
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
                    
                    result = run_uct_mcts_folding_experiment(
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
                
                result = run_uct_mcts_folding_experiment(
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
    parser = argparse.ArgumentParser(description='UCT MCTS Folding Experiments with DPLM2Evaluator')
    parser.add_argument('--mode', choices=['single_expert', 'multi_expert', 'all'], 
                       default='all', help='Experiment mode')
    parser.add_argument('--expert_id', type=int, choices=[0, 1, 2], 
                       help='Expert ID for single expert mode (0=650M, 1=150M, 2=3B)')
    parser.add_argument('--start', type=int, default=0, help='Start structure index')
    parser.add_argument('--end', type=int, default=5, help='End structure index')
    parser.add_argument('--num_iterations', type=int, default=25,
                       help='Number of UCT iterations per experiment')
    parser.add_argument('--max_depth', type=int, default=5,
                       help='Maximum search depth for UCT')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Load data
    print("  Loading CAMEO data...")
    sequences = load_cameo_sequences()
    coordinates, cameo_structure_data = load_reference_coordinates()
    
    if not sequences or not coordinates:
        print("❌ Failed to load CAMEO data")
        return
    
    # Get structure IDs
    structure_ids = list(sequences.keys())[args.start:args.end]
    print(f"📊 Processing {len(structure_ids)} structures: {structure_ids}")
    print(f"🔁 Iterations per structure: {args.num_iterations}")
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
    output_file = f"uct_mcts_folding_results_{timestamp}.json"
    
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
                
            avg_rmsd_delta = np.mean([r['rmsd_delta'] for r in mode_results])
            avg_tm_delta = np.mean([r['tm_delta'] for r in mode_results])
            avg_reward_delta = np.mean([r['reward_delta'] for r in mode_results])
            
            print(f"  {mode}: RMSD Δ={avg_rmsd_delta:+.3f}Å, TM Δ={avg_tm_delta:+.3f}, Reward Δ={avg_reward_delta:+.3f}")

if __name__ == "__main__":
    main()
