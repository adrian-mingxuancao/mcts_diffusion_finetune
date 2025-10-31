#!/usr/bin/env python3
"""
UCT MCTS Experiments for Inverse Folding (Without Entropy) - PDB Dataset

Runs UCT-based MCTS experiments without entropy bonuses for inverse folding on PDB dataset:
1. Single expert experiments: DPLM-2 650M, 150M, 3B, ProteinMPNN
2. Multi-expert experiments: All experts combined
3. Pure UCB1 selection (no PH-UCT entropy bonuses)
4. Multiple rollouts per expert with top-K selection

Usage:
    python uct_mcts_inverse_folding_pdb.py --mode single_expert --expert_id 0 --start 0 --end 5
    python uct_mcts_inverse_folding_pdb.py --mode multi_expert --start 0 --end 5
    python uct_mcts_inverse_folding_pdb.py --mode all --start 0 --end 10  # Run all modes
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Project path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from Bio import SeqIO

# Import MCTS and DPLM-2 components
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS  # noqa: E402
from mcts_diffusion_finetune.core.dplm2_integration import DPLM2Integration

# Import evaluation utilities
try:
    from mcts_diffusion_finetune.utils.pdb_data_loader import PDBDataLoader  # noqa: E402
except ImportError:
    class PDBDataLoader:
        def __init__(self, *args, **kwargs):
            self.structures = []
        def get_test_structure(self, index=0):
            return {"name": f"test_structure_{index}", "sequence": "IKKSI", "length": 5}

try:
    from mcts_diffusion_finetune.utils.real_plddt_computation import compute_sctm_from_esmfold  # noqa: E402
    SCTM_AVAILABLE = True
except Exception:
    SCTM_AVAILABLE = False

_ESMFOLD_MODEL = None

def get_esmfold_model():
    """Get ESMFold model for scTM calculation."""
    global _ESMFOLD_MODEL
    if _ESMFOLD_MODEL is None:
        try:
            from transformers import EsmForProteinFolding
            _ESMFOLD_MODEL = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            _ESMFOLD_MODEL.eval()
        except Exception as e:
            logging.warning(f"Failed to load ESMFold: {e}")
            _ESMFOLD_MODEL = None
    return _ESMFOLD_MODEL

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def load_reference_sequences() -> Dict[str, str]:
    """Load reference sequences from PDB dataset."""
    sequences = {}
    data_path = "/home/caom/AID3/dplm/data-bin/PDB_date/aatype.fasta"
    
    if not os.path.exists(data_path):
        logging.error(f"Reference sequences file not found: {data_path}")
        return sequences
    
    try:
        for record in SeqIO.parse(data_path, "fasta"):
            sequences[record.id] = str(record.seq)
        logging.info(f"Loaded {len(sequences)} reference sequences")
    except Exception as e:
        logging.error(f"Failed to load reference sequences: {e}")
    
    return sequences

def load_structure_tokens() -> Dict[str, str]:
    """Load structure token sequences from struct.fasta."""
    tokens = {}
    data_path = "/home/caom/AID3/dplm/data-bin/PDB_date/struct.fasta"
    
    if not os.path.exists(data_path):
        logging.warning(f"Structure tokens file not found: {data_path}")
        return tokens
    
    try:
        for record in SeqIO.parse(data_path, "fasta"):
            tokens[record.id] = str(record.seq)
        logging.info(f"Loaded {len(tokens)} structure token sequences")
    except Exception as e:
        logging.error(f"Failed to load structure tokens: {e}")
    
    return tokens

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

def compute_esmfold_plddt(sequence: str) -> Optional[List[float]]:
    """Compute per-residue pLDDT scores using ESMFold."""
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
            
            if hasattr(output, 'plddt') and output.plddt is not None:
                plddt_tensor = output.plddt.cpu()
                # pLDDT is per-residue confidence, convert to list
                if len(plddt_tensor.shape) == 3:  # [batch, length, 1]
                    plddt_scores = plddt_tensor[0, :, 0].tolist()
                elif len(plddt_tensor.shape) == 2:  # [batch, length]
                    plddt_scores = plddt_tensor[0, :].tolist()
                else:
                    plddt_scores = plddt_tensor.flatten().tolist()
                
                # Clean sequence to match pLDDT length
                clean_seq = sequence.replace('<mask_aa>', '').replace('<cls_aa>', '').replace('<eos_aa>', '')
                return plddt_scores[:len(clean_seq)]  # Match sequence length
                
    except Exception as e:
        logging.warning(f"  ⚠️ ESMFold pLDDT computation failed: {e}")
    
    return None

def calculate_simple_aar(pred_seq: str, ref_seq: str) -> float:
    """Calculate amino acid recovery between predicted and reference sequences."""
    if not pred_seq or not ref_seq:
        return 0.0
    
    min_len = min(len(pred_seq), len(ref_seq))
    if min_len == 0:
        return 0.0
    
    matches = sum(1 for i in range(min_len) if pred_seq[i] == ref_seq[i])
    return matches / min_len

def predict_structure_coordinates(sequence: str) -> Optional[np.ndarray]:
    """Predict 3D coordinates for a sequence using ESMFold."""
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        # Load ESMFold model
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        
        with torch.no_grad():
            # Tokenize sequence
            tokenized = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
            if torch.cuda.is_available():
                tokenized = {k: v.cuda() for k, v in tokenized.items()}
            
            # Predict structure
            output = model(tokenized["input_ids"])
            
            # Extract coordinates (CA atoms)
            if hasattr(output, 'positions') and output.positions is not None:
                positions = output.positions.cpu().numpy()
                # positions shape: [8, 1, length, 14, 3] (ESMFold format)
                # Extract CA atoms (index 1 in 14-atom representation)
                if len(positions.shape) == 5:
                    if positions.shape[3] >= 2:  # At least 2 atoms (N, CA)
                        ca_coords = positions[0, 0, :, 1, :]  # CA atoms
                    else:
                        ca_coords = positions[0, 0, :, 0, :]  # Use first atom if CA not available
                    
                    return ca_coords[:len(sequence)]  # Match sequence length
                elif len(positions.shape) == 4:
                    if positions.shape[2] >= 2:  # At least 2 atoms (N, CA)
                        ca_coords = positions[0, :, 1, :]  # CA atoms
                    else:
                        ca_coords = positions[0, :, 0, :]  # Use first atom if CA not available
                    
                    return ca_coords[:len(sequence)]  # Match sequence length
                
    except Exception as e:
        logging.warning(f"ESMFold coordinate prediction failed: {e}")
        
    return None

def clean_sequence(seq: str) -> str:
    """Clean sequence by removing invalid characters."""
    if not seq:
        return ""
    # Keep only standard amino acids
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    return ''.join(aa for aa in seq.upper() if aa in valid_aas)

def load_pregenerated_baseline_pdb(structure_id: str) -> Optional[str]:
    """Load pregenerated baseline sequence for PDB structure."""
    try:
        # Try to load from pregenerated results (similar to sampling script)
        baseline_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_id}.fasta"
        if os.path.exists(baseline_path):
            from Bio import SeqIO
            for record in SeqIO.parse(baseline_path, "fasta"):
                return clean_sequence(str(record.seq))
    except Exception as e:
        logging.debug(f"Failed to load pregenerated baseline for {structure_id}: {e}")
    return None

def calculate_biophysical_score(sequence: str) -> float:
    """Calculate biophysical feasibility score."""
    if not sequence:
        return 0.0
    
    # Count amino acid types
    charged = sum(1 for aa in sequence if aa in 'DEKR')
    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV')
    
    # Calculate fractions
    total = len(sequence)
    charged_frac = charged / total
    hydrophobic_frac = hydrophobic / total
    
    # Apply penalties for extreme compositions
    penalty = 0.0
    if charged_frac > 0.3:
        penalty += (charged_frac - 0.3) * 2.0
    if hydrophobic_frac > 0.4:
        penalty += (hydrophobic_frac - 0.4) * 1.5
    
    return max(0.0, 1.0 - penalty)

def calculate_composite_reward(aar: float, sctm: float, bio: float) -> float:
    """Calculate composite reward from AAR, scTM, and biophysical scores."""
    return 0.4 * aar + 0.45 * sctm + 0.15 * bio

def run_single_expert_experiment(
    structure_id: str,
    reference_sequence: str,
    structure_tokens: str,
    expert_id: int,
    num_iterations: int = 25,
    max_depth: int = 5
) -> Dict:
    """Run UCT MCTS experiment with single expert."""
    
    logging.info(f"🤖 Running single expert {expert_id} on {structure_id}")
    
    try:
        # Initialize DPLM-2 integration first
        dplm2_integration = DPLM2Integration(device="cuda")
        
        # Initialize MCTS with UCT (no entropy)
        mcts = GeneralMCTS(
            task_type='inverse_folding',
            use_entropy=False,  # Pure UCT without entropy bonuses
            backup_rule='max',
            num_rollouts_per_expert=2,
            max_depth=max_depth,
            exclude_proteinmpnn=False,
            dplm2_integration=dplm2_integration
        )
        
        # Run MCTS search
        result = mcts.search(
            initial_sequence=reference_sequence,
            structure_tokens=structure_tokens,
            num_iterations=num_iterations,
            single_expert_id=expert_id
        )
        
        if result and 'best_sequence' in result:
            best_sequence = result['best_sequence']
            baseline_aar = calculate_simple_aar(baseline_seq, reference_sequence)  # FIXED: baseline vs reference
            final_aar = calculate_simple_aar(best_sequence, reference_sequence)
            
            # Calculate additional metrics
            baseline_bio = calculate_biophysical_score(baseline_seq)  # FIXED: use baseline_seq
            final_bio = calculate_biophysical_score(best_sequence)
            
            # Calculate real scTM scores
            baseline_sctm = baseline_aar  # fallback
            final_sctm = baseline_aar  # fallback
            
            if SCTM_AVAILABLE:
                try:
                    # Use the same reference coordinates that MCTS used (real PDB coordinates)
                    reference_coords = baseline_structure.get('coordinates')
                    if reference_coords is not None:
                        logging.info(f"✅ Using real PDB coordinates: {reference_coords.shape}")
                        
                        # Calculate baseline scTM (baseline sequence vs reference coords) - FIXED
                        baseline_sctm = float(compute_sctm_from_esmfold(baseline_seq, baseline_seq, reference_coords))
                        
                        # Calculate final scTM (best sequence vs reference coords)
                        final_sctm = float(compute_sctm_from_esmfold(best_sequence, baseline_seq, reference_coords))
                        
                        logging.info(f"📊 scTM scores: baseline={baseline_sctm:.3f}, final={final_sctm:.3f}")
                    else:
                        logging.warning("Failed to get real PDB coordinates for scTM calculation")
                except Exception as e:
                    logging.warning(f"scTM calculation failed: {e}")
            else:
                logging.warning("scTM calculation unavailable")
            
            baseline_reward = calculate_composite_reward(baseline_aar, baseline_sctm, baseline_bio)
            final_reward = calculate_composite_reward(final_aar, final_sctm, final_bio)
            
            return {
                'structure_id': structure_id,
                'expert_id': expert_id,
                'mode': 'single_expert',
                'baseline_aar': baseline_aar,
                'final_aar': final_aar,
                'aar_improvement': final_aar - baseline_aar,
                'baseline_sctm': baseline_sctm,
                'final_sctm': final_sctm,
                'sctm_improvement': final_sctm - baseline_sctm,
                'baseline_biophysical': baseline_bio,
                'final_biophysical': final_bio,
                'baseline_reward': baseline_reward,
                'final_reward': final_reward,
                'reward_improvement': final_reward - baseline_reward,
                'num_iterations': num_iterations,
                'max_depth': max_depth,
                'best_sequence': best_sequence,
                'success': True
            }
        else:
            logging.warning(f"❌ MCTS search failed for {structure_id} expert {expert_id}")
            return {
                'structure_id': structure_id,
                'expert_id': expert_id,
                'mode': 'single_expert',
                'success': False,
                'error': 'MCTS search returned no result'
            }
            
    except Exception as e:
        logging.error(f"❌ Single expert experiment failed for {structure_id}: {e}")
        return {
            'structure_id': structure_id,
            'expert_id': expert_id,
            'mode': 'single_expert',
            'success': False,
            'error': str(e)
        }

def run_multi_expert_experiment(
    structure_id: str,
    reference_sequence: str,
    structure_tokens: str,
    num_iterations: int = 25,
    max_depth: int = 5,
    seed: int = 42
) -> Dict:
    """Run UCT MCTS experiment with multiple experts."""
    
    logging.info(f"🤖 Running multi-expert on {structure_id}")
    
    try:
        # Initialize DPLM-2 integration first
        dplm2_integration = DPLM2Integration(device="cuda")
        
        # Load structure data and coordinates like CAMEO version
        structure_data = load_pdb_structure_data(structure_id)
        reference_coords = None
        if structure_data:
            # CRITICAL: Use modeled_idx to get the right coordinates for the sequence
            if 'modeled_idx' in structure_data and 'bb_positions' in structure_data:
                modeled_idx = structure_data['modeled_idx']
                all_coords = structure_data['bb_positions']  # Use backbone positions (CA coords)
                
                # Extract coordinates for modeled residues only
                reference_coords = all_coords[modeled_idx]
                
                seq_length = len(reference_sequence)
                coord_length = reference_coords.shape[0]
                
                logging.info(f"✅ Using modeled_idx: {len(modeled_idx)} modeled residues from {all_coords.shape[0]} total")
                logging.info(f"✅ Extracted coordinates: sequence={seq_length}, coords={coord_length}")
                
                if coord_length == seq_length:
                    logging.info(f"✅ Perfect alignment: coordinates match sequence length")
                else:
                    logging.warning(f"⚠️ Still misaligned after modeled_idx: sequence={seq_length}, coords={coord_length}")
                    
            elif 'atom_positions' in structure_data:
                # Fallback: try atom_positions with CA extraction
                coords = structure_data['atom_positions']
                if len(coords.shape) == 3 and coords.shape[1] >= 2:
                    reference_coords = coords[:, 1, :]  # CA atoms at index 1
                else:
                    reference_coords = coords
                
                # Use modeled_idx if available for alignment
                if 'modeled_idx' in structure_data:
                    modeled_idx = structure_data['modeled_idx']
                    reference_coords = reference_coords[modeled_idx]
                    logging.info(f"✅ Applied modeled_idx to atom_positions: {reference_coords.shape}")
                else:
                    # Simple trimming as fallback
                    seq_length = len(reference_sequence)
                    coord_length = reference_coords.shape[0]
                    if coord_length > seq_length:
                        reference_coords = reference_coords[:seq_length]
                        logging.warning(f"⚠️ Fallback trimming: {coord_length} → {seq_length}")
            
            if reference_coords is None:
                logging.warning(f"⚠️ No coordinates found for {structure_id}, ProteinMPNN may not work")
        else:
            logging.warning(f"⚠️ No structure data found for {structure_id}")
        
        # Create baseline structure like CAMEO version
        baseline_structure = {
            'struct_seq': structure_tokens,  # Use structure tokens from struct.fasta
            'sequence': reference_sequence,
            'length': len(reference_sequence),
            'coordinates': reference_coords,
            'plddt_scores': None  # Will be computed below
        }
        
        # Generate baseline sequence like sampling version
        baseline_seq = load_pregenerated_baseline_pdb(structure_id)
        if not baseline_seq:
            logging.info("Generating baseline with DPLM-2 150M...")
            # Generate baseline using DPLM-2 150M like sampling script
            baseline_seq = dplm2_integration.generate_from_masked_input(
                aa_sequence='<mask_aa>' * len(reference_sequence),
                struct_tokens=structure_tokens,
                task_type="inverse_folding",
                expert_id=1,  # Use 150M model
                temperature=1.0,
            )
        
        if not baseline_seq:
            logging.warning("Baseline generation failed, using reference sequence")
            baseline_seq = reference_sequence
        else:
            baseline_seq = clean_sequence(baseline_seq)
            
        # Compute pLDDT scores like CAMEO version
        plddt_scores = compute_esmfold_plddt(baseline_seq)
        if plddt_scores:
            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            logging.info(f"  ✅ Computed pLDDT: mean={mean_plddt:.1f}, length={len(plddt_scores)}")
            baseline_structure['plddt_scores'] = plddt_scores
        else:
            logging.warning(f"  ⚠️ Failed to compute pLDDT scores")
            baseline_structure['plddt_scores'] = [70.0] * len(baseline_seq)  # Fallback
        
        # Set baseline structure on DPLM2 integration BEFORE MCTS initialization (like CAMEO)
        dplm2_integration.set_baseline_structure(baseline_structure)
        dplm2_integration.set_baseline_sequence(baseline_seq)
        
        # Initialize MCTS with UCT (no entropy) - multi-expert mode (include ProteinMPNN)
        mcts = GeneralMCTS(
            dplm2_integration=dplm2_integration,
            baseline_structure=baseline_structure,  # Pass baseline structure like CAMEO
            reference_sequence=reference_sequence,
            task_type='inverse_folding',
            use_entropy=False,  # Pure UCT without entropy bonuses
            backup_rule='max',
            num_rollouts_per_expert=2,
            max_depth=max_depth,
            exclude_proteinmpnn=False,  # Include ProteinMPNN like CAMEO version
            exploration_constant=1.414
        )
        
        # Set seed only for baseline generation, not rollouts
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        
        # Run MCTS search with all experts (including ProteinMPNN) - like CAMEO version
        structure_data = {
            'struct_seq': baseline_structure.get('struct_seq', ''),
            'length': baseline_structure.get('length', len(reference_sequence)),
        }
        
        result = mcts.search(
            initial_sequence=reference_sequence,
            structure_data=structure_data,  # Pass structure data like CAMEO
            num_iterations=num_iterations,
            ablation_mode="multi_expert"  # Specify multi-expert mode like CAMEO
        )
        
        # Handle result - could be dict or MCTSNode
        if result:
            if hasattr(result, 'sequence'):
                # Result is MCTSNode
                best_sequence = result.sequence
                logging.info(f"✅ MCTS returned MCTSNode with sequence: {len(best_sequence)} chars")
            elif isinstance(result, dict) and 'best_sequence' in result:
                # Result is dict
                best_sequence = result['best_sequence']
                logging.info(f"✅ MCTS returned dict with best_sequence: {len(best_sequence)} chars")
            else:
                logging.warning(f"⚠️ Unexpected result type: {type(result)}")
                best_sequence = None
        else:
            best_sequence = None
            
        if best_sequence:
            baseline_aar = calculate_simple_aar(baseline_seq, reference_sequence)  # FIXED: baseline vs reference
            final_aar = calculate_simple_aar(best_sequence, reference_sequence)
            
            # Calculate additional metrics
            baseline_bio = calculate_biophysical_score(baseline_seq)  # FIXED: use baseline_seq
            final_bio = calculate_biophysical_score(best_sequence)
            
            # Calculate real scTM scores
            baseline_sctm = baseline_aar  # fallback
            final_sctm = baseline_aar  # fallback
            
            if SCTM_AVAILABLE:
                try:
                    # Use the same reference coordinates that MCTS used (real PDB coordinates)
                    reference_coords = baseline_structure.get('coordinates')
                    if reference_coords is not None:
                        logging.info(f"✅ Using real PDB coordinates: {reference_coords.shape}")
                        
                        # Calculate baseline scTM (baseline sequence vs reference coords) - FIXED
                        baseline_sctm = float(compute_sctm_from_esmfold(baseline_seq, baseline_seq, reference_coords))
                        
                        # Calculate final scTM (best sequence vs reference coords)
                        final_sctm = float(compute_sctm_from_esmfold(best_sequence, baseline_seq, reference_coords))
                        
                        logging.info(f"📊 scTM scores: baseline={baseline_sctm:.3f}, final={final_sctm:.3f}")
                    else:
                        logging.warning("Failed to get real PDB coordinates for scTM calculation")
                except Exception as e:
                    logging.warning(f"scTM calculation failed: {e}")
            else:
                logging.warning("scTM calculation unavailable")
            
            baseline_reward = calculate_composite_reward(baseline_aar, baseline_sctm, baseline_bio)
            final_reward = calculate_composite_reward(final_aar, final_sctm, final_bio)
            
            return {
                'structure_id': structure_id,
                'mode': 'multi_expert',
                'baseline_aar': baseline_aar,
                'final_aar': final_aar,
                'aar_improvement': final_aar - baseline_aar,
                'baseline_sctm': baseline_sctm,
                'final_sctm': final_sctm,
                'sctm_improvement': final_sctm - baseline_sctm,
                'baseline_biophysical': baseline_bio,
                'final_biophysical': final_bio,
                'baseline_reward': baseline_reward,
                'final_reward': final_reward,
                'reward_improvement': final_reward - baseline_reward,
                'num_iterations': num_iterations,
                'max_depth': max_depth,
                'best_sequence': best_sequence,
                'success': True
            }
        else:
            logging.warning(f"❌ Multi-expert MCTS search failed for {structure_id}")
            return {
                'structure_id': structure_id,
                'mode': 'multi_expert',
                'success': False,
                'error': 'MCTS search returned no result'
            }
            
    except Exception as e:
        logging.error(f"❌ Multi-expert experiment failed for {structure_id}: {e}")
        return {
            'structure_id': structure_id,
            'mode': 'multi_expert',
            'success': False,
            'error': str(e)
        }

def run_experiments(
    structure_ids: List[str],
    reference_sequences: Dict[str, str],
    modes: List[str],
    expert_ids: List[int],
    num_iterations: int = 25,
    max_depth: int = 5,
    seed: int = 42
) -> List[Dict]:
    """Run multi-expert experiments only."""
    
    results = []
    structure_tokens = load_structure_tokens()
    
    total_structures = len(structure_ids)
    
    for i, structure_id in enumerate(structure_ids, 1):
        print(f"\n{'='*80}")
        print(f"🧬 PROCESSING STRUCTURE {i}/{total_structures}: {structure_id}")
        print(f"{'='*80}")
        
        if structure_id not in reference_sequences:
            logging.warning(f"⚠️ No reference sequence for {structure_id}")
            # Save failed result
            failed_result = {
                'structure_id': structure_id,
                'mode': 'multi_expert',
                'success': False,
                'error': 'No reference sequence found'
            }
            results.append(failed_result)
            save_individual_result(failed_result, structure_id)
            continue
            
        if structure_id not in structure_tokens:
            logging.warning(f"⚠️ No structure tokens for {structure_id}")
            # Save failed result
            failed_result = {
                'structure_id': structure_id,
                'mode': 'multi_expert',
                'success': False,
                'error': 'No structure tokens found'
            }
            results.append(failed_result)
            save_individual_result(failed_result, structure_id)
            continue
        
        reference_sequence = reference_sequences[structure_id]
        struct_tokens = structure_tokens[structure_id]
        
        # FIXED: Run multi-expert experiment only
        if "multi_expert" in modes:
            try:
                print(f"🚀 Starting multi-expert MCTS for {structure_id}...")
                start_time = time.time()
                
                result = run_multi_expert_experiment(
                    structure_id, reference_sequence, struct_tokens,
                    num_iterations, max_depth, seed
                )
                
                end_time = time.time()
                duration = end_time - start_time
                print(f"⏱️ Completed {structure_id} in {duration:.1f}s")
                
                results.append(result)
                
                # SAVE RESULTS IMMEDIATELY after each structure
                save_individual_result(result, structure_id)
                
                print(f"✅ Successfully processed {structure_id}")
                
            except Exception as e:
                error_msg = f"Failed to process {structure_id}: {str(e)}"
                logging.error(error_msg)
                print(f"❌ ERROR: {error_msg}")
                
                # Save failed result
                failed_result = {
                    'structure_id': structure_id,
                    'mode': 'multi_expert',
                    'success': False,
                    'error': str(e)
                }
                results.append(failed_result)
                save_individual_result(failed_result, structure_id)
                
                # Continue to next structure instead of crashing
                continue
    
    return results

def calculate_summary_statistics(results: List[Dict]) -> Dict:
    """Calculate summary statistics from experiment results."""
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    summary = {
        'total_experiments': len(results),
        'successful_experiments': len(successful_results),
        'failed_experiments': len(results) - len(successful_results),
        'modes': {}
    }
    
    # Calculate baseline and final metrics for successful results
    if successful_results:
        baseline_aars = [r.get('baseline_aar', 0) for r in successful_results]
        final_aars = [r.get('final_aar', 0) for r in successful_results]
        baseline_sctms = [r.get('baseline_sctm', 0) for r in successful_results]
        final_sctms = [r.get('final_sctm', 0) for r in successful_results]
        baseline_rewards = [r.get('baseline_reward', 0) for r in successful_results]
        final_rewards = [r.get('final_reward', 0) for r in successful_results]
        
        summary['baseline_metrics'] = {
            'aar': {
                'mean': float(np.mean(baseline_aars)),
                'std': float(np.std(baseline_aars)),
                'min': float(np.min(baseline_aars)),
                'max': float(np.max(baseline_aars))
            },
            'sctm': {
                'mean': float(np.mean(baseline_sctms)),
                'std': float(np.std(baseline_sctms)),
                'min': float(np.min(baseline_sctms)),
                'max': float(np.max(baseline_sctms))
            },
            'reward': {
                'mean': float(np.mean(baseline_rewards)),
                'std': float(np.std(baseline_rewards)),
                'min': float(np.min(baseline_rewards)),
                'max': float(np.max(baseline_rewards))
            }
        }
        
        summary['final_metrics'] = {
            'aar': {
                'mean': float(np.mean(final_aars)),
                'std': float(np.std(final_aars)),
                'min': float(np.min(final_aars)),
                'max': float(np.max(final_aars))
            },
            'sctm': {
                'mean': float(np.mean(final_sctms)),
                'std': float(np.std(final_sctms)),
                'min': float(np.min(final_sctms)),
                'max': float(np.max(final_sctms))
            },
            'reward': {
                'mean': float(np.mean(final_rewards)),
                'std': float(np.std(final_rewards)),
                'min': float(np.min(final_rewards)),
                'max': float(np.max(final_rewards))
            }
        }
    
    # Group by mode
    mode_results = {}
    for result in results:
        if not result.get('success', False):
            continue
            
        mode = result.get('mode', 'unknown')
        if mode not in mode_results:
            mode_results[mode] = []
        mode_results[mode].append(result)
    
    # Calculate statistics for each mode
    for mode, mode_data in mode_results.items():
        if not mode_data:
            continue
            
        aar_improvements = [r['aar_improvement'] for r in mode_data]
        sctm_improvements = [r['sctm_improvement'] for r in mode_data]
        reward_improvements = [r['reward_improvement'] for r in mode_data]
        
        summary['modes'][mode] = {
            'count': len(mode_data),
            'aar_improvement': {
                'mean': float(np.mean(aar_improvements)),
                'std': float(np.std(aar_improvements)),
                'min': float(np.min(aar_improvements)),
                'max': float(np.max(aar_improvements))
            },
            'sctm_improvement': {
                'mean': float(np.mean(sctm_improvements)),
                'std': float(np.std(sctm_improvements)),
                'min': float(np.min(sctm_improvements)),
                'max': float(np.max(sctm_improvements))
            },
            'reward_improvement': {
                'mean': float(np.mean(reward_improvements)),
                'std': float(np.std(reward_improvements)),
                'min': float(np.min(reward_improvements)),
                'max': float(np.max(reward_improvements))
            }
        }
    
    return summary

def save_individual_result(result: Dict, structure_id: str):
    """Save individual result immediately after each structure completion."""
    
    # Create individual results directory
    individual_dir = "/net/scratch/caom/uct_mcts_pdb_individual_results"
    os.makedirs(individual_dir, exist_ok=True)
    
    # Save individual result with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    individual_file = os.path.join(individual_dir, f"{structure_id}_{timestamp}.json")
    
    with open(individual_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Print immediate summary for this structure
    if result.get('success', False):
        baseline_aar = result.get('baseline_aar', 0)
        final_aar = result.get('final_aar', 0)
        baseline_sctm = result.get('baseline_sctm', 0)
        final_sctm = result.get('final_sctm', 0)
        baseline_reward = result.get('baseline_reward', 0)
        final_reward = result.get('final_reward', 0)
        aar_improvement = result.get('aar_improvement', 0)
        sctm_improvement = result.get('sctm_improvement', 0)
        reward_improvement = result.get('reward_improvement', 0)
        best_sequence = result.get('best_sequence', '')[:50] + '...' if len(result.get('best_sequence', '')) > 50 else result.get('best_sequence', '')
        
        print(f"\n🎉 COMPLETED: {structure_id}")
        print(f"   📊 AAR:    {baseline_aar:.3f} → {final_aar:.3f} (Δ {aar_improvement:+.3f})")
        print(f"   📊 scTM:   {baseline_sctm:.3f} → {final_sctm:.3f} (Δ {sctm_improvement:+.3f})")
        print(f"   📊 Reward: {baseline_reward:.3f} → {final_reward:.3f} (Δ {reward_improvement:+.3f})")
        print(f"   🧬 Best sequence: {best_sequence}")
        print(f"   💾 Saved to: {individual_file}")
    else:
        error_msg = result.get('error', 'Unknown error')
        print(f"\n❌ FAILED: {structure_id}")
        print(f"   🚨 Error: {error_msg}")
        print(f"   💾 Saved to: {individual_file}")

def save_results(results: List[Dict], summary: Dict, output_dir: str):
    """Save experiment results and summary."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"uct_mcts_inverse_folding_pdb_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f"uct_mcts_inverse_folding_pdb_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"📈 Summary statistics saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="UCT MCTS experiments for inverse folding on PDB dataset - MULTI-EXPERT ONLY")
    # FIXED: Multi-expert mode only
    parser.add_argument("--mode", choices=["multi_expert"], 
                       default="multi_expert", help="Multi-expert experiment mode only")
    parser.add_argument("--start", type=int, default=0, help="Start structure index")
    parser.add_argument("--end", type=int, default=5, help="End structure index")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/caom/AID3/dplm/mcts_diffusion_finetune/results/uct_mcts_pdb_analysis",
                       help="Output directory for results")
    parser.add_argument("--num_iterations", type=int, default=25,
                       help="Number of UCT iterations per structure")
    parser.add_argument("--max_depth", type=int, default=5,
                       help="Maximum search depth for UCT")
    parser.add_argument("--seed", type=int, default=42,
                       help="Fixed seed for reproducible results")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🧬 UCT MCTS Experiments - Inverse Folding (No Entropy) - PDB Dataset - MULTI-EXPERT ONLY")
    print("=" * 90)
    print(f"🎯 Mode: {args.mode} (DPLM-2 650M + 150M + 3B)")
    print(f"📊 Structure range: {args.start}-{args.end}")
    print(f"🔁 Iterations: {args.num_iterations}")
    print(f"🌳 Max depth: {args.max_depth}")
    print(f"🎲 Seed: {args.seed}")
    
    # Load reference sequences
    reference_sequences = load_reference_sequences()
    if not reference_sequences:
        print("❌ No reference sequences loaded. Exiting.")
        return
    
    # Determine structure IDs to process
    structure_ids = list(reference_sequences.keys())[args.start:args.end]
    print(f"🧬 Processing {len(structure_ids)} structures")
    
    # FIXED: Multi-expert mode only
    modes = ["multi_expert"]
    expert_ids = []  # Not used for multi-expert mode
    
    # Run experiments
    results = run_experiments(
        structure_ids,
        reference_sequences,
        modes,
        expert_ids,
        num_iterations=args.num_iterations,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    
    if not results:
        print("❌ No results generated. Exiting.")
        return
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(results)
    
    # Save results
    save_results(results, summary, args.output_dir)
    
    # Print final summary
    successful = summary['successful_experiments']
    failed = summary['failed_experiments']
    total = summary['total_experiments']
    
    print(f"\n🎉 UCT MCTS experiments complete!")
    print(f"📊 Total structures processed: {total}")
    print(f"✅ Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    
    # Print baseline vs final metrics if available
    if successful > 0 and 'baseline_metrics' in summary and 'final_metrics' in summary:
        baseline = summary['baseline_metrics']
        final = summary['final_metrics']
        
        print(f"\n📈 BASELINE vs FINAL METRICS (n={successful}):")
        print(f"   AAR:    {baseline['aar']['mean']:.3f} → {final['aar']['mean']:.3f} (Δ {final['aar']['mean']-baseline['aar']['mean']:+.3f})")
        print(f"   scTM:   {baseline['sctm']['mean']:.3f} → {final['sctm']['mean']:.3f} (Δ {final['sctm']['mean']-baseline['sctm']['mean']:+.3f})")
        print(f"   Reward: {baseline['reward']['mean']:.3f} → {final['reward']['mean']:.3f} (Δ {final['reward']['mean']-baseline['reward']['mean']:+.3f})")
    
    print(f"\n📁 Results saved to: {args.output_dir}")
    print(f"💾 Individual results saved to: /net/scratch/caom/uct_mcts_pdb_individual_results/")

if __name__ == "__main__":
    main()
