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
import pickle
import glob

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

def load_cached_baselines(cache_dir="/net/scratch/caom/cached_baselines"):
    """
    Load cached baselines from pickle files
    """
    cached_baselines = {}
    
    if not os.path.exists(cache_dir):
        print(f"⚠️ Cache directory not found: {cache_dir}")
        return cached_baselines
    
    # Find all cached baseline files
    cache_files = glob.glob(os.path.join(cache_dir, "cached_baselines_*.pkl"))
    
    if not cache_files:
        print(f"⚠️ No cached baseline files found in: {cache_dir}")
        return cached_baselines
    
    print(f"🔄 Loading cached baselines from {len(cache_files)} files...")
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                file_baselines = pickle.load(f)
                cached_baselines.update(file_baselines)
                print(f"  ✅ Loaded {len(file_baselines)} baselines from {os.path.basename(cache_file)}")
        except Exception as e:
            print(f"  ⚠️ Failed to load {cache_file}: {e}")
    
    print(f"✅ Total cached baselines loaded: {len(cached_baselines)}")
    return cached_baselines

def generate_dplm2_baseline_sequence(structure, dplm2, structure_idx=None, fixed_baseline=True, seed_offset=0):
    # **FIXED BASELINE**: Set deterministic seed for baseline generation only
    import torch
    import numpy as np
    import random
    
    # Save current random states
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    
    if fixed_baseline:
        # Create deterministic seed based on structure properties (consistent across runs)
        structure_name = structure.get('name', '').replace('CAMEO ', '')
        pdb_id = structure.get('pdb_id', '')
        chain_id = structure.get('chain_id', '')
        if pdb_id and chain_id:
            seed_string = f"{pdb_id}_{chain_id}"
        else:
            seed_string = structure_name
        
        # Convert string to deterministic seed
        baseline_seed = (hash(seed_string) % (2**31)) + seed_offset  # Ensure positive 32-bit int
        
        print(f"  🎯 Using fixed baseline seed {baseline_seed} for {seed_string} (offset: {seed_offset})")
        
        # Set deterministic seed for baseline generation
        torch.manual_seed(baseline_seed)
        np.random.seed(baseline_seed)
        random.seed(baseline_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(baseline_seed)
            torch.cuda.manual_seed_all(baseline_seed)
    else:
        print(f"  🎲 Using random baseline generation (no fixed seed)")
    
    try:
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
            struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
            pdb_id = structure.get('pdb_id', '')
            chain_id = structure.get('chain_id', '')
            structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
            from utils.struct_loader import load_struct_seq_from_fasta
            struct_seq = load_struct_seq_from_fasta(struct_fasta_path, structure_name)
            baseline_structure['struct_seq'] = struct_seq
        
        # Try inverse folding generation first, fallback to pregenerated if it fails
        pdb_id = structure.get('pdb_id', '')
        chain_id = structure.get('chain_id', '')
        structure_name = f"{pdb_id}_{chain_id}" if pdb_id and chain_id else structure.get('name', '').replace('CAMEO ', '')
        
        # First attempt: Try direct inverse folding generation
        print(f"  🔄 Attempting direct inverse folding generation...")
        # Use the correct baseline generation method
        baseline_seq = dplm2.generate_baseline_sequence(
            structure_tokens=baseline_structure['struct_seq'], 
            target_length=target_length, 
            expert_id=1
        )
        if baseline_seq and len(baseline_seq) > 0:
            print(f"  ✅ Generated baseline sequence: {len(baseline_seq)} chars")
            
            # Generate ESMFold pLDDT scores for the baseline sequence
            try:
                print(f"  🔄 Computing ESMFold pLDDT for baseline sequence...")
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
    
    finally:
        # **RESTORE RANDOM STATES**: Restore randomness for MCTS rollouts
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        print(f"  🔄 Restored random states for MCTS rollouts")

def run_one_structure(structure, structure_name, dplm2, correct_reference_sequences, ablation_mode, single_expert_id=None, structure_idx=None, loader=None, cached_baselines=None, fixed_baseline=True, seed_offset=0, max_depth=5, num_iterations=25, use_plddt_masking=True, device="cuda"):
    print(f"\n🧬 [{ablation_mode}{'' if single_expert_id is None else f'_{single_expert_id}'}] {structure_name}")
    ref_id = structure_name.replace('CAMEO ', '')
    ref_seq = correct_reference_sequences.get(ref_id)
    if not ref_seq:
        print(f"  ❌ no reference: {ref_id}")
        return None

    # **LOAD PREGENERATED BASELINE**: Use pregenerated DPLM-2 150M sequences as fixed baselines
    print(f"  🎯 Loading pregenerated baseline for {ref_id}")
    
    pregenerated_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{ref_id}.fasta"
    baseline_seq = None
    
    try:
        from Bio import SeqIO
        
        def clean_aa(seq_str: str) -> str:
            valid = set("ACDEFGHIKLMNPQRSTVWY")
            return "".join(c for c in str(seq_str).upper() if c in valid)
        
        if os.path.exists(pregenerated_path):
            for record in SeqIO.parse(pregenerated_path, "fasta"):
                baseline_seq = clean_aa(record.seq)
                print(f"  ✅ Loaded pregenerated baseline: {len(baseline_seq)} chars")
                break
        else:
            print(f"  ❌ Pregenerated baseline not found: {pregenerated_path}")
            print(f"  🔄 Fallback: Generating new baseline on-the-fly")
            baseline_seq, baseline_struct = generate_dplm2_baseline_sequence(structure, dplm2, structure_idx, fixed_baseline, seed_offset)
            if not baseline_seq:
                print("  ❌ Baseline generation failed")
                return None
            
    except Exception as e:
        print(f"  ❌ Failed to load pregenerated baseline: {e}")
        return None
    
    # Use the loaded structure data (contains struct_seq from struct.fasta)
    baseline_struct = structure.copy()
    
    # **COMPUTE ESMFold pLDDT**: Generate pLDDT scores for the pregenerated baseline sequence
    print(f"  🔄 Computing ESMFold pLDDT for pregenerated baseline sequence...")
    try:
        import torch
        from transformers import EsmTokenizer, EsmForProteinFolding
        
        # Load ESMFold model
        tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
        esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        target_device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        esmfold_model = esmfold_model.to(target_device)
        esmfold_model.eval()
        
        with torch.no_grad():
            tokenized = tokenizer(baseline_seq, return_tensors="pt", add_special_tokens=False)
            tokenized = {k: v.to(target_device) for k, v in tokenized.items()}
            output = esmfold_model(tokenized['input_ids'])
            
            if hasattr(output, 'plddt') and output.plddt is not None:
                plddt_tensor = output.plddt[0].cpu().numpy()  # [L, 37]
                
                # Use Cα atom confidence (atom index 1)
                if len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                    plddt_scores = plddt_tensor[:, 1].tolist()  # Cα atom confidence
                else:
                    plddt_scores = plddt_tensor.mean(axis=1).tolist() if len(plddt_tensor.shape) == 2 else plddt_tensor.tolist()
                
                # Add pLDDT scores to baseline structure
                baseline_struct['plddt_scores'] = plddt_scores
                print(f"  ✅ Added ESMFold pLDDT: mean={sum(plddt_scores)/len(plddt_scores):.1f}, length={len(plddt_scores)}")
            else:
                print(f"  ⚠️ ESMFold pLDDT not available")
        
        # Clean up
        del esmfold_model
        if target_device.type == "cuda":
            torch.cuda.empty_cache()
        
    except Exception as plddt_e:
        print(f"  ⚠️ ESMFold pLDDT generation failed: {plddt_e}")
    
    baseline_aar = calculate_simple_aar(baseline_seq, ref_seq)
    print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")

    # **COORDINATES**: Use cached coordinates if available, otherwise load from .pkl
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
    
    # **CRITICAL FIX**: Initialize MCTS with external experts and ablation mode
    # Load external experts if needed
    external_experts = []
    if ablation_mode in ["single_expert", "multi_expert"]:
        try:
            from external_models.real_direct_models import create_real_external_experts
            external_experts = create_real_external_experts()
            print(f"✅ Loaded {len(external_experts)} external experts: {[e.get_name() for e in external_experts]}")
        except Exception as e:
            print(f"⚠️ Failed to load external experts: {e}")
            external_experts = []
    
    # Add external experts and single_expert_id to baseline structure for backup MCTS
    if external_experts:
        baseline_struct['external_experts'] = external_experts
    if single_expert_id is not None:
        baseline_struct['single_expert_id'] = single_expert_id
    
    # Initialize CORRECTED MCTS with proper multi-expert rollouts and entropy fix
    mcts = GeneralMCTS(
        dplm2_integration=dplm2,
        baseline_structure=baseline_struct,
        reference_sequence=ref_seq,
        max_depth=max_depth,
        exploration_constant=1.414,
        ablation_mode=ablation_mode,
        single_expert_id=single_expert_id,
        external_experts=external_experts,
        num_rollouts_per_expert=2,  # N rollouts per expert
        top_k_candidates=2,  # Top-K selection
        task_type="inverse_folding",
        num_simulations=num_iterations,
        temperature=1.0,
        use_plddt_masking=use_plddt_masking
    )
    
    # **CRITICAL FIX**: Set baseline structure AND sequence in DPLM2Integration
    dplm2.set_baseline_structure(baseline_struct)
    dplm2.set_baseline_sequence(baseline_seq)  # This ensures MCTS uses the pregenerated baseline
    
    # Configure ablation mode - this will be handled in the search method
    if ablation_mode == "random_no_expert":
        print("🎲 Configuring random no expert mode")
    elif ablation_mode == "single_expert":
        expert_id = int(single_expert_id or 0)
        print(f"🎯 Configuring single expert mode with expert {expert_id}")
    else:
        print("🤖 Configuring multi-expert mode")

    # Define biophysical score calculation function
    def calculate_biophysical_score(sequence):
        if not sequence:
            return 0.0
        # Biophysical penalties for extreme compositions
        hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)
        charged = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
        polar = sum(1 for aa in sequence if aa in 'NQSTY') / len(sequence)
        
        # Penalties for extreme distributions
        charge_penalty = max(0, charged - 0.3) * 2  # Penalty if >30% charged
        hydrophobic_penalty = max(0, hydrophobic - 0.4) * 2  # Penalty if >40% hydrophobic
        
        # Base score with penalties
        base_score = 1.0 - charge_penalty - hydrophobic_penalty
        return max(0.0, min(1.0, base_score))

    # Baseline reward using composite formula (same as MCTS rollouts)
    try:
        # Calculate baseline scTM if not already calculated
        if baseline_sctm is None and baseline_struct.get('coordinates') is not None:
            from utils.sctm_calculation import calculate_sctm_score
            reference_coords = baseline_struct.get('coordinates')
            baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
        
        # Calculate baseline biophysical score
        baseline_biophysical = calculate_biophysical_score(baseline_seq)
        
        # Use composite reward formula (same as MCTS rollouts)
        if baseline_sctm is not None:
            baseline_reward = 0.4 * baseline_aar + 0.45 * baseline_sctm + 0.15 * baseline_biophysical
            print(f"  📈 Baseline reward breakdown: AAR={baseline_aar:.3f}, scTM={baseline_sctm:.3f}, B={baseline_biophysical:.3f} → R={baseline_reward:.3f}")
        else:
            # Fallback to AAR-only if scTM unavailable
            baseline_reward = baseline_aar
            print(f"  📈 Baseline reward (AAR-only, no coords): {baseline_reward:.3f}")
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
    
    root_node = mcts.search(
        initial_sequence=baseline_seq,
        reference_sequence=ref_seq,  # Pass reference sequence for AAR calculation
        num_iterations=num_iterations,
        structure_data=structure_data  # Use corrected MCTS parameters
    )
    elapsed = time.time() - t0

    # Process results from search
    if root_node is None:
        print("  ❌ MCTS search failed")
        return None
    
    # Find best sequence from the tree
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
    
    # Compute scTM scores using ESMFold prediction vs reference structure (coordinates already loaded)
    baseline_sctm, final_sctm = None, None
    try:
        from utils.sctm_calculation import calculate_sctm_score
        
        # Use coordinates that were already loaded before MCTS
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

    # Calculate final reward using composite formula (same as baseline and MCTS rollouts)
    try:
        # Calculate final biophysical score
        final_biophysical = calculate_biophysical_score(best_seq)
        
        # Use composite reward formula (same as baseline and MCTS rollouts)
        if final_sctm is not None:
            final_reward = 0.4 * best_aar + 0.45 * final_sctm + 0.15 * final_biophysical
            print(f"  🎯 Final reward breakdown: AAR={best_aar:.3f}, scTM={final_sctm:.3f}, B={final_biophysical:.3f} → R={final_reward:.3f}")
        else:
            # Fallback to AAR-only if scTM unavailable
            final_reward = best_aar
            print(f"  🎯 Final reward (AAR-only, no coords): {final_reward:.3f}")
    except Exception as e:
        print(f"  ⚠️ Final reward computation failed: {e}")
        final_reward = best_aar

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
    # Small per-structure summary (AAR, scTM, Reward)
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
    
    # Print incremental summary for real-time tracking (parseable format)
    print(f"  📈 Summary: {structure_name},{structure['length']},{baseline_aar:.3f},{best_aar:.3f},"
          f"{best_aar - baseline_aar:+.3f},{baseline_reward:.3f},{final_reward:.3f},"
          f"{final_reward - baseline_reward:+.3f},"
          f"{baseline_sctm if baseline_sctm is not None else 0.0:.3f},"
          f"{final_sctm if final_sctm is not None else 0.0:.3f},"
          f"{(final_sctm - baseline_sctm) if (baseline_sctm is not None and final_sctm is not None) else 0.0:+.3f},"
          f"True,{elapsed:.1f}")
    
    return out

def main():
    print("🧬 MCTS Ablation Study - CAMEO 2022 Evaluation (with Cached Baselines)")
    print("=" * 70)
    
    # CLI: structure range + method switching
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("start", nargs='?', type=int, default=0, help="start index (inclusive)")
    parser.add_argument("end", nargs='?', type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--mode", choices=["random_no_expert","single_expert","multi_expert","all"], default="all")
    parser.add_argument("--single_expert_id", type=int, default=None, help="single expert id (0/1/2/3) - 3=ProteinMPNN")
    parser.add_argument("--cache_dir", type=str, default="/net/scratch/caom/cached_baselines", help="Directory containing cached baselines")
    parser.add_argument("--fixed_baseline", action="store_true", default=True, help="Use fixed seed for baseline generation (fallback only)")
    parser.add_argument("--baseline_seed_offset", type=int, default=0, help="Offset for baseline seed (fallback only)")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum MCTS depth (default: 5)")
    parser.add_argument("--num_iterations", type=int, default=25, help="Number of MCTS iterations per structure (default: 25)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for DPLM-2/ESMFold inference")
    parser.add_argument("--disable_plddt_masking", action="store_true", help="Disable progressive pLDDT masking")
    parser.add_argument("--dry_run", action="store_true", help="Print configuration and exit without running")
    args = parser.parse_args()

    start_idx = args.start
    end_idx = args.end
    print(f"🎯 Structure range: {start_idx}-{end_idx if end_idx is not None else 'end'} | Mode: {args.mode}")
    print(f"🔍 Debug: args.single_expert_id = {args.single_expert_id}")
    print(f"💾 Cache directory: {args.cache_dir}")
    print(f"🎯 Fixed baseline: {args.fixed_baseline} | Seed offset: {args.baseline_seed_offset}")
    print(f"🌲 MCTS max_depth = {args.max_depth}, num_iterations = {args.num_iterations}")
    print(f"🖥️ Device: {args.device}")
    print(f"🎭 pLDDT masking: {'disabled' if args.disable_plddt_masking else 'enabled'}")
    
    if args.dry_run:
        print("🛠️ Dry run requested; exiting before model initialization.")
        return
    
    # **SIMPLIFIED BASELINE APPROACH**: Use pregenerated FASTA sequences directly
    print(f"✅ Using pregenerated DPLM-2 150M baselines from generation-results/")
    cached_baselines = None  # Not using cached baselines - using pregenerated FASTA directly
    
    # Load data using same pattern as test_mcts_with_real_data.py
    loader = CAMEODataLoader()
    refs = load_correct_reference_sequences()
    
    # Initialize DPLM-2 (using default parameters: max_iter=150, temperature=1.0)
    dplm2 = DPLM2Integration(device=args.device)
    
    results = []
    
    # Load structures using the same approach as our working test script
    all_structures = []
    struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
    
    # Load structure sequences directly from struct.fasta (same as test script)
    struct_records = {}
    if os.path.exists(struct_fasta_path):
        from Bio import SeqIO
        for record in SeqIO.parse(struct_fasta_path, "fasta"):
            struct_records[record.id] = str(record.seq)
        print(f"✅ Loaded {len(struct_records)} structure sequences from struct.fasta")
    else:
        print(f"❌ struct.fasta not found: {struct_fasta_path}")
        return
    
    # Create structures using the same format as our test script
    for idx, structure_file in enumerate(loader.structures):
        # Get the base name (e.g., "7dz2_C" from "7dz2_C.pkl")
        base_name = structure_file.replace('.pkl', '')
        
        if base_name in struct_records:
            # Create structure dict in the same format as our test script
            struct_seq = struct_records[base_name]
            structure = {
                'struct_seq': struct_seq,
                'length': len(struct_seq.split(',')),
                'pdb_id': base_name.split('_')[0] if '_' in base_name else base_name,
                'chain_id': base_name.split('_')[1] if '_' in base_name else 'A',
                'name': f"CAMEO {base_name}"
            }
            all_structures.append((idx, structure))
            print(f"✅ Created structure {idx}: {base_name} (length: {structure['length']})")
        else:
            print(f"⚠️ No struct sequence found for {base_name}")
    
    # Select structure range
    if end_idx is not None:
        test_structures = all_structures[start_idx:end_idx]
    else:
        test_structures = all_structures[start_idx:]
    
    print(f"📊 Selected {len(test_structures)} structures to process")
    
    # Method-by-method switching
    for idx, structure in test_structures:
        name = f"CAMEO {structure.get('pdb_id','test')}_{structure.get('chain_id','A')}"

        if args.mode == "random_no_expert":
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert", structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
            if r: results.append(r)
            continue

        if args.mode == "single_expert":
            if args.single_expert_id is None:
                ids = [0,1,2,3]  # Include ProteinMPNN (expert 3)
            else:
                ids = [int(args.single_expert_id)]
            for eid in ids:
                r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid, structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
                if r: results.append(r)
            continue

        if args.mode == "multi_expert":
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert", structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
            if r: results.append(r)
            continue

        # args.mode == "all": run all variants
        r = run_one_structure(structure, name, dplm2, refs, ablation_mode="random_no_expert", structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
        if r: results.append(r)
        for eid in [0,1,2,3]:  # Include ProteinMPNN (expert 3)
            r = run_one_structure(structure, name, dplm2, refs, ablation_mode="single_expert", single_expert_id=eid, structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
            if r: results.append(r)
        r = run_one_structure(structure, name, dplm2, refs, ablation_mode="multi_expert", structure_idx=idx, loader=loader, cached_baselines=cached_baselines, fixed_baseline=args.fixed_baseline, seed_offset=args.baseline_seed_offset, max_depth=args.max_depth, num_iterations=args.num_iterations, use_plddt_masking=not args.disable_plddt_masking, device=args.device)
        if r: results.append(r)

    # save grouped-by-mode summary for easier analysis
    out_dir = "/net/scratch/caom/cameo_evaluation_results"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(out_dir, f"mcts_ablation_results_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Saved ablation results → {out_json}")

    # --- Build per-mode summary table similar to tree_search summary ---
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[r["mode"]].append(r)

    summary_path = os.path.join(out_dir, f"mcts_ablation_summary_{ts}.txt")
    try:
        with open(summary_path, "w") as sf:
            sf.write("MCTS Ablation Summary (grouped by mode)\n")
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
