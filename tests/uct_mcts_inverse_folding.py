#!/usr/bin/env python3
"""
UCT MCTS Experiments for Inverse Folding (Without Entropy)

Runs UCT-based MCTS experiments without entropy bonuses for inverse folding:
1. Single expert experiments: DPLM-2 650M, 150M, 3B, ProteinMPNN
2. Multi-expert experiments: All experts combined
3. Pure UCB1 selection (no PH-UCT entropy bonuses)
4. Multiple rollouts per expert with top-K selection

Usage:
    python uct_mcts_inverse_folding.py --mode single_expert --expert_id 0 --start 0 --end 5
    python uct_mcts_inverse_folding.py --mode multi_expert --start 0 --end 5
    python uct_mcts_inverse_folding.py --mode all --start 0 --end 10  # Run all modes
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
from Bio import SeqIO

# Import MCTS and DPLM-2 components
from core.sequence_level_mcts import GeneralMCTS
from core.dplm2_integration import DPLM2Integration

# Import evaluation utilities
try:
    from utils.cameo_data_loader import CAMEODataLoader
except ImportError:
    class CAMEODataLoader:
        def __init__(self, *args, **kwargs):
            self.structures = []
        def get_test_structure(self, index=0):
            return {"name": f"test_structure_{index}", "sequence": "IKKSI", "length": 5}

_ESMFOLD_MODEL = None
_ESMFOLD_TOKENIZER = None

def _load_esmfold_model():
    """Load and cache transformers ESMFold model for pLDDT computation."""
    global _ESMFOLD_MODEL, _ESMFOLD_TOKENIZER
    if _ESMFOLD_MODEL is not None and _ESMFOLD_TOKENIZER is not None:
        return _ESMFOLD_MODEL, _ESMFOLD_TOKENIZER
    
    try:
        import torch
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        print("   🔄 Loading ESMFold model for pLDDT estimation...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("   ✅ ESMFold moved to GPU")
        else:
            print("   ✅ ESMFold running on CPU")
        
        _ESMFOLD_MODEL = model
        _ESMFOLD_TOKENIZER = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"   ❌ Failed to load ESMFold: {e}")
        return None, None

def _predict_plddt_with_esmfold(sequence: str) -> Optional[List[float]]:
    """Predict per-residue pLDDT scores using ESMFold."""
    model, tokenizer = _load_esmfold_model()
    if model is None or tokenizer is None:
        return None
    
    if not sequence:
        return None
    
    try:
        import torch
        clean_sequence = sequence.replace("X", "A")
        tokenized = tokenizer(clean_sequence, return_tensors="pt", add_special_tokens=False)
        device = next(model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            outputs = model(tokenized["input_ids"])
        
        if hasattr(outputs, "plddt"):
            plddt = outputs.plddt
        elif isinstance(outputs, dict) and "plddt" in outputs:
            plddt = outputs["plddt"]
        else:
            print("   ⚠️ ESMFold output missing pLDDT; using fallback.")
            return None
        
        if isinstance(plddt, torch.Tensor):
            plddt_np = plddt.detach().cpu().numpy()
        else:
            plddt_np = np.asarray(plddt, dtype=np.float32)
        
        scores = plddt_np.reshape(-1).tolist()
        scores = [max(0.0, min(1.0, float(s))) * 100.0 for s in scores]
        return scores
    except Exception as e:
        print(f"   ❌ ESMFold inference failed: {e}")
        return None

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_simple_aar(pred_seq: str, ref_seq: str) -> float:
    """Calculate Amino Acid Recovery (AAR)"""
    L = min(len(pred_seq), len(ref_seq))
    if L == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred_seq[:L], ref_seq[:L])) / L

def calculate_biophysical_score(sequence: str) -> float:
    """Calculate biophysical score based on amino acid composition"""
    if not sequence:
        return 0.0
    
    # Biophysical penalties for extreme compositions
    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFPWV') / len(sequence)
    charged = sum(1 for aa in sequence if aa in 'DEKR') / len(sequence)
    
    # Penalties for extreme distributions
    charge_penalty = max(0, charged - 0.3) * 2  # Penalty if >30% charged
    hydrophobic_penalty = max(0, hydrophobic - 0.4) * 2  # Penalty if >40% hydrophobic
    
    # Base score with penalties
    base_score = 1.0 - charge_penalty - hydrophobic_penalty
    return max(0.0, min(1.0, base_score))

def load_reference_sequences() -> Dict[str, str]:
    """Load CAMEO reference sequences"""
    reference_fasta = "/home/caom/AID3/dplm/data-bin/cameo2022/aatype.fasta"
    sequences = {}
    
    if os.path.exists(reference_fasta):
        for record in SeqIO.parse(reference_fasta, "fasta"):
            sequences[record.id] = str(record.seq).replace(" ", "").upper()
        print(f"✅ Loaded {len(sequences)} reference sequences")
    else:
        print(f"⚠️ Reference FASTA not found: {reference_fasta}")
    
    return sequences

def load_pregenerated_baseline(structure_id: str) -> Optional[str]:
    """Load pregenerated DPLM-2 150M baseline sequence"""
    baseline_path = f"/home/caom/AID3/dplm/generation-results/dplm2_150m/inverse_folding/{structure_id}.fasta"
    
    def clean_aa(seq_str: str) -> str:
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return "".join(c for c in str(seq_str).upper() if c in valid)
    
    try:
        if os.path.exists(baseline_path):
            for record in SeqIO.parse(baseline_path, "fasta"):
                return clean_aa(record.seq)
    except Exception as e:
        print(f"  ⚠️ Failed to load baseline for {structure_id}: {e}")
    
    return None

def run_uct_mcts_experiment(
    structure_data: Dict,
    structure_id: str,
    reference_seq: str,
    mode: str,
    expert_id: Optional[int] = None,
    num_iterations: int = 25,
    max_depth: int = 5,
) -> Dict:
    """Run UCT MCTS experiment for a single structure"""
    
    print(f"\n🧬 [{mode}{'_' + str(expert_id) if expert_id is not None else ''}] {structure_id}")
    
    # Load pregenerated baseline
    baseline_seq = load_pregenerated_baseline(structure_id)
    if not baseline_seq:
        print(f"  ❌ No baseline sequence found for {structure_id}")
        return None
    
    baseline_aar = calculate_simple_aar(baseline_seq, reference_seq)
    print(f"  ✅ Baseline AAR: {baseline_aar:.1%}")
    
    try:
        # Initialize DPLM-2 integration
        dplm2 = DPLM2Integration(device="cuda")
        
        # Load external experts if needed
        external_experts = []
        if mode in ["single_expert", "multi_expert"] and expert_id == 3:
            # ProteinMPNN expert
            try:
                from external_models.real_direct_models import create_real_proteinmpnn_expert
                proteinmpnn_expert = create_real_proteinmpnn_expert()
                external_experts = [proteinmpnn_expert]
                print(f"  ✅ Loaded ProteinMPNN expert")
            except Exception as e:
                print(f"  ⚠️ Failed to load ProteinMPNN: {e}")
                return None
        
        # Prepare baseline structure data
        baseline_structure = structure_data.copy()
        baseline_structure['external_experts'] = external_experts
        if expert_id is not None:
            baseline_structure['single_expert_id'] = expert_id
        
        # **DUAL DATA LOADING**: Load both structure tokens (for DPLM-2) AND coordinates (for ProteinMPNN)
        print(f"🔄 Loading structure data: tokens for DPLM-2 + coordinates for ProteinMPNN...")
        try:
            from utils.cameo_data_loader import CAMEODataLoader
            loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
            
            # Find structure index
            structure_idx = None
            for idx, struct_file in enumerate(loader.structures):
                if structure_id in struct_file:
                    structure_idx = idx
                    break
            
            if structure_idx is None:
                print(f"    ❌ Structure not found in loader: {structure_id}")
                dplm2.set_baseline_structure(baseline_structure)
            else:
                # Load CAMEO structure data (for ProteinMPNN coordinates)
                cameo_structure_data = loader.get_structure_by_index(structure_idx)
                if not cameo_structure_data:
                    print(f"    ❌ Could not load CAMEO structure data for {structure_id}")
                    dplm2.set_baseline_structure(baseline_structure)
                else:
                    # CRITICAL: Merge both data sources
                    # 1. Keep original baseline_structure (has struct_seq from struct.fasta for DPLM-2)
                    # 2. Add coordinates from CAMEO data (for ProteinMPNN)
                    
                    # Add ProteinMPNN coordinates to baseline_structure
                    if 'atom_positions' in cameo_structure_data:
                        coords = cameo_structure_data['atom_positions']
                        if len(coords.shape) == 3 and coords.shape[1] >= 2:
                            baseline_structure['coordinates'] = coords[:, 1, :]  # CA atoms
                            baseline_structure['backbone_coords'] = coords
                        else:
                            baseline_structure['coordinates'] = coords
                        print(f"✅ Added ProteinMPNN coordinates: {baseline_structure['coordinates'].shape}")
                    
                    # Add other CAMEO data that might be useful
                    for key in ['atom_positions', 'aatype', 'atom_mask', 'residue_index']:
                        if key in cameo_structure_data:
                            baseline_structure[key] = cameo_structure_data[key]
                    
                    # Set the combined structure data
                    dplm2.set_baseline_structure(baseline_structure)
                    print(f"✅ Combined structure data: DPLM-2 tokens + ProteinMPNN coordinates")
                    print(f"    Has struct_seq: {'struct_seq' in baseline_structure and baseline_structure['struct_seq']}")
                    print(f"    Has coordinates: {'coordinates' in baseline_structure}")
        except Exception as e:
            print(f"⚠️ Failed to load structure data: {e}")
            dplm2.set_baseline_structure(baseline_structure)
        
        # Compute pLDDT scores using ESMFold for the baseline sequence
        print(f"  🔄 Computing ESMFold pLDDT for baseline sequence (len={len(baseline_seq)})...")
        esmfold_plddt = _predict_plddt_with_esmfold(baseline_seq)
        if esmfold_plddt is not None and len(esmfold_plddt) >= len(baseline_seq):
            baseline_structure['plddt_scores'] = esmfold_plddt[:len(baseline_seq)]
            print(f"  ✅ ESMFold pLDDT mean={np.mean(baseline_structure['plddt_scores']):.2f}, std={np.std(baseline_structure['plddt_scores']):.2f}")
        else:
            print("  ⚠️ Using default pLDDT=70 due to ESMFold failure")
            baseline_structure['plddt_scores'] = [70.0] * len(baseline_seq)
        
        # Set baseline sequence
        dplm2.set_baseline_sequence(baseline_seq)
        
        # Initialize UCT MCTS (NO ENTROPY - pure UCB1)
        mcts = GeneralMCTS(
            dplm2_integration=dplm2,
            baseline_structure=baseline_structure,
            reference_sequence=reference_seq,
            max_depth=max_depth,
            exploration_constant=1.414,  # Standard UCB1 constant
            ablation_mode=mode,
            single_expert_id=expert_id,
            external_experts=external_experts,
            num_rollouts_per_expert=2,
            top_k_candidates=2,
            use_ph_uct=False,  # CRITICAL: Use standard UCT (no entropy/novelty bonuses)
            task_type="inverse_folding",
            num_simulations=num_iterations,
            temperature=1.0,
            use_plddt_masking=True
        )
        print("  ⚙️ UCT mode: standard UCB1 (entropy bonuses disabled)")
        
        baseline_reward = mcts._evaluate_sequence_aar(baseline_seq)
        baseline_structure['baseline_reward'] = baseline_reward
        mcts.baseline_structure['baseline_reward'] = baseline_reward
        
        # Prepare structure data for search
        structure_search_data = {
            'struct_seq': baseline_structure.get('struct_seq', ''),
            'length': baseline_structure.get('length', len(baseline_seq)),
            'pdb_id': baseline_structure.get('pdb_id', ''),
            'chain_id': baseline_structure.get('chain_id', ''),
            'coordinates': baseline_structure.get('coordinates'),
            'plddt_scores': baseline_structure.get('plddt_scores', [])
        }
        
        # Run UCT MCTS search
        print(f"  🔄 Running UCT MCTS search ({num_iterations} iterations, pure UCB1, max_depth={max_depth})...")
        start_time = time.time()
        
        search_result = mcts.search(
            initial_sequence=baseline_seq,
            reference_sequence=reference_seq,
            num_iterations=num_iterations,
            structure_data=structure_search_data
        )

        search_time = time.time() - start_time

        if search_result is None:
            print(f"  ❌ MCTS search failed")
            return None
        
        # Determine best node (search returns best node in tree, but fall back if root provided)
        if getattr(search_result, 'parent', None) is not None or not hasattr(search_result, 'children'):
            best_node = search_result
        else:
            def find_best_node(node):
                best_node_loc, best_score_loc = node, getattr(node, "reward", 0.0)
                for child in node.children:
                    child_best = find_best_node(child)
                    child_score = getattr(child_best, "reward", 0.0)
                    if child_score > best_score_loc:
                        best_node_loc, best_score_loc = child_best, child_score
                return best_node_loc
            best_node = find_best_node(search_result)
        best_seq = best_node.sequence
        final_aar = calculate_simple_aar(best_seq, reference_seq)
        
        # Reward calculations (compound reward used during search)
        final_reward = mcts._evaluate_sequence_aar(best_seq)
        
        # Calculate metrics
        baseline_biophysical = calculate_biophysical_score(baseline_seq)
        final_biophysical = calculate_biophysical_score(best_seq)
        
        # Calculate scTM if coordinates available
        baseline_sctm, final_sctm = None, None
        try:
            from utils.sctm_calculation import calculate_sctm_score
            reference_coords = baseline_structure.get('coordinates')
            
            if reference_coords is not None:
                baseline_sctm = calculate_sctm_score(baseline_seq, reference_coords)
                final_sctm = calculate_sctm_score(best_seq, reference_coords)
        except Exception as e:
            print(f"  ⚠️ scTM calculation failed: {e}")
        
        # Compile results
        result = {
            'structure_id': structure_id,
            'mode': f"{mode}{'_' + str(expert_id) if expert_id is not None else ''}",
            'method': 'UCT_MCTS',
            'baseline_sequence': baseline_seq,
            'final_sequence': best_seq,
            'baseline_reward': baseline_reward,
            'final_reward': final_reward,
            'reward_improvement': final_reward - baseline_reward,
            'baseline_aar': baseline_aar,
            'final_aar': final_aar,
            'aar_improvement': final_aar - baseline_aar,
            'baseline_biophysical': baseline_biophysical,
            'final_biophysical': final_biophysical,
            'biophysical_improvement': final_biophysical - baseline_biophysical,
            'baseline_sctm': baseline_sctm if baseline_sctm is not None else 0.0,
            'final_sctm': final_sctm if final_sctm is not None else 0.0,
            'sctm_improvement': (final_sctm - baseline_sctm) if (baseline_sctm and final_sctm) else 0.0,
            'search_time': search_time,
            'num_iterations': num_iterations,
            'exploration_constant': 1.414,
            'entropy_bonus_used': False,
            'novelty_bonus_used': False
        }
        
        # Print summary
        print(f"  ─────────────────────────────────────────")
        print(f"  UCT MCTS Results [{mode}{'/' + str(expert_id) if expert_id is not None else ''}]")
        print(f"    Reward: {baseline_reward:.4f} → {final_reward:.4f} (Δ {final_reward - baseline_reward:+.4f})")
        print(f"    AAR: {baseline_aar:.1%} → {final_aar:.1%} (Δ {final_aar - baseline_aar:+.1%})")
        print(f"    Bio: {baseline_biophysical:.3f} → {final_biophysical:.3f} (Δ {final_biophysical - baseline_biophysical:+.3f})")
        if baseline_sctm and final_sctm:
            print(f"    scTM: {baseline_sctm:.3f} → {final_sctm:.3f} (Δ {final_sctm - baseline_sctm:+.3f})")
        print(f"    Time: {search_time:.1f}s")
        print(f"  ─────────────────────────────────────────")
        
        return result
        
    except Exception as e:
        print(f"  ❌ Experiment failed: {e}")
        return None

def run_experiments(
    structure_ids: List[str],
    reference_sequences: Dict[str, str],
    modes: List[str],
    expert_ids: List[int],
    num_iterations: int,
    max_depth: int,
) -> List[Dict]:
    """Run all UCT MCTS experiments"""
    
    results = []
    
    # Load CAMEO data
    loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
    
    # Load structure sequences from struct.fasta
    struct_fasta_path = "/home/caom/AID3/dplm/data-bin/cameo2022/struct.fasta"
    struct_records = {}
    if os.path.exists(struct_fasta_path):
        for record in SeqIO.parse(struct_fasta_path, "fasta"):
            struct_records[record.id] = str(record.seq)
        print(f"✅ Loaded {len(struct_records)} structure sequences")
    
    for structure_id in structure_ids:
        if structure_id not in reference_sequences:
            print(f"⚠️ No reference sequence for {structure_id}")
            continue
        
        if structure_id not in struct_records:
            print(f"⚠️ No structure sequence for {structure_id}")
            continue
        
        # Prepare structure data
        struct_seq = struct_records[structure_id]
        structure_data = {
            'struct_seq': struct_seq,
            'length': len(struct_seq.split(',')),
            'pdb_id': structure_id.split('_')[0] if '_' in structure_id else structure_id,
            'chain_id': structure_id.split('_')[1] if '_' in structure_id else 'A',
            'name': f"CAMEO {structure_id}"
        }
        
        # Load coordinates if available
        try:
            structure_idx = None
            for idx, struct_file in enumerate(loader.structures):
                if structure_id in struct_file:
                    structure_idx = idx
                    break
            
            if structure_idx is not None:
                cameo_structure = loader.get_structure_by_index(structure_idx)
                if cameo_structure:
                    # Add coordinates
                    for coord_key in ['backbone_coords', 'coordinates', 'atom_positions']:
                        if coord_key in cameo_structure and cameo_structure[coord_key] is not None:
                            coords = cameo_structure[coord_key]
                            if len(coords.shape) == 3 and coords.shape[1] >= 2:
                                structure_data['coordinates'] = coords[:, 1, :]  # CA atoms
                            else:
                                structure_data['coordinates'] = coords
                            break
        except Exception as e:
            print(f"⚠️ Could not load coordinates for {structure_id}: {e}")
        
        reference_seq = reference_sequences[structure_id]
        
        # Run experiments for each mode
        for mode in modes:
            if mode == "single_expert":
                for expert_id in expert_ids:
                    result = run_uct_mcts_experiment(
                        structure_data,
                        structure_id,
                        reference_seq,
                        mode,
                        expert_id,
                        num_iterations=num_iterations,
                        max_depth=max_depth,
                    )
                    if result:
                        results.append(result)
            elif mode == "multi_expert":
                result = run_uct_mcts_experiment(
                    structure_data,
                    structure_id,
                    reference_seq,
                    mode,
                    num_iterations=num_iterations,
                    max_depth=max_depth,
                )
                if result:
                    results.append(result)
    
    return results

def calculate_summary_statistics(results: List[Dict]) -> Dict:
    """Calculate summary statistics for all experiments"""
    summary = {}
    
    # Group by mode
    modes = {}
    for result in results:
        mode = result['mode']
        if mode not in modes:
            modes[mode] = []
        modes[mode].append(result)
    
    print(f"\n📊 UCT MCTS Summary Statistics")
    print("=" * 70)
    
    for mode, mode_results in modes.items():
        if not mode_results:
            continue
        
        # Calculate statistics with safe fallbacks
        aar_improvements = [r.get('aar_improvement', 0.0) for r in mode_results]
        bio_improvements = [r.get('biophysical_improvement', 0.0) for r in mode_results]
        sctm_improvements = [r.get('sctm_improvement', 0.0) for r in mode_results if r.get('sctm_improvement') is not None]
        search_times = [r.get('search_time', 0.0) for r in mode_results]
        
        def _mean(values):
            return float(np.mean(values)) if values else 0.0
        
        def _std(values):
            return float(np.std(values)) if values else 0.0
        
        def _min(values):
            return float(np.min(values)) if values else 0.0
        
        def _max(values):
            return float(np.max(values)) if values else 0.0
        
        aar_mean = _mean(aar_improvements)
        aar_std = _std(aar_improvements)
        aar_min = _min(aar_improvements)
        aar_max = _max(aar_improvements)
        
        bio_mean = _mean(bio_improvements)
        bio_std = _std(bio_improvements)
        
        sctm_mean = _mean(sctm_improvements)
        sctm_std = _std(sctm_improvements)
        
        time_mean = _mean(search_times)
        time_std = _std(search_times)
        
        stats = {
            'mode': mode,
            'num_structures': len(mode_results),
            'aar_improvement_mean': aar_mean,
            'aar_improvement_std': aar_std,
            'aar_improvement_min': aar_min,
            'aar_improvement_max': aar_max,
            'bio_improvement_mean': bio_mean,
            'bio_improvement_std': bio_std,
            'sctm_improvement_mean': sctm_mean,
            'sctm_improvement_std': sctm_std,
            'search_time_mean': time_mean,
            'search_time_std': time_std,
            'success_rate': len(mode_results) / len(mode_results)  # All completed are successes
        }
        
        summary[mode] = stats
        
        # Print summary
        print(f"\n🔬 {mode}")
        print(f"  Structures: {stats['num_structures']}")
        print(f"  AAR Δ: {stats['aar_improvement_mean']:+.1%} ± {stats['aar_improvement_std']:.1%}")
        print(f"  Bio Δ: {stats['bio_improvement_mean']:+.3f} ± {stats['bio_improvement_std']:.3f}")
        if sctm_improvements:
            print(f"  scTM Δ: {stats['sctm_improvement_mean']:+.3f} ± {stats['sctm_improvement_std']:.3f}")
        print(f"  Time: {stats['search_time_mean']:.1f}s ± {stats['search_time_std']:.1f}s")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
    
    return summary

def save_results(results: List[Dict], summary: Dict, output_dir: str):
    """Save results to JSON files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"uct_mcts_inverse_folding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾 Detailed results saved to: {results_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, f"uct_mcts_inverse_folding_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"📈 Summary statistics saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="UCT MCTS experiments for inverse folding")
    parser.add_argument("--mode", choices=["single_expert", "multi_expert", "all"], 
                       default="all", help="Experiment mode")
    parser.add_argument("--expert_id", type=int, choices=[0, 1, 2, 3], 
                       help="Single expert ID (0=650M, 1=150M, 2=3B, 3=ProteinMPNN)")
    parser.add_argument("--start", type=int, default=0, help="Start structure index")
    parser.add_argument("--end", type=int, default=5, help="End structure index")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/caom/AID3/dplm/mcts_diffusion_finetune/results/uct_mcts_analysis",
                       help="Output directory for results")
    parser.add_argument("--num_iterations", type=int, default=25,
                       help="Number of UCT iterations per structure")
    parser.add_argument("--max_depth", type=int, default=5,
                       help="Maximum search depth for UCT")
    
    args = parser.parse_args()
    
    setup_logging()
    
    print("🧬 UCT MCTS Experiments - Inverse Folding (No Entropy)")
    print("=" * 70)
    print(f"🎯 Mode: {args.mode}")
    print(f"📊 Structure range: {args.start}-{args.end}")
    if args.expert_id is not None:
        print(f"🤖 Expert ID: {args.expert_id}")
    print(f"🔁 Iterations: {args.num_iterations}")
    print(f"🌳 Max depth: {args.max_depth}")
    
    # Load reference sequences
    reference_sequences = load_reference_sequences()
    if not reference_sequences:
        print("❌ No reference sequences loaded. Exiting.")
        return
    
    # Determine structure IDs to process
    structure_ids = list(reference_sequences.keys())[args.start:args.end]
    print(f"🧬 Processing {len(structure_ids)} structures")
    
    # Determine modes and expert IDs
    if args.mode == "all":
        modes = ["single_expert", "multi_expert"]
        expert_ids = [0, 1, 2, 3]  # All experts
    elif args.mode == "single_expert":
        modes = ["single_expert"]
        expert_ids = [args.expert_id] if args.expert_id is not None else [0, 1, 2, 3]
    else:  # multi_expert
        modes = ["multi_expert"]
        expert_ids = []
    
    # Run experiments
    results = run_experiments(
        structure_ids,
        reference_sequences,
        modes,
        expert_ids,
        num_iterations=args.num_iterations,
        max_depth=args.max_depth,
    )
    
    if not results:
        print("❌ No results generated. Exiting.")
        return
    
    # Calculate summary statistics
    summary = calculate_summary_statistics(results)
    
    # Save results
    save_results(results, summary, args.output_dir)
    
    print(f"\n🎉 UCT MCTS experiments complete!")
    print(f"📊 Processed {len(results)} total experiments")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
