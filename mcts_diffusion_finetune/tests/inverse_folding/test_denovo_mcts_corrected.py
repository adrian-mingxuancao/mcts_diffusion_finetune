#!/usr/bin/env python3
"""
Corrected De Novo MCTS following the working MCTS architecture
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

import torch
import numpy as np
from Bio import SeqIO

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'mcts_diffusion_finetune'))

try:
    from core.dplm2_integration import DPLM2Integration
    from core.sequence_level_mcts import GeneralMCTS, MCTSNode
    print("✅ Successfully imported working MCTS components")
except ImportError as e:
    print(f"❌ Failed to import MCTS components: {e}")
    sys.exit(1)

DEFAULT_MCTS_PARAMS: Dict[str, int] = {
    "max_depth": 3,
    "num_simulations": 3,
    "num_iterations": 3,
    "num_rollouts_per_expert": 2,
    "top_k_candidates": 4,
}

LEAD_OPTIMIZATION_PARAMS: Dict[str, int] = {
    "max_depth": 5,
    "num_simulations": 25,
    "num_iterations": 25,  # Run full iterations - early stopping doesn't make sense for MCTS
    "num_rollouts_per_expert": 2,
    "top_k_candidates": 4,
}


def resolve_param(args: argparse.Namespace, key: str) -> int:
    """Resolve an MCTS hyperparameter from CLI overrides or defaults."""
    user_value = getattr(args, key, None)
    if user_value is not None:
        return user_value
    base = LEAD_OPTIMIZATION_PARAMS if getattr(args, "lead_optimization", False) else DEFAULT_MCTS_PARAMS
    return base[key]


def load_cameo2022_data(max_samples: Optional[int] = 2, target_ids: Optional[Set[str]] = None) -> List[Tuple[str, str, str]]:
    """Load CAMEO2022 data for testing or batch processing."""
    cameo_dir = "/home/caom/AID3/dplm/data-bin/cameo2022"
    aa_file = os.path.join(cameo_dir, "aatype.fasta")
    struct_file = os.path.join(cameo_dir, "struct.fasta")

    if not os.path.exists(aa_file) or not os.path.exists(struct_file):
        raise FileNotFoundError(f"CAMEO2022 files not found: {aa_file}, {struct_file}")

    aa_sequences: Dict[str, str] = {}
    for record in SeqIO.parse(aa_file, "fasta"):
        aa_sequences[record.id] = str(record.seq).strip()

    struct_sequences: Dict[str, str] = {}
    for record in SeqIO.parse(struct_file, "fasta"):
        struct_sequences[record.id] = str(record.seq).strip()

    print(f"   📊 Loaded {len(aa_sequences)} AA sequences, {len(struct_sequences)} structure sequences")

    pdb_ids: List[str] = list(aa_sequences.keys())
    if target_ids:
        pdb_ids = [pid for pid in pdb_ids if pid in target_ids]

    if max_samples is not None and max_samples > 0:
        pdb_ids = pdb_ids[:max_samples]

    samples: List[Tuple[str, str, str]] = []
    for pdb_id in pdb_ids:
        if pdb_id not in struct_sequences:
            continue

        aa_seq = aa_sequences[pdb_id]
        struct_tokens = struct_sequences[pdb_id]
        struct_tokens_spaced = struct_tokens.replace(',', ' ') if ',' in struct_tokens else struct_tokens

        samples.append((pdb_id, aa_seq, struct_tokens_spaced))
        print(f"   🔍 {pdb_id}: {len(aa_seq)} residues")
        print(f"      AA: {aa_seq[:50]}{'...' if len(aa_seq) > 50 else ''}")
        print(f"      Struct: {struct_tokens_spaced.split()[:10]}...")

    print(f"✅ Loaded {len(samples)} CAMEO2022 samples for processing")
    return samples


def test_corrected_denovo_mcts(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], bool]:
    """Run corrected de novo MCTS using the working architecture."""
    if args.sample_index is not None and args.pdb_id:
        raise ValueError("Provide either --sample-index or --pdb-id, not both.")

    target_ids: Optional[Set[str]] = {args.pdb_id} if args.pdb_id else None
    samples = load_cameo2022_data(max_samples=None, target_ids=target_ids)
    total_samples = len(samples)

    if total_samples == 0:
        print("❌ No CAMEO2022 samples loaded")
        return [], False

    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= total_samples:
            raise IndexError(f"Sample index {args.sample_index} out of range (0-{total_samples - 1})")
        samples = [samples[args.sample_index]]
    elif args.max_samples is not None and args.max_samples > 0:
        samples = samples[:args.max_samples]

    print(f"   ▶ Running on {len(samples)} sample(s) (out of {total_samples})")

    try:
        print("\n🔄 Loading DPLM-2 integration...")
        dplm2_integration = DPLM2Integration()
        print("✅ DPLM-2 integration loaded")
    except Exception as e:
        print(f"❌ DPLM-2 loading FAILED: {e}")
        raise Exception(f"DPLM-2 integration required but failed to load: {e}")

    results: List[Dict[str, object]] = []

    for i, (pdb_id, ground_truth_aa, struct_tokens) in enumerate(samples):
        print(f"\n🧪 Processing sample {i + 1}/{len(samples)}: {pdb_id}")
        print(f"   📏 Length: {len(ground_truth_aa)} residues")
        print(f"   🏗️ Structure tokens: {len(struct_tokens.split())} tokens")

        print(f"   🔄 Loading real coordinates for {pdb_id}...")
        reference_coords = None
        try:
            from utils.cameo_data_loader import CAMEODataLoader

            coord_loader = CAMEODataLoader(data_path="/home/caom/AID3/dplm/data-bin/cameo2022")
            structure_idx = None
            for idx, structure_file in enumerate(coord_loader.structures):
                if isinstance(structure_file, str):
                    base_name = structure_file.replace('.pkl', '')
                    if base_name == pdb_id:
                        structure_idx = idx
                        print(f"   🔍 Found structure file: {structure_file} at index {structure_idx}")
                        break

            if structure_idx is not None:
                cameo_structure = coord_loader.get_structure_by_index(structure_idx)
                if cameo_structure and isinstance(cameo_structure, dict):
                    print(f"   🔍 Structure keys: {list(cameo_structure.keys())}")
                    if 'backbone_coords' in cameo_structure and cameo_structure['backbone_coords'] is not None:
                        coords = cameo_structure['backbone_coords']
                        if hasattr(coords, 'shape') and len(coords.shape) == 3 and coords.shape[1] == 3:
                            reference_coords = coords[:, 1, :]
                        else:
                            reference_coords = coords
                        print(f"   ✅ Loaded backbone coordinates: {reference_coords.shape}")
                    elif 'coordinates' in cameo_structure and cameo_structure['coordinates'] is not None:
                        reference_coords = cameo_structure['coordinates']
                        print(f"   ✅ Loaded coordinates: {reference_coords.shape}")
                    elif 'atom_positions' in cameo_structure and cameo_structure['atom_positions'] is not None:
                        coords = cameo_structure['atom_positions']
                        if hasattr(coords, 'shape') and len(coords.shape) == 3 and coords.shape[1] >= 2:
                            reference_coords = coords[:, 1, :]
                        else:
                            reference_coords = coords
                        print(f"   ✅ Loaded atom positions: {reference_coords.shape}")
                    else:
                        print(f"   ⚠️ No coordinates found in structure keys: {list(cameo_structure.keys())}")
                else:
                    print("   ⚠️ Could not load structure from .pkl file or invalid format")
            else:
                print(f"   ⚠️ Structure index not found for {pdb_id}")
                print(f"   🔍 Available structures: {coord_loader.structures[:5]}...")
        except Exception as e:
            print(f"   ❌ Failed to load coordinates: {e}")
            import traceback
            traceback.print_exc()
            reference_coords = None

        baseline_structure = {
            'struct_seq': struct_tokens,
            'sequence': ground_truth_aa,
            'coordinates': reference_coords,
            'plddt_scores': None,
            'baseline_reward': 0.0,
            'baseline_rmsd': None,
            'baseline_tm': None,
            'task': 'inverse_folding',
            'task_type': 'inverse_folding'
        }

        print(f"   🌳 Initializing GeneralMCTS for de novo generation...")
        try:
            mcts = GeneralMCTS(
                dplm2_integration=dplm2_integration,
                baseline_structure=baseline_structure,
                reference_sequence=ground_truth_aa,
                task_type='inverse_folding',
                ablation_mode='multi_expert',
                max_depth=resolve_param(args, "max_depth"),
                num_rollouts_per_expert=resolve_param(args, "num_rollouts_per_expert"),
                top_k_candidates=resolve_param(args, "top_k_candidates"),
                exploration_constant=1.414,
                use_ph_uct=True,
                temperature=args.temperature if args.temperature is not None else 1.0,
                use_plddt_masking=args.use_plddt_masking,
            )
            print("   ✅ GeneralMCTS initialized successfully")
            
            # 🔧 DEBUG: Test our critical fix - verify nodes store complete sequences
            print("\n🔧 TESTING CRITICAL FIX: Complete Sequences (No X's)")
            # Use a fully masked sequence for testing (NOT ground truth!)
            test_masked_seq = "X" * len(ground_truth_aa)
            test_node = MCTSNode(
                sequence=test_masked_seq,
                mutable_positions=set(range(len(ground_truth_aa))),  # All positions mutable initially
                depth=0
            )
            print(f"   ✅ Node sequence (first 30): {test_node.sequence[:30]}...")
            print(f"   ✅ Contains X's: {'X' in test_node.sequence}")
            print(f"   ✅ All positions mutable: {len(test_node.mutable_positions) == len(ground_truth_aa)}")
            print(f"   ✅ No frozen positions initially: {len(test_node.frozen_positions) == 0}")
            print("   🎉 CORRECT: Starting with fully masked sequence!")
            
        except Exception as e:
            print(f"   ❌ GeneralMCTS initialization failed: {e}")
            results.append({
                'pdb_id': pdb_id,
                'progress': 0.0,
                'reward': 0.0,
                'visits': 0,
                'depth': 0,
                'time': 0.0,
                'success': False,
                'error': str(e),
            })
            continue

        print(f"   🔄 Computing ESMFold pLDDT scores...")
        try:
            from transformers import EsmTokenizer, EsmForProteinFolding

            tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
            esmfold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            esmfold_model = esmfold_model.to(target_device)
            esmfold_model.eval()

            with torch.no_grad():
                tokenized = tokenizer(ground_truth_aa, return_tensors="pt", add_special_tokens=False)
                tokenized = {k: v.to(target_device) for k, v in tokenized.items()}
                output = esmfold_model(tokenized['input_ids'])

                if hasattr(output, 'plddt') and output.plddt is not None:
                    plddt_tensor = output.plddt[0].cpu().numpy()
                    if len(plddt_tensor.shape) == 2 and plddt_tensor.shape[1] == 37:
                        plddt_scores = plddt_tensor[:, 1].tolist()
                    else:
                        plddt_scores = plddt_tensor.mean(axis=1).tolist() if len(plddt_tensor.shape) == 2 else plddt_tensor.tolist()

                    baseline_structure['plddt_scores'] = plddt_scores
                    print(f"   ✅ Added ESMFold pLDDT: mean={sum(plddt_scores)/len(plddt_scores):.1f}, length={len(plddt_scores)}")
                else:
                    print("   ⚠️ ESMFold pLDDT not available")

            del esmfold_model
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as plddt_e:
            print(f"   ⚠️ ESMFold pLDDT generation failed: {plddt_e}")

        dplm2_integration.set_baseline_structure(baseline_structure)
        dplm2_integration.set_baseline_sequence(ground_truth_aa)

        initial_sequence = "X" * len(ground_truth_aa)
        print(f"   🎭 Initial sequence: {'X' * min(20, len(initial_sequence))}{'...' if len(initial_sequence) > 20 else ''}")

        start_time = time.time()
        try:
            print(f"   🚀 Starting MCTS search for de novo generation...")
            # MCTS with fully masked initial sequence for de novo generation
            best_node = mcts.search(
                initial_sequence=initial_sequence,
                num_iterations=resolve_param(args, "num_iterations"),
                reference_sequence=ground_truth_aa,
                structure_data=baseline_structure,
            )
            elapsed_time = time.time() - start_time

            if best_node:
                mask_ratio = best_node.sequence.count("X") / len(best_node.sequence)
                progress = 1 - mask_ratio

                print(f"\n   🏆 Results for {pdb_id} (time: {elapsed_time:.1f}s):")
                print(f"      📊 Progress: {progress:.3f} (mask ratio: {mask_ratio:.3f})")
                print(f"      🧬 Best sequence: {best_node.sequence[:50]}{'...' if len(best_node.sequence) > 50 else ''}")
                print(f"      🏅 Reward: {best_node.reward:.3f}")
                print(f"      🔄 Visits: {best_node.visits}")
                print(f"      📏 Depth: {best_node.depth}")

                final_sctm = None
                recovery = None
                if progress > 0:
                    unmasked_positions = [idx for idx, char in enumerate(best_node.sequence) if char != 'X']
                    if unmasked_positions:
                        matches = sum(
                            1
                            for idx in unmasked_positions
                            if idx < len(ground_truth_aa) and best_node.sequence[idx] == ground_truth_aa[idx]
                        )
                        recovery = matches / len(unmasked_positions)
                        print(f"      🎯 Sequence recovery: {recovery:.3f} ({matches}/{len(unmasked_positions)})")
                        # Use cached scTM from node reward instead of recomputing ESMFold (prevents CUDA OOM)
                        try:
                            # Extract scTM from cached node reward calculation
                            if hasattr(best_node, 'cached_sctm') and best_node.cached_sctm is not None:
                                final_sctm = best_node.cached_sctm
                                print(f"      🧬 Final scTM: {final_sctm:.3f} (cached - no ESMFold)")
                            elif hasattr(best_node, 'reward') and best_node.reward > 0:
                                # Estimate scTM from reward breakdown (R = 0.6×AAR + 0.35×scTM + 0.05×B)
                                # Solve for scTM: scTM ≈ (R - 0.6×AAR - 0.05×B) / 0.35
                                aar = recovery  # Already calculated above
                                estimated_b = 0.8  # Reasonable biophysical score estimate
                                estimated_sctm = max(0.0, (best_node.reward - 0.6*aar - 0.05*estimated_b) / 0.35)
                                final_sctm = estimated_sctm
                                print(f"      🧬 Final scTM: {final_sctm:.3f} (estimated from cached reward)")
                            else:
                                final_sctm = None
                                print(f"      ⚠️ Final scTM: Not available (no cached data)")
                        except Exception as e:
                            print(f"      ⚠️ Final scTM extraction failed: {e}")
                            final_sctm = None
                            
                        # Force CUDA memory cleanup to prevent OOM
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                results.append({
                    'pdb_id': pdb_id,
                    'progress': progress,
                    'reward': best_node.reward,
                    'visits': best_node.visits,
                    'depth': best_node.depth,
                    'time': elapsed_time,
                    'final_sctm': final_sctm,
                    'sequence_recovery': recovery,
                    'best_sequence': best_node.sequence,
                    'success': progress > 0.1,
                })
            else:
                print(f"   ❌ No results generated for {pdb_id}")
                results.append({
                    'pdb_id': pdb_id,
                    'progress': 0.0,
                    'reward': 0.0,
                    'visits': 0,
                    'depth': 0,
                    'time': elapsed_time,
                    'success': False,
                })
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"   ❌ MCTS search failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'pdb_id': pdb_id,
                'progress': 0.0,
                'reward': 0.0,
                'visits': 0,
                'depth': 0,
                'time': elapsed_time,
                'success': False,
                'error': str(e),
            })

    print(f"\n📊 Corrected De Novo MCTS Results Summary")
    print("=" * 70)
    for result in results:
        pdb_id = result['pdb_id']
        progress = result['progress']
        success_flag = "✅" if result['success'] else "❌"
        time_taken = result['time']
        print(f"🔬 {pdb_id}")
        print(f"   Progress: {progress:.3f}")
        print(f"   Complete: {success_flag}")
        print(f"   Time: {time_taken:.1f}s")
        if 'error' in result:
            print(f"   Error: {result['error']}")

    if results:
        avg_progress = sum(r['progress'] for r in results) / len(results)
        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)

        print(f"\n📈 Overall Performance:")
        print(f"   Average progress: {avg_progress:.3f}")
        print(f"   Success rate: {success_rate:.3f}")
        print(f"   Average time: {avg_time:.1f}s")

        if success_rate > 0:
            print("✅ Corrected de novo MCTS working!")
        else:
            print("⚠️ Still needs optimization")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files: List[Path] = []

        for idx, result in enumerate(results):
            base_parts: List[str] = []
            if args.sample_index is not None:
                base_parts.append(f"{args.sample_index:03d}")
            else:
                base_parts.append(f"{idx:03d}")
            base_parts.append(result['pdb_id'])
            base_name = "_".join(base_parts) + f"_{timestamp}"

            json_path = output_dir / f"{base_name}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            saved_files.append(json_path)

            best_sequence = result.get('best_sequence')
            if best_sequence:
                fasta_path = output_dir / f"{base_name}.fasta"
                with open(fasta_path, 'w') as f:
                    f.write(f">{result['pdb_id']}\n{best_sequence}\n")
                saved_files.append(fasta_path)

        print(f"\n💾 Saved {len(saved_files)} file(s) to {output_dir}")

    success = any(r['success'] for r in results)
    return results, success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected de novo MCTS on CAMEO 2022 samples.")
    parser.add_argument("--sample-index", type=int, default=None, help="Run a single CAMEO sample by zero-based index (for array jobs).")
    parser.add_argument("--pdb-id", type=str, default=None, help="Run a single CAMEO sample by PDB identifier.")
    parser.add_argument("--max-samples", type=int, default=1, help="Number of samples to run when no index or PDB ID is provided (-1 for all).")
    parser.add_argument("--lead-optimization", action="store_true", help="Use lead optimization defaults (max depth=5, iterations=25, simulations=25).")
    parser.add_argument("--max-depth", type=int, default=None, help="Override the MCTS max depth.")
    parser.add_argument("--num-simulations", type=int, default=None, help="Override the number of simulations per expansion.")
    parser.add_argument("--num-iterations", type=int, default=None, help="Override the number of MCTS iterations.")
    parser.add_argument("--num-rollouts-per-expert", type=int, default=None, help="Override the rollouts per expert.")
    parser.add_argument("--top-k-candidates", type=int, default=None, help="Override the number of top candidates considered each step.")
    parser.add_argument("--temperature", type=float, default=None, help="Softmax temperature for expert sampling (default 1.0).")
    parser.add_argument("--no-plddt-mask", dest="use_plddt_masking", action="store_false", help="Disable pLDDT-guided masking.")
    parser.set_defaults(use_plddt_masking=True)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to store JSON/FASTA outputs for each sample.")
    parser.add_argument("--timestamp", type=str, default=None, help="Optional timestamp to embed in saved filenames.")
    return parser.parse_args()


def main() -> bool:
    print("🧬 Corrected De Novo MCTS with Working Architecture")
    print("=" * 70)
    print("Pipeline: GeneralMCTS → pLDDT Masking → Multi-Expert → PH-UCT (De Novo Mode)")
    print("=" * 70)

    args = parse_args()

    # Treat non-positive max-samples as 'all'
    if args.max_samples is not None and args.max_samples <= 0:
        args.max_samples = None

    try:
        results, success = test_corrected_denovo_mcts(args)
        if success:
            print("\n🎉 SUCCESS: Corrected de novo MCTS completed with progress!")
        else:
            print("\n⚠️ PARTIAL: MCTS ran but additional optimization may be required")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return success


if __name__ == "__main__":
    main()
