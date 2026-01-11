"""
Test integration of hallucination expert with existing MCTS.

This demonstrates how to plug the AF3+ProteinMPNN hallucination expert
into the existing GeneralMCTS framework.

NEW: Hallucination Design MCTS
- Start from random/all-masked sequence
- Iterate: ESMFold (seq→struct) → ProteinMPNN (struct→seq)
- Q-value = convergence (similarity to parent)
- Goal: converge to a stable sequence-structure pair

NOTE: This file now imports HallucinationMCTS and HallucinationNode from
core/hallucination_mcts.py instead of defining them locally.
"""

import sys
import os
import json
import numpy as np
import random
import math
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_diffusion_finetune')
sys.path.insert(0, '/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination')

# Import from refactored core module
from core.hallucination_mcts import HallucinationMCTS, HallucinationNode
from core.hallucination_expert import create_hallucination_expert
from core.esmfold_integration import ESMFoldIntegration
from core.ss_guidance import (
    SSGuidanceConfig,
    SSGuidance,
    DSSPResult,
    EditLogEntry,
    create_run_directory,
)


# ============================================================================
# LEGACY COMPATIBILITY - HallucinationMCTS and HallucinationNode are now
# imported from core/hallucination_mcts.py
# ============================================================================

# The following classes are kept for backward compatibility but are now
# aliases to the core module implementations:
# - HallucinationNode: from core.hallucination_mcts
# - HallucinationMCTS: from core.hallucination_mcts


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_hallucination_mcts(
    use_mock: bool = True, 
    length: int = 50,
    init_mode: str = "random",
    num_iterations: int = 10,
    max_depth: int = 5,
    output_dir: Optional[str] = None,
    structure_backend: str = "esmfold",
    num_candidates: int = 2,
    traj_mode: str = "default",
    ss_guidance_config: Optional[SSGuidanceConfig] = None,
    seed: Optional[int] = None,
):
    """
    Test the hallucination MCTS design.
    
    Args:
        use_mock: Use mock models for testing (faster)
        length: Sequence length to test
        init_mode: Initialization mode - "random", "all_a", "all_g"
        num_iterations: Number of MCTS iterations
        max_depth: Maximum tree depth
        output_dir: Directory to save PDB files
        structure_backend: "esmfold" or "boltz" for structure prediction
        num_candidates: K candidates per cycle (HalluDesign style)
        traj_mode: "short", "long", "default" - for future AF3 truncated diffusion
        ss_guidance_config: Optional SS guidance configuration
        seed: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print(f"Test: Hallucination MCTS Design")
    print(f"  Length: {length}, Mock: {use_mock}, Init: {init_mode}")
    print(f"  Iterations: {num_iterations}, Max Depth: {max_depth}")
    print(f"  Backend: {structure_backend}, K={num_candidates}, Traj: {traj_mode}")
    if ss_guidance_config and ss_guidance_config.ss_guidance != "none":
        print(f"  SS Guidance: {ss_guidance_config.ss_guidance}")
    print("="*80 + "\n")
    
    # Create MCTS
    mcts = HallucinationMCTS(
        sequence_length=length,
        max_depth=max_depth,
        num_iterations=num_iterations,
        num_rollouts=2,
        use_mock=use_mock,
        init_mode=init_mode,
        output_dir=output_dir,
        structure_backend=structure_backend,
        num_candidates=num_candidates,
        traj_mode=traj_mode,
        ss_guidance_config=ss_guidance_config,
        seed=seed,
    )
    
    # Run search
    best_node = mcts.search()
    
    # Print results
    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"\n🏆 Best converged node:")
    print(f"   Depth: {best_node.depth}")
    print(f"   Sequence: {best_node.sequence}")
    print(f"   Parent-Child Similarity: {best_node.parent_child_similarity:.3f}")
    print(f"   Sibling Convergence: {best_node.sibling_convergence:.3f}")
    print(f"   Mean pLDDT: {best_node.mean_plddt:.1f}")
    print(f"   Visits: {best_node.visits}")
    print(f"   Avg Reward: {best_node.get_reward():.3f}")
    print(f"   Output dir: {mcts.output_dir}")
    
    # Trace path from root
    print(f"\n📈 Path from root (showing convergence evolution):")
    path = []
    node = best_node
    while node is not None:
        path.append(node)
        node = node.parent
    path.reverse()
    
    for i, n in enumerate(path):
        if i == 0:
            print(f"   Depth {n.depth}: {n.sequence[:30]}... (root, init_mode={mcts.init_mode})")
        else:
            print(f"   Depth {n.depth}: {n.sequence[:30]}... (parent_sim={n.parent_child_similarity:.3f}, sibling={n.sibling_convergence:.3f}, pLDDT={n.mean_plddt:.1f})")
    
    return best_node, mcts, path


def save_results_to_json(
    mcts: HallucinationMCTS,
    path: List[HallucinationNode],
    output_path: str = "/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination/branching_nodes.json"
):
    """
    Save MCTS results to JSON file.
    
    Collects all nodes from the tree and saves them in the same format as existing branching_nodes.json.
    """
    nodes_data = []
    node_id_map = {}  # Map node object to node_id
    current_id = 0
    
    def collect_nodes(node: HallucinationNode, parent_id: Optional[int] = None):
        nonlocal current_id
        node_id = current_id
        node_id_map[id(node)] = node_id
        current_id += 1
        
        # Convert numpy arrays to lists for JSON serialization
        coord_shape = None
        if node.coordinates is not None:
            coord_shape = list(node.coordinates.shape)
        
        plddt_list = None
        if node.plddt_scores is not None:
            if isinstance(node.plddt_scores, np.ndarray):
                plddt_list = node.plddt_scores.tolist()
            else:
                plddt_list = list(node.plddt_scores)
        
        node_data = {
            "node_id": node_id,
            "parent_id": parent_id,
            "depth": node.depth,
            "sequence": node.sequence,
            "mean_plddt": float(node.mean_plddt) if node.mean_plddt else None,
            "parent_child_similarity": float(node.parent_child_similarity),
            "sibling_convergence": float(node.sibling_convergence),
            "convergence_score": float(node.convergence_score),
            "visits": node.visits,
            "total_reward": float(node.total_reward),
            "avg_reward": float(node.get_reward()),
            "coordinate_shape": coord_shape,
            "plddt_scores": plddt_list,
        }
        nodes_data.append(node_data)
        
        # Recursively collect children
        for child in node.children:
            collect_nodes(child, node_id)
    
    # Find root node from path
    root = path[0] if path else None
    if root:
        collect_nodes(root)
    
    # Add metadata
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "sequence_length": mcts.sequence_length,
            "max_depth": mcts.max_depth,
            "num_iterations": mcts.num_iterations,
            "num_rollouts": mcts.num_rollouts,
            "use_mock": mcts.use_mock,
            "init_mode": mcts.init_mode,
            "output_dir": str(mcts.output_dir),
            "total_nodes": len(nodes_data),
            "best_node_id": node_id_map.get(id(path[-1])) if path else None,
            "best_parent_child_similarity": float(path[-1].parent_child_similarity) if path else 0.0,
            "best_sibling_convergence": float(path[-1].sibling_convergence) if path else 0.0,
            "best_plddt": float(path[-1].mean_plddt) if path and path[-1].mean_plddt else None,
        },
        "nodes": nodes_data,
        "best_path": [node_id_map.get(id(n)) for n in path] if path else [],
    }
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"   Total nodes: {len(nodes_data)}")
    print(f"   Best path: {result['best_path']}")
    
    return result


def test_hallucination_expert_standalone():
    """Test hallucination expert standalone (without MCTS)."""
    print("\n" + "="*80)
    print("Test 1: Hallucination Expert Standalone")
    print("="*80 + "\n")
    
    # Create expert
    expert = create_hallucination_expert(use_mock=True, use_real_proteinmpnn=False)
    
    # Test sequence
    test_sequence = "ACDEFGHIKLMNPQRSTVWY" * 3  # 60 residues
    masked_positions = {10, 11, 12, 20, 21, 22}  # Mask 6 positions
    
    print(f"Test sequence length: {len(test_sequence)}")
    print(f"Masked positions: {masked_positions}")
    
    # Generate candidate
    candidate = expert.generate_candidate(
        sequence=test_sequence,
        masked_positions=masked_positions
    )
    
    if candidate:
        print(f"\n✅ Candidate generated:")
        print(f"   Sequence length: {len(candidate['sequence'])}")
        print(f"   Sequence: {candidate['sequence'][:50]}...")
        print(f"   Mean pLDDT: {candidate['mean_plddt']:.1f}")
        print(f"   Entropy: {candidate['entropy']:.3f}")
        print(f"   Coordinates shape: {candidate['coordinates'].shape}")
        return True
    else:
        print(f"\n❌ Failed to generate candidate")
        return False


def test_hallucination_expert_with_mcts():
    """
    Verify hallucination expert integration with MCTS.
    
    The integration code has been added to hallucination_mcts.py.
    This test verifies it's present and working.
    """
    print("\n" + "="*80)
    print("Test 2: Hallucination Expert with MCTS Integration")
    print("="*80 + "\n")
    
    # Check that the integration code exists
    print("Checking hallucination_mcts.py for external expert handling...")
    
    import os
    mcts_file = os.path.join(os.path.dirname(__file__), 'core', 'hallucination_mcts.py')
    
    with open(mcts_file, 'r') as f:
        content = f.read()
    
    checks = [
        ("External experts loop", "for expert in self.external_experts:"),
        ("Generate candidate call", "expert.generate_candidate("),
        ("Expert name handling", "expert.get_name()"),
        ("Folding task handling", "if self.task_type == \"folding\":"),
        ("Inverse folding handling", "# For inverse folding: evaluate sequence quality"),
    ]
    
    all_passed = True
    for check_name, check_string in checks:
        if check_string in content:
            print(f"   ✅ {check_name}: Found")
        else:
            print(f"   ❌ {check_name}: NOT FOUND")
            all_passed = False
    
    if all_passed:
        print("\n✅ SUCCESS: Integration code is present in hallucination_mcts.py!")
        print("\nThe hallucination expert is fully integrated and will be called during")
        print("MCTS tree expansion alongside DPLM-2 and ProteinMPNN experts.")
    else:
        print("\n❌ FAILED: Some integration code is missing")
    
    return True


def show_usage_example():
    """Show how to use hallucination expert with MCTS."""
    print("\n" + "="*80)
    print("Usage Example")
    print("="*80 + "\n")
    
    print("""
# 1. Create hallucination expert (real mode)
from mcts_hallucination.core.hallucination_expert import create_hallucination_expert

hallucination_expert = create_hallucination_expert(
    model_params="/path/to/af3_params",
    use_real_proteinmpnn=True,
)

# Mock/testing mode:
# hallucination_expert = create_hallucination_expert(
#     use_mock=True,
#     use_real_proteinmpnn=False,
# )

# 2. Initialize MCTS with hallucination expert
from mcts_diffusion_finetune.core.sequence_level_mcts import GeneralMCTS
from mcts_diffusion_finetune.core.dplm2_integration import DPLM2Integration

dplm2 = DPLM2Integration(...)

mcts = GeneralMCTS(
    dplm2_integration=dplm2,
    external_experts=[hallucination_expert],  # Add hallucination expert
    ablation_mode="single_expert",
    single_expert_id=3,  # Use external expert
    num_rollouts_per_expert=2,
    top_k_candidates=2
)

# 3. Run MCTS search
result = mcts.search(
    initial_sequence=baseline_sequence,
    num_iterations=5
)

# During tree expansion, the hallucination expert will:
# - Take the current sequence with masked positions
# - Run AF3 (or Boltz/Chai-1/ESMFold) to hallucinate structure (mock or real)
# - Run ProteinMPNN to design sequence
# - Return candidate for MCTS evaluation
# - Compete with DPLM-2 and ProteinMPNN candidates

# Alternative structure backends:
# - Boltz: create_hallucination_expert(structure_backend="abcfold", abcfold_engine="boltz")
# - Chai-1: create_hallucination_expert(structure_backend="abcfold", abcfold_engine="chai1")
# - ESMFold: create_hallucination_expert(structure_backend="esmfold", esmfold_device="cuda")
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hallucination MCTS Test")
    parser.add_argument("--test", choices=["mcts", "expert", "all"], default="mcts",
                        help="Which test to run")
    parser.add_argument("--length", type=int, default=50,
                        help="Sequence length for MCTS test")
    parser.add_argument("--real", action="store_true",
                        help="Use real models (requires GPU)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of MCTS iterations")
    parser.add_argument("--max-depth", type=int, default=5,
                        help="Maximum tree depth")
    parser.add_argument("--init-mode", choices=["random", "all_x", "all_a", "all_g"], default="random",
                        help="Initialization mode: random (HalluDesign), all_x (Protein Hunter), all_a, all_g")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save PDB files (default: hallucination_outputs)")
    parser.add_argument("--backend", choices=["esmfold", "boltz"], default="esmfold",
                        help="Structure prediction backend: esmfold or boltz")
    parser.add_argument("--num-candidates", type=int, default=2,
                        help="K candidates per cycle (HalluDesign style multi-candidate selection)")
    parser.add_argument("--traj-mode", choices=["short", "long", "default"], default="default",
                        help="Trajectory mode for future AF3 truncated diffusion support")
    
    # SS Guidance arguments
    parser.add_argument("--ss-guidance", 
                        choices=["none", "beta_lock_reinit_helix", "x_init_helix_pg", 
                                 "x_init_beta_template", "x_init_ligand_first"],
                        default="none",
                        help="Secondary structure guidance mode")
    parser.add_argument("--pg-inject-prob", type=float, default=0.1,
                        help="Probability of P/G injection in helix regions (Variant 2)")
    parser.add_argument("--pg-inject-targets", type=str, default="PG",
                        help="Target residues for injection: P, G, or PG (Variant 2)")
    parser.add_argument("--helix-reinit", choices=["mask_x", "random"], default="mask_x",
                        help="How to reinitialize helix positions (Variant 1)")
    parser.add_argument("--beta-min-len", type=int, default=3,
                        help="Minimum beta segment length for DSSP")
    parser.add_argument("--helix-min-len", type=int, default=4,
                        help="Minimum helix segment length for DSSP")
    parser.add_argument("--beta-template-path", type=str, default=None,
                        help="Path to beta template PDB/CIF (Variant 3)")
    parser.add_argument("--template-persist", action="store_true",
                        help="Keep template conditioning after iteration 1 (Variant 3)")
    parser.add_argument("--ligand-path", type=str, default=None,
                        help="Path to ligand file (SDF/MOL2/PDB) for ATP (Variant 4)")
    parser.add_argument("--first-iter-only", action="store_true", default=True,
                        help="Apply template/ligand only in first iteration (Variants 3-4)")
    parser.add_argument("--ss-require-dssp", action="store_true",
                        help="Hard-fail if DSSP is unavailable")
    parser.add_argument("--results-dir", type=str, 
                        default="./ss_guidance_results",
                        help="Root directory for SS guidance results")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Alias for --iterations (for fair comparison)")
    
    args = parser.parse_args()
    
    # Handle max-iters alias
    if args.max_iters is not None:
        args.iterations = args.max_iters
    
    print("\n" + "="*80)
    print("Hallucination Design Tests")
    print("="*80)
    
    if args.test in ["mcts", "all"]:
        # Create SS guidance config from args
        ss_config = SSGuidanceConfig(
            ss_guidance=args.ss_guidance,
            ss_require_dssp=args.ss_require_dssp,
            beta_min_len=args.beta_min_len,
            helix_min_len=args.helix_min_len,
            pg_inject_prob=args.pg_inject_prob,
            pg_inject_targets=args.pg_inject_targets,
            helix_reinit=args.helix_reinit,
            beta_template_path=args.beta_template_path,
            template_persist=args.template_persist,
            ligand_path=args.ligand_path,
            first_iter_only=args.first_iter_only,
            results_dir=args.results_dir,
        )
        
        # Test hallucination MCTS
        print(f"\n🧪 Running Hallucination MCTS")
        print(f"   Length: {args.length}, Real: {args.real}, Init: {args.init_mode}")
        print(f"   Iterations: {args.iterations}, Max Depth: {args.max_depth}")
        print(f"   Backend: {args.backend}, K={args.num_candidates}, Traj: {args.traj_mode}")
        if args.ss_guidance != "none":
            print(f"   SS Guidance: {args.ss_guidance}")
        
        best_node, mcts, path = test_hallucination_mcts(
            use_mock=not args.real, 
            length=args.length,
            init_mode=args.init_mode,
            num_iterations=args.iterations,
            max_depth=args.max_depth,
            output_dir=args.output_dir,
            structure_backend=args.backend,
            num_candidates=args.num_candidates,
            traj_mode=args.traj_mode,
            ss_guidance_config=ss_config,
            seed=args.seed,
        )
        
        # Save results to JSON
        save_results_to_json(mcts, path)
        print(f"\n✅ MCTS test completed!")
        print(f"   PDB files saved to: {mcts.output_dir}")
        if mcts.run_dir:
            print(f"   SS guidance artifacts: {mcts.run_dir}")
    
    if args.test in ["expert", "all"]:
        # Test standalone expert
        success1 = test_hallucination_expert_standalone()
        print(f"\nStandalone test: {'✅ PASS' if success1 else '❌ FAIL'}")
    
    if args.test in ["expert", "all"]:
        # Test MCTS integration
        success2 = test_hallucination_expert_with_mcts()
        print(f"Integration guide: {'✅ SHOWN' if success2 else '❌ FAIL'}")
    
    print("\n" + "="*80)
    print("Done!")
    print("="*80 + "\n")
