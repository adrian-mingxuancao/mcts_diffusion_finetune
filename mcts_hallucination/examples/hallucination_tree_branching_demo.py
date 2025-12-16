#!/usr/bin/env python3
"""
Branching hallucination tree demo.

This script grows a small tree of hallucinated sequences/structures without
running the full MCTS stack. Each node:
  1. Masks a random subset of residues.
  2. Asks the hallucination expert to hallucinate a structure + inverse-folded sequence.
  3. Stores summary statistics (sequence, mean pLDDT, entropy, optional refold result).

Use this to sanity-check new structure backends (AF3, Boltz, Chai-1, ESMFold) or
to visualize how repeated hallucination steps evolve candidates.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import List, Optional, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.hallucination_expert import create_hallucination_expert  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Branching hallucination tree demo")
    parser.add_argument("--length", type=int, default=50, help="Initial sequence length")
    parser.add_argument("--depth", type=int, default=3, help="Tree depth (root depth=0)")
    parser.add_argument("--branching", type=int, default=2, help="Number of children per node")
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Fraction of residues to re-mask when expanding a node")
    parser.add_argument("--structure-backend", choices=["abcfold", "esmfold"], default="abcfold")
    parser.add_argument("--abcfold-engine", choices=["af3", "boltz", "chai1"], default="af3")
    parser.add_argument("--model-params", type=str, default=None, help="AF3 model params path (only for AF3 engine)")
    parser.add_argument("--abcfold-database-dir", type=str, default=None, help="AF3 database dir when MMseqs2 disabled")
    parser.add_argument("--disable-mmseqs", action="store_true", help="Disable the --mmseqs2 flag when calling ABCFold")
    parser.add_argument("--esmfold-model", type=str, default="facebook/esmfold_v1", help="HuggingFace model to load for ESMFold backend")
    parser.add_argument("--esmfold-device", type=str, default="cuda", help="Device for ESMFold inference (cuda/cpu)")
    parser.add_argument(
        "--no-real-proteinmpnn",
        action="store_false",
        dest="use_real_proteinmpnn",
        help="Disable the real ProteinMPNN model (mock inverse folding).",
    )
    parser.set_defaults(use_real_proteinmpnn=True)
    parser.add_argument("--proteinmpnn-device", type=str, default="cuda", help="Device for ProteinMPNN inference")
    parser.add_argument("--proteinmpnn-temperature", type=float, default=1.0, help="ProteinMPNN sampling temperature")
    parser.add_argument("--use-mock", action="store_true", help="Force mock predictions for the entire pipeline (testing only)")
    parser.add_argument("--initial-sequence", type=str, default=None, help="Optional starting sequence; defaults to all 'X'")
    parser.add_argument("--refold-designed", action="store_true", help="Re-run structure prediction on the designed sequence for validation")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save tree nodes as JSON")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def generate_initial_sequence(length: int, seq: Optional[str]) -> str:
    if seq:
        seq = seq.strip().upper()
        if len(seq) != length:
            raise ValueError(f"Provided initial sequence length ({len(seq)}) != requested {length}")
        return seq
    return "X" * length


def sample_mask_positions(seq_len: int, mask_ratio: float) -> List[int]:
    mask_count = max(1, int(mask_ratio * seq_len))
    return random.sample(range(seq_len), mask_count)


@dataclass
class TreeNode:
    node_id: int
    parent_id: Optional[int]
    depth: int
    sequence: str
    masked_positions: List[int]
    mean_plddt: float
    entropy: float
    pae_mean: float
    coordinate_shape: List[int]
    validation_mean_plddt: Optional[float] = None


def expand_node(
    node: TreeNode,
    expert,
    branching: int,
    mask_ratio: float,
    refold: bool,
) -> List[TreeNode]:
    children: List[TreeNode] = []
    seq_len = len(node.sequence)
    
    for _ in range(branching):
        masked_positions = sample_mask_positions(seq_len, mask_ratio)
        masked_positions_set = set(masked_positions)
        candidate = expert.generate_candidate(
            sequence=node.sequence,
            masked_positions=masked_positions_set,
        )
        if candidate is None:
            print(f"   ⚠️ Candidate generation failed for node {node.node_id}; skipping child.")
            continue
        
        validation_mean = None
        if refold:
            print("      🔄 Refolding designed sequence for validation...")
            validation = expert.structure_predictor.predict_structure(candidate["sequence"])
            validation_mean = float(validation["confidence"].mean())
        
        child = TreeNode(
            node_id=-1,  # placeholder, will be set later
            parent_id=node.node_id,
            depth=node.depth + 1,
            sequence=candidate["sequence"],
            masked_positions=masked_positions,
            mean_plddt=float(candidate["mean_plddt"]),
            entropy=float(candidate["entropy"]),
            pae_mean=float(candidate["pae_mean"]),
            coordinate_shape=list(candidate["coordinates"].shape),
            validation_mean_plddt=validation_mean,
        )
        children.append(child)
    
    return children


def main():
    args = parse_args()
    random.seed(args.seed)
    
    expert = create_hallucination_expert(
        model_params=args.model_params,
        use_mock=args.use_mock,
        structure_backend=args.structure_backend,
        abcfold_engine=args.abcfold_engine,
        abcfold_database_dir=args.abcfold_database_dir,
        abcfold_use_mmseqs=not args.disable_mmseqs,
        esmfold_model_name=args.esmfold_model,
        esmfold_device=args.esmfold_device,
        use_real_proteinmpnn=args.use_real_proteinmpnn,
        proteinmpnn_device=args.proteinmpnn_device,
        proteinmpnn_temperature=args.proteinmpnn_temperature,
    )
    
    root_sequence = generate_initial_sequence(args.length, args.initial_sequence)
    node_counter = 0
    root = TreeNode(
        node_id=node_counter,
        parent_id=None,
        depth=0,
        sequence=root_sequence,
        masked_positions=[],
        mean_plddt=float("nan"),
        entropy=0.0,
        pae_mean=float("nan"),
        coordinate_shape=[args.length, 3],
    )
    
    nodes: List[TreeNode] = [root]
    frontier = [root]
    
    while frontier:
        current = frontier.pop(0)
        if current.depth >= args.depth:
            continue
        
        print(f"\n=== Expanding node {current.node_id} at depth {current.depth} ===")
        children = expand_node(
            node=current,
            expert=expert,
            branching=args.branching,
            mask_ratio=args.mask_ratio,
            refold=args.refold_designed,
        )
        
        for child in children:
            node_counter += 1
            child.node_id = node_counter
            nodes.append(child)
            frontier.append(child)
            print(
                f"   ✅ Child {child.node_id}: depth={child.depth}, "
                f"mean pLDDT={child.mean_plddt:.1f}, entropy={child.entropy:.3f}"
            )
    
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [asdict(node) for node in nodes]
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nSaved {len(nodes)} nodes to {out_path}")
    else:
        print(f"\nGenerated {len(nodes)} nodes (root + children).")


if __name__ == "__main__":
    main()
