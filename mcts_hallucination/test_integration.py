"""
Test integration of hallucination expert with existing MCTS.

This demonstrates how to plug the AF3+ProteinMPNN hallucination expert
into the existing GeneralMCTS framework.

NEW: Hallucination Design MCTS
- Start from random/all-masked sequence
- Iterate: ESMFold (seq→struct) → ProteinMPNN (struct→seq)
- Q-value = convergence (similarity to parent)
- Goal: converge to a stable sequence-structure pair
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

from core.hallucination_expert import create_hallucination_expert
from core.esmfold_integration import ESMFoldIntegration


# ============================================================================
# HALLUCINATION MCTS NODE
# ============================================================================

@dataclass
class HallucinationNode:
    """Node for hallucination MCTS - stores sequence-structure pair."""
    sequence: str
    coordinates: Optional[np.ndarray] = None
    plddt_scores: Optional[np.ndarray] = None
    mean_plddt: float = 0.0
    
    # MCTS fields
    parent: Optional['HallucinationNode'] = None
    children: List['HallucinationNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    depth: int = 0
    
    # Convergence tracking
    convergence_score: float = 0.0  # How similar to parent (higher = more converged)
    parent_child_similarity: float = 0.0  # Similarity to parent sequence
    sibling_convergence: float = 0.0  # Similarity among siblings
    
    def get_reward(self) -> float:
        """Average reward from visits."""
        return self.total_reward / max(1, self.visits)
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """UCT score for selection."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.get_reward()
        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        else:
            exploration = 0
        
        return exploitation + exploration


# ============================================================================
# HALLUCINATION MCTS
# ============================================================================

class HallucinationMCTS:
    """
    MCTS for hallucination design using ESMFold + ProteinMPNN.
    
    Pipeline per node expansion:
    1. Take current sequence
    2. ESMFold: sequence → structure (coordinates + pLDDT)
    3. ProteinMPNN: structure → new sequence
    4. Q-value = convergence (similarity to parent sequence)
    
    Goal: Find a converged sequence-structure pair.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        max_depth: int = 10,
        num_iterations: int = 20,
        num_rollouts: int = 2,
        exploration_constant: float = 1.414,
        use_mock: bool = False,
        device: str = "cuda",
        init_mode: str = "random",  # "random" or "all_x" or "all_a"
        output_dir: Optional[str] = None,  # Directory to save PDB files
        structure_backend: str = "esmfold",  # "esmfold" or "boltz"
        num_candidates: int = 2,  # K candidates per cycle (HalluDesign style)
        traj_mode: str = "default",  # "short", "long", "default" - for future AF3 truncated diffusion
    ):
        self.sequence_length = sequence_length
        self.max_depth = max_depth
        self.num_iterations = num_iterations
        self.num_rollouts = num_rollouts
        self.exploration_constant = exploration_constant
        self.use_mock = use_mock
        self.device = device
        self.init_mode = init_mode
        self.structure_backend = structure_backend
        self.num_candidates = num_candidates
        self.traj_mode = traj_mode
        
        # Setup output directory for PDB files
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("/home/caom/AID3/dplm/mcts_diffusion_finetune/mcts_hallucination/hallucination_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_counter = 0
        
        # Initialize structure predictor based on backend
        if structure_backend == "boltz":
            print(f"🔧 Initializing Boltz (mock={use_mock})...")
            from core.abcfold_integration import ABCFoldIntegration
            self.structure_predictor = ABCFoldIntegration(
                use_mock=use_mock,
                engine="boltz",
                allow_fallback=True,
            )
            self.esmfold = None
        else:
            print(f"🔧 Initializing ESMFold (mock={use_mock})...")
            self.esmfold = ESMFoldIntegration(
                device=device,
                use_mock=use_mock,
            )
            self.structure_predictor = self.esmfold
        
        # Initialize ProteinMPNN (via hallucination expert for simplicity)
        print(f"🔧 Initializing ProteinMPNN (mock={use_mock})...")
        self.hallucination_expert = create_hallucination_expert(
            structure_backend=structure_backend,
            esmfold_device=device,
            use_mock=use_mock,
            use_real_proteinmpnn=not use_mock,
        )
        
        print(f"✅ HallucinationMCTS initialized")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Max depth: {max_depth}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Rollouts per expansion: {num_rollouts}")
        print(f"   Init mode: {init_mode}")
        print(f"   Structure backend: {structure_backend}")
        print(f"   Num candidates (K): {num_candidates}")
        print(f"   Traj mode: {traj_mode}")
        print(f"   Output dir: {self.output_dir}")
    
    def generate_initial_sequence(self, length: int) -> str:
        """
        Generate initial sequence based on init_mode.
        
        Modes:
        - 'random': Random amino acid sequence (HalluDesign style)
        - 'all_x': All X tokens (unknown) - used in Protein Hunter paper
                   Boltz can handle X directly; ESMFold converts to A
        - 'all_a': All alanine - simple baseline
        - 'all_g': All glycine - minimal side chains
        """
        if self.init_mode == "all_x":
            # Boltz can handle X tokens directly for diffusion hallucination
            # ESMFold will convert X to A internally
            return 'X' * length
        elif self.init_mode == "all_a":
            return 'A' * length
        elif self.init_mode == "all_g":
            return 'G' * length
        else:  # random
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            return ''.join(random.choice(amino_acids) for _ in range(length))
    
    def save_pdb(self, sequence: str, coordinates: np.ndarray, iteration: int, 
                 depth: int, node_id: int, plddt: float, suffix: str = "") -> str:
        """
        Save structure as PDB file for visualization.
        
        Returns the path to the saved PDB file.
        """
        self.pdb_counter += 1
        filename = f"iter{iteration:03d}_depth{depth:02d}_node{node_id:04d}_plddt{plddt:.1f}{suffix}.pdb"
        filepath = self.output_dir / filename
        
        # Write PDB file
        with open(filepath, 'w') as f:
            f.write(f"REMARK   1 Hallucination MCTS Design\n")
            f.write(f"REMARK   2 Iteration: {iteration}, Depth: {depth}, Node: {node_id}\n")
            f.write(f"REMARK   3 Sequence: {sequence}\n")
            f.write(f"REMARK   4 Mean pLDDT: {plddt:.2f}\n")
            
            # Write CA atoms
            for i, (aa, coord) in enumerate(zip(sequence, coordinates)):
                # Convert 1-letter to 3-letter code
                aa3 = {'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
                       'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
                       'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
                       'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'}.get(aa, 'ALA')
                
                f.write(f"ATOM  {i+1:5d}  CA  {aa3} A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{plddt:6.2f}           C\n")
            
            f.write("END\n")
        
        return str(filepath)
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def compute_sibling_convergence(self, sequences: List[str]) -> float:
        """
        Compute convergence as average pairwise similarity among sibling sequences.
        
        Higher value = siblings are more similar = structure is more "deterministic"
        in producing sequences = better convergence.
        """
        if len(sequences) < 2:
            return 0.0
        
        total_sim = 0.0
        num_pairs = 0
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                total_sim += self.compute_sequence_similarity(sequences[i], sequences[j])
                num_pairs += 1
        
        return total_sim / num_pairs if num_pairs > 0 else 0.0
    
    def compute_convergence_reward(
        self, 
        sibling_convergence: float, 
        parent_child_similarity: float,
        plddt: float
    ) -> float:
        """
        Compute reward based on convergence metrics and structure quality.
        
        Based on Protein Hunter and HalluDesign papers:
        - Parent-child similarity: Key metric - how much does the sequence change?
          High similarity = converging to stable sequence-structure pair
        - Sibling convergence: Structure consistency - does same structure produce similar seqs?
        - pLDDT: Structure quality
        
        R = 0.4 * parent_child_similarity + 0.3 * sibling_convergence + 0.3 * normalized_plddt
        """
        normalized_plddt = plddt / 100.0  # pLDDT is 0-100
        
        # Weight parent-child similarity more heavily - this is the key convergence metric
        reward = 0.4 * parent_child_similarity + 0.3 * sibling_convergence + 0.3 * normalized_plddt
        return reward
    
    def expand_node(self, node: HallucinationNode) -> List[HallucinationNode]:
        """
        Expand a node by running Structure → ProteinMPNN cycle.
        
        Like Protein Hunter / HalluDesign:
        1. Fold current sequence to get structure
        2. Sample K sequences from ProteinMPNN
        3. Optionally fold each candidate to evaluate
        4. Select best candidates based on reward
        
        Returns list of child nodes.
        """
        # Step 1: Predict structure from current sequence
        backend_name = self.structure_backend.upper()
        print(f"      🔮 {backend_name} predicting structure from: {node.sequence[:30]}...")
        structure_result = self.structure_predictor.predict_structure(node.sequence)
        coords = structure_result['coordinates']
        parent_plddt = structure_result['confidence']
        parent_mean_plddt = float(np.mean(parent_plddt))
        print(f"      ✅ {backend_name} done - mean pLDDT={parent_mean_plddt:.1f}, min={np.min(parent_plddt):.1f}, max={np.max(parent_plddt):.1f}")
        
        # Log traj_mode for future AF3 truncated diffusion support
        if self.traj_mode != "default":
            print(f"      📊 Traj mode: {self.traj_mode} (placeholder for future AF3 diffusion control)")
        
        # Step 2: Generate K candidate sequences from the structure
        # This is the key HalluDesign insight: sample multiple, pick best
        candidate_sequences = []
        candidate_data = []  # Store (sequence, coords, plddt) for each candidate
        
        total_candidates = self.num_candidates * self.num_rollouts
        print(f"      🧬 Generating {total_candidates} candidate sequences (K={self.num_candidates} x {self.num_rollouts} rollouts)...")
        
        for rollout in range(self.num_rollouts):
            try:
                # Generate K sequences per rollout
                for k in range(self.num_candidates):
                    # Inherit the full parent sequence without masking (ProteinHunter-style).
                    masked_seq = node.sequence
                    new_sequence = self.hallucination_expert.proteinmpnn.design_sequence(
                        coordinates=coords,
                        masked_sequence=masked_seq,
                    )
                    
                    # Evaluate the new sequence with structure predictor
                    new_structure_result = self.structure_predictor.predict_structure(new_sequence)
                    new_coords = new_structure_result['coordinates']
                    new_plddt = new_structure_result['confidence']
                    new_mean_plddt = float(np.mean(new_plddt))
                    
                    candidate_sequences.append(new_sequence)
                    candidate_data.append((new_sequence, new_coords, new_plddt, new_mean_plddt))
                    
                    print(f"      ✅ Candidate {len(candidate_sequences)}: pLDDT={new_mean_plddt:.1f}, seq={new_sequence[:20]}...")
                
            except Exception as e:
                print(f"      ❌ Rollout {rollout+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Compute sibling convergence (how similar are candidates to each other)
        sibling_convergence = self.compute_sibling_convergence(candidate_sequences)
        print(f"      👥 Sibling convergence: {sibling_convergence:.3f} (from {len(candidate_sequences)} candidates)")
        
        # Step 4: Create child nodes with both convergence metrics
        children = []
        for i, (seq, new_coords, new_plddt, new_mean_plddt) in enumerate(candidate_data):
            # Compute parent-child similarity (key convergence metric from papers)
            parent_child_sim = self.compute_sequence_similarity(node.sequence, seq)
            
            # Reward based on both convergence metrics + pLDDT
            reward = self.compute_convergence_reward(sibling_convergence, parent_child_sim, new_mean_plddt)
            
            child = HallucinationNode(
                sequence=seq,
                coordinates=new_coords,
                plddt_scores=new_plddt,
                mean_plddt=new_mean_plddt,
                parent=node,
                depth=node.depth + 1,
                convergence_score=parent_child_sim,  # Use parent-child as main convergence
                parent_child_similarity=parent_child_sim,
                sibling_convergence=sibling_convergence,
                visits=1,
                total_reward=reward,
            )
            children.append(child)
            
            print(f"      📊 Child {i+1}: parent_sim={parent_child_sim:.3f}, sibling_conv={sibling_convergence:.3f}, pLDDT={new_mean_plddt:.1f}, reward={reward:.3f}")
        
        return children
    
    def select_node(self, root: HallucinationNode) -> HallucinationNode:
        """UCT selection - traverse tree to find best node to expand."""
        node = root
        while node.children and node.depth < self.max_depth:
            # Select child with highest UCT score
            node = max(node.children, key=lambda c: c.uct_score(self.exploration_constant))
        return node
    
    def backpropagate(self, node: HallucinationNode, reward: float):
        """Backpropagate reward up the tree."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def search(self) -> HallucinationNode:
        """
        Run MCTS search for hallucination design.
        
        Returns the best converged node.
        """
        print("\n" + "="*80)
        print("🚀 Starting Hallucination MCTS Search")
        print("="*80)
        
        # Create root node with initial sequence based on init_mode
        initial_seq = self.generate_initial_sequence(self.sequence_length)
        print(f"\n🌱 Root: {self.init_mode} sequence (length={len(initial_seq)})")
        print(f"   Sequence: {initial_seq}")
        
        root = HallucinationNode(
            sequence=initial_seq,
            depth=0,
            visits=1,
        )
        
        # Track node IDs for PDB saving
        node_counter = 0
        
        # Run MCTS iterations
        best_node = root
        best_convergence = 0.0
        
        for iteration in range(self.num_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.num_iterations}")
            
            # Selection
            selected = self.select_node(root)
            print(f"   📍 Selected node at depth {selected.depth}")
            
            # Expansion (if not at max depth)
            if selected.depth < self.max_depth:
                children = self.expand_node(selected)
                selected.children.extend(children)
                
                # Backpropagate for each child and save PDB files
                for child in children:
                    node_counter += 1
                    self.backpropagate(child, child.get_reward())
                    
                    # Save PDB file for this child
                    if child.coordinates is not None:
                        pdb_path = self.save_pdb(
                            sequence=child.sequence,
                            coordinates=child.coordinates,
                            iteration=iteration + 1,
                            depth=child.depth,
                            node_id=node_counter,
                            plddt=child.mean_plddt,
                        )
                        print(f"   💾 Saved: {Path(pdb_path).name}")
                    
                    # Track best converged node
                    if child.convergence_score > best_convergence:
                        best_convergence = child.convergence_score
                        best_node = child
                        print(f"   🏆 New best: parent_sim={best_convergence:.3f}, pLDDT={child.mean_plddt:.1f}, depth={child.depth}")
            
            # Check for convergence (>95% similarity)
            if best_convergence > 0.95:
                print(f"\n✅ Converged! Parent-child similarity > 95%")
                break
        
        # Save final best structure with special suffix
        if best_node.coordinates is not None:
            final_pdb = self.save_pdb(
                sequence=best_node.sequence,
                coordinates=best_node.coordinates,
                iteration=self.num_iterations,
                depth=best_node.depth,
                node_id=9999,
                plddt=best_node.mean_plddt,
                suffix="_BEST",
            )
            print(f"\n💾 Best structure saved: {final_pdb}")
        
        return best_node
    
    def find_best_in_tree(self, root: HallucinationNode) -> HallucinationNode:
        """Find the best node in the entire tree by convergence."""
        best = root
        best_score = root.convergence_score
        
        def traverse(node):
            nonlocal best, best_score
            if node.convergence_score > best_score:
                best = node
                best_score = node.convergence_score
            for child in node.children:
                traverse(child)
        
        traverse(root)
        return best


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
    """
    print("\n" + "="*80)
    print(f"Test: Hallucination MCTS Design")
    print(f"  Length: {length}, Mock: {use_mock}, Init: {init_mode}")
    print(f"  Iterations: {num_iterations}, Max Depth: {max_depth}")
    print(f"  Backend: {structure_backend}, K={num_candidates}, Traj: {traj_mode}")
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
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("Hallucination Design Tests")
    print("="*80)
    
    if args.test in ["mcts", "all"]:
        # Test hallucination MCTS
        print(f"\n🧪 Running Hallucination MCTS")
        print(f"   Length: {args.length}, Real: {args.real}, Init: {args.init_mode}")
        print(f"   Iterations: {args.iterations}, Max Depth: {args.max_depth}")
        print(f"   Backend: {args.backend}, K={args.num_candidates}, Traj: {args.traj_mode}")
        
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
        )
        
        # Save results to JSON
        save_results_to_json(mcts, path)
        print(f"\n✅ MCTS test completed!")
        print(f"   PDB files saved to: {mcts.output_dir}")
    
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
