"""
Hallucination MCTS - Standalone MCTS for Protein Hallucination Design

This module implements MCTS-guided protein hallucination using:
- Structure prediction: ESMFold, Boltz, or Chai-1 (via ABCFold)
- Inverse folding: ProteinMPNN or NA-MPNN
- SS guidance: DSSP-based secondary structure analysis

Pipeline per iteration:
1. Sequence -> Structure (ESMFold/Boltz/Chai)
2. Structure -> New Sequence (ProteinMPNN)
3. Evaluate convergence (parent-child similarity, sibling convergence, pLDDT)
4. UCT selection and backpropagation

This is a STANDALONE module that does NOT depend on dplm2_integration.
All structure prediction and inverse folding use local core modules.
"""

import math
import random
import csv
import json
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import sys
import os
import numpy as np

# Add the mcts_hallucination directory to path
HALLUCINATION_DIR = Path(__file__).resolve().parent.parent
if str(HALLUCINATION_DIR) not in sys.path:
    sys.path.insert(0, str(HALLUCINATION_DIR))

# Import local core modules
from core.hallucination_expert import create_hallucination_expert, HallucinationExpert
from core.esmfold_integration import ESMFoldIntegration
from core.abcfold_integration import ABCFoldIntegration
from core.ss_guidance import (
    SSGuidanceConfig,
    SSGuidance,
    DSSPResult,
    EditLogEntry,
    create_run_directory,
)


@dataclass
class HallucinationNode:
    """Node for hallucination MCTS - stores sequence-structure pair.
    
    Aligned with MCTSNode from sequence_level_mcts.py for compatibility.
    """
    sequence: str
    coordinates: Optional[np.ndarray] = None
    plddt_scores: Optional[np.ndarray] = None
    mean_plddt: float = 0.0
    
    # Position tracking (aligned with sequence_level_mcts.py)
    mutable_positions: Set[int] = field(default_factory=set)
    frozen_positions: Set[int] = None
    
    # MCTS fields
    parent: Optional['HallucinationNode'] = None
    children: List['HallucinationNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    reward: float = 0.0  # Single reward value (aligned with sequence_level_mcts.py)
    depth: int = 0
    
    # Structure tokens (aligned with sequence_level_mcts.py)
    structure_tokens: Optional[str] = None
    rmsd: Optional[float] = None
    tm_score: Optional[float] = None
    
    # Convergence tracking (hallucination-specific)
    convergence_score: float = 0.0
    parent_child_similarity: float = 0.0
    sibling_convergence: float = 0.0
    
    # PH-UCT components
    entropy: float = 0.0
    novelty: float = 0.0
    expert_source: str = None
    
    def __post_init__(self):
        """Auto-compute frozen positions if not provided."""
        if self.frozen_positions is None and self.mutable_positions:
            all_positions = set(range(len(self.sequence)))
            self.frozen_positions = all_positions - self.mutable_positions
    
    @property
    def masked_positions(self) -> Set[int]:
        """Backward compatibility - return mutable positions."""
        return self.mutable_positions
    
    def get_reward(self) -> float:
        """Average reward from visits."""
        return self.total_reward / max(1, self.visits)
    
    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """Standard UCT score (aligned with sequence_level_mcts.py)."""
        if self.visits == 0:
            return float('inf')
        
        # Base UCB1 score
        if self.parent and self.parent.visits > 0:
            exploitation = self.total_reward / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration
        else:
            return self.total_reward / self.visits
    
    def ph_uct_score(self, exploration_constant: float = 1.414,
                     entropy_weight: float = 0.1, novelty_weight: float = 0.05) -> float:
        """PH-UCT score with entropy and novelty bonuses."""
        if self.visits == 0:
            return float('inf')
        
        # Base UCB1 score
        if self.parent and self.parent.visits > 0:
            exploitation = self.total_reward / self.visits
            exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
            ucb_score = exploitation + exploration
        else:
            ucb_score = self.total_reward / self.visits
        
        # Add entropy and novelty bonuses for PH-UCT
        entropy_bonus = entropy_weight * self.entropy
        novelty_bonus = novelty_weight * self.novelty
        
        return ucb_score + entropy_bonus + novelty_bonus


class HallucinationMCTS:
    """
    MCTS for hallucination design using ESMFold/Boltz/Chai + ProteinMPNN.
    
    Pipeline per node expansion:
    1. Take current sequence
    2. Structure predictor: sequence -> structure (coordinates + pLDDT)
    3. ProteinMPNN: structure -> new sequence
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
        init_mode: str = "random",
        output_dir: Optional[str] = None,
        structure_backend: str = "esmfold",
        abcfold_engine: str = "boltz",
        num_candidates: int = 2,
        traj_mode: str = "default",
        ss_guidance_config: Optional[SSGuidanceConfig] = None,
        seed: Optional[int] = None,
        use_ph_uct: bool = True,
        entropy_weight: float = 0.1,
        novelty_weight: float = 0.05,
    ):
        """
        Initialize HallucinationMCTS.
        
        Args:
            sequence_length: Length of sequences to design
            max_depth: Maximum tree depth
            num_iterations: Number of MCTS iterations
            num_rollouts: Rollouts per expansion
            exploration_constant: UCT exploration constant
            use_mock: Use mock models for testing
            device: Device for models (cuda/cpu)
            init_mode: Initialization mode (random, all_x, all_a, all_g)
            output_dir: Directory to save PDB files
            structure_backend: Structure prediction backend (esmfold, boltz, chai1, abcfold)
            abcfold_engine: Engine for ABCFold (af3, boltz, chai1)
            num_candidates: K candidates per cycle (HalluDesign style)
            traj_mode: Trajectory mode for future AF3 diffusion control
            ss_guidance_config: Secondary structure guidance configuration
            seed: Random seed for reproducibility
            use_ph_uct: Use PH-UCT (with entropy/novelty) vs standard UCT
            entropy_weight: Weight for entropy bonus in PH-UCT
            novelty_weight: Weight for novelty bonus in PH-UCT
        """
        self.sequence_length = sequence_length
        self.max_depth = max_depth
        self.num_iterations = num_iterations
        self.num_rollouts = num_rollouts
        self.exploration_constant = exploration_constant
        self.use_mock = use_mock
        self.device = device
        self.init_mode = init_mode
        self.structure_backend = structure_backend
        self.abcfold_engine = abcfold_engine
        self.num_candidates = num_candidates
        self.traj_mode = traj_mode
        self.use_ph_uct = use_ph_uct
        self.entropy_weight = entropy_weight
        self.novelty_weight = novelty_weight
        
        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = HALLUCINATION_DIR / "hallucination_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_counter = 0
        
        # Initialize structure predictor
        self._init_structure_predictor()
        
        # Initialize inverse folding (ProteinMPNN via hallucination expert)
        print(f"Initializing ProteinMPNN (mock={use_mock})...")
        self.hallucination_expert = create_hallucination_expert(
            structure_backend=structure_backend,
            abcfold_engine=abcfold_engine,
            esmfold_device=device,
            use_mock=use_mock,
            use_real_proteinmpnn=not use_mock,
        )
        
        # Initialize SS guidance
        self.ss_guidance_config = ss_guidance_config or SSGuidanceConfig()
        self.ss_guidance = None
        if self.ss_guidance_config.ss_guidance != "none":
            try:
                self.ss_guidance = SSGuidance(self.ss_guidance_config)
            except Exception as e:
                print(f"Warning: SS guidance initialization failed: {e}")
                self.ss_guidance = None
        
        # Set random seed
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create run directory for SS guidance artifacts
        self.run_dir = None
        if self.ss_guidance and self.ss_guidance_config.ss_guidance != "none":
            self.run_dir = create_run_directory(self.ss_guidance_config, seed)
            print(f"   SS guidance: {self.ss_guidance_config.ss_guidance}")
            print(f"   Run dir: {self.run_dir}")
        
        # Override init_mode based on SS guidance variant
        if self.ss_guidance_config.ss_guidance == "beta_lock_reinit_helix":
            self.init_mode = "random"
        elif self.ss_guidance_config.ss_guidance in ("x_init_helix_pg", "x_init_beta_template", "x_init_ligand_first"):
            self.init_mode = "all_x"
        
        print(f"HallucinationMCTS initialized")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Max depth: {max_depth}")
        print(f"   Iterations: {num_iterations}")
        print(f"   Init mode: {self.init_mode}")
        print(f"   Structure backend: {structure_backend}")
        print(f"   Use PH-UCT: {use_ph_uct}")
        print(f"   Output dir: {self.output_dir}")
    
    def _init_structure_predictor(self):
        """Initialize structure prediction backend."""
        backend = self.structure_backend.lower()
        
        if backend == "esmfold":
            print(f"Initializing ESMFold (mock={self.use_mock})...")
            self.structure_predictor = ESMFoldIntegration(
                device=self.device,
                use_mock=self.use_mock,
            )
        elif backend in ("boltz", "chai1", "af3"):
            print(f"Initializing ABCFold/{backend} (mock={self.use_mock})...")
            self.structure_predictor = ABCFoldIntegration(
                use_mock=self.use_mock,
                engine=backend if backend != "af3" else "af3",
                allow_fallback=True,
            )
        elif backend == "abcfold":
            print(f"Initializing ABCFold/{self.abcfold_engine} (mock={self.use_mock})...")
            self.structure_predictor = ABCFoldIntegration(
                use_mock=self.use_mock,
                engine=self.abcfold_engine,
                allow_fallback=True,
            )
        else:
            raise ValueError(f"Unknown structure backend: {backend}")
    
    def generate_initial_sequence(self, length: int) -> str:
        """Generate initial sequence based on init_mode."""
        if self.init_mode == "all_x":
            return 'X' * length
        elif self.init_mode == "all_a":
            return 'A' * length
        elif self.init_mode == "all_g":
            return 'G' * length
        else:  # random
            amino_acids = "ACDEFGHIKLMNPQRSTVWY"
            return ''.join(random.choice(amino_acids) for _ in range(length))
    
    def compute_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence identity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
        return matches / len(seq1)
    
    def compute_sibling_convergence(self, sequences: List[str]) -> float:
        """Compute average pairwise similarity among sibling sequences."""
        if len(sequences) < 2:
            return 0.0
        
        total_sim = 0.0
        num_pairs = 0
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                total_sim += self.compute_sequence_similarity(sequences[i], sequences[j])
                num_pairs += 1
        
        return total_sim / num_pairs if num_pairs > 0 else 0.0
    
    def compute_novelty(self, sequence: str, parent_node: HallucinationNode) -> float:
        """Compute novelty based on Hamming distance to siblings."""
        if not parent_node.children:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for sibling in parent_node.children:
            if sibling.sequence != sequence:
                distance = sum(1 for a, b in zip(sequence, sibling.sequence) if a != b)
                total_distance += distance / len(sequence)
                count += 1
        
        return total_distance / count if count > 0 else 1.0
    
    def compute_entropy(self, sequence: str, coordinates: np.ndarray) -> float:
        """
        Compute sequence entropy based on amino acid distribution.
        
        This is a simplified entropy calculation that doesn't require dplm2_integration.
        For more sophisticated entropy, use the hallucination_expert's internal methods.
        """
        # Shannon entropy of amino acid distribution
        aa_counts = {}
        for aa in sequence:
            if aa != 'X':
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        total = sum(aa_counts.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in aa_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by max entropy (log2(20) for 20 amino acids)
        max_entropy = math.log2(20)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def compute_convergence_reward(
        self, 
        sibling_convergence: float, 
        parent_child_similarity: float,
        plddt: float,
        entropy: float = 0.0,
    ) -> float:
        """
        Compute reward based on convergence metrics and structure quality.
        
        R = 0.4 * parent_child_similarity + 0.3 * sibling_convergence + 0.3 * normalized_plddt
        """
        normalized_plddt = plddt / 100.0
        reward = 0.4 * parent_child_similarity + 0.3 * sibling_convergence + 0.3 * normalized_plddt
        return reward
    
    def save_pdb(self, sequence: str, coordinates: np.ndarray, iteration: int, 
                 depth: int, node_id: int, plddt: float, suffix: str = "") -> str:
        """Save structure as PDB file."""
        self.pdb_counter += 1
        filename = f"iter{iteration:03d}_depth{depth:02d}_node{node_id:04d}_plddt{plddt:.1f}{suffix}.pdb"
        filepath = self.output_dir / filename
        
        aa3_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
            'X': 'UNK'
        }
        
        with open(filepath, 'w') as f:
            f.write(f"REMARK   1 Hallucination MCTS Design\n")
            f.write(f"REMARK   2 Iteration: {iteration}, Depth: {depth}, Node: {node_id}\n")
            f.write(f"REMARK   3 Sequence: {sequence}\n")
            f.write(f"REMARK   4 Mean pLDDT: {plddt:.2f}\n")
            
            for i, (aa, coord) in enumerate(zip(sequence, coordinates)):
                aa3 = aa3_map.get(aa, 'ALA')
                f.write(f"ATOM  {i+1:5d}  CA  {aa3} A{i+1:4d}    "
                        f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00{plddt:6.2f}           C\n")
            
            f.write("END\n")
        
        return str(filepath)
    
    def _save_pdb_for_dssp(self, sequence: str, coordinates: np.ndarray, 
                           plddt_scores: np.ndarray, filepath: str):
        """Save PDB with backbone atoms for DSSP analysis."""
        aa3_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
            'X': 'UNK'
        }
        
        with open(filepath, 'w') as f:
            f.write(f"REMARK   1 Structure for DSSP analysis\n")
            atom_num = 1
            
            for i, (aa, ca_coord) in enumerate(zip(sequence, coordinates)):
                aa3 = aa3_map.get(aa, 'ALA')
                plddt = float(plddt_scores[i]) if i < len(plddt_scores) else 50.0
                res_num = i + 1
                
                # Generate approximate backbone atoms from CA position
                n_coord = ca_coord + np.array([-1.46, 0.0, 0.0])
                c_coord = ca_coord + np.array([1.52, 0.0, 0.0])
                o_coord = c_coord + np.array([0.0, 1.23, 0.0])
                
                for atom_name, coord in [("N", n_coord), ("CA", ca_coord), ("C", c_coord), ("O", o_coord)]:
                    elem = atom_name[0]
                    f.write(f"ATOM  {atom_num:5d}  {atom_name:<3s} {aa3} A{res_num:4d}    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                            f"  1.00{plddt:6.2f}           {elem}\n")
                    atom_num += 1
            
            f.write("END\n")
    
    def expand_node(self, node: HallucinationNode, iteration: int = 0) -> List[HallucinationNode]:
        """
        Expand a node by running Structure -> ProteinMPNN cycle.
        
        Returns list of child nodes.
        """
        edit_log = []
        dssp_result = None
        
        # Step 1: Predict structure from current sequence
        backend_name = self.structure_backend.upper()
        print(f"      {backend_name} predicting structure from: {node.sequence[:30]}...")
        
        # Get template/ligand conditioning for SS guidance Variants 3 & 4
        template_cond = None
        ligand_cond = None
        if self.ss_guidance:
            template_cond = self.ss_guidance.get_template_conditioning(iteration)
            ligand_cond = self.ss_guidance.get_ligand_conditioning(iteration)
        
        # Call structure predictor
        if template_cond or ligand_cond:
            if hasattr(self.structure_predictor, 'predict_structure_with_conditioning'):
                structure_result = self.structure_predictor.predict_structure_with_conditioning(
                    node.sequence, template=template_cond, ligand=ligand_cond)
            else:
                print(f"      Warning: Backend does not support conditioning")
                structure_result = self.structure_predictor.predict_structure(node.sequence)
        else:
            structure_result = self.structure_predictor.predict_structure(node.sequence)
        
        coords = structure_result['coordinates']
        parent_plddt = structure_result['confidence']
        parent_mean_plddt = float(np.mean(parent_plddt))
        print(f"      Done - mean pLDDT={parent_mean_plddt:.1f}")
        
        # Step 1.5: Run DSSP for SS analysis if enabled
        if self.ss_guidance and self.run_dir:
            temp_pdb_path = self.run_dir / f"iter_{iteration:03d}" / "structure_pred.pdb"
            temp_pdb_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_pdb_for_dssp(node.sequence, coords, parent_plddt, str(temp_pdb_path))
            
            dssp_result = self.ss_guidance.run_dssp(str(temp_pdb_path), len(node.sequence))
            if dssp_result:
                print(f"      DSSP: {len(dssp_result.helix_positions)} helix, {len(dssp_result.beta_positions)} beta")
        
        # Step 2: Generate K candidate sequences
        candidate_sequences = []
        candidate_data = []
        
        total_candidates = self.num_candidates * self.num_rollouts
        print(f"      Generating {total_candidates} candidates...")
        
        for rollout in range(self.num_rollouts):
            try:
                for k in range(self.num_candidates):
                    masked_seq = node.sequence
                    
                    # Variant 1: Apply beta-lock before MPNN
                    if (self.ss_guidance and dssp_result and 
                        self.ss_guidance_config.ss_guidance == "beta_lock_reinit_helix"):
                        masked_seq, variant1_edits = self.ss_guidance.apply_variant1_beta_lock_reinit_helix(
                            node.sequence, dssp_result)
                        edit_log.extend(variant1_edits)
                    
                    # Generate new sequence via ProteinMPNN
                    new_sequence = self.hallucination_expert.proteinmpnn.design_sequence(
                        coordinates=coords, masked_sequence=masked_seq)
                    
                    # Variant 2: Apply PG injection after MPNN
                    candidate_edit_log = []
                    if (self.ss_guidance and dssp_result and 
                        self.ss_guidance_config.ss_guidance == "x_init_helix_pg"):
                        new_sequence, variant2_edits = self.ss_guidance.apply_variant2_helix_pg_injection(
                            new_sequence, dssp_result)
                        candidate_edit_log.extend(variant2_edits)
                    
                    # Evaluate new sequence
                    new_structure_result = self.structure_predictor.predict_structure(new_sequence)
                    new_coords = new_structure_result['coordinates']
                    new_plddt = new_structure_result['confidence']
                    new_mean_plddt = float(np.mean(new_plddt))
                    
                    # Compute entropy
                    entropy = self.compute_entropy(new_sequence, new_coords)
                    
                    candidate_sequences.append(new_sequence)
                    candidate_data.append({
                        'sequence': new_sequence,
                        'coordinates': new_coords,
                        'plddt': new_plddt,
                        'mean_plddt': new_mean_plddt,
                        'entropy': entropy,
                        'edit_log': candidate_edit_log,
                    })
                    
                    print(f"      Candidate {len(candidate_sequences)}: pLDDT={new_mean_plddt:.1f}")
                    
            except Exception as e:
                print(f"      Rollout {rollout+1} failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 3: Compute sibling convergence
        sibling_convergence = self.compute_sibling_convergence(candidate_sequences)
        print(f"      Sibling convergence: {sibling_convergence:.3f}")
        
        # Step 4: Create child nodes
        children = []
        for i, cand in enumerate(candidate_data):
            parent_child_sim = self.compute_sequence_similarity(node.sequence, cand['sequence'])
            novelty = self.compute_novelty(cand['sequence'], node)
            
            reward = self.compute_convergence_reward(
                sibling_convergence, parent_child_sim, cand['mean_plddt'], cand['entropy'])
            
            child = HallucinationNode(
                sequence=cand['sequence'],
                coordinates=cand['coordinates'],
                plddt_scores=cand['plddt'],
                mean_plddt=cand['mean_plddt'],
                parent=node,
                depth=node.depth + 1,
                convergence_score=parent_child_sim,
                parent_child_similarity=parent_child_sim,
                sibling_convergence=sibling_convergence,
                entropy=cand['entropy'],
                novelty=novelty,
                visits=1,
                total_reward=reward,
            )
            children.append(child)
            
            print(f"      Child {i+1}: sim={parent_child_sim:.3f}, pLDDT={cand['mean_plddt']:.1f}, reward={reward:.3f}")
        
        return children
    
    def select_node(self, root: HallucinationNode) -> HallucinationNode:
        """UCT/PH-UCT selection - traverse tree to find best node to expand."""
        node = root
        while node.children and node.depth < self.max_depth:
            if self.use_ph_uct:
                node = max(node.children, key=lambda c: c.ph_uct_score(
                    self.exploration_constant, self.entropy_weight, self.novelty_weight))
            else:
                node = max(node.children, key=lambda c: c.uct_score(self.exploration_constant))
        return node
    
    def backpropagate(self, node: HallucinationNode, reward: float):
        """Backpropagate reward up the tree (sum rule)."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent
    
    def _backpropagate_max_rule(self, node: HallucinationNode):
        """Backpropagate using max rule (W <- max(W, v)) - aligned with sequence_level_mcts.py."""
        current = node
        while current:
            current.visits += 1
            current.total_reward = max(current.total_reward, node.reward)
            current = current.parent
    
    def get_best_child(self, node: HallucinationNode) -> HallucinationNode:
        """Get the best child node based on reward - aligned with sequence_level_mcts.py."""
        if not node.children:
            return node
        
        best_child = node
        best_reward = node.reward
        
        for child in node.children:
            child_reward = child.reward
            if child_reward > best_reward:
                best_child = child
                best_reward = child_reward
        
        return best_child
    
    def _prepare_plddt_scores(self, sequence_length: int, raw_scores) -> List[float]:
        """Normalize raw pLDDT inputs - aligned with sequence_level_mcts.py."""
        try:
            if raw_scores is None:
                raise ValueError("No pLDDT scores provided")
            
            # Handle dict-style payloads
            if isinstance(raw_scores, dict):
                candidates = ['plddt_scores', 'plddt', 'scores', 'confidence', 'lddts']
                selected = None
                for key in candidates:
                    if key in raw_scores and raw_scores[key]:
                        selected = raw_scores[key]
                        break
                if selected is None and raw_scores:
                    for value in raw_scores.values():
                        if value:
                            selected = value
                            break
                raw_scores = selected if selected is not None else []
            
            # Convert numpy arrays / tensors to list
            if hasattr(raw_scores, 'cpu'):
                raw_scores = raw_scores.cpu()
            if hasattr(raw_scores, 'numpy'):
                raw_scores = raw_scores.numpy()
            if isinstance(raw_scores, np.ndarray):
                raw_scores = raw_scores.tolist()
            
            if not isinstance(raw_scores, (list, tuple)):
                raw_scores = [float(raw_scores)]
            
            scores = [float(x) for x in raw_scores if x is not None]
        except Exception as e:
            print(f"   Warning: Failed to parse pLDDT scores ({e}); using default 70.0.")
            scores = []
        
        if not scores:
            scores = [70.0] * sequence_length
        else:
            max_score = max(scores)
            if max_score <= 1.5:
                scores = [min(1.0, max(0.0, s)) * 100.0 for s in scores]
            if len(scores) >= sequence_length:
                scores = scores[:sequence_length]
            else:
                pad_value = float(np.mean(scores)) if scores else 70.0
                scores = scores + [pad_value] * (sequence_length - len(scores))
        
        return scores
    
    def _compute_progressive_plddt_masking(self, sequence: str, plddt_scores: List[float], depth: int) -> Set[int]:
        """Progressive pLDDT masking - aligned with sequence_level_mcts.py."""
        if plddt_scores is None or len(plddt_scores) == 0:
            plddt_scores = [70.0] * len(sequence)
        elif len(plddt_scores) != len(sequence):
            plddt_scores = self._prepare_plddt_scores(len(sequence), plddt_scores)
        
        # Progressive thresholds by depth
        if depth == 0:
            threshold = 70.0
            target_ratio = 0.25
        elif depth == 1:
            threshold = 75.0
            target_ratio = 0.20
        elif depth == 2:
            threshold = 80.0
            target_ratio = 0.15
        elif depth == 3:
            threshold = 85.0
            target_ratio = 0.05
        else:
            threshold = 100.0
            target_ratio = 0.0
        
        # Threshold-based masking
        threshold_masked = set([i for i, score in enumerate(plddt_scores) if score < threshold])
        
        min_positions = max(3, int(len(sequence) * 0.05))
        max_positions = int(len(sequence) * 0.30)
        
        if min_positions <= len(threshold_masked) <= max_positions:
            masked_positions = threshold_masked
        else:
            # Quantile-based fallback
            num_to_mask = max(min_positions, int(len(sequence) * target_ratio))
            num_to_mask = min(num_to_mask, max_positions)
            
            position_scores = [(i, score) for i, score in enumerate(plddt_scores)]
            position_scores.sort(key=lambda x: x[1])
            
            masked_positions = set([pos for pos, _ in position_scores[:num_to_mask]])
        
        return masked_positions
    
    def _evaluate_sequence_aar(self, sequence: str, reference_sequence: str = None) -> float:
        """Evaluate sequence using AAR - aligned with sequence_level_mcts.py."""
        if reference_sequence is None or len(sequence) != len(reference_sequence):
            return 0.5
        
        matches = sum(1 for a, b in zip(sequence, reference_sequence) if a == b)
        aar = matches / len(sequence)
        return aar
    
    def search(self) -> HallucinationNode:
        """Run MCTS search for hallucination design."""
        print("\n" + "="*80)
        print("Starting Hallucination MCTS Search")
        if self.ss_guidance_config.ss_guidance != "none":
            print(f"   SS Guidance: {self.ss_guidance_config.ss_guidance}")
        print("="*80)
        
        # Create root node
        initial_seq = self.generate_initial_sequence(self.sequence_length)
        print(f"\nRoot: {self.init_mode} sequence (length={len(initial_seq)})")
        
        root = HallucinationNode(sequence=initial_seq, depth=0, visits=1)
        
        node_counter = 0
        summary_rows = []
        best_node = root
        best_convergence = 0.0
        
        for iteration in range(self.num_iterations):
            print(f"\nIteration {iteration + 1}/{self.num_iterations}")
            
            # Selection
            selected = self.select_node(root)
            print(f"   Selected node at depth {selected.depth}")
            
            # Expansion
            iter_best_plddt = 0.0
            iter_best_reward = 0.0
            iter_num_valid = 0
            
            if selected.depth < self.max_depth:
                children = self.expand_node(selected, iteration=iteration)
                selected.children.extend(children)
                
                for child in children:
                    node_counter += 1
                    self.backpropagate(child, child.get_reward())
                    iter_num_valid += 1
                    iter_best_plddt = max(iter_best_plddt, child.mean_plddt)
                    iter_best_reward = max(iter_best_reward, child.get_reward())
                    
                    # Save PDB
                    if child.coordinates is not None:
                        pdb_path = self.save_pdb(
                            child.sequence, child.coordinates, iteration + 1,
                            child.depth, node_counter, child.mean_plddt)
                        print(f"   Saved: {Path(pdb_path).name}")
                    
                    # Track best
                    if child.convergence_score > best_convergence:
                        best_convergence = child.convergence_score
                        best_node = child
                        print(f"   New best: sim={best_convergence:.3f}, pLDDT={child.mean_plddt:.1f}")
            
            summary_rows.append({
                "iteration": iteration + 1,
                "mean_plddt": iter_best_plddt,
                "best_reward": iter_best_reward,
                "num_valid": iter_num_valid,
                "best_convergence": best_convergence,
                "best_depth": best_node.depth,
            })
            
            if best_convergence > 0.95:
                print(f"\nConverged! Parent-child similarity > 95%")
                break
        
        # Save final best
        if best_node.coordinates is not None:
            final_pdb = self.save_pdb(
                best_node.sequence, best_node.coordinates, self.num_iterations,
                best_node.depth, 9999, best_node.mean_plddt, suffix="_BEST")
            print(f"\nBest structure saved: {final_pdb}")
        
        # Save summary CSV
        if self.run_dir:
            self._save_summary_csv(summary_rows)
        
        return best_node
    
    def _save_summary_csv(self, summary_rows: List[Dict]):
        """Save summary CSV with per-iteration metrics."""
        csv_path = self.run_dir / "summary.csv"
        
        if not summary_rows:
            return
        
        fieldnames = list(summary_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        
        print(f"   Summary saved: {csv_path}")
    
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


# Backward compatibility alias
GeneralMCTS = HallucinationMCTS
MCTSNode = HallucinationNode
