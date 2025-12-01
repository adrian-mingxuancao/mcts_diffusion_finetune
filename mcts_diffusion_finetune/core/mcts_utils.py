"""
Utility functions for MCTS masked diffusion search.
"""
import hashlib
import random
from typing import List, Set, Tuple, Dict, Any
import torch
import math


def apply_patch(parent_seq: str, proposal_seq: str, mask_idxs: Set[int]) -> str:
    """Apply edits only at the chosen mask positions."""
    out = list(parent_seq)
    for i in mask_idxs:
        if i < len(proposal_seq):
            out[i] = proposal_seq[i]
    return ''.join(out)


def compute_mask_schedule(sequence: str, plddt_scores: List[float], depth: int, max_depth: int = 4) -> Set[int]:
    """
    Quantile-based progressive masking strategy for motif scaffolding.
    
    Strategy: Use quantile-based approach that becomes more conservative at deeper levels:
    - Depth 0: Mask bottom 80% of low-confidence positions
    - Depth 1: Mask bottom 40% of low-confidence positions  
    - Depth 2: Mask bottom 20% of low-confidence positions
    - Depth 3: Mask bottom 10% of low-confidence positions
    - Depth 4+: Mask bottom 5% of low-confidence positions
    
    This ensures meaningful exploration while becoming more focused at deeper levels.
    """
    import numpy as np
    
    # Handle both numpy arrays and lists
    if isinstance(plddt_scores, np.ndarray):
        # Handle multi-dimensional pLDDT (e.g., per-atom confidence)
        if plddt_scores.ndim > 1:
            # Use MIN across atoms for per-residue confidence (most conservative)
            plddt_per_residue = np.min(plddt_scores, axis=-1)
        else:
            plddt_per_residue = plddt_scores
    else:
        plddt_per_residue = np.array(plddt_scores)
    
    seq_len = len(sequence)
    
    # **NEW**: Quantile-based masking percentages
    if depth == 0:
        mask_percentage = 0.80  # 80% at depth 0
    elif depth == 1:
        mask_percentage = 0.40  # 40% at depth 1
    elif depth == 2:
        mask_percentage = 0.20  # 20% at depth 2
    elif depth == 3:
        mask_percentage = 0.10  # 10% at depth 3
    else:
        mask_percentage = 0.05  # 5% at depth 4+
    
    # Calculate number of positions to mask
    num_to_mask = max(1, int(seq_len * mask_percentage))  # At least 1 position
    
    # **NEW**: Use quantile-based selection instead of threshold-based
    # Sort all positions by confidence (lowest first)
    position_confidence = [(i, float(plddt_per_residue[i])) for i in range(seq_len)]
    position_confidence.sort(key=lambda x: x[1])  # Sort by pLDDT ascending
    
    # Take the lowest confidence positions up to the percentage
    mask_positions = [pos for pos, _ in position_confidence[:num_to_mask]]
    
    # Calculate the actual confidence threshold used
    if num_to_mask < len(position_confidence):
        threshold_plddt = position_confidence[num_to_mask-1][1]
    else:
        threshold_plddt = position_confidence[-1][1] if position_confidence else 0.0
    
    print(f"🎯 Depth {depth}: masking {len(mask_positions)} positions ({mask_percentage*100:.0f}% quantile), pLDDT threshold ≤ {threshold_plddt:.1f}")
    
    return set(mask_positions)


def compute_mask_schedule_inverse_folding(sequence: str, plddt_scores: List[float], depth: int, max_depth: int = 4) -> Set[int]:
    """
    Hybrid threshold + quantile progressive masking strategy for inverse folding.
    
    For inverse folding, we have high confidence in the structure, so we prefer to mask
    low pLDDT areas. Uses a hybrid approach:
    1. **Primary**: Threshold-based masking (conservative)
    2. **Fallback**: Quantile-based masking if too few positions
    
    Strategy: 
    - **Threshold approach**: Use conservative pLDDT thresholds that become more stringent at deeper levels:
      - Depth 0: Mask positions with pLDDT < 70
      - Depth 1: Mask positions with pLDDT < 60  
      - Depth 2: Mask positions with pLDDT < 50
      - Depth 3: Mask positions with pLDDT < 40
      - Depth 4+: Mask positions with pLDDT < 30
    
    - **Fallback**: If threshold approach gives < 5% of sequence, use quantile-based:
      - Depth 0: Mask bottom 30% by pLDDT (broad exploration)
      - Depth 1: Mask bottom 20% by pLDDT
      - Depth 2: Mask bottom 15% by pLDDT
      - Depth 3: Mask bottom 10% by pLDDT
      - Depth 4+: Mask bottom 5% by pLDDT (focused refinement)
    
    This ensures we always have sufficient positions for meaningful MCTS exploration while
    preserving high-confidence structure regions when possible.
    """
    import numpy as np
    
    # Handle both numpy arrays and lists
    if isinstance(plddt_scores, np.ndarray):
        # Handle multi-dimensional pLDDT (e.g., per-atom confidence)
        if plddt_scores.ndim > 1:
            # Use MIN across atoms for per-residue confidence (most conservative)
            plddt_per_residue = np.min(plddt_scores, axis=-1)
        else:
            plddt_per_residue = plddt_scores
    else:
        plddt_per_residue = np.array(plddt_scores)
    
    seq_len = len(sequence)
    
    # **CONSERVATIVE**: Threshold-based masking for inverse folding
    if depth == 0:
        threshold = 70.0  # Only mask really low confidence regions
    elif depth == 1:
        threshold = 60.0  # Slightly more aggressive
    elif depth == 2:
        threshold = 50.0  # Moderate threshold
    elif depth == 3:
        threshold = 40.0  # Lower threshold
    else:
        threshold = 30.0  # Very low threshold for deep exploration
    
    # Find positions below the threshold
    mask_positions = []
    for i in range(seq_len):
        if float(plddt_per_residue[i]) < threshold:
            mask_positions.append(i)
    
    # **HYBRID APPROACH**: Threshold first, quantile fallback if too few positions
    min_positions_needed = max(3, seq_len // 20)  # At least 5% of sequence or minimum 3 positions
    max_positions_allowed = min(seq_len // 4, 50)  # At most 25% of sequence or 50 positions
    
    if len(mask_positions) < min_positions_needed:
        print(f"🎯 Depth {depth}: threshold approach gave {len(mask_positions)} positions, falling back to quantile approach")
        
        # **FALLBACK**: Use quantile-based masking to ensure sufficient positions
        # Progressive quantiles based on depth (MORE masking at shallow depths, LESS at deep depths)
        if depth == 0:
            quantile = 0.3  # Mask bottom 30% (broad exploration)
        elif depth == 1:
            quantile = 0.2  # Mask bottom 20%
        elif depth == 2:
            quantile = 0.15  # Mask bottom 15%
        elif depth == 3:
            quantile = 0.1  # Mask bottom 10%
        else:
            quantile = 0.05  # Mask bottom 5% (focused refinement at terminal nodes)
        
        # Calculate quantile threshold
        quantile_threshold = np.percentile(plddt_per_residue, quantile * 100)
        
        # Find positions below quantile threshold
        mask_positions = []
        for i in range(seq_len):
            if float(plddt_per_residue[i]) < quantile_threshold:
                mask_positions.append(i)
        
        # Ensure we have enough positions
        if len(mask_positions) < min_positions_needed:
            # Take the lowest confidence positions
            position_confidence = [(i, float(plddt_per_residue[i])) for i in range(seq_len)]
            position_confidence.sort(key=lambda x: x[1])  # Sort by pLDDT ascending
            mask_positions = [pos for pos, _ in position_confidence[:min_positions_needed]]
            print(f"🎯 Depth {depth}: quantile fallback insufficient, masking {len(mask_positions)} lowest confidence positions")
        else:
            print(f"🎯 Depth {depth}: quantile fallback masking {len(mask_positions)} positions below {quantile_threshold:.1f} ({len(mask_positions)/seq_len*100:.1f}%)")
    
    elif len(mask_positions) > max_positions_allowed:
        # If too many positions, take the lowest confidence ones
        position_confidence = [(i, float(plddt_per_residue[i])) for i in mask_positions]
        position_confidence.sort(key=lambda x: x[1])  # Sort by pLDDT ascending
        mask_positions = [pos for pos, _ in position_confidence[:max_positions_allowed]]
        print(f"🎯 Depth {depth}: threshold approach gave too many positions, limiting to {len(mask_positions)} lowest confidence")
    else:
        print(f"🎯 Depth {depth}: threshold approach masking {len(mask_positions)} positions with pLDDT < {threshold:.1f} ({len(mask_positions)/seq_len*100:.1f}%)")
    
    return set(mask_positions)


def compute_sequence_hash(sequence: str) -> str:
    """Compute hash for sequence deduplication."""
    return hashlib.md5(sequence.encode()).hexdigest()[:16]


def compute_hamming_distance(seq1: str, seq2: str) -> int:
    """Compute Hamming distance between two sequences."""
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


def dedupe_and_filter(proposals: List[Tuple[str, float, str]], min_hamming: int = 8) -> List[Tuple[str, float, str]]:
    """Remove near-duplicates and enforce diversity."""
    if not proposals:
        return []
    
    # Sort by score first
    proposals = sorted(proposals, key=lambda x: x[1], reverse=True)
    
    # Keep diverse proposals
    kept = [proposals[0]]  # Always keep the best
    
    for seq, score, meta in proposals[1:]:
        # Check diversity against all kept sequences
        diverse = all(compute_hamming_distance(seq, kept_seq) >= min_hamming 
                     for kept_seq, _, _ in kept)
        if diverse:
            kept.append((seq, score, meta))
    
    return kept


def compute_fast_aar(sequence: str, reference: str) -> float:
    """Fast AAR calculation for proxy scoring."""
    if len(sequence) != len(reference):
        min_len = min(len(sequence), len(reference))
        sequence = sequence[:min_len]
        reference = reference[:min_len]
    
    matches = sum(s == r for s, r in zip(sequence, reference))
    return matches / len(sequence) if sequence else 0.0


def ph_uct_score(average_value: float, visit_count: int, parent_visits: int, 
                 c: float = 1.414, w_ent: float = 0.1, w_div: float = 0.1, 
                 entropy_proposals: float = 0.0, novelty_vs_parent: float = 0.0) -> float:
    """
    PH-UCT score matching ERP paper formula with multiplication.
    
    ERP formula: Q + c * sqrt(log(N) / n) * entropy_factor + diversity_bonus
    """
    if visit_count == 0:
        return float('inf')
    
    # Core UCB exploration term
    ucb_exploration = c * math.sqrt(math.log(parent_visits + 1) / (visit_count + 1))
    
    # ERP paper: multiply exploration by entropy factor
    entropy_factor = (1.0 / math.e) * entropy_proposals if entropy_proposals > 0 else 1.0
    ph_exploration = ucb_exploration * entropy_factor
    
    # Diversity bonus is additive
    diversity_bonus = w_div * novelty_vs_parent
    
    return average_value + ph_exploration + diversity_bonus


class SequenceCache:
    """Cache for sequence evaluations to avoid recomputation."""
    
    def __init__(self):
        self.aar_cache = {}
        self.sctm_cache = {}
        self.biophys_cache = {}
        self.rewards = {}
        self.cache = {}  # General cache for deduplication
    
    def get_aar(self, seq_hash: str) -> float:
        return self.aar_cache.get(seq_hash)
    
    def set_aar(self, seq_hash: str, aar: float):
        self.aar_cache[seq_hash] = aar
    
    def get_sctm(self, seq_hash: str) -> float:
        return self.sctm_cache.get(seq_hash)
    
    def set_sctm(self, seq_hash: str, sctm: float):
        self.sctm_cache[seq_hash] = sctm
    
    def get_biophys(self, seq_hash: str) -> float:
        return self.biophys_cache.get(seq_hash)
    
    def set_biophys(self, seq_hash: str, biophys: float):
        self.biophys_cache[seq_hash] = biophys
    
    def get_reward(self, seq_hash: str) -> float:
        return self.rewards.get(seq_hash)
    
    def set_reward(self, seq_hash: str, reward: float):
        self.rewards[seq_hash] = reward
